//! Command-line entry point for the AX100 ASM+Golay (CSP) packet decoder.
//!
//! Usage:
//!   askew_demod_from_file <audio_file.wav|.ogg> [more files...]
//!
//! Each already-doppler-corrected audio file is decoded independently.
//! Decoded CSP frames are written to stdout as JSONL (one JSON object per
//! line, fields: data_length_bytes, time_in_file_ms, tier,
//! sync_bit_errors, golay_corrected_bit_count, header_flags,
//! rs_corrected_error_count, rs_correctable, crc_pass, combined_copies,
//! rssi_db, data_hex —
//! plus filename if `--show-filename` is passed).
//!
//! The decoder labels every candidate frame with a `pipeline::FrameTier`
//! rather than dropping the ones it doesn't believe, and `--output-filter`
//! picks how much evidence is worth printing. It defaults to `verified`
//! (RS-decoded with a verifying CRC32C); loosening it one step at a time
//! goes `believable` (also real-but-too-corrupt-to-repair bursts), then
//! `all`, which prints raw syncword hits and is a diagnostic mode, not a
//! decode — on a multi-minute capture most of what it prints is noise.
//!
//! All other diagnostics go to stderr so stdout stays valid JSONL.

use askew_radio_tools::pipeline::{FrameTier, PacketRecord};
use askew_radio_tools::{audio_check, pipeline};
use clap::{Parser, ValueEnum};

/// How much evidence a frame needs before it's included in the JSONL
/// output. Each level keeps a minimum `pipeline::FrameTier`; the doc
/// comments below are what `--help` prints, so they're written for a
/// terminal rather than for rustdoc.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFilter {
    /// Every candidate frame whose Golay header decoded — raw syncword
    /// hits included.
    ///
    /// A diagnostic mode, not a decode: on a multi-minute capture most of
    /// what this prints is noise that happened to match the syncword. Use
    /// it to answer "did this recording contain anything at all?", and
    /// judge each record by its "tier", "sync_bit_errors",
    /// "golay_corrected_bit_count" and "header_flags" fields rather than
    /// by its payload.
    All,
    /// Frames the decoder believes were really transmitted, repairable or
    /// not (tier "believable" and above).
    ///
    /// Adds bursts that Reed-Solomon could not correct but whose syncword
    /// and Golay header are too clean for noise to explain. Their payloads
    /// are best-effort and probably still corrupt — but they are real
    /// transmissions, not noise.
    Believable,
    /// Only frames where Reed-Solomon corrected the codeword (tier
    /// "rs_correctable_crc_error" and above).
    ///
    /// CRC failures are still included. In practice this tier is empty:
    /// on every capture measured so far, RS success and CRC success
    /// coincide.
    RsCorrectableCrcError,
    /// Only frames that are RS-correctable *and* whose CSP CRC32C trailer
    /// verifies (tier "verified").
    ///
    /// The strictest filter, for perfectly good frames only.
    Verified,
}

impl OutputFilter {
    /// The weakest tier this filter still prints.
    fn min_tier(self) -> FrameTier {
        match self {
            OutputFilter::All => FrameTier::Candidate,
            OutputFilter::Believable => FrameTier::Believable,
            OutputFilter::RsCorrectableCrcError => FrameTier::RsCorrectableCrcError,
            OutputFilter::Verified => FrameTier::Verified,
        }
    }

    fn keep(self, record: &PacketRecord) -> bool {
        record.tier >= self.min_tier()
    }
}

/// Which over-the-air protocol (modulation + framing + FEC) to decode.
/// More will be added; each variant picks a whole decode chain.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Protocol {
    /// GomSpace AX100 Mode 5: 2-FSK/GFSK, ASM syncword + Golay(24,12)
    /// length header, CCSDS scrambler, RS(255,223), CSP with CRC32C.
    #[value(name = "ax100-mode-5")]
    Ax100Mode5,
}

#[derive(Parser)]
#[command(
    version,
    about = "AX100 ASM+Golay (CSP) packet decoder — emits JSONL to stdout"
)]
struct Cli {
    /// Audio files to decode (.wav or .ogg), already Doppler-corrected.
    #[arg(required = true)]
    audio_files: Vec<String>,

    /// Include the source filename in each JSONL record (excluded by default).
    #[arg(long)]
    show_filename: bool,

    /// Which decoded frames to emit.
    #[arg(long, value_enum, default_value_t = OutputFilter::Verified)]
    output_filter: OutputFilter,

    /// Transmitter symbol rate, in baud.
    #[arg(long, default_value_t = 9600, value_parser = clap::value_parser!(u32).range(1..))]
    baud_rate: u32,

    /// Over-the-air protocol (modulation, framing, and FEC) to decode.
    #[arg(long, value_enum, default_value_t = Protocol::Ax100Mode5)]
    protocol: Protocol,
}

fn main() {
    let cli = Cli::parse();
    let mut had_error = false;

    for path in &cli.audio_files {
        if let Err(e) = decode_and_print(path, &cli) {
            had_error = true;
            eprintln!("{path}: error: {e}");
        }
    }

    if had_error {
        std::process::exit(1);
    }
}

fn decode_and_print(path: &str, cli: &Cli) -> Result<(), askew_radio_tools::DecodeError> {
    let output_filter = cli.output_filter;
    let audio = askew_radio_tools::audio::load_audio(path)?;
    let metrics = audio_check::check(&audio);
    eprintln!("{path}: {}", metrics.verdict);

    let records = match cli.protocol {
        Protocol::Ax100Mode5 => pipeline::decode_audio(&audio, path, f64::from(cli.baud_rate)),
    };
    let believable = records
        .iter()
        .filter(|r| r.tier >= FrameTier::Believable)
        .count();
    eprintln!(
        "{path}: {} candidate frame(s) decoded, {believable} believable",
        records.len()
    );

    let filtered: Vec<&PacketRecord> = records.iter().filter(|r| output_filter.keep(r)).collect();
    eprintln!(
        "{path}: {} frame(s) emitted after --output-filter={:?}",
        filtered.len(),
        output_filter
    );

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    for record in filtered {
        // A single serde_json struct can't fail to serialize here (no
        // maps/floats that are NaN/inf), so this is safe to unwrap.
        use std::io::Write;
        let mut value = serde_json::to_value(record).unwrap();
        if !cli.show_filename {
            // `.remove()` is a `swap_remove` under the `preserve_order`
            // feature (moves the last field into "filename"'s slot), which
            // would scramble the remaining field order — `shift_remove`
            // keeps it intact.
            value.as_object_mut().unwrap().shift_remove("filename");
        }
        writeln!(handle, "{}", serde_json::to_string(&value).unwrap())
            .expect("failed to write to stdout");
    }

    Ok(())
}
