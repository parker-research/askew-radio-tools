//! Command-line entry point for the AX100 ASM+Golay (CSP) beacon decoder.
//!
//! Usage:
//!   askew_demod_from_file <audio_file.wav|.ogg> [more files...]
//!
//! Each already-doppler-corrected audio file is decoded independently.
//! Decoded CSP frames are written to stdout as JSONL (one JSON object per
//! line, fields: data_length_bytes, time_in_file_ms,
//! rs_corrected_error_count, rs_correctable, crc_pass, data_hex — plus
//! filename if `--show-filename` is passed). Frames with uncorrectable
//! Reed-Solomon errors are still emitted (rs_correctable: false) rather
//! than dropped, unless filtered out via `--output-filter`. All other
//! diagnostics go to stderr so stdout stays valid JSONL.

use askew_radio_tools::pipeline::BeaconRecord;
use askew_radio_tools::{audio_check, pipeline};
use clap::{Parser, ValueEnum};

/// Which decoded frames to include in the JSONL output.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFilter {
    /// Emit every decoded frame, including RS-uncorrectable ones.
    All,
    /// Only frames where Reed-Solomon successfully corrected the codeword
    /// (`rs_correctable: true`) — CRC failures are still included.
    RsCorrectable,
    /// Only frames that are RS-correctable *and* whose CSP CRC (when
    /// present) passes — the strictest filter, for "perfectly good" frames
    /// only. A frame with no CRC field at all doesn't count against this,
    /// since there's nothing to have failed.
    Good,
}

impl OutputFilter {
    fn keep(self, record: &BeaconRecord) -> bool {
        match self {
            OutputFilter::All => true,
            OutputFilter::RsCorrectable => record.rs_correctable,
            OutputFilter::Good => record.rs_correctable && record.crc_pass != Some(false),
        }
    }
}

#[derive(Parser)]
#[command(
    version,
    about = "AX100 ASM+Golay (CSP) beacon decoder — emits JSONL to stdout"
)]
struct Cli {
    /// Audio files to decode (.wav or .ogg), already Doppler-corrected.
    #[arg(required = true)]
    audio_files: Vec<String>,

    /// Include the source filename in each JSONL record (excluded by default).
    #[arg(long)]
    show_filename: bool,

    /// Which decoded frames to emit.
    #[arg(long, value_enum, default_value_t = OutputFilter::All)]
    output_filter: OutputFilter,
}

fn main() {
    let cli = Cli::parse();
    let mut had_error = false;

    for path in &cli.audio_files {
        if let Err(e) = decode_and_print(path, cli.show_filename, cli.output_filter) {
            had_error = true;
            eprintln!("{path}: error: {e}");
        }
    }

    if had_error {
        std::process::exit(1);
    }
}

fn decode_and_print(
    path: &str,
    show_filename: bool,
    output_filter: OutputFilter,
) -> Result<(), askew_radio_tools::DecodeError> {
    let audio = askew_radio_tools::audio::load_audio(path)?;
    let metrics = audio_check::check(&audio);
    eprintln!("{path}: {}", metrics.verdict);

    let records = pipeline::decode_audio(&audio, path);
    eprintln!("{path}: {} frame(s) decoded", records.len());

    let filtered: Vec<&BeaconRecord> = records.iter().filter(|r| output_filter.keep(r)).collect();
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
        if !show_filename {
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
