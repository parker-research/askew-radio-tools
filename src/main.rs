//! Command-line entry point for the AX100 Mode 5 (CSP) beacon decoder.
//!
//! Usage:
//!   ax100-radio-csp-decoder <audio_file.wav|.ogg> [more files...]
//!
//! Each already-doppler-corrected audio file is decoded independently.
//! Decoded CSP frames are written to stdout as JSONL (one JSON object per
//! line, fields: filename, data_hex, data_length_bytes,
//! start_time_in_file_ms, rs_corrected_error_count, crc_pass). All other
//! diagnostics go to stderr so stdout stays valid JSONL.

use ax100_radio_csp_decoder::{audio_check, pipeline};
use clap::Parser;

#[derive(Parser)]
#[command(about = "AX100 Mode 5 (CSP) beacon decoder — emits JSONL to stdout")]
struct Cli {
    /// Audio files to decode (.wav or .ogg), already Doppler-corrected.
    #[arg(required = true)]
    audio_files: Vec<String>,
}

fn main() {
    let cli = Cli::parse();
    let mut had_error = false;

    for path in &cli.audio_files {
        if let Err(e) = decode_and_print(path) {
            had_error = true;
            eprintln!("{path}: error: {e}");
        }
    }

    if had_error {
        std::process::exit(1);
    }
}

fn decode_and_print(path: &str) -> Result<(), ax100_radio_csp_decoder::DecodeError> {
    let audio = ax100_radio_csp_decoder::audio::load_audio(path)?;
    let metrics = audio_check::check(&audio);
    eprintln!("{path}: {}", metrics.verdict);

    let records = pipeline::decode_audio(&audio, path);
    eprintln!("{path}: {} frame(s) decoded", records.len());

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    for record in &records {
        // A single serde_json struct can't fail to serialize here (no
        // maps/floats that are NaN/inf), so this is safe to unwrap.
        use std::io::Write;
        writeln!(handle, "{}", serde_json::to_string(record).unwrap())
            .expect("failed to write to stdout");
    }

    Ok(())
}
