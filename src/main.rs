//! Command-line entry point for the AX100 beacon decoder.
//!
//! Usage:
//!   decode <audio_file.wav|.ogg>
//!
//! Outputs decoded CSP packet payload bytes (hex) for each valid frame found.
use anyhow::{Context, Result};
use ax100_radio_csp_decoder::{audio, dsp, fec, framing};
use clap::Parser;

#[derive(Parser)]
#[command(about = "AX100 beacon decoder")]
struct Cli {
    /// Audio file to decode (.wav or .ogg)
    audio_file: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let path = &cli.audio_file;
    println!("Loading audio: {}", path);

    // 1. Load audio
    let audio =
        audio::load_audio(path).with_context(|| format!("Failed to load audio from {}", path))?;

    println!(
        "  {} samples @ {} Hz ({:.1} s, {} ch)",
        audio.samples.len(),
        audio.sample_rate,
        audio.samples.len() as f64 / audio.sample_rate as f64,
        audio.channels,
    );
    println!(
        "  Samples per symbol at 9600 baud: {:.2}",
        audio.samples_per_symbol()
    );

    // 2. DSP front-end: FM discriminator → LPF → Gardner TED → slicer
    println!("\nRunning DSP front-end...");
    let bitstream = dsp::fm_discriminate_and_filter(&audio);
    println!(
        "  Recovered {} bits, symbol rate ≈ {:.1} Hz",
        bitstream.bits.len(),
        bitstream.recovered_symbol_rate,
    );

    // 3. Sync word search + frame extraction
    println!("\nSearching for frames...");
    let frames = framing::find_frames(&bitstream.bits);
    println!("  Found {} candidate frame(s)", frames.len());

    if frames.is_empty() {
        println!("\nNo frames found. Check that the audio contains a valid AX100 beacon.");
        return Ok(());
    }

    // 4. Process each frame through FEC pipeline
    let mut valid = 0usize;
    for (idx, frame) in frames.iter().enumerate() {
        println!(
            "\n--- Frame {} (sync at bit {}) ---",
            idx + 1,
            frame.sync_bit_offset
        );
        println!("  Raw payload: {} bytes", frame.data.len());

        // Clone so we can mutate
        let mut data = frame.data.clone();

        // Pad or trim to RS codeword length if necessary
        let rs_len = 255;
        if data.len() < rs_len {
            data.resize(rs_len, 0);
        } else {
            data.truncate(rs_len);
        }

        // 4a. CCSDS de-randomize
        fec::ccsds_derandomize(&mut data);

        // 4b. Reed-Solomon decode
        let rs_result = fec::rs_decode(&data);
        let rs_data = match rs_result {
            Ok(d) => d,
            Err(e) => {
                println!("  RS decode failed: {}", e);
                continue;
            }
        };
        println!(
            "  RS decoded: {} bytes (32 parity bytes stripped)",
            rs_data.len()
        );

        // 4c. CRC-32C verify
        let payload = match fec::crc32c_verify_and_strip(&rs_data) {
            Ok(p) => p,
            Err(e) => {
                println!("  CRC check failed: {}", e);
                continue;
            }
        };

        valid += 1;
        println!("  ✓ Valid frame! CSP payload: {} bytes", payload.len());
        print!("  Hex: ");
        for (i, byte) in payload.iter().enumerate() {
            if i > 0 && i % 16 == 0 {
                print!("\n        ");
            }
            print!("{:02X} ", byte);
        }
        println!();
    }

    println!("\n{}/{} frames decoded successfully.", valid, frames.len());
    Ok(())
}
