//! Ties the whole decode chain together for one audio file:
//! audio -> DSP front-end -> AX100 ASM+Golay framing -> Golay+RS -> CSP CRC.

use std::collections::HashSet;

use serde::Serialize;

use crate::audio::AudioSamples;
use crate::{audio, dsp, fec, framing, DecodeError};

/// One decoded AX100 Mode 5 / CSP beacon frame, ready to be serialised as a
/// JSONL record.
#[derive(Debug, Clone, Serialize)]
pub struct BeaconRecord {
    pub filename: String,
    pub data_hex: String,
    pub data_length_bytes: usize,
    pub start_time_in_file_ms: f64,
    pub rs_corrected_error_count: u32,
    pub crc_pass: bool,
}

/// Run the full decode pipeline on one audio file and return every frame
/// that passed Golay+Reed-Solomon decoding (frames that fail either are
/// dropped, matching `u482c_decode_impl`'s behaviour of only publishing on
/// success).
pub fn decode_file(path: &str) -> Result<Vec<BeaconRecord>, DecodeError> {
    let audio = audio::load_audio(path)?;
    Ok(decode_audio(&audio, path))
}

/// Same as [`decode_file`], but operating on already-loaded audio (so
/// callers that also want to run [`crate::audio_check`] don't have to
/// decode the file twice).
///
/// Runs the DSP front-end once per gain pair in
/// [`dsp::GARDNER_GAIN_CANDIDATES`] and merges the decoded frames
/// (deduplicated by payload bytes). A single fixed symbol-timing loop
/// bandwidth isn't reliable across a whole multi-minute capture — see the
/// comment on `gardner_ted` in `dsp.rs` — so trying a few and taking the
/// union catches real frames that any one gain's phase drift would miss.
pub fn decode_audio(audio: &AudioSamples, filename: &str) -> Vec<BeaconRecord> {
    let mut seen_payloads: HashSet<Vec<u8>> = HashSet::new();
    let mut records = Vec::new();

    for &(alpha, beta) in dsp::GARDNER_GAIN_CANDIDATES {
        let bitstream = dsp::fm_discriminate_and_filter_with_gains(audio, alpha, beta);
        let raw_frames = framing::find_frames(&bitstream.bits);

        for raw in &raw_frames {
            let (payload, rs_corrected_error_count) = match fec::ax100_asm_golay_decode(&raw.data)
            {
                Ok(v) => v,
                Err(_) => continue,
            };

            if !seen_payloads.insert(payload.clone()) {
                continue; // already found via an earlier gain pair
            }

            let crc_pass = fec::csp_crc32c_check(&payload);
            let start_time_in_file_ms = bitstream
                .bit_times_ms
                .get(raw.sync_bit_offset)
                .copied()
                .unwrap_or(0.0);

            records.push(BeaconRecord {
                filename: filename.to_string(),
                data_hex: hex_encode(&payload),
                data_length_bytes: payload.len(),
                start_time_in_file_ms,
                rs_corrected_error_count,
                crc_pass,
            });
        }
    }

    records.sort_by(|a, b| a.start_time_in_file_ms.total_cmp(&b.start_time_in_file_ms));
    records
}

fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
