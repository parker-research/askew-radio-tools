//! Ties the whole decode chain together for one audio file:
//! audio -> DSP front-end -> AX100 Mode 5 framing -> Reed-Solomon -> CSP CRC.

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
/// that passed Reed-Solomon decoding (frames RS can't correct are dropped,
/// matching `ax100_decode_impl`'s behaviour of only publishing on success).
pub fn decode_file(path: &str) -> Result<Vec<BeaconRecord>, DecodeError> {
    let audio = audio::load_audio(path)?;
    Ok(decode_audio(&audio, path))
}

/// Same as [`decode_file`], but operating on already-loaded audio (so
/// callers that also want to run [`crate::audio_check`] don't have to
/// decode the file twice).
pub fn decode_audio(audio: &AudioSamples, filename: &str) -> Vec<BeaconRecord> {
    let bitstream = dsp::fm_discriminate_and_filter(audio);
    let raw_frames = framing::find_frames(&bitstream.bits);

    let mut records = Vec::with_capacity(raw_frames.len());
    for raw in &raw_frames {
        let (payload, rs_corrected_error_count) = match fec::ax100_rs_decode(&raw.data) {
            Ok(v) => v,
            Err(_) => continue,
        };

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

    records
}

fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
