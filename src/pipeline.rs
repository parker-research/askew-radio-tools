//! Ties the whole decode chain together for one audio file:
//! audio -> DSP front-end -> AX100 ASM+Golay framing -> Golay+RS -> CSP CRC.

use std::collections::HashSet;

use serde::Serialize;

use crate::audio::AudioSamples;
use crate::{DecodeError, audio, dsp, fec, framing};

/// One decoded AX100 Mode 5 / CSP beacon frame, ready to be serialised as a
/// JSONL record.
#[derive(Debug, Clone, Serialize)]
pub struct BeaconRecord {
    pub filename: String,
    pub data_length_bytes: usize,
    pub time_in_file_ms: f64,
    pub rs_corrected_error_count: Option<u32>,
    /// `true` unless Reed-Solomon found more errors in the frame than it
    /// can correct (>16 symbol errors) — in that case `data_hex` is a
    /// best-effort, likely-still-corrupt payload, kept in the output
    /// rather than dropped so callers can see/inspect what was received.
    pub rs_correctable: bool,
    /// `true`/`false` if the CSP frame's header declares a CRC trailer and
    /// it matches/doesn't; `None` if the frame doesn't include one (per
    /// the CSP header's `crc` flag) — there's nothing to check.
    pub crc_pass: Option<bool>,
    pub data_hex: String,
}

/// Run the full decode pipeline on one audio file and return every frame
/// whose Golay-coded header decoded successfully (frames where the header
/// itself is uncorrectable are dropped — there's no reliable frame length
/// to extract anything from). Frames where Reed-Solomon couldn't correct
/// all errors are still included, with `rs_correctable: false`.
pub fn decode_file(path: &str) -> Result<Vec<BeaconRecord>, DecodeError> {
    let audio = audio::load_audio(path)?;
    Ok(decode_audio(&audio, path))
}

/// Same as [`decode_file`], but operating on already-loaded audio (so
/// callers that also want to run [`crate::audio_check`] don't have to
/// decode the file twice).
///
/// Runs the DSP front-end (a close port of gr-satellites'
/// `fsk_demodulator` — see the module doc on `dsp.rs`) once over the whole
/// file and decodes every frame whose Golay-coded header decodes.
pub fn decode_audio(audio: &AudioSamples, filename: &str) -> Vec<BeaconRecord> {
    let mut seen_payloads: HashSet<Vec<u8>> = HashSet::new();
    let mut records = Vec::new();

    let bitstream = dsp::fm_discriminate_and_filter(audio);
    let raw_frames = framing::find_frames(&bitstream.bits);

    for raw in &raw_frames {
        let decoded = match fec::ax100_asm_golay_decode(&raw.data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if !seen_payloads.insert(decoded.payload.clone()) {
            continue; // duplicate syncword hit on the same real frame
        }

        let crc_pass = fec::csp_crc32c_check(&decoded.payload);
        let time_in_file_ms = bitstream
            .bit_times_ms
            .get(raw.sync_bit_offset)
            .copied()
            .unwrap_or(0.0);
        // Round to microsecond precision (i.e. the nearest 0.001 ms).
        let time_in_file_ms = (time_in_file_ms * 1000.0).round() / 1000.0;

        records.push(BeaconRecord {
            filename: filename.to_string(),
            data_length_bytes: decoded.payload.len(),
            time_in_file_ms,
            rs_corrected_error_count: decoded.rs_corrected_error_count,
            rs_correctable: decoded.rs_correctable,
            crc_pass,
            data_hex: hex_encode(&decoded.payload),
        });
    }

    records.sort_by(|a, b| a.time_in_file_ms.total_cmp(&b.time_in_file_ms));
    records
}

fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
