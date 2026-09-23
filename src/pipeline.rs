//! Ties the whole decode chain together for one audio file:
//! audio -> DSP front-end -> AX100 ASM+Golay framing -> Golay+RS -> CSP CRC.

use std::collections::HashMap;

use serde::Serialize;

use crate::audio::AudioSamples;
use crate::fec::AsmGolayDecoded;
use crate::framing::RawFrame;
use crate::{DecodeError, audio, dsp, fec, framing};

// ---------------------------------------------------------------------------
// How much the decoder trusts a frame
// ---------------------------------------------------------------------------
//
// A frame that decodes RS-clean *and* passes its CSP CRC32C proves itself:
// the odds of noise producing a valid RS(255,223) codeword whose trailing
// CRC also matches are negligible. Everything else has to earn its place
// from the 56 bits that sit in front of the payload — the syncword and the
// Golay-coded header — because that's the only part of the frame whose
// correct value we know independently of the payload.
//
// Those 56 bits are weak evidence on their own, which is why "keep
// RS-uncorrectable frames so callers can inspect them" is, unqualified, a
// recipe for emitting mostly noise. On a multi-minute 9600-baud capture
// the bit stream holds ~10^6 candidate positions per demodulator pass,
// and:
//
//   * the syncword search accepts <=4 bit errors over 32 bits (the
//     gr-satellites default), i.e. 41449/2^32 ~ 1e-5 of random positions,
//     so a few dozen noise hits per file is normal, not exceptional; and
//   * Golay(24,12) corrects up to 3 bit errors, so it "successfully"
//     decodes 2325/4096 = 57% of the random headers behind those hits.
//
// Tightening each of those to the *low-error* end of its range is what
// costs noise dearly while costing real frames nothing, since a real burst
// that is clean enough to sync on is overwhelmingly clean enough to sync
// on exactly. Measured over seven real SatNOGS captures (1455 candidate
// frames, 261 of them RS+CRC-verified), the [`FrameTier::Believable`]
// gates keep 261/261 verified frames while admitting only 151 of the 1194
// unverified ones, and what survives has the transmitted length
// distribution of real FrontierSat traffic (86/138/170/238 bytes on the
// wire) rather than a uniform one.
//
// Frames are labelled rather than dropped: deciding how much evidence is
// worth printing belongs to the caller (see `main.rs`'s `--output-filter`),
// not to the decoder.

/// Syncword bit errors tolerated for a frame RS couldn't verify. The full
/// [`framing::SYNC_THRESHOLD`] of 4 is ~80x more likely to fire on noise
/// than this is (41449 vs 529 of the 2^32 possible words).
pub const MAX_UNVERIFIED_SYNC_BIT_ERRORS: u32 = 1;

/// Golay header bit errors tolerated for a frame RS couldn't verify.
/// Golay's full 3-bit correction radius swallows 57% of random headers;
/// a 1-bit radius swallows 25/4096 = 0.6% of them.
pub const MAX_UNVERIFIED_GOLAY_BIT_ERRORS: u32 = 1;

/// The header flag nibble (`[unused][RS][scrambler][viterbi]`) every real
/// frame from this project's satellites carries. `ax100_deframer`'s ASM
/// path ignores these bits when *decoding* (it uses its own fixed
/// configuration — see [`fec::ax100_asm_golay_decode`]), which leaves them
/// free to serve as 4 bits of known-value check digit: noise spreads
/// uniformly over all 16 values, real frames only ever show this one.
///
/// This is a property of *this project's* transmitter, not of AX100 in
/// general — it is the single strongest noise discriminator the decoder
/// has, so a satellite whose firmware sets any of these bits needs this
/// constant updated to match.
pub const EXPECTED_HEADER_FLAGS: u8 = 0x0;

/// Shortest CSP payload that could be a real frame: a 4-byte CSP header
/// plus the 4-byte CRC32C trailer, with nothing in between.
pub const MIN_CSP_PAYLOAD_LEN: usize = 8;

/// How much evidence there is that a decoded frame was really transmitted,
/// weakest first. Ordering is the point of the type: a caller keeps
/// everything at or above the tier it's willing to trust, and
/// [`decode_audio`] uses it to pick a winner when two demodulator passes
/// recover the same payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameTier {
    /// The Golay-coded header decoded, and nothing beyond that holds up.
    /// Consistent with a noise-driven syncword hit, and on a real capture
    /// usually *is* one — these outnumber genuine frames several-fold.
    Candidate,
    /// Reed-Solomon couldn't correct the codeword, so `data_hex` is
    /// best-effort and probably still corrupt — but the syncword and Golay
    /// header are clean enough that noise is an implausible explanation
    /// for them. A real burst, received too badly to repair.
    Believable,
    /// Reed-Solomon decoded the codeword, but the CSP CRC32C trailer
    /// didn't verify. On real captures this tier stays empty: RS success
    /// and CRC success have so far always coincided.
    RsCorrectableCrcError,
    /// Reed-Solomon decoded the codeword *and* its CRC32C verifies. The
    /// frame is exactly what the satellite sent.
    Verified,
}

impl FrameTier {
    /// Classify one candidate frame. See the module-level commentary above
    /// for the measurements behind the [`Believable`](Self::Believable)
    /// thresholds.
    pub fn classify(raw: &RawFrame, decoded: &AsmGolayDecoded, crc_pass: Option<bool>) -> Self {
        if decoded.rs_correctable {
            return if crc_pass == Some(true) {
                FrameTier::Verified
            } else {
                FrameTier::RsCorrectableCrcError
            };
        }

        let header_too_clean_for_noise = raw.sync_bit_errors <= MAX_UNVERIFIED_SYNC_BIT_ERRORS
            && decoded.golay_corrected_bit_count <= MAX_UNVERIFIED_GOLAY_BIT_ERRORS
            && decoded.header_flags == EXPECTED_HEADER_FLAGS
            && decoded.payload.len() >= MIN_CSP_PAYLOAD_LEN;

        if header_too_clean_for_noise {
            FrameTier::Believable
        } else {
            FrameTier::Candidate
        }
    }
}

/// One decoded AX100 Mode 5 / CSP frame, ready to be serialised as a
/// JSONL record.
#[derive(Debug, Clone, Serialize)]
pub struct PacketRecord {
    pub filename: String,
    pub data_length_bytes: usize,
    /// Start of the frame's syncword, snapped to the nearest two-symbol
    /// period (e.g. 2 / 9600 baud ≈ 0.2083 ms) and rounded to 0.001 ms, so it's
    /// identical across platforms and build profiles.
    pub time_in_file_ms: f64,
    /// How much evidence there is that this frame was really transmitted.
    /// Callers that want only real traffic keep
    /// [`FrameTier::Believable`] and above.
    pub tier: FrameTier,
    /// Bit errors in this frame's syncword match (0..=4), bit errors the
    /// Golay(24,12) decoder corrected in its 3-byte header (0..=3), and
    /// that header's flag nibble (`[unused][RS][scrambler][viterbi]`; 0 on
    /// every real frame — see [`EXPECTED_HEADER_FLAGS`]). All three are
    /// the inputs to [`FrameTier::classify`]: on a frame RS couldn't
    /// correct, they are the whole of the evidence that it was a real
    /// transmission rather than noise, so they're reported rather than
    /// left implicit in `tier`.
    pub sync_bit_errors: u32,
    pub golay_corrected_bit_count: u32,
    pub header_flags: u8,
    pub rs_corrected_error_count: Option<u32>,
    /// `true` unless Reed-Solomon found more errors in the frame than it
    /// can correct (>16 symbol errors) — in that case `data_hex` is a
    /// best-effort, likely-still-corrupt payload, kept in the output
    /// rather than dropped so callers can see/inspect what was received.
    /// Cross-check `tier` before reading anything into such a payload:
    /// [`FrameTier::Candidate`] means the decoder has no evidence the
    /// frame was transmitted at all.
    pub rs_correctable: bool,
    /// `true`/`false` for whether the frame's trailing CRC32C matches;
    /// `None` only if the frame is too short to even hold a 4-byte
    /// trailer. See [`fec::csp_crc32c_check`] for why this doesn't depend
    /// on the CSP header's `crc` flag.
    pub crc_pass: Option<bool>,
    /// Relative received signal strength over the frame (syncword through
    /// payload), in dB. Measured from the demodulator's local signal
    /// amplitude just before AGC normalisation — see
    /// [`dsp::BitStream::bit_rssi_db`] — so it isn't calibrated to an
    /// absolute RF power, but is comparable between packets within (and
    /// across) files: a higher value means a relatively stronger receive.
    pub rssi_db: f64,
    pub data_hex: String,
}

/// Run the full decode pipeline on one audio file and return every
/// candidate frame whose Golay-coded header decoded, each labelled with a
/// [`FrameTier`]. Nothing is filtered here: most of what comes back from a
/// multi-minute capture is [`FrameTier::Candidate`] noise, and it is the
/// caller's job to decide how much evidence it wants (see `main.rs`'s
/// `--output-filter`, which defaults to [`FrameTier::Verified`] only).
///
/// `baud_rate` is the transmitter's symbol rate in Hz (9600 for
/// FRONTIERSAT — see [`dsp::DEFAULT_SYMBOL_RATE_HZ`]).
pub fn decode_file(path: &str, baud_rate: f64) -> Result<Vec<PacketRecord>, DecodeError> {
    let audio = audio::load_audio(path)?;
    Ok(decode_audio(&audio, path, baud_rate))
}

/// Same as [`decode_file`], but operating on already-loaded audio (so
/// callers that also want to run [`crate::audio_check`] don't have to
/// decode the file twice).
///
/// Runs the DSP front-end (a close port of gr-satellites'
/// `fsk_demodulator` — see the module doc on `dsp.rs`) once per loop
/// bandwidth in [`dsp::CLK_BW_CANDIDATES`], plus the structurally
/// different Mueller-Müller front-end
/// ([`dsp::fm_discriminate_and_filter_mueller_muller`]), and merges all
/// the decoded frames, deduplicated by payload bytes: the highest
/// [`FrameTier`] wins, and within a tier the first pass to find it wins —
/// the accurately-timestamped `CLK_BW_CANDIDATES` passes run first, so
/// they take any tie against the Mueller-Müller pass's less-precisely-
/// compensated timestamps. Tier has to outrank arrival order here,
/// because the same payload can surface as a marginal
/// [`FrameTier::Candidate`] in one pass and a clean
/// [`FrameTier::Verified`] frame in the next; keeping the first sighting
/// would throw away the better evidence. A single continuous timing loop over a whole
/// multi-minute capture can still momentarily lose lock at one specific
/// point and miss a real burst there, even well-tuned — see
/// [`dsp::CLK_BW`]'s comment — so trying a couple of nearby bandwidths
/// *and* a structurally different algorithm, then taking the union,
/// catches frames that no amount of tuning the primary chain alone would
/// recover.
pub fn decode_audio(audio: &AudioSamples, filename: &str, baud_rate: f64) -> Vec<PacketRecord> {
    // payload bytes -> index into `records` of the best decode seen so far.
    let mut best_by_payload: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut records = Vec::new();

    for bitstream in
        dsp::fm_discriminate_and_filter_multi_bw(audio, baud_rate, dsp::CLK_BW_CANDIDATES)
    {
        decode_bitstream(
            &bitstream,
            filename,
            baud_rate,
            &mut best_by_payload,
            &mut records,
        );
    }

    let mm_bitstream = dsp::fm_discriminate_and_filter_mueller_muller(audio, baud_rate);
    decode_bitstream(
        &mm_bitstream,
        filename,
        baud_rate,
        &mut best_by_payload,
        &mut records,
    );

    records.sort_by(|a, b| a.time_in_file_ms.total_cmp(&b.time_in_file_ms));
    records
}

/// Search one [`dsp::BitStream`] for frames and merge them into `records`,
/// deduplicated across calls by payload bytes via `best_by_payload` (see
/// [`decode_audio`] for the tie-breaking rule).
fn decode_bitstream(
    bitstream: &dsp::BitStream,
    filename: &str,
    baud_rate: f64,
    best_by_payload: &mut HashMap<Vec<u8>, usize>,
    records: &mut Vec<PacketRecord>,
) {
    let raw_frames = framing::find_frames(&bitstream.bits);

    for raw in &raw_frames {
        let decoded = match fec::ax100_asm_golay_decode(&raw.data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let crc_pass = fec::csp_crc32c_check(&decoded.payload);
        let tier = FrameTier::classify(raw, &decoded, crc_pass);

        // Duplicate syncword hit on the same real frame? Keep whichever
        // decode carries the stronger evidence, and leave the incumbent in
        // place on a tie so the earlier (better-timestamped) pass wins.
        let incumbent = best_by_payload.get(&decoded.payload).copied();
        if let Some(index) = incumbent
            && records[index].tier >= tier
        {
            continue;
        }

        let time_in_file_ms = quantize_time_in_file_ms(
            bitstream
                .bit_times_ms
                .get(raw.sync_bit_offset)
                .copied()
                .unwrap_or(0.0),
            baud_rate,
        );

        let frame_end_bit =
            (raw.sync_bit_offset + 32 + raw.data.len() * 8).min(bitstream.bit_rssi_db.len());
        let rssi_db = mean_rssi_db(&bitstream.bit_rssi_db[raw.sync_bit_offset..frame_end_bit]);

        let record = PacketRecord {
            filename: filename.to_string(),
            data_length_bytes: decoded.payload.len(),
            time_in_file_ms,
            tier,
            sync_bit_errors: raw.sync_bit_errors,
            golay_corrected_bit_count: decoded.golay_corrected_bit_count,
            header_flags: decoded.header_flags,
            rs_corrected_error_count: decoded.rs_corrected_error_count,
            rs_correctable: decoded.rs_correctable,
            crc_pass,
            rssi_db,
            data_hex: hex_encode(&decoded.payload),
        };

        match incumbent {
            Some(index) => records[index] = record,
            None => {
                best_by_payload.insert(decoded.payload, records.len());
                records.push(record);
            }
        }
    }
}

/// Snap a frame's start time to the nearest multiple of two symbol periods
/// (e.g. 2 / 9600 baud ≈ 0.2083 ms), then round to 3 decimal places (in ms).
///
/// The timing loop's sub-symbol interpolation is built on transcendental
/// float functions whose last-bit results differ between platform math
/// libraries (glibc vs macOS vs Windows) and between debug and release
/// builds, so the raw interpolated time isn't reproducible across them.
/// Snapping to a coarse grid, using only exactly-rounded IEEE operations,
/// makes the reported time the same everywhere.
fn quantize_time_in_file_ms(time_in_file_ms: f64, baud_rate: f64) -> f64 {
    const GRID_SYMBOLS: f64 = 2.0;
    let grid_steps = (time_in_file_ms * baud_rate / (GRID_SYMBOLS * 1000.0)).round();
    let snapped_ms = grid_steps * (GRID_SYMBOLS * 1000.0) / baud_rate;
    (snapped_ms * 1000.0).round() / 1000.0
}

/// Average a frame's per-bit RSSI (dB) values, rounded to 0.01 dB. Averages
/// in the power domain (mean of the underlying RMS-squared values, not a
/// plain mean of dB) since dB is already a log quantity.
fn mean_rssi_db(bit_rssi_db: &[f64]) -> f64 {
    if bit_rssi_db.is_empty() {
        // Finite silence floor, not `-inf` — this value flows into JSON
        // output (`serde_json` can't serialize non-finite floats).
        return -240.0;
    }
    let mean_power: f64 = bit_rssi_db
        .iter()
        .map(|&db| 10f64.powf(db / 10.0))
        .sum::<f64>()
        / bit_rssi_db.len() as f64;
    (10.0 * mean_power.log10() * 100.0).round() / 100.0
}

fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::ASM_FRAME_LEN_BYTES;

    #[test]
    fn test_quantize_time_in_file_ms_snaps_to_two_symbol_grid() {
        // Grid step is 2 / 9600 s = 0.208333... ms.
        assert_eq!(quantize_time_in_file_ms(0.0, 9600.0), 0.0);
        assert_eq!(quantize_time_in_file_ms(0.1, 9600.0), 0.0);
        assert_eq!(quantize_time_in_file_ms(0.11, 9600.0), 0.208);
        assert_eq!(quantize_time_in_file_ms(0.3, 9600.0), 0.208);
        assert_eq!(quantize_time_in_file_ms(0.32, 9600.0), 0.417);
        // Sub-symbol jitter around a grid point collapses to the same value.
        assert_eq!(quantize_time_in_file_ms(150182.85, 9600.0), 150182.917);
        assert_eq!(quantize_time_in_file_ms(150182.99, 9600.0), 150182.917);
        assert_eq!(quantize_time_in_file_ms(150183.0, 9600.0), 150182.917);
    }

    /// A frame that is clean on every axis the classifier looks at, but
    /// that RS could not correct — the shape of a real burst received too
    /// badly to repair.
    fn clean_header_but_rs_failed() -> (RawFrame, AsmGolayDecoded) {
        (
            RawFrame {
                sync_bit_offset: 0,
                sync_bit_errors: 0,
                data: [0u8; ASM_FRAME_LEN_BYTES],
            },
            AsmGolayDecoded {
                golay_corrected_bit_count: 0,
                header_flags: EXPECTED_HEADER_FLAGS,
                frame_len: 170,
                payload: vec![0u8; 138],
                rs_corrected_error_count: None,
                rs_correctable: false,
            },
        )
    }

    fn classify(raw: &RawFrame, decoded: &AsmGolayDecoded, crc_pass: Option<bool>) -> FrameTier {
        FrameTier::classify(raw, decoded, crc_pass)
    }

    #[test]
    fn test_tiers_are_ordered_weakest_first() {
        assert!(FrameTier::Candidate < FrameTier::Believable);
        assert!(FrameTier::Believable < FrameTier::RsCorrectableCrcError);
        assert!(FrameTier::RsCorrectableCrcError < FrameTier::Verified);
    }

    #[test]
    fn test_rs_and_crc_together_verify_a_frame_whatever_its_header_looked_like() {
        let (mut raw, mut decoded) = clean_header_but_rs_failed();
        decoded.rs_correctable = true;
        decoded.rs_corrected_error_count = Some(16);
        // Deliberately fail every one of the header checks: a valid RS
        // codeword with a matching CRC stands on its own.
        raw.sync_bit_errors = framing::SYNC_THRESHOLD;
        decoded.golay_corrected_bit_count = 3;
        decoded.header_flags = 0xf;

        assert_eq!(classify(&raw, &decoded, Some(true)), FrameTier::Verified);
    }

    #[test]
    fn test_rs_correctable_crc_error_without_a_matching_crc_is_its_own_tier() {
        let (raw, mut decoded) = clean_header_but_rs_failed();
        decoded.rs_correctable = true;
        decoded.rs_corrected_error_count = Some(0);

        assert_eq!(
            classify(&raw, &decoded, Some(false)),
            FrameTier::RsCorrectableCrcError
        );
        assert_eq!(
            classify(&raw, &decoded, None),
            FrameTier::RsCorrectableCrcError
        );
    }

    #[test]
    fn test_clean_header_rescues_an_rs_uncorrectable_frame() {
        let (raw, decoded) = clean_header_but_rs_failed();
        assert_eq!(classify(&raw, &decoded, Some(false)), FrameTier::Believable);
    }

    #[test]
    fn test_sloppy_syncword_demotes_to_candidate() {
        let (mut raw, decoded) = clean_header_but_rs_failed();
        raw.sync_bit_errors = MAX_UNVERIFIED_SYNC_BIT_ERRORS + 1;
        assert_eq!(classify(&raw, &decoded, Some(false)), FrameTier::Candidate);
    }

    #[test]
    fn test_heavily_corrected_golay_header_demotes_to_candidate() {
        let (raw, mut decoded) = clean_header_but_rs_failed();
        decoded.golay_corrected_bit_count = MAX_UNVERIFIED_GOLAY_BIT_ERRORS + 1;
        assert_eq!(classify(&raw, &decoded, Some(false)), FrameTier::Candidate);
    }

    #[test]
    fn test_unexpected_header_flags_demote_to_candidate() {
        let (raw, mut decoded) = clean_header_but_rs_failed();
        for flags in 1..=0xfu8 {
            decoded.header_flags = flags;
            assert_eq!(
                classify(&raw, &decoded, Some(false)),
                FrameTier::Candidate,
                "header flags {flags:#x} should not be accepted"
            );
        }
    }

    #[test]
    fn test_payload_too_short_to_hold_a_csp_frame_demotes_to_candidate() {
        let (raw, mut decoded) = clean_header_but_rs_failed();
        decoded.payload = vec![0u8; MIN_CSP_PAYLOAD_LEN - 1];
        assert_eq!(classify(&raw, &decoded, None), FrameTier::Candidate);
    }

    #[test]
    fn test_tier_serialises_as_a_snake_case_string() {
        let json = serde_json::to_string(&FrameTier::RsCorrectableCrcError).unwrap();
        assert_eq!(json, "\"rs_correctable_crc_error\"");
    }
}
