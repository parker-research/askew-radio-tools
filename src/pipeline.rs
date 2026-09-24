//! Ties the whole decode chain together for one audio file:
//! audio -> DSP front-end -> AX100 ASM+Golay framing -> Golay+RS -> CSP CRC.

use std::collections::{BTreeSet, HashMap, HashSet};

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
    /// Bit errors in this frame's syncword match (0..=4 from the blind
    /// search; up to [`CHAIN_SYNC_THRESHOLD`] for a frame found by chain
    /// following), bit errors in its 3-byte Golay(24,12) header (0..=3
    /// when Golay corrected it; more when the length was instead taken
    /// from the frame's chain neighbours — see `decode_bitstream`), and
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
    /// How many *other* receptions of this same frame — retransmissions of
    /// it at other times in the file — had their soft symbols combined
    /// with this one's to decode it. 0 (the usual case) means this
    /// reception decoded on its own. See `combine_retransmissions` for
    /// what is checked before a frame is credited this way; when this is
    /// non-zero, `rs_corrected_error_count` counts the bytes of *this*
    /// reception that differed from the decoded codeword, which can exceed
    /// what RS could have corrected alone.
    pub combined_copies: u32,
    /// Relative received signal strength over the frame (syncword through
    /// payload), in dB. Measured from the demodulator's local signal
    /// amplitude just before AGC normalisation — see
    /// [`dsp::BitStream::bit_power`] — so it isn't calibrated to an
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
/// Runs an ensemble of demodulator front-ends over the audio and merges
/// what each recovers, since no single timing-recovery method catches
/// every frame:
///
/// - the Gardner loop (a close port of gr-satellites' `fsk_demodulator` —
///   see the module doc on `dsp.rs`) once per loop bandwidth in
///   [`dsp::CLK_BW_CANDIDATES`]: a single continuous loop over a
///   multi-minute capture can still momentarily lose lock at one specific
///   point and miss a real burst there, even well-tuned (see
///   [`dsp::CLK_BW`]'s comment);
/// - feed-forward (Oerder-Meyr) timing once per window in
///   [`dsp::FEEDFORWARD_WINDOW_SYMBOLS`], which has no loop jitter on long
///   runs of back-to-back frames;
/// - the structurally different Mueller-Müller front-end
///   ([`dsp::fm_discriminate_and_filter_mueller_muller`]).
///
/// Each pass's frames go through `decode_bitstream` (syncword search,
/// soft-decision RS, and chain following) and are merged one record per
/// transmission — see `Collected::insert`: when passes recover the same
/// transmission, the highest [`FrameTier`] wins, and within a tier the
/// first pass to find it wins (the accurately-timestamped Gardner passes
/// run first, so they take any tie). Tier has to outrank arrival order
/// here, because the same payload can surface as a marginal
/// [`FrameTier::Candidate`] in one pass and a clean
/// [`FrameTier::Verified`] frame in the next.
///
/// Finally, receptions no pass could verify on its own are combined with
/// retransmissions of the same frame elsewhere in the file
/// (`combine_retransmissions`).
pub fn decode_audio(audio: &AudioSamples, filename: &str, baud_rate: f64) -> Vec<PacketRecord> {
    let mut collected = Collected::default();

    for bitstream in dsp::fm_discriminate_and_filter_ensemble(
        audio,
        baud_rate,
        dsp::CLK_BW_CANDIDATES,
        dsp::FEEDFORWARD_WINDOW_SYMBOLS,
    ) {
        decode_bitstream(&bitstream, filename, baud_rate, &mut collected);
    }

    let mm_bitstream = dsp::fm_discriminate_and_filter_mueller_muller(audio, baud_rate);
    decode_bitstream(&mm_bitstream, filename, baud_rate, &mut collected);

    combine_retransmissions(&mut collected, filename);

    let mut records = collected.records;
    records.sort_by(|a, b| a.time_in_file_ms.total_cmp(&b.time_in_file_ms));
    records
}

/// Syncword bit errors tolerated at a bit position where a neighbouring
/// frame says the next (or previous) frame must start — far looser than
/// [`framing::SYNC_THRESHOLD`]'s blind search, because the position isn't
/// a guess. Nothing is emitted on the strength of it: a frame found this
/// way only counts if Reed-Solomon *and* its CRC32C verify.
pub const CHAIN_SYNC_THRESHOLD: u32 = 12;

/// How many consecutive undecodable frames a chain of back-to-back frames
/// may step over (assuming each has the same length as the last good one)
/// while still looking for the next decodable one.
pub const MAX_CHAIN_GAP_FRAMES: usize = 2;

/// Payload bytes -> indices into the record list of every transmission of
/// that payload decoded so far (one per distinct moment it was sent).
type PayloadIndex = HashMap<Vec<u8>, Vec<usize>>;

/// Two decodes of the same payload whose start times are at most this far
/// apart are the same transmission, recovered by different demodulator
/// passes (whose timestamps agree to well under a millisecond). Anything
/// further apart is a retransmission. Far shorter than the shortest frame
/// (~30 ms on the air at 9600 baud), so two back-to-back sends of one
/// payload can't be confused for one.
const SAME_TRANSMISSION_WINDOW_MS: f64 = 10.0;

/// Bits one frame occupies on the air: syncword, Golay header, codeword.
fn frame_span_bits(frame_len: usize) -> usize {
    32 + (fec::HEADER_LEN + frame_len) * 8
}

/// Search one [`dsp::BitStream`] for frames and merge them into
/// `collected` (see `Collected::insert` for how the passes' results are
/// deduplicated).
///
/// Two stages:
///
/// 1. A blind syncword search ([`framing::find_frames`]), each hit decoded
///    by hard-decision RS and, failing that, by RS with the least reliable
///    bytes erased ([`fec::ax100_rs_decode_with_erasures`]).
/// 2. Chain following. Satellites send long runs of frames back-to-back
///    with no gap or preamble between them — hundreds in a row during a
///    bulk downlink — so every real frame pins down exactly where its
///    neighbours start: the next syncword begins on the bit right after
///    this frame's codeword ends. Walking the chain from every frame
///    believed to be real (forwards, stepping over up to
///    [`MAX_CHAIN_GAP_FRAMES`] undecodable ones, and backwards) finds
///    frames the blind search can't: syncwords too corrupted for its
///    threshold, and Golay headers that failed or decoded to the wrong
///    length (the lengths seen elsewhere in the chain are tried instead).
///    Only CRC-verified frames come out of this stage.
fn decode_bitstream(
    bitstream: &dsp::BitStream,
    filename: &str,
    baud_rate: f64,
    collected: &mut Collected,
) {
    let mut sink = RecordSink {
        bitstream,
        filename,
        baud_rate,
        collected,
    };

    // (sync bit offset, frame length) of every frame believed to be real.
    let mut anchors: Vec<(usize, usize)> = Vec::new();
    // Sync offsets of the frames this bitstream has already verified.
    let mut verified_offsets: HashSet<usize> = HashSet::new();
    // Frame lengths seen on verified frames: the candidates tried when a
    // chain position's own header can't be trusted.
    let mut verified_lens: BTreeSet<usize> = BTreeSet::new();

    // Positions worth handing to `RecordSink::decode_at`, with a length
    // hint (0 for none).
    let mut to_try: Vec<(usize, usize)> = Vec::new();

    for raw in framing::find_frames(&bitstream.bits, &bitstream.soft) {
        let mut decoded = match fec::ax100_asm_golay_decode(&raw.data) {
            Ok(v) => v,
            Err(_) => {
                // An uncorrectable header behind a near-perfect syncword is
                // far more likely a real frame whose header took a burst of
                // errors than noise: try it with the lengths real frames in
                // this bitstream turn out to have.
                if raw.sync_bit_errors <= MAX_UNVERIFIED_SYNC_BIT_ERRORS {
                    to_try.push((raw.sync_bit_offset, 0));
                }
                continue;
            }
        };

        // Hard-decision RS failed: retry with the frame's least reliable
        // bytes erased. Only a CRC-verified result is taken.
        if !decoded.rs_correctable
            && let Some(soft) = fec::ax100_rs_decode_with_erasures(
                &raw.data,
                decoded.frame_len,
                &raw.byte_reliability,
            )
        {
            decoded.payload = soft.payload;
            decoded.rs_corrected_error_count = Some(soft.corrected_symbols);
            decoded.rs_correctable = true;
        }

        let crc_pass = fec::csp_crc32c_check(&decoded.payload);
        let tier = FrameTier::classify(&raw, &decoded, crc_pass);
        if tier >= FrameTier::Believable {
            anchors.push((raw.sync_bit_offset, decoded.frame_len));
        }
        if tier == FrameTier::Verified {
            verified_offsets.insert(raw.sync_bit_offset);
            verified_lens.insert(decoded.frame_len);
        }
        sink.merge(&raw, decoded, crc_pass, tier);
    }

    // Believable frames RS couldn't fix may just have had their length
    // header misread: give them the chain treatment at their own position
    // too, not only at their neighbours'.
    to_try.extend(
        anchors
            .iter()
            .filter(|(offset, _)| !verified_offsets.contains(offset))
            .copied(),
    );
    let mut tried: HashSet<usize> = HashSet::new();
    while let Some((offset, frame_len)) = to_try.pop() {
        if tried.insert(offset)
            && let Some(len) = sink.decode_at(offset, frame_len, &verified_lens)
        {
            verified_offsets.insert(offset);
            verified_lens.insert(len);
            anchors.push((offset, len));
        }
    }

    while let Some((offset, frame_len)) = anchors.pop() {
        // Forwards: the next frame starts right where this one ends.
        let mut next = offset;
        let mut assumed_len = frame_len;
        for _ in 0..=MAX_CHAIN_GAP_FRAMES {
            next += frame_span_bits(assumed_len);
            if verified_offsets.contains(&next) || !tried.insert(next) {
                break;
            }
            if let Some(len) = sink.decode_at(next, frame_len, &verified_lens) {
                verified_offsets.insert(next);
                verified_lens.insert(len);
                anchors.push((next, len));
                break;
            }
            assumed_len = frame_len;
        }

        // Backwards: this frame starts right where the previous one ended,
        // for whichever length the previous one had.
        let prev_lens: Vec<usize> = verified_lens.iter().copied().chain([frame_len]).collect();
        for prev_len in prev_lens {
            let Some(prev) = offset.checked_sub(frame_span_bits(prev_len)) else {
                continue;
            };
            if verified_offsets.contains(&prev) || !tried.insert(prev) {
                continue;
            }
            if let Some(len) = sink.decode_at(prev, prev_len, &verified_lens) {
                verified_offsets.insert(prev);
                verified_lens.insert(len);
                anchors.push((prev, len));
            }
        }
    }
}

/// Where `decode_bitstream` puts what it finds: turns decoded frames into
/// [`PacketRecord`]s and merges them into the file-wide, payload-
/// deduplicated record list.
struct RecordSink<'a> {
    bitstream: &'a dsp::BitStream,
    filename: &'a str,
    baud_rate: f64,
    collected: &'a mut Collected,
}

impl RecordSink<'_> {
    /// Try to decode a frame whose syncword a chain neighbour says starts
    /// at bit `offset`. Lengths tried, in order: whatever the frame's own
    /// Golay header says (if it decodes), `len_hint`, then `known_lens`.
    /// Merges and returns the length of the first CRC-verified decode.
    fn decode_at(
        &mut self,
        offset: usize,
        len_hint: usize,
        known_lens: &BTreeSet<usize>,
    ) -> Option<usize> {
        let bits = &self.bitstream.bits;
        if offset + 32 + fec::ASM_FRAME_LEN_BYTES * 8 > bits.len() {
            return None;
        }
        let sync_bit_errors = framing::sync_bit_errors_at(bits, offset);
        if sync_bit_errors > CHAIN_SYNC_THRESHOLD {
            return None;
        }
        let raw = framing::extract_frame(bits, &self.bitstream.soft, offset, sync_bit_errors);

        let header = fec::ax100_asm_golay_decode(&raw.data).ok();
        let mut lens: Vec<usize> = Vec::new();
        for len in header
            .as_ref()
            .map(|h| h.frame_len)
            .into_iter()
            .chain([len_hint])
            .chain(known_lens.iter().copied())
        {
            if !lens.contains(&len) {
                lens.push(len);
            }
        }

        for frame_len in lens {
            let Some(soft) =
                fec::ax100_rs_decode_with_erasures(&raw.data, frame_len, &raw.byte_reliability)
            else {
                continue;
            };
            let header_flags = match &header {
                Some(h) if h.frame_len == frame_len => h.header_flags,
                _ => EXPECTED_HEADER_FLAGS,
            };
            let decoded = AsmGolayDecoded {
                golay_corrected_bit_count: fec::header_bit_errors(
                    &raw.data,
                    frame_len,
                    header_flags,
                ),
                header_flags,
                frame_len,
                payload: soft.payload,
                rs_corrected_error_count: Some(soft.corrected_symbols),
                rs_correctable: true,
            };
            self.merge(&raw, decoded, Some(true), FrameTier::Verified);
            return Some(frame_len);
        }

        // Undecodable on its own, but a neighbour vouches for a frame being
        // here: keep its soft symbols in case a retransmission elsewhere in
        // the file can make up the difference (see
        // `combine_retransmissions`, which checks the copies really match
        // before combining them, so keeping one where there turns out to
        // be nothing costs only a comparison).
        let (copy_len, golay_corrected_bit_count, header_flags) = match &header {
            Some(h) if h.frame_len == len_hint || known_lens.contains(&h.frame_len) => {
                (h.frame_len, h.golay_corrected_bit_count, h.header_flags)
            }
            _ if len_hint > 0 => (
                len_hint,
                fec::header_bit_errors(&raw.data, len_hint, EXPECTED_HEADER_FLAGS),
                EXPECTED_HEADER_FLAGS,
            ),
            _ => return None,
        };
        let copy = self.soft_copy(&raw, copy_len, golay_corrected_bit_count, header_flags);
        self.collected.copies.push(copy);
        None
    }

    /// Package `raw`'s soft symbols (header and `frame_len`-byte codeword)
    /// for `combine_retransmissions`.
    fn soft_copy(
        &self,
        raw: &RawFrame,
        frame_len: usize,
        golay_corrected_bit_count: u32,
        header_flags: u8,
    ) -> SoftCopy {
        let bitstream = self.bitstream;
        let time_in_file_ms = quantize_time_in_file_ms(
            bitstream
                .bit_times_ms
                .get(raw.sync_bit_offset)
                .copied()
                .unwrap_or(0.0),
            self.baud_rate,
        );
        let frame_end_bit =
            (raw.sync_bit_offset + 32 + raw.data.len() * 8).min(bitstream.bit_power.len());
        let rssi_db = mean_rssi_db(&bitstream.bit_power[raw.sync_bit_offset..frame_end_bit]);
        let start = raw.sync_bit_offset + 32;
        let soft = bitstream.soft[start..start + (fec::HEADER_LEN + frame_len) * 8].to_vec();
        SoftCopy {
            time_in_file_ms,
            frame_len,
            snr: soft_snr(&soft[fec::HEADER_LEN * 8..]),
            soft,
            sync_bit_errors: raw.sync_bit_errors,
            golay_corrected_bit_count,
            header_flags,
            rssi_db,
        }
    }

    /// Record one decoded frame (see `Collected::insert`), and keep its
    /// soft symbols for `combine_retransmissions` if it's believed real.
    fn merge(
        &mut self,
        raw: &RawFrame,
        decoded: AsmGolayDecoded,
        crc_pass: Option<bool>,
        tier: FrameTier,
    ) {
        let copy = self.soft_copy(
            raw,
            decoded.frame_len,
            decoded.golay_corrected_bit_count,
            decoded.header_flags,
        );
        let (time_in_file_ms, rssi_db) = (copy.time_in_file_ms, copy.rssi_db);
        if tier >= FrameTier::Believable {
            self.collected.copies.push(copy);
        }

        let record = PacketRecord {
            filename: self.filename.to_string(),
            data_length_bytes: decoded.payload.len(),
            time_in_file_ms,
            tier,
            sync_bit_errors: raw.sync_bit_errors,
            golay_corrected_bit_count: decoded.golay_corrected_bit_count,
            header_flags: decoded.header_flags,
            rs_corrected_error_count: decoded.rs_corrected_error_count,
            rs_correctable: decoded.rs_correctable,
            crc_pass,
            combined_copies: 0,
            rssi_db,
            data_hex: hex_encode(&decoded.payload),
        };
        self.collected.insert(decoded.payload, record);
    }
}

/// Everything decoded from one file so far, across all demodulator passes.
#[derive(Default)]
struct Collected {
    records: Vec<PacketRecord>,
    best_by_payload: PayloadIndex,
    /// Soft symbols of every reception believed real (tier
    /// [`FrameTier::Believable`] and up), from every pass — the raw
    /// material for `combine_retransmissions`.
    copies: Vec<SoftCopy>,
}

impl Collected {
    /// Add `record` (whose payload is `payload`), unless an equal-or-better
    /// decode of the same transmission is already recorded.
    ///
    /// Duplicate syncword hit on the same real frame (another pass, or
    /// another syncword hit of this one, at the same moment)? Keep
    /// whichever decode carries the stronger evidence, and leave the
    /// incumbent in place on a tie so the earlier (better-timestamped) pass
    /// wins. The same payload at a different time is a separate
    /// transmission — satellites resend whole bulk downlinks — and gets its
    /// own record.
    fn insert(&mut self, payload: Vec<u8>, record: PacketRecord) {
        let incumbent = self.best_by_payload.get(&payload).and_then(|indices| {
            indices.iter().copied().find(|&index| {
                (self.records[index].time_in_file_ms - record.time_in_file_ms).abs()
                    <= SAME_TRANSMISSION_WINDOW_MS
            })
        });
        match incumbent {
            Some(index) if self.records[index].tier >= record.tier => {}
            Some(index) => self.records[index] = record,
            None => {
                self.best_by_payload
                    .entry(payload)
                    .or_default()
                    .push(self.records.len());
                self.records.push(record);
            }
        }
    }

    /// Whether a CRC-verified record already covers the transmission at
    /// `time_in_file_ms`.
    fn has_verified_at(&self, time_in_file_ms: f64) -> bool {
        self.records.iter().any(|r| {
            r.tier == FrameTier::Verified
                && (r.time_in_file_ms - time_in_file_ms).abs() <= SAME_TRANSMISSION_WINDOW_MS
        })
    }
}

// ---------------------------------------------------------------------------
// Combining retransmissions
// ---------------------------------------------------------------------------
//
// Satellites resend: a bulk file downlink is often sent in full several
// times per pass, and the copies are bit-for-bit identical on the air. A
// frame received too noisily to decode — most often at the low-elevation
// ends of a pass — may well have a copy elsewhere in the file, and adding
// two noisy receptions' soft symbols together is worth up to 3 dB of SNR,
// which is the whole difference between a frame RS can't correct and one
// it can.
//
// Copies are found by bit similarity alone (no decoding needed): two
// receptions of one frame differ only where noise flipped bits, a few
// percent at most, while unrelated frames differ in ~50% of bits, and even
// beacons with mostly-constant telemetry in 15%+.
//
// The combined decode must then pass RS *and* the CSP CRC32C, and — before
// the frame is credited to a reception — that reception's own bits have
// to be within [`MAX_COMBINED_OWN_BIT_DISAGREEMENT`] of the decoded
// codeword. That last check is what rules out crediting a reception with
// a *different* frame's content: two distinct RS(255,223) codewords differ
// in at least 33 bytes, which even at the code's minimum distance is ~130
// bits (every such byte differing in ~4 of its 8 bits) — more than 6.5%
// of even the longest frame, and far more of a short one.

/// A reception believed to be real, kept for `combine_retransmissions`.
struct SoftCopy {
    time_in_file_ms: f64,
    frame_len: usize,
    /// Soft symbols of the Golay header and codeword, in on-air order.
    soft: Vec<f32>,
    /// Soft-symbol SNR over the codeword (linear; see [`soft_snr`]).
    snr: f64,
    sync_bit_errors: u32,
    golay_corrected_bit_count: u32,
    header_flags: u8,
    rssi_db: f64,
}

/// Largest fraction of codeword bits two receptions may disagree on and
/// still be combined as copies of one frame. Well above the few percent
/// noise causes between two real copies, well below the 15%+ of
/// look-alike beacons.
pub const MAX_COPY_BIT_DISAGREEMENT: f64 = 0.10;

/// Largest fraction of its own codeword bits a reception may disagree with
/// a combined decode on and still be credited with it. See the section
/// comment above for why this can't admit a different frame's content.
pub const MAX_COMBINED_OWN_BIT_DISAGREEMENT: f64 = 0.05;

/// `mean(|soft|)^2 / var(|soft|)`: how cleanly the soft symbols separate
/// from the decision threshold (linear, not dB). Used to weight copies
/// when combining them.
fn soft_snr(soft: &[f32]) -> f64 {
    if soft.is_empty() {
        return 0.0;
    }
    let n = soft.len() as f64;
    let mean = soft.iter().map(|s| s.abs() as f64).sum::<f64>() / n;
    let var = soft
        .iter()
        .map(|s| (s.abs() as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    if var > 0.0 { mean * mean / var } else { 0.0 }
}

/// Fraction of positions where `a` and `b` (soft symbols, compared by sign
/// only) disagree.
fn sign_disagreement(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let differing = a
        .iter()
        .zip(b)
        .filter(|&(&x, &y)| (x >= 0.0) != (y >= 0.0))
        .count();
    differing as f64 / n as f64
}

/// Try to decode every believed-real reception that no pass could verify
/// on its own by combining it with other receptions of the same frame
/// (see the section comment above), adding a CRC-verified record for each
/// one this rescues.
fn combine_retransmissions(collected: &mut Collected, filename: &str) {
    // One copy per transmission: the cleanest of the passes' receptions.
    // Passes can disagree on a frame's length (a misread header), so it's
    // one per length the transmission was read with.
    let mut copies: Vec<&SoftCopy> = collected.copies.iter().collect();
    copies.sort_by(|a, b| a.time_in_file_ms.total_cmp(&b.time_in_file_ms));
    let mut transmissions: Vec<&SoftCopy> = Vec::new();
    let mut cluster_start = 0;
    for copy in copies {
        if transmissions.last().is_some_and(|last| {
            copy.time_in_file_ms - last.time_in_file_ms > SAME_TRANSMISSION_WINDOW_MS
        }) {
            cluster_start = transmissions.len();
        }
        match transmissions[cluster_start..]
            .iter_mut()
            .find(|t| t.frame_len == copy.frame_len)
        {
            Some(best) if copy.snr > best.snr => *best = copy,
            Some(_) => {}
            None => transmissions.push(copy),
        }
    }

    let header_bits = fec::HEADER_LEN * 8;
    let mut rescued: Vec<(Vec<u8>, PacketRecord)> = Vec::new();
    for target in &transmissions {
        if collected.has_verified_at(target.time_in_file_ms) {
            continue;
        }
        let target_codeword = &target.soft[header_bits..];

        let mut partners: Vec<(f64, &SoftCopy)> = transmissions
            .iter()
            .filter(|other| {
                other.frame_len == target.frame_len
                    && (other.time_in_file_ms - target.time_in_file_ms).abs()
                        > SAME_TRANSMISSION_WINDOW_MS
            })
            .map(|other| {
                (
                    sign_disagreement(target_codeword, &other.soft[header_bits..]),
                    *other,
                )
            })
            .filter(|&(disagreement, _)| disagreement <= MAX_COPY_BIT_DISAGREEMENT)
            .collect();
        if partners.is_empty() {
            continue;
        }
        partners.sort_by(|a, b| a.0.total_cmp(&b.0));

        // All the copies first; if a stray look-alike spoils that, just the
        // closest one.
        let partner_sets: Vec<Vec<&SoftCopy>> = if partners.len() > 1 {
            vec![
                partners.iter().map(|&(_, p)| p).collect(),
                vec![partners[0].1],
            ]
        } else {
            vec![vec![partners[0].1]]
        };

        for partner_set in partner_sets {
            if let Some(result) = decode_combined(target, &partner_set, filename) {
                rescued.push(result);
                break;
            }
        }
    }

    for (payload, record) in rescued {
        collected.insert(payload, record);
    }
}

/// Combine `target`'s soft symbols with `partners'` (each bit voting with
/// its reliability, weighted by its copy's SNR), decode the result, and —
/// if it verifies and `target`'s own bits agree with it (see
/// [`MAX_COMBINED_OWN_BIT_DISAGREEMENT`]) — build `target`'s record.
fn decode_combined(
    target: &SoftCopy,
    partners: &[&SoftCopy],
    filename: &str,
) -> Option<(Vec<u8>, PacketRecord)> {
    let n_bits = target.soft.len();
    let mut combined = vec![0f32; n_bits];
    for copy in std::iter::once(&target).chain(partners.iter()) {
        // Each bit's vote is its reliability (so a click-inflated symbol
        // counts for little or nothing, rather than for the most — see
        // `framing::bit_reliability`), scaled by how clean the copy is
        // overall.
        let mean_magnitude = framing::mean_soft_magnitude(&copy.soft);
        let weight = copy.snr as f32;
        for (acc, &s) in combined.iter_mut().zip(&copy.soft) {
            let vote = framing::bit_reliability(s, mean_magnitude).max(0.0);
            *acc += weight * vote.copysign(s);
        }
    }

    // Lay the combined decisions out as a frame (header + codeword, the
    // rest zero) for the RS stage.
    let mut frame = [0u8; fec::ASM_FRAME_LEN_BYTES];
    let mut byte_reliability = [f32::INFINITY; fec::ASM_FRAME_LEN_BYTES];
    for (i, &s) in combined.iter().enumerate() {
        if s >= 0.0 {
            frame[i / 8] |= 0x80 >> (i % 8);
        }
        byte_reliability[i / 8] = byte_reliability[i / 8].min(s.abs());
    }

    let decoded = fec::ax100_rs_decode_with_erasures(&frame, target.frame_len, &byte_reliability)?;

    // Credit the frame to `target` only if its own reception matches it.
    let header_bits = fec::HEADER_LEN * 8;
    let own = &target.soft[header_bits..];
    let mut differing_bits = 0usize;
    let mut differing_bytes = 0u32;
    for (byte_idx, &on_air) in decoded.codeword_on_air.iter().enumerate() {
        let mut received = 0u8;
        for bit in 0..8 {
            if own[byte_idx * 8 + bit] >= 0.0 {
                received |= 0x80 >> bit;
            }
        }
        let diff = (received ^ on_air).count_ones() as usize;
        differing_bits += diff;
        differing_bytes += u32::from(diff > 0);
    }
    if differing_bits as f64 / own.len() as f64 > MAX_COMBINED_OWN_BIT_DISAGREEMENT {
        return None;
    }

    let record = PacketRecord {
        filename: filename.to_string(),
        data_length_bytes: decoded.payload.len(),
        time_in_file_ms: target.time_in_file_ms,
        tier: FrameTier::Verified,
        sync_bit_errors: target.sync_bit_errors,
        golay_corrected_bit_count: target.golay_corrected_bit_count,
        header_flags: target.header_flags,
        rs_corrected_error_count: Some(differing_bytes),
        rs_correctable: true,
        crc_pass: Some(true),
        combined_copies: partners.len() as u32,
        rssi_db: target.rssi_db,
        data_hex: hex_encode(&decoded.payload),
    };
    Some((decoded.payload, record))
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

/// Average a frame's per-bit signal power (linear, see
/// [`dsp::BitStream::bit_power`]) and express it in dB, rounded to 0.01 dB.
/// Averaging happens in the power domain, not over per-bit dB values,
/// since dB is already a log quantity.
fn mean_rssi_db(bit_power: &[f64]) -> f64 {
    if bit_power.is_empty() {
        // Finite silence floor, not `-inf` — this value flows into JSON
        // output (`serde_json` can't serialize non-finite floats).
        return -240.0;
    }
    let mean_power: f64 = bit_power.iter().sum::<f64>() / bit_power.len() as f64;
    (10.0 * libm::log10(mean_power) * 100.0).round() / 100.0
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

    // --- Synthetic bitstreams of back-to-back frames ---

    /// On-air bits of one frame: syncword, then the header and codeword
    /// (the zero padding [`fec::encode_frame_for_test`] adds past the
    /// codeword isn't transmitted).
    fn on_air_bits(payload: &[u8]) -> Vec<bool> {
        let frame = fec::encode_frame_for_test(payload);
        let n_bytes = fec::HEADER_LEN + payload.len() + 32;
        let mut bits: Vec<bool> = (0..32)
            .rev()
            .map(|k| (framing::SYNC_WORD >> k) & 1 == 1)
            .collect();
        for &byte in &frame[..n_bytes] {
            bits.extend((0..8).rev().map(|k| (byte >> k) & 1 == 1));
        }
        bits
    }

    /// Deterministic filler bits (idle/noise between and after frames).
    fn filler(n: usize, seed: u32) -> Vec<bool> {
        (0..n as u32)
            .map(|i| (i ^ seed).wrapping_mul(2654435761).rotate_left(11) & 4 != 0)
            .collect()
    }

    /// A clean bitstream carrying `bits`, at 9600 baud from t = 0.
    fn bitstream_of(bits: Vec<bool>) -> dsp::BitStream {
        let n = bits.len();
        dsp::BitStream {
            soft: bits.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect(),
            bits,
            bit_times_ms: (0..n).map(|i| i as f64 / 9.6).collect(),
            bit_power: vec![1.0; n],
            recovered_symbol_rate: 9600.0,
        }
    }

    fn verified_payloads(records: &[PacketRecord]) -> Vec<String> {
        records
            .iter()
            .filter(|r| r.tier == FrameTier::Verified)
            .map(|r| r.data_hex.clone())
            .collect()
    }

    #[test]
    fn test_chain_following_finds_a_frame_whose_syncword_is_too_corrupt_to_search_for() {
        let payloads: Vec<Vec<u8>> = (0..3).map(|i| fec::csp_payload_for_test(100, i)).collect();
        let mut bits = filler(300, 1);
        let second_sync = bits.len() + on_air_bits(&payloads[0]).len();
        for p in &payloads {
            bits.extend(on_air_bits(p));
        }
        bits.extend(filler(fec::ASM_FRAME_LEN_BYTES * 8, 2));
        // 8 syncword errors: twice what the blind search tolerates.
        for k in [0, 3, 7, 12, 18, 21, 26, 30] {
            bits[second_sync + k] = !bits[second_sync + k];
        }
        let bitstream = bitstream_of(bits);
        assert!(
            framing::find_frames(&bitstream.bits, &bitstream.soft)
                .iter()
                .all(|f| f.sync_bit_offset != second_sync)
        );

        let mut collected = Collected::default();
        decode_bitstream(&bitstream, "t", 9600.0, &mut collected);

        let found = verified_payloads(&collected.records);
        for p in &payloads {
            assert!(found.contains(&hex_encode(p)), "missing {}", hex_encode(p));
        }
    }

    #[test]
    fn test_uncorrectable_header_behind_a_clean_syncword_is_tried_at_known_lengths() {
        let first = fec::csp_payload_for_test(120, 4);
        let second = fec::csp_payload_for_test(120, 5);
        let mut bits = filler(300, 3);
        bits.extend(on_air_bits(&first));
        bits.extend(filler(777, 4));
        let header_start = bits.len() + 32;
        bits.extend(on_air_bits(&second));
        bits.extend(filler(fec::ASM_FRAME_LEN_BYTES * 8, 5));
        // 4 header errors: always detected, never corrected, by Golay(24,12).
        for k in [1, 6, 13, 20] {
            bits[header_start + k] = !bits[header_start + k];
        }

        let mut collected = Collected::default();
        decode_bitstream(&bitstream_of(bits), "t", 9600.0, &mut collected);

        let found = verified_payloads(&collected.records);
        assert!(found.contains(&hex_encode(&first)));
        assert!(found.contains(&hex_encode(&second)));
        let record = collected
            .records
            .iter()
            .find(|r| r.data_hex == hex_encode(&second))
            .unwrap();
        assert_eq!(record.golay_corrected_bit_count, 4);
    }

    fn test_record(payload: &[u8], time_in_file_ms: f64, tier: FrameTier) -> PacketRecord {
        PacketRecord {
            filename: "t".into(),
            data_length_bytes: payload.len(),
            time_in_file_ms,
            tier,
            sync_bit_errors: 0,
            golay_corrected_bit_count: 0,
            header_flags: 0,
            rs_corrected_error_count: None,
            rs_correctable: tier >= FrameTier::RsCorrectableCrcError,
            crc_pass: None,
            combined_copies: 0,
            rssi_db: 0.0,
            data_hex: hex_encode(payload),
        }
    }

    #[test]
    fn test_same_payload_is_one_record_per_transmission() {
        let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut collected = Collected::default();
        collected.insert(
            payload.clone(),
            test_record(&payload, 1000.0, FrameTier::Believable),
        );
        // Another pass's sighting of the same transmission: upgrades it.
        collected.insert(
            payload.clone(),
            test_record(&payload, 1000.2, FrameTier::Verified),
        );
        // A weaker sighting of it again: ignored.
        collected.insert(
            payload.clone(),
            test_record(&payload, 999.9, FrameTier::Candidate),
        );
        // The same payload a minute later is a retransmission: kept.
        collected.insert(
            payload.clone(),
            test_record(&payload, 61000.0, FrameTier::Verified),
        );

        let times: Vec<(f64, FrameTier)> = collected
            .records
            .iter()
            .map(|r| (r.time_in_file_ms, r.tier))
            .collect();
        assert_eq!(
            times,
            vec![
                (1000.2, FrameTier::Verified),
                (61000.0, FrameTier::Verified)
            ]
        );
    }

    /// A noisy reception of `payload`'s header + codeword: soft symbols of
    /// magnitude 1, except that one bit in each of the `bad_bytes`
    /// codeword bytes is received weakly with the wrong sign.
    fn noisy_copy(payload: &[u8], time_in_file_ms: f64, bad_bytes: &[usize]) -> SoftCopy {
        let bits = on_air_bits(payload);
        let mut soft: Vec<f32> = bits[32..]
            .iter()
            .map(|&b| if b { 1.0 } else { -1.0 })
            .collect();
        for &byte in bad_bytes {
            let bit = (fec::HEADER_LEN + byte) * 8 + byte % 8;
            soft[bit] *= -0.3;
        }
        SoftCopy {
            time_in_file_ms,
            frame_len: payload.len() + 32,
            snr: soft_snr(&soft[fec::HEADER_LEN * 8..]),
            soft,
            sync_bit_errors: 0,
            golay_corrected_bit_count: 0,
            header_flags: 0,
            rssi_db: 0.0,
        }
    }

    #[test]
    fn test_retransmissions_too_noisy_alone_decode_combined() {
        let payload = fec::csp_payload_for_test(150, 9);
        // 40 corrupted bytes each — beyond RS even with erasures — but
        // never the same bytes in both copies.
        let first: Vec<usize> = (0..40).map(|i| i * 4).collect();
        let second: Vec<usize> = (0..40).map(|i| i * 4 + 2).collect();
        let mut collected = Collected::default();
        collected.copies.push(noisy_copy(&payload, 1000.0, &first));
        collected
            .copies
            .push(noisy_copy(&payload, 90000.0, &second));

        combine_retransmissions(&mut collected, "t");

        let mut records = collected.records.clone();
        records.sort_by(|a, b| a.time_in_file_ms.total_cmp(&b.time_in_file_ms));
        assert_eq!(records.len(), 2);
        for (record, time) in records.iter().zip([1000.0, 90000.0]) {
            assert_eq!(record.time_in_file_ms, time);
            assert_eq!(record.tier, FrameTier::Verified);
            assert_eq!(record.data_hex, hex_encode(&payload));
            assert_eq!(record.combined_copies, 1);
            assert_eq!(record.rs_corrected_error_count, Some(40));
        }
    }

    #[test]
    fn test_combining_never_credits_a_reception_with_a_different_frame() {
        // Two frames one byte apart, each received twice, all four too
        // noisy to decode alone.
        let a = fec::csp_payload_for_test(150, 11);
        let mut b = a.clone();
        b[40] ^= 0x10;
        let crc = crc32c::crc32c(&b[..b.len() - 4]);
        let n = b.len();
        b[n - 4..].copy_from_slice(&crc.to_be_bytes());

        let bad = |offset: usize| -> Vec<usize> { (0..40).map(|i| i * 4 + offset).collect() };
        let truth = [(1000.0, &a), (2000.0, &b), (90000.0, &a), (91000.0, &b)];
        let mut collected = Collected::default();
        for (i, (time, payload)) in truth.iter().enumerate() {
            collected
                .copies
                .push(noisy_copy(payload, *time, &bad(i % 4)));
        }

        combine_retransmissions(&mut collected, "t");
        // Each frame's own two copies combine; neither frame's copies are
        // credited with the other's content.
        assert_eq!(collected.records.len(), 4);

        for record in &collected.records {
            let (_, payload) = truth
                .iter()
                .find(|(t, _)| *t == record.time_in_file_ms)
                .unwrap();
            assert_eq!(
                record.data_hex,
                hex_encode(payload),
                "reception at {} credited with the wrong frame",
                record.time_in_file_ms
            );
        }
    }

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
                byte_reliability: [1.0; ASM_FRAME_LEN_BYTES],
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
