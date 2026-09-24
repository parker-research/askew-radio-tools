//! Syncword search and fixed-length frame extraction for AX100 "ASM+Golay"
//! framing.
//!
//! Ported closely from gr-satellites' `ax100_deframer` (`mode='ASM'`) and
//! its `sync_to_pdu_packed` hierarchical block:
//!
//! ```text
//! sliced bits -> correlate_access_code_tag_bb(
//!     sync='10010011000010110101000111011110', threshold=4)
//!   -> fixedlen_to_pdu(packlen=258*8)
//! ```
//!
//! Unlike AX100's "RS" mode, ASM mode does **not** run a bit-level
//! self-synchronizing descrambler ahead of the syncword search
//! (`digital.descrambler_bb` is only wired in for `mode='RS'` in
//! `ax100_deframer.py`) — the sliced bits are searched directly. Byte-level
//! CCSDS derandomization happens later, inside the FEC stage
//! ([`crate::fec::ax100_asm_golay_decode`]), and only covers the payload
//! region (not the 3-byte Golay header).
//!
//! Frame layout (the [`ASM_FRAME_LEN_BYTES`] bytes captured after the
//! syncword):
//! ```text
//! byte 0..3:   Golay(24,12)-encoded length/flags header
//! byte 3..258: up to 255 bytes of [CSP frame | 32 RS parity bytes]
//!              (CCSDS-scrambled), zero-padded at the end if the
//!              transmitted frame was shorter than 255 bytes
//! ```

use crate::fec::ASM_FRAME_LEN_BYTES;

/// 32-bit AX100 syncword, matching gr-satellites'
/// `_syncword = '10010011000010110101000111011110'` (shared default for
/// both ASM and RS modes).
pub const SYNC_WORD: u32 = 0x930B_51DE;

/// Bit errors tolerated in the syncword (gr-satellites' default
/// `syncword_threshold`).
pub const SYNC_THRESHOLD: u32 = 4;

/// A raw frame extracted from the bit stream, before Golay/RS decoding.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Bit index where the syncword started.
    pub sync_bit_offset: usize,
    /// Number of bit errors in the matched syncword (0..=[`SYNC_THRESHOLD`]
    /// from [`find_frames`]; up to
    /// [`crate::pipeline::CHAIN_SYNC_THRESHOLD`] for a frame whose position
    /// a neighbouring frame predicted). Only ~1 in 10^5 random bit
    /// positions matches within the (gr-satellites default) threshold of
    /// 4, but a multi-minute capture holds millions of them, so
    /// noise-driven hits are routine — see
    /// [`crate::pipeline::FrameTier::classify`], which uses this to decide
    /// whether a frame RS couldn't vouch for is believable.
    pub sync_bit_errors: u32,
    /// The `ASM_FRAME_LEN_BYTES` raw bytes following the syncword.
    pub data: [u8; ASM_FRAME_LEN_BYTES],
    /// How trustworthy each byte of `data` is: the smallest soft-symbol
    /// magnitude among its 8 bits (one weak bit is all it takes to corrupt
    /// a byte). Only comparable within one frame; feeds
    /// [`crate::fec::ax100_rs_decode_with_erasures`].
    pub byte_reliability: [f32; ASM_FRAME_LEN_BYTES],
}

/// Search `bits` (as recovered by the slicer, with no descrambling) for all
/// AX100 ASM+Golay candidate frames. `soft` holds the soft symbol each bit
/// was sliced from (same length as `bits`); only its magnitudes are used.
pub fn find_frames(bits: &[bool], soft: &[f32]) -> Vec<RawFrame> {
    debug_assert_eq!(bits.len(), soft.len());
    let n = bits.len();
    let frame_bits = ASM_FRAME_LEN_BYTES * 8;
    let mut frames = Vec::new();

    if n < 32 + frame_bits {
        return frames;
    }

    // Scan every bit position, like GNU Radio's `correlate_access_code_tag_bb`
    // (a streaming tagger, not a state machine) — it does *not* skip ahead
    // after a match. That matters here: a weak/coincidental near-match a
    // few bits before a real frame is common enough (threshold=4 over 32
    // bits) that skipping past it would swallow the real frame's syncword
    // before we ever get to test it.
    // The 32 bits starting at `i`, as a shift register: one bit in per
    // position instead of re-reading all 32 (same value as
    // `read_u32_msb(bits, i)`).
    let mut window = read_u32_msb(bits, 0);
    for i in 0..=(n - 32) {
        if i > 0 {
            window = (window << 1) | (bits[i + 31] as u32);
        }
        let errors = (window ^ SYNC_WORD).count_ones();

        if errors <= SYNC_THRESHOLD {
            let payload_start = i + 32;
            if payload_start + frame_bits <= n {
                frames.push(extract_frame(bits, soft, i, errors));
            }
        }
    }

    frames
}

/// Bit errors between [`SYNC_WORD`] and the 32 bits of `bits` starting at
/// `offset`.
pub fn sync_bit_errors_at(bits: &[bool], offset: usize) -> u32 {
    (read_u32_msb(bits, offset) ^ SYNC_WORD).count_ones()
}

/// Cut the frame whose syncword starts at bit `sync_bit_offset` out of
/// `bits`/`soft`. The caller guarantees the whole frame is in range.
pub fn extract_frame(
    bits: &[bool],
    soft: &[f32],
    sync_bit_offset: usize,
    sync_bit_errors: u32,
) -> RawFrame {
    let payload_start = sync_bit_offset + 32;
    let frame_soft = &soft[payload_start..payload_start + ASM_FRAME_LEN_BYTES * 8];
    let mean_magnitude = mean_soft_magnitude(frame_soft);
    let mut byte_reliability = [0f32; ASM_FRAME_LEN_BYTES];
    for (byte_idx, r) in byte_reliability.iter_mut().enumerate() {
        *r = frame_soft[byte_idx * 8..byte_idx * 8 + 8]
            .iter()
            .map(|&s| bit_reliability(s, mean_magnitude))
            .fold(f32::INFINITY, f32::min);
    }
    RawFrame {
        sync_bit_offset,
        sync_bit_errors,
        data: bits_to_frame(bits, payload_start),
        byte_reliability,
    }
}

/// Mean of `|soft|` — the scale [`bit_reliability`] measures against.
pub fn mean_soft_magnitude(soft: &[f32]) -> f32 {
    if soft.is_empty() {
        return 0.0;
    }
    soft.iter().map(|s| s.abs()).sum::<f32>() / soft.len() as f32
}

/// How trustworthy a bit sliced from soft symbol `soft` is, given the
/// surrounding frame's mean soft magnitude: larger is more reliable, and
/// values at or below zero mean "no better than a coin flip". Comparable
/// between bits (and frames); used to pick erasures and to weight copies
/// when combining.
///
/// The obvious measure — `|soft|`, distance from the decision threshold —
/// is only half right for this signal. SatNOGS audio is the output of an
/// FM discriminator, and at the low SNRs where bits start failing, FM
/// "click" noise dominates: a momentary 2*pi phase slip in the receiver
/// injects a pulse several times the signal's own deviation, flipping the
/// symbol it lands on *and* making it look unusually confident. Measured
/// over 190k bits of known codewords on SatNOGS observation 15039753, the
/// error rate is U-shaped in `r = |soft| / mean|soft|`: lowest (<1%) near
/// `r = 1`, rising to 12-39% as `r` falls below 0.3 as expected — and
/// *also* rising, to 29% at `r = 2..2.5` and 66% beyond, as it grows. A
/// bit at large `r` is about as trustworthy as one at `1.09 - 0.41 * r`
/// on the small side, which is what this returns there.
pub fn bit_reliability(soft: f32, mean_magnitude: f32) -> f32 {
    if mean_magnitude <= 0.0 {
        return soft.abs();
    }
    let r = soft.abs() / mean_magnitude;
    r.min(CLICK_RELIABILITY_INTERCEPT - CLICK_RELIABILITY_SLOPE * r)
}

/// See [`bit_reliability`].
const CLICK_RELIABILITY_INTERCEPT: f32 = 1.09;
/// See [`bit_reliability`].
const CLICK_RELIABILITY_SLOPE: f32 = 0.41;

fn read_u32_msb(bits: &[bool], offset: usize) -> u32 {
    let mut v = 0u32;
    for k in 0..32 {
        v = (v << 1) | (bits[offset + k] as u32);
    }
    v
}

fn bits_to_frame(bits: &[bool], offset: usize) -> [u8; ASM_FRAME_LEN_BYTES] {
    let mut out = [0u8; ASM_FRAME_LEN_BYTES];
    for (byte_idx, byte) in out.iter_mut().enumerate() {
        let mut b = 0u8;
        for bit in 0..8 {
            b = (b << 1) | (bits[offset + byte_idx * 8 + bit] as u8);
        }
        *byte = b;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame_bits(preamble_bits: usize, frame: &[u8; ASM_FRAME_LEN_BYTES]) -> Vec<bool> {
        let mut bits = Vec::new();
        for _ in 0..preamble_bits {
            bits.push(false);
            bits.push(true);
        }
        for k in (0..32).rev() {
            bits.push((SYNC_WORD >> k) & 1 == 1);
        }
        for &byte in frame {
            for bit in (0..8).rev() {
                bits.push((byte >> bit) & 1 == 1);
            }
        }
        bits
    }

    #[test]
    fn test_find_frames_recovers_one_frame() {
        let mut frame = [0u8; ASM_FRAME_LEN_BYTES];
        frame[0] = 0xAB;
        for (i, b) in frame.iter_mut().enumerate().skip(1) {
            *b = i as u8;
        }

        let bits = make_frame_bits(64, &frame);
        let frames = find_frames(&bits, &vec![1.0; bits.len()]);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, frame);
        assert_eq!(frames[0].sync_bit_errors, 0);
    }

    #[test]
    fn test_find_frames_tolerates_bit_errors_in_syncword() {
        let frame = [0u8; ASM_FRAME_LEN_BYTES];
        let mut bits = make_frame_bits(0, &frame);

        bits[3] = !bits[3];
        bits[10] = !bits[10];

        let frames = find_frames(&bits, &vec![1.0; bits.len()]);
        assert_eq!(frames.len(), 1, "should tolerate 2 bit errors (<=4)");
        assert_eq!(frames[0].sync_bit_errors, 2);
    }

    #[test]
    fn test_bit_reliability_distrusts_click_sized_symbols() {
        let mean = 1.0;
        // Rises from zero at the decision threshold...
        assert!(bit_reliability(0.1, mean) < bit_reliability(0.5, mean));
        assert!(bit_reliability(0.5, mean) < bit_reliability(0.75, mean));
        // ...but falls again for implausibly large symbols (FM clicks),
        // down to "worthless" well above the normal level.
        assert!(bit_reliability(2.0, mean) < bit_reliability(1.0, mean));
        assert!(bit_reliability(2.0, mean) < bit_reliability(0.5, mean));
        assert!(bit_reliability(3.0, mean) <= 0.0);
        // Only relative magnitude matters, and sign never does.
        assert_eq!(bit_reliability(-0.4, 2.0), bit_reliability(0.2, 1.0));
    }

    #[test]
    fn test_extract_frame_marks_the_weakest_bit_of_each_byte() {
        let frame = [0u8; ASM_FRAME_LEN_BYTES];
        let bits = make_frame_bits(0, &frame);
        let mut soft = vec![1.0f32; bits.len()];
        // Byte 5: one bit barely off the threshold. Byte 9: one click-sized.
        soft[32 + 5 * 8 + 3] = 0.05;
        soft[32 + 9 * 8 + 6] = 3.0;
        let raw = extract_frame(&bits, &soft, 0, 0);
        let typical = raw.byte_reliability[0];
        assert!(raw.byte_reliability[5] < typical);
        assert!(raw.byte_reliability[9] < typical);
        assert_eq!(raw.data, frame);
    }

    #[test]
    fn test_find_frames_empty_on_short_input() {
        let bits = vec![false; 100];
        assert!(find_frames(&bits, &vec![1.0; bits.len()]).is_empty());
    }
}
