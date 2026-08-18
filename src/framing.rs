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
    /// The `ASM_FRAME_LEN_BYTES` raw bytes following the syncword.
    pub data: [u8; ASM_FRAME_LEN_BYTES],
}

/// Search `bits` (as recovered by the slicer, with no descrambling) for all
/// AX100 ASM+Golay candidate frames.
pub fn find_frames(bits: &[bool]) -> Vec<RawFrame> {
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
    for i in 0..=(n - 32) {
        let candidate = read_u32_msb(bits, i);
        let errors = (candidate ^ SYNC_WORD).count_ones();

        if errors <= SYNC_THRESHOLD {
            let payload_start = i + 32;
            if payload_start + frame_bits <= n {
                let data = bits_to_frame(bits, payload_start);
                frames.push(RawFrame {
                    sync_bit_offset: i,
                    data,
                });
            }
        }
    }

    frames
}

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
        let frames = find_frames(&bits);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, frame);
    }

    #[test]
    fn test_find_frames_tolerates_bit_errors_in_syncword() {
        let frame = [0u8; ASM_FRAME_LEN_BYTES];
        let mut bits = make_frame_bits(0, &frame);

        bits[3] = !bits[3];
        bits[10] = !bits[10];

        let frames = find_frames(&bits);
        assert_eq!(frames.len(), 1, "should tolerate 2 bit errors (<=4)");
    }

    #[test]
    fn test_find_frames_empty_on_short_input() {
        let bits = vec![false; 100];
        assert!(find_frames(&bits).is_empty());
    }
}
