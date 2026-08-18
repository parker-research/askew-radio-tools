//! Syncword search and fixed-length frame extraction for AX100 Mode 5
//! (Reed-Solomon / CSP) framing.
//!
//! Ported closely from gr-satellites' `ax100_deframer` (`mode='RS'`) and its
//! `sync_to_pdu_packed` hierarchical block:
//!
//! ```text
//! sliced bits -> descrambler_bb(0x21,0,16) -> correlate_access_code_tag_bb(
//!     sync='10010011000010110101000111011110', threshold=4)
//!   -> fixedlen_to_pdu(packlen=256*8)
//! ```
//!
//! i.e. the whole bit stream is descrambled first ([`crate::descramble`]),
//! then searched for the 32-bit syncword (tolerating up to
//! [`SYNC_THRESHOLD`] bit errors), and the [`FRAME_LEN_BYTES`] bytes
//! immediately following each match are packed up as a candidate frame.
//!
//! Frame layout (the `d_data` buffer handled by `ax100_decode_impl`):
//! ```text
//! byte 0:      LEN   — drives the RS "pad" (shortened-code) length
//! byte 1..256: up to 255 bytes of [CSP frame | 32 RS parity bytes],
//!              zero-padded at the end if the transmitted frame was
//!              shorter than 255 bytes
//! ```

use crate::descramble;

/// 32-bit AX100 "RS mode" syncword, matching gr-satellites'
/// `_syncword = '10010011000010110101000111011110'`.
pub const SYNC_WORD: u32 = 0x930B_51DE;

/// Bit errors tolerated in the syncword (gr-satellites' default
/// `syncword_threshold`).
pub const SYNC_THRESHOLD: u32 = 4;

/// Fixed PDU length pulled after each syncword match (`packlen=256` bytes
/// for RS mode).
pub const FRAME_LEN_BYTES: usize = 256;

/// A raw frame extracted from the (descrambled) bit stream, before RS
/// decoding.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Bit index (into the descrambled stream, same indexing as the
    /// original recovered bitstream) where the syncword started.
    pub sync_bit_offset: usize,
    /// The `FRAME_LEN_BYTES` raw bytes following the syncword.
    pub data: [u8; FRAME_LEN_BYTES],
}

/// Descramble `bits` and search for all AX100 Mode 5 candidate frames.
pub fn find_frames(bits: &[bool]) -> Vec<RawFrame> {
    let descrambled = descramble::descramble(bits);
    let n = descrambled.len();
    let frame_bits = FRAME_LEN_BYTES * 8;
    let mut frames = Vec::new();

    if n < 32 + frame_bits {
        return frames;
    }

    let mut i = 0;
    while i <= n - 32 {
        let candidate = read_u32_msb(&descrambled, i);
        let errors = (candidate ^ SYNC_WORD).count_ones();

        if errors <= SYNC_THRESHOLD {
            let payload_start = i + 32;
            if payload_start + frame_bits <= n {
                let data = bits_to_frame(&descrambled, payload_start);
                frames.push(RawFrame {
                    sync_bit_offset: i,
                    data,
                });
                // Skip past this frame so we don't re-detect a syncword
                // that happens to occur inside its payload/parity bytes.
                i = payload_start + frame_bits;
                continue;
            }
        }

        i += 1;
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

fn bits_to_frame(bits: &[bool], offset: usize) -> [u8; FRAME_LEN_BYTES] {
    let mut out = [0u8; FRAME_LEN_BYTES];
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

    /// Build a *transmitted* (i.e. already-scrambled) bit stream containing
    /// one AX100 Mode 5 frame, the way `find_frames` expects to receive it
    /// straight off the slicer.
    fn make_scrambled_frame_bits(preamble_bits: usize, frame: &[u8; 256]) -> Vec<bool> {
        let mut plain = Vec::new();
        for _ in 0..preamble_bits {
            plain.push(false);
            plain.push(true);
        }
        for k in (0..32).rev() {
            plain.push((SYNC_WORD >> k) & 1 == 1);
        }
        for &byte in frame {
            for bit in (0..8).rev() {
                plain.push((byte >> bit) & 1 == 1);
            }
        }
        // find_frames descrambles on the way in, so build the "transmitted"
        // (scrambled) bit stream using the transmit-side scrambler.
        descramble::scramble(&plain)
    }

    #[test]
    fn test_find_frames_recovers_one_frame() {
        let mut frame = [0u8; 256];
        frame[0] = 0xAB;
        for (i, b) in frame.iter_mut().enumerate().skip(1) {
            *b = i as u8;
        }

        let bits = make_scrambled_frame_bits(64, &frame);
        let frames = find_frames(&bits);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data, frame);
    }

    #[test]
    fn test_find_frames_tolerates_bit_errors_in_syncword() {
        let frame = [0u8; 256];
        let mut bits = make_scrambled_frame_bits(0, &frame);

        // Flip one bit in the *transmitted* (scrambled) stream. Because
        // AX100_SCRAMBLER_MASK (0x21) has 2 taps, this self-synchronizing
        // descrambler multiplies a single channel bit error into up to
        // 3 bit errors in the descrambled syncword region — still within
        // SYNC_THRESHOLD (4).
        bits[5] = !bits[5];

        let frames = find_frames(&bits);
        assert_eq!(
            frames.len(),
            1,
            "should tolerate the descrambler's error multiplication of a single bit flip"
        );
    }

    #[test]
    fn test_find_frames_empty_on_short_input() {
        let bits = vec![false; 100];
        assert!(find_frames(&bits).is_empty());
    }
}
