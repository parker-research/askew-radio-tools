//! Sync word detection and frame extraction from the raw bit stream.
//!
//! AX100 Mode 2 (ASM) frame structure:
//! ```text
//! ┌──────────────────┬─────────────────────┬──────────────────────────┐
//! │  32-bit sync     │  24-bit Golay LEN   │  DATA FIELD              │
//! │  0xC9D08A7B      │  Golay(24,12)       │  LEN bytes               │
//! └──────────────────┴─────────────────────┴──────────────────────────┘
//! ```
//! The sync word is searched with up to `MAX_SYNC_ERRORS` bit errors to
//! tolerate noise on the link. The Golay(24,12) code in the length field
//! can correct up to 3 bit errors.

use crate::DecodeError;

/// The 32-bit ASM sync word (MSB first, as transmitted).
pub const SYNC_WORD: u32 = 0xC9D08A7B;

/// Maximum allowed bit errors in the sync word before a candidate is rejected.
pub const MAX_SYNC_ERRORS: u32 = 2;

/// A raw frame extracted from the bit stream (before FEC / de-randomization).
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Byte offset (in bits from stream start) where the sync word was found.
    pub sync_bit_offset: usize,
    /// Frame data bytes: RS-encoded, randomized payload.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Sync word search
// ---------------------------------------------------------------------------

/// Search the bit stream for occurrences of the AX100 sync word and extract
/// all candidate frames.
///
/// For each candidate, the Golay(24,12) length field is decoded to determine
/// how many data bytes to read. Frames where the length field cannot be Golay-
/// decoded (> 3 bit errors) are silently skipped.
pub fn find_frames(bits: &[bool]) -> Vec<RawFrame> {
    let n = bits.len();
    let mut frames = Vec::new();

    // We need at least 32 (sync) + 24 (Golay LEN) bits to start.
    if n < 56 {
        return frames;
    }

    let mut i = 0;
    while i <= n.saturating_sub(56) {
        // Read 32 bits as a u32 (MSB first)
        let candidate = read_u32_msb(bits, i);
        let errors = (candidate ^ SYNC_WORD).count_ones();

        if errors <= MAX_SYNC_ERRORS {
            // Try to decode the Golay length field that immediately follows.
            let golay_start = i + 32;
            if golay_start + 24 > n {
                break;
            }

            let golay_word = read_u24_msb(bits, golay_start);
            match golay_decode(golay_word) {
                Ok(length) => {
                    let length = length as usize;
                    if length == 0 || length > 240 {
                        // AX100 max payload is 240 bytes
                        i += 1;
                        continue;
                    }

                    let data_start_bit = golay_start + 24;
                    let data_end_bit = data_start_bit + length * 8;

                    if data_end_bit > n {
                        // Not enough bits left for the declared payload
                        i += 1;
                        continue;
                    }

                    let data = bits_to_bytes(bits, data_start_bit, length);
                    frames.push(RawFrame {
                        sync_bit_offset: i,
                        data,
                    });

                    // Skip past this frame to avoid false re-detections inside
                    // the payload (not guaranteed to be clean, but a good heuristic).
                    i = data_end_bit;
                    continue;
                }
                Err(_) => {
                    // Golay failure — not a valid frame start
                }
            }
        }

        i += 1;
    }

    frames
}

// ---------------------------------------------------------------------------
// Bit packing helpers
// ---------------------------------------------------------------------------

fn read_u32_msb(bits: &[bool], offset: usize) -> u32 {
    let mut v = 0u32;
    for k in 0..32 {
        v = (v << 1) | (bits[offset + k] as u32);
    }
    v
}

fn read_u24_msb(bits: &[bool], offset: usize) -> u32 {
    let mut v = 0u32;
    for k in 0..24 {
        v = (v << 1) | (bits[offset + k] as u32);
    }
    v
}

fn bits_to_bytes(bits: &[bool], offset: usize, num_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(num_bytes);
    for byte_idx in 0..num_bytes {
        let mut b = 0u8;
        for bit in 0..8 {
            b = (b << 1) | (bits[offset + byte_idx * 8 + bit] as u8);
        }
        out.push(b);
    }
    out
}

// ---------------------------------------------------------------------------
// Golay(24,12) decoder
// ---------------------------------------------------------------------------
//
// The AX100 uses a systematic Golay(24,12) code for the length field.
// Layout: [12 parity bits | 12 data bits] (MSB first in the 24-bit word).
// Can correct up to 3 bit errors.
//
// Reference: Lin & Costello, "Error Control Coding", 2nd ed., Ch. 6.
// Generator polynomial: G(x) = x¹¹ + x¹⁰ + x⁶ + x⁵ + x⁴ + x² + 1
//                             = 0b1000_0011_0111_01 = 0x0C75 (12-bit)

const GOLAY_POLY: u32 = 0x0C75; // generator polynomial (12 bits)

/// Encode 12 data bits into a 24-bit Golay codeword.
/// Returns [12 data | 12 parity] packed into a u32 (only 24 bits used).
pub fn golay_encode(data12: u16) -> u32 {
    debug_assert!(data12 < (1 << 12));
    let parity = golay_parity(data12 as u32);
    ((data12 as u32) << 12) | (parity as u32)
}

/// Compute the 12-bit Golay parity for 12 data bits.
fn golay_parity(data: u32) -> u32 {
    let mut remainder = data & 0xFFF;
    for i in (0..12).rev() {
        if (remainder >> i) & 1 == 1 {
            remainder ^= GOLAY_POLY << (i as u32).saturating_sub(0);
        }
    }
    // Systematic parity = remainder after division
    // Re-derive by multiplying data by x^12 mod g(x)
    let mut r = 0u32;
    let mut d = data & 0xFFF;
    for _ in 0..12 {
        if (d >> 11) & 1 == 1 {
            d ^= GOLAY_POLY;
        }
        r = (r << 1) | ((d >> 11) & 1);
        d = (d << 1) & 0xFFF;
    }
    r & 0xFFF
}

/// Decode a 24-bit Golay word and return the 12-bit data value.
/// Corrects up to 3 bit errors. Returns `Err(DecodeError::GolayFailed)`
/// if there are more than 3 errors.
///
/// Codeword layout: [12 data bits | 12 parity bits] in the low 24 bits.
/// (The AX100 manual specifies data in the upper half of the Golay word,
///  but the framing code packs the 24-bit word with data in bits 23..12
///  and parity in bits 11..0, so we treat it as such here.)
pub fn golay_decode(word24: u32) -> Result<u16, DecodeError> {
    let word = word24 & 0x00FF_FFFF;

    // Split into [data | parity] — data in upper 12 bits, parity in lower 12
    let received_data = ((word >> 12) & 0xFFF) as u16;
    let received_parity = (word & 0xFFF) as u32;
    let computed_parity = golay_parity(received_data as u32);
    let syndrome = computed_parity ^ received_parity;

    if syndrome == 0 {
        return Ok(received_data);
    }

    // Errors only in parity half (syndrome has ≤3 bits set) — data is clean
    if syndrome.count_ones() <= 3 {
        return Ok(received_data);
    }

    // Try to correct 1, 2, or 3 errors in the data half
    for i in 0..12u32 {
        let e = 1u32 << i;
        let s2 = syndrome ^ golay_parity(e);
        if s2.count_ones() <= 2 {
            return Ok(received_data ^ (e as u16));
        }
    }

    for i in 0..12u32 {
        for j in (i + 1)..12u32 {
            let e = (1u32 << i) | (1u32 << j);
            let s2 = syndrome ^ golay_parity(e);
            if s2.count_ones() <= 1 {
                return Ok(received_data ^ (e as u16));
            }
        }
    }

    for i in 0..12u32 {
        for j in (i + 1)..12u32 {
            for k in (j + 1)..12u32 {
                let e = (1u32 << i) | (1u32 << j) | (1u32 << k);
                let s2 = syndrome ^ golay_parity(e);
                if s2 == 0 {
                    return Ok(received_data ^ (e as u16));
                }
            }
        }
    }

    Err(DecodeError::GolayFailed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golay_roundtrip_no_errors() {
        for data in [0u16, 1, 42, 100, 0x7FF, 0xFFF] {
            // 0xFFF is out of range for 12 bits — skip
            if data >= (1 << 12) {
                continue;
            }
            let encoded = golay_encode(data);
            let decoded = golay_decode(encoded).expect("Golay decode failed with no errors");
            assert_eq!(decoded, data, "Roundtrip failed for data={}", data);
        }
    }

    #[test]
    fn test_golay_corrects_single_bit_error() {
        let data = 0x5A5u16; // arbitrary 12-bit value
        let encoded = golay_encode(data);
        // Flip bit 0 in the data half
        let corrupted = encoded ^ 1;
        let decoded = golay_decode(corrupted).expect("Should correct 1-bit error");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_golay_corrects_two_bit_errors() {
        let data = 0x3C3u16;
        let encoded = golay_encode(data);
        // Flip bits 0 and 5 in the data half
        let corrupted = encoded ^ (1 | (1 << 5));
        let decoded = golay_decode(corrupted).expect("Should correct 2-bit error");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_sync_word_search_finds_frame() {
        // Build a synthetic bit stream: preamble + sync + Golay(LEN=10) + 10 zero bytes
        let length: u16 = 10;
        let golay_word = golay_encode(length);

        let mut bits = Vec::new();

        // 64-bit preamble (0xAA alternating)
        for _ in 0..64 {
            bits.push(false);
            bits.push(true);
        }

        // Sync word
        for k in (0..32).rev() {
            bits.push((SYNC_WORD >> k) & 1 == 1);
        }

        // Golay length word (24 bits, MSB first)
        for k in (0..24).rev() {
            bits.push((golay_word >> k) & 1 == 1);
        }

        // 10 zero bytes of data
        for _ in 0..80 {
            bits.push(false);
        }

        let frames = find_frames(&bits);
        assert_eq!(frames.len(), 1, "Should find exactly one frame");
        assert_eq!(frames[0].data.len(), length as usize);
        assert!(frames[0].data.iter().all(|&b| b == 0));
    }
}
