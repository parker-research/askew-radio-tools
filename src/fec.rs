//! FEC decoding and CSP-level CRC verification for AX100 "ASM+Golay" mode
//! (`framing: AX100 ASM+Golay`, `scrambler: CCSDS` in gr-satellites YAML —
//! this is FRONTIERSAT's mode).
//!
//! Ported closely from gr-satellites' `u482c_decode` block
//! (`lib/u482c_decode_impl.cc`), as invoked by `ax100_deframer` for
//! `mode='ASM'`: `u482c_decode(verbose, viterbi=0, scrambler=(1 if
//! CCSDS else 0), rs=1)` — i.e. Viterbi is always forced off, RS is
//! always forced on, and the scrambler follows the satellite's YAML.
//!
//! Frame layout (the 258 bytes captured immediately after the syncword by
//! `sync_to_pdu_packed(packlen=258, ...)`):
//! ```text
//! byte 0..3:   Golay(24,12) header — NOT descrambled, NOT RS-corrected
//! byte 3..258: payload region:
//!                1. (Viterbi — always skipped for ax100_deframer)
//!                2. CCSDS-derandomize in place, first `frame_len` bytes
//!                3. RS(255,223) decode in place, pad = 255 - frame_len
//!                4. output = first (frame_len - 32) bytes
//! ```
//! The Golay-corrected header's low 12 bits are `[frame_len: 8][viterbi
//! flag][scrambler flag][RS flag]`; `ax100_deframer`'s ASM path ignores the
//! viterbi/scrambler/RS flag bits and uses its own fixed configuration
//! instead (`lib/u482c_decode_impl.cc`'s `msg_handler`).
//!
//! The Reed-Solomon decoder is a close port of Phil Karn's `decode_rs_8`
//! (`libfec`, vendored into gr-satellites at `lib/libfec/decode_rs_8.c` +
//! `decode_rs.h`), with the fixed CCSDS (255,223) parameters from
//! `lib/libfec/fixed.h`: `NN=255`, `NROOTS=32`, `FCR=112`, `PRIM=11` (note:
//! *not* consecutive roots), GF(2^8) poly `x^8+x^7+x^2+x+1` (`0x187`). Both
//! AX100 modes share this exact codec (`ax100_decode_impl.cc` for RS mode
//! and `u482c_decode_impl.cc` for ASM mode both call `decode_rs_8`
//! directly — no dual-basis conversion needed).
//!
//! The Golay(24,12) decoder is a literal port of `lib/golay24.c`'s
//! syndrome-decoding algorithm (Morelos-Zaragoza, *The Art of Error
//! Correcting Coding*, §2.2.3).
//!
//! The CCSDS derandomizer is a literal port of `lib/randomizer.c`'s
//! `ccsds_generate_sequence`/`ccsds_xor_sequence` (polynomial
//! `x^8+x^7+x^5+x^3+1`, all-ones seed, regenerated fresh per frame).
//!
//! CSP-level CRC verification (`crc_pass`) mirrors gr-satellites'
//! `crcs.crc32c()` helper (`python/crcs.py`), which is `libcsp`'s CRC32C
//! (Castagnoli) over the frame, present only when the CSP header's `crc`
//! flag bit is set.

use crate::DecodeError;

// ---------------------------------------------------------------------------
// Golay(24,12) decoder (port of lib/golay24.c)
// ---------------------------------------------------------------------------

const GOLAY_N: usize = 12;
const GOLAY_H: [u32; GOLAY_N] = [
    0x8008ed, 0x4001db, 0x2003b5, 0x100769, 0x080ed1, 0x040da3, 0x020b47, 0x01068f, 0x008d1d,
    0x004a3b, 0x002477, 0x001ffe,
];

fn golay_b(i: usize) -> u32 {
    GOLAY_H[i] & 0xfff
}

/// Decode a 24-bit Golay(24,12) codeword in place. Returns the number of
/// corrected bit errors (0-3) on success, or `Err` if uncorrectable (≥4
/// errors). The corrected word's low 12 bits are the message; see
/// `lib/golay24.c`'s `encode_golay24` for the encode-side convention
/// (`*data = (parity & 0xfff) << 12 | message`).
fn golay24_decode(data: &mut u32) -> Result<u32, DecodeError> {
    let r = *data;

    // Step 1: s = H*r
    let mut s: u32 = 0;
    for h in GOLAY_H.iter() {
        s <<= 1;
        s |= (h & r).count_ones() & 1;
    }

    // Step 2: if w(s) <= 3, e = (s, 0)
    if s.count_ones() <= 3 {
        let e = s << GOLAY_N;
        *data = r ^ e;
        return Ok(e.count_ones());
    }

    // Step 3: if w(s ^ B(i)) <= 2, e = (s ^ B(i), e_{i+1})
    for i in 0..GOLAY_N {
        let cand = s ^ golay_b(i);
        if cand.count_ones() <= 2 {
            let e = (cand << GOLAY_N) | (1 << (GOLAY_N - i - 1));
            *data = r ^ e;
            return Ok(e.count_ones());
        }
    }

    // Step 4: q = B*s
    let mut q: u32 = 0;
    for i in 0..GOLAY_N {
        q <<= 1;
        q |= (golay_b(i) & s).count_ones() & 1;
    }

    // Step 5: if w(q) <= 3, e = (0, q)
    if q.count_ones() <= 3 {
        let e = q;
        *data = r ^ e;
        return Ok(e.count_ones());
    }

    // Step 6: if w(q ^ B(i)) <= 2, e = (e_{i+1}, q ^ B(i))
    for i in 0..GOLAY_N {
        let cand = q ^ golay_b(i);
        if cand.count_ones() <= 2 {
            let e = (1 << (2 * GOLAY_N - i - 1)) | cand;
            *data = r ^ e;
            return Ok(e.count_ones());
        }
    }

    // Step 7: uncorrectable
    Err(DecodeError::GolayFailed)
}

// ---------------------------------------------------------------------------
// CCSDS randomizer (port of lib/randomizer.c)
// ---------------------------------------------------------------------------

/// Generate `len` bytes of the CCSDS pseudo-random sequence
/// (`h(x) = x^8+x^7+x^5+x^3+1`, all-ones seed), MSB-first per byte. This is
/// a literal (Fibonacci-LFSR) port of `ccsds_generate_sequence`, always
/// starting fresh — `u482c_decode` regenerates/reuses this from index 0
/// for every frame, never continuing state across frames.
fn ccsds_sequence(len: usize) -> Vec<u8> {
    let mut x = [1u8; 9];
    let mut seq = vec![0u8; len];
    for i in 0..len * 8 {
        seq[i / 8] |= x[1] << (7 - (i % 8));
        let fb = x[8] ^ x[6] ^ x[4] ^ x[1];
        for k in 1..8 {
            x[k] = x[k + 1];
        }
        x[8] = fb;
    }
    seq
}

/// XOR `data` in place against a freshly generated CCSDS sequence.
fn ccsds_derandomize(data: &mut [u8]) {
    let seq = ccsds_sequence(data.len());
    for (byte, mask) in data.iter_mut().zip(seq.iter()) {
        *byte ^= mask;
    }
}

// ---------------------------------------------------------------------------
// CCSDS GF(2^8) tables (see lib/libfec/ccsds.c)
// ---------------------------------------------------------------------------

const NN: usize = 255;
const NROOTS: usize = 32;
const FCR: i32 = 112;
const PRIM: i32 = 11;
const IPRIM: i32 = 116; // modular inverse of PRIM mod 255
const GF_POLY: u16 = 0x187;
const A0: u8 = NN as u8; // sentinel: "index of zero"

struct GfTables {
    alpha_to: [u8; 256],
    index_of: [u8; 256],
}

impl GfTables {
    fn new() -> Self {
        let mut alpha_to = [0u8; 256];
        let mut index_of = [0u8; 256];
        index_of[0] = A0;

        let mut x: u16 = 1;
        #[allow(clippy::needless_range_loop)]
        for i in 0..255usize {
            alpha_to[i] = x as u8;
            index_of[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= GF_POLY;
            }
        }
        GfTables { alpha_to, index_of }
    }
}

/// `mod255(x)` from `fixed.h`: fast reduction assuming `x >= 0`.
fn modnn(mut x: i32) -> u8 {
    while x >= 255 {
        x -= 255;
        x = (x >> 8) + (x & 255);
    }
    x as u8
}

// ---------------------------------------------------------------------------
// Reed-Solomon decode (port of decode_rs.h, specialised to no_eras=0)
// ---------------------------------------------------------------------------

/// Decode `data` (the `NN - pad` real symbols, i.e. the tail of a virtual
/// `NN`-symbol codeword whose first `pad` symbols are implicitly zero) in
/// place. Returns the number of corrected symbols, or `-1` if uncorrectable.
fn decode_rs8(gf: &GfTables, data: &mut [u8], pad: usize) -> i32 {
    let n_data = NN - pad;
    debug_assert_eq!(data.len(), n_data);

    let alpha_to = &gf.alpha_to;
    let index_of = &gf.index_of;

    let mut lambda = [0u8; NROOTS + 1];
    let mut s = [0u8; NROOTS];
    let mut b = [0u8; NROOTS + 1];
    let mut t = [0u8; NROOTS + 1];
    let mut omega = [0u8; NROOTS + 1];
    let mut root = [0i32; NROOTS];
    let mut reg = [0u8; NROOTS + 1];
    let mut loc = [0i32; NROOTS];

    // --- Syndromes: evaluate data(x) at the NROOTS code roots ---
    for slot in s.iter_mut() {
        *slot = data[0];
    }
    for &byte in &data[1..n_data] {
        for i in 0..NROOTS {
            if s[i] == 0 {
                s[i] = byte;
            } else {
                s[i] = byte
                    ^ alpha_to
                        [modnn(index_of[s[i] as usize] as i32 + (FCR + i as i32) * PRIM) as usize];
            }
        }
    }

    let mut syn_error = 0u8;
    for i in 0..NROOTS {
        syn_error |= s[i];
        s[i] = index_of[s[i] as usize];
    }
    if syn_error == 0 {
        // Zero syndrome: data is already a valid codeword.
        return 0;
    }

    lambda[0] = 1;
    for i in 0..=NROOTS {
        b[i] = index_of[lambda[i] as usize];
    }

    // --- Berlekamp-Massey ---
    let mut r: i32 = 0;
    let mut el: i32 = 0;
    while r < NROOTS as i32 {
        r += 1;
        let mut discr_r: u8 = 0;
        for i in 0..r as usize {
            if lambda[i] != 0 && s[r as usize - i - 1] != A0 {
                discr_r ^= alpha_to[modnn(
                    index_of[lambda[i] as usize] as i32 + s[r as usize - i - 1] as i32,
                ) as usize];
            }
        }
        let discr_r_idx = index_of[discr_r as usize];
        if discr_r_idx == A0 {
            for i in (1..=NROOTS).rev() {
                b[i] = b[i - 1];
            }
            b[0] = A0;
        } else {
            t[0] = lambda[0];
            for i in 0..NROOTS {
                if b[i] != A0 {
                    t[i + 1] =
                        lambda[i + 1] ^ alpha_to[modnn(discr_r_idx as i32 + b[i] as i32) as usize];
                } else {
                    t[i + 1] = lambda[i + 1];
                }
            }
            if 2 * el < r {
                el = r - el;
                for i in 0..=NROOTS {
                    b[i] = if lambda[i] == 0 {
                        A0
                    } else {
                        modnn(index_of[lambda[i] as usize] as i32 - discr_r_idx as i32 + NN as i32)
                    };
                }
            } else {
                for i in (1..=NROOTS).rev() {
                    b[i] = b[i - 1];
                }
                b[0] = A0;
            }
            lambda.copy_from_slice(&t);
        }
    }

    // Convert lambda to index form, find deg(lambda).
    let mut deg_lambda = 0usize;
    for i in 0..=NROOTS {
        lambda[i] = index_of[lambda[i] as usize];
        if lambda[i] != A0 {
            deg_lambda = i;
        }
    }

    // --- Chien search: find roots of the error locator polynomial ---
    reg[1..=NROOTS].copy_from_slice(&lambda[1..=NROOTS]);
    let mut count = 0usize;
    let mut k: i32 = IPRIM - 1;
    let mut i: i32 = 1;
    while i <= NN as i32 {
        let mut q: u8 = 1;
        for j in (1..=deg_lambda).rev() {
            if reg[j] != A0 {
                reg[j] = modnn(reg[j] as i32 + j as i32);
                q ^= alpha_to[reg[j] as usize];
            }
        }
        if q == 0 {
            root[count] = i;
            loc[count] = k;
            count += 1;
            if count == deg_lambda {
                break;
            }
        }
        i += 1;
        k = modnn(k + IPRIM) as i32;
    }

    if deg_lambda != count {
        // Number of roots doesn't match deg(lambda) => uncorrectable.
        return -1;
    }

    // --- Forney: compute error-evaluator poly and apply corrections ---
    let deg_omega: i32 = deg_lambda as i32 - 1;
    if deg_omega >= 0 {
        for i in 0..=(deg_omega as usize) {
            let mut tmp: u8 = 0;
            for j in (0..=i).rev() {
                if s[i - j] != A0 && lambda[j] != A0 {
                    tmp ^= alpha_to[modnn(s[i - j] as i32 + lambda[j] as i32) as usize];
                }
            }
            omega[i] = index_of[tmp as usize];
        }
    }

    for j in (0..count).rev() {
        let mut num1: u8 = 0;
        if deg_omega >= 0 {
            for i in (0..=(deg_omega as usize)).rev() {
                if omega[i] != A0 {
                    num1 ^= alpha_to[modnn(omega[i] as i32 + i as i32 * root[j]) as usize];
                }
            }
        }
        let num2 = alpha_to[modnn(root[j] * (FCR - 1) + NN as i32) as usize];

        let mut den: u8 = 0;
        let upper = deg_lambda.min(NROOTS - 1) & !1usize;
        let mut ii = upper as i32;
        while ii >= 0 {
            if lambda[(ii + 1) as usize] != A0 {
                den ^= alpha_to[modnn(lambda[(ii + 1) as usize] as i32 + ii * root[j]) as usize];
            }
            ii -= 2;
        }
        if den == 0 {
            // The reference decoder only checks this under DEBUG; treat it
            // as uncorrectable rather than applying a garbage correction.
            return -1;
        }

        if num1 != 0 && loc[j] >= pad as i32 {
            let idx = (loc[j] - pad as i32) as usize;
            data[idx] ^= alpha_to[modnn(
                index_of[num1 as usize] as i32 + index_of[num2 as usize] as i32 + NN as i32
                    - index_of[den as usize] as i32,
            ) as usize];
        }
    }

    count as i32
}

// ---------------------------------------------------------------------------
// Top-level ASM+Golay frame decode (port of u482c_decode_impl::msg_handler)
// ---------------------------------------------------------------------------

const HEADER_LEN: usize = 3;

/// Total bytes captured after the syncword for AX100 ASM+Golay framing
/// (`packlen=258` in gr-satellites' `sync_to_pdu_packed`).
pub const ASM_FRAME_LEN_BYTES: usize = HEADER_LEN + NN;

/// Result of decoding one AX100 ASM+Golay frame.
pub struct AsmGolayDecoded {
    /// CSP frame bytes (RS parity stripped). Best-effort: if
    /// `rs_correctable` is `false`, this is the derandomized-but-otherwise-
    /// uncorrected payload (RS detected more errors than it can fix, so
    /// any partial corrections it made along the way aren't trustworthy).
    pub payload: Vec<u8>,
    /// Number of symbol errors the RS decoder corrected, or `None` if
    /// `rs_correctable` is `false` (there's no meaningful count when RS
    /// couldn't actually correct the codeword).
    pub rs_corrected_error_count: Option<u32>,
    /// Whether RS decoding succeeded (`false` means the codeword had more
    /// than 16 symbol errors — uncorrectable).
    pub rs_correctable: bool,
}

/// Decode one AX100 ASM+Golay frame (the 258 bytes following the syncword),
/// mirroring `u482c_decode_impl::msg_handler` with `ax100_deframer`'s fixed
/// configuration (Viterbi off, RS on, CCSDS scrambler on).
///
/// Returns `Err` only when the frame can't be interpreted at all (the
/// Golay-coded header itself is uncorrectable, or it decodes to a length
/// that isn't a valid RS(255,223) pad) — there's no frame length to work
/// from in that case. An RS-uncorrectable *payload* is still returned as
/// `Ok` (see [`AsmGolayDecoded::rs_correctable`]), so callers can choose
/// whether to keep or drop those.
pub fn ax100_asm_golay_decode(
    frame: &[u8; ASM_FRAME_LEN_BYTES],
) -> Result<AsmGolayDecoded, DecodeError> {
    let mut header = ((frame[0] as u32) << 16) | ((frame[1] as u32) << 8) | frame[2] as u32;
    golay24_decode(&mut header)?;

    let frame_len = (header & 0xff) as i32;
    // header bits 8/9/10 (viterbi/scrambler/RS flags) are intentionally
    // ignored here, matching ax100_deframer's fixed viterbi=off, rs=on,
    // scrambler=CCSDS-per-YAML configuration rather than the frame's
    // self-described flags.

    let pad = NN as i32 - frame_len;
    if !(0..=222).contains(&pad) {
        return Err(DecodeError::ReedSolomonFailed);
    }
    let frame_len = frame_len as usize;
    let pad = pad as usize;
    if frame_len < NROOTS {
        return Err(DecodeError::ReedSolomonFailed);
    }

    let mut packet = frame[HEADER_LEN..HEADER_LEN + frame_len].to_vec();
    ccsds_derandomize(&mut packet);

    let gf = GfTables::new();
    let count = decode_rs8(&gf, &mut packet, pad);
    let rs_correctable = count >= 0;
    let rs_corrected_error_count = rs_correctable.then_some(count as u32);

    let payload_len = frame_len - NROOTS;
    Ok(AsmGolayDecoded {
        payload: packet[..payload_len].to_vec(),
        rs_corrected_error_count,
        rs_correctable,
    })
}

// ---------------------------------------------------------------------------
// CSP-level CRC32C verification
// ---------------------------------------------------------------------------

/// Check the CSP CRC on a decoded CSP frame, following the `crc` flag bit
/// in the CSP header (bit 0 of the big-endian 32-bit header word — see
/// `python/csp_header.py`'s `CSP.crc`). If the flag is clear there's no
/// trailer to check, so this returns `true` (nothing failed).
pub fn csp_crc32c_check(frame: &[u8]) -> bool {
    if frame.len() < 4 {
        return false;
    }
    let header = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]);
    let crc_flag = header & 1 != 0;
    if !crc_flag {
        return true;
    }
    if frame.len() < 8 {
        return false;
    }

    let (body, crc_bytes) = frame.split_at(frame.len() - 4);
    let stored = u32::from_be_bytes(crc_bytes.try_into().unwrap());
    crc32c::crc32c(body) == stored
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn golay_encode(data12: u32) -> u32 {
        let r = data12 & 0xfff;
        let mut s: u32 = 0;
        for h in GOLAY_H.iter() {
            s <<= 1;
            s |= (h & r).count_ones() & 1;
        }
        ((s & 0xfff) << GOLAY_N) | r
    }

    #[test]
    fn test_golay_roundtrip_no_errors() {
        for data in [0u32, 1, 42, 0x123, 0xABC, 0xFFF] {
            let encoded = golay_encode(data);
            let mut w = encoded;
            let errors = golay24_decode(&mut w).expect("should decode cleanly");
            assert_eq!(errors, 0);
            assert_eq!(w & 0xfff, data);
        }
    }

    #[test]
    fn test_golay_corrects_up_to_3_bit_errors() {
        let data = 0x2A5u32;
        let encoded = golay_encode(data);
        for mask in [0b1u32, 0b101, 0b10100001, 1 << 23] {
            let mut corrupted = encoded ^ mask;
            let errors = golay24_decode(&mut corrupted).expect("should correct <=3 bit errors");
            assert_eq!(errors, mask.count_ones());
            assert_eq!(corrupted & 0xfff, data);
        }
    }

    #[test]
    fn test_golay_rejects_too_many_errors() {
        let data = 0x055u32;
        let encoded = golay_encode(data);
        // Flip 7 widely spread bits - well beyond the 3-bit correction radius.
        let mut corrupted = encoded ^ 0b101_0101_0101_0101_0101;
        assert!(golay24_decode(&mut corrupted).is_err());
    }

    #[test]
    fn test_ccsds_sequence_deterministic_and_nonzero() {
        let seq = ccsds_sequence(16);
        assert!(seq.iter().any(|&b| b != 0));
        assert_eq!(seq, ccsds_sequence(16));
    }

    #[test]
    fn test_ccsds_derandomize_involution() {
        let original = vec![0xDEu8, 0xAD, 0xBE, 0xEF, 0x12, 0x34];
        let mut data = original.clone();
        ccsds_derandomize(&mut data);
        ccsds_derandomize(&mut data);
        assert_eq!(data, original);
    }

    fn make_header(frame_len: u8) -> [u8; HEADER_LEN] {
        let word = golay_encode(frame_len as u32);
        [(word >> 16) as u8, (word >> 8) as u8, word as u8]
    }

    #[test]
    fn test_asm_decode_rejects_short_frame_len() {
        let mut frame = [0u8; ASM_FRAME_LEN_BYTES];
        frame[..HEADER_LEN].copy_from_slice(&make_header(10)); // pad=245 > 222
        assert!(matches!(
            ax100_asm_golay_decode(&frame),
            Err(DecodeError::ReedSolomonFailed)
        ));
    }

    #[test]
    fn test_asm_decode_all_zero_codeword() {
        // frame_len=255 (max): pad=0, payload_len=255-32=223.
        // The all-zero RS codeword derandomizes to the CCSDS sequence
        // itself, which is *not* all-zero, so instead we scramble a
        // known-valid (all-zero) codeword forward so that after
        // derandomization inside the decoder we get back to all-zero.
        let mut frame = [0u8; ASM_FRAME_LEN_BYTES];
        frame[..HEADER_LEN].copy_from_slice(&make_header(255));
        let seq = ccsds_sequence(255);
        frame[HEADER_LEN..].copy_from_slice(&seq);

        let decoded = ax100_asm_golay_decode(&frame).expect("should decode cleanly");
        assert_eq!(decoded.payload.len(), 223);
        assert!(decoded.payload.iter().all(|&b| b == 0));
        assert_eq!(decoded.rs_corrected_error_count, Some(0));
        assert!(decoded.rs_correctable);
    }

    #[test]
    fn test_asm_decode_corrects_single_byte_error() {
        let mut frame = [0u8; ASM_FRAME_LEN_BYTES];
        frame[..HEADER_LEN].copy_from_slice(&make_header(255));
        let seq = ccsds_sequence(255);
        frame[HEADER_LEN..].copy_from_slice(&seq);
        // Corrupt one transmitted (scrambled) byte inside the codeword.
        frame[HEADER_LEN + 50] ^= 0x5A;

        let decoded = ax100_asm_golay_decode(&frame).expect("should correct 1 error");
        assert!(decoded.payload.iter().all(|&b| b == 0));
        assert_eq!(decoded.rs_corrected_error_count, Some(1));
        assert!(decoded.rs_correctable);
    }

    #[test]
    fn test_asm_decode_reports_uncorrectable_rs_with_best_effort_payload() {
        let mut frame = [0u8; ASM_FRAME_LEN_BYTES];
        frame[..HEADER_LEN].copy_from_slice(&make_header(255));
        let seq = ccsds_sequence(255);
        frame[HEADER_LEN..].copy_from_slice(&seq);
        // Corrupt far more than the 16 symbols RS(255,223) can fix.
        for i in 0..40 {
            frame[HEADER_LEN + i] ^= 0x5A;
        }

        let decoded = ax100_asm_golay_decode(&frame).expect(
            "Golay header is fine; frame should still decode to Ok with rs_correctable=false",
        );
        assert!(!decoded.rs_correctable);
        assert_eq!(decoded.rs_corrected_error_count, None);
        assert_eq!(decoded.payload.len(), 223);
    }

    #[test]
    fn test_csp_crc_check_no_crc_flag_passes() {
        // header with crc bit (LSB) = 0
        let frame = [0x00u8, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];
        assert!(csp_crc32c_check(&frame));
    }

    #[test]
    fn test_csp_crc_check_valid() {
        let mut frame = vec![0x00u8, 0x00, 0x00, 0x01]; // crc bit set
        frame.extend_from_slice(b"hello world");
        let crc = crc32c::crc32c(&frame);
        frame.extend_from_slice(&crc.to_be_bytes());

        assert!(csp_crc32c_check(&frame));
    }

    #[test]
    fn test_csp_crc_check_invalid() {
        let mut frame = vec![0x00u8, 0x00, 0x00, 0x01];
        frame.extend_from_slice(b"hello world");
        let crc = crc32c::crc32c(&frame);
        frame.extend_from_slice(&crc.to_be_bytes());
        *frame.last_mut().unwrap() ^= 0xFF;

        assert!(!csp_crc32c_check(&frame));
    }
}
