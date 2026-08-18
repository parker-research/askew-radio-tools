//! Reed-Solomon decoding and CSP-level CRC verification for AX100 Mode 5.
//!
//! The Reed-Solomon decoder is a close port of Phil Karn's `decode_rs_8`
//! (`libfec`, as vendored into gr-satellites at `lib/libfec/decode_rs_8.c`
//! + `decode_rs.h`), with the fixed CCSDS (255,223) parameters from
//! `lib/libfec/fixed.h`:
//!
//!   - `NN=255`, `NROOTS=32` (32 parity bytes, corrects up to 16 errors)
//!   - `FCR=112`, `PRIM=11` (note: *not* consecutive roots — this is the
//!     detail that's easy to get wrong when porting from a "normal" RS
//!     decoder)
//!   - GF(2^8) generator polynomial `x^8+x^7+x^2+x+1` (`0x187`)
//!
//! `ax100_decode_impl.cc`'s `msg_handler` calls `decode_rs_8` directly (not
//! `decode_rs_ccsds`), i.e. **no dual-basis conversion** — the AX100 modem
//! already produces/expects the conventional basis, so we don't need the
//! `Taltab`/`Tal1tab` transform either.
//!
//! CSP-level CRC verification (`crc_pass`) mirrors gr-satellites'
//! `crcs.crc32c()` helper (`python/crcs.py`), which is `libcsp`'s CRC32C
//! (Castagnoli) over the frame, present only when the CSP header's `crc`
//! flag bit is set.

use crate::DecodeError;

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
                    ^ alpha_to[modnn(index_of[s[i] as usize] as i32 + (FCR + i as i32) * PRIM)
                        as usize];
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
            if 2 * el <= r - 1 {
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

/// Decode one AX100 Mode 5 frame (the 256 bytes following the syncword:
/// `[LEN, ...codeword tail]`), mirroring `ax100_decode_impl::msg_handler`.
///
/// Returns the recovered CSP frame bytes (RS parity stripped) and the
/// number of symbol errors the RS decoder corrected.
pub fn ax100_rs_decode(frame: &[u8; 256]) -> Result<(Vec<u8>, u32), DecodeError> {
    let len_byte = frame[0] as i32;
    let pad = 255 - len_byte + 1;
    if !(0..=222).contains(&pad) {
        return Err(DecodeError::ReedSolomonFailed);
    }
    let pad = pad as usize;
    let real_len = NN - pad; // == len_byte - 1
    if real_len < NROOTS {
        return Err(DecodeError::ReedSolomonFailed);
    }

    let gf = GfTables::new();
    let mut data = frame[1..1 + real_len].to_vec();
    let count = decode_rs8(&gf, &mut data, pad);
    if count < 0 {
        return Err(DecodeError::ReedSolomonFailed);
    }

    let payload_len = real_len - NROOTS;
    Ok((data[..payload_len].to_vec(), count as u32))
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

    #[test]
    fn test_rs_decode_rejects_short_len_byte() {
        let mut frame = [0u8; 256];
        frame[0] = 10; // pad = 256-10 = 246 > 222 -> invalid
        assert!(matches!(
            ax100_rs_decode(&frame),
            Err(DecodeError::ReedSolomonFailed)
        ));
    }

    #[test]
    fn test_rs_decode_all_zero_codeword() {
        // All-zero data is trivially a valid codeword (zero syndrome).
        let mut frame = [0u8; 256];
        frame[0] = 255; // pad=1, real_len=254, payload_len=254-32=222
        let (payload, corrected) = ax100_rs_decode(&frame).expect("should decode cleanly");
        assert_eq!(payload.len(), 222);
        assert!(payload.iter().all(|&b| b == 0));
        assert_eq!(corrected, 0);
    }

    #[test]
    fn test_rs_decode_corrects_single_byte_error() {
        // Build a valid all-zero codeword, corrupt one parity byte, and
        // confirm the decoder both detects and repairs it.
        let mut frame = [0u8; 256];
        frame[0] = 255;
        frame[50] ^= 0x5A; // corrupt one byte inside the codeword region

        let (payload, corrected) = ax100_rs_decode(&frame).expect("should correct 1 error");
        assert!(payload.iter().all(|&b| b == 0));
        assert_eq!(corrected, 1);
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
