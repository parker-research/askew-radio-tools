//! FEC and integrity: CCSDS de-randomization, Reed-Solomon, CRC-32C.
//!
//! Applied in order after frame extraction:
//!  1. CCSDS de-randomize (XOR with PN sequence)
//!  2. Reed-Solomon RS(255,223) decode — strips 32 parity bytes
//!  3. CRC-32C verify + strip — strips 4 CRC bytes
//!
//! The output is the raw CSP packet payload bytes.

use crate::DecodeError;

// ---------------------------------------------------------------------------
// CCSDS Pseudo-Random De-randomizer
// ---------------------------------------------------------------------------
//
// CCSDS pseudo-random sequence generator:
//   polynomial: h(x) = x⁸ + x⁷ + x⁵ + x³ + 1  (0xA9 in Galois form)
//   seed: 0xFF at start of each frame
//
// The same XOR sequence is used to randomize and de-randomize (it's its own
// inverse), so this function both randomizes and de-randomizes.

/// CCSDS LFSR polynomial taps (feedback polynomial h(x) = x⁸+x⁷+x⁵+x³+1).
const CCSDS_POLY: u8 = 0xA9; // bits 7,5,3,0 set

/// Generate `len` bytes of the CCSDS pseudo-random sequence starting from
/// the standard initial state (0xFF). Pre-compute on first call; callers
/// can also call directly per-frame.
pub fn ccsds_sequence(len: usize) -> Vec<u8> {
    let mut seq = Vec::with_capacity(len);
    let mut lfsr: u8 = 0xFF; // initial state per CCSDS standard

    for _ in 0..len {
        let mut byte = 0u8;
        for bit in 0..8 {
            // Output bit is the MSB of the LFSR
            let out_bit = (lfsr >> 7) & 1;
            byte |= out_bit << (7 - bit);

            // Feedback: XOR taps and shift
            let feedback = if out_bit == 1 { CCSDS_POLY } else { 0 };
            lfsr = lfsr.wrapping_shl(1) ^ feedback;
        }
        seq.push(byte);
    }
    seq
}

/// Apply (or reverse) CCSDS pseudo-randomization in-place.
pub fn ccsds_derandomize(data: &mut Vec<u8>) {
    let seq = ccsds_sequence(data.len());
    for (byte, mask) in data.iter_mut().zip(seq.iter()) {
        *byte ^= mask;
    }
}

// ---------------------------------------------------------------------------
// Reed-Solomon RS(255, 223) decoder
// ---------------------------------------------------------------------------
//
// CCSDS uses GF(2⁸) with primitive polynomial p(x) = x⁸+x⁷+x²+x+1 (0x187),
// primitive root α=2, first consecutive root b=112, block length 255,
// data symbols 223, parity symbols 32.
//
// We implement a pure-Rust RS decoder following the Berlekamp-Massey /
// Forney algorithm over GF(2⁸). This is a complete self-contained
// implementation targeting the exact CCSDS parameters.

/// CCSDS GF(2⁸) primitive polynomial: x⁸+x⁷+x²+x+1 = 0x187
const GF_POLY: u16 = 0x187;
const GF_SIZE: usize = 256;
const RS_N: usize = 255; // codeword length
const RS_K: usize = 223; // data symbols
const RS_T: usize = 16; // symbol errors correctable (2T = 32 parity bytes)
const RS_B: usize = 112; // first consecutive root (CCSDS)

/// Galois Field GF(2⁸) arithmetic tables.
struct Gf256 {
    exp: [u8; 512], // α^i for i in 0..511 (duplicated for wrap-around)
    log: [u8; 256], // log_α(x) for x in 1..255; log[0] is undefined
}

impl Gf256 {
    fn new() -> Self {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u16 = 1;
        for i in 0..255usize {
            exp[i] = x as u8;
            exp[i + 255] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= GF_POLY;
            }
        }
        Gf256 { exp, log }
    }

    #[inline]
    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        self.exp[(self.log[a as usize] as usize) + (self.log[b as usize] as usize)]
    }

    #[inline]
    fn div(&self, a: u8, b: u8) -> u8 {
        debug_assert!(b != 0, "GF division by zero");
        if a == 0 {
            return 0;
        }
        let la = self.log[a as usize] as usize;
        let lb = self.log[b as usize] as usize;
        self.exp[(la + 255 - lb) % 255]
    }

    #[inline]
    fn pow(&self, base: u8, exp: usize) -> u8 {
        if base == 0 {
            return 0;
        }
        self.exp[(self.log[base as usize] as usize * exp) % 255]
    }

    #[inline]
    fn alpha_pow(&self, exp: usize) -> u8 {
        self.exp[exp % 255]
    }
}

/// Decode a Reed-Solomon RS(255,223) codeword (255 bytes) and return the
/// 223 corrected data bytes. Strips the 32 parity bytes.
///
/// # Errors
/// Returns [`DecodeError::ReedSolomonFailed`] if there are more than 16
/// symbol errors (uncorrectable).
pub fn rs_decode(codeword: &[u8]) -> Result<Vec<u8>, DecodeError> {
    if codeword.len() != RS_N {
        return Err(DecodeError::ReedSolomonFailed);
    }

    let gf = Gf256::new();

    // 1. Compute syndromes S_i = r(α^(b+i)) for i = 0..2T-1
    let mut syndromes = vec![0u8; 2 * RS_T];
    for (i, s) in syndromes.iter_mut().enumerate() {
        let root = gf.alpha_pow(RS_B + i);
        let mut acc = 0u8;
        for &byte in codeword {
            acc = gf.mul(acc, root) ^ byte;
        }
        *s = acc;
    }

    if syndromes.iter().all(|&s| s == 0) {
        // No errors — return data portion directly
        return Ok(codeword[..RS_K].to_vec());
    }

    // 2. Berlekamp-Massey to find error locator polynomial Λ(x)
    let lambda = berlekamp_massey(&gf, &syndromes);

    if lambda.len() - 1 > RS_T {
        return Err(DecodeError::ReedSolomonFailed);
    }

    // 3. Chien search to find error locations
    let mut error_positions: Vec<usize> = Vec::new();
    for j in 0..RS_N {
        // Evaluate Λ(α^(-j)) — error locator root
        let alpha_inv = gf.alpha_pow(255 - j % 255);
        let mut val = 0u8;
        let mut alpha_pow = 1u8;
        for &coeff in &lambda {
            val ^= gf.mul(coeff, alpha_pow);
            alpha_pow = gf.mul(alpha_pow, alpha_inv);
        }
        if val == 0 {
            error_positions.push(j);
        }
    }

    if error_positions.len() != lambda.len() - 1 {
        return Err(DecodeError::ReedSolomonFailed);
    }

    // 4. Forney algorithm to compute error magnitudes
    let mut corrected = codeword.to_vec();
    let omega = error_evaluator(&gf, &syndromes, &lambda);

    for &pos in &error_positions {
        let x_k = gf.alpha_pow(pos);
        let x_k_inv = gf.alpha_pow(255 - pos % 255);

        // Evaluate Ω(X_k^-1)
        let mut omega_val = 0u8;
        let mut x_pow = 1u8;
        for &coeff in &omega {
            omega_val ^= gf.mul(coeff, x_pow);
            x_pow = gf.mul(x_pow, x_k_inv);
        }

        // Evaluate Λ'(X_k^-1) — formal derivative (even powers vanish in GF(2))
        let mut lambda_prime = 0u8;
        let mut x_p = 1u8;
        for (i, &coeff) in lambda.iter().enumerate() {
            if i % 2 == 1 {
                lambda_prime ^= gf.mul(coeff, x_p);
            }
            x_p = gf.mul(x_p, x_k_inv);
        }

        if lambda_prime == 0 {
            return Err(DecodeError::ReedSolomonFailed);
        }

        let magnitude = gf.mul(gf.div(omega_val, lambda_prime), x_k);
        // Position in codeword: α^pos corresponds to codeword[N-1-pos]
        let codeword_pos = (RS_N - 1 - pos % RS_N) % RS_N;
        corrected[codeword_pos] ^= magnitude;
    }

    Ok(corrected[..RS_K].to_vec())
}

/// Berlekamp-Massey algorithm: finds the minimal-length LFSR (error locator
/// polynomial) that generates the given syndrome sequence.
fn berlekamp_massey(gf: &Gf256, syndromes: &[u8]) -> Vec<u8> {
    let n = syndromes.len();
    let mut lambda = vec![0u8; n + 1];
    let mut b = vec![0u8; n + 1];
    lambda[0] = 1;
    b[0] = 1;
    let mut l = 0usize;
    let mut m = 1usize;
    let mut b_val = 1u8;

    for i in 0..n {
        // Compute discrepancy delta
        let mut delta = syndromes[i];
        for j in 1..=l {
            delta ^= gf.mul(lambda[j], syndromes[i - j]);
        }

        if delta == 0 {
            m += 1;
            continue;
        }

        let t = lambda.clone();
        let coeff = gf.div(delta, b_val);
        for j in m..=n {
            lambda[j] ^= gf.mul(coeff, b[j - m]);
        }

        if 2 * l <= i {
            l = i + 1 - l;
            b = t;
            b_val = delta;
            m = 1;
        } else {
            m += 1;
        }
    }

    lambda.truncate(l + 1);
    lambda
}

/// Error evaluator polynomial Ω(x) = S(x)·Λ(x) mod x^(2T).
fn error_evaluator(gf: &Gf256, syndromes: &[u8], lambda: &[u8]) -> Vec<u8> {
    let two_t = 2 * RS_T;
    let mut omega = vec![0u8; two_t];
    for i in 0..two_t {
        for j in 0..lambda.len().min(i + 1) {
            if i - j < syndromes.len() {
                omega[i] ^= gf.mul(lambda[j], syndromes[i - j]);
            }
        }
    }
    // Trim trailing zeros
    while omega.len() > 1 && *omega.last().unwrap() == 0 {
        omega.pop();
    }
    omega
}

// ---------------------------------------------------------------------------
// CRC-32C verification
// ---------------------------------------------------------------------------

/// Verify the CRC-32C appended to a Reed-Solomon decoded payload and return
/// the payload without the 4-byte CRC trailer.
///
/// The last 4 bytes of `payload` must be the CRC-32C checksum (little-endian)
/// over the preceding bytes.
pub fn crc32c_verify_and_strip(payload: &[u8]) -> Result<Vec<u8>, DecodeError> {
    if payload.len() < 4 {
        return Err(DecodeError::CrcMismatch);
    }

    let (data, crc_bytes) = payload.split_at(payload.len() - 4);
    let stored_crc = u32::from_le_bytes(crc_bytes.try_into().unwrap());
    let computed_crc = crc32c::crc32c(data);

    if computed_crc != stored_crc {
        return Err(DecodeError::CrcMismatch);
    }

    Ok(data.to_vec())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccsds_sequence_first_bytes() {
        // Verify the CCSDS PN sequence is non-trivial and deterministic.
        let seq = ccsds_sequence(16);
        assert!(
            seq.iter().any(|&b| b != 0),
            "PN sequence should be non-zero"
        );
        let seq2 = ccsds_sequence(16);
        assert_eq!(seq, seq2, "PN sequence must be deterministic");
    }

    #[test]
    fn test_ccsds_derandomize_involution() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78];
        let mut data = original.clone();
        ccsds_derandomize(&mut data);
        ccsds_derandomize(&mut data); // apply twice — should restore original
        assert_eq!(
            data, original,
            "CCSDS derandomize should be its own inverse"
        );
    }

    #[test]
    fn test_rs_no_errors() {
        // Build a valid RS(255,223) codeword with all-zero data and
        // verify it round-trips cleanly. We simulate by encoding
        // a known sequence and checking the decoded output.
        // (Full encoder omitted here — we test the zero-error path.)
        // A valid codeword of 255 zero bytes has zero syndrome → passes.
        let codeword = vec![0u8; RS_N];
        let result = rs_decode(&codeword).expect("All-zero codeword should decode cleanly");
        assert_eq!(result.len(), RS_K);
        assert!(result.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_crc32c_verify_and_strip_valid() {
        let data = b"CSP beacon payload data here!";
        let crc = crc32c::crc32c(data);
        let mut payload = data.to_vec();
        payload.extend_from_slice(&crc.to_le_bytes());

        let stripped = crc32c_verify_and_strip(&payload).expect("CRC should match");
        assert_eq!(stripped, data);
    }

    #[test]
    fn test_crc32c_verify_and_strip_invalid() {
        let data = b"CSP beacon payload data here!";
        let crc = crc32c::crc32c(data);
        let mut payload = data.to_vec();
        payload.extend_from_slice(&crc.to_le_bytes());
        // Corrupt the last byte
        *payload.last_mut().unwrap() ^= 0xFF;

        let result = crc32c_verify_and_strip(&payload);
        assert!(matches!(result, Err(DecodeError::CrcMismatch)));
    }
}
