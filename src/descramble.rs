//! Self-synchronizing (multiplicative) scrambler used by the GOMspace AX100
//! modem on its raw bit stream — a port of GNU Radio's
//! `digital.descrambler_bb(mask, seed, len)` as used by gr-satellites'
//! `ax100_deframer` (mode='RS'):
//!
//! ```text
//! sliced bits -> descrambler_bb(0x21, 0, 16) -> syncword search -> ...
//! ```
//!
//! This is *not* the autonomous/additive scrambler (where a keystream is
//! generated independently of the data and XORed in) — with `seed=0` that
//! would be a permanent no-op, since the keystream LFSR would never leave
//! the all-zero state. `descrambler_bb` is self-synchronizing: the shift
//! register is continuously fed by the actual incoming bit stream, so it
//! synchronizes to the remote scrambler within `len` bits regardless of the
//! initial seed. Concretely, for each bit:
//!
//! ```text
//! bit_out = bit_in XOR parity(shift_register & mask)
//! shift_register = ((shift_register << 1) | bit_in) & ((1 << len) - 1)
//! ```
//!
//! Because the register is driven by the same transmitted-bit sequence on
//! both ends, this is applied continuously across the whole capture (not
//! reset per frame), and it is the true inverse of the transmit-side
//! scrambler — not a simple XOR involution.

/// Feedback polynomial mask used by the AX100 RS-mode deframer.
pub const AX100_SCRAMBLER_MASK: u32 = 0x21;
/// Initial LFSR state (self-synchronizing, so this value doesn't matter
/// much beyond the first `AX100_SCRAMBLER_LEN` bits).
pub const AX100_SCRAMBLER_SEED: u32 = 0x0;
/// Shift register length in bits.
pub const AX100_SCRAMBLER_LEN: u32 = 16;

struct Lfsr {
    reg: u32,
    mask: u32,
    reg_mask: u32,
}

impl Lfsr {
    fn new(mask: u32, seed: u32, len: u32) -> Self {
        Lfsr {
            reg: seed,
            mask,
            reg_mask: (1u32 << len) - 1,
        }
    }

    fn parity(&self) -> bool {
        (self.reg & self.mask).count_ones() & 1 != 0
    }

    fn shift_in(&mut self, bit: bool) {
        self.reg = ((self.reg << 1) | (bit as u32)) & self.reg_mask;
    }
}

/// Receive-side descrambler: recovers the original bits from a scrambled
/// stream. The shift register is fed by the (scrambled) input bits.
pub fn descramble(bits: &[bool]) -> Vec<bool> {
    let mut lfsr = Lfsr::new(
        AX100_SCRAMBLER_MASK,
        AX100_SCRAMBLER_SEED,
        AX100_SCRAMBLER_LEN,
    );
    bits.iter()
        .map(|&b| {
            let out = b ^ lfsr.parity();
            lfsr.shift_in(b);
            out
        })
        .collect()
}

/// Transmit-side scrambler: the counterpart to [`descramble`], used here
/// only to build scrambled test fixtures. The shift register is fed by the
/// (scrambled) output bits, so `descramble(scramble(x)) == x`.
#[cfg(test)]
pub fn scramble(bits: &[bool]) -> Vec<bool> {
    let mut lfsr = Lfsr::new(
        AX100_SCRAMBLER_MASK,
        AX100_SCRAMBLER_SEED,
        AX100_SCRAMBLER_LEN,
    );
    bits.iter()
        .map(|&b| {
            let out = b ^ lfsr.parity();
            lfsr.shift_in(out);
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descramble_inverts_scramble() {
        let bits: Vec<bool> = (0..500).map(|i| i % 7 < 3).collect();
        let scrambled = scramble(&bits);
        let recovered = descramble(&scrambled);
        assert_eq!(recovered, bits);
    }

    #[test]
    fn test_descramble_changes_nonzero_data() {
        let bits = vec![true; 64];
        let out = descramble(&bits);
        assert_ne!(out, bits, "scrambler should not be a no-op on real data");
    }
}
