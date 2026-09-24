//! AX100 "ASM+Golay" packet decoder for SatNOGS audio captures
//! (FRONTIERSAT / NORAD 69015 config: `framing: AX100 ASM+Golay`,
//! `scrambler: CCSDS`, 9600 baud, 3200 Hz deviation).
//!
//! Implements the full decode chain, closely ported from gr-satellites'
//! `ax100_deframer(mode='ASM')` + `u482c_decode`:
//!   Audio (WAV/OGG) → LPF → symbol timing → bit decisions →
//!   syncword search → Golay(24,12) length header → CCSDS derandomize →
//!   Reed-Solomon (255,223) → CSP frame → CSP CRC-32C check
//!
//! On top of that reference chain, to recover more of what's actually in
//! a noisy capture (see [`pipeline::decode_audio`]): an ensemble of timing
//! recovery methods, soft-decision (erasure) Reed-Solomon decoding,
//! following runs of back-to-back frames to find ones whose syncword or
//! header is too damaged to search for, and combining retransmissions of
//! the same frame. Every frame reported as verified still has to pass
//! Reed-Solomon and its CSP CRC-32C.

pub mod audio;
pub mod audio_check;
pub mod dsp;
pub mod error;
pub mod fec;
pub mod framing;
mod pfb_taps;
pub mod pipeline;

pub use error::DecodeError;
