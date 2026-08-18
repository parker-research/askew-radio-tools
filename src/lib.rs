//! AX100 Mode 5 (Reed-Solomon / CSP) beacon decoder for SatNOGS audio
//! captures.
//!
//! Implements the full decode chain, closely ported from gr-satellites'
//! `ax100_deframer(mode='RS')` + `ax100_decode`:
//!   Audio (WAV/OGG) → FM discriminator → LPF → symbol timing →
//!   bit decisions → additive descrambler → syncword search →
//!   Reed-Solomon (255,223) → CSP frame → CSP CRC-32C check

pub mod audio;
pub mod audio_check;
pub mod descramble;
pub mod dsp;
pub mod error;
pub mod fec;
pub mod framing;
pub mod pipeline;

pub use error::DecodeError;
