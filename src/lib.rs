//! AX100 "ASM+Golay" beacon decoder for SatNOGS audio captures
//! (FRONTIERSAT / NORAD 69015 config: `framing: AX100 ASM+Golay`,
//! `scrambler: CCSDS`, 9600 baud, 3200 Hz deviation).
//!
//! Implements the full decode chain, closely ported from gr-satellites'
//! `ax100_deframer(mode='ASM')` + `u482c_decode`:
//!   Audio (WAV/OGG) → LPF → symbol timing → bit decisions →
//!   syncword search → Golay(24,12) length header → CCSDS derandomize →
//!   Reed-Solomon (255,223) → CSP frame → CSP CRC-32C check

pub mod audio;
pub mod audio_check;
pub mod dsp;
pub mod error;
pub mod fec;
pub mod framing;
pub mod pipeline;

pub use error::DecodeError;
