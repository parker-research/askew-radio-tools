//! AX100 satellite beacon decoder for SatNOGS audio captures.
//!
//! Implements the full decode chain:
//!   Audio (WAV/OGG) → FM discriminator → LPF → symbol timing →
//!   bit decisions → sync word search → Golay length decode →
//!   CCSDS de-randomization → Reed-Solomon → CRC-32C → CSP payload

pub mod audio;
pub mod dsp;
pub mod error;
pub mod fec;
pub mod framing;

pub use error::DecodeError;
