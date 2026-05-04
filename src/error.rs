use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("Audio I/O error: {0}")]
    Audio(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Unsupported sample rate {rate} Hz — need at least {min} Hz for 9600 baud")]
    SampleRateTooLow { rate: u32, min: u32 },

    #[error("Sync word not found in audio")]
    SyncNotFound,

    #[error("Golay decode failed — too many errors in length field")]
    GolayFailed,

    #[error("Reed-Solomon decode failed — frame has >{0} byte errors", 16)]
    ReedSolomonFailed,

    #[error("CRC-32C mismatch — frame corrupted beyond RS correction")]
    CrcMismatch,

    #[error("Frame length {0} out of valid range (0–223 bytes)")]
    InvalidFrameLength(usize),
}
