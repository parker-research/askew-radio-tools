use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("Audio I/O error: {0}")]
    Audio(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("Unsupported sample rate {rate} Hz — need at least {min} Hz for 9600 baud")]
    SampleRateTooLow { rate: u32, min: u32 },

    #[error("Reed-Solomon decode failed — frame has invalid LEN byte or >16 byte errors")]
    ReedSolomonFailed,
}
