use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("bad magic: expected BASE, got {0:?}")]
    BadMagic([u8; 4]),

    #[error("unsupported format version: {0} (this build supports {1})")]
    UnsupportedVersion(u32, u32),

    #[error("file too small to contain a valid .base prefix")]
    Truncated,

    #[error("header length {0} exceeds file size")]
    HeaderOverflow(u64),

    #[error("tensor {name:?} not found")]
    TensorNotFound { name: String },

    #[error("tensor {name:?} offset/length out of bounds (offset={offset}, length={length}, blob_size={blob_size})")]
    TensorOutOfBounds {
        name: String,
        offset: u64,
        length: u64,
        blob_size: u64,
    },

    #[error("tensor {name:?} checksum mismatch (expected {expected:016x}, got {actual:016x})")]
    ChecksumMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },

    #[error("tensor {name:?} marked SSM_A_MATRIX but is {dtype} in region {region:?} — must be f32 in cpu")]
    InvalidSsmAMatrix {
        name: String,
        dtype: String,
        region: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
