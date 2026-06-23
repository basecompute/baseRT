//! On-disk format for `.base` files.
//!
//! See `FORMAT.md` at the workspace root for the authoritative spec.

mod error;
mod header;
mod reader;
mod slots;
mod writer;

pub use error::{Error, Result};
pub use header::{
    AlignmentConfig, CalibrationInfo, ComputeRegion, Header, HeaderFlags, LayerDescriptor,
    LayerKind, LayerPrecision, Layout, ModelConfig, QuantScheme, ResidencyHint, ScaleDtype,
    Signature, SourceInfo, TargetBackend, TensorDtype, TensorEntry, TensorFlags, TokenizerBlob,
};
pub use reader::BaseReader;
pub use slots::{read_slots, write_slots, Slot, SlotFlags, SlotKind};
pub use writer::{BaseWriter, TensorPayload};

/// Magic bytes at offset 0.
pub const MAGIC: [u8; 4] = *b"BASE";

/// Current format version. Bump on any breaking layout change.
pub const FORMAT_VERSION: u32 = 1;

/// Weights blob starts at a 64 KiB boundary within the file. This is the
/// maximum page size across supported platforms (NVIDIA/AMD), so a single
/// blob start works everywhere.
pub const BLOB_ALIGNMENT: u64 = 64 * 1024;

/// Fixed-size prefix: magic (4) + version (4) + header_len (8).
pub const PREFIX_LEN: u64 = 16;

/// Default alignment for the accelerator region.
/// 64 B covers Apple ANE DMA (64 B cacheline) and baseline CPU SIMD.
/// Override via `AlignmentConfig::accel_align_log2` for NVIDIA Tensor Core
/// (128 B) or AMD Matrix Core (128 B) targets.
pub const DEFAULT_ACCEL_ALIGN: u64 = 64;

/// Default alignment for the GPU region.
/// 16 KiB is the Apple Metal page size — the minimum for zero-copy
/// MTLBuffer.makeBufferWithBytesNoCopy. Override to 64 KiB for CUDA/ROCm
/// (cudaHostRegister / hipHostRegister requirement).
pub const DEFAULT_GPU_PAGE: u64 = 16 * 1024;

/// CPU region alignment — one cache line on all supported platforms.
pub const DEFAULT_CPU_ALIGN: u64 = 64;
