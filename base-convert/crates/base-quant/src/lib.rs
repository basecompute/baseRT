//! Quantization packing routines for the `.base` canonical schemes.
//!
//! All packers produce a `Packed` value carrying three byte streams that
//! the BaseWriter places in distinct regions of the weights blob:
//!
//! - `packed_weights`  — the quantized bytes proper
//! - `scales`          — per-group fp16 scales
//! - `biases`          — per-group fp16 zero-point biases (asymmetric
//!   schemes only; empty for symmetric)
//!
//! Packing is bit-exact against a reference (worktree's `quants.py`
//! `simple_affine_q4`) so the Rust converter and the study's Python
//! reference produce identical on-disk bytes for the same input.

pub mod base_q2;
pub mod base_q3;
pub mod base_q4;
pub mod base_q5;
pub mod base_q6;
pub mod base_q8;
pub mod base_qn;
pub mod mxfp4;
pub mod nvfp4;
pub mod profile;
pub mod rtn;

pub use profile::{QuantProfile, ResolvedQuant, RuleEntry};
pub use rtn::{pack as pack_rtn, unpack as unpack_rtn, RtnConfig};

/// One packed tensor with its three byte streams.
#[derive(Debug, Clone)]
pub struct Packed {
    pub packed_weights: Vec<u8>,
    pub scales: Vec<u8>,
    pub biases: Vec<u8>,
    pub group_size: u32,
    /// Per-group scale storage dtype actually used to encode `scales` and
    /// `biases`. The runtime reads this from `TensorEntry.scale_dtype` to
    /// pick the matching `_sbf16` GEMV/GEMM kernel sibling. `None` for
    /// non-quantized packs (dense bf16/f16 fallbacks) where there are no
    /// per-group scales.
    pub scale_dtype: Option<base_format::ScaleDtype>,
}

impl Packed {
    pub fn is_symmetric(&self) -> bool {
        self.biases.is_empty()
    }
}
