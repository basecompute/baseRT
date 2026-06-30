//! GGUF (v2/v3) parser.
//!
//! Wire format:
//! ```text
//!   magic        : u32 = b"GGUF" little-endian
//!   version      : u32 = 2 or 3
//!   tensor_count : u64
//!   kv_count     : u64
//!   kv_entries   : [KvEntry; kv_count]
//!   tensor_infos : [TensorInfo; tensor_count]
//!   <align to ALIGNMENT>
//!   tensor_data  : bytes (addressed by per-tensor offset from tensor_data base)
//! ```
//!
//! All integers little-endian. Strings are `u64 len + bytes` (UTF-8, no NUL).
//! The tensor_data base is the first multiple of `general.alignment` (default
//! 32) at or after the end of tensor_infos.

mod dequant;
mod parse;

pub use dequant::{dequant_to_f32, ggml_type_name, GgmlType};
pub use parse::{GgufFile, KvValue, TensorInfo};
