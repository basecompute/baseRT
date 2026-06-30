//! Safetensors (v0.4.x) parser.
//!
//! Wire format:
//! ```text
//!   u64  header_len      (little-endian)
//!   u8[header_len]       header_json (UTF-8)
//!   u8[]                 tensor_data (addressed by per-tensor
//!                                     data_offsets in the header)
//! ```
//!
//! Header JSON shape:
//! ```json
//! {
//!   "tensor.name": {
//!     "dtype": "F32" | "F16" | "BF16" | "I8" | "U32" | ...,
//!     "shape": [10, 20],
//!     "data_offsets": [start, end]
//!   },
//!   "__metadata__": { "format": "pt", ... }
//! }
//! ```
//! `data_offsets` are relative to the start of tensor_data (i.e. relative
//! to `8 + header_len` from file start).

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    F32,
    F16,
    Bf16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl StDtype {
    // The signature can't satisfy std::str::FromStr because anyhow::Error
    // is not the trait's associated `Err` shape, so we keep the inherent
    // method and allow the should_implement_trait lint.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => StDtype::F32,
            "F16" => StDtype::F16,
            "BF16" => StDtype::Bf16,
            "F64" => StDtype::F64,
            "I8" => StDtype::I8,
            "I16" => StDtype::I16,
            "I32" => StDtype::I32,
            "I64" => StDtype::I64,
            "U8" => StDtype::U8,
            "U16" => StDtype::U16,
            "U32" => StDtype::U32,
            "U64" => StDtype::U64,
            "BOOL" => StDtype::Bool,
            other => bail!("unknown safetensors dtype: {other}"),
        })
    }

    pub fn byte_width(self) -> usize {
        match self {
            StDtype::F32 | StDtype::I32 | StDtype::U32 => 4,
            StDtype::F16 | StDtype::Bf16 | StDtype::I16 | StDtype::U16 => 2,
            StDtype::F64 | StDtype::I64 | StDtype::U64 => 8,
            StDtype::I8 | StDtype::U8 | StDtype::Bool => 1,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Clone)]
pub struct StTensorInfo {
    pub name: String,
    pub dtype: StDtype,
    pub shape: Vec<u64>,
    /// Offset relative to the file start (tensor_data_base + local).
    pub file_offset: u64,
    pub length: u64,
}

/// Mmap-backed safetensors file.
pub struct SafetensorsFile {
    mmap: Mmap,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<StTensorInfo>,
}

impl SafetensorsFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("opening {:?}", path.as_ref()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let bytes = mmap.as_ref();
        if bytes.len() < 8 {
            bail!("safetensors file too short");
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        if header_end > bytes.len() {
            bail!(
                "safetensors header declares {} bytes, file has {}",
                header_len,
                bytes.len() - 8
            );
        }

        let header_bytes = &bytes[8..header_end];
        let raw: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(header_bytes).context("parsing safetensors header")?;

        let mut metadata = BTreeMap::new();
        let mut tensors = Vec::new();
        for (name, value) in raw.into_iter() {
            if name == "__metadata__" {
                if let serde_json::Value::Object(m) = value {
                    for (k, v) in m {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k, s.to_string());
                        }
                    }
                }
                continue;
            }
            let entry: RawTensorEntry = serde_json::from_value(value)
                .with_context(|| format!("parsing tensor entry {:?}", name))?;
            let dtype = StDtype::from_str(&entry.dtype)?;
            let local_begin = entry.data_offsets[0];
            let local_end = entry.data_offsets[1];
            let length = local_end - local_begin;
            let file_offset = header_end as u64 + local_begin;
            tensors.push(StTensorInfo {
                name,
                dtype,
                shape: entry.shape,
                file_offset,
                length,
            });
        }
        Ok(Self {
            mmap,
            metadata,
            tensors,
        })
    }

    /// Zero-copy view of a tensor's bytes.
    pub fn tensor_bytes(&self, info: &StTensorInfo) -> &[u8] {
        let start = info.file_offset as usize;
        let end = start + info.length as usize;
        &self.mmap[start..end]
    }
}
