use super::dequant::GgmlType;
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// One metadata-KV value. GGUF supports 13 scalar types + arrays.
#[derive(Debug, Clone)]
pub enum KvValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<KvValue>),
}

/// Per-tensor descriptor from the GGUF header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset from the start of the tensor_data region (not the file).
    pub data_offset: u64,
}

/// A parsed, mmap-backed GGUF file. Tensor data is accessed lazily via
/// `tensor_bytes`.
pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: BTreeMap<String, KvValue>,
    pub tensors: Vec<TensorInfo>,
    tensor_data_base: u64,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("opening {:?}", path.as_ref()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let mut cur = Cursor::new(&mmap);

        let magic = cur.read_bytes(4)?;
        if magic != MAGIC {
            bail!("not a GGUF file: bad magic {:?}", magic);
        }
        let version = cur.read_u32()?;
        if version != 2 && version != 3 {
            bail!("unsupported GGUF version: {version}");
        }

        let tensor_count = cur.read_u64()?;
        let kv_count = cur.read_u64()?;

        // Metadata KV store.
        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cur.read_string()?;
            let value = cur.read_value()?;
            metadata.insert(key, value);
        }

        // Tensor info array.
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cur.read_u64()?);
            }
            let ggml_ty = GgmlType::from_u32(cur.read_u32()?)?;
            let data_offset = cur.read_u64()?;
            tensors.push(TensorInfo {
                name,
                shape,
                ggml_type: ggml_ty,
                data_offset,
            });
        }

        // Align cursor to tensor_data base. "general.alignment" override
        // lives in metadata; default 32.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                KvValue::U32(n) => Some(*n as u64),
                KvValue::U64(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);
        let tensor_data_base = align_up(cur.pos, alignment);

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            tensor_data_base,
        })
    }

    /// Zero-copy view of a tensor's raw bytes. Length is derived from
    /// shape + ggml_type block size — GGUF doesn't store per-tensor
    /// byte length, but it's computable.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let nbytes = tensor_byte_length(info)?;
        let start = (self.tensor_data_base + info.data_offset) as usize;
        let end = start + nbytes;
        if end > self.mmap.len() {
            bail!(
                "tensor {:?} byte range [{}, {}) exceeds file size {}",
                info.name,
                start,
                end,
                self.mmap.len()
            );
        }
        Ok(&self.mmap[start..end])
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Architecture string from metadata (`general.architecture`).
    pub fn arch(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(KvValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Compute a tensor's total byte length from shape and ggml_type.
pub fn tensor_byte_length(info: &TensorInfo) -> Result<usize> {
    let n: u64 = info.shape.iter().product();
    let (block_size, bytes_per_block) = info.ggml_type.block_geometry();
    if n % (block_size as u64) != 0 {
        bail!(
            "tensor {:?}: element count {} not divisible by block_size {}",
            info.name,
            n,
            block_size
        );
    }
    let n_blocks = n / block_size as u64;
    Ok(n_blocks as usize * bytes_per_block)
}

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let start = self.pos as usize;
        let end = start + n;
        if end > self.buf.len() {
            bail!("cursor overflow: {} > {}", end, self.buf.len());
        }
        self.pos = end as u64;
        Ok(&self.buf[start..end])
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_bytes(1)?[0] as i8)
    }
    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }
    fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }
    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }
    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }
    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Ok(std::str::from_utf8(bytes)
            .map_err(|e| anyhow::anyhow!("non-UTF-8 GGUF string: {e}"))?
            .to_string())
    }

    fn read_value(&mut self) -> Result<KvValue> {
        let ty = self.read_u32()?;
        self.read_value_of_type(ty)
    }

    fn read_value_of_type(&mut self, ty: u32) -> Result<KvValue> {
        Ok(match ty {
            0 => KvValue::U8(self.read_u8()?),
            1 => KvValue::I8(self.read_i8()?),
            2 => KvValue::U16(self.read_u16()?),
            3 => KvValue::I16(self.read_i16()?),
            4 => KvValue::U32(self.read_u32()?),
            5 => KvValue::I32(self.read_i32()?),
            6 => KvValue::F32(self.read_f32()?),
            7 => KvValue::Bool(self.read_bool()?),
            8 => KvValue::String(self.read_string()?),
            9 => {
                let elem_ty = self.read_u32()?;
                let n = self.read_u64()? as usize;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n {
                    out.push(self.read_value_of_type(elem_ty)?);
                }
                KvValue::Array(out)
            }
            10 => KvValue::U64(self.read_u64()?),
            11 => KvValue::I64(self.read_i64()?),
            12 => KvValue::F64(self.read_f64()?),
            other => bail!("unknown GGUF value type: {other}"),
        })
    }
}

impl KvValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            KvValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            KvValue::U8(n) => Some(*n as u64),
            KvValue::U16(n) => Some(*n as u64),
            KvValue::U32(n) => Some(*n as u64),
            KvValue::U64(n) => Some(*n),
            KvValue::I8(n) if *n >= 0 => Some(*n as u64),
            KvValue::I16(n) if *n >= 0 => Some(*n as u64),
            KvValue::I32(n) if *n >= 0 => Some(*n as u64),
            KvValue::I64(n) if *n >= 0 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            KvValue::F32(f) => Some(*f),
            KvValue::F64(f) => Some(*f as f32),
            _ => None,
        }
    }
}
