use crate::header::{ComputeRegion, Header, TensorDtype, TensorFlags};
use crate::slots::{read_slots, Slot};
use crate::{Error, Result, BLOB_ALIGNMENT, FORMAT_VERSION, MAGIC, PREFIX_LEN};
use memmap2::Mmap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Zero-copy mmap reader for `.base` files.
pub struct BaseReader {
    mmap: Mmap,
    header: Header,
    blob_offset: u64,
}

impl BaseReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // Safety: the file is expected to be immutable for the lifetime of
        // the reader. Callers that mutate concurrently will see tearing —
        // documented invariant.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let bytes = mmap.as_ref();
        if (bytes.len() as u64) < PREFIX_LEN {
            return Err(Error::Truncated);
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(Error::BadMagic(magic));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version, FORMAT_VERSION));
        }

        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let header_end = PREFIX_LEN + header_len;
        if header_end > bytes.len() as u64 {
            return Err(Error::HeaderOverflow(header_len));
        }

        let header_bytes = &bytes[PREFIX_LEN as usize..header_end as usize];
        let header = Header::from_json_bytes(header_bytes)?;

        // Enforce the SSM A-matrix invariant at open time. An A-matrix
        // stored at the wrong dtype or region is a silent correctness bug
        // that manifests as NaN after ~100 recurrent steps — much better
        // to fail loud at load.
        for t in header.tensors.iter() {
            if t.flags.contains(TensorFlags::SSM_A_MATRIX) {
                let dtype_ok = matches!(t.dtype, TensorDtype::F32);
                let region_ok = matches!(t.compute_region, ComputeRegion::Cpu);
                if !dtype_ok || !region_ok {
                    return Err(Error::InvalidSsmAMatrix {
                        name: t.name.clone(),
                        dtype: format!("{:?}", t.dtype),
                        region: format!("{:?}", t.compute_region),
                    });
                }
            }
        }

        let blob_offset = align_up(header_end, BLOB_ALIGNMENT);

        Ok(Self {
            mmap,
            header,
            blob_offset,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Parse only the header without mmapping the whole file. Reads the
    /// 16-byte prefix to learn `header_len`, then reads exactly the header
    /// JSON. Cheap enough to scan a directory of multi-GB `.base` files for
    /// a `list`-style catalog — `open()` would mmap each blob in full.
    pub fn read_header<P: AsRef<Path>>(path: P) -> Result<Header> {
        let mut file = File::open(path)?;

        let mut prefix = [0u8; PREFIX_LEN as usize];
        file.read_exact(&mut prefix).map_err(|e| match e.kind() {
            std::io::ErrorKind::UnexpectedEof => Error::Truncated,
            _ => Error::Io(e),
        })?;

        let magic: [u8; 4] = prefix[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(Error::BadMagic(magic));
        }
        let version = u32::from_le_bytes(prefix[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion(version, FORMAT_VERSION));
        }
        let header_len = u64::from_le_bytes(prefix[8..16].try_into().unwrap());

        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes).map_err(|e| match e.kind() {
            std::io::ErrorKind::UnexpectedEof => Error::HeaderOverflow(header_len),
            _ => Error::Io(e),
        })?;
        Ok(Header::from_json_bytes(&header_bytes)?)
    }

    pub fn blob_offset(&self) -> u64 {
        self.blob_offset
    }

    /// Get a zero-copy slice of a tensor's raw bytes.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::TensorNotFound {
                name: name.to_string(),
            })?;

        let abs_start = self.blob_offset + entry.offset;
        let abs_end = abs_start + entry.length;
        let total = self.mmap.len() as u64;
        if abs_end > total {
            return Err(Error::TensorOutOfBounds {
                name: name.to_string(),
                offset: entry.offset,
                length: entry.length,
                blob_size: total - self.blob_offset,
            });
        }

        Ok(&self.mmap[abs_start as usize..abs_end as usize])
    }

    /// Absolute file offset for a tensor's first byte. Useful for
    /// constructing zero-copy buffers (MTLBuffer.makeBufferWithBytesNoCopy,
    /// cudaHostRegister, etc.) without holding a slice.
    pub fn tensor_file_offset(&self, name: &str) -> Result<u64> {
        let entry = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::TensorNotFound {
                name: name.to_string(),
            })?;
        Ok(self.blob_offset + entry.offset)
    }

    /// Whether a tensor's file offset is aligned to its declared region's
    /// alignment. Callers use this as a precondition for zero-copy buffer
    /// creation — a misaligned GPU-region tensor means the page-aligned
    /// invariant was violated and `makeBufferWithBytesNoCopy` would
    /// silently fall back to a copy.
    pub fn tensor_is_zero_copy_eligible(&self, name: &str) -> Result<bool> {
        let entry = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::TensorNotFound {
                name: name.to_string(),
            })?;
        let align = self.header.alignment.align_for(entry.compute_region);
        let abs = self.blob_offset + entry.offset;
        Ok(abs % align == 0)
    }

    /// Iterate tensors filtered by region.
    pub fn tensors_in_region(
        &self,
        region: ComputeRegion,
    ) -> impl Iterator<Item = &crate::header::TensorEntry> {
        self.header
            .tensors
            .iter()
            .filter(move |t| t.compute_region == region)
    }

    /// Parse extension slots. Returns an empty vec if none are present.
    /// Slots live after the weights blob; this scans to find their start.
    pub fn slots(&self) -> Result<Vec<Slot>> {
        let slots_offset = self.slots_offset();
        if slots_offset >= self.mmap.len() as u64 {
            return Ok(Vec::new());
        }
        let mut cursor = std::io::Cursor::new(&self.mmap[slots_offset as usize..]);
        read_slots(&mut cursor)
    }

    /// Compute the file offset where the slots section begins. One byte
    /// past the last tensor's end, rounded up to 8 bytes.
    fn slots_offset(&self) -> u64 {
        let blob_end = self
            .header
            .tensors
            .iter()
            .map(|t| self.blob_offset + t.offset + t.length)
            .max()
            .unwrap_or(self.blob_offset);
        (blob_end + 7) & !7u64
    }

    /// Verify a single tensor's xxhash64. Callers invoke this lazily on
    /// first use, or eagerly in a `--validate` mode. Returns Ok(()) when
    /// the checksum matches OR when no checksum is recorded. Returns
    /// Err(ChecksumMismatch) when the recorded checksum disagrees with
    /// the current bytes.
    pub fn verify_tensor(&self, name: &str) -> Result<()> {
        let entry = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| Error::TensorNotFound {
                name: name.to_string(),
            })?;
        let Some(expected) = entry.checksum_xxh64 else {
            return Ok(());
        };
        let bytes = self.tensor_bytes(name)?;
        let actual = xxhash_rust::xxh64::xxh64(bytes, 0);
        if actual != expected {
            return Err(Error::ChecksumMismatch {
                name: name.to_string(),
                expected,
                actual,
            });
        }
        Ok(())
    }
}

fn align_up(x: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}
