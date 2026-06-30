use crate::header::{Header, MmprojBundle, TensorEntry};
use crate::slots::{write_slots, Slot};
use crate::{Error, Result, BLOB_ALIGNMENT, FORMAT_VERSION, MAGIC, PREFIX_LEN};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

/// A tensor payload staged for writing.
///
/// `offset`/`length` in the final file are assigned by the writer; callers
/// populate everything except those two fields.
pub struct TensorPayload {
    pub entry: TensorEntry,
    pub data: Vec<u8>,
}

/// Writer for the `.base` single-file format.
///
/// Usage:
/// 1. Create a `BaseWriter` with a target header (tensor offsets ignored).
/// 2. Push each tensor payload with `add_tensor`.
/// 3. Call `finish` to commit — partitions tensors by `compute_region`,
///    assigns per-region alignments, serializes the canonical-JSON header,
///    writes prefix + padding + blob.
pub struct BaseWriter<W: Write + Seek> {
    inner: W,
    header: Header,
    payloads: Vec<TensorPayload>,
    /// Tensors destined for the multimodal sub-bundle (vision/audio
    /// towers + projector). Stored in the same weights blob as the LM
    /// tensors but listed under `header.mmproj.tensors` instead of
    /// `header.tensors`. Empty when the model is text-only.
    mmproj_payloads: Vec<TensorPayload>,
    /// `header.mmproj.arch` value when `mmproj_payloads` is non-empty.
    /// "gemma4_vision_audio", "gemma4_vision", etc.
    mmproj_arch: Option<String>,
    /// `header.mmproj.config` block (vision_config / audio_config /
    /// multimodal token IDs). Carried verbatim from the source HF config
    /// so the runtime can populate vision/audio fields without a
    /// separate config file.
    mmproj_config: std::collections::BTreeMap<String, serde_json::Value>,
    slots: Vec<Slot>,
}

impl BaseWriter<BufWriter<File>> {
    pub fn create<P: AsRef<Path>>(path: P, header: Header) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            inner: BufWriter::new(file),
            header,
            payloads: Vec::new(),
            mmproj_payloads: Vec::new(),
            mmproj_arch: None,
            mmproj_config: std::collections::BTreeMap::new(),
            slots: Vec::new(),
        })
    }
}

impl<W: Write + Seek> BaseWriter<W> {
    pub fn new(inner: W, header: Header) -> Self {
        Self {
            inner,
            header,
            payloads: Vec::new(),
            mmproj_payloads: Vec::new(),
            mmproj_arch: None,
            mmproj_config: std::collections::BTreeMap::new(),
            slots: Vec::new(),
        }
    }

    pub fn add_tensor(&mut self, payload: TensorPayload) {
        self.payloads.push(payload);
    }

    /// Add a tensor that belongs to the multimodal sub-bundle
    /// (vision/audio tower or projector head). The payload is written
    /// into the same weights blob, but its entry lands under
    /// `header.mmproj.tensors` instead of `header.tensors`.
    pub fn add_mmproj_tensor(&mut self, payload: TensorPayload) {
        self.mmproj_payloads.push(payload);
    }

    /// Set the mmproj sub-bundle arch tag (e.g. "gemma4_vision_audio").
    /// Required when any `add_mmproj_tensor` was called.
    pub fn set_mmproj_arch(&mut self, arch: impl Into<String>) {
        self.mmproj_arch = Some(arch.into());
    }

    /// Set the mmproj sub-bundle config block. Keys mirror the source HF
    /// config: `vision_config`, `audio_config`, `image_token_id`,
    /// `boi_token_id`, `eoi_token_id`, `audio_token_id`, `boa_token_id`,
    /// `eoa_token_id`, `vision_soft_tokens_per_image`, `audio_seq_length`,
    /// `audio_ms_per_token`, `image_processor.pooling_kernel_size`.
    pub fn set_mmproj_config(
        &mut self,
        cfg: std::collections::BTreeMap<String, serde_json::Value>,
    ) {
        self.mmproj_config = cfg;
    }

    pub fn add_slot(&mut self, slot: Slot) {
        self.slots.push(slot);
    }

    pub fn finish(mut self) -> Result<()> {
        let alignment = self.header.alignment;

        // Assign each tensor an offset honoring its compute-region's
        // alignment. We walk the payloads in user-specified order (the
        // residency convention cares about ordering) and pad to the
        // per-tensor alignment before each write.
        let mut blob_cursor: u64 = 0;
        let mut entries: Vec<TensorEntry> = Vec::with_capacity(self.payloads.len());
        for p in &self.payloads {
            let align = alignment.align_for(p.entry.compute_region);
            let aligned = align_up(blob_cursor, align);
            let mut entry = p.entry.clone();
            entry.offset = aligned;
            entry.length = p.data.len() as u64;
            if entry.checksum_xxh64.is_none() {
                entry.checksum_xxh64 = Some(xxhash_rust::xxh64::xxh64(&p.data, 0));
            }
            entries.push(entry);
            blob_cursor = aligned + p.data.len() as u64;
        }
        self.header.tensors = entries;

        // Multimodal sub-bundle entries land in the same weights blob,
        // continuing past the LM tensors. Their entries go into
        // `header.mmproj.tensors` so the runtime can decide whether to
        // load them based on the active task.
        if !self.mmproj_payloads.is_empty() {
            let mut mmproj_entries: Vec<TensorEntry> =
                Vec::with_capacity(self.mmproj_payloads.len());
            for p in &self.mmproj_payloads {
                let align = alignment.align_for(p.entry.compute_region);
                let aligned = align_up(blob_cursor, align);
                let mut entry = p.entry.clone();
                entry.offset = aligned;
                entry.length = p.data.len() as u64;
                if entry.checksum_xxh64.is_none() {
                    entry.checksum_xxh64 = Some(xxhash_rust::xxh64::xxh64(&p.data, 0));
                }
                mmproj_entries.push(entry);
                blob_cursor = aligned + p.data.len() as u64;
            }
            let arch = self
                .mmproj_arch
                .clone()
                .unwrap_or_else(|| "mmproj".to_string());
            self.header.mmproj = Some(MmprojBundle {
                arch,
                config: std::mem::take(&mut self.mmproj_config),
                tensors: mmproj_entries,
            });
        }

        // Serialize header (canonical JSON, sorted keys via BTreeMap).
        let header_json = self.header.to_canonical_json().map_err(Error::Json)?;
        let header_len = header_json.len() as u64;

        // Prefix: magic + version + header_len.
        self.inner.write_all(&MAGIC)?;
        self.inner.write_all(&FORMAT_VERSION.to_le_bytes())?;
        self.inner.write_all(&header_len.to_le_bytes())?;

        // Header.
        self.inner.write_all(&header_json)?;

        // Pad to blob start. The blob always starts at the maximum
        // platform page size (BLOB_ALIGNMENT = 64 KiB) so that a tensor
        // aligned to 16 KiB (Apple) or 64 KiB (NVIDIA) within the blob is
        // also page-aligned in the file.
        let header_end = PREFIX_LEN + header_len;
        let blob_start = align_up(header_end, BLOB_ALIGNMENT);
        let pad_bytes = (blob_start - header_end) as usize;
        if pad_bytes > 0 {
            let zeros = vec![0u8; pad_bytes];
            self.inner.write_all(&zeros)?;
        }

        // Tensor data with per-tensor alignment padding.
        let mut blob_written: u64 = 0;
        for (p, e) in self.payloads.iter().zip(self.header.tensors.iter()) {
            if e.offset > blob_written {
                let pad = (e.offset - blob_written) as usize;
                let zeros = vec![0u8; pad];
                self.inner.write_all(&zeros)?;
                blob_written += pad as u64;
            }
            self.inner.write_all(&p.data)?;
            blob_written += p.data.len() as u64;
        }

        // Mmproj data continues past the LM blob. The header has both
        // tensor lists already pointing into the same offset space.
        if let Some(mmproj) = &self.header.mmproj {
            for (p, e) in self.mmproj_payloads.iter().zip(mmproj.tensors.iter()) {
                if e.offset > blob_written {
                    let pad = (e.offset - blob_written) as usize;
                    let zeros = vec![0u8; pad];
                    self.inner.write_all(&zeros)?;
                    blob_written += pad as u64;
                }
                self.inner.write_all(&p.data)?;
                blob_written += p.data.len() as u64;
            }
        }

        // Extension slots, if any. Pad to 8-byte boundary first so the
        // slots section starts aligned.
        if !self.slots.is_empty() {
            let pad = (8 - (blob_written % 8)) % 8;
            if pad > 0 {
                self.inner.write_all(&[0u8; 8][..pad as usize])?;
            }
            write_slots(&mut self.inner, &self.slots)?;
        }

        self.inner.flush()?;
        Ok(())
    }
}

fn align_up(x: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two(), "alignment must be power of two");
    (x + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(100, 65536), 65536);
    }
}
