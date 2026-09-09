use crate::header::{Header, MmprojBundle, TensorDtype, TensorEntry};
use crate::slots::{write_slots, Slot};
use crate::{Error, Result, BLOB_ALIGNMENT, FORMAT_VERSION, MAGIC, PREFIX_LEN};
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// A tensor payload staged for writing.
///
/// `offset`/`length` in the final file are assigned by the writer; callers
/// populate everything except those two fields.
pub struct TensorPayload {
    pub entry: TensorEntry,
    pub data: Vec<u8>,
}

/// Streaming blob sink: the weights blob is written incrementally to a
/// sibling temp file as tensors are added, so the writer never holds the
/// whole (multi-hundred-GB) blob in RAM. Only the tiny per-tensor
/// `TensorEntry` metadata is retained. At `finish` the header is
/// serialized (all offsets/lengths/checksums are already known) and the
/// temp blob is streamed into the final file after it. The on-disk format
/// is byte-identical to the buffered path.
struct BlobStream {
    file: BufWriter<File>,
    path: PathBuf,
    /// Bytes written to the blob so far (blob-relative cursor). Tensor
    /// `entry.offset` values are relative to the blob start, matching the
    /// reader's `blob_offset + entry.offset` addressing.
    cursor: u64,
    lm_entries: Vec<TensorEntry>,
    mmproj_entries: Vec<TensorEntry>,
}

/// Writer for the `.base` single-file format.
///
/// Usage:
/// 1. Create a `BaseWriter` with a target header (tensor offsets ignored).
/// 2. Push each tensor payload with `add_tensor`.
/// 3. Call `finish` to commit — partitions tensors by `compute_region`,
///    assigns per-region alignments, serializes the canonical-JSON header,
///    writes prefix + padding + blob.
///
/// File-backed writers (via [`BaseWriter::create`]) STREAM the blob to a
/// temp file (constant memory). Generic writers (via [`BaseWriter::new`],
/// e.g. an in-memory `Cursor` in tests) buffer payloads in RAM.
pub struct BaseWriter<W: Write + Seek> {
    inner: W,
    header: Header,
    /// Streaming blob sink (Some for file-backed `create`, None for the
    /// buffered generic path).
    stream: Option<BlobStream>,
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
    /// First error encountered while streaming a payload to the temp blob
    /// (add_tensor is infallible for API compatibility); reported at finish.
    stream_error: Option<Error>,
    /// Direct-write mode (`create_direct`): the blob streams straight into
    /// the final file after a fixed reserved header region, so no `.blobtmp`
    /// sibling (and no 2× disk peak) exists. Holds the reserved header byte
    /// count; the header JSON is space-padded to exactly this length at
    /// finish so the reader's `align_up(prefix + header_len)` blob-start
    /// computation lands on the offset the blob was streamed at.
    direct_reserve: Option<u64>,
}

impl BaseWriter<BufWriter<File>> {
    pub fn create<P: AsRef<Path>>(path: P, header: Header) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)?;
        // Sibling temp file for the streamed blob (same directory ⇒ same
        // filesystem, so the final copy is a fast local sequential I/O).
        let tmp_path = {
            let mut s = path.as_os_str().to_os_string();
            s.push(".blobtmp");
            PathBuf::from(s)
        };
        // Read+write: we stream the blob in, then read it back at finish to
        // copy it into the final file (File::create is write-only → EBADF on read).
        let tmp = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        Ok(Self {
            inner: BufWriter::new(file),
            header,
            stream: Some(BlobStream {
                file: BufWriter::new(tmp),
                path: tmp_path,
                cursor: 0,
                lm_entries: Vec::new(),
                mmproj_entries: Vec::new(),
            }),
            payloads: Vec::new(),
            mmproj_payloads: Vec::new(),
            mmproj_arch: None,
            mmproj_config: std::collections::BTreeMap::new(),
            slots: Vec::new(),
            stream_error: None,
            direct_reserve: None,
        })
    }

    /// Direct-write variant of [`BaseWriter::create`]: reserve
    /// `header_reserve` bytes for the header up front and stream the blob
    /// straight into the final file at the fixed blob start — no `.blobtmp`
    /// sibling, so peak disk usage is the bundle size instead of 2×. The
    /// header JSON is space-padded (valid trailing whitespace) to exactly
    /// the reserve at finish, which keeps the reader's
    /// `align_up(prefix + header_len)` blob-start computation equal to the
    /// offset the blob was streamed at; finish errors if the header doesn't
    /// fit (re-run with a larger reserve). Intended for very large bundles
    /// where the temp-blob copy would not fit on disk.
    pub fn create_direct<P: AsRef<Path>>(
        path: P,
        header: Header,
        header_reserve: u64,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create(path)?; // header handle (truncates)
                                        // Second handle for the blob stream, positioned at the fixed blob
                                        // start. Writing beyond EOF leaves the header region as a hole
                                        // until finish backfills it.
        let mut blob = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        let blob_start = align_up(PREFIX_LEN + header_reserve, BLOB_ALIGNMENT);
        blob.seek(SeekFrom::Start(blob_start))?;
        Ok(Self {
            inner: BufWriter::new(file),
            header,
            stream: Some(BlobStream {
                file: BufWriter::new(blob),
                path: path.to_path_buf(),
                cursor: 0,
                lm_entries: Vec::new(),
                mmproj_entries: Vec::new(),
            }),
            payloads: Vec::new(),
            mmproj_payloads: Vec::new(),
            mmproj_arch: None,
            mmproj_config: std::collections::BTreeMap::new(),
            slots: Vec::new(),
            stream_error: None,
            direct_reserve: Some(header_reserve),
        })
    }
}

impl<W: Write + Seek> BaseWriter<W> {
    pub fn new(inner: W, header: Header) -> Self {
        Self {
            inner,
            header,
            stream: None,
            payloads: Vec::new(),
            mmproj_payloads: Vec::new(),
            mmproj_arch: None,
            mmproj_config: std::collections::BTreeMap::new(),
            slots: Vec::new(),
            stream_error: None,
            direct_reserve: None,
        }
    }

    /// Stream one payload into the temp blob, assigning its aligned
    /// blob-relative offset and computing its checksum. The payload's
    /// `data` is dropped immediately after the write, keeping memory flat.
    fn stream_payload(
        alignment: crate::header::AlignmentConfig,
        s: &mut BlobStream,
        payload: TensorPayload,
        is_mmproj: bool,
    ) -> Result<()> {
        let align = alignment.align_for(payload.entry.compute_region);
        let aligned = align_up(s.cursor, align);
        if aligned > s.cursor {
            write_zeros(&mut s.file, (aligned - s.cursor) as usize)?;
        }
        let mut entry = payload.entry;
        entry.offset = aligned;
        entry.length = payload.data.len() as u64;
        if entry.checksum_xxh64.is_none() {
            entry.checksum_xxh64 = Some(xxhash_rust::xxh64::xxh64(&payload.data, 0));
        }
        s.file.write_all(&payload.data)?;
        s.cursor = aligned + payload.data.len() as u64;
        if is_mmproj {
            s.mmproj_entries.push(entry);
        } else {
            s.lm_entries.push(entry);
        }
        Ok(())
    }

    pub fn add_tensor(&mut self, payload: TensorPayload) {
        if let Some(s) = self.stream.as_mut() {
            // Streaming path: file-backed writers never buffer the blob.
            // (add_tensor's signature is infallible for API compatibility;
            // a write error here is surfaced by re-checking at finish via
            // the BufWriter's retained error state on flush.)
            let alignment = self.header.alignment;
            if let Err(e) = Self::stream_payload(alignment, s, payload, false) {
                // Stash the error to report at finish; keep the API simple.
                self.stream_error.get_or_insert(e);
            }
        } else {
            self.payloads.push(payload);
        }
    }

    /// Add a tensor that belongs to the multimodal sub-bundle
    /// (vision/audio tower or projector head). The payload is written
    /// into the same weights blob, but its entry lands under
    /// `header.mmproj.tensors` instead of `header.tensors`.
    pub fn add_mmproj_tensor(&mut self, payload: TensorPayload) {
        if let Some(s) = self.stream.as_mut() {
            // mmproj tensors continue the blob after all LM tensors; callers
            // add every LM tensor before the first mmproj tensor, so the
            // streamed order matches the buffered path.
            let alignment = self.header.alignment;
            if let Err(e) = Self::stream_payload(alignment, s, payload, true) {
                self.stream_error.get_or_insert(e);
            }
        } else {
            self.mmproj_payloads.push(payload);
        }
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

    /// Attach conversion provenance to the header. Call before `finish()`;
    /// accepts both the structured converter record and JSON-producing
    /// architecture-specific paths.
    pub fn set_provenance<T: serde::Serialize>(&mut self, prov: T) {
        self.header.provenance = Some(serde_json::to_value(prov).expect("serializing provenance"));
    }

    /// Stamp `target_backend` from CONTENT instead of trusting the caller's
    /// default (every construction site hardcodes Metal, so CUDA-only bundles
    /// carried a `metal` tag and failed a kernel lookup only after a full
    /// download + load). The one CUDA-only content class today: base_q6 MoE
    /// expert slabs (`*_exps.*` tensors) — Metal ships no q6 MoE kernels.
    /// bf16-scale q8/q4 bundles stay `metal` (universal): both backends carry
    /// the `_sbf16` kernel families.
    ///
    /// Call from EVERY finish path. It used to live inline in the buffered
    /// `finish` only, which the streaming rewrite silently turned into dead
    /// code for real bundles: `create`/`create_direct` both set `stream`, so
    /// they return through `finish_streaming`/`finish_direct` and never
    /// reached it. Requires `header.tensors` to already be populated.
    fn stamp_target_backend_from_content(&mut self) {
        let q6_experts = |ts: &[TensorEntry]| {
            ts.iter()
                .any(|t| t.name.contains("_exps.") && t.dtype == TensorDtype::BaseQ6)
        };
        if q6_experts(&self.header.tensors) {
            self.header.target_backend = crate::header::TargetBackend::CudaSm121;
        }
    }

    pub fn finish(mut self) -> Result<()> {
        // Surface any error stashed while streaming payloads to the temp blob.
        if let Some(e) = self.stream_error.take() {
            return Err(e);
        }
        if let Some(reserve) = self.direct_reserve {
            return self.finish_direct(reserve);
        }
        if self.stream.is_some() {
            return self.finish_streaming();
        }

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

        self.stamp_target_backend_from_content();

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

    /// Streaming commit (file-backed writers). The blob was written to a
    /// temp file as tensors were added, so here we only finalize the header
    /// (offsets/lengths/checksums already recorded) and copy the temp blob
    /// into the final file after the header. Constant memory throughout.
    fn finish_streaming(mut self) -> Result<()> {
        let mut s = self.stream.take().expect("finish_streaming without stream");

        // Finalize the header from the streamed entries.
        self.header.tensors = std::mem::take(&mut s.lm_entries);
        self.stamp_target_backend_from_content();
        if !s.mmproj_entries.is_empty() {
            let arch = self
                .mmproj_arch
                .clone()
                .unwrap_or_else(|| "mmproj".to_string());
            self.header.mmproj = Some(MmprojBundle {
                arch,
                config: std::mem::take(&mut self.mmproj_config),
                tensors: std::mem::take(&mut s.mmproj_entries),
            });
        }

        // Remove the temp blob on EVERY exit from here on, not just the happy
        // path. A failure below (the destination filling mid-copy is the
        // realistic one) used to return through `?` and strand a model-sized
        // `.blobtmp` next to the partial output — hundreds of GB that the user
        // has to find by hand, and that makes every retry fail for space.
        struct TmpGuard(std::path::PathBuf);
        impl Drop for TmpGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let _tmp_guard = TmpGuard(s.path.clone());

        // Flush + rewind the temp blob for reading.
        s.file.flush()?;
        let mut tmp = s.file.into_inner().map_err(|e| e.into_error())?;
        tmp.seek(SeekFrom::Start(0))?;

        // Prefix + header.
        let header_json = self.header.to_canonical_json().map_err(Error::Json)?;
        let header_len = header_json.len() as u64;
        self.inner.write_all(&MAGIC)?;
        self.inner.write_all(&FORMAT_VERSION.to_le_bytes())?;
        self.inner.write_all(&header_len.to_le_bytes())?;
        self.inner.write_all(&header_json)?;

        // Pad to blob start (BLOB_ALIGNMENT), then copy the streamed blob.
        let header_end = PREFIX_LEN + header_len;
        let blob_start = align_up(header_end, BLOB_ALIGNMENT);
        write_zeros(&mut self.inner, (blob_start - header_end) as usize)?;
        {
            let mut reader = BufReader::new(&mut tmp);
            let copied = std::io::copy(&mut reader, &mut self.inner)?;
            debug_assert_eq!(copied, s.cursor, "streamed blob length mismatch");
        }

        // Extension slots after the blob (8-byte aligned).
        if !self.slots.is_empty() {
            let pad = ((8 - (s.cursor % 8)) % 8) as usize;
            if pad > 0 {
                self.inner.write_all(&[0u8; 8][..pad])?;
            }
            write_slots(&mut self.inner, &self.slots)?;
        }

        self.inner.flush()?;
        drop(tmp);
        // _tmp_guard removes the blob here, and on every `?` above.
        Ok(())
    }

    /// Finish for direct-write mode (`create_direct`): the blob already sits
    /// at its final offset, so this only appends the extension slots to the
    /// blob stream and backfills the reserved header region — no copy, no
    /// temp file to delete.
    fn finish_direct(mut self, reserve: u64) -> Result<()> {
        let mut s = self.stream.take().expect("finish_direct without stream");

        self.header.tensors = std::mem::take(&mut s.lm_entries);
        self.stamp_target_backend_from_content();
        if !s.mmproj_entries.is_empty() {
            let arch = self
                .mmproj_arch
                .clone()
                .unwrap_or_else(|| "mmproj".to_string());
            self.header.mmproj = Some(MmprojBundle {
                arch,
                config: std::mem::take(&mut self.mmproj_config),
                tensors: std::mem::take(&mut s.mmproj_entries),
            });
        }

        // Extension slots continue past the blob (8-byte aligned) on the
        // blob handle — it is already positioned at the blob end.
        if !self.slots.is_empty() {
            let pad = ((8 - (s.cursor % 8)) % 8) as usize;
            if pad > 0 {
                s.file.write_all(&[0u8; 8][..pad])?;
            }
            write_slots(&mut s.file, &self.slots)?;
        }
        s.file.flush()?;
        drop(s);

        // Backfill the reserved header region. The JSON is space-padded to
        // exactly `reserve` bytes (trailing whitespace is valid JSON), so
        // header_len = reserve and the reader's blob-start computation
        // matches the offset the blob was streamed at.
        let header_json = self.header.to_canonical_json().map_err(Error::Json)?;
        if header_json.len() as u64 > reserve {
            return Err(Error::Io(std::io::Error::other(format!(
                "direct-write header ({} bytes) exceeds the reserved {} bytes — \
                 re-run with a larger header reserve",
                header_json.len(),
                reserve
            ))));
        }
        self.inner.write_all(&MAGIC)?;
        self.inner.write_all(&FORMAT_VERSION.to_le_bytes())?;
        self.inner.write_all(&reserve.to_le_bytes())?;
        self.inner.write_all(&header_json)?;
        write_fill(
            &mut self.inner,
            b' ',
            (reserve - header_json.len() as u64) as usize,
        )?;
        let header_end = PREFIX_LEN + reserve;
        let blob_start = align_up(header_end, BLOB_ALIGNMENT);
        write_zeros(&mut self.inner, (blob_start - header_end) as usize)?;
        self.inner.flush()?;
        Ok(())
    }
}

/// Write `n` zero bytes to `w` without allocating an n-sized buffer.
fn write_zeros<W: Write>(w: &mut W, n: usize) -> std::io::Result<()> {
    write_fill(w, 0u8, n)
}

/// Write `n` copies of `byte` to `w` without allocating an n-sized buffer.
fn write_fill<W: Write>(w: &mut W, byte: u8, mut n: usize) -> std::io::Result<()> {
    let buf = [byte; 8192];
    while n > 0 {
        let chunk = n.min(buf.len());
        w.write_all(&buf[..chunk])?;
        n -= chunk;
    }
    Ok(())
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
