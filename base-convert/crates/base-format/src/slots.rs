//! Extension slots — typed, length-prefixed, skippable payloads that live
//! after the weights blob.
//!
//! Design goals:
//! - Forward compatibility: a loader that doesn't recognize a slot type
//!   skips it using `byte_length`, no re-encode needed to add new types.
//! - No re-write of the main bundle when adding LoRA / speculator /
//!   compiled graph — slots append.
//! - Keep the main header JSON small; large binary blobs (compiled
//!   MPSGraph, LoRA weights) live in slot payloads instead.
//!
//! Wire format (each slot, back-to-back after the weights blob end):
//!
//! ```text
//!   u16  slot_kind       (see SlotKind)
//!   u16  slot_flags      (bit 0 = REQUIRED, bit 1 = COMPRESSED_ZSTD)
//!   u64  payload_length  (bytes after this header, before padding)
//!   u64  payload_xxh64   (xxhash64 of payload bytes, zero if omitted)
//!   u8[payload_length]   payload
//!   <pad to 8-byte boundary>
//! ```
//!
//! The slots section begins at the first 64-byte aligned offset after
//! the weights blob ends. Its start is recorded in the `slots_offset`
//! field of the header. An `u32 n_slots` precedes the slot records.

use crate::{Error, Result};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

/// Slot types. Unknown values are skipped by the loader using
/// `payload_length`, preserving forward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum SlotKind {
    /// LoRA delta weights (rank-decomposed) that fuse onto the base model.
    /// Payload format: `u16 rank`, `u32 n_tensors`, then TensorEntry+data
    /// pairs (like a mini-bundle).
    LoraWeights = 0x0001,
    /// Pre-compiled compute graph (MPSGraph archive on Apple, CUDA graph
    /// on NVIDIA). Payload starts with `u8 graph_platform`, then graph-
    /// specific bytes.
    ComputeGraph = 0x0002,
    /// KV-cache warmup token sequence. Payload: `u32 n_tokens`, then
    /// `u32[n_tokens]` token IDs.
    KvWarmup = 0x0003,
    /// Correctness trace: fixed 16-token input + expected fp32 logits.
    /// Payload: `u32[16]` input tokens, `u32 vocab_size`,
    /// `f32[vocab_size]` expected logits.
    TraceReference = 0x0004,
    /// Precomputed RoPE cos/sin tables. Payload: `u32 max_seq_len`,
    /// `u8 dtype` (0=f32, 1=f16, 2=bf16), then cos table then sin table.
    RopeTables = 0x0005,
    /// Per-tensor activation statistics from AWQ calibration. Payload is
    /// JSON (variable length), not fixed binary, for auditability.
    CalibrationData = 0x0006,
    /// Vendor-defined. Payload is a length-prefixed vendor ID + schema
    /// version + opaque bytes.
    Custom = 0x00FF,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct SlotFlags: u16 {
        /// Loader must reject the file if it cannot process this slot.
        /// Use sparingly — most slots should be optional.
        const REQUIRED        = 1 << 0;
        /// Payload is zstd-compressed. Slot length is the compressed size;
        /// an inline u64 at the start of the payload holds the uncompressed
        /// size. Not implemented in v1 — reserved bit.
        const COMPRESSED_ZSTD = 1 << 1;
    }
}

/// A slot as it appears in memory. On disk the payload is adjacent to
/// the record; in Rust we carry it in a Vec for ergonomics.
#[derive(Debug, Clone)]
pub struct Slot {
    pub kind_raw: u16,
    pub flags: SlotFlags,
    pub payload: Vec<u8>,
    /// Expected xxh64 of payload. Zero means "unchecked".
    pub payload_xxh64: u64,
}

impl Slot {
    pub fn new(kind: SlotKind, payload: Vec<u8>) -> Self {
        let payload_xxh64 = xxhash_rust::xxh64::xxh64(&payload, 0);
        Self {
            kind_raw: kind as u16,
            flags: SlotFlags::empty(),
            payload,
            payload_xxh64,
        }
    }

    pub fn with_flags(mut self, flags: SlotFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn kind(&self) -> Option<SlotKind> {
        match self.kind_raw {
            0x0001 => Some(SlotKind::LoraWeights),
            0x0002 => Some(SlotKind::ComputeGraph),
            0x0003 => Some(SlotKind::KvWarmup),
            0x0004 => Some(SlotKind::TraceReference),
            0x0005 => Some(SlotKind::RopeTables),
            0x0006 => Some(SlotKind::CalibrationData),
            0x00FF => Some(SlotKind::Custom),
            _ => None,
        }
    }
}

/// Write `n_slots + [Slot; n_slots]` to a writer. Caller is responsible
/// for positioning the writer at the slots section start.
pub fn write_slots<W: Write>(writer: &mut W, slots: &[Slot]) -> Result<()> {
    writer.write_all(&(slots.len() as u32).to_le_bytes())?;
    for s in slots {
        writer.write_all(&s.kind_raw.to_le_bytes())?;
        writer.write_all(&s.flags.bits().to_le_bytes())?;
        writer.write_all(&(s.payload.len() as u64).to_le_bytes())?;
        writer.write_all(&s.payload_xxh64.to_le_bytes())?;
        writer.write_all(&s.payload)?;
        let pad = (8 - (s.payload.len() % 8)) % 8;
        if pad > 0 {
            writer.write_all(&[0u8; 8][..pad])?;
        }
    }
    Ok(())
}

/// Parse slots from a reader. Unknown kinds are retained (skipped at
/// dispatch time) so forward compatibility is preserved.
pub fn read_slots<R: Read>(reader: &mut R) -> Result<Vec<Slot>> {
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let n = u32::from_le_bytes(buf4) as usize;

    let mut slots = Vec::with_capacity(n);
    for _ in 0..n {
        let mut buf2 = [0u8; 2];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf2)?;
        let kind_raw = u16::from_le_bytes(buf2);
        reader.read_exact(&mut buf2)?;
        let flags = SlotFlags::from_bits_truncate(u16::from_le_bytes(buf2));
        reader.read_exact(&mut buf8)?;
        let payload_len = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let payload_xxh64 = u64::from_le_bytes(buf8);

        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;

        // Verify checksum if non-zero.
        if payload_xxh64 != 0 {
            let got = xxhash_rust::xxh64::xxh64(&payload, 0);
            if got != payload_xxh64 {
                return Err(Error::ChecksumMismatch {
                    name: format!("slot@kind=0x{:04x}", kind_raw),
                    expected: payload_xxh64,
                    actual: got,
                });
            }
        }

        // Skip pad bytes.
        let pad = (8 - (payload_len % 8)) % 8;
        if pad > 0 {
            let mut sink = [0u8; 8];
            reader.read_exact(&mut sink[..pad])?;
        }

        slots.push(Slot {
            kind_raw,
            flags,
            payload,
            payload_xxh64,
        });
    }
    Ok(slots)
}
