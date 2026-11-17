//! MLX quantized safetensors reader.
//!
//! MLX stores quantized weights in HF-safetensors directories with a
//! specific convention:
//!
//! - `config.json` has a `"quantization": {"bits": 4, "group_size": 64}`
//!   key declaring the scheme.
//! - A quantized weight `foo.weight` is stored as a `U32` tensor with
//!   shape `[out_features, in_features * bits / 32]`: each row is a
//!   contiguous little-endian bitstream of `bits`-wide codes. For bits
//!   that divide 32 (2/4/8) every code sits at a fixed slot inside one
//!   u32; for 3/5/6-bit codes cross byte and word boundaries, so the
//!   stream must be decoded at bit granularity (MLX supports
//!   bits ∈ {2, 3, 4, 5, 6, 8} for affine quant).
//! - Two additional tensors accompany it: `foo.scales` and `foo.biases`,
//!   both `F16`, shape `[out_features, in_features / group_size]`.
//! - Dequant: `x[i, j] = q[i, j] * scale[i, j / group_size]
//!                       + bias[i, j / group_size]`
//!   where `q[i, j]` is the `bits`-wide field at bit offset `j * bits`
//!   of row `i`'s bitstream.
//!
//! Some MLX models carry per-tensor AWQ overrides in
//! `config.quantization_config.{tensor_name}`; those are read verbatim
//! and written into the `.base` header's `calibration.per_layer_alpha`.

use crate::hf::HfDir;
use crate::safetensors::StDtype;
use anyhow::{bail, Context, Result};
use half::{bf16, f16};

#[derive(Debug, Clone, Copy)]
pub struct MlxQuant {
    pub bits: u32,
    pub group_size: u32,
}

impl MlxQuant {
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        let q = config.get("quantization")?;
        let bits = q.get("bits")?.as_u64()? as u32;
        let group_size = q.get("group_size")?.as_u64()? as u32;
        Some(Self { bits, group_size })
    }
}

pub struct MlxDir {
    pub hf: HfDir,
    pub quant: MlxQuant,
}

/// Borrowed view of an MLX-quantized tensor's raw storage, for
/// passthrough conversion (see [`MlxDir::tensor_packed`]).
pub struct MlxPackedTensor<'a> {
    /// Packed weight bytes (nibble stream, low-nibble first).
    pub packed: &'a [u8],
    /// Per-group scales, stored as `scale_dtype` (BF16 on current
    /// checkpoints, F16 on pre-0.20 ones).
    pub scales: &'a [u8],
    /// Per-group biases, same dtype as scales.
    pub biases: &'a [u8],
    pub group_size: u32,
    pub bits: u32,
    pub scale_dtype: StDtype,
}

impl MlxDir {
    pub fn open<P: AsRef<std::path::Path>>(dir: P) -> Result<Self> {
        let hf = HfDir::open(dir)?;
        let quant = MlxQuant::from_config(&hf.config)
            .context("config.json has no `quantization` block — not an MLX directory")?;
        Ok(Self { hf, quant })
    }

    /// Per-tensor quant override resolved against the `.quantization`
    /// block in `config.json`. MLX checkpoints often raise the precision
    /// of a few sensitive tensors (Gemma 4 26B-A4B: shared-FFN
    /// `mlp.{gate,down,up}_proj` + `router.proj` ship at 8-bit while
    /// everything else is 4-bit). Each override is keyed by the tensor's
    /// stem (HF safetensors name minus the `.weight` suffix). Returns
    /// the global setting when no override applies.
    pub fn quant_for_tensor(&self, name: &str) -> MlxQuant {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let qcfg = match self.hf.config.get("quantization") {
            Some(q) => q,
            None => return self.quant,
        };
        if let Some(override_obj) = qcfg.get(stem) {
            if let (Some(bits), Some(gs)) = (
                override_obj.get("bits").and_then(|v| v.as_u64()),
                override_obj.get("group_size").and_then(|v| v.as_u64()),
            ) {
                return MlxQuant {
                    bits: bits as u32,
                    group_size: gs as u32,
                };
            }
        }
        self.quant
    }

    /// Dequant an MLX tensor to f32. Handles both packed-quant tensors
    /// (looks for `.scales` and `.biases` siblings) and plain tensors
    /// (F16/BF16/F32 passthrough).
    pub fn tensor_to_f32(&self, name: &str) -> Result<Vec<f32>> {
        // If this tensor has a `.scales` sibling, it's MLX-packed.
        let scales_name = quant_sibling(name, "scales");
        if let Some(sn) = &scales_name {
            if self.hf.tensor_info(sn).is_some() {
                return self.dequant_packed(name);
            }
        }
        // Plain tensor — defer to HF.
        self.hf.tensor_to_f32(name)
    }

    fn dequant_packed(&self, name: &str) -> Result<Vec<f32>> {
        let packed = self
            .hf
            .tensor_info(name)
            .with_context(|| format!("packed tensor {name} missing"))?;
        // quant_sibling returns None only if `name` does not end in
        // `.weight`. dequant_packed is invoked from tensor_to_f32
        // exactly when that suffix is present, so the unwrap can't
        // panic — but we propagate the error explicitly anyway to
        // keep the panic-free contract for untrusted-input paths.
        let scales_name = quant_sibling(name, "scales")
            .with_context(|| format!("expected `.weight`-suffixed name, got {name}"))?;
        let biases_name = quant_sibling(name, "biases")
            .with_context(|| format!("expected `.weight`-suffixed name, got {name}"))?;
        let scales_info = self
            .hf
            .tensor_info(&scales_name)
            .with_context(|| format!("scales {scales_name} missing"))?;
        let biases_info = self.hf.tensor_info(&biases_name);
        // MLX scales/biases are F16 on older checkpoints (mlx-lm < ~0.20)
        // and BF16 on newer ones (Gemma 4 4-bit, recent Qwen3 MoE).
        // Reading BF16 bytes as F16 silently returns wildly wrong
        // exponents → corrupted dequant + degenerate decode (`<pad>`,
        // mojibake). Dispatch on the stored dtype.
        let scales_dtype = scales_info.dtype;
        let biases_dtype = biases_info.map(|t| t.dtype);

        if packed.shape.len() < 2 {
            bail!(
                "MLX packed tensor {:?} must be ≥2-D (got {:?})",
                name,
                packed.shape
            );
        }
        // Per-tensor quant override (Gemma 4 26B-A4B 4-bit MLX bumps
        // shared-FFN + router to 8-bit). Reading them with the global
        // 4-bit settings would unpack 8-bit data as nibbles and group
        // 64 elements with the wrong scale stride.
        let q = self.quant_for_tensor(name);
        let bits = q.bits as usize;
        let group_size = q.group_size as usize;
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            bail!(
                "MLX packed tensor {:?}: unsupported bits={} (MLX affine quant packs 2/3/4/5/6/8)",
                name,
                bits
            );
        }

        // Batch dims are everything except the last; last-2 dims are
        // [out_features, packed_in]. For 2-D tensors batch is empty.
        let (batch_dims, packed_in) = packed.shape.split_at(packed.shape.len() - 1);
        let packed_in = packed_in[0] as usize;
        let (batch_dims_split, out_dim_slice) = batch_dims.split_at(batch_dims.len() - 1);
        let out_features = out_dim_slice[0] as usize;
        if packed_in * 32 % bits != 0 {
            bail!(
                "MLX packed tensor {:?}: packed width {} u32 does not hold a whole number of {}-bit codes",
                name,
                packed_in,
                bits
            );
        }
        let in_features = packed_in * 32 / bits;
        let batch: usize = batch_dims_split.iter().product::<u64>() as usize;
        let batch = batch.max(1);

        if in_features % group_size != 0 {
            bail!(
                "MLX packed tensor {:?}: in_features {} not divisible by group_size {}",
                name,
                in_features,
                group_size
            );
        }

        let packed_bytes = self
            .hf
            .tensor_bytes(name)
            .with_context(|| format!("tensor_bytes({name}) missing after tensor_info()"))?;
        let scales_bytes = self
            .hf
            .tensor_bytes(&scales_name)
            .with_context(|| format!("tensor_bytes({scales_name}) missing after tensor_info()"))?;
        let biases_bytes = match biases_dtype {
            Some(_) => self.hf.tensor_bytes(&biases_name).with_context(|| {
                format!("tensor_bytes({biases_name}) missing after tensor_info()")
            })?,
            None => &[],
        };

        let mask = (1u32 << bits) - 1;
        let groups_per_row = in_features / group_size;

        let slice_packed_len = out_features * packed_in;
        let slice_scales_len = out_features * groups_per_row;
        let slice_out_len = out_features * in_features;

        let mut out = vec![0f32; batch * slice_out_len];
        for b in 0..batch {
            let p_base = b * slice_packed_len;
            let s_base = b * slice_scales_len;
            let o_base = b * slice_out_len;
            for i in 0..out_features {
                let row_bytes = (p_base + i * packed_in) * 4;
                let row_s = s_base + i * groups_per_row;
                for gj in 0..groups_per_row {
                    let scale = read_half(scales_bytes, row_s + gj, scales_dtype);
                    let bias = if !biases_bytes.is_empty() {
                        let dt = biases_dtype.unwrap_or(StDtype::F16);
                        read_half(biases_bytes, row_s + gj, dt)
                    } else {
                        0.0
                    };
                    for lj in 0..group_size {
                        let j = gj * group_size + lj;
                        // Code j occupies bits [j*bits, (j+1)*bits) of the
                        // row's little-endian bitstream. With bits ≤ 8 a
                        // code spans at most two bytes, and the second
                        // byte is only touched when the code actually
                        // crosses into it — never past the row's end.
                        let bit = j * bits;
                        let byte0 = row_bytes + bit / 8;
                        let sh = bit % 8;
                        let mut word = packed_bytes[byte0] as u32;
                        if sh + bits > 8 {
                            word |= (packed_bytes[byte0 + 1] as u32) << 8;
                        }
                        let q = (word >> sh) & mask;
                        out[o_base + i * in_features + j] = (q as f32) * scale + bias;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Raw packed payload of an MLX-quantized tensor, verbatim.
    ///
    /// MLX's `U32 [.., in/(32/bits)]` little-endian nibble packing is
    /// byte-identical to `base_q4`'s two-per-byte low-nibble-first
    /// stream (verified against `mx.dequantize` at 0.0 difference), so
    /// a passthrough conversion can reuse these bytes without the
    /// dequant→requant round trip that costs ~4.4% of weight RMS.
    ///
    /// Returns `Ok(None)` for unquantized tensors (no `.scales`
    /// sibling). Errors if the tensor's (bits, group_size) differ from
    /// `expect_bits`/`expect_group_size` — a passthrough caller must
    /// fail loudly rather than silently requantize, or the "weights
    /// are bit-identical to the source" contract breaks.
    pub fn tensor_packed(
        &self,
        name: &str,
        expect_bits: u32,
        expect_group_size: u32,
    ) -> Result<Option<MlxPackedTensor<'_>>> {
        let Some(scales_name) = quant_sibling(name, "scales") else {
            return Ok(None);
        };
        if self.hf.tensor_info(&scales_name).is_none() {
            return Ok(None);
        }
        let q = self.quant_for_tensor(name);
        if q.bits != expect_bits || q.group_size != expect_group_size {
            bail!(
                "MLX tensor {name:?} is {}-bit gs={} — passthrough expects {}-bit gs={}",
                q.bits,
                q.group_size,
                expect_bits,
                expect_group_size
            );
        }
        let scales_info = self
            .hf
            .tensor_info(&scales_name)
            .with_context(|| format!("scales {scales_name} missing"))?;
        let scale_dtype = scales_info.dtype;
        let biases_name = quant_sibling(name, "biases")
            .with_context(|| format!("expected `.weight`-suffixed name, got {name}"))?;
        let packed = self
            .hf
            .tensor_bytes(name)
            .with_context(|| format!("tensor_bytes({name}) missing"))?;
        let scales = self
            .hf
            .tensor_bytes(&scales_name)
            .with_context(|| format!("tensor_bytes({scales_name}) missing"))?;
        let biases = self
            .hf
            .tensor_bytes(&biases_name)
            .with_context(|| format!("MLX affine tensor {name:?} has scales but no biases"))?;
        Ok(Some(MlxPackedTensor {
            packed,
            scales,
            biases,
            group_size: q.group_size,
            bits: q.bits,
            scale_dtype,
        }))
    }

    /// Logical shape of an MLX-packed tensor (unpacking the last dim).
    /// Returns `Some(shape)` if the tensor is packed, `None` otherwise.
    /// Resolves bits per-tensor — Gemma 4 26B-A4B 4-bit checkpoints
    /// override `mlp.{gate,up,down}_proj` and `router.proj` to 8-bit, so
    /// using the global bits here unpacks 8-bit data as 4-bit and the
    /// resulting `last_dim * 32 / 4` (instead of `* 32 / 8`) doubles the
    /// logical in_features the runtime expects.
    pub fn unpacked_shape(&self, name: &str) -> Option<Vec<u64>> {
        let info = self.hf.tensor_info(name)?;
        quant_sibling(name, "scales").and_then(|sn| self.hf.tensor_info(&sn))?;
        if info.shape.len() < 2 {
            return None;
        }
        let q = self.quant_for_tensor(name);
        let bits = q.bits as u64;
        let mut shape = info.shape.clone();
        let last = shape.len() - 1;
        // packed_in = in_features * bits / 32, exactly — a non-exact
        // inverse means the tensor isn't MLX-packed with these settings.
        if bits == 0 || (shape[last] * 32) % bits != 0 {
            return None;
        }
        shape[last] = shape[last] * 32 / bits;
        Some(shape)
    }

    /// Zero-loss transplant of an MLX affine-quantized tensor into
    /// `base_q4`'s on-disk layout.
    ///
    /// `base_q4` and MLX-affine 4-bit are the *same* scheme: INT4
    /// asymmetric, `value = q * scale + bias`, one f16 scale and f16
    /// bias per group of 64. At 4 bits MLX's little-endian bitstream is
    /// byte-for-byte `base_q4`'s low-nibble-first packing
    /// (`byte = (q[2i+1] << 4) | q[2i]`), so the weight bytes transplant
    /// verbatim and only the scale/bias regions need re-laying-out.
    ///
    /// Taking this path instead of dequant → requant matters for more
    /// than speed: re-deriving `scale = (max - min) / 15` from already
    /// quantized values lands on a *different* grid whenever a group's
    /// codes don't span the full 0..15 range, so the round trip is not
    /// the identity. Transplanting reproduces the reference engine's
    /// weights exactly, which is what makes an MLX-vs-baseRT numerical
    /// comparison attributable to engine math.
    ///
    /// Returns `None` (rather than an error) whenever the tensor is not
    /// an exact match for the target scheme — different bits, a
    /// different group size, a symmetric tensor with no `.biases`, or a
    /// plain unquantized tensor. Callers fall back to dequant → requant.
    pub fn packed_base_q4(&self, name: &str, group_size: u32) -> Result<Option<MlxPackedQ4>> {
        let Some(scales_name) = quant_sibling(name, "scales") else {
            return Ok(None);
        };
        let Some(scales_info) = self.hf.tensor_info(&scales_name) else {
            return Ok(None); // not MLX-packed
        };
        let q = self.quant_for_tensor(name);
        if q.bits != 4 || q.group_size != group_size {
            return Ok(None); // 8-bit override, or a group size base_q4 can't express
        }
        let Some(biases_name) = quant_sibling(name, "biases") else {
            return Ok(None);
        };
        let Some(biases_info) = self.hf.tensor_info(&biases_name) else {
            return Ok(None); // symmetric tensor — base_q4 is asymmetric
        };
        let packed = self
            .hf
            .tensor_info(name)
            .with_context(|| format!("packed tensor {name} missing"))?;
        if packed.shape.len() < 2 {
            return Ok(None);
        }

        let group_size_usize = group_size as usize;
        let (batch_dims, packed_last) = packed.shape.split_at(packed.shape.len() - 1);
        let packed_in = packed_last[0] as usize;
        let (batch_dims_split, out_dim_slice) = batch_dims.split_at(batch_dims.len() - 1);
        let out_features = out_dim_slice[0] as usize;
        // 4-bit: 8 codes per u32.
        let in_features = packed_in * 8;
        if in_features % group_size_usize != 0 {
            return Ok(None);
        }
        let batch = (batch_dims_split.iter().product::<u64>() as usize).max(1);
        let total_values = batch * out_features * in_features;
        let n_groups = total_values / group_size_usize;

        let packed_bytes = self
            .hf
            .tensor_bytes(name)
            .with_context(|| format!("tensor_bytes({name}) missing after tensor_info()"))?;
        // The transplant is only sound if the source is exactly as large
        // as the layout implies — a short/long buffer means our shape
        // arithmetic disagrees with the file, and copying it verbatim
        // would silently produce a corrupt bundle.
        if packed_bytes.len() != total_values / 2 {
            bail!(
                "MLX packed tensor {:?}: {} weight bytes but shape {:?} implies {} \
                 (4-bit codes, 2 per byte)",
                name,
                packed_bytes.len(),
                packed.shape,
                total_values / 2
            );
        }
        let scales_bytes = self
            .hf
            .tensor_bytes(&scales_name)
            .with_context(|| format!("tensor_bytes({scales_name}) missing after tensor_info()"))?;
        let biases_bytes = self
            .hf
            .tensor_bytes(&biases_name)
            .with_context(|| format!("tensor_bytes({biases_name}) missing after tensor_info()"))?;
        if scales_bytes.len() != n_groups * 2 || biases_bytes.len() != n_groups * 2 {
            bail!(
                "MLX packed tensor {:?}: scales/biases are {}/{} bytes but shape {:?} implies \
                 {} groups of {} (2 bytes each)",
                name,
                scales_bytes.len(),
                biases_bytes.len(),
                packed.shape,
                n_groups,
                group_size
            );
        }

        // base_q4 stores f16 scales and biases. F16 sources copy
        // verbatim; bf16 sources (mlx-lm ≳ 0.20) are widened to f32 and
        // renarrowed — exact in the mantissa (bf16 has 8 bits, f16 has
        // 11) but bf16's wider exponent range can overflow to inf or
        // flush to zero, so report how many did.
        let (scales, scales_narrowed) = narrow_to_f16_le(scales_bytes, scales_info.dtype)?;
        let (biases, biases_narrowed) = narrow_to_f16_le(biases_bytes, biases_info.dtype)?;

        let out_of_f16_range = count_non_finite(&scales) + count_non_finite(&biases);

        Ok(Some(MlxPackedQ4 {
            packed_weights: packed_bytes.to_vec(),
            scales,
            biases,
            group_size,
            narrowed_from_bf16: scales_narrowed || biases_narrowed,
            out_of_f16_range,
        }))
    }
}

/// An MLX affine-q4 tensor in `base_q4`'s on-disk layout, ready to write
/// without a dequant → requant round trip. Field names mirror
/// `base_quant::Packed` so the caller's conversion is a plain move.
pub struct MlxPackedQ4 {
    /// 4-bit codes, two per byte, low nibble first — MLX's bytes verbatim.
    pub packed_weights: Vec<u8>,
    /// One f16 scale per group, little-endian.
    pub scales: Vec<u8>,
    /// One f16 bias per group, little-endian.
    pub biases: Vec<u8>,
    pub group_size: u32,
    /// Source scales/biases were bf16 and had to be renarrowed to f16.
    pub narrowed_from_bf16: bool,
    /// Scales/biases that left f16's representable range in the process.
    pub out_of_f16_range: usize,
}

/// Re-encode a buffer of f16-or-bf16 halves as little-endian f16.
/// Returns `(bytes, narrowed)` where `narrowed` marks a bf16 source.
/// Public so `--validate` can compare against the bytes the WRITE path
/// actually emitted: a bf16-scaled checkpoint is narrowed on the way in, so a
/// byte-compare against the raw source scales would fail on every tensor.
pub fn narrow_to_f16_le(bytes: &[u8], dtype: StDtype) -> Result<(Vec<u8>, bool)> {
    match dtype {
        StDtype::F16 => Ok((bytes.to_vec(), false)),
        StDtype::Bf16 => {
            let mut out = Vec::with_capacity(bytes.len());
            for c in bytes.chunks_exact(2) {
                let v = bf16::from_le_bytes([c[0], c[1]]).to_f32();
                out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
            Ok((out, true))
        }
        other => bail!("MLX scales/biases have unsupported dtype {other:?} (expected f16 or bf16)"),
    }
}

fn count_non_finite(f16_le: &[u8]) -> usize {
    f16_le
        .chunks_exact(2)
        .filter(|c| !f16::from_le_bytes([c[0], c[1]]).to_f32().is_finite())
        .count()
}

fn quant_sibling(name: &str, suffix: &str) -> Option<String> {
    let stem = name.strip_suffix(".weight")?;
    Some(format!("{stem}.{suffix}"))
}

/// Read one f16-or-bf16 value at `idx`. MLX scales/biases are stored
/// as f16 on older checkpoints (pre mlx-lm 0.20) and bf16 on newer
/// ones; mis-dispatching produces silently corrupted dequant.
fn read_half(bytes: &[u8], idx: usize, dtype: StDtype) -> f32 {
    let lo = bytes[idx * 2];
    let hi = bytes[idx * 2 + 1];
    match dtype {
        StDtype::Bf16 => bf16::from_le_bytes([lo, hi]).to_f32(),
        StDtype::F16 => f16::from_le_bytes([lo, hi]).to_f32(),
        // F32 / other: shouldn't happen for scales/biases, but degrade
        // gracefully by reading as f16 to match historical behavior.
        _ => f16::from_le_bytes([lo, hi]).to_f32(),
    }
}
