//! MLX quantized safetensors reader.
//!
//! MLX stores quantized weights in HF-safetensors directories with a
//! specific convention:
//!
//! - `config.json` has a `"quantization": {"bits": 4, "group_size": 64}`
//!   key declaring the scheme.
//! - A quantized weight `foo.weight` is stored as a `U32` tensor with
//!   shape `[out_features, in_features / (32 / bits)]` — packed nibbles
//!   (or bytes for 8-bit).
//! - Two additional tensors accompany it: `foo.scales` and `foo.biases`,
//!   both `F16`, shape `[out_features, in_features / group_size]`.
//! - Dequant: `x[i, j] = q[i, j] * scale[i, j / group_size]
//!                       + bias[i, j / group_size]`
//!   where `q[i, j]` is extracted from `packed[i, j / (32/bits)]` at
//!   nibble position `j % (32/bits)` (low-nibble first).
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
        let vals_per_u32 = 32 / bits;

        // Batch dims are everything except the last; last-2 dims are
        // [out_features, packed_in]. For 2-D tensors batch is empty.
        let (batch_dims, packed_in) = packed.shape.split_at(packed.shape.len() - 1);
        let packed_in = packed_in[0] as usize;
        let (batch_dims_split, out_dim_slice) = batch_dims.split_at(batch_dims.len() - 1);
        let out_features = out_dim_slice[0] as usize;
        let in_features = packed_in * vals_per_u32;
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
            Some(_) => self
                .hf
                .tensor_bytes(&biases_name)
                .with_context(|| format!("tensor_bytes({biases_name}) missing after tensor_info()"))?,
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
                let row_p = p_base + i * packed_in;
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
                        let u32_idx = row_p + j / vals_per_u32;
                        let u = u32::from_le_bytes(
                            packed_bytes[u32_idx * 4..u32_idx * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        let slot = j % vals_per_u32;
                        let q = (u >> (bits * slot)) & mask;
                        out[o_base + i * in_features + j] =
                            (q as f32) * scale + bias;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Logical shape of an MLX-packed tensor (unpacking the last dim).
    /// Returns `Some(shape)` if the tensor is packed, `None` otherwise.
    /// Resolves bits per-tensor — Gemma 4 26B-A4B 4-bit checkpoints
    /// override `mlp.{gate,up,down}_proj` and `router.proj` to 8-bit, so
    /// using the global bits here unpacks 8-bit data as 4-bit and the
    /// resulting `last_dim *= 8` (instead of *= 4) doubles the logical
    /// in_features the runtime expects.
    pub fn unpacked_shape(&self, name: &str) -> Option<Vec<u64>> {
        let info = self.hf.tensor_info(name)?;
        quant_sibling(name, "scales")
            .and_then(|sn| self.hf.tensor_info(&sn))?;
        if info.shape.len() < 2 {
            return None;
        }
        let q = self.quant_for_tensor(name);
        let vals_per_u32 = 32 / q.bits as u64;
        let mut shape = info.shape.clone();
        let last = shape.len() - 1;
        shape[last] *= vals_per_u32;
        Some(shape)
    }

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
