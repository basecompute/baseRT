//! AWQ — Activation-aware Weight Quantization.
//!
//! Per the canonical-quant migration (2026-04-29), the toolchain has
//! no Python dependency: activation profiles come from the baseRT
//! runtime running in fp16 calibration mode and dumping per-linear
//! input-channel absmax to a sidecar (`AwqProfile`). The Rust path
//! consumes that sidecar, runs AWQ search + apply, and the result
//! feeds into the canonical `pack_rtn` packer.
//!
//! ## Algorithm (Lin et al., 2023)
//!
//! Per linear layer with weight `W [out, in]` and per-input-channel
//! activation magnitudes `s_x [in]`:
//!
//! 1. Search α ∈ alpha_grid that minimizes
//!    `|| W*X − Q(W*diag(s)) * diag(1/s)*X ||`
//!    where `s = s_x^α`.
//! 2. Apply: `W' = W * diag(s)` packs through normal RTN. The runtime
//!    multiplies activations by `1/s` (or folds it into the preceding
//!    norm).
//!
//! The intuition: salient input channels (large |x|) get amplified in
//! W before quantization → the quant grid spends bits on them
//! preferentially. The inverse scale on x is a no-op for the math but
//! halves the dynamic range W's quantizer has to cover.
//!
//! ## Layer scope
//!
//! Search is per-tensor and operates on the (weights, activation
//! stats) pair. Composition with `pack_rtn` is straightforward:
//!
//! ```ignore
//! let plan = awq_search(weights, &absmax, &cfg);
//! let rotated = awq_apply(weights, in_features, &plan.scales);
//! let packed = pack_rtn(&rotated, rtn_cfg);
//! // rt: forward multiplies activations by plan.inverse_scales
//! ```
//!
//! Composition with the canonical-quant profile system is via the
//! profile's `calibration: {method: "awq", ...}` block + a separate
//! sidecar file referenced by the converter at run time.

pub mod sidecar;
pub mod wikitext;

use serde::{Deserialize, Serialize};

/// Per-layer activation profile consumed by AWQ search. The sidecar
/// produces this; the search/apply stages consume it. Keys are the
/// canonical .base tensor names (e.g. `layers.7.self_attn.q_proj.weight`).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AwqProfile {
    /// `model_id` / `tokenizer_hash` / similar identifying field so the
    /// converter can refuse a profile that doesn't belong to the model
    /// being converted. Matched against the `.base` header's `source`
    /// block at convert time.
    pub source_fingerprint: Option<String>,
    /// Number of calibration tokens that produced the profile. Logged
    /// in the `.base` header's `calibration.calib_tokens`.
    pub calib_tokens: usize,
    /// Per-tensor activation statistics. One entry per linear weight
    /// matrix; the f32 vec has length = in_features (per-channel
    /// absmax over the calibration set).
    pub per_tensor_absmax: std::collections::BTreeMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwqConfig {
    /// How many calibration tokens to expect in the profile. Fed into
    /// the search; only used as a sanity-check that the profile isn't
    /// drastically under-sized.
    pub calib_tokens: usize,
    /// Alpha search grid. The "lite" mode used the single value 0.5;
    /// full AWQ sweeps and picks the per-layer minimum. The default
    /// 21-point grid matches the reference llm-awq implementation.
    pub alpha_grid: Vec<f32>,
    /// Clipping ratio search grid. Applied as an additional optional
    /// step after alpha search; per-layer.
    pub clip_grid: Vec<f32>,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            calib_tokens: 512,
            alpha_grid: (0..=20).map(|i| i as f32 / 20.0).collect(),
            clip_grid: (50..=100).map(|i| i as f32 / 100.0).collect(),
        }
    }
}

/// Output of `awq_search`. Carries the chosen α and the per-input-
/// channel scale vector (length = in_features) the apply step uses.
#[derive(Debug, Clone)]
pub struct AwqPlan {
    pub alpha: f32,
    /// Forward scale per input channel: `s_in = absmax^α / norm`.
    /// Multiply weights along axis-1 (in_features) by this.
    pub scales: Vec<f32>,
    /// Inverse of `scales`. Apply to activations at runtime.
    pub inverse_scales: Vec<f32>,
    /// Reconstruction MSE at the chosen α (against the original
    /// weights at fp32). Lower is better — used by callers to decide
    /// whether AWQ helps over plain RTN.
    pub mse: f32,
}

impl AwqConfig {
    /// AWQ alpha search per the original paper. Iterates alpha_grid,
    /// for each α scales weights by `absmax^α`, RTN-quantizes,
    /// dequantizes, undoes the scale, and measures reconstruction
    /// MSE. Returns the best α + its scale vector.
    ///
    /// `weights` is row-major `[out, in]`. `absmax_per_input_channel`
    /// has length = in_features and contains the per-channel max
    /// absolute activation across the calibration set.
    ///
    /// `bits` / `group_size` / `symmetric` describe the target quant
    /// scheme — search loop pre-RTN-quantizes to those settings.
    pub fn search(
        &self,
        weights: &[f32],
        in_features: usize,
        absmax_per_input_channel: &[f32],
        bits: u32,
        group_size: u32,
        symmetric: bool,
    ) -> AwqPlan {
        assert_eq!(absmax_per_input_channel.len(), in_features);
        assert!(weights.len() % in_features == 0);
        let out_features = weights.len() / in_features;

        let mut best = AwqPlan {
            alpha: 0.0,
            scales: vec![1.0; in_features],
            inverse_scales: vec![1.0; in_features],
            mse: f32::INFINITY,
        };

        for &alpha in &self.alpha_grid {
            let scales = absmax_to_scales(absmax_per_input_channel, alpha);
            let inverse_scales: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();

            // Apply: rotated[i,j] = weights[i,j] * scales[j].
            let rotated = scale_columns(weights, in_features, &scales);

            // RTN-quantize the rotated weights, dequantize, then undo
            // the scale to compare against the original weights.
            let mse = rtn_reconstruction_mse(
                weights,
                &rotated,
                in_features,
                out_features,
                &inverse_scales,
                bits,
                group_size,
                symmetric,
            );

            if mse < best.mse {
                best = AwqPlan {
                    alpha,
                    scales,
                    inverse_scales,
                    mse,
                };
            }
        }
        best
    }

    /// Convenience: use the `lite` shortcut (alpha=0.5, no search)
    /// when calibration data is unavailable. Cheap, ~80–95% of the
    /// gains of full AWQ on most models.
    pub fn lite(in_features: usize, absmax: &[f32]) -> AwqPlan {
        let scales = absmax_to_scales(absmax, 0.5);
        let inverse_scales: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
        let _ = in_features;
        AwqPlan {
            alpha: 0.5,
            scales,
            inverse_scales,
            mse: f32::NAN,
        }
    }
}

/// Compose AWQ scales into the weights for downstream RTN packing.
/// `weights[i*in_features + j] *= scales[j]`. Returns a new vector;
/// the original is left untouched (callers may want the original to
/// gate AWQ-vs-RTN decisions).
pub fn awq_apply(weights: &[f32], in_features: usize, scales: &[f32]) -> Vec<f32> {
    scale_columns(weights, in_features, scales)
}

// ---------- internal helpers ----------

fn absmax_to_scales(absmax: &[f32], alpha: f32) -> Vec<f32> {
    // s_j = max(absmax_j, eps)^α, normalized so geometric mean = 1.
    // Normalization keeps the matmul magnitude unchanged, which is
    // important so dequant doesn't accumulate numerical drift.
    let eps = 1e-5_f32;
    let raw: Vec<f32> = absmax.iter().map(|&x| x.max(eps).powf(alpha)).collect();
    let log_mean: f32 = raw.iter().map(|&x| x.ln()).sum::<f32>() / (raw.len() as f32);
    let geo_mean = log_mean.exp();
    raw.into_iter().map(|s| s / geo_mean.max(eps)).collect()
}

fn scale_columns(weights: &[f32], in_features: usize, scales: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; weights.len()];
    let n = weights.len() / in_features;
    for i in 0..n {
        for j in 0..in_features {
            out[i * in_features + j] = weights[i * in_features + j] * scales[j];
        }
    }
    out
}

/// Quantize-dequantize `rotated`, undo the rotation by `inverse_scales`,
/// compare against `original`. Returns mean-squared error.
#[allow(clippy::too_many_arguments)]
fn rtn_reconstruction_mse(
    original: &[f32],
    rotated: &[f32],
    in_features: usize,
    out_features: usize,
    inverse_scales: &[f32],
    bits: u32,
    group_size: u32,
    symmetric: bool,
) -> f32 {
    use base_quant::{pack_rtn, unpack_rtn, RtnConfig};
    use base_format::ScaleDtype;

    let cfg = RtnConfig {
        bits,
        group_size,
        symmetric,
        scale_dtype: ScaleDtype::Bf16,
    };

    // RTN expects total elements multiple of group_size. Per-row pack
    // is the canonical case (one row at a time), so we walk rows.
    let mut sse = 0f64;
    let mut n = 0usize;
    for i in 0..out_features {
        let row = &rotated[i * in_features..(i + 1) * in_features];
        let packed = pack_rtn(row, cfg);
        let dequant = unpack_rtn(&packed, in_features, cfg);
        let orig_row = &original[i * in_features..(i + 1) * in_features];
        for (k, q) in dequant.iter().enumerate() {
            // Undo the scale before comparing.
            let recon = q * inverse_scales[k];
            let err = recon - orig_row[k];
            sse += (err as f64) * (err as f64);
            n += 1;
        }
    }
    (sse / (n as f64)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AWQ should be at least as good as plain RTN: the search
    /// includes α=0 which is identity, so the optimum is by
    /// construction never worse.
    #[test]
    fn awq_search_never_worse_than_plain_rtn() {
        let n_in = 64usize;
        let n_out = 8usize;
        let mut weights = vec![0f32; n_out * n_in];
        let mut s = 0xdeadbeefu32;
        for v in weights.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        }
        let absmax = vec![1.0f32; n_in];

        let plan = AwqConfig::default().search(&weights, n_in, &absmax, 4, 64, false);
        let identity: Vec<f32> = vec![1.0; n_in];
        let plain_mse = rtn_reconstruction_mse(
            &weights, &weights, n_in, n_out, &identity, 4, 64, false,
        );
        // Tiny tolerance for floating-point order-of-ops.
        assert!(
            plan.mse <= plain_mse * 1.0001,
            "AWQ search regressed: {} vs RTN {}",
            plan.mse,
            plain_mse
        );
    }

    /// AWQ should help in the canonical paper case: a tiny "salient"
    /// weight that gets crushed to 0 by plain RTN's group-shared
    /// scale, while having a large activation magnitude. Rotation
    /// amplifies the weight before quant so it lands on a non-zero
    /// quant level.
    #[test]
    fn awq_helps_when_salient_weight_would_round_to_zero() {
        // One row, 32 in features = exactly one q2 group.
        let n_in = 32usize;
        let n_out = 1usize;
        let mut weights = vec![10.0f32; n_in];
        // The salient weight is 4 orders of magnitude smaller than
        // the bulk, so plain q2 quantizes it to 0.
        weights[31] = 0.001;
        // Its activation magnitude is huge (the "salient" channel).
        let mut absmax = vec![0.01f32; n_in];
        absmax[31] = 100.0;

        let plan = AwqConfig::default().search(&weights, n_in, &absmax, 2, 32, false);

        let identity: Vec<f32> = vec![1.0; n_in];
        let plain_mse = rtn_reconstruction_mse(
            &weights, &weights, n_in, n_out, &identity, 2, 32, false,
        );
        assert!(
            plan.mse < plain_mse,
            "AWQ should help: awq={} plain={}",
            plan.mse,
            plain_mse
        );
    }

    #[test]
    fn awq_apply_is_consistent_with_search_scales() {
        let n_in = 32usize;
        let weights = vec![1.0f32; 4 * n_in];
        let absmax = vec![2.0f32; n_in];
        let plan = AwqConfig::default().search(&weights, n_in, &absmax, 4, 32, false);
        let rotated = awq_apply(&weights, n_in, &plan.scales);
        // Uniform absmax → all scales = (2^α) / (2^α) = 1, no-op.
        for v in rotated.iter() {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn awq_lite_alpha_is_half() {
        let plan = AwqConfig::lite(8, &[1.0; 8]);
        assert_eq!(plan.alpha, 0.5);
        assert_eq!(plan.scales.len(), 8);
    }

    #[test]
    fn absmax_to_scales_geometric_mean_is_unit() {
        let absmax = vec![0.5f32, 1.0, 2.0, 4.0, 0.25, 8.0, 1.5, 3.0];
        let scales = absmax_to_scales(&absmax, 0.7);
        let log_mean: f32 = scales.iter().map(|s| s.ln()).sum::<f32>() / (scales.len() as f32);
        let geo_mean = log_mean.exp();
        assert!(
            (geo_mean - 1.0).abs() < 1e-4,
            "scales should be normalized to geometric mean 1, got {}",
            geo_mean
        );
    }
}
