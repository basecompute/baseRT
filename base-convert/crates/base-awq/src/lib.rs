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

use base_format::ScaleDtype;
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
    /// Optional per-channel mean of `x_j²` over the calibration
    /// positions (same keys and lengths as `per_tensor_absmax`). When
    /// present it is the error weight of the alpha search (the faithful
    /// diagonal proxy of the AWQ output-error objective); otherwise the
    /// search falls back to `absmax²`.
    #[serde(default)]
    pub per_tensor_meansq: std::collections::BTreeMap<String, Vec<f32>>,
    /// Fused 3-D MoE expert tensors (`[E, N, K]`): per-expert
    /// per-input-channel absmax, `[E][K]`, measured over the (token,
    /// slot) rows routed to each expert. The pooled `per_tensor_*`
    /// entries of the same name (max / row-weighted mean over experts)
    /// stay the shared-scale statistic; these drive per-expert folds
    /// (`moe_down`). Experts that saw no rows hold zeros.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub per_expert_absmax: std::collections::BTreeMap<String, Vec<Vec<f32>>>,
    /// Per-expert `E[x_j²]`, `[E][K]` (same keys as `per_expert_absmax`).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub per_expert_meansq: std::collections::BTreeMap<String, Vec<Vec<f32>>>,
    /// Routed (token, slot) rows each expert saw, `[E]`.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub per_expert_rows: std::collections::BTreeMap<String, Vec<u64>>,
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
    /// Rows of each member the alpha search scores: 0 = every row; N =
    /// the evenly strided subset `rows[::out/N]` (the stride autoquant's
    /// proxy search uses). The objective is a mean over rows, so the
    /// subset is an unbiased estimate; it trades search fidelity for
    /// time on very tall matrices (a 248k-row lm_head). Only the search
    /// samples — the fold and the pack always touch every row.
    #[serde(default)]
    pub search_rows: usize,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            calib_tokens: 512,
            alpha_grid: (0..=20).map(|i| i as f32 / 20.0).collect(),
            clip_grid: (50..=100).map(|i| i as f32 / 100.0).collect(),
            search_rows: 0,
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
    /// dequantizes, undoes the scale, and measures the
    /// ACTIVATION-WEIGHTED reconstruction error — each channel's error
    /// counts in proportion to its calibration absmax, i.e.
    /// ‖(Q(W·s)·s⁻¹ − W)·diag(a)‖². Returns the best α + its scales.
    ///
    /// The weighting is the point of AWQ: with plain (unweighted) MSE
    /// the α=0 identity always wins, because RTN already minimizes
    /// per-group weight-space MSE — which is exactly the bug this
    /// implementation shipped with (α=0.00 on every real tensor).
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
        let col_weight = activation_weights(absmax_per_input_channel);
        self.search_weighted(
            weights,
            in_features,
            absmax_per_input_channel,
            &col_weight,
            bits,
            group_size,
            symmetric,
        )
    }

    /// `search` with an explicit per-input-channel error weight (see
    /// `activation_weights_from`: `E[x_j²]` when the sidecar carries it,
    /// `absmax_j²` otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn search_weighted(
        &self,
        weights: &[f32],
        in_features: usize,
        absmax_per_input_channel: &[f32],
        col_weight: &[f32],
        bits: u32,
        group_size: u32,
        symmetric: bool,
    ) -> AwqPlan {
        assert!(weights.len() % in_features == 0);
        // The single-matrix convenience prices the cell with bf16 scales;
        // the fold plan goes through `search_group_*` with each member's
        // resolved scale dtype.
        let member = [GroupMember {
            weights: MemberData::F32(weights),
            in_features,
            bits,
            group_size,
            symmetric,
            scale_dtype: ScaleDtype::Bf16,
        }];
        self.search_group_weighted_curve(&member, absmax_per_input_channel, col_weight)
            .0
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

/// A member's weights as the search reads them: one row at a time,
/// converted to f32 on the fly. The converter hands the search the
/// source's own bf16/f16 bytes (a slice of the shard's mmap), so a
/// 27B lm_head is searched without an f32 copy of the matrix — and
/// without the per-alpha whole-matrix copies the old path made (18
/// threads × 4.7 GB on a 248320×5120 head took a 128 GB box down).
#[derive(Debug, Clone, Copy)]
pub enum MemberData<'a> {
    F32(&'a [f32]),
    /// Little-endian IEEE f16 bytes, row-major `[out, in]`.
    F16(&'a [u8]),
    /// Little-endian bf16 bytes, row-major `[out, in]`.
    Bf16(&'a [u8]),
}

impl MemberData<'_> {
    /// Number of weights.
    pub fn len(&self) -> usize {
        match self {
            MemberData::F32(w) => w.len(),
            MemberData::F16(b) | MemberData::Bf16(b) => b.len() / 2,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i` of the `[out, in_features]` matrix, as f32, into `out`.
    pub fn row_into(&self, i: usize, in_features: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), in_features);
        let (a, b) = (i * in_features, (i + 1) * in_features);
        match self {
            MemberData::F32(w) => out.copy_from_slice(&w[a..b]),
            MemberData::F16(bytes) => {
                for (o, c) in out.iter_mut().zip(bytes[2 * a..2 * b].chunks_exact(2)) {
                    *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            MemberData::Bf16(bytes) => {
                for (o, c) in out.iter_mut().zip(bytes[2 * a..2 * b].chunks_exact(2)) {
                    *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
        }
    }
}

impl<'a> MemberData<'a> {
    /// Elements `[start, start + len)` (in weights, not bytes) — one
    /// expert's `[N, K]` slab of a fused `[E, N, K]` expert tensor.
    pub fn slice(&self, start: usize, len: usize) -> MemberData<'a> {
        match *self {
            MemberData::F32(w) => MemberData::F32(&w[start..start + len]),
            MemberData::F16(b) => MemberData::F16(&b[2 * start..2 * (start + len)]),
            MemberData::Bf16(b) => MemberData::Bf16(&b[2 * start..2 * (start + len)]),
        }
    }
}

impl<'a> From<&'a [f32]> for MemberData<'a> {
    fn from(w: &'a [f32]) -> Self {
        MemberData::F32(w)
    }
}

impl<'a> From<&'a Vec<f32>> for MemberData<'a> {
    fn from(w: &'a Vec<f32>) -> Self {
        MemberData::F32(w.as_slice())
    }
}

/// One member of a fold group for `AwqConfig::search_group`: a linear
/// weight matrix `[out, in]` (row-major) plus the quant cell its
/// profile rule assigns. All members of a group share `in_features`
/// (they consume the same activation vector).
#[derive(Debug, Clone, Copy)]
pub struct GroupMember<'a> {
    pub weights: MemberData<'a>,
    pub in_features: usize,
    pub bits: u32,
    pub group_size: u32,
    pub symmetric: bool,
    /// The dtype the producer packs this member's group scales in. The
    /// search quantizes on the same grid, so the curve `awq-search`
    /// reports is the error the pack realizes (f16 and bf16 scales round
    /// differently).
    pub scale_dtype: ScaleDtype,
}

impl AwqConfig {
    /// AWQ alpha search for a FOLD GROUP: several linears that consume
    /// the same input (q/k/v after one norm, gate/up after one norm,
    /// qkv/z/a/b in a linear-attention block). One scale vector `s`
    /// must serve all of them because the inverse is folded into the
    /// single upstream tensor, so the search minimizes the SUMMED
    /// reconstruction SSE across members (each quantized at its own
    /// cell) over the alpha grid. α=0 is in the default grid, so the
    /// result is never worse than plain RTN on every member combined.
    pub fn search_group(&self, members: &[GroupMember], absmax: &[f32]) -> AwqPlan {
        let col_weight = activation_weights(absmax);
        self.search_group_weighted(members, absmax, &col_weight)
    }

    /// `search_group` with an explicit per-input-channel error weight.
    pub fn search_group_weighted(
        &self,
        members: &[GroupMember],
        absmax: &[f32],
        col_weight: &[f32],
    ) -> AwqPlan {
        self.search_group_weighted_curve(members, absmax, col_weight)
            .0
    }

    /// `search_group_weighted`, also returning the whole `(alpha,
    /// weighted mse)` curve in grid order — what `basert awq-search`
    /// reports so a prober can see how flat the objective is around
    /// the chosen alpha.
    ///
    /// Memory: the search streams. Each rayon task owns two
    /// `in_features`-long f32 buffers (the row and its rotation) and one
    /// f64 accumulator per grid point; it reads a row of the member (bf16
    /// → f32 on the fly), scores it at every alpha, and moves on. Nothing
    /// the size of a member is ever allocated, whatever the thread count.
    /// Work is the same as the old alpha-parallel loop (every row packed
    /// and unpacked once per alpha), spread over rows instead of the 21
    /// grid points — which also keeps all cores busy on a wide box.
    pub fn search_group_weighted_curve(
        &self,
        members: &[GroupMember],
        absmax: &[f32],
        col_weight: &[f32],
    ) -> (AwqPlan, Vec<(f32, f32)>) {
        use base_quant::{pack_rtn, unpack_rtn, RtnConfig};
        use rayon::prelude::*;

        assert!(
            !members.is_empty(),
            "search_group needs at least one member"
        );
        let in_features = members[0].in_features;
        assert_eq!(absmax.len(), in_features);
        assert_eq!(col_weight.len(), in_features);
        for m in members {
            assert_eq!(
                m.in_features, in_features,
                "fold group members must share in_features"
            );
            assert!(m.weights.len() % in_features == 0);
        }
        let total_n: f64 = members.iter().map(|m| m.weights.len() as f64).sum();

        // (alpha, s, 1/s) per grid point, shared read-only by every task
        let grid: Vec<(f32, Vec<f32>, Vec<f32>)> = self
            .alpha_grid
            .iter()
            .map(|&alpha| {
                let scales = absmax_to_scales(absmax, alpha);
                let inverse: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
                (alpha, scales, inverse)
            })
            .collect();
        let n_alpha = grid.len();

        // Objective per alpha: Σ_members mean-over-rows(weighted SSE) × member
        // size, / total size — the element-weighted mean over the group.
        // With `search_rows` the per-member mean is over the strided row
        // subset (still weighted by the member's full size).
        let mut sse_per_alpha = vec![0f64; n_alpha];
        for m in members {
            let out_features = m.weights.len() / in_features;
            let step = if self.search_rows > 0 && out_features > self.search_rows {
                out_features / self.search_rows
            } else {
                1
            };
            let rows: Vec<usize> = (0..out_features).step_by(step).collect();
            let cfg = RtnConfig {
                bits: m.bits,
                group_size: m.group_size,
                symmetric: m.symmetric,
                scale_dtype: m.scale_dtype,
            };
            let weights = m.weights;
            let acc: Vec<f64> = rows
                .par_iter()
                .fold(
                    || {
                        (
                            vec![0f32; in_features],
                            vec![0f32; in_features],
                            vec![0f64; n_alpha],
                        )
                    },
                    |(mut row, mut rot, mut acc), &i| {
                        weights.row_into(i, in_features, &mut row);
                        for (k, (_, scales, inverse)) in grid.iter().enumerate() {
                            for j in 0..in_features {
                                rot[j] = row[j] * scales[j];
                            }
                            let packed = pack_rtn(&rot, cfg);
                            let dequant = unpack_rtn(&packed, in_features, cfg);
                            let mut e = 0f64;
                            for j in 0..in_features {
                                // undo the scale before comparing
                                let err = dequant[j] * inverse[j] - row[j];
                                e += (err as f64) * (err as f64) * col_weight[j] as f64;
                            }
                            acc[k] += e;
                        }
                        (row, rot, acc)
                    },
                )
                .map(|(_, _, acc)| acc)
                .reduce(
                    || vec![0f64; n_alpha],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                        a
                    },
                );
            let seen = (rows.len() * in_features) as f64;
            let size = weights.len() as f64;
            for (total, part) in sse_per_alpha.iter_mut().zip(acc) {
                *total += part / seen * size;
            }
        }

        let evaluated: Vec<(f32, f32)> = grid
            .iter()
            .zip(&sse_per_alpha)
            .map(|((alpha, _, _), &sse)| (*alpha, (sse / total_n) as f32))
            .collect();
        let mut best = AwqPlan {
            alpha: 0.0,
            scales: vec![1.0; in_features],
            inverse_scales: vec![1.0; in_features],
            mse: f32::INFINITY,
        };
        for (k, &(alpha, mse)) in evaluated.iter().enumerate() {
            // strict `<` keeps the first (smallest) alpha on ties, α=0 first
            if mse < best.mse {
                best = AwqPlan {
                    alpha,
                    scales: grid[k].1.clone(),
                    inverse_scales: grid[k].2.clone(),
                    mse,
                };
            }
        }
        (best, evaluated)
    }
}

impl AwqConfig {
    /// Build the plan for an EXPLICIT alpha (no search): the same
    /// `s = absmax^alpha / geomean` normalisation the search uses, so a
    /// probe that chose alpha by measured ΔKL reproduces the converter's
    /// scales exactly. `mse` is left NaN (nothing was evaluated).
    pub fn plan_for_alpha(absmax: &[f32], alpha: f32) -> AwqPlan {
        assert!((0.0..=1.0).contains(&alpha), "alpha {alpha} out of 0..=1");
        let scales = absmax_to_scales(absmax, alpha);
        let inverse_scales: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
        AwqPlan {
            alpha,
            scales,
            inverse_scales,
            mse: f32::NAN,
        }
    }
}

// ---------- fold helpers (exact in FP; the runtime applies nothing) ----------

/// `w[i, j] *= s[j]` for a row-major `[out, in]` matrix. This is the
/// AWQ rotation on a consumer of the scaled activation.
pub fn scale_columns_in_place(weights: &mut [f32], in_features: usize, s: &[f32]) {
    assert_eq!(s.len(), in_features);
    assert!(weights.len() % in_features == 0);
    for row in weights.chunks_mut(in_features) {
        for (v, &sj) in row.iter_mut().zip(s) {
            *v *= sj;
        }
    }
}

/// `w[i, j] *= r[i]` for a row-major `[out, in]` matrix. Used to push
/// `1/s` of a down_proj fold into up_proj's output rows: down's input
/// is `silu(gate·x) ⊙ (up·x)`, so scaling up's row i by 1/s_i scales
/// down's input channel i by exactly 1/s_i. Row scaling never changes
/// per-group affine codes along in_features (only scale/bias move).
pub fn scale_rows_in_place(weights: &mut [f32], in_features: usize, r: &[f32]) {
    assert!(weights.len() % in_features == 0);
    assert_eq!(r.len(), weights.len() / in_features);
    for (row, &ri) in weights.chunks_mut(in_features).zip(r) {
        for v in row.iter_mut() {
            *v *= ri;
        }
    }
}

/// `gamma[j] *= inv[j]` — fold the inverse AWQ scale into an RMSNorm
/// gain: `rmsnorm(x) * (gamma / s)` equals `(rmsnorm(x) * gamma) / s`
/// elementwise, so every consumer sees `x / s` exactly.
pub fn scale_vec_in_place(gamma: &mut [f32], inv: &[f32]) {
    assert_eq!(gamma.len(), inv.len());
    for (g, &i) in gamma.iter_mut().zip(inv) {
        *g *= i;
    }
}

/// Merge the sidecar absmax vectors of tensors that share one input.
/// They should be identical (same activation); returns the elementwise
/// max and whether every vector agreed with it within `rel_tol`.
pub fn merge_absmax(vectors: &[&[f32]], rel_tol: f32) -> (Vec<f32>, bool) {
    assert!(!vectors.is_empty());
    let n = vectors[0].len();
    let mut out = vec![0f32; n];
    for v in vectors {
        assert_eq!(
            v.len(),
            n,
            "absmax vectors of a fold group must have equal length"
        );
        for (o, &x) in out.iter_mut().zip(v.iter()) {
            if x > *o {
                *o = x;
            }
        }
    }
    let mut agree = true;
    for v in vectors {
        for (&o, &x) in out.iter().zip(v.iter()) {
            if (o - x).abs() > rel_tol * o.abs().max(1e-6) {
                agree = false;
            }
        }
    }
    (out, agree)
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

/// Per-input-channel error weights for the search objective.
///
/// AWQ minimizes the OUTPUT error `‖X Wᵀ − X Ŵᵀ‖²`, not the weight
/// error. With only per-channel activation absmax available (the
/// sidecar), the diagonal proxy is `Σ_j absmax_j² · Σ_i (W_ij − Ŵ_ij)²`:
/// a weight error on a channel that carries large activations costs
/// proportionally more. Plain (unweighted) weight MSE would make α=0
/// win on every unstructured tensor and turn AWQ into a no-op.
/// Normalized to mean 1 so the returned MSE stays comparable to the
/// unweighted number.
pub fn activation_weights(absmax: &[f32]) -> Vec<f32> {
    let eps = 1e-5_f32;
    let raw: Vec<f32> = absmax.iter().map(|&a| a.max(eps) * a.max(eps)).collect();
    normalize_mean_one(raw)
}

/// Error weights from the best available statistic: `E[x_j²]` (the
/// diagonal of the activation second moment, the faithful proxy of the
/// AWQ output-error objective) when the sidecar carries it, else
/// `absmax_j²`. Normalized to mean 1.
pub fn activation_weights_from(meansq: Option<&[f32]>, absmax: &[f32]) -> Vec<f32> {
    match meansq {
        Some(m) if m.len() == absmax.len() => {
            let eps = 1e-10_f32;
            normalize_mean_one(m.iter().map(|&v| v.max(eps)).collect())
        }
        _ => activation_weights(absmax),
    }
}

fn normalize_mean_one(raw: Vec<f32>) -> Vec<f32> {
    let eps = 1e-10_f32;
    let mean = raw.iter().sum::<f32>() / raw.len().max(1) as f32;
    raw.into_iter().map(|w| w / mean.max(eps)).collect()
}

/// Quantize-dequantize `rotated`, undo the rotation by `inverse_scales`,
/// compare against `original` with per-column weights `col_weight`
/// (use all-ones for the plain weight MSE). Returns the weighted
/// mean-squared error.
#[allow(clippy::too_many_arguments)]
pub fn rtn_reconstruction_mse(
    original: &[f32],
    rotated: &[f32],
    in_features: usize,
    out_features: usize,
    inverse_scales: &[f32],
    col_weight: &[f32],
    bits: u32,
    group_size: u32,
    symmetric: bool,
    scale_dtype: ScaleDtype,
) -> f32 {
    use base_quant::{pack_rtn, unpack_rtn, RtnConfig};

    let cfg = RtnConfig {
        bits,
        group_size,
        symmetric,
        scale_dtype,
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
            // Undo the scale before comparing, then weight the error by the
            // channel's activation magnitude — a weight error on a channel
            // that never fires costs (almost) nothing; one on a hot channel
            // is what actually perturbs the layer output.
            let recon = q * inverse_scales[k];
            let err = recon - orig_row[k];
            sse += (err as f64) * (err as f64) * col_weight[k] as f64;
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
        let cw = activation_weights(&absmax);
        let plain_mse = rtn_reconstruction_mse(
            &weights,
            &weights,
            n_in,
            n_out,
            &identity,
            &cw,
            4,
            64,
            false,
            ScaleDtype::Bf16,
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
        let cw = activation_weights(&absmax);
        let plain_mse = rtn_reconstruction_mse(
            &weights,
            &weights,
            n_in,
            n_out,
            &identity,
            &cw,
            2,
            32,
            false,
            ScaleDtype::Bf16,
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

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter().zip(gamma).map(|(v, g)| v * inv * g).collect()
    }

    fn matvec(w: &[f32], in_features: usize, x: &[f32]) -> Vec<f32> {
        w.chunks(in_features)
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(1e-3))
            .fold(0.0, f32::max)
    }

    /// Norm-fed fold group: rmsnorm -> {q, k} sharing one gamma. After
    /// folding s into the linears' columns and 1/s into gamma, both
    /// outputs are unchanged in FP.
    #[test]
    fn norm_group_fold_is_exact() {
        let n_in = 64usize;
        let mut seed = 7u32;
        let gamma: Vec<f32> = (0..n_in).map(|_| 1.0 + 0.3 * lcg(&mut seed)).collect();
        let q: Vec<f32> = (0..16 * n_in).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..8 * n_in).map(|_| lcg(&mut seed)).collect();
        let absmax: Vec<f32> = (0..n_in)
            .map(|_| 0.05 + lcg(&mut seed).abs() * 8.0)
            .collect();
        let x: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed) * 3.0).collect();

        let members = [
            GroupMember {
                weights: MemberData::F32(&q),
                in_features: n_in,
                bits: 4,
                group_size: 64,
                symmetric: false,
                scale_dtype: ScaleDtype::Bf16,
            },
            GroupMember {
                weights: MemberData::F32(&k),
                in_features: n_in,
                bits: 5,
                group_size: 32,
                symmetric: false,
                scale_dtype: ScaleDtype::Bf16,
            },
        ];
        let plan = AwqConfig::default().search_group(&members, &absmax);
        assert!(
            plan.alpha > 0.0,
            "salient absmax spread should pick a nonzero alpha"
        );

        let y_ref = rmsnorm(&x, &gamma, 1e-6);
        let q_ref = matvec(&q, n_in, &y_ref);
        let k_ref = matvec(&k, n_in, &y_ref);

        let (mut g2, mut q2, mut k2) = (gamma.clone(), q.clone(), k.clone());
        scale_vec_in_place(&mut g2, &plan.inverse_scales);
        scale_columns_in_place(&mut q2, n_in, &plan.scales);
        scale_columns_in_place(&mut k2, n_in, &plan.scales);
        let y2 = rmsnorm(&x, &g2, 1e-6);
        let q_f = matvec(&q2, n_in, &y2);
        let k_f = matvec(&k2, n_in, &y2);
        assert!(
            max_rel_err(&q_ref, &q_f) < 1e-4,
            "q fold err {}",
            max_rel_err(&q_ref, &q_f)
        );
        assert!(
            max_rel_err(&k_ref, &k_f) < 1e-4,
            "k fold err {}",
            max_rel_err(&k_ref, &k_f)
        );
    }

    /// down_proj fold: down's columns *= s, up's rows *= 1/s; the SwiGLU
    /// product silu(gate x) * (up x) -> down is unchanged in FP.
    #[test]
    fn down_up_fold_is_exact() {
        let hidden = 32usize;
        let inter = 64usize;
        let mut seed = 11u32;
        let gate: Vec<f32> = (0..inter * hidden).map(|_| lcg(&mut seed)).collect();
        let up: Vec<f32> = (0..inter * hidden).map(|_| lcg(&mut seed)).collect();
        let down: Vec<f32> = (0..hidden * inter).map(|_| lcg(&mut seed)).collect();
        let absmax: Vec<f32> = (0..inter)
            .map(|_| 0.02 + lcg(&mut seed).abs() * 5.0)
            .collect();
        let x: Vec<f32> = (0..hidden).map(|_| lcg(&mut seed)).collect();

        let silu = |v: f32| v / (1.0 + (-v).exp());
        let h_ref: Vec<f32> = matvec(&gate, hidden, &x)
            .iter()
            .zip(matvec(&up, hidden, &x))
            .map(|(g, u)| silu(*g) * u)
            .collect();
        let out_ref = matvec(&down, inter, &h_ref);

        let members = [GroupMember {
            weights: MemberData::F32(&down),
            in_features: inter,
            bits: 4,
            group_size: 64,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }];
        let plan = AwqConfig::default().search_group(&members, &absmax);
        let (mut up2, mut down2) = (up.clone(), down.clone());
        scale_rows_in_place(&mut up2, hidden, &plan.inverse_scales);
        scale_columns_in_place(&mut down2, inter, &plan.scales);
        let h2: Vec<f32> = matvec(&gate, hidden, &x)
            .iter()
            .zip(matvec(&up2, hidden, &x))
            .map(|(g, u)| silu(*g) * u)
            .collect();
        let out_f = matvec(&down2, inter, &h2);
        assert!(
            max_rel_err(&out_ref, &out_f) < 1e-4,
            "down fold err {}",
            max_rel_err(&out_ref, &out_f)
        );
    }

    /// Group search never regresses the members' combined RTN error
    /// (α=0 is the identity and sits in the grid).
    #[test]
    fn group_search_never_worse_than_rtn() {
        let n_in = 64usize;
        let mut seed = 3u32;
        let a: Vec<f32> = (0..8 * n_in).map(|_| lcg(&mut seed)).collect();
        let b: Vec<f32> = (0..4 * n_in).map(|_| lcg(&mut seed) * 0.1).collect();
        let absmax: Vec<f32> = (0..n_in)
            .map(|_| 0.1 + lcg(&mut seed).abs() * 3.0)
            .collect();
        let members = [
            GroupMember {
                weights: MemberData::F32(&a),
                in_features: n_in,
                bits: 4,
                group_size: 64,
                symmetric: false,
                scale_dtype: ScaleDtype::Bf16,
            },
            GroupMember {
                weights: MemberData::F32(&b),
                in_features: n_in,
                bits: 3,
                group_size: 32,
                symmetric: false,
                scale_dtype: ScaleDtype::Bf16,
            },
        ];
        let plan = AwqConfig::default().search_group(&members, &absmax);
        let identity = vec![1.0f32; n_in];
        let cw = activation_weights(&absmax);
        let sse_rtn = rtn_reconstruction_mse(
            &a,
            &a,
            n_in,
            8,
            &identity,
            &cw,
            4,
            64,
            false,
            ScaleDtype::Bf16,
        ) as f64
            * a.len() as f64
            + rtn_reconstruction_mse(
                &b,
                &b,
                n_in,
                4,
                &identity,
                &cw,
                3,
                32,
                false,
                ScaleDtype::Bf16,
            ) as f64
                * b.len() as f64;
        let mse_rtn = (sse_rtn / (a.len() + b.len()) as f64) as f32;
        assert!(
            plan.mse <= mse_rtn * 1.0001,
            "group awq {} vs rtn {}",
            plan.mse,
            mse_rtn
        );
    }

    /// Random weights, strongly non-uniform activation energy: the
    /// weighted objective must move off α=0 (plain RTN would never
    /// pick a scale under the unweighted criterion).
    #[test]
    fn nonuniform_activation_energy_picks_positive_alpha() {
        let n_in = 128usize;
        let n_out = 64usize;
        let mut seed = 99u32;
        let weights: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
        // 8 hot channels carry 30× the activation magnitude of the rest.
        let absmax: Vec<f32> = (0..n_in)
            .map(|j| if j % 16 == 0 { 30.0 } else { 1.0 })
            .collect();
        let meansq: Vec<f32> = absmax.iter().map(|a| a * a * 0.3).collect();
        let cw = activation_weights_from(Some(&meansq), &absmax);
        let plan = AwqConfig::default().search_weighted(&weights, n_in, &absmax, &cw, 4, 64, false);
        assert!(
            plan.alpha > 0.0,
            "weighted search should pick alpha > 0, got {}",
            plan.alpha
        );
        let identity = vec![1.0f32; n_in];
        let rtn = rtn_reconstruction_mse(
            &weights,
            &weights,
            n_in,
            n_out,
            &identity,
            &cw,
            4,
            64,
            false,
            ScaleDtype::Bf16,
        );
        assert!(
            plan.mse < rtn,
            "awq {} should beat rtn {} on the weighted loss",
            plan.mse,
            rtn
        );
        // Fallback (absmax²) also leaves α=0 on this input.
        let plan2 = AwqConfig::default().search(&weights, n_in, &absmax, 4, 64, false);
        assert!(plan2.alpha > 0.0);
    }

    /// An explicit alpha is applied verbatim: scales are absmax^alpha
    /// normalised to geometric mean 1, inverse is the reciprocal.
    #[test]
    fn explicit_alpha_is_applied_verbatim() {
        let absmax = vec![0.5f32, 1.0, 2.0, 4.0, 0.25, 8.0, 1.5, 3.0];
        let alpha = 0.35f32;
        let plan = AwqConfig::plan_for_alpha(&absmax, alpha);
        assert_eq!(plan.alpha, alpha);
        let raw: Vec<f32> = absmax.iter().map(|a| a.powf(alpha)).collect();
        let geo = (raw.iter().map(|x| x.ln()).sum::<f32>() / raw.len() as f32).exp();
        for (i, (&s, r)) in plan.scales.iter().zip(&raw).enumerate() {
            assert!((s - r / geo).abs() < 1e-5, "scale {i}: {s} vs {}", r / geo);
            assert!((plan.inverse_scales[i] * s - 1.0).abs() < 1e-5);
        }
        assert!(plan.mse.is_nan());
        // alpha=0 is the identity
        let id = AwqConfig::plan_for_alpha(&absmax, 0.0);
        assert!(id.scales.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    /// The streaming row-parallel search scores exactly what the
    /// whole-matrix reference (`scale_columns` + `rtn_reconstruction_mse`)
    /// scores: same curve, same alpha.
    #[test]
    fn streaming_search_matches_whole_matrix_reference() {
        let (n_in, n_out) = (64usize, 40usize);
        let mut seed = 11u32;
        let weights: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
        let absmax: Vec<f32> = (0..n_in)
            .map(|j| {
                if j % 8 == 0 {
                    20.0
                } else {
                    0.5 + lcg(&mut seed).abs()
                }
            })
            .collect();
        let cw = activation_weights(&absmax);
        let cfg = AwqConfig::default();
        let member = [GroupMember {
            weights: MemberData::F32(&weights),
            in_features: n_in,
            bits: 4,
            group_size: 32,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }];
        let (plan, curve) = cfg.search_group_weighted_curve(&member, &absmax, &cw);
        let mut best = (0.0f32, f32::INFINITY);
        for (&alpha, &(a2, mse_stream)) in cfg.alpha_grid.iter().zip(&curve) {
            assert_eq!(alpha, a2);
            let scales = absmax_to_scales(&absmax, alpha);
            let inv: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
            let rotated = scale_columns(&weights, n_in, &scales);
            let mse_ref = rtn_reconstruction_mse(
                &weights,
                &rotated,
                n_in,
                n_out,
                &inv,
                &cw,
                4,
                32,
                false,
                ScaleDtype::Bf16,
            );
            assert!(
                (mse_stream - mse_ref).abs() <= 1e-5 * mse_ref.max(1e-12),
                "alpha {alpha}: {mse_stream} vs {mse_ref}"
            );
            if mse_ref < best.1 {
                best = (alpha, mse_ref);
            }
        }
        assert_eq!(plan.alpha, best.0);
    }

    /// bf16 source bytes score identically to the same values as f32.
    #[test]
    fn bf16_member_equals_f32_member() {
        let (n_in, n_out) = (32usize, 24usize);
        let mut seed = 5u32;
        let vals: Vec<f32> = (0..n_out * n_in)
            .map(|_| half::bf16::from_f32(lcg(&mut seed)).to_f32())
            .collect();
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let absmax: Vec<f32> = (0..n_in)
            .map(|_| 0.1 + lcg(&mut seed).abs() * 4.0)
            .collect();
        let cw = activation_weights(&absmax);
        let cfg = AwqConfig::default();
        let f = [GroupMember {
            weights: MemberData::F32(&vals),
            in_features: n_in,
            bits: 3,
            group_size: 32,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }];
        let b = [GroupMember {
            weights: MemberData::Bf16(&bytes),
            in_features: n_in,
            bits: 3,
            group_size: 32,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }];
        let (pf, cf) = cfg.search_group_weighted_curve(&f, &absmax, &cw);
        let (pb, cb) = cfg.search_group_weighted_curve(&b, &absmax, &cw);
        assert_eq!(pf.alpha, pb.alpha);
        assert_eq!(cf, cb);
        assert_eq!(MemberData::Bf16(&bytes).len(), vals.len());
    }

    /// `search_rows` scores the evenly strided subset and weights it by
    /// the member's full size — a member with identical rows gives the
    /// same curve subsampled or not.
    #[test]
    fn search_rows_subsamples_rows() {
        let (n_in, n_out) = (32usize, 64usize);
        let mut seed = 21u32;
        let row: Vec<f32> = (0..n_in).map(|_| lcg(&mut seed)).collect();
        let weights: Vec<f32> = (0..n_out).flat_map(|_| row.clone()).collect();
        let absmax: Vec<f32> = (0..n_in)
            .map(|_| 0.1 + lcg(&mut seed).abs() * 4.0)
            .collect();
        let cw = activation_weights(&absmax);
        let m = [GroupMember {
            weights: MemberData::F32(&weights),
            in_features: n_in,
            bits: 4,
            group_size: 32,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }];
        let full = AwqConfig::default().search_group_weighted_curve(&m, &absmax, &cw);
        let sub = AwqConfig {
            search_rows: 8,
            ..Default::default()
        }
        .search_group_weighted_curve(&m, &absmax, &cw);
        assert_eq!(full.0.alpha, sub.0.alpha);
        for ((a, x), (b, y)) in full.1.iter().zip(&sub.1) {
            assert_eq!(a, b);
            assert!((x - y).abs() <= 1e-6 * x.max(1e-12), "{x} vs {y}");
        }
    }

    #[test]
    fn merge_absmax_reports_disagreement() {
        let (m, agree) = merge_absmax(&[&[1.0, 2.0], &[1.0, 2.0]], 1e-3);
        assert_eq!(m, vec![1.0, 2.0]);
        assert!(agree);
        let (m, agree) = merge_absmax(&[&[1.0, 2.0], &[1.5, 2.0]], 1e-3);
        assert_eq!(m, vec![1.5, 2.0]);
        assert!(!agree);
    }

    /// The alpha search quantizes each member with ITS scale dtype: the
    /// curve `awq-search --scale-dtype f16` reports is the error the f16
    /// pack realizes, not the bf16 grid's. Per dtype the streamed curve
    /// equals the whole-matrix pack/unpack score at that dtype, and the
    /// two dtypes' curves differ.
    #[test]
    fn search_honours_member_scale_dtype() {
        let (n_in, n_out) = (64usize, 40usize);
        let mut seed = 17u32;
        let weights: Vec<f32> = (0..n_out * n_in).map(|_| lcg(&mut seed)).collect();
        let absmax: Vec<f32> = (0..n_in)
            .map(|j| {
                if j % 8 == 0 {
                    20.0
                } else {
                    0.5 + lcg(&mut seed).abs()
                }
            })
            .collect();
        let cw = activation_weights(&absmax);
        let cfg = AwqConfig::default();
        let mut curves = Vec::new();
        for sd in [ScaleDtype::Bf16, ScaleDtype::F16] {
            let member = [GroupMember {
                weights: MemberData::F32(&weights),
                in_features: n_in,
                bits: 4,
                group_size: 32,
                symmetric: false,
                scale_dtype: sd,
            }];
            let (_, curve) = cfg.search_group_weighted_curve(&member, &absmax, &cw);
            for (&alpha, &(_, mse_stream)) in cfg.alpha_grid.iter().zip(&curve) {
                let scales = absmax_to_scales(&absmax, alpha);
                let inv: Vec<f32> = scales.iter().map(|&s| 1.0 / s).collect();
                let rotated = scale_columns(&weights, n_in, &scales);
                let mse_ref = rtn_reconstruction_mse(
                    &weights, &rotated, n_in, n_out, &inv, &cw, 4, 32, false, sd,
                );
                assert!(
                    (mse_stream - mse_ref).abs() <= 1e-5 * mse_ref.max(1e-12),
                    "{sd:?} alpha {alpha}: {mse_stream} vs {mse_ref}"
                );
            }
            curves.push(curve);
        }
        assert_ne!(
            curves[0], curves[1],
            "bf16 and f16 scales round differently; equal curves mean the dtype was ignored"
        );
    }

    /// down_proj fold on a BIASED up_proj: up's rows and up's bias take
    /// 1/s, down's columns take s, and silu(gate x) * (up x + b_up) -> down
    /// is unchanged in FP. Scaling the rows alone leaves the bias term
    /// multiplied by s (the full-precision function changes), which the
    /// second assertion pins.
    #[test]
    fn biased_down_up_fold_is_exact() {
        let hidden = 32usize;
        let inter = 64usize;
        let mut seed = 23u32;
        let gate: Vec<f32> = (0..inter * hidden).map(|_| lcg(&mut seed)).collect();
        let up: Vec<f32> = (0..inter * hidden).map(|_| lcg(&mut seed)).collect();
        let b_up: Vec<f32> = (0..inter).map(|_| 0.5 * lcg(&mut seed)).collect();
        let down: Vec<f32> = (0..hidden * inter).map(|_| lcg(&mut seed)).collect();
        let absmax: Vec<f32> = (0..inter)
            .map(|_| 0.02 + lcg(&mut seed).abs() * 5.0)
            .collect();
        let x: Vec<f32> = (0..hidden).map(|_| lcg(&mut seed)).collect();

        let silu = |v: f32| v / (1.0 + (-v).exp());
        let mlp = |up_w: &[f32], up_b: &[f32], down_w: &[f32]| -> Vec<f32> {
            let h: Vec<f32> = matvec(&gate, hidden, &x)
                .iter()
                .zip(matvec(up_w, hidden, &x))
                .zip(up_b)
                .map(|((g, u), b)| silu(*g) * (u + b))
                .collect();
            matvec(down_w, inter, &h)
        };
        let out_ref = mlp(&up, &b_up, &down);

        let members = [GroupMember {
            weights: MemberData::F32(&down),
            in_features: inter,
            bits: 4,
            group_size: 64,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }];
        let plan = AwqConfig::default().search_group(&members, &absmax);
        assert!(
            plan.alpha > 0.0,
            "nonuniform absmax should pick a nonzero alpha"
        );
        let (mut up2, mut b2, mut down2) = (up.clone(), b_up.clone(), down.clone());
        scale_rows_in_place(&mut up2, hidden, &plan.inverse_scales);
        scale_vec_in_place(&mut b2, &plan.inverse_scales);
        scale_columns_in_place(&mut down2, inter, &plan.scales);
        let out_f = mlp(&up2, &b2, &down2);
        assert!(
            max_rel_err(&out_ref, &out_f) < 1e-4,
            "biased down fold err {}",
            max_rel_err(&out_ref, &out_f)
        );
        // Rows scaled, bias not: the bias term rides s and the output moves.
        let out_bad = mlp(&up2, &b_up, &down2);
        assert!(
            max_rel_err(&out_ref, &out_bad) > 1e-2,
            "an unscaled bias should change the output (err {})",
            max_rel_err(&out_ref, &out_bad)
        );
    }

    /// o_proj fold on a BIASED v_proj (Qwen2-style attention, one head so
    /// o's input channel is v's output row): v's rows and v.bias take 1/s,
    /// o's columns take s, and W_o (W_v x + b_v) is unchanged in FP.
    #[test]
    fn biased_o_proj_fold_is_exact() {
        let hidden = 32usize;
        let hd = 32usize;
        let mut seed = 29u32;
        let v: Vec<f32> = (0..hd * hidden).map(|_| lcg(&mut seed)).collect();
        let b_v: Vec<f32> = (0..hd).map(|_| 0.5 * lcg(&mut seed)).collect();
        let o: Vec<f32> = (0..hidden * hd).map(|_| lcg(&mut seed)).collect();
        let absmax: Vec<f32> = (0..hd).map(|_| 0.02 + lcg(&mut seed).abs() * 5.0).collect();
        let x: Vec<f32> = (0..hidden).map(|_| lcg(&mut seed)).collect();

        let attn = |v_w: &[f32], v_b: &[f32], o_w: &[f32]| -> Vec<f32> {
            let val: Vec<f32> = matvec(v_w, hidden, &x)
                .iter()
                .zip(v_b)
                .map(|(a, b)| a + b)
                .collect();
            matvec(o_w, hd, &val)
        };
        let out_ref = attn(&v, &b_v, &o);

        let members = [GroupMember {
            weights: MemberData::F32(&o),
            in_features: hd,
            bits: 4,
            group_size: 32,
            symmetric: false,
            scale_dtype: ScaleDtype::F16,
        }];
        let plan = AwqConfig::default().search_group(&members, &absmax);
        assert!(plan.alpha > 0.0);
        let (mut v2, mut b2, mut o2) = (v.clone(), b_v.clone(), o.clone());
        scale_rows_in_place(&mut v2, hidden, &plan.inverse_scales);
        scale_vec_in_place(&mut b2, &plan.inverse_scales);
        scale_columns_in_place(&mut o2, hd, &plan.scales);
        let out_f = attn(&v2, &b2, &o2);
        assert!(
            max_rel_err(&out_ref, &out_f) < 1e-4,
            "biased o fold err {}",
            max_rel_err(&out_ref, &out_f)
        );
        let out_bad = attn(&v2, &b_v, &o2);
        assert!(max_rel_err(&out_ref, &out_bad) > 1e-2);
    }
}
