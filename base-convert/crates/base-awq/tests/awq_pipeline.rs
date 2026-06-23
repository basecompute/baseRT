//! End-to-end test of the AWQ + canonical-quant pipeline.
//!
//! Generates synthetic fp16 weights with a salient-channel pattern,
//! runs AWQ search to find per-channel scales, applies the rotation,
//! packs through the canonical RTN packer, and verifies that the
//! AWQ path beats plain RTN on reconstruction MSE for the salient
//! channels. This is the contract Phase 5 (legacy kernel deletion)
//! relies on: the canonical path matches or beats the v1.0 quant
//! quality at the same bit budget.

use base_awq::{awq_apply, AwqConfig};
use base_format::ScaleDtype;
use base_quant::{pack_rtn, unpack_rtn, RtnConfig};

/// Run plain RTN on `weights` row-by-row at `(bits, group_size,
/// symmetric, scale_dtype)`. Returns mean reconstruction MSE.
fn rtn_mse(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    bits: u32,
    group_size: u32,
    symmetric: bool,
) -> f32 {
    let cfg = RtnConfig {
        bits,
        group_size,
        symmetric,
        scale_dtype: ScaleDtype::Bf16,
    };
    let mut sse = 0f64;
    let mut n = 0usize;
    for i in 0..out_features {
        let row = &weights[i * in_features..(i + 1) * in_features];
        let packed = pack_rtn(row, cfg);
        let dequant = unpack_rtn(&packed, in_features, cfg);
        for (a, b) in row.iter().zip(dequant.iter()) {
            sse += ((a - b) as f64).powi(2);
            n += 1;
        }
    }
    (sse / (n as f64)) as f32
}

fn awq_then_rtn_mse(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    inverse_scales: &[f32],
    bits: u32,
    group_size: u32,
    symmetric: bool,
) -> f32 {
    let cfg = RtnConfig {
        bits,
        group_size,
        symmetric,
        scale_dtype: ScaleDtype::Bf16,
    };
    let mut sse = 0f64;
    let mut n = 0usize;
    // `weights` here is post-AWQ rotation; inverse_scales undo it.
    let original_unrotated: Vec<f32> = (0..out_features)
        .flat_map(|i| {
            (0..in_features).map(move |j| {
                weights[i * in_features + j] * inverse_scales[j]
            })
        })
        .collect();
    for i in 0..out_features {
        let row = &weights[i * in_features..(i + 1) * in_features];
        let packed = pack_rtn(row, cfg);
        let dequant = unpack_rtn(&packed, in_features, cfg);
        for (k, q) in dequant.iter().enumerate() {
            let recon = q * inverse_scales[k];
            let orig = original_unrotated[i * in_features + k];
            sse += ((recon - orig) as f64).powi(2);
            n += 1;
        }
    }
    (sse / (n as f64)) as f32
}

/// AWQ + RTN at q2 should beat plain RTN at q2 on the canonical
/// salient-weight regression case. q2 is the regime where AWQ matters
/// most — at q4+ plain RTN already preserves most channels well.
#[test]
fn awq_plus_rtn_beats_plain_rtn_at_q2() {
    let n_in = 32usize;
    let n_out = 4usize;
    // Pattern: 31 channels with weight ~1.0, channel 31 with weight 0.001.
    // Plain q2 quant rounds the small weight to 0.
    let mut weights = vec![1.0f32; n_out * n_in];
    for i in 0..n_out {
        weights[i * n_in + 31] = 0.001;
    }
    // Activation magnitudes: small everywhere except channel 31.
    let mut absmax = vec![0.01f32; n_in];
    absmax[31] = 100.0;

    let plan = AwqConfig::default().search(&weights, n_in, &absmax, 2, 32, false);
    let rotated = awq_apply(&weights, n_in, &plan.scales);

    let plain = rtn_mse(&weights, n_in, n_out, 2, 32, false);
    let awq = awq_then_rtn_mse(
        &rotated,
        n_in,
        n_out,
        &plan.inverse_scales,
        2,
        32,
        false,
    );

    assert!(
        awq < plain,
        "AWQ+RTN ({}) should beat plain RTN ({}) at q2",
        awq,
        plain
    );
}

/// At q4 with a similar pattern, AWQ should still help (or at worst tie).
/// q4 is sufficient to preserve a 4-order-of-magnitude smaller weight
/// in most groups, so the gain is much smaller than at q2.
#[test]
fn awq_plus_rtn_at_q4_does_not_regress() {
    let n_in = 64usize;
    let n_out = 2usize;
    let mut weights = vec![1.0f32; n_out * n_in];
    for i in 0..n_out {
        weights[i * n_in + 7] = 0.002;
    }
    let mut absmax = vec![0.01f32; n_in];
    absmax[7] = 50.0;

    let plan = AwqConfig::default().search(&weights, n_in, &absmax, 4, 64, false);
    let rotated = awq_apply(&weights, n_in, &plan.scales);

    let plain = rtn_mse(&weights, n_in, n_out, 4, 64, false);
    let awq = awq_then_rtn_mse(
        &rotated,
        n_in,
        n_out,
        &plan.inverse_scales,
        4,
        64,
        false,
    );

    // Non-regression invariant: AWQ search includes α=0, so the
    // optimum is by construction never worse than plain.
    assert!(
        awq <= plain * 1.01,
        "AWQ+RTN ({}) should not regress against plain RTN ({}) at q4",
        awq,
        plain
    );
}

/// AWQ-lite (α=0.5, no search) should always be applicable as a
/// fallback when no calibration data is available — and should be
/// no worse than +50% of plain RTN MSE.
#[test]
fn awq_lite_is_a_safe_fallback() {
    let n_in = 64usize;
    let n_out = 4usize;
    let mut weights = vec![0f32; n_out * n_in];
    let mut s = 0xfeedu32;
    for v in weights.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    let absmax = vec![1.0f32; n_in];

    let plan = AwqConfig::lite(n_in, &absmax);
    let rotated = awq_apply(&weights, n_in, &plan.scales);

    let plain = rtn_mse(&weights, n_in, n_out, 4, 64, false);
    let lite = awq_then_rtn_mse(
        &rotated,
        n_in,
        n_out,
        &plan.inverse_scales,
        4,
        64,
        false,
    );

    // Uniform absmax → AWQ-lite is identity; lite ≈ plain.
    assert!(
        lite < plain * 1.5,
        "AWQ-lite ({}) should not be wildly worse than RTN ({})",
        lite,
        plain
    );
}
