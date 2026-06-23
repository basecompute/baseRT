//! `base_q8` — INT8 asymmetric group-wise quantization.
//!
//! Per group of `GROUP_SIZE` consecutive values along the last dim:
//!   - `scale = (max - min) / 255.0`   (or 1.0 if range is zero)
//!   - `bias  = min`
//!   - `q[i]  = clip(round((x[i] - min) / scale), 0, 255)` stored as u8
//!
//! Both scale and bias are stored — same wire layout as `base_q4` so
//! the same `[packed | scales | biases]` writer code works, and so the
//! existing `simd_gemm_q8` / `gemv_q8` Metal kernels (which
//! read both `scales[g]` and `biases[g]` per group) decode correctly.
//!
//! The earlier symmetric variant (signed int8 + scale only) shipped a
//! padded-zeros region where the kernel read biases, so the runtime
//! computed `q * scale + 0` against UNSIGNED-reinterpreted bytes —
//! correct for q ≥ 0 but flipping sign on negative entries (i8=-1
//! became u8=255). Asymmetric matches the kernel's expected dequant
//! `x_hat = q * scale + bias` exactly.
//!
//! Default group size is 128 (one scale + one bias per 128 values).
//! Smaller than `base_q4`'s 64 because q8 has more dynamic range per
//! quantum and per-group overhead is less critical.

use crate::Packed;
use base_format;
use half::f16;

pub const GROUP_SIZE: usize = 128;

pub fn pack(weights: &[f32]) -> Packed {
    pack_with_group_size(weights, GROUP_SIZE)
}

pub fn pack_with_group_size(weights: &[f32], group_size: usize) -> Packed {
    assert!(group_size > 0);
    assert!(
        weights.len() % group_size == 0,
        "weights.len()={} must be a multiple of group_size={}",
        weights.len(),
        group_size
    );

    let n_groups = weights.len() / group_size;
    let mut packed = vec![0u8; weights.len()];
    let mut scales_f16 = Vec::with_capacity(n_groups);
    let mut biases_f16 = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let group = &weights[g * group_size..(g + 1) * group_size];
        let (mn, mx) = minmax(group);
        let raw_scale = (mx - mn) / 255.0;
        let scale_f32 = if raw_scale == 0.0 { 1.0 } else { raw_scale };

        // Round-trip scale and bias through fp16 so quantization matches
        // what kernels will compute at dequant time.
        let scale_h = f16::from_f32(scale_f32);
        let bias_h = f16::from_f32(mn);
        let scale = f16::to_f32(scale_h);
        let bias = f16::to_f32(bias_h);

        scales_f16.push(scale_h);
        biases_f16.push(bias_h);

        let inv_scale = 1.0 / scale;
        for i in 0..group_size {
            let q = ((group[i] - bias) * inv_scale).round().clamp(0.0, 255.0) as u8;
            packed[g * group_size + i] = q;
        }
    }

    Packed {
        packed_weights: packed,
        scales: f16_vec_to_bytes(&scales_f16),
        biases: f16_vec_to_bytes(&biases_f16),
        group_size: group_size as u32,
        scale_dtype: Some(base_format::ScaleDtype::F16),
    }
}

pub fn unpack(packed: &Packed, total_values: usize) -> Vec<f32> {
    assert_eq!(packed.packed_weights.len(), total_values);
    let group_size = packed.group_size as usize;
    let n_groups = total_values / group_size;
    assert_eq!(packed.scales.len(), n_groups * 2);
    assert_eq!(packed.biases.len(), n_groups * 2);

    let scales = bytes_to_f16_vec(&packed.scales);
    let biases = bytes_to_f16_vec(&packed.biases);

    let mut out = vec![0f32; total_values];
    for g in 0..n_groups {
        let scale = scales[g].to_f32();
        let bias = biases[g].to_f32();
        for i in 0..group_size {
            let flat = g * group_size + i;
            let q = packed.packed_weights[flat] as u32;
            out[flat] = (q as f32) * scale + bias;
        }
    }
    out
}

fn minmax(xs: &[f32]) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in xs {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    (mn, mx)
}

fn f16_vec_to_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for h in v {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

fn bytes_to_f16_vec(bytes: &[u8]) -> Vec<f16> {
    assert_eq!(bytes.len() % 2, 0);
    bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_round_trip_is_tight() {
        let xs: Vec<f32> = (-64..64).map(|i| i as f32).collect(); // 128 values
        let p = pack(&xs);
        assert_eq!(p.packed_weights.len(), 128);
        assert_eq!(p.scales.len(), 2);
        assert_eq!(p.biases.len(), 2);

        let recon = unpack(&p, xs.len());
        // 128-value range across 256 levels → max error ≈ 0.5/2 = 0.25.
        let max_err = xs
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 0.6, "max err {} too high", max_err);
    }

    #[test]
    fn negative_values_round_trip() {
        // The previous symmetric-i8 variant silently corrupted negatives
        // (i8=-1 reinterpreted as u8=255). Verify asymmetric handles the
        // full negative range.
        let xs: Vec<f32> = (-127..1).map(|i| i as f32).collect(); // 128 values
        let p = pack(&xs);
        let recon = unpack(&p, xs.len());
        let max_err = xs
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 0.6, "negative-range max err {} too high", max_err);
    }

    #[test]
    fn bit_exact_for_small_fixture() {
        // group_size=4, values [0, 1, 2, 3]: mn=0, mx=3, scale=3/255≈0.0118,
        // q = round(x/scale) = [0, 85, 170, 255] (after fp16 rounding of scale)
        let xs = vec![0f32, 1f32, 2f32, 3f32];
        let p = pack_with_group_size(&xs, 4);
        assert_eq!(p.packed_weights[0], 0);
        assert_eq!(p.packed_weights[3], 255);
        // Middle two depend on exact fp16 rounding of 3/255.
        assert!(p.packed_weights[1] >= 84 && p.packed_weights[1] <= 86);
        assert!(p.packed_weights[2] >= 169 && p.packed_weights[2] <= 171);
    }

    #[test]
    fn zeros_group_gets_unit_scale() {
        let xs = vec![0f32; 128];
        let p = pack(&xs);
        let scale = f16::from_le_bytes([p.scales[0], p.scales[1]]).to_f32();
        let bias = f16::from_le_bytes([p.biases[0], p.biases[1]]).to_f32();
        assert_eq!(scale, 1.0);
        assert_eq!(bias, 0.0);
        let recon = unpack(&p, xs.len());
        for v in recon {
            assert_eq!(v, 0.0);
        }
    }
}
