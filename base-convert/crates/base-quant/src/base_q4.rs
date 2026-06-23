//! `base_q4` — INT4 asymmetric group-wise quantization.
//!
//! Per group of `GROUP_SIZE` consecutive values along the last dim:
//!   - `scale = (max - min) / 15.0`   (or 1.0 if range is zero)
//!   - `bias  = min`
//!   - `q[i]  = clip(round((x[i] - min) / scale), 0, 15)`
//!
//! Scales and biases are round-tripped through fp16 during packing so
//! the dequantization math matches what kernels do at runtime. Weights
//! are packed two-per-byte with low-nibble-first ordering:
//!
//!   `byte = (q[2i+1] << 4) | q[2i]`
//!
//! so `q[2i]` is `byte & 0x0F` and `q[2i+1]` is `byte >> 4`. This
//! matches MLX's convention and keeps kernel unpack to a single mask
//! and a shift.

use crate::Packed;
use base_format;
use half::f16;

/// Canonical group size for `base_q4`.
pub const GROUP_SIZE: usize = 64;

/// Pack a contiguous f32 tensor into `base_q4`. Length must be a
/// multiple of `GROUP_SIZE`.
pub fn pack(weights: &[f32]) -> Packed {
    pack_with_group_size(weights, GROUP_SIZE)
}

/// Pack with an explicit group size (tests use smaller groups for
/// hand-verified fixtures).
pub fn pack_with_group_size(weights: &[f32], group_size: usize) -> Packed {
    assert!(group_size > 0 && group_size % 2 == 0, "group_size must be even");
    assert!(
        weights.len() % group_size == 0,
        "weights.len()={} must be a multiple of group_size={}",
        weights.len(),
        group_size
    );

    let n_groups = weights.len() / group_size;
    let mut packed_weights = vec![0u8; weights.len() / 2];
    let mut scales_f16 = Vec::with_capacity(n_groups);
    let mut biases_f16 = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let group = &weights[g * group_size..(g + 1) * group_size];
        let (mn, mx) = minmax(group);
        let raw_scale = (mx - mn) / 15.0;
        let scale_f32 = if raw_scale == 0.0 { 1.0 } else { raw_scale };

        // Round-trip scale and bias through fp16 so quantization matches
        // what kernels will compute at dequant time.
        let scale_h = f16::from_f32(scale_f32);
        let bias_h = f16::from_f32(mn);
        let scale = f16::to_f32(scale_h);
        let bias = f16::to_f32(bias_h);

        scales_f16.push(scale_h);
        biases_f16.push(bias_h);

        // Quantize with the fp16-rounded scale/bias (matches kernel math).
        let inv_scale = 1.0 / scale;
        for (i, &val) in group.iter().enumerate().take(group_size) {
            let q = ((val - bias) * inv_scale).round().clamp(0.0, 15.0) as u8;
            let byte_idx = (g * group_size + i) / 2;
            if i % 2 == 0 {
                packed_weights[byte_idx] =
                    (packed_weights[byte_idx] & 0xF0) | (q & 0x0F);
            } else {
                packed_weights[byte_idx] =
                    (packed_weights[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
            }
        }
    }

    Packed {
        packed_weights,
        scales: f16_vec_to_bytes(&scales_f16),
        biases: f16_vec_to_bytes(&biases_f16),
        group_size: group_size as u32,
        scale_dtype: Some(base_format::ScaleDtype::F16),
    }
}

/// Dequantize a `base_q4` packed tensor. Returns f32 values in original
/// logical order. Used for tests and for the `--validate` trace gate.
pub fn unpack(packed: &Packed, total_values: usize) -> Vec<f32> {
    assert_eq!(packed.packed_weights.len() * 2, total_values);
    let group_size = packed.group_size as usize;
    assert_eq!(total_values % group_size, 0);
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
            let byte = packed.packed_weights[flat / 2];
            let q = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
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
    fn all_zeros_round_trip() {
        let xs = vec![0f32; 64];
        let packed = pack(&xs);
        let recon = unpack(&packed, xs.len());
        assert_eq!(packed.packed_weights.len(), 32);
        assert_eq!(packed.scales.len(), 2);
        assert_eq!(packed.biases.len(), 2);
        for (a, b) in xs.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn ramp_round_trip_is_tight() {
        // A 0..64 ramp quantized to 4 bits per group of 64: scale is
        // 63/15 ≈ 4.2, reconstruction error should be ~±2.1 per value.
        let xs: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let packed = pack(&xs);
        let recon = unpack(&packed, xs.len());
        let max_err = xs
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // Expected max error: half the step size = (63/15)/2 ≈ 2.1.
        assert!(max_err < 2.5, "max err {} too high", max_err);
    }

    #[test]
    fn bit_exact_for_known_fixture() {
        // Group size 4 for hand-verified expected bytes. Values 0..4:
        // mn=0, mx=3, scale=3/15=0.2, bias=0
        // q = round((x-0)/0.2) = round(5x) = [0, 5, 10, 15]
        // packed (low-nibble first, 2 per byte):
        //   byte0 = (q[1]<<4) | q[0] = (5<<4) | 0 = 0x50
        //   byte1 = (q[3]<<4) | q[2] = (15<<4) | 10 = 0xFA
        let xs = vec![0f32, 1f32, 2f32, 3f32];
        let p = pack_with_group_size(&xs, 4);
        assert_eq!(p.packed_weights, vec![0x50, 0xFA]);
        assert_eq!(p.scales.len(), 2); // one f16
        assert_eq!(p.biases.len(), 2);
    }

    #[test]
    fn constant_group_uses_unit_scale() {
        // When max == min, scale would be 0 → we force scale=1 and store
        // all q=0. Dequant returns bias = original constant.
        // (1.25 is fp16-exact and not near any clippy::approx_constant
        // value, so the test stays predictable and the lint stays happy.)
        let xs = vec![1.25f32; 64];
        let p = pack(&xs);
        let recon = unpack(&p, xs.len());
        // scale should round-trip 1.0 through fp16 fine.
        let scale = half::f16::from_le_bytes([p.scales[0], p.scales[1]]).to_f32();
        let bias = half::f16::from_le_bytes([p.biases[0], p.biases[1]]).to_f32();
        assert_eq!(scale, 1.0);
        assert!((bias - 1.25).abs() < 1e-2);
        for v in &recon {
            assert!((v - 1.25).abs() < 1e-2);
        }
    }

    #[test]
    fn symmetric_ramp_centered_at_zero() {
        let xs: Vec<f32> = (-32..32).map(|i| i as f32).collect();
        let p = pack(&xs);
        let recon = unpack(&p, xs.len());
        // Max error bounded by half-step ≈ 2.1 as above.
        for (a, b) in xs.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 2.5);
        }
    }
}
