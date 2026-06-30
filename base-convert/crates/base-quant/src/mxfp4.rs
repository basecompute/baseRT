//! MXFP4 — OCP Microscaling FP4 (E2M1) with per-block E8M0 shared scale.
//!
//! Per group of 32 values: one 8-bit E8M0 scale (power of 2 only) and
//! 32 × 4 bits of FP4 E2M1 values. FP4 E2M1 levels are non-uniform:
//! {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} — sign + 3-bit magnitude index.
//!
//! Target: NVIDIA Blackwell tensor cores, AMD MI400, Intel Gaudi3
//! native; emulated on Apple Silicon / CPU.
//!
//! On-disk layout per 32-value block:
//!   [packed_fp4: 16 bytes (2 values per byte, low nibble first)]
//!   [scale: 1 byte (E8M0 exponent biased by 127)]
//! Total: 17 bytes per 32 values ≈ 4.25 bits/value.

use crate::Packed;
use base_format;

pub const GROUP_SIZE: usize = 32;

/// FP4 E2M1 positive representable levels.
const FP4_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// Round-to-nearest midpoints between adjacent levels.
const FP4_THRESHOLDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
/// Maximum absolute value representable by FP4 E2M1.
const FP4_MAX: f32 = 6.0;

pub fn pack(weights: &[f32]) -> Packed {
    assert!(
        weights.len() % GROUP_SIZE == 0,
        "weights.len()={} must be a multiple of {}",
        weights.len(),
        GROUP_SIZE
    );
    let n_groups = weights.len() / GROUP_SIZE;
    let mut packed = vec![0u8; weights.len() / 2];
    let mut scales = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let group = &weights[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let absmax = group.iter().copied().map(f32::abs).fold(0f32, f32::max);

        // E8M0: scale = 2^e where e = ceil(log2(absmax / 6)). Clamp to
        // i8 range (biased storage uses u8 = e + 127).
        let e_raw = if absmax > 0.0 {
            (absmax / FP4_MAX).log2().ceil()
        } else {
            0.0
        };
        let e = e_raw.clamp(-127.0, 127.0) as i32;
        let scale = 2f32.powi(e);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
        scales.push((e + 127) as u8);

        for (i, &val) in group.iter().enumerate().take(GROUP_SIZE) {
            let normed = val * inv_scale;
            let idx = fp4_index(normed);
            let byte_idx = (g * GROUP_SIZE + i) / 2;
            if i % 2 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | (idx & 0x0F);
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((idx & 0x0F) << 4);
            }
        }
    }

    Packed {
        packed_weights: packed,
        scales,
        biases: Vec::new(),
        group_size: GROUP_SIZE as u32,
        scale_dtype: Some(base_format::ScaleDtype::E8m0),
    }
}

pub fn unpack(packed: &Packed, total_values: usize) -> Vec<f32> {
    assert_eq!(packed.packed_weights.len() * 2, total_values);
    let n_groups = total_values / GROUP_SIZE;
    assert_eq!(packed.scales.len(), n_groups);

    let mut out = vec![0f32; total_values];
    for g in 0..n_groups {
        let e = packed.scales[g] as i32 - 127;
        let scale = 2f32.powi(e);
        for i in 0..GROUP_SIZE {
            let flat = g * GROUP_SIZE + i;
            let byte = packed.packed_weights[flat / 2];
            let idx = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[flat] = fp4_value(idx) * scale;
        }
    }
    out
}

/// Encode an f32 into its 4-bit FP4 E2M1 index (sign bit + 3-bit magnitude).
fn fp4_index(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x8 } else { 0 };
    let abs = x.abs();
    let mut idx = 7u8; // default to max magnitude
    for (i, &t) in FP4_THRESHOLDS.iter().enumerate() {
        if abs < t {
            idx = i as u8;
            break;
        }
    }
    // Special-case: true +0 and -0 both map to 0 (ignore sign).
    if idx == 0 {
        0
    } else {
        sign | idx
    }
}

fn fp4_value(idx: u8) -> f32 {
    let sign = if idx & 0x8 != 0 { -1.0 } else { 1.0 };
    sign * FP4_LEVELS[(idx & 0x7) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_levels_round_trip() {
        for &lvl in &FP4_LEVELS {
            assert_eq!(fp4_value(fp4_index(lvl)), lvl);
            if lvl > 0.0 {
                assert_eq!(fp4_value(fp4_index(-lvl)), -lvl);
            }
        }
    }

    #[test]
    fn pack_unpack_within_error() {
        // Values in [-6, 6] should quantize with bounded error (roughly
        // half the smallest gap = 0.25 near zero, larger elsewhere).
        let xs: Vec<f32> = (-16..16).map(|i| i as f32 * 0.375).collect(); // 32 values
        let p = pack(&xs);
        assert_eq!(p.packed_weights.len(), 16);
        assert_eq!(p.scales.len(), 1);
        let recon = unpack(&p, xs.len());
        let max_err = xs
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 1.0, "max err {}", max_err);
    }

    #[test]
    fn all_zeros_preserved() {
        let xs = vec![0f32; 32];
        let p = pack(&xs);
        let recon = unpack(&p, xs.len());
        for v in recon {
            assert_eq!(v, 0.0);
        }
    }
}
