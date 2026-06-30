//! NVFP4 — NVIDIA's variant: FP4 E2M1 values with E4M3 fp8 per-block scale.
//!
//! Higher scale precision than MXFP4 (4-bit exponent + 3-bit mantissa
//! in the block scale vs. MXFP4's power-of-2-only E8M0) at the cost of
//! a slightly denser group size (16 vs 32). Storage:
//!   [packed_fp4: 8 bytes]  [scale: 1 byte (E4M3)]
//! = 9 bytes per 16 values ≈ 4.5 bits/value.
//!
//! Target: NVIDIA Blackwell tensor cores.

use crate::Packed;
use base_format;

pub const GROUP_SIZE: usize = 16;

const FP4_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const FP4_THRESHOLDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
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

        let scale_continuous = absmax / FP4_MAX;
        let (e4m3_byte, scale) = to_e4m3(scale_continuous);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
        scales.push(e4m3_byte);

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
        scale_dtype: Some(base_format::ScaleDtype::E4m3),
    }
}

pub fn unpack(packed: &Packed, total_values: usize) -> Vec<f32> {
    assert_eq!(packed.packed_weights.len() * 2, total_values);
    let n_groups = total_values / GROUP_SIZE;
    assert_eq!(packed.scales.len(), n_groups);

    let mut out = vec![0f32; total_values];
    for g in 0..n_groups {
        let scale = from_e4m3(packed.scales[g]);
        for i in 0..GROUP_SIZE {
            let flat = g * GROUP_SIZE + i;
            let byte = packed.packed_weights[flat / 2];
            let idx = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[flat] = fp4_value(idx) * scale;
        }
    }
    out
}

/// Round an f32 scale to its E4M3 (sign + 4-bit exp + 3-bit mantissa,
/// bias=7) representation. Returns (stored byte, reconstructed f32).
fn to_e4m3(x: f32) -> (u8, f32) {
    if x <= 0.0 || !x.is_finite() {
        return (0, 1.0); // fall back to unit scale for degenerate blocks
    }
    let exp = x.log2().floor() as i32;
    let mantissa_f = x / 2f32.powi(exp); // in [1, 2)
    // 3-bit mantissa → 8 bins in [1, 2) spanning mantissa = 1.0..1.875
    // (values are 1 + n/8).
    let man_q_bits = ((mantissa_f - 1.0) * 8.0).round().clamp(0.0, 7.0) as u32;
    let man_q = 1.0 + (man_q_bits as f32) / 8.0;

    let biased_exp = (exp + 7).clamp(0, 15) as u32;
    let byte = ((biased_exp << 3) | man_q_bits) as u8;
    let recon = 2f32.powi(biased_exp as i32 - 7) * man_q;
    (byte, recon)
}

fn from_e4m3(byte: u8) -> f32 {
    if byte == 0 {
        return 1.0;
    }
    let biased_exp = (byte >> 3) & 0x0F;
    let man_q_bits = byte & 0x07;
    let man = 1.0 + (man_q_bits as f32) / 8.0;
    2f32.powi(biased_exp as i32 - 7) * man
}

fn fp4_index(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x8 } else { 0 };
    let abs = x.abs();
    let mut idx = 7u8;
    for (i, &t) in FP4_THRESHOLDS.iter().enumerate() {
        if abs < t {
            idx = i as u8;
            break;
        }
    }
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
    fn pack_unpack_bounded_error() {
        let xs: Vec<f32> = (-8..8).map(|i| i as f32 * 0.35).collect(); // 16 values
        let p = pack(&xs);
        assert_eq!(p.packed_weights.len(), 8);
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
    fn e4m3_round_trip_reasonable() {
        for v in [0.1f32, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
            let (byte, recon) = to_e4m3(v);
            let recon2 = from_e4m3(byte);
            assert!((recon - recon2).abs() < 1e-6);
            assert!(
                (v - recon).abs() / v < 0.1,
                "e4m3({v}) = {recon} (relerr {})",
                (v - recon).abs() / v
            );
        }
    }
}
