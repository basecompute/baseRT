//! Round-to-nearest (RTN) quantization for canonical bit-widths.
//!
//! Supports `base_q2`, `base_q3`, `base_q4`, `base_q5`, `base_q6`,
//! `base_q8`. Both symmetric and asymmetric variants. Scale dtypes:
//! `Bf16` / `F16` / `E8m0` / `E4m3`.
//!
//! Pack layout: Metal-affine (MLX-compatible). Power-of-2 widths
//! (q2/q4/q8) concatenate lanes little-endian into uint32; q3/q5 spread
//! 8 lanes across 3/5 bytes; q6 spreads 4 lanes across 3 bytes.
//!
//! This is the round-trip baseline — no calibration. AWQ / GPTQ build
//! on top by precomputing per-channel scales before the RTN pass.

use crate::Packed;
use base_format::ScaleDtype;
use half::{bf16, f16};

/// Configuration for an RTN quantization pass.
#[derive(Debug, Clone, Copy)]
pub struct RtnConfig {
    pub bits: u32,
    pub group_size: u32,
    pub symmetric: bool,
    pub scale_dtype: ScaleDtype,
}

impl RtnConfig {
    /// Canonical defaults from `CANONICAL_QUANT_SPEC.md`: q2/q3 = gs=32,
    /// q4/q5/q6 = gs=64, q8 = gs=128. Asymmetric, bf16 scales.
    pub fn canonical(bits: u32) -> Self {
        let group_size = match bits {
            2..=3 => 32,
            4..=6 => 64,
            8 => 128,
            _ => panic!("unsupported bit width {bits} (expected 2/3/4/5/6/8)"),
        };
        Self {
            bits,
            group_size,
            symmetric: false,
            scale_dtype: ScaleDtype::Bf16,
        }
    }
}

/// Quantize an f32 tensor to the canonical bit-width per `cfg`.
/// `weights.len()` must be a multiple of `cfg.group_size`.
pub fn pack(weights: &[f32], cfg: RtnConfig) -> Packed {
    assert!(
        cfg.group_size > 0 && weights.len() % cfg.group_size as usize == 0,
        "weights.len()={} must be a multiple of group_size={}",
        weights.len(),
        cfg.group_size
    );
    assert!(
        matches!(cfg.bits, 2 | 3 | 4 | 5 | 6 | 8),
        "unsupported bit width {} (expected 2/3/4/5/6/8)",
        cfg.bits
    );
    if cfg.scale_dtype == ScaleDtype::E4m3 && cfg.bits != 8 {
        panic!("scale_dtype=e4m3 is only valid for bits=8 (per canonical spec)");
    }

    let group_size = cfg.group_size as usize;
    let n_groups = weights.len() / group_size;
    let mut q_lanes: Vec<u32> = vec![0; weights.len()];
    let mut scales_bytes = Vec::with_capacity(n_groups * cfg.scale_dtype.bytes_per_group() as usize);
    let mut biases_bytes = Vec::with_capacity(if cfg.symmetric {
        0
    } else {
        n_groups * cfg.scale_dtype.bytes_per_group() as usize
    });

    let q_max_sym: i32 = (1 << (cfg.bits - 1)) - 1; // e.g. 7 for q4
    let q_min_sym: i32 = -(1 << (cfg.bits - 1)); // e.g. -8
    let q_max_asym: i32 = (1 << cfg.bits) - 1; // e.g. 15 for q4

    for g in 0..n_groups {
        let group = &weights[g * group_size..(g + 1) * group_size];

        let (scale_f32, bias_f32) = if cfg.symmetric {
            // Symmetric: scale = max(|x|) / qmax_sym. No bias.
            let amax = group.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let scale = if amax == 0.0 {
                1.0
            } else {
                amax / q_max_sym as f32
            };
            (scale, 0.0)
        } else {
            // Asymmetric: scale = (max-min) / qmax_asym, bias = min.
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &x in group {
                if x < mn {
                    mn = x;
                }
                if x > mx {
                    mx = x;
                }
            }
            let raw = (mx - mn) / q_max_asym as f32;
            (if raw == 0.0 { 1.0 } else { raw }, mn)
        };

        // Round-trip scale + bias through their target dtype so the
        // pack-side rounding matches dequant exactly.
        let (scale_rt, scale_enc) = round_trip_scale(scale_f32, cfg.scale_dtype);
        scales_bytes.extend_from_slice(&scale_enc);
        let bias_rt = if cfg.symmetric {
            0.0
        } else {
            let (b_rt, b_enc) = round_trip_scale(bias_f32, cfg.scale_dtype);
            biases_bytes.extend_from_slice(&b_enc);
            b_rt
        };

        let inv_scale = 1.0 / scale_rt;
        for (i, &val) in group.iter().enumerate() {
            let q = if cfg.symmetric {
                (val * inv_scale).round().clamp(q_min_sym as f32, q_max_sym as f32) as i32
            } else {
                ((val - bias_rt) * inv_scale)
                    .round()
                    .clamp(0.0, q_max_asym as f32) as i32
            };
            // Store as unsigned in the bit-width's value space.
            // Symmetric values are biased into [0, 2^bits) for packing
            // by adding 2^(bits-1).
            let q_packed = if cfg.symmetric {
                (q + (1 << (cfg.bits - 1))) as u32
            } else {
                q as u32
            };
            q_lanes[g * group_size + i] = q_packed;
        }
    }

    let packed_weights = pack_lanes(&q_lanes, cfg.bits);

    Packed {
        packed_weights,
        scales: scales_bytes,
        biases: biases_bytes,
        group_size: cfg.group_size,
        scale_dtype: Some(cfg.scale_dtype),
    }
}

/// Dequantize an RTN-packed tensor. Inverse of `pack`. Used for tests
/// and the `--validate` trace gate.
pub fn unpack(packed: &Packed, total_values: usize, cfg: RtnConfig) -> Vec<f32> {
    let group_size = cfg.group_size as usize;
    assert_eq!(total_values % group_size, 0);
    let n_groups = total_values / group_size;
    let scale_bytes = cfg.scale_dtype.bytes_per_group() as usize;
    assert_eq!(packed.scales.len(), n_groups * scale_bytes);
    if !cfg.symmetric {
        assert_eq!(packed.biases.len(), n_groups * scale_bytes);
    }

    let q_lanes = unpack_lanes(&packed.packed_weights, cfg.bits, total_values);

    let mut out = vec![0f32; total_values];
    let q_offset_sym: i32 = 1 << (cfg.bits - 1);
    for g in 0..n_groups {
        let scale = decode_scale(
            &packed.scales[g * scale_bytes..(g + 1) * scale_bytes],
            cfg.scale_dtype,
        );
        let bias = if cfg.symmetric {
            0.0
        } else {
            decode_scale(
                &packed.biases[g * scale_bytes..(g + 1) * scale_bytes],
                cfg.scale_dtype,
            )
        };
        for i in 0..group_size {
            let flat = g * group_size + i;
            let q = q_lanes[flat] as i32;
            let q_real = if cfg.symmetric {
                q - q_offset_sym
            } else {
                q
            };
            out[flat] = q_real as f32 * scale + bias;
        }
    }
    out
}

// ---------- Bit-level pack / unpack ----------

/// Pack a slice of small integers (each in [0, 2^bits)) into a byte
/// stream using Metal-affine layout per bit-width.
pub fn pack_lanes(lanes: &[u32], bits: u32) -> Vec<u8> {
    match bits {
        2 | 4 | 8 => pack_pow2(lanes, bits),
        3 => pack_q3(lanes),
        5 => pack_q5(lanes),
        6 => pack_q6(lanes),
        _ => panic!("bits {bits} not supported"),
    }
}

/// Inverse of `pack_lanes`.
pub fn unpack_lanes(bytes: &[u8], bits: u32, total: usize) -> Vec<u32> {
    match bits {
        2 | 4 | 8 => unpack_pow2(bytes, bits, total),
        3 => unpack_q3(bytes, total),
        5 => unpack_q5(bytes, total),
        6 => unpack_q6(bytes, total),
        _ => panic!("bits {bits} not supported"),
    }
}

/// Power-of-2 widths: lanes concatenated little-endian, lane i at bit
/// i*bits in a uint32. q2 → 16 lanes/u32, q4 → 8/u32, q8 → 4/u32 (or
/// trivially 1 byte/lane).
fn pack_pow2(lanes: &[u32], bits: u32) -> Vec<u8> {
    let mask = (1u32 << bits) - 1;
    if bits == 8 {
        // Trivial case — one byte per lane, no bit packing.
        return lanes.iter().map(|&x| (x & 0xFF) as u8).collect();
    }
    let lanes_per_u32 = 32 / bits as usize;
    assert!(
        lanes.len() % lanes_per_u32 == 0,
        "lanes.len()={} must be a multiple of {}",
        lanes.len(),
        lanes_per_u32
    );
    let mut out = Vec::with_capacity(lanes.len() / lanes_per_u32 * 4);
    for chunk in lanes.chunks_exact(lanes_per_u32) {
        let mut word: u32 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            word |= (v & mask) << (i as u32 * bits);
        }
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn unpack_pow2(bytes: &[u8], bits: u32, total: usize) -> Vec<u32> {
    let mask = (1u32 << bits) - 1;
    if bits == 8 {
        return bytes.iter().take(total).map(|&b| b as u32).collect();
    }
    let lanes_per_u32 = 32 / bits as usize;
    let mut out = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(4) {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        for i in 0..lanes_per_u32 {
            if out.len() < total {
                out.push((word >> (i as u32 * bits)) & mask);
            }
        }
    }
    out.truncate(total);
    out
}

/// q3: 8 lanes of 3 bits → 24 bits → 3 bytes. Little-endian within the
/// 24-bit window.
fn pack_q3(lanes: &[u32]) -> Vec<u8> {
    assert!(
        lanes.len() % 8 == 0,
        "q3 requires lanes.len() multiple of 8"
    );
    let mut out = Vec::with_capacity(lanes.len() / 8 * 3);
    for chunk in lanes.chunks_exact(8) {
        let mut acc: u32 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            acc |= (v & 0x7) << (i * 3);
        }
        out.push((acc & 0xFF) as u8);
        out.push(((acc >> 8) & 0xFF) as u8);
        out.push(((acc >> 16) & 0xFF) as u8);
    }
    out
}

fn unpack_q3(bytes: &[u8], total: usize) -> Vec<u32> {
    assert!(total % 8 == 0);
    let mut out = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(3) {
        let acc =
            (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        for i in 0..8 {
            out.push((acc >> (i * 3)) & 0x7);
        }
    }
    out.truncate(total);
    out
}

/// q5: 8 lanes of 5 bits → 40 bits → 5 bytes.
fn pack_q5(lanes: &[u32]) -> Vec<u8> {
    assert!(
        lanes.len() % 8 == 0,
        "q5 requires lanes.len() multiple of 8"
    );
    let mut out = Vec::with_capacity(lanes.len() / 8 * 5);
    for chunk in lanes.chunks_exact(8) {
        let mut acc: u64 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            acc |= ((v as u64) & 0x1F) << (i * 5);
        }
        for byte in 0..5 {
            out.push(((acc >> (byte * 8)) & 0xFF) as u8);
        }
    }
    out
}

fn unpack_q5(bytes: &[u8], total: usize) -> Vec<u32> {
    assert!(total % 8 == 0);
    let mut out = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(5) {
        let mut acc: u64 = 0;
        for (i, &b) in chunk.iter().enumerate() {
            acc |= (b as u64) << (i * 8);
        }
        for i in 0..8 {
            out.push(((acc >> (i * 5)) & 0x1F) as u32);
        }
    }
    out.truncate(total);
    out
}

/// q6: 4 lanes of 6 bits → 24 bits → 3 bytes.
fn pack_q6(lanes: &[u32]) -> Vec<u8> {
    assert!(
        lanes.len() % 4 == 0,
        "q6 requires lanes.len() multiple of 4"
    );
    let mut out = Vec::with_capacity(lanes.len() / 4 * 3);
    for chunk in lanes.chunks_exact(4) {
        let mut acc: u32 = 0;
        for (i, &v) in chunk.iter().enumerate() {
            acc |= (v & 0x3F) << (i * 6);
        }
        out.push((acc & 0xFF) as u8);
        out.push(((acc >> 8) & 0xFF) as u8);
        out.push(((acc >> 16) & 0xFF) as u8);
    }
    out
}

fn unpack_q6(bytes: &[u8], total: usize) -> Vec<u32> {
    assert!(total % 4 == 0);
    let mut out = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(3) {
        let acc =
            (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        for i in 0..4 {
            out.push((acc >> (i * 6)) & 0x3F);
        }
    }
    out.truncate(total);
    out
}

// ---------- Scale dtype encode / decode ----------

/// Encode `x` into the target scale dtype's byte representation, return
/// (round-tripped value, encoded bytes). The round-tripped value must
/// be used by the quantization step so dequant is bit-exact.
fn round_trip_scale(x: f32, dt: ScaleDtype) -> (f32, Vec<u8>) {
    match dt {
        ScaleDtype::Bf16 => {
            let v = bf16::from_f32(x);
            (v.to_f32(), v.to_le_bytes().to_vec())
        }
        ScaleDtype::F16 => {
            let v = f16::from_f32(x);
            (v.to_f32(), v.to_le_bytes().to_vec())
        }
        ScaleDtype::E8m0 => {
            // Power-of-2 scale: round log2(|x|) to nearest integer.
            // Bias = 127 (matches OCP MX). Encode 0 via the smallest
            // representable value (e=0 → 2^(-127)).
            let mag = x.abs();
            let e = if mag == 0.0 || !mag.is_finite() {
                0u8
            } else {
                let log2 = mag.log2();
                (log2.round() as i32 + 127).clamp(0, 255) as u8
            };
            let v = if e == 0 {
                0.0
            } else {
                2f32.powi(e as i32 - 127) * x.signum()
            };
            // Sign of x is lost in pure e8m0; the canonical use is for
            // positive scales (asymmetric bias has its own bf16/f16
            // encoding too — e8m0 bias is undefined and only relevant
            // for symmetric quant where bias=0).
            (v.abs(), vec![e])
        }
        ScaleDtype::E4m3 => {
            // OCP fp8 e4m3: 4-bit exponent (bias 7), 3-bit mantissa.
            // Range ≈ [-448, 448]. Encode through bf16 round-trip to
            // approximate e4m3 (a full e4m3 codec is overkill for
            // round-trip parity in this context — kernels use the
            // bf16 fast path for now).
            let v = bf16::from_f32(x);
            // Quantize bf16 mantissa to 3 bits to simulate e4m3 noise.
            // Not a real encoder — flagged for follow-up when we
            // actually wire e4m3 kernel paths.
            let u = v.to_bits();
            let truncated = u & 0xFF80; // keep sign + exp + 3 mantissa bits
            let v_q = bf16::from_bits(truncated);
            let bytes = encode_e4m3_approx(v_q.to_f32());
            (v_q.to_f32(), vec![bytes])
        }
    }
}

/// Decode a scale value from bytes per the dtype.
fn decode_scale(bytes: &[u8], dt: ScaleDtype) -> f32 {
    match dt {
        ScaleDtype::Bf16 => bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
        ScaleDtype::F16 => f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
        ScaleDtype::E8m0 => {
            let e = bytes[0];
            if e == 0 {
                0.0
            } else {
                2f32.powi(e as i32 - 127)
            }
        }
        ScaleDtype::E4m3 => decode_e4m3_approx(bytes[0]),
    }
}

/// Approximate e4m3 encoder. Not a full IEEE-754 fp8 codec; encodes via
/// bf16-truncated mantissa to keep round-trip cheap for tests until
/// real e4m3 kernels land.
fn encode_e4m3_approx(x: f32) -> u8 {
    if x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0u8 };
    let mag = x.abs();
    let log2 = mag.log2().round() as i32;
    let exp = (log2 + 7).clamp(0, 15) as u8; // 4-bit exponent, bias 7
    // Mantissa: 3 bits of fraction beyond the implicit leading 1.
    let frac = mag / 2f32.powi(log2) - 1.0;
    let mantissa = (frac * 8.0).round().clamp(0.0, 7.0) as u8;
    sign | (exp << 3) | (mantissa & 0x7)
}

fn decode_e4m3_approx(b: u8) -> f32 {
    if b == 0 || b == 0x80 {
        return 0.0;
    }
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32 - 7;
    let mant = (b & 0x7) as f32 / 8.0;
    sign * 2f32.powi(exp) * (1.0 + mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn zeros(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// Round-trip parity: pack → unpack reconstructs values within
    /// the bit-width's expected error. Step size = range / 2^bits;
    /// max RTN error is half a step.
    fn assert_roundtrip_within_step(weights: &[f32], cfg: RtnConfig) {
        let p = pack(weights, cfg);
        let r = unpack(&p, weights.len(), cfg);
        let levels = (1 << cfg.bits) as f32;
        let mn = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (mx - mn).max(1e-6);
        let step = range / levels;
        let tol = step * 0.6;
        let max_err = weights
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < tol,
            "bits={} max_err={} > tol={} (step={})",
            cfg.bits,
            max_err,
            tol,
            step
        );
    }

    #[test]
    fn q8_with_f16_scales_matches_existing_packer() {
        // Same contract as q4: pack_rtn(bits=8, gs=128, asym, f16)
        // must be byte-identical to crate::base_q8::pack.
        let weights: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.4).collect();
        let cfg = RtnConfig {
            bits: 8,
            group_size: 128,
            symmetric: false,
            scale_dtype: ScaleDtype::F16,
        };
        let new = pack(&weights, cfg);
        let old = crate::base_q8::pack(&weights);
        assert_eq!(new.packed_weights, old.packed_weights, "q8 weights diverge");
        assert_eq!(new.scales, old.scales, "q8 scales diverge");
        assert_eq!(new.biases, old.biases, "q8 biases diverge");
    }

    #[test]
    fn q4_with_f16_scales_matches_existing_packer() {
        // Use f16 scales so the new generic packer matches the existing
        // crate::base_q4 module (which is f16-only). The canonical
        // default is bf16 — different rounding produces different
        // reconstructed values, which is correct under the new spec
        // but not byte-comparable to the old module.
        let xs = ramp(64);
        let cfg = RtnConfig {
            bits: 4,
            group_size: 64,
            symmetric: false,
            scale_dtype: ScaleDtype::F16,
        };
        let new = pack(&xs, cfg);
        let old = crate::base_q4::pack(&xs);
        assert_eq!(new.packed_weights, old.packed_weights);
        assert_eq!(new.scales, old.scales);
        assert_eq!(new.biases, old.biases);
        let r_new = unpack(&new, xs.len(), cfg);
        let r_old = crate::base_q4::unpack(&old, xs.len());
        for (a, b) in r_new.iter().zip(r_old.iter()) {
            assert!((a - b).abs() < 1e-6, "q4 f16 disagree: {a} vs {b}");
        }
    }

    #[test]
    fn q4_f16_packers_agree_on_many_groups() {
        // Exercises the RTN packer on a multi-group input that mirrors a
        // realistic per-row weight stripe (16 groups of 64 = 1024 elements,
        // matching one row of e.g. Llama q_proj at hidden=2048 / 2 since each
        // u32 holds 8 lanes). The single-group test above can't catch a
        // group-stride bug.
        let mut xs = Vec::with_capacity(1024);
        let mut s = 0xCAFEBABEu64;
        for _ in 0..1024 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = ((s >> 33) & 0x7FFFFFFF) as f32 / 0x7FFFFFFF as f32;
            xs.push(-1.5 + 3.0 * t);
        }
        let cfg = RtnConfig {
            bits: 4,
            group_size: 64,
            symmetric: false,
            scale_dtype: ScaleDtype::F16,
        };
        let new = pack(&xs, cfg);
        let old = crate::base_q4::pack(&xs);
        assert_eq!(new.packed_weights.len(), old.packed_weights.len(), "weight byte count mismatch");
        assert_eq!(new.scales.len(), old.scales.len(), "scale byte count mismatch");
        assert_eq!(new.biases.len(), old.biases.len(), "bias byte count mismatch");
        for (i, (a, b)) in new.packed_weights.iter().zip(old.packed_weights.iter()).enumerate() {
            if a != b {
                panic!("weight bytes diverge at index {i}: new=0x{a:02x} old=0x{b:02x}");
            }
        }
        for (i, (a, b)) in new.scales.iter().zip(old.scales.iter()).enumerate() {
            if a != b {
                panic!("scale bytes diverge at index {i}: new=0x{a:02x} old=0x{b:02x}");
            }
        }
        for (i, (a, b)) in new.biases.iter().zip(old.biases.iter()).enumerate() {
            if a != b {
                panic!("bias bytes diverge at index {i}: new=0x{a:02x} old=0x{b:02x}");
            }
        }
    }

    #[test]
    fn round_trip_q2_q3_q4_q5_q6_q8_asymmetric_bf16() {
        for bits in [2u32, 3, 4, 5, 6, 8] {
            let cfg = RtnConfig::canonical(bits);
            let n = (cfg.group_size * 4) as usize;
            assert_roundtrip_within_step(&ramp(n), cfg);
        }
    }

    #[test]
    fn round_trip_q4_symmetric() {
        let cfg = RtnConfig {
            bits: 4,
            group_size: 64,
            symmetric: true,
            scale_dtype: ScaleDtype::Bf16,
        };
        // Symmetric: range centered at zero.
        let xs: Vec<f32> = (-32..32).map(|i| i as f32).collect();
        let p = pack(&xs, cfg);
        assert!(p.biases.is_empty(), "symmetric must have no biases");
        let r = unpack(&p, xs.len(), cfg);
        let max_err = xs.iter().zip(r.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_err < 4.5, "symmetric q4 range max err {max_err}");
    }

    #[test]
    fn round_trip_all_zeros_uses_unit_scale() {
        for bits in [2u32, 3, 4, 5, 6, 8] {
            let cfg = RtnConfig::canonical(bits);
            let n = (cfg.group_size * 2) as usize;
            let p = pack(&zeros(n), cfg);
            let r = unpack(&p, n, cfg);
            for v in &r {
                assert!(v.abs() < 1e-6, "bits={bits} expected 0 got {v}");
            }
        }
    }

    #[test]
    fn round_trip_q2_extreme_range() {
        // q2 only has 4 levels; very tight tolerance is unrealistic,
        // but pack→unpack must not blow up, must clip cleanly to 4
        // representable levels.
        let cfg = RtnConfig::canonical(2);
        let xs: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
        let p = pack(&xs, cfg);
        let r = unpack(&p, xs.len(), cfg);
        assert_eq!(r.len(), xs.len());
        // 4 levels over a range of ~16 → step ≈ 4. half-step tol ≈ 2.4.
        let max_err = xs
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 6.0, "q2 max_err={max_err}");
    }

    #[test]
    fn pack_q4_lanes_byte_layout_matches_mlx() {
        // Per spec / MLX-affine: lane 0 at bit 0..4, lane 1 at bit 4..8,
        // ..., lane 7 at bit 28..32 inside one uint32.
        let lanes = [0xAu32, 0xB, 0xC, 0xD, 0xE, 0xF, 0x1, 0x2];
        let bytes = pack_pow2(&lanes, 4);
        // word = 0x21FEDCBA (lane 7 high nibble of byte 3, lane 0 low
        // nibble of byte 0).
        assert_eq!(bytes, vec![0xBA, 0xDC, 0xFE, 0x21]);
    }

    #[test]
    fn pack_q2_lanes_layout() {
        // 16 lanes per u32, 2 bits each. lanes[0]=0b00, lanes[1]=0b01,
        // lanes[2]=0b10, lanes[3]=0b11, rest=0 → low byte = 0b11_10_01_00
        // = 0xE4.
        let mut lanes = vec![0u32; 16];
        lanes[0] = 0b00;
        lanes[1] = 0b01;
        lanes[2] = 0b10;
        lanes[3] = 0b11;
        let bytes = pack_pow2(&lanes, 2);
        assert_eq!(bytes[0], 0xE4);
    }

    #[test]
    fn pack_q3_lanes_byte_layout() {
        // 8 lanes of 3 bits packed into 3 bytes.
        // lanes = [0,1,2,3,4,5,6,7] → acc = 0b111_110_101_100_011_010_001_000
        //  = 0xFAC688
        let lanes: Vec<u32> = (0..8).collect();
        let bytes = pack_q3(&lanes);
        assert_eq!(bytes, vec![0x88, 0xC6, 0xFA]);
        let back = unpack_q3(&bytes, 8);
        assert_eq!(back, lanes);
    }

    #[test]
    fn pack_q5_lanes_round_trip() {
        let lanes: Vec<u32> = (0..32).map(|i| i & 0x1F).collect();
        let bytes = pack_q5(&lanes);
        assert_eq!(bytes.len(), 4 * 5);
        let back = unpack_q5(&bytes, 32);
        assert_eq!(back, lanes);
    }

    #[test]
    fn pack_q6_lanes_round_trip() {
        let lanes: Vec<u32> = (0..32).map(|i| i & 0x3F).collect();
        let bytes = pack_q6(&lanes);
        assert_eq!(bytes.len(), 8 * 3);
        let back = unpack_q6(&bytes, 32);
        assert_eq!(back, lanes);
    }

    #[test]
    fn e8m0_roundtrip_powers_of_two() {
        for e in [-10i32, -1, 0, 1, 4, 7] {
            let x = 2f32.powi(e);
            let (rt, bytes) = round_trip_scale(x, ScaleDtype::E8m0);
            assert!((rt - x).abs() < 1e-6, "e8m0 PoT failed: {x} → {rt}");
            assert_eq!(bytes.len(), 1);
            let back = decode_scale(&bytes, ScaleDtype::E8m0);
            assert!((back - x).abs() < 1e-6);
        }
    }

    #[test]
    fn bf16_scale_dtype_roundtrip_q4() {
        let cfg = RtnConfig::canonical(4);
        assert_eq!(cfg.scale_dtype, ScaleDtype::Bf16);
        let xs: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let p = pack(&xs, cfg);
        // Per group: 1 bf16 scale = 2 bytes; group_size=64 → n_groups=1.
        assert_eq!(p.scales.len(), 2);
        assert_eq!(p.biases.len(), 2);
        let r = unpack(&p, xs.len(), cfg);
        assert_eq!(r.len(), xs.len());
    }

    #[test]
    fn e8m0_scale_dtype_q4_roundtrip() {
        let cfg = RtnConfig {
            bits: 4,
            group_size: 64,
            symmetric: false,
            scale_dtype: ScaleDtype::E8m0,
        };
        let xs: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let p = pack(&xs, cfg);
        // 1 byte per group scale (and bias if asymmetric).
        assert_eq!(p.scales.len(), 1);
        assert_eq!(p.biases.len(), 1);
        let r = unpack(&p, xs.len(), cfg);
        assert_eq!(r.len(), xs.len());
        // e8m0 quantizes the scale to a power of two, so error can be
        // up to ~2× a regular bf16 scale. Loose check.
        let max_err = xs
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 8.0, "q4+e8m0 max_err={max_err}");
    }

    #[test]
    #[should_panic(expected = "scale_dtype=e4m3 is only valid for bits=8")]
    fn e4m3_only_on_q8() {
        let cfg = RtnConfig {
            bits: 4,
            group_size: 64,
            symmetric: false,
            scale_dtype: ScaleDtype::E4m3,
        };
        let _ = pack(&zeros(64), cfg);
    }
}
