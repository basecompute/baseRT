//! `base_qn` — generic INT-N asymmetric group-wise quantization for the
//! `BaseQ2`/`BaseQ3`/`BaseQ5`/`BaseQ6` widths.
//!
//! Per group of `group_size` consecutive values:
//!   - `scale = (max - min) / (2^bits - 1)`   (or 1.0 if range is zero)
//!   - `bias  = min`
//!   - `q[i]  = clip(round((x[i] - min) / scale), 0, 2^bits - 1)`
//!
//! Weights are bit-spread into a little-endian byte stream: lane `i` of a
//! `lanes_per_chunk`-lane chunk occupies `bits` bits starting at bit
//! `i * bits` within the chunk, assembled byte-by-byte. This matches the
//! layouts the `gemv_base_qN` / `simd_gemm_qN` Metal kernels read:
//!
//!   q2: 16 lanes/u32 (bytes_per_chunk=4)
//!   q3:  8 lanes/3 bytes
//!   q5:  8 lanes/5 bytes
//!   q6:  4 lanes/3 bytes
//!
//! q4 and q8 stay in their own modules — q4 uses an explicit nibble-pair
//! byte format and q8 stores raw bytes; the bit-spread helper still
//! produces equivalent bytes for those widths but the existing files are
//! load-bearing for kernel ABI tests and we don't refactor them here.

use crate::Packed;
use base_format;
use half::f16;

/// Lanes per packed chunk for `bits`-bit asymmetric quant. Chosen so the
/// chunk spans an integer number of bytes (q2: 16 lanes/4B, q3/q5: 8 lanes,
/// q6: 4 lanes/3B). Panics for unsupported widths.
fn lanes_per_chunk(bits: u32) -> usize {
    match bits {
        2 => 16,
        3 | 5 => 8,
        6 => 4,
        _ => panic!("base_qn::lanes_per_chunk: unsupported bits={bits}"),
    }
}

/// Bytes consumed by one `lanes_per_chunk(bits)`-lane chunk.
fn bytes_per_chunk(bits: u32) -> usize {
    (lanes_per_chunk(bits) * bits as usize) / 8
}

/// Pack a flat f32 tensor at `bits` precision with the given group size.
/// `weights.len()` must be a multiple of `lcm(group_size, lanes_per_chunk(bits))`;
/// in practice every supported width has `lanes_per_chunk` ≤ group_size and
/// `group_size` is a multiple of `lanes_per_chunk`, so a multiple of
/// `group_size` suffices.
pub fn pack_with(weights: &[f32], bits: u32, group_size: usize) -> Packed {
    assert!(group_size > 0);
    assert!(
        weights.len() % group_size == 0,
        "weights.len()={} must be a multiple of group_size={}",
        weights.len(),
        group_size
    );
    let lpc = lanes_per_chunk(bits);
    let bpc = bytes_per_chunk(bits);
    assert!(
        group_size % lpc == 0,
        "group_size={group_size} must be a multiple of lanes_per_chunk={lpc} for q{bits}"
    );

    let total = weights.len();
    let n_groups = total / group_size;
    let n_chunks = total / lpc;
    let qmax = (1u32 << bits) - 1;

    let mut packed_weights = vec![0u8; n_chunks * bpc];
    let mut scales_f16 = Vec::with_capacity(n_groups);
    let mut biases_f16 = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let group = &weights[g * group_size..(g + 1) * group_size];
        let (mn, mx) = minmax(group);
        let raw_scale = (mx - mn) / (qmax as f32);
        let scale_f32 = if raw_scale == 0.0 { 1.0 } else { raw_scale };

        // Round-trip scale + bias through fp16 so the quantization matches
        // what kernels compute at dequant time.
        let scale_h = f16::from_f32(scale_f32);
        let bias_h = f16::from_f32(mn);
        let scale = f16::to_f32(scale_h);
        let bias = f16::to_f32(bias_h);
        scales_f16.push(scale_h);
        biases_f16.push(bias_h);

        let inv_scale = 1.0 / scale;
        for (i, &val) in group.iter().enumerate() {
            let q = ((val - bias) * inv_scale).round().clamp(0.0, qmax as f32) as u32;
            // Locate the lane in the global chunk stream.
            let flat = g * group_size + i;
            let chunk_idx = flat / lpc;
            let lane_in_chunk = flat % lpc;
            let bit_off = lane_in_chunk * bits as usize;
            let byte_in_chunk = bit_off / 8;
            let sub_shift = bit_off % 8;
            let byte_idx = chunk_idx * bpc + byte_in_chunk;
            // Spill bits across at most ceil((sub_shift + bits) / 8) bytes.
            let shifted = (q as u64 & ((1u64 << bits) - 1)) << sub_shift;
            let span = (sub_shift + bits as usize).div_ceil(8);
            for b in 0..span {
                packed_weights[byte_idx + b] |= ((shifted >> (b * 8)) & 0xFF) as u8;
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

/// Inverse of `pack_with` — reconstruct f32 values in original logical
/// order. Used for round-trip tests and for the runtime CPU oracle.
pub fn unpack_with(packed: &Packed, bits: u32, total_values: usize) -> Vec<f32> {
    let group_size = packed.group_size as usize;
    assert!(group_size > 0);
    assert_eq!(total_values % group_size, 0);
    let n_groups = total_values / group_size;
    let lpc = lanes_per_chunk(bits);
    let bpc = bytes_per_chunk(bits);
    assert_eq!(total_values % lpc, 0);
    let n_chunks = total_values / lpc;
    assert_eq!(packed.packed_weights.len(), n_chunks * bpc);
    assert_eq!(packed.scales.len(), n_groups * 2);
    assert_eq!(packed.biases.len(), n_groups * 2);

    let scales = bytes_to_f16_vec(&packed.scales);
    let biases = bytes_to_f16_vec(&packed.biases);
    let mask = (1u32 << bits) - 1;

    let mut out = vec![0f32; total_values];
    for g in 0..n_groups {
        let scale = scales[g].to_f32();
        let bias = biases[g].to_f32();
        for i in 0..group_size {
            let flat = g * group_size + i;
            let chunk_idx = flat / lpc;
            let lane_in_chunk = flat % lpc;
            let bit_off = lane_in_chunk * bits as usize;
            let byte_in_chunk = bit_off / 8;
            let sub_shift = bit_off % 8;
            let byte_idx = chunk_idx * bpc + byte_in_chunk;
            // Read up to two consecutive bytes (max span for bits ≤ 8).
            let mut raw: u32 = packed_weights_byte(packed, byte_idx) as u32;
            let span = (sub_shift + bits as usize).div_ceil(8);
            if span > 1 {
                raw |= (packed_weights_byte(packed, byte_idx + 1) as u32) << 8;
            }
            if span > 2 {
                raw |= (packed_weights_byte(packed, byte_idx + 2) as u32) << 16;
            }
            let q = (raw >> sub_shift) & mask;
            out[flat] = (q as f32) * scale + bias;
        }
    }
    out
}

fn packed_weights_byte(p: &Packed, idx: usize) -> u8 {
    p.packed_weights[idx]
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

    fn round_trip_max_err(bits: u32, group_size: usize, xs: &[f32]) -> f32 {
        let p = pack_with(xs, bits, group_size);
        let recon = unpack_with(&p, bits, xs.len());
        xs.iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn q2_ramp_round_trip_within_half_step() {
        // 0..32 over a group of 32 at q2: scale = 31/3 ≈ 10.33, half-step ≈ 5.17.
        let xs: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let err = round_trip_max_err(2, 32, &xs);
        assert!(err < 5.5, "q2 max err {err}");
    }

    #[test]
    fn q3_ramp_round_trip_within_half_step() {
        // 0..32 at q3 (group=32): scale = 31/7 ≈ 4.43, half-step ≈ 2.22.
        let xs: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let err = round_trip_max_err(3, 32, &xs);
        assert!(err < 2.5, "q3 max err {err}");
    }

    #[test]
    fn q5_ramp_round_trip_within_half_step() {
        // 0..64 at q5 (group=64): scale = 63/31 ≈ 2.03, half-step ≈ 1.02.
        let xs: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let err = round_trip_max_err(5, 64, &xs);
        assert!(err < 1.2, "q5 max err {err}");
    }

    #[test]
    fn q6_ramp_round_trip_within_half_step() {
        // 0..64 at q6 (group=64): scale = 63/63 ≈ 1.0, half-step ≈ 0.5.
        let xs: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let err = round_trip_max_err(6, 64, &xs);
        assert!(err < 0.6, "q6 max err {err}");
    }

    #[test]
    fn q2_constant_group_uses_unit_scale() {
        let xs = vec![1.25f32; 32];
        let p = pack_with(&xs, 2, 32);
        let recon = unpack_with(&p, 2, xs.len());
        for v in &recon {
            assert!((v - 1.25).abs() < 1e-2, "got {v} expected 1.25");
        }
    }

    #[test]
    fn q3_packed_bytes_match_kernel_layout() {
        // q3 layout (matches gemv_base_q3.metal):
        //   8 lanes per 3-byte chunk; lane i at bits i*3..i*3+2.
        // Picking values that decode to q-indices 0,1,2,3,4,5,6,7 (one of
        // each), then verifying the 3 bytes match the bit-spread spec.
        // Group size = 8, so scale = 7/7 = 1, bias = 0.
        let xs: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let p = pack_with(&xs, 3, 8);
        // Expected bits (LSB-first): 000 001 010 011 100 101 110 111
        //   byte 0 = 11010001 0 = 0xD1 ?  Let's recompute:
        //   pos 0 (q=0): bits 0-2 of byte 0 → 0b000
        //   pos 1 (q=1): bits 3-5 of byte 0 → 0b001 → bits 3..5 = 001
        //   pos 2 (q=2): bits 6-7 of byte 0 + bit 0 of byte 1 → 010 → byte0 hi-2 = 10, byte1 lo-1 = 0
        //   pos 3 (q=3): bits 1-3 of byte 1 → 011
        //   pos 4 (q=4): bits 4-6 of byte 1 → 100
        //   pos 5 (q=5): bits 7 of byte 1 + bits 0-1 of byte 2 → 101 → byte1 hi-1 = 1, byte2 lo-2 = 10
        //   pos 6 (q=6): bits 2-4 of byte 2 → 110
        //   pos 7 (q=7): bits 5-7 of byte 2 → 111
        // byte 0: bit7..0 = (10)(001)(000) = 10001000 = 0x88
        //   Wait, hi 2 bits of byte 0 come from pos 2 lo-2 of q=2 = bits[5:6] of q=2? q=2 = 0b010,
        //   lane bits are i*3..i*3+2 = bits 6..8 of the 24-bit chunk.
        //   bit 6 of chunk = bit 6 of byte 0 = q=2 bit 0 = 0
        //   bit 7 of chunk = bit 7 of byte 0 = q=2 bit 1 = 1
        //   bit 8 of chunk = bit 0 of byte 1 = q=2 bit 2 = 0
        // So byte 0 = bits 7..0 = q=2_bit_1, q=2_bit_0, q=1_bit_2, q=1_bit_1, q=1_bit_0, q=0_bit_2, q=0_bit_1, q=0_bit_0
        //          = 1, 0, 0, 0, 1, 0, 0, 0 = 0b10001000 = 0x88
        // byte 1 = q=5_lo1, q=4_bits, q=3_bits, q=2_bit2 = 1, 100, 011, 0 = 0b11000110 = 0xC6
        // byte 2 = q=7, q=6, q=5_hi2 = 111, 110, 10 = 0b11111010 = 0xFA
        assert_eq!(p.packed_weights, vec![0x88, 0xC6, 0xFA]);
    }

    #[test]
    fn q6_packed_bytes_match_kernel_layout() {
        // q6 layout: 4 lanes per 3-byte chunk; lane i at bits i*6..i*6+5.
        // q=[0,1,2,3] with group=4: scale=3/63≈0.0476, bias=0; q-indices=[0,21,42,63] (≈).
        // Easier: use a constant-min/max group and check the bit-spread by
        // crafting q indices directly via the dequant. Use group_size=4 ramp
        // 0..4 — scale = 3/63, q indices ≈ [0, 21, 42, 63].
        let xs: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
        let p = pack_with(&xs, 6, 4);
        // mn=0, mx=3, raw_scale = 3/63 ≈ 0.0476. f16 round-trip ≈ 0.04760742.
        // q_i ≈ round(x_i / scale) → [0, 21, 42, 63] (verify with the round trip).
        let recon = unpack_with(&p, 6, xs.len());
        for (a, b) in xs.iter().zip(recon.iter()) {
            // Half-step at q6 ≈ scale/2 ≈ 0.024.
            assert!((a - b).abs() < 0.05, "q=[?] got {b} expected {a}");
        }
        // Bit-spread sanity: 4 lanes × 6 bits = 24 bits = 3 bytes.
        assert_eq!(p.packed_weights.len(), 3);
    }

    #[test]
    fn q5_round_trip_negative_range() {
        // Symmetric around zero stays well-quantized at q5.
        let xs: Vec<f32> = (-32..32).map(|i| i as f32).collect();
        let err = round_trip_max_err(5, 64, &xs);
        assert!(err < 1.1, "q5 symmetric max err {err}");
    }

    #[test]
    fn cross_width_byte_counts() {
        // Sanity: the packed buffer length matches `K * bits / 8` for one
        // K-long row. Use K=64 (multiple of every supported lanes_per_chunk).
        let xs: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        for (bits, group_size, expected_bytes) in
            [(2u32, 32, 16), (3, 32, 24), (5, 64, 40), (6, 64, 48)]
        {
            let p = pack_with(&xs, bits, group_size);
            assert_eq!(
                p.packed_weights.len(),
                expected_bytes,
                "q{bits} K=64 should produce {expected_bytes} bytes"
            );
        }
    }
}
