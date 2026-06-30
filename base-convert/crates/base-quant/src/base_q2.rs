//! `base_q2` — INT2 asymmetric group-wise quantization.
//!
//! Per group of `GROUP_SIZE` consecutive values:
//!   - `scale = (max - min) / 3.0`   (or 1.0 if range is zero)
//!   - `bias  = min`
//!   - `q[i]  = clip(round((x[i] - min) / scale), 0, 3)`
//!
//! Weights are bit-spread into 4-byte chunks (16 lanes/u32). Matches the
//! `gemv_base_q2` / `simd_gemm_q2` kernel layout — see `base_qn` for the
//! shared implementation.

use crate::{base_qn, Packed};

/// Canonical group size for `base_q2` (matches `TensorDtype::native_group_size`).
pub const GROUP_SIZE: usize = 32;
const BITS: u32 = 2;

pub fn pack(weights: &[f32]) -> Packed {
    base_qn::pack_with(weights, BITS, GROUP_SIZE)
}

pub fn pack_with_group_size(weights: &[f32], group_size: usize) -> Packed {
    base_qn::pack_with(weights, BITS, group_size)
}

pub fn unpack(packed: &Packed, total_values: usize) -> Vec<f32> {
    base_qn::unpack_with(packed, BITS, total_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_native_group() {
        let xs: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let p = pack(&xs);
        let recon = unpack(&p, xs.len());
        assert_eq!(recon.len(), xs.len());
        assert_eq!(p.group_size, GROUP_SIZE as u32);
        assert_eq!(p.packed_weights.len(), xs.len() / 4); // 2 bits/elem
    }
}
