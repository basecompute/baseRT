//! `base_q6` — INT6 asymmetric group-wise quantization. See `base_qn`.

use crate::{base_qn, Packed};

pub const GROUP_SIZE: usize = 64;
const BITS: u32 = 6;

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
        let xs: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let p = pack(&xs);
        let recon = unpack(&p, xs.len());
        assert_eq!(p.group_size, GROUP_SIZE as u32);
        // q6: 4 lanes / 3 bytes → 128 lanes / 32 chunks = 96 bytes
        assert_eq!(p.packed_weights.len(), xs.len() / 4 * 3);
        for (a, b) in xs.iter().zip(recon.iter()) {
            assert!((a - b).abs() < 0.12, "q6 mismatch a={a} b={b}");
        }
    }
}
