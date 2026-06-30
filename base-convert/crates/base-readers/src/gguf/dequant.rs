//! GGUF ggml-type dequant to f32.
//!
//! Coverage at this phase: F32, F16, BF16, Q8_0, Q4_0. Enough to convert
//! Llama-3.2-1B in Q8_0 or Q4_0 GGUF form. Q4_K / Q6_K / IQx lands in a
//! follow-up commit.

use super::parse::TensorInfo;
use anyhow::{bail, Result};
use half::{bf16, f16};

/// ggml-type codes (from ggml.h). Only a subset is dequantized here; the
/// parser accepts any type, but dequant may refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(n: u32) -> Result<Self> {
        Ok(match n {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::Iq2Xxs,
            17 => GgmlType::Iq2Xs,
            18 => GgmlType::Iq3Xxs,
            19 => GgmlType::Iq1S,
            20 => GgmlType::Iq4Nl,
            21 => GgmlType::Iq3S,
            22 => GgmlType::Iq2S,
            23 => GgmlType::Iq4Xs,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::Iq1M,
            30 => GgmlType::BF16,
            other => bail!("unknown ggml_type: {other}"),
        })
    }

    /// (elements_per_block, bytes_per_block). Used to compute tensor
    /// total byte length from logical element count.
    pub fn block_geometry(self) -> (usize, usize) {
        use GgmlType::*;
        match self {
            F32 | I32 => (1, 4),
            F16 | BF16 | I16 => (1, 2),
            I8 => (1, 1),
            I64 | F64 => (1, 8),
            Q4_0 => (32, 18),  // fp16 d + 16 bytes (32 × 4-bit)
            Q4_1 => (32, 20),  // fp16 d + fp16 m + 16 bytes
            Q5_0 => (32, 22),  // fp16 d + 4-byte qh + 16 bytes (32 × 5-bit)
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),  // fp16 d + 32 × i8
            Q8_1 => (32, 36),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
            other => panic!("block_geometry not defined for {other:?}"),
        }
    }
}

pub fn ggml_type_name(ty: GgmlType) -> &'static str {
    use GgmlType::*;
    match ty {
        F32 => "F32",
        F16 => "F16",
        BF16 => "BF16",
        Q4_0 => "Q4_0",
        Q4_1 => "Q4_1",
        Q5_0 => "Q5_0",
        Q5_1 => "Q5_1",
        Q8_0 => "Q8_0",
        Q8_1 => "Q8_1",
        Q2K => "Q2_K",
        Q3K => "Q3_K",
        Q4K => "Q4_K",
        Q5K => "Q5_K",
        Q6K => "Q6_K",
        Q8K => "Q8_K",
        Iq2Xxs => "IQ2_XXS",
        Iq2Xs => "IQ2_XS",
        Iq3Xxs => "IQ3_XXS",
        Iq1S => "IQ1_S",
        Iq4Nl => "IQ4_NL",
        Iq3S => "IQ3_S",
        Iq2S => "IQ2_S",
        Iq4Xs => "IQ4_XS",
        Iq1M => "IQ1_M",
        I8 => "I8",
        I16 => "I16",
        I32 => "I32",
        I64 => "I64",
        F64 => "F64",
    }
}

/// Dequantize a full tensor's raw GGUF bytes into f32. `info.shape`
/// product gives the logical element count.
pub fn dequant_to_f32(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f32>> {
    let n: usize = info.shape.iter().product::<u64>() as usize;
    match info.ggml_type {
        GgmlType::F32 => f32_from_bytes(bytes, n),
        GgmlType::F16 => f16_from_bytes(bytes, n),
        GgmlType::BF16 => bf16_from_bytes(bytes, n),
        GgmlType::Q8_0 => dequant_q8_0(bytes, n),
        GgmlType::Q4_0 => dequant_q4_0(bytes, n),
        GgmlType::Q4_1 => dequant_q4_1(bytes, n),
        GgmlType::Q5_0 => dequant_q5_0(bytes, n),
        GgmlType::Q5_1 => dequant_q5_1(bytes, n),
        GgmlType::Q2K => dequant_q2_k(bytes, n),
        GgmlType::Q3K => dequant_q3_k(bytes, n),
        GgmlType::Q4K => dequant_q4_k(bytes, n),
        GgmlType::Q5K => dequant_q5_k(bytes, n),
        GgmlType::Q6K => dequant_q6_k(bytes, n),
        other => bail!(
            "dequant_to_f32: {:?} not yet implemented (tensor {:?})",
            other,
            info.name
        ),
    }
}

fn f32_from_bytes(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() < n * 4 {
        bail!("F32 byte length mismatch: got {}, need {}", bytes.len(), n * 4);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let arr: [u8; 4] = bytes[i * 4..i * 4 + 4]
            .try_into()
            .expect("4-byte slice from bounds-checked range");
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn f16_from_bytes(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() < n * 2 {
        bail!("F16 byte length mismatch: got {}, need {}", bytes.len(), n * 2);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32());
    }
    Ok(out)
}

fn bf16_from_bytes(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() < n * 2 {
        bail!("BF16 byte length mismatch: got {}, need {}", bytes.len(), n * 2);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(bf16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32());
    }
    Ok(out)
}

const Q8_0_BLOCK: usize = 34; // 2 (fp16 d) + 32 (i8)
const QK8_0: usize = 32;

fn dequant_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK8_0 != 0 {
        bail!("Q8_0 element count {n} not a multiple of {QK8_0}");
    }
    let n_blocks = n / QK8_0;
    if bytes.len() != n_blocks * Q8_0_BLOCK {
        bail!(
            "Q8_0 byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q8_0_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q8_0_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        for i in 0..QK8_0 {
            let q = bytes[base + 2 + i] as i8;
            out.push((q as f32) * d);
        }
    }
    Ok(out)
}

const Q4_0_BLOCK: usize = 18; // 2 (fp16 d) + 16 (packed 4-bit)
const QK4_0: usize = 32;

fn dequant_q4_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK4_0 != 0 {
        bail!("Q4_0 element count {n} not a multiple of {QK4_0}");
    }
    let n_blocks = n / QK4_0;
    if bytes.len() != n_blocks * Q4_0_BLOCK {
        bail!(
            "Q4_0 byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q4_0_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q4_0_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        // First 16 values: low nibbles of qs[0..16], minus 8 (signed).
        // Next 16 values: high nibbles of qs[0..16], minus 8.
        for i in 0..16 {
            let byte = bytes[base + 2 + i];
            let v0 = (byte & 0x0F) as i32 - 8;
            out.push((v0 as f32) * d);
        }
        for i in 0..16 {
            let byte = bytes[base + 2 + i];
            let v1 = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push((v1 as f32) * d);
        }
    }
    Ok(out)
}

const Q4_1_BLOCK: usize = 20; // 2 (fp16 d) + 2 (fp16 m) + 16 (qs)

fn dequant_q4_1(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK4_0 != 0 {
        bail!("Q4_1 element count {n} not a multiple of {QK4_0}");
    }
    let n_blocks = n / QK4_0;
    if bytes.len() != n_blocks * Q4_1_BLOCK {
        bail!(
            "Q4_1 byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q4_1_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q4_1_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        let m = f16::from_le_bytes([bytes[base + 2], bytes[base + 3]]).to_f32();
        for i in 0..16 {
            let byte = bytes[base + 4 + i];
            out.push((byte & 0x0F) as f32 * d + m);
        }
        for i in 0..16 {
            let byte = bytes[base + 4 + i];
            out.push(((byte >> 4) & 0x0F) as f32 * d + m);
        }
    }
    Ok(out)
}

const Q5_0_BLOCK: usize = 22; // 2 (fp16 d) + 4 (qh) + 16 (qs)
const QK5_0: usize = 32;

fn dequant_q5_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK5_0 != 0 {
        bail!("Q5_0 element count {n} not a multiple of {QK5_0}");
    }
    let n_blocks = n / QK5_0;
    if bytes.len() != n_blocks * Q5_0_BLOCK {
        bail!(
            "Q5_0 byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q5_0_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q5_0_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        let qh = u32::from_le_bytes([
            bytes[base + 2],
            bytes[base + 3],
            bytes[base + 4],
            bytes[base + 5],
        ]);
        for i in 0..16 {
            let byte = bytes[base + 6 + i];
            let hi_bit = ((qh >> i) & 1) as u8;
            let v = ((byte & 0x0F) | (hi_bit << 4)) as i32 - 16;
            out.push((v as f32) * d);
        }
        for i in 0..16 {
            let byte = bytes[base + 6 + i];
            let hi_bit = ((qh >> (i + 16)) & 1) as u8;
            let v = ((byte >> 4) | (hi_bit << 4)) as i32 - 16;
            out.push((v as f32) * d);
        }
    }
    Ok(out)
}

const Q5_1_BLOCK: usize = 24; // 2 (fp16 d) + 2 (fp16 m) + 4 (qh) + 16 (qs)

fn dequant_q5_1(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK5_0 != 0 {
        bail!("Q5_1 element count {n} not a multiple of {QK5_0}");
    }
    let n_blocks = n / QK5_0;
    if bytes.len() != n_blocks * Q5_1_BLOCK {
        bail!(
            "Q5_1 byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q5_1_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q5_1_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        let m = f16::from_le_bytes([bytes[base + 2], bytes[base + 3]]).to_f32();
        let qh = u32::from_le_bytes([
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
            bytes[base + 7],
        ]);
        for i in 0..16 {
            let byte = bytes[base + 8 + i];
            let hi_bit = ((qh >> i) & 1) as u8;
            let v = (byte & 0x0F) | (hi_bit << 4);
            out.push(v as f32 * d + m);
        }
        for i in 0..16 {
            let byte = bytes[base + 8 + i];
            let hi_bit = ((qh >> (i + 16)) & 1) as u8;
            let v = (byte >> 4) | (hi_bit << 4);
            out.push(v as f32 * d + m);
        }
    }
    Ok(out)
}

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

const Q2_K_BLOCK: usize = 84; // 16 scales + 64 qs + 2 d + 2 dmin
fn dequant_q2_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q2_K element count {n} not a multiple of {QK_K}");
    }
    let n_blocks = n / QK_K;
    if bytes.len() != n_blocks * Q2_K_BLOCK {
        bail!(
            "Q2_K byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q2_K_BLOCK
        );
    }
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q2_K_BLOCK;
        let scales = &bytes[base..base + 16];
        let qs = &bytes[base + 16..base + 16 + 64];
        let d = f16::from_le_bytes([bytes[base + 80], bytes[base + 81]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[base + 82], bytes[base + 83]]).to_f32();

        let out_base = b * QK_K;
        let mut is = 0usize;
        let mut y_off = 0usize;
        for n_off in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _j in 0..4 {
                let sc1 = scales[is];
                let dl1 = d * (sc1 & 0x0F) as f32;
                let ml1 = dmin * (sc1 >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    let q = ((qs[(n_off / 4) + l] >> shift) & 3) as i32;
                    out[out_base + y_off + l] = dl1 * q as f32 - ml1;
                }
                let sc2 = scales[is];
                let dl2 = d * (sc2 & 0x0F) as f32;
                let ml2 = dmin * (sc2 >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    let q = ((qs[(n_off / 4) + l + 16] >> shift) & 3) as i32;
                    out[out_base + y_off + 16 + l] = dl2 * q as f32 - ml2;
                }
                y_off += 32;
                shift += 2;
            }
        }
    }
    Ok(out)
}

const Q3_K_BLOCK: usize = 110; // 32 hmask + 64 qs + 12 scales + 2 d

/// Matches llama.cpp's Q3_K scale shuffle. `scales_in` is 12 bytes; we
/// produce 16 6-bit signed scales in `scales_out` (offset-32).
fn q3_k_unpack_scales(scales_in: &[u8; 12], scales_out: &mut [i8; 16]) {
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes([
            scales_in[i * 4],
            scales_in[i * 4 + 1],
            scales_in[i * 4 + 2],
            scales_in[i * 4 + 3],
        ]);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    for i in 0..4 {
        let bytes = aux[i].to_le_bytes();
        for j in 0..4 {
            scales_out[i * 4 + j] = bytes[j] as i8 - 32;
        }
    }
}

fn dequant_q3_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q3_K element count {n} not a multiple of {QK_K}");
    }
    let n_blocks = n / QK_K;
    if bytes.len() != n_blocks * Q3_K_BLOCK {
        bail!(
            "Q3_K byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q3_K_BLOCK
        );
    }
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q3_K_BLOCK;
        let hmask = &bytes[base..base + 32];
        let qs = &bytes[base + 32..base + 32 + 64];
        let mut scales_in = [0u8; 12];
        scales_in.copy_from_slice(&bytes[base + 32 + 64..base + 32 + 64 + 12]);
        let d_all = f16::from_le_bytes([bytes[base + 108], bytes[base + 109]]).to_f32();

        let mut scales = [0i8; 16];
        q3_k_unpack_scales(&scales_in, &mut scales);

        let out_base = b * QK_K;
        let mut is = 0usize;
        let mut y_off = 0usize;
        let mut m: u8 = 1;
        for n_off in (0..QK_K).step_by(128) {
            let mut shift = 0;
            for _j in 0..4 {
                let dl1 = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let q = ((qs[n_off / 4 + l] >> shift) & 3) as i32;
                    let hi = if hmask[l] & m != 0 { 0 } else { 4 };
                    out[out_base + y_off + l] = dl1 * (q - hi) as f32;
                }
                let dl2 = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let q = ((qs[n_off / 4 + l + 16] >> shift) & 3) as i32;
                    let hi = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    out[out_base + y_off + 16 + l] = dl2 * (q - hi) as f32;
                }
                y_off += 32;
                shift += 2;
                m <<= 1;
            }
        }
    }
    Ok(out)
}

const Q4_K_BLOCK: usize = 144; // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)

/// Extract a 6-bit scale and 6-bit min from the packed K_SCALE_SIZE
/// scales array for sub-block `j` (0..8). Matches llama.cpp's
/// `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q4_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q4_K element count {n} not a multiple of {QK_K}");
    }
    let n_blocks = n / QK_K;
    if bytes.len() != n_blocks * Q4_K_BLOCK {
        bail!(
            "Q4_K byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q4_K_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q4_K_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[base + 2], bytes[base + 3]]).to_f32();
        let scales = &bytes[base + 4..base + 4 + K_SCALE_SIZE];
        let qs = &bytes[base + 4 + K_SCALE_SIZE..base + Q4_K_BLOCK];

        let mut is = 0usize;
        let mut q_off = 0usize;
        // Outer loop over pairs of sub-blocks (0..2, 2..4, 4..6, 6..8).
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                let q = qs[q_off + l] & 0x0F;
                out.push(d1 * q as f32 - m1);
            }
            for l in 0..32 {
                let q = qs[q_off + l] >> 4;
                out.push(d2 * q as f32 - m2);
            }
            q_off += 32;
            is += 2;
        }
    }
    Ok(out)
}

const Q5_K_BLOCK: usize = 176; // 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs)

fn dequant_q5_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q5_K element count {n} not a multiple of {QK_K}");
    }
    let n_blocks = n / QK_K;
    if bytes.len() != n_blocks * Q5_K_BLOCK {
        bail!(
            "Q5_K byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q5_K_BLOCK
        );
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let base = b * Q5_K_BLOCK;
        let d = f16::from_le_bytes([bytes[base], bytes[base + 1]]).to_f32();
        let dmin = f16::from_le_bytes([bytes[base + 2], bytes[base + 3]]).to_f32();
        let scales = &bytes[base + 4..base + 4 + K_SCALE_SIZE];
        let qh = &bytes[base + 4 + K_SCALE_SIZE..base + 4 + K_SCALE_SIZE + 32];
        let qs = &bytes[base + 4 + K_SCALE_SIZE + 32..base + Q5_K_BLOCK];

        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                let lo = qs[q_off + l] & 0x0F;
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * (lo + hi) as f32 - m1);
            }
            for l in 0..32 {
                let lo = qs[q_off + l] >> 4;
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * (lo + hi) as f32 - m2);
            }
            q_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

const Q6_K_BLOCK: usize = 210; // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)

fn dequant_q6_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if n % QK_K != 0 {
        bail!("Q6_K element count {n} not a multiple of {QK_K}");
    }
    let n_blocks = n / QK_K;
    if bytes.len() != n_blocks * Q6_K_BLOCK {
        bail!(
            "Q6_K byte length mismatch: got {}, expected {}",
            bytes.len(),
            n_blocks * Q6_K_BLOCK
        );
    }
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let base = b * Q6_K_BLOCK;
        let ql = &bytes[base..base + 128];
        let qh = &bytes[base + 128..base + 128 + 64];
        // scales are signed i8.
        let sc: &[i8] = unsafe {
            std::slice::from_raw_parts(bytes[base + 128 + 64..].as_ptr() as *const i8, 16)
        };
        let d = f16::from_le_bytes([bytes[base + 128 + 64 + 16], bytes[base + 128 + 64 + 17]])
            .to_f32();

        let out_base = b * QK_K;
        // Each block processes two 128-value halves.
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let y_off = half * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0x0F)
                    | ((qh[qh_off + l] & 0x03) << 4)) as i32
                    - 32;
                let q2 = ((ql[ql_off + l + 32] & 0x0F)
                    | (((qh[qh_off + l] >> 2) & 0x03) << 4)) as i32
                    - 32;
                let q3 = ((ql[ql_off + l] >> 4)
                    | (((qh[qh_off + l] >> 4) & 0x03) << 4)) as i32
                    - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4)
                    | (((qh[qh_off + l] >> 6) & 0x03) << 4)) as i32
                    - 32;
                out[out_base + y_off + l] = d * sc[sc_off + is] as f32 * q1 as f32;
                out[out_base + y_off + l + 32] = d * sc[sc_off + is + 2] as f32 * q2 as f32;
                out[out_base + y_off + l + 64] = d * sc[sc_off + is + 4] as f32 * q3 as f32;
                out[out_base + y_off + l + 96] = d * sc[sc_off + is + 6] as f32 * q4 as f32;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_hand_verified_block() {
        // d = 2.0, q = [-128, -64, 0, 64, 127, ...] — expect
        // dequant [-256.0, -128.0, 0.0, 128.0, 254.0, ...].
        let mut bytes = Vec::new();
        let d = half::f16::from_f32(2.0);
        bytes.extend_from_slice(&d.to_le_bytes());
        let qs: [i8; 32] = [
            -128, -64, 0, 64, 127, 1, -1, 100, 10, 20, 30, 40, 50, 60, 70, 80, 90, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(qs.as_ptr() as *const u8, 32) });
        let out = dequant_q8_0(&bytes, 32).unwrap();
        assert_eq!(out[0], -256.0);
        assert_eq!(out[1], -128.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 128.0);
        assert_eq!(out[4], 254.0);
    }

    #[test]
    fn q4_0_hand_verified_block() {
        // d = 1.0, nibbles packed so value[0]=0 value[1]=15 value[16]=8 value[17]=7.
        // Element layout: low nibbles of qs[0..16] = values[0..16]
        //                 high nibbles of qs[0..16] = values[16..32]
        // We want value[0]=-8 (nibble=0, minus 8), so low nibble of qs[0] = 0.
        // We want value[1]=7 (nibble=15, minus 8), so low nibble of qs[1] = 15.
        // Hmm wait: the packed byte holds nibbles, value[i] uses qs[i]'s low nibble
        // for i<16. So:
        //   qs[0] lo=0 (value[0] = -8)
        //   qs[0] hi=8 (value[16] = 0)
        // byte[0] = (hi<<4) | lo = (8<<4) | 0 = 0x80.
        //   qs[1] lo=15 (value[1] = 7)
        //   qs[1] hi=0 (value[17] = -8)
        // byte[1] = (0<<4) | 15 = 0x0F.
        let mut bytes = Vec::new();
        let d = half::f16::from_f32(1.0);
        bytes.extend_from_slice(&d.to_le_bytes());
        let mut qs = [0u8; 16];
        qs[0] = 0x80;
        qs[1] = 0x0F;
        bytes.extend_from_slice(&qs);

        let out = dequant_q4_0(&bytes, 32).unwrap();
        assert_eq!(out[0], -8.0);
        assert_eq!(out[1], 7.0);
        assert_eq!(out[16], 0.0);
        assert_eq!(out[17], -8.0);
    }
}
