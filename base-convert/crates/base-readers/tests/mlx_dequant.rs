//! MLX packed-tensor dequant against ground truth generated with
//! `mlx.core.quantize` / `mlx.core.dequantize` (mlx 0.26, f16 weights,
//! group_size=32, seed 42). The fixture arrays in
//! `fixtures/mlx_dequant_fixtures.rs` are the verbatim packed words,
//! f16 scale/bias bit patterns, and mlx's own dequantized output.
//!
//! 3/5/6-bit are the interesting cases: their codes cross byte and u32
//! boundaries (contiguous little-endian bitstream), which the old
//! `32 / bits` slot math silently mis-decoded (issue #20 on the public
//! repo: 6-bit checkpoints failed with a bogus group-size error).

// The generated fixture literals carry full round-trip precision.
#![allow(clippy::excessive_precision)]

use base_readers::mlx::MlxDir;
use std::io::Write;
use std::path::Path;

include!("fixtures/mlx_dequant_fixtures.rs");

/// mlx dequantize emits f16; our reader recomputes in f32 from the same
/// f16 scales/biases, so results differ by at most one f16 rounding of
/// values drawn from N(0,1). A mis-decoded bitstream is off by O(1).
const TOL: f32 = 5e-3;

fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[u64], Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape, bytes) in tensors {
        let end = offset + bytes.len() as u64;
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hdr).unwrap();
    for (_, _, _, bytes) in tensors {
        f.write_all(bytes).unwrap();
    }
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Build a one-tensor MLX checkpoint dir and return the opened reader.
fn mlx_dir(
    dir: &Path,
    bits: u32,
    packed_shape: &[u64],
    scales_shape: &[u64],
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
) -> MlxDir {
    let config = serde_json::json!({
        "model_type": "test",
        "quantization": { "bits": bits, "group_size": 32 },
    });
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    write_safetensors(
        &dir.join("model.safetensors"),
        &[
            ("layer.weight", "U32", packed_shape, u32_bytes(packed)),
            ("layer.scales", "F16", scales_shape, u16_bytes(scales)),
            ("layer.biases", "F16", scales_shape, u16_bytes(biases)),
        ],
    );
    MlxDir::open(dir).unwrap()
}

fn check_case(
    bits: u32,
    logical_shape: &[u64],
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    expected: &[f32],
) {
    let tmp = tempfile::tempdir().unwrap();
    let mut packed_shape = logical_shape.to_vec();
    let last = packed_shape.len() - 1;
    packed_shape[last] = packed_shape[last] * bits as u64 / 32;
    let mut scales_shape = logical_shape.to_vec();
    scales_shape[last] = logical_shape[last] / 32;
    let mlx = mlx_dir(
        tmp.path(),
        bits,
        &packed_shape,
        &scales_shape,
        packed,
        scales,
        biases,
    );

    assert_eq!(
        mlx.unpacked_shape("layer.weight").unwrap(),
        logical_shape,
        "bits={bits}: unpacked_shape"
    );

    let got = mlx.tensor_to_f32("layer.weight").unwrap();
    assert_eq!(got.len(), expected.len(), "bits={bits}: element count");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= TOL,
            "bits={bits}: element {i} mismatch: got {g}, mlx says {e}"
        );
    }
}

#[test]
fn dequant_matches_mlx_3bit() {
    check_case(3, &[3, 64], B3_PACKED, B3_SCALES, B3_BIASES, B3_EXPECTED);
}

#[test]
fn dequant_matches_mlx_4bit() {
    check_case(4, &[3, 64], B4_PACKED, B4_SCALES, B4_BIASES, B4_EXPECTED);
}

#[test]
fn dequant_matches_mlx_5bit() {
    check_case(5, &[3, 64], B5_PACKED, B5_SCALES, B5_BIASES, B5_EXPECTED);
}

#[test]
fn dequant_matches_mlx_6bit() {
    check_case(6, &[3, 64], B6_PACKED, B6_SCALES, B6_BIASES, B6_EXPECTED);
}

#[test]
fn dequant_matches_mlx_8bit() {
    check_case(8, &[3, 64], B8_PACKED, B8_SCALES, B8_BIASES, B8_EXPECTED);
}

/// Stacked-experts layout: batch dim ahead of [out, packed_in], as in
/// MLX-LM MoE checkpoints (switch_mlp stacks experts on dim 0).
#[test]
fn dequant_matches_mlx_6bit_batched() {
    check_case(
        6,
        &[2, 3, 64],
        B6BATCH_PACKED,
        B6BATCH_SCALES,
        B6BATCH_BIASES,
        B6BATCH_EXPECTED,
    );
}

#[test]
fn unsupported_bits_is_a_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let mlx = mlx_dir(
        tmp.path(),
        7,
        &[3, 14],
        &[3, 2],
        &[0u32; 42],
        &[0u16; 6],
        &[0u16; 6],
    );
    let err = mlx.tensor_to_f32("layer.weight").unwrap_err().to_string();
    assert!(err.contains("unsupported bits=7"), "got: {err}");
}
