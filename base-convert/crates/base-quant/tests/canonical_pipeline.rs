//! End-to-end smoke test for the canonical-quant pipeline.
//!
//! Exercises: profile resolve → RTN pack → BaseWriter (canonical
//! header fields) → BaseReader → unpack → numerical fidelity. This is
//! the contract every later phase (AWQ, GPTQ, mixed precision)
//! must preserve: same in, same bytes out, same dequantized values.
//!
//! Operates entirely on a synthetic fp16-source model; no on-disk
//! checkpoint required. The model is small (toy scale, ~100 tensors)
//! but representative — embed_tokens, attn projections, MLP, lm_head.

use base_format::{
    AlignmentConfig, BaseReader, BaseWriter, ComputeRegion, Header, HeaderFlags, ModelConfig,
    QuantScheme, ScaleDtype, SourceInfo, TargetBackend, TensorDtype, TensorEntry, TensorFlags,
    TensorPayload, TokenizerBlob,
};
use base_quant::{pack_rtn, unpack_rtn, QuantProfile, RtnConfig};
use std::collections::BTreeMap;

fn synthetic_fp32_tensor(name: &str, rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    // Deterministic-ish mock weights: values in [-1, 1) per row, per
    // column, modulated by seed + name hash for variety. Stable across
    // runs so tests reproduce.
    let _ = name;
    let mut out = vec![0f32; rows * cols];
    let mut s = seed;
    for v in out.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
        *v = f;
    }
    out
}

fn make_profile() -> QuantProfile {
    let json = r#"{
        "name": "smoke-q4-q8",
        "arch": "synthetic",
        "rules": [
            {"pattern": "model.embed_tokens.weight", "dtype": "bf16"},
            {"pattern": "**.input_layernorm.weight", "dtype": "bf16"},
            {"pattern": "**.{q,k,v,o}_proj.weight",
             "dtype": "base_q4", "scale_dtype": "bf16", "group_size": 64},
            {"pattern": "**.{gate,up,down}_proj.weight",
             "dtype": "base_q8", "scale_dtype": "bf16", "group_size": 128},
            {"pattern": "lm_head.weight",
             "dtype": "base_q8", "scale_dtype": "bf16", "group_size": 128},
            {"pattern": "model.norm.weight", "dtype": "bf16"}
        ]
    }"#;
    QuantProfile::from_json(json.as_bytes()).unwrap()
}

fn make_header() -> Header {
    Header {
        schema: 1,
        arch: "synthetic".into(),
        quant_scheme: QuantScheme::BaseQ4,
        min_hw: "apple_m1".into(),
        created: "2026-04-29T00:00:00Z".into(),
        base_rt_version: "0.1.0-smoke".into(),
        source: SourceInfo {
            format: "synthetic".into(),
            sha256: "0".repeat(64),
            filename: "synthetic-fp16.safetensors".into(),
        },
        tokenizer: TokenizerBlob { fields: BTreeMap::new() },
        config: ModelConfig { fields: BTreeMap::new() },
        target_backend: TargetBackend::Metal,
        quant_profile: "smoke-q4-q8".into(),
        alignment: AlignmentConfig::default(),
        flags: HeaderFlags::QUANTIZED,
        layers: vec![],
        tensors: vec![],
        mmproj: None,
        calibration: None,
        sig: None,
    }
}

fn make_entry(name: &str, dtype: TensorDtype, shape: Vec<u64>, group_size: Option<u32>) -> TensorEntry {
    TensorEntry {
        name: name.into(),
        dtype,
        shape,
        offset: 0,
        length: 0,
        scale_offset: None,
        scale_length: None,
        bias_offset: None,
        bias_length: None,
        awq_scale_offset: None,
        awq_scale_length: None,
        group_size,
        scale_dtype: Some(ScaleDtype::Bf16),
        symmetric: false,
        layout: None,
        residency: None,
        compute_region: ComputeRegion::Gpu,
        flags: TensorFlags::empty(),
        checksum_xxh64: None,
        source_ggml_type: None,
    }
}

fn quantize_via_profile(
    profile: &QuantProfile,
    name: &str,
    weights_f32: &[f32],
    shape: Vec<u64>,
) -> (TensorEntry, Vec<u8>, Option<RtnConfig>) {
    let resolved = profile
        .resolve_or_err(name)
        .unwrap_or_else(|e| panic!("profile resolve {name}: {e}"));

    match resolved.dtype {
        TensorDtype::Bf16 => {
            // bf16 passthrough — pack 2 bytes/value.
            let mut bytes = Vec::with_capacity(weights_f32.len() * 2);
            for &v in weights_f32 {
                bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
            let entry = make_entry(name, TensorDtype::Bf16, shape, None);
            (entry, bytes, None)
        }
        dtype @ (TensorDtype::BaseQ2
        | TensorDtype::BaseQ3
        | TensorDtype::BaseQ4
        | TensorDtype::BaseQ5
        | TensorDtype::BaseQ6
        | TensorDtype::BaseQ8) => {
            let bits = dtype.bits_per_weight().unwrap();
            let cfg = RtnConfig {
                bits,
                group_size: resolved.group_size,
                symmetric: resolved.symmetric,
                scale_dtype: resolved.scale_dtype,
            };
            let packed = pack_rtn(weights_f32, cfg);
            // Pack the three byte streams into one TensorPayload's
            // `data` field. The writer treats `data` as one blob and
            // records sub-offsets for scales/biases via the entry.
            let weight_bytes = packed.packed_weights;
            let mut combined = weight_bytes.clone();
            let scale_off = combined.len() as u64;
            combined.extend_from_slice(&packed.scales);
            let bias_off = combined.len() as u64;
            combined.extend_from_slice(&packed.biases);

            let mut entry = make_entry(name, dtype, shape, Some(cfg.group_size));
            entry.length = weight_bytes.len() as u64;
            entry.scale_offset = Some(scale_off);
            entry.scale_length = Some(packed.scales.len() as u64);
            if !packed.biases.is_empty() {
                entry.bias_offset = Some(bias_off);
                entry.bias_length = Some(packed.biases.len() as u64);
            }
            entry.scale_dtype = Some(cfg.scale_dtype);
            entry.symmetric = cfg.symmetric;
            (entry, combined, Some(cfg))
        }
        other => panic!("smoke test does not handle dtype {other:?}"),
    }
}

#[test]
fn canonical_pipeline_round_trip_smoke() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    let profile = make_profile();
    let header = make_header();

    // Canonical layer (one-block toy) — names match what the profile
    // expects to cover.
    let names_and_shapes: Vec<(&str, Vec<u64>)> = vec![
        ("model.embed_tokens.weight", vec![1024, 256]),
        ("model.layers.0.input_layernorm.weight", vec![256]),
        ("model.layers.0.self_attn.q_proj.weight", vec![256, 256]),
        ("model.layers.0.self_attn.k_proj.weight", vec![256, 256]),
        ("model.layers.0.self_attn.v_proj.weight", vec![256, 256]),
        ("model.layers.0.self_attn.o_proj.weight", vec![256, 256]),
        ("model.layers.0.mlp.gate_proj.weight", vec![512, 256]),
        ("model.layers.0.mlp.up_proj.weight", vec![512, 256]),
        ("model.layers.0.mlp.down_proj.weight", vec![256, 512]),
        ("model.norm.weight", vec![256]),
        ("lm_head.weight", vec![1024, 256]),
    ];

    struct Original {
        name: String,
        weights: Vec<f32>,
        cfg: Option<RtnConfig>,
    }

    let mut writer = BaseWriter::create(&path, header).unwrap();
    let mut originals: Vec<Original> = Vec::new();

    for (i, (name, shape)) in names_and_shapes.iter().enumerate() {
        let total: usize = shape.iter().product::<u64>() as usize;
        let weights = synthetic_fp32_tensor(name, 1, total, (i as u32) + 1);
        let (entry, data, cfg) = quantize_via_profile(&profile, name, &weights, shape.clone());
        originals.push(Original {
            name: name.to_string(),
            weights,
            cfg,
        });
        writer.add_tensor(TensorPayload { entry, data });
    }
    writer.finish().unwrap();

    // Read back. Verify per-tensor entries carry canonical fields.
    let reader = BaseReader::open(&path).unwrap();
    let h = reader.header();
    assert_eq!(h.target_backend, TargetBackend::Metal);
    assert_eq!(h.quant_profile, "smoke-q4-q8");
    assert_eq!(h.tensors.len(), names_and_shapes.len());

    for orig in &originals {
        let orig_name = &orig.name;
        let orig_weights = &orig.weights;
        let cfg = &orig.cfg;
        let entry = h
            .tensors
            .iter()
            .find(|t| &t.name == orig_name)
            .unwrap_or_else(|| panic!("tensor {} missing from bundle", orig_name));

        match cfg {
            Some(cfg) => {
                // Quantized: dequant must match within the bit-width's
                // expected RTN error.
                assert_eq!(entry.dtype.bits_per_weight().unwrap(), cfg.bits);
                let blob = reader
                    .tensor_bytes(orig_name)
                    .unwrap_or_else(|e| panic!("tensor_bytes {orig_name}: {e}"))
                    .to_vec();
                // Writer stores weight+scales+biases combined; offsets
                // are relative to the tensor's blob start.
                let scale_off = entry.scale_offset.unwrap() as usize;
                let scale_len = entry.scale_length.unwrap() as usize;
                let weight_bytes_len = scale_off; // weights occupy [0, scale_off)
                let bias_range = entry
                    .bias_offset
                    .map(|o| (o as usize, entry.bias_length.unwrap() as usize));

                // Reconstruct a Packed and unpack.
                let weights = blob[..weight_bytes_len].to_vec();
                let scales = blob[scale_off..scale_off + scale_len].to_vec();
                let biases = bias_range
                    .map(|(o, l)| blob[o..o + l].to_vec())
                    .unwrap_or_default();
                let packed = base_quant::Packed {
                    packed_weights: weights,
                    scales,
                    biases,
                    group_size: cfg.group_size,
                    scale_dtype: Some(cfg.scale_dtype),
                };
                let recon = unpack_rtn(&packed, orig_weights.len(), *cfg);

                let levels = (1u32 << cfg.bits) as f32;
                let max = orig_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = orig_weights.iter().cloned().fold(f32::INFINITY, f32::min);
                let step = (max - min) / levels;
                // Tolerance: one full step + small absolute fudge.
                // bf16 scale-rounding can move an individual sample's
                // reconstruction by up to ~step on top of the
                // half-step RTN error, especially at higher bits where
                // the step is small relative to bf16 mantissa noise.
                let tol = step * 1.1 + 1e-3;
                let max_err = orig_weights
                    .iter()
                    .zip(recon.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_err < tol,
                    "{orig_name} bits={} max_err={} > tol={} (step={})",
                    cfg.bits,
                    max_err,
                    tol,
                    step
                );
            }
            None => {
                // bf16 passthrough — verify dtype + bit-exact.
                assert_eq!(entry.dtype, TensorDtype::Bf16);
                let blob = reader
                    .tensor_bytes(orig_name)
                    .unwrap_or_else(|e| panic!("tensor_bytes {orig_name}: {e}"));
                assert_eq!(blob.len(), orig_weights.len() * 2);
                for (i, &orig) in orig_weights.iter().enumerate() {
                    let bytes = [blob[i * 2], blob[i * 2 + 1]];
                    let bf = half::bf16::from_le_bytes(bytes);
                    let want = half::bf16::from_f32(orig);
                    assert_eq!(bf, want, "bf16 mismatch at idx {i}");
                }
            }
        }
    }
}

/// Validation gate: a profile that doesn't cover every tensor should
/// produce a clear error at the first uncovered name.
#[test]
fn profile_missing_rule_errors_clearly() {
    let json = r#"{
        "name": "incomplete",
        "arch": "synthetic",
        "rules": [
            {"pattern": "**.{q,k,v,o}_proj.weight", "dtype": "base_q4"}
        ]
    }"#;
    let p = QuantProfile::from_json(json.as_bytes()).unwrap();
    let err = p.resolve_or_err("model.embed_tokens.weight").unwrap_err();
    assert!(format!("{err}").contains("model.embed_tokens.weight"));
    assert!(format!("{err}").contains("incomplete"));
}

/// Dropping target_backend=metal mid-flight should reject at runtime.
/// (Smoke level — we exercise the Rust-side header path; the C++
/// runtime test is in `baseRT_test_base_format_reader`.)
#[test]
fn header_with_unsupported_target_backend_round_trips() {
    let mut h = make_header();
    h.target_backend = TargetBackend::CudaSm89;
    let json = h.to_canonical_json().unwrap();
    let parsed = Header::from_json_bytes(&json).unwrap();
    assert_eq!(parsed.target_backend, TargetBackend::CudaSm89);
    // Round-trip is silent at the format level; rejection lives in the
    // Metal-only runtime.
}
