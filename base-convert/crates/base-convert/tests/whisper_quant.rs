//! End-to-end whisper quant conversion: build a synthetic HF whisper
//! checkpoint, convert it through the real binary with the shipped
//! whisper quant profiles, and pin the on-disk contract:
//!
//! - block linear projections (attn/cross_attn query/key/value/out,
//!   mlp.0/mlp.2 — encoder and decoder) carry the canonical packed
//!   scheme ([W | scales | biases], q8/gs=128 or q4/gs=64, bf16 scales);
//! - conv1/conv2, positional embeddings, token_embedding, norms and
//!   biases stay f16;
//! - header quant_scheme / QUANTIZED flag / quant_profile are correct;
//! - the default (no-profile) conversion stays all-f16.

use base_format::{BaseReader, HeaderFlags, QuantScheme, ScaleDtype, TensorDtype};
use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_basert")
}

fn profiles_dir() -> PathBuf {
    // crates/base-convert → ../../profiles
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("profiles")
}

// Synthetic whisper dims. All linear in-features (128 / 256) divide both
// canonical group sizes (q8 gs=128, q4 gs=64), like every real whisper
// size (d_model 384…1280, ffn = 4·d_model).
const D_MODEL: u64 = 128;
const FFN: u64 = 256;
const MELS: u64 = 8;
const VOCAB: u64 = 64;
const SRC_POS: u64 = 16;
const TGT_POS: u64 = 24;

/// Minimal single-shard safetensors writer (LE u64 header length +
/// JSON header + raw F32 data), enough for the converter's reader.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<u64>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        let numel: u64 = shape.iter().product();
        // Deterministic, non-constant data so quant groups get real
        // min/max spread.
        let data: Vec<u8> = (0..numel)
            .flat_map(|j| {
                let x = ((i as f32) * 0.37 + (j as f32) * 0.11).sin() * 0.5;
                x.to_le_bytes()
            })
            .collect();
        let end = offset + data.len() as u64;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
        blobs.push(data);
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = Vec::with_capacity(8 + header_bytes.len() + offset as usize);
    file.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file.extend_from_slice(&header_bytes);
    for b in &blobs {
        file.extend_from_slice(b);
    }
    std::fs::write(path, file).unwrap();
}

/// Lay down a synthetic HF whisper checkpoint dir (config.json,
/// tokenizer.json, model.safetensors) with 1 encoder + 1 decoder layer.
fn write_synthetic_whisper_dir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "model_type": "whisper",
            "d_model": D_MODEL,
            "decoder_layers": 1,
            "decoder_attention_heads": 2,
            "decoder_ffn_dim": FFN,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": FFN,
            "num_mel_bins": MELS,
            "max_source_positions": SRC_POS,
            "max_target_positions": TGT_POS,
            "vocab_size": VOCAB,
            "bos_token_id": 50257,
            "eos_token_id": 50256,
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::json!({
            "added_tokens": [
                {"id": 50256, "content": "<|endoftext|>"},
                {"id": 50257, "content": "<|startoftranscript|>"},
                {"id": 50258, "content": "<|en|>"},
                {"id": 50358, "content": "<|translate|>"},
                {"id": 50359, "content": "<|transcribe|>"},
                {"id": 50361, "content": "<|startofprev|>"},
                {"id": 50362, "content": "<|nocaptions|>"},
                {"id": 50363, "content": "<|notimestamps|>"},
            ]
        })
        .to_string(),
    )
    .unwrap();

    let mut t: Vec<(String, Vec<u64>)> = vec![
        ("model.encoder.conv1.weight".into(), vec![D_MODEL, MELS, 3]),
        ("model.encoder.conv1.bias".into(), vec![D_MODEL]),
        ("model.encoder.conv2.weight".into(), vec![D_MODEL, D_MODEL, 3]),
        ("model.encoder.conv2.bias".into(), vec![D_MODEL]),
        ("model.encoder.embed_positions.weight".into(), vec![SRC_POS, D_MODEL]),
        ("model.encoder.layer_norm.weight".into(), vec![D_MODEL]),
        ("model.encoder.layer_norm.bias".into(), vec![D_MODEL]),
        ("model.decoder.embed_tokens.weight".into(), vec![VOCAB, D_MODEL]),
        ("model.decoder.embed_positions.weight".into(), vec![TGT_POS, D_MODEL]),
        ("model.decoder.layer_norm.weight".into(), vec![D_MODEL]),
        ("model.decoder.layer_norm.bias".into(), vec![D_MODEL]),
    ];
    let attn = |t: &mut Vec<(String, Vec<u64>)>, prefix: &str, attn: &str| {
        for (suffix, shape) in [
            ("q_proj.weight", vec![D_MODEL, D_MODEL]),
            ("q_proj.bias", vec![D_MODEL]),
            ("k_proj.weight", vec![D_MODEL, D_MODEL]),
            ("v_proj.weight", vec![D_MODEL, D_MODEL]),
            ("v_proj.bias", vec![D_MODEL]),
            ("out_proj.weight", vec![D_MODEL, D_MODEL]),
            ("out_proj.bias", vec![D_MODEL]),
        ] {
            t.push((format!("{prefix}.{attn}.{suffix}"), shape));
        }
        t.push((format!("{prefix}.{attn}_layer_norm.weight"), vec![D_MODEL]));
        t.push((format!("{prefix}.{attn}_layer_norm.bias"), vec![D_MODEL]));
    };
    let mlp = |t: &mut Vec<(String, Vec<u64>)>, prefix: &str| {
        t.push((format!("{prefix}.fc1.weight"), vec![FFN, D_MODEL]));
        t.push((format!("{prefix}.fc1.bias"), vec![FFN]));
        t.push((format!("{prefix}.fc2.weight"), vec![D_MODEL, FFN]));
        t.push((format!("{prefix}.fc2.bias"), vec![D_MODEL]));
        t.push((format!("{prefix}.final_layer_norm.weight"), vec![D_MODEL]));
        t.push((format!("{prefix}.final_layer_norm.bias"), vec![D_MODEL]));
    };
    let enc = "model.encoder.layers.0";
    attn(&mut t, enc, "self_attn");
    mlp(&mut t, enc);
    let dec = "model.decoder.layers.0";
    attn(&mut t, dec, "self_attn");
    attn(&mut t, dec, "encoder_attn");
    mlp(&mut t, dec);

    write_safetensors(&dir.join("model.safetensors"), &t);
}

/// Canonical names of the quant-eligible linears in the 1+1-layer
/// synthetic checkpoint.
fn expected_quant_linears() -> Vec<String> {
    let mut v = Vec::new();
    for p in ["query", "key", "value", "out"] {
        v.push(format!("encoder.blocks.0.attn.{p}.weight"));
    }
    v.push("encoder.blocks.0.mlp.0.weight".into());
    v.push("encoder.blocks.0.mlp.2.weight".into());
    for attn in ["attn", "cross_attn"] {
        for p in ["query", "key", "value", "out"] {
            v.push(format!("decoder.blocks.0.{attn}.{p}.weight"));
        }
    }
    v.push("decoder.blocks.0.mlp.0.weight".into());
    v.push("decoder.blocks.0.mlp.2.weight".into());
    v
}

fn convert(dir: &Path, out: &Path, profile: Option<&Path>) {
    let mut cmd = Command::new(bin());
    cmd.arg("convert").arg(dir).arg("-o").arg(out);
    if let Some(p) = profile {
        cmd.arg("--profile").arg(p);
    }
    let output = cmd.output().expect("run basert convert");
    assert!(
        output.status.success(),
        "convert failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Shared assertions for a quantized whisper bundle.
fn assert_quant_bundle(
    out: &Path,
    scheme: QuantScheme,
    dtype: TensorDtype,
    profile_name: &str,
    group_size: u32,
    packed_bytes_per_weight_num: u64, // packed W bytes = numel * num / den
    packed_bytes_per_weight_den: u64,
) {
    let reader = BaseReader::open(out).unwrap();
    let h = reader.header();
    assert_eq!(h.arch, "whisper");
    assert_eq!(h.quant_scheme, scheme);
    assert_eq!(h.quant_profile, profile_name);
    assert!(h.flags.contains(HeaderFlags::QUANTIZED), "flags: {:?}", h.flags);
    assert!(h.flags.contains(HeaderFlags::TIED_EMBEDDINGS), "flags: {:?}", h.flags);

    let quant_names = expected_quant_linears();
    let mut seen_quant = 0usize;
    for t in h.tensors.iter() {
        let numel: u64 = t.shape.iter().product();
        if quant_names.contains(&t.name) {
            seen_quant += 1;
            assert_eq!(t.dtype, dtype, "{}", t.name);
            assert_eq!(t.group_size, Some(group_size), "{}", t.name);
            assert_eq!(t.scale_dtype, Some(ScaleDtype::Bf16), "{}", t.name);
            let packed = numel * packed_bytes_per_weight_num / packed_bytes_per_weight_den;
            let n_groups = numel / group_size as u64;
            // [W | scales | biases]: bf16 scales + biases, one each per group.
            assert_eq!(t.scale_offset, Some(packed), "{}", t.name);
            assert_eq!(t.scale_length, Some(n_groups * 2), "{}", t.name);
            assert_eq!(t.bias_offset, Some(packed + n_groups * 2), "{}", t.name);
            assert_eq!(t.bias_length, Some(n_groups * 2), "{}", t.name);
            assert_eq!(t.length, packed + n_groups * 4, "{}", t.name);
        } else {
            // Conv / positional embeddings / token_embedding / norms /
            // biases: raw f16, no quant sidecars.
            assert_eq!(t.dtype, TensorDtype::F16, "{}", t.name);
            assert_eq!(t.length, numel * 2, "{}", t.name);
            assert_eq!(t.scale_offset, None, "{}", t.name);
            assert_eq!(t.bias_offset, None, "{}", t.name);
            assert_eq!(t.group_size, None, "{}", t.name);
            assert_eq!(t.scale_dtype, None, "{}", t.name);
        }
    }
    assert_eq!(
        seen_quant,
        quant_names.len(),
        "all {} block linears must be quantized",
        quant_names.len()
    );
}

#[test]
fn whisper_q8_profile_quantizes_linears_only() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("hf");
    write_synthetic_whisper_dir(&src);
    let out = tmp.path().join("whisper-q8.base");
    convert(&src, &out, Some(&profiles_dir().join("whisper-q8.json")));
    // base_q8: 1 byte per weight, gs=128.
    assert_quant_bundle(
        &out,
        QuantScheme::BaseQ8,
        TensorDtype::BaseQ8,
        "whisper-q8",
        128,
        1,
        1,
    );
}

#[test]
fn whisper_q4_profile_quantizes_linears_only() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("hf");
    write_synthetic_whisper_dir(&src);
    let out = tmp.path().join("whisper-q4.base");
    convert(&src, &out, Some(&profiles_dir().join("whisper-q4.json")));
    // base_q4: half a byte per weight, gs=64.
    assert_quant_bundle(
        &out,
        QuantScheme::BaseQ4,
        TensorDtype::BaseQ4,
        "whisper-q4",
        64,
        1,
        2,
    );
}

/// No profile → the committed all-f16 behavior, byte for byte: every
/// tensor f16, header f16, no QUANTIZED flag, empty quant_profile.
#[test]
fn whisper_default_stays_all_f16() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("hf");
    write_synthetic_whisper_dir(&src);
    let out = tmp.path().join("whisper-f16.base");
    convert(&src, &out, None);

    let reader = BaseReader::open(&out).unwrap();
    let h = reader.header();
    assert_eq!(h.arch, "whisper");
    assert_eq!(h.quant_scheme, QuantScheme::F16);
    assert_eq!(h.quant_profile, "");
    assert!(!h.flags.contains(HeaderFlags::QUANTIZED), "flags: {:?}", h.flags);
    assert!(h.flags.contains(HeaderFlags::TIED_EMBEDDINGS));
    for t in h.tensors.iter() {
        let numel: u64 = t.shape.iter().product();
        assert_eq!(t.dtype, TensorDtype::F16, "{}", t.name);
        assert_eq!(t.length, numel * 2, "{}", t.name);
        assert_eq!(t.scale_offset, None, "{}", t.name);
        assert_eq!(t.group_size, None, "{}", t.name);
    }
}
