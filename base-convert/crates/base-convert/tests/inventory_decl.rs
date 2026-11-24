//! `basert inventory` is the one source of classes and folds (PIPELINE_V3
//! §4.8), and its `canonical` names are the bundle header's names: on
//! tiny synthesized HF checkpoints of three pipeline architectures, the
//! inventory (schema 3) lists every tensor of every bundle section with a
//! class, its fold groups list only `linear` / `expert_stack` consumers,
//! and the set of its canonical names equals the set of tensor names in
//! the bundle `convert --target bf16` writes — the tied head copy, the
//! MTP `fc` split and the GLM `kv_b` split included.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn basert() -> Command {
    Command::new(env!("CARGO_BIN_EXE_basert"))
}

/// A single-shard bf16 safetensors checkpoint (zeros) plus its config.
fn write_hf(dir: &Path, config: &serde_json::Value, tensors: &[(String, Vec<u64>)]) {
    write_hf_typed(dir, config, tensors, |_| None);
}

/// `write_hf` with a per-tensor override: `typed(name)` returns the
/// safetensors dtype and the one element's little-endian bytes every value
/// of that tensor holds (None = bf16 1.0, as `write_hf`).
fn write_hf_typed(
    dir: &Path,
    config: &serde_json::Value,
    tensors: &[(String, Vec<u64>)],
    typed: impl Fn(&str) -> Option<(&'static str, Vec<u8>)>,
) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    // 0x3f80 = bf16 1.0: a non-degenerate value for every weight
    let one = ("BF16", vec![0x80u8, 0x3f]);
    let mut header = serde_json::Map::new();
    let mut off = 0u64;
    let mut buf = Vec::new();
    for (name, shape) in tensors {
        let n: u64 = shape.iter().product();
        let (dtype, elem) = typed(name).unwrap_or_else(|| one.clone());
        let len = elem.len() as u64 * n;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [off, off + len]}),
        );
        off += len;
        for _ in 0..n {
            buf.extend_from_slice(&elem);
        }
    }
    let mut hb = serde_json::Value::Object(header).to_string().into_bytes();
    while hb.len() % 8 != 0 {
        hb.push(b' ');
    }
    let mut f = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hb).unwrap();
    f.write_all(&buf).unwrap();
}

fn t(names: &[(&str, &[u64])]) -> Vec<(String, Vec<u64>)> {
    names
        .iter()
        .map(|(n, s)| (n.to_string(), s.to_vec()))
        .collect()
}

const H: u64 = 64;
const FF: u64 = 128;
const V: u64 = 160;

fn dense_layer(p: &str, i: usize, qk_norm: bool) -> Vec<(String, Vec<u64>)> {
    let mut v = t(&[
        (
            &format!("{p}layers.{i}.self_attn.q_proj.weight"),
            &[2 * 32, H],
        ),
        (&format!("{p}layers.{i}.self_attn.k_proj.weight"), &[32, H]),
        (&format!("{p}layers.{i}.self_attn.v_proj.weight"), &[32, H]),
        (&format!("{p}layers.{i}.self_attn.o_proj.weight"), &[H, 64]),
        (&format!("{p}layers.{i}.mlp.gate_proj.weight"), &[FF, H]),
        (&format!("{p}layers.{i}.mlp.up_proj.weight"), &[FF, H]),
        (&format!("{p}layers.{i}.mlp.down_proj.weight"), &[H, FF]),
        (&format!("{p}layers.{i}.input_layernorm.weight"), &[H]),
        (
            &format!("{p}layers.{i}.post_attention_layernorm.weight"),
            &[H],
        ),
    ]);
    if qk_norm {
        v.extend(t(&[
            (&format!("{p}layers.{i}.self_attn.q_norm.weight"), &[32]),
            (&format!("{p}layers.{i}.self_attn.k_norm.weight"), &[32]),
        ]));
    }
    v
}

fn llama_tied(dir: &Path) {
    let cfg = serde_json::json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": H, "intermediate_size": FF, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 32, "vocab_size": V, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "max_position_embeddings": 512, "tie_word_embeddings": true,
        "bos_token_id": 1, "eos_token_id": 2});
    let mut ts = t(&[
        ("model.embed_tokens.weight", &[V, H]),
        ("model.norm.weight", &[H]),
    ]);
    for i in 0..2 {
        ts.extend(dense_layer("model.", i, false));
    }
    write_hf(dir, &cfg, &ts);
}

fn qwen35_dense(dir: &Path) {
    let (lkh, lvh, lkd, lvd, conv) = (2u64, 4u64, 16u64, 16u64, 4u64);
    let tc = serde_json::json!({"model_type": "qwen3_5_text", "hidden_size": H, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": V,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "intermediate_size": FF,
        "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
        "linear_conv_kernel_dim": conv, "linear_key_head_dim": lkd, "linear_num_key_heads": lkh,
        "linear_num_value_heads": lvh, "linear_value_head_dim": lvd, "attn_output_gate": true,
        "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": true,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": true}});
    let cfg = serde_json::json!({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": tc, "tie_word_embeddings": true, "bos_token_id": 1, "eos_token_id": 2});
    let p = "model.language_model.";
    let (kd, vd) = (lkh * lkd, lvh * lvd);
    let mut ts = t(&[
        (&format!("{p}embed_tokens.weight"), &[V, H]),
        (&format!("{p}norm.weight"), &[H]),
    ]);
    ts.extend(t(&[
        (
            &format!("{p}layers.0.linear_attn.in_proj_qkv.weight"),
            &[2 * kd + vd, H],
        ),
        (
            &format!("{p}layers.0.linear_attn.in_proj_z.weight"),
            &[vd, H],
        ),
        (
            &format!("{p}layers.0.linear_attn.in_proj_a.weight"),
            &[lvh, H],
        ),
        (
            &format!("{p}layers.0.linear_attn.in_proj_b.weight"),
            &[lvh, H],
        ),
        (
            &format!("{p}layers.0.linear_attn.out_proj.weight"),
            &[H, vd],
        ),
        (
            &format!("{p}layers.0.linear_attn.conv1d.weight"),
            &[2 * kd + vd, 1, conv],
        ),
        (&format!("{p}layers.0.linear_attn.norm.weight"), &[lvd]),
        (&format!("{p}layers.0.linear_attn.A_log"), &[lvh]),
        (&format!("{p}layers.0.linear_attn.dt_bias"), &[lvh]),
        (&format!("{p}layers.0.input_layernorm.weight"), &[H]),
        (
            &format!("{p}layers.0.post_attention_layernorm.weight"),
            &[H],
        ),
        (&format!("{p}layers.0.mlp.gate_proj.weight"), &[FF, H]),
        (&format!("{p}layers.0.mlp.up_proj.weight"), &[FF, H]),
        (&format!("{p}layers.0.mlp.down_proj.weight"), &[H, FF]),
    ]));
    ts.extend(dense_layer(p, 1, true));
    // a vision tower (mmproj) and the MTP head (speculator) with its fused fc
    ts.extend(t(&[
        ("model.visual.blocks.0.attn.qkv.weight", &[96, 32]),
        ("model.visual.blocks.0.attn.proj.weight", &[32, 32]),
        ("model.visual.blocks.0.norm1.weight", &[32]),
        ("model.visual.merger.linear_fc1.weight", &[64, 128]),
        ("mtp.fc.weight", &[H, 2 * H]),
        ("mtp.pre_fc_norm_embedding.weight", &[H]),
        ("mtp.pre_fc_norm_hidden.weight", &[H]),
        ("mtp.norm.weight", &[H]),
    ]));
    ts.extend(dense_layer("mtp.", 0, true));
    write_hf(dir, &cfg, &ts);
}

fn glm_dsa(dir: &Path) {
    let (qlr, kvlr, nope, rope, vhd, nh) = (32u64, 16u64, 16u64, 8u64, 16u64, 2u64);
    let (ih, ihd, e, mf) = (2u64, 16u64, 4u64, 32u64);
    let cfg = serde_json::json!({"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": H, "num_hidden_layers": 2, "num_attention_heads": nh, "num_key_value_heads": nh,
        "vocab_size": V, "intermediate_size": FF, "moe_intermediate_size": mf, "n_routed_experts": e,
        "num_experts_per_tok": 2, "n_shared_experts": 1, "n_group": 1, "topk_group": 1,
        "routed_scaling_factor": 1.0, "scoring_func": "sigmoid", "norm_topk_prob": true,
        "first_k_dense_replace": 1, "q_lora_rank": qlr, "kv_lora_rank": kvlr, "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope, "v_head_dim": vhd, "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0}, "max_position_embeddings": 512,
        "tie_word_embeddings": false, "eos_token_id": [2], "bos_token_id": 1,
        "index_n_heads": ih, "index_head_dim": ihd, "index_topk": 64, "indexer_types": ["full", "sparse"]});
    let mut ts = t(&[
        ("model.embed_tokens.weight", &[V, H]),
        ("model.norm.weight", &[H]),
        ("lm_head.weight", &[V, H]),
    ]);
    for i in 0..2 {
        let p = format!("model.layers.{i}.");
        ts.extend(t(&[
            (&format!("{p}input_layernorm.weight"), &[H]),
            (&format!("{p}post_attention_layernorm.weight"), &[H]),
            (&format!("{p}self_attn.q_a_proj.weight"), &[qlr, H]),
            (&format!("{p}self_attn.q_a_layernorm.weight"), &[qlr]),
            (
                &format!("{p}self_attn.q_b_proj.weight"),
                &[nh * (nope + rope), qlr],
            ),
            (
                &format!("{p}self_attn.kv_a_proj_with_mqa.weight"),
                &[kvlr + rope, H],
            ),
            (&format!("{p}self_attn.kv_a_layernorm.weight"), &[kvlr]),
            (
                &format!("{p}self_attn.kv_b_proj.weight"),
                &[nh * (nope + vhd), kvlr],
            ),
            (&format!("{p}self_attn.o_proj.weight"), &[H, nh * vhd]),
            (
                &format!("{p}self_attn.indexer.wq_b.weight"),
                &[ih * ihd, qlr],
            ),
            (&format!("{p}self_attn.indexer.wk.weight"), &[ihd, H]),
            (
                &format!("{p}self_attn.indexer.weights_proj.weight"),
                &[ih, H],
            ),
            (&format!("{p}self_attn.indexer.k_norm.weight"), &[ihd]),
            (&format!("{p}self_attn.indexer.k_norm.bias"), &[ihd]),
        ]));
        if i == 0 {
            ts.extend(t(&[
                (&format!("{p}mlp.gate_proj.weight"), &[FF, H]),
                (&format!("{p}mlp.up_proj.weight"), &[FF, H]),
                (&format!("{p}mlp.down_proj.weight"), &[H, FF]),
            ]));
        } else {
            ts.extend(t(&[
                (&format!("{p}mlp.gate.weight"), &[e, H]),
                (&format!("{p}mlp.gate.e_score_correction_bias"), &[e]),
                (&format!("{p}mlp.switch_mlp.gate_proj.weight"), &[e, mf, H]),
                (&format!("{p}mlp.switch_mlp.up_proj.weight"), &[e, mf, H]),
                (&format!("{p}mlp.switch_mlp.down_proj.weight"), &[e, H, mf]),
                (&format!("{p}mlp.shared_experts.gate_proj.weight"), &[FF, H]),
                (&format!("{p}mlp.shared_experts.up_proj.weight"), &[FF, H]),
                (&format!("{p}mlp.shared_experts.down_proj.weight"), &[H, FF]),
            ]));
        }
    }
    write_hf(dir, &cfg, &ts);
}

/// Qwen3.8-Flash-Next with one PLE layer whose n-gram table arrives as the
/// row shards `(index, rows)` and/or `whole` as one table of that shape.
/// ngram_size 2 x heads_per_ngram 1 is one hash head; the first prime
/// from ngram_vocab_size_base 100 is 101, padded to a multiple of 16: the
/// table's `ngram_padded_vocab` is 112 rows.
fn qwen4_exp_ngram(dir: &Path, shards: &[(usize, u64)], whole: Option<&[u64]>) {
    qwen4_exp_ngram_typed(dir, shards, whole, |_| None);
}

/// `qwen4_exp_ngram` with `write_hf_typed`'s per-tensor dtype override.
fn qwen4_exp_ngram_typed(
    dir: &Path,
    shards: &[(usize, u64)],
    whole: Option<&[u64]>,
    typed: impl Fn(&str) -> Option<(&'static str, Vec<u8>)>,
) {
    let (lkh, lvh, lkd, lvd, conv) = (2u64, 4u64, 16u64, 16u64, 4u64);
    let tc = serde_json::json!({"model_type": "qwen4_exp_text", "hidden_size": H, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "vocab_size": V,
        "rms_norm_eps": 1e-6, "max_position_embeddings": 512, "intermediate_size": FF,
        "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
        "linear_conv_kernel_dim": conv, "linear_key_head_dim": lkd, "linear_num_key_heads": lkh,
        "linear_num_value_heads": lvh, "linear_value_head_dim": lvd, "attn_output_gate": true,
        "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": true,
        "ple_layer_ids": [0], "ple_embed_dim": 32, "ngram_size": 2, "heads_per_ngram": 1,
        "ngram_vocab_size_base": 100, "make_ngram_vocab_size_divisible_by": 16, "seed": 1,
        "split_ngram_parts": 2,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": true}});
    let cfg = serde_json::json!({"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": tc, "tie_word_embeddings": true, "bos_token_id": 1, "eos_token_id": 2});
    let p = "model.language_model.";
    let mut ts = t(&[
        (&format!("{p}embed_tokens.weight"), &[V, H]),
        (&format!("{p}norm.weight"), &[H]),
    ]);
    for (e, rows) in shards {
        ts.push((
            format!("{p}layers.0.ple.ple_embedding.ngram_embedding.shard_{e}.weight"),
            vec![*rows, 32],
        ));
    }
    if let Some(shape) = whole {
        ts.push((
            format!("{p}layers.0.ple.ple_embedding.ngram_embedding.weight"),
            shape.to_vec(),
        ));
    }
    write_hf_typed(dir, &cfg, &ts, typed);
}

fn run(mut c: Command) -> String {
    let out = c.output().unwrap();
    assert!(
        out.status.success(),
        "{:?} failed:\n{}",
        c,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Runs `c` expecting failure; returns its stderr.
fn fails(mut c: Command) -> String {
    let out = c.output().unwrap();
    assert!(
        !out.status.success(),
        "{:?} succeeded:\n{}",
        c,
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

const NGRAM_TABLE: &str = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight";

/// `inventory` and `convert --target bf16` of `dir` both fail, each naming
/// every `wants` fragment.
fn ngram_rejected(dir: &Path, wants: &[&str]) {
    let mut inv = basert();
    inv.arg("inventory")
        .arg(dir)
        .arg("--out")
        .arg(dir.with_extension("inventory.json"));
    let mut conv = basert();
    conv.arg("convert")
        .arg(dir)
        .arg("-o")
        .arg(dir.with_extension("base"))
        .arg("--target")
        .arg("bf16");
    for err in [fails(inv), fails(conv)] {
        for w in wants {
            assert!(err.contains(w), "missing {w:?} in:\n{err}");
        }
    }
}

/// `inventory` of `dir` lists the one n-gram table as a [112, 32] embedding.
fn ngram_listed(dir: &Path) {
    let inv_path = dir.with_extension("inventory.json");
    let mut c = basert();
    c.arg("inventory").arg(dir).arg("--out").arg(&inv_path);
    run(c);
    let inv: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&inv_path).unwrap()).unwrap();
    let table = tensor(&inv, "layers.0.ple.ple_embedding.ngram_embedding.weight");
    assert_eq!(table["shape"], serde_json::json!([112, 32]));
    assert_eq!(table["class"], "embedding");
}

fn bundle_names(path: &Path) -> BTreeSet<String> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[16..16 + n]).unwrap();
    let mut names = BTreeSet::new();
    let mut take = |v: Option<&serde_json::Value>| {
        if let Some(ts) = v.and_then(|v| v.as_array()) {
            for t in ts {
                names.insert(t["name"].as_str().unwrap().to_string());
            }
        }
    };
    take(hdr.get("tensors"));
    take(hdr.get("mmproj").and_then(|m| m.get("tensors")));
    take(hdr.get("speculator").and_then(|m| m.get("tensors")));
    names
}

/// Inventory + bf16 convert of one checkpoint; returns the inventory.
fn check(dir: &Path, bundle: &Path) -> serde_json::Value {
    let inv_path = dir.with_extension("inventory.json");
    let mut c = basert();
    c.arg("inventory").arg(dir).arg("--out").arg(&inv_path);
    run(c);
    let inv: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&inv_path).unwrap()).unwrap();
    assert_eq!(inv["schema"], 3);
    let classes = ["linear", "expert_stack", "embedding", "other"];
    let mut class_of = std::collections::BTreeMap::new();
    for t in inv["tensors"].as_array().unwrap() {
        let c = t["class"].as_str().unwrap();
        assert!(classes.contains(&c), "{t}");
        assert!(
            ["main", "mmproj", "speculator"].contains(&t["section"].as_str().unwrap()),
            "{t}"
        );
        class_of.insert(t["canonical"].as_str().unwrap().to_string(), c.to_string());
    }
    for g in inv["fold_groups"].as_array().unwrap() {
        for c in g["consumers"].as_array().unwrap() {
            let k = &class_of[c.as_str().unwrap()];
            assert!(k == "linear" || k == "expert_stack", "{g}");
        }
        match g["fold"].as_str().unwrap() {
            "gamma" => assert!(g["gamma_shift"] == 0 || g["gamma_shift"] == 1, "{g}"),
            "rows" => assert!(
                g["upstream_rows"] == "all" || g["upstream_rows"] == "second_half",
                "{g}"
            ),
            other => panic!("fold {other}"),
        }
        // the tiling the probe applies: null, a GQA triple or a period; a refused group says why
        let tile = &g["tile"];
        assert!(
            tile.is_null()
                || tile["gqa"].as_array().is_some_and(|a| a.len() == 3)
                || tile["period"].as_u64().is_some(),
            "{g}"
        );
        if g["foldable"] == false {
            assert!(g["refusal"].as_str().is_some_and(|r| !r.is_empty()), "{g}");
        } else {
            assert!(g.get("refusal").is_none(), "{g}");
        }
    }
    let mut c = basert();
    c.arg("convert")
        .arg(dir)
        .arg("-o")
        .arg(bundle)
        .arg("--target")
        .arg("bf16");
    run(c);
    let inv_names: BTreeSet<String> = class_of.keys().cloned().collect();
    let names = bundle_names(bundle);
    assert_eq!(
        inv_names, names,
        "inventory canonical names must be the bundle's tensor names\n  inventory-only: {:?}\n  bundle-only: {:?}",
        inv_names.difference(&names).collect::<Vec<_>>(),
        names.difference(&inv_names).collect::<Vec<_>>()
    );
    inv
}

fn tensor<'a>(inv: &'a serde_json::Value, canonical: &str) -> &'a serde_json::Value {
    inv["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["canonical"] == canonical)
        .unwrap_or_else(|| panic!("{canonical} not in the inventory"))
}

#[test]
fn llama_tied_lists_the_head_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("llama");
    llama_tied(&dir);
    let inv = check(&dir, &tmp.path().join("llama.base"));
    let head = tensor(&inv, "lm_head.weight");
    assert_eq!(head["class"], "linear");
    assert_eq!(head["source"], "model.embed_tokens.weight");
    assert_eq!(tensor(&inv, "embed_tokens.weight")["class"], "embedding");
    assert_eq!(
        tensor(&inv, "layers.0.self_attn.k_proj.weight")["class"],
        "linear"
    );
    assert_eq!(tensor(&inv, "layers.0.input_norm.weight")["class"], "other");
    let kinds: BTreeSet<&str> = inv["fold_groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        ["attn", "down", "mlp", "o"].into_iter().collect(),
        "tied: no head group"
    );
}

#[test]
fn qwen35_lists_every_section_and_the_split_mtp_fc() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("qwen35");
    qwen35_dense(&dir);
    let inv = check(&dir, &tmp.path().join("qwen35.base"));
    assert_eq!(tensor(&inv, "lm_head.weight")["section"], "main");
    assert_eq!(
        tensor(&inv, "layers.0.linear_attn.in_proj_a.weight")["class"],
        "linear"
    );
    assert_eq!(
        tensor(&inv, "layers.0.linear_attn.conv1d.weight")["class"],
        "other"
    );
    for (n, class) in [
        ("mtp.fc_embed.weight", "linear"),
        ("mtp.fc_hidden.weight", "linear"),
        ("mtp.layers.0.self_attn.q_proj.weight", "linear"),
        ("mtp.pre_fc_norm_hidden.weight", "other"),
    ] {
        let t = tensor(&inv, n);
        assert_eq!(
            (t["section"].as_str().unwrap(), t["class"].as_str().unwrap()),
            ("speculator", class),
            "{n}"
        );
    }
    for (n, class) in [
        ("visual.blocks.0.attn.qkv.weight", "linear"),
        ("visual.blocks.0.norm1.weight", "other"),
    ] {
        let t = tensor(&inv, n);
        assert_eq!(
            (t["section"].as_str().unwrap(), t["class"].as_str().unwrap()),
            ("mmproj", class),
            "{n}"
        );
    }
    let gdn: Vec<&serde_json::Value> = inv["fold_groups"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|g| g["kind"] == "gdn_out")
        .collect();
    assert_eq!(gdn.len(), 1);
    assert_eq!(gdn[0]["gamma_shift"], 0);
    // the lvd-wide gated-norm gain in front of out_proj's lvh x lvd columns: a periodic tile
    assert_eq!(gdn[0]["foldable"], true, "{}", gdn[0]);
    assert_eq!(
        gdn[0]["tile"],
        serde_json::json!({"period": 16}),
        "{}",
        gdn[0]
    );
    let o = inv["fold_groups"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["kind"] == "o")
        .expect("the full-attention layer declares an o fold");
    assert_eq!(o["tile"], serde_json::json!({"gqa": [2, 1, 32]}), "{o}");
    let attn0 = inv["fold_groups"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["kind"] == "attn" && g["layer"] == 0)
        .unwrap();
    assert_eq!(
        attn0["consumers"].as_array().unwrap().len(),
        4,
        "the GDN attn group lists in_proj_a/b too"
    );
    assert_eq!(attn0["gamma_shift"], 1);
    assert!(attn0["tile"].is_null(), "{attn0}");
}

#[test]
fn glm_lists_the_split_kv_b_projections() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("glm");
    glm_dsa(&dir);
    let inv = check(&dir, &tmp.path().join("glm.base"));
    assert_eq!(
        tensor(&inv, "layers.0.self_attn.k_b_proj.weight")["class"],
        "other"
    );
    assert_eq!(
        tensor(&inv, "layers.0.self_attn.v_b_proj.weight")["class"],
        "other"
    );
    assert!(inv["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .all(|t| t["canonical"] != "layers.0.self_attn.kv_b_proj.weight"));
    assert_eq!(
        tensor(&inv, "layers.1.mlp.experts.up_proj.weight")["class"],
        "expert_stack"
    );
    assert_eq!(tensor(&inv, "layers.1.mlp.router.weight")["class"], "other");
    let md = inv["fold_groups"]
        .as_array()
        .unwrap()
        .iter()
        .find(|g| g["kind"] == "moe_down")
        .unwrap();
    assert_eq!(md["upstream"], "layers.1.mlp.experts.up_proj.weight");
    assert_eq!(md["upstream_rows"], "all");
}

/// Shards 0 and 2 without shard 1 are not a table: concatenating them
/// would put shard 2's rows at shard 1's ids under a plausible shape.
#[test]
fn qwen4_exp_ngram_shards_must_be_shard_0_to_split_ngram_parts() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("missing_middle");
    qwen4_exp_ngram(&dir, &[(0, 64), (2, 48)], None);
    ngram_rejected(
        &dir,
        &[
            NGRAM_TABLE,
            "split_ngram_parts 2",
            "missing [1]",
            "unexpected [2]",
        ],
    );
    let dir = tmp.path().join("extra_shard");
    qwen4_exp_ngram(&dir, &[(0, 32), (1, 32), (2, 48)], None);
    ngram_rejected(&dir, &[NGRAM_TABLE, "missing []", "unexpected [2]"]);
}

/// A complete shard sequence whose rows do not add up to the layer's
/// `ngram_padded_vocab` is refused too; one that does lists as one table.
#[test]
fn qwen4_exp_ngram_rows_must_equal_the_padded_vocab() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("short");
    qwen4_exp_ngram(&dir, &[(0, 64), (1, 32)], None);
    ngram_rejected(
        &dir,
        &[
            NGRAM_TABLE,
            "2 n-gram shards hold 96 rows",
            "ngram_padded_vocab is 112",
        ],
    );
    let dir = tmp.path().join("complete");
    qwen4_exp_ngram(&dir, &[(0, 64), (1, 48)], None);
    ngram_listed(&dir);
}

/// Every configured PLE layer needs its table, sharded or whole — a bundle
/// must not advertise PLE without the weights — and a whole table's rows
/// are checked like a stacked one's; a layer never has both forms.
#[test]
fn qwen4_exp_ngram_needs_one_table_per_ple_layer() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("no_table");
    qwen4_exp_ngram(&dir, &[], None);
    ngram_rejected(
        &dir,
        &["0 n-gram table(s) (`…ple.ple_embedding.ngram_embedding.weight`, sharded or whole) for 1 PLE layer(s)"],
    );
    let dir = tmp.path().join("whole");
    qwen4_exp_ngram(&dir, &[], Some(&[112, 32]));
    ngram_listed(&dir);
    let dir = tmp.path().join("whole_short");
    qwen4_exp_ngram(&dir, &[], Some(&[96, 32]));
    ngram_rejected(
        &dir,
        &[
            NGRAM_TABLE,
            "the n-gram table holds 96 rows",
            "ngram_padded_vocab is 112",
        ],
    );
    let dir = tmp.path().join("whole_and_shards");
    qwen4_exp_ngram(&dir, &[(0, 64), (1, 48)], Some(&[112, 32]));
    ngram_rejected(&dir, &[NGRAM_TABLE, "both whole and as shards"]);
    // A whole table must be [rows, dim]: a 1-D tensor with the right row
    // count would otherwise be written with one value per row, and a scalar
    // has no row count to read at all.
    let dir = tmp.path().join("whole_1d");
    qwen4_exp_ngram(&dir, &[], Some(&[112]));
    ngram_rejected(&dir, &[NGRAM_TABLE, "is not 2-D", "shape [112]"]);
    let dir = tmp.path().join("whole_scalar");
    qwen4_exp_ngram(&dir, &[], Some(&[]));
    ngram_rejected(&dir, &[NGRAM_TABLE, "is not 2-D", "shape []"]);
}

/// The n-gram table of the bundle `convert --target bf16` writes from
/// `dir`: its dtype and bytes.
fn converted_ngram(dir: &Path) -> (base_format::TensorDtype, Vec<u8>) {
    let out = dir.with_extension("base");
    let mut c = basert();
    c.arg("convert")
        .arg(dir)
        .arg("-o")
        .arg(&out)
        .arg("--target")
        .arg("bf16");
    run(c);
    let r = base_format::BaseReader::open(&out).unwrap();
    let e = r
        .all_entries()
        .find(|e| e.name.ends_with("ngram_embedding.weight"))
        .expect("the bundle has the n-gram table")
        .clone();
    (e.dtype, r.tensor_bytes(&e.name).unwrap().to_vec())
}

/// `--target bf16` carries a sharded n-gram table losslessly whatever the
/// shards' dtype, exactly as it carries the same table stored whole: the
/// narrowest of f16 / bf16 / f32 that holds every value of every shard.
/// Only bf16 shards were accepted before (their bytes concatenated); f16 or
/// f32 shards failed with "n-gram shards are not plain bf16 tensors".
#[test]
fn qwen4_exp_ngram_shards_carry_like_a_whole_table_at_bf16() {
    use base_format::TensorDtype;
    let tmp = tempfile::tempdir().unwrap();
    let is_ngram = |n: &str| n.contains("ngram_embedding");
    let fill = |n: usize, elem: &[u8]| elem.repeat(n);
    let (rows0, rows1, dim) = (64usize, 48usize, 32usize);

    // f16 shards of 1.0: f16 as the whole table is.
    let f16_one = |n: &str| is_ngram(n).then(|| ("F16", 0x3c00u16.to_le_bytes().to_vec()));
    let whole = tmp.path().join("f16_whole");
    qwen4_exp_ngram_typed(&whole, &[], Some(&[112, 32]), f16_one);
    let sharded = tmp.path().join("f16_shards");
    qwen4_exp_ngram_typed(&sharded, &[(0, 64), (1, 48)], None, f16_one);
    let (wd, wb) = converted_ngram(&whole);
    assert_eq!(wd, TensorDtype::F16);
    assert_eq!(converted_ngram(&sharded), (wd, wb));

    // f32 shards, both on the f16 grid (1.0): f16 as the whole table is.
    let f32_one = |n: &str| is_ngram(n).then(|| ("F32", 1.0f32.to_le_bytes().to_vec()));
    let whole = tmp.path().join("f32_whole");
    qwen4_exp_ngram_typed(&whole, &[], Some(&[112, 32]), f32_one);
    let sharded = tmp.path().join("f32_shards");
    qwen4_exp_ngram_typed(&sharded, &[(0, 64), (1, 48)], None, f32_one);
    let (wd, wb) = converted_ngram(&whole);
    assert_eq!(wd, TensorDtype::F16);
    assert_eq!(converted_ngram(&sharded), (wd, wb));

    // Shard 0 fits f16, shard 1 (65536 > f16 max) only bf16: the whole table
    // is bf16, shard 0's already-packed rows re-encoded.
    let split = |v1: f32| {
        move |n: &str| {
            if n.ends_with("shard_0.weight") {
                Some(("F32", 1.0f32.to_le_bytes().to_vec()))
            } else if n.ends_with("shard_1.weight") {
                Some(("F32", v1.to_le_bytes().to_vec()))
            } else {
                None
            }
        }
    };
    let dir = tmp.path().join("f16_then_bf16");
    qwen4_exp_ngram_typed(&dir, &[(0, 64), (1, 48)], None, split(65536.0));
    let mut want = fill(rows0 * dim, &0x3f80u16.to_le_bytes());
    want.extend(fill(rows1 * dim, &0x4780u16.to_le_bytes()));
    assert_eq!(converted_ngram(&dir), (TensorDtype::Bf16, want));

    // Shard 1 on neither grid (1 + 2^-23): f32 for the whole table.
    let v1 = f32::from_bits(0x3f80_0001);
    let dir = tmp.path().join("f16_then_f32");
    qwen4_exp_ngram_typed(&dir, &[(0, 64), (1, 48)], None, split(v1));
    let mut want = fill(rows0 * dim, &1.0f32.to_le_bytes());
    want.extend(fill(rows1 * dim, &v1.to_le_bytes()));
    assert_eq!(converted_ngram(&dir), (TensorDtype::F32, want));
}
