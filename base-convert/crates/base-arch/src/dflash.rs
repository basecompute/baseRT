//! Block drafters (DFlash / DSpark) — speculative-decoding drafters that
//! ship as their own HF checkpoints and convert to a SIDECAR `.base` bundle
//! (arch `dflash` / `dspark`) the runtime loads against its target
//! (`baseRT_load_drafter`).
//!
//! A block drafter is a short qwen3-shaped decoder (5-6 layers of the
//! target's width) plus a context projection: `fc` ([dim, n_taps * dim])
//! over the concatenation of the target's hidden states at
//! `target_layer_ids` (hidden_states[id + 1]), then `hidden_norm`. The
//! checkpoint's `model_type` is the backbone's (`qwen3`), so the drafter
//! is recognised by its `architectures` entry (`DFlashDraftModel`,
//! `DFlash2DraftModel`, `Qwen3DSparkModel`) — converting it as a Qwen3 would
//! drop `fc` / `hidden_norm` and write a broken LM.
//!
//! Tensor policy: every tensor stays f16 (the drafter's acceptance rate is
//! the whole point of the scheme, and the runtime's block step is bound by
//! the target's verify anyway). `fc.weight` is split by input column into
//! one `fc.{i}.weight` per tap (base-convert's `FcSplitProvider`), so the
//! runtime runs one GEMM per tap block of the tap-major input.
//!
//! Header config: the drafter's own geometry in the HF-style keys the
//! runtime parses (hidden_size, num_hidden_layers, heads, head_dim,
//! intermediate_size, rms_norm_eps, rope_theta, vocab_size, sliding
//! window as a period) plus the drafting fields under
//! [`ArchConfig::extra_config`]: `speculator_kind`, `block_size`,
//! `mask_token_id`, `logits_start`, `target_layer_ids`,
//! `num_target_layers`, `markov_rank`, `confidence_head`, and DFlash 2's
//! `conv_kernel_size` / `conv_group_size` / `selector_rank` /
//! `selector_top_k` when present.

use crate::{ArchConfig, HfMapper, ValueTransform};
use anyhow::{anyhow, bail, Result};

/// Scale folded into every `fc.{i}.weight` (see [`ValueTransform::Scale`]):
/// the fc output feeds only `hidden_norm`, so the drafter's math is
/// unchanged while the f16 output stays far from overflow. Recorded in the
/// header as `fc_scale`.
pub const FC_SCALE_DIV: u32 = 256;

fn fc_transform(canonical: &str) -> Option<ValueTransform> {
    let rest = canonical.strip_prefix("fc.")?;
    let (idx, tail) = rest.split_once('.')?;
    (tail == "weight" && idx.parse::<u32>().is_ok()).then_some(ValueTransform::DivBy(FC_SCALE_DIV))
}

pub struct DflashHfMapper;
pub struct DsparkHfMapper;

/// Canonical arch strings of the block-drafter family.
pub fn is_block_drafter_arch(arch: &str) -> bool {
    arch == "dflash" || arch == "dspark"
}

/// Pick a drafter mapper from `config.json`'s `architectures`; None for an
/// ordinary LM checkpoint (which routes by `model_type` as before).
pub fn hf_mapper_for_architectures(config: &serde_json::Value) -> Option<&'static dyn HfMapper> {
    let archs = config.get("architectures")?.as_array()?;
    for a in archs.iter().filter_map(|v| v.as_str()) {
        if a.starts_with("DFlash") {
            return Some(&DflashHfMapper);
        }
        if a.contains("DSpark") {
            return Some(&DsparkHfMapper);
        }
    }
    None
}

fn u64_at(c: &serde_json::Value, dc: Option<&serde_json::Value>, key: &str) -> Option<u64> {
    dc.and_then(|d| d.get(key))
        .or_else(|| c.get(key))
        .and_then(|v| v.as_u64())
}

/// Shared config extraction. DFlash nests its drafting fields under
/// `dflash_config`; DSpark keeps them at the top level. Either way the
/// backbone fields are the standard HF ones.
/// RedHat's `speculators` packaging of a block drafter
/// (`speculators_model_type` "dflash" / "dflash2" / "dspark"): the layer config nested
/// under `transformer_layer_config`, the drafting fields at the top level,
/// taps as `aux_hidden_state_layer_ids` in hidden_states indices (z-lab's
/// `target_layer_ids` + 1), causality of the sliding layers stated as
/// `sliding_window_non_causal`, and `sample_from_anchor` choosing whether the
/// anchor row drafts. Flattened into the z-lab form read below.
fn flatten_speculators_block(c: &serde_json::Value) -> Result<serde_json::Value> {
    let mut flat = c
        .get("transformer_layer_config")
        .cloned()
        .ok_or_else(|| anyhow!("speculators config without transformer_layer_config"))?;
    let o = flat
        .as_object_mut()
        .ok_or_else(|| anyhow!("transformer_layer_config is not an object"))?;
    for k in [
        "architectures",
        "block_size",
        "mask_token_id",
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
    ] {
        if let Some(v) = c.get(k).filter(|v| !v.is_null()) {
            o.insert(k.to_string(), v.clone());
        }
    }
    // A reduced draft vocabulary (RedHat's Qwen3-30B-A3B / Nemotron DFlash:
    // 32000 rows of lm_head plus a `d2t` offset table back to target ids)
    // is carried through; drafter_config_from_hf decides which kinds take it.
    let vocab = o.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0);
    if let Some(dv) = c.get("draft_vocab_size").and_then(|v| v.as_u64()) {
        if dv == 0 || dv > vocab {
            bail!("draft_vocab_size {dv} is outside 1..={vocab}");
        }
        if dv < vocab {
            o.insert("draft_vocab_size".into(), serde_json::json!(dv));
        }
    }
    let ids: Vec<i64> = c
        .get("aux_hidden_state_layer_ids")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default();
    if ids.iter().any(|&i| i < 1) {
        bail!("speculators aux_hidden_state_layer_ids {ids:?} name the embedding output; not supported");
    }
    o.insert(
        "target_layer_ids".into(),
        serde_json::json!(ids.iter().map(|i| i - 1).collect::<Vec<_>>()),
    );
    // Sliding layers attend causally unless the packaging says otherwise;
    // full layers are bidirectional either way (the z-lab rule, stated
    // explicitly here).
    if let Some(nc) = c.get("sliding_window_non_causal").and_then(|v| v.as_bool()) {
        if nc {
            o.insert("is_causal".into(), serde_json::json!(false));
        }
    }
    if let Some(a) = c.get("sample_from_anchor").and_then(|v| v.as_bool()) {
        o.insert(
            "logits_start".into(),
            serde_json::json!(if a { 0 } else { 1 }),
        );
    }
    Ok(flat)
}

fn drafter_config_from_hf(c: &serde_json::Value, kind: &str) -> Result<ArchConfig> {
    let flat;
    let speculators = c
        .get("speculators_model_type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| t.starts_with("dflash") || t == "dspark");
    let c = if speculators {
        flat = flatten_speculators_block(c)?;
        &flat
    } else {
        c
    };
    let backbone = c.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    // Gemma 4 backbone (deepseek-ai/dspark_gemma4_*: `Gemma4DSparkModel`,
    // model_type gemma4_text): the Gemma 4 layer config, recorded as
    // `drafter_backbone` so the runtime runs the block rows through its
    // Gemma 4 encoder and the context pass in Gemma 4 form.
    let gemma4 = backbone.starts_with("gemma4");
    // RedHat's speculators DFlash heads for Qwen3-30B-A3B and Nemotron say
    // `llama` but carry Qwen3's per-head q/k norms. The label alone cannot
    // tell those from a genuinely norm-free Llama backbone, which the runtime
    // cannot run (it loads every dflash / dspark sidecar as Qwen3 and demands
    // q/k norms on every layer), so the tensors decide:
    // [`check_block_drafter_qk_norms`] refuses a sidecar without them.
    if backbone != "qwen3" && backbone != "llama" && !gemma4 {
        bail!(
            "block drafter backbone model_type {backbone:?} is not qwen3, llama or gemma4 — the engine runs \
             llama/qwen3- and gemma4-shaped drafters only"
        );
    }
    if gemma4 && kind != "dspark" {
        bail!("a gemma4-backbone block drafter is wired for DSpark only");
    }
    let mut cfg = if gemma4 {
        use crate::HfMapper;
        crate::gemma::Gemma4HfMapper.config_from_hf(c)?
    } else {
        crate::llama::hf_generic_config(c)?
    };
    // The logit head is the target's (DFlash) or the drafter's own
    // `lm_head.weight` (DSpark): never synthesize a tied copy.
    cfg.tie_word_embeddings = false;

    let dc = c.get("dflash_config");
    let target_layer_ids: Vec<u64> = dc
        .and_then(|d| d.get("target_layer_ids"))
        .or_else(|| c.get("target_layer_ids"))
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        .unwrap_or_default();
    if target_layer_ids.is_empty() {
        bail!("config.json: target_layer_ids missing (dflash_config or top level) — the drafter's taps");
    }
    if target_layer_ids.len() > 8 {
        bail!(
            "config.json: {} target_layer_ids; the engine taps at most 8 layers per forward",
            target_layer_ids.len()
        );
    }
    // Two DSpark packagings. DeepSeek's DeepSpec (`Qwen3DSparkModel`) ships its
    // own embed / lm_head and predicts from block row 0 (anchor as position
    // 0). SpecForge's (`DSparkDraftModel`, `projector_type: "dspark"`; e.g.
    // RadixArk/Qwen3.8-27B-DSpark) is a DFlash backbone: drafts from rows 1..
    // (`draft_hidden[:, -block_size + 1:]`) through the TARGET's lm_head.
    let archs: Vec<&str> = c
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    // RedHat's speculators DSpark reuses the `DSparkDraftModel` name but ships
    // its own embed / lm_head (DeepSpec-style), so it is not SpecForge's.
    let specforge_dspark = kind == "dspark" && archs.contains(&"DSparkDraftModel") && !speculators;
    let ds_cfg = c.get("dspark_config");
    let markov_type = ds_cfg
        .and_then(|d| d.get("markov_head_type"))
        .or_else(|| dc.and_then(|d| d.get("markov_head_type")))
        .or_else(|| c.get("markov_head_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("vanilla");
    if kind == "dspark" && markov_type != "vanilla" {
        bail!("DSpark markov_head_type {markov_type:?} is not implemented (vanilla only)");
    }
    let default_block = if kind == "dspark" { 7 } else { 16 };
    let block_size = u64_at(c, dc, "block_size").unwrap_or(default_block);
    if !(2..=16).contains(&block_size) {
        bail!("config.json: block_size {block_size} outside 2..16 (the verify feed width)");
    }
    let mask_token_id = u64_at(c, dc, "mask_token_id").ok_or_else(|| {
        anyhow!("config.json: mask_token_id missing — the token the draft block is padded with")
    })?;
    // DFlash reserves block row 0 as the anchor and drafts from the mask
    // rows (logits_start 1); DeepSpec-native DSpark heads predict from
    // row 0 (anchor-as-position-0). A checkpoint may say so explicitly.
    let logits_start =
        u64_at(c, dc, "logits_start").unwrap_or(if kind == "dspark" && !specforge_dspark {
            0
        } else {
            1
        });
    let num_target_layers = u64_at(c, dc, "num_target_layers").unwrap_or(0);
    let markov_rank = c
        .get("markov_rank")
        .or_else(|| ds_cfg.and_then(|d| d.get("markov_rank")))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let confidence_head = c
        .get("enable_confidence_head")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Sliding-window schedule → the runtime's period form (layer i is
    // global iff (i + 1) % pattern == 0). [S,S,S,S,S,F] → 6; all sliding →
    // n_layers + 1 (never global).
    let layer_types: Vec<String> = c
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let window = c
        .get("sliding_window")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let use_sliding = c
        .get("use_sliding_window")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let sliding: Vec<bool> = layer_types
        .iter()
        .map(|t| t == "sliding_attention")
        .collect();
    if use_sliding && window > 0 && sliding.iter().any(|&s| s) {
        let n = sliding.len();
        let pattern = (1..=n + 1).find(|&p| (0..n).all(|i| sliding[i] == ((i + 1) % p != 0)));
        match pattern {
            Some(p) => {
                cfg.sliding_window = window;
                cfg.sliding_window_pattern = p as u32;
            }
            None => bail!(
                "config.json: irregular sliding-window schedule {layer_types:?} — the runtime expresses windows \
                 as a period"
            ),
        }
    }

    use serde_json::json;
    let x = &mut cfg.extra_config;
    x.insert("speculator_kind".into(), json!(kind));
    x.insert("block_size".into(), json!(block_size));
    x.insert("mask_token_id".into(), json!(mask_token_id));
    x.insert("logits_start".into(), json!(logits_start));
    x.insert("target_layer_ids".into(), json!(target_layer_ids));
    x.insert("fc_scale".into(), json!(1.0 / FC_SCALE_DIV as f64));
    if num_target_layers > 0 {
        x.insert("num_target_layers".into(), json!(num_target_layers));
    }
    if markov_rank > 0 {
        x.insert("markov_rank".into(), json!(markov_rank));
    }
    if confidence_head {
        x.insert("confidence_head".into(), json!(true));
    }
    if specforge_dspark {
        // The runtime then lets the logit head fall through to the target's.
        x.insert("drafter_head".into(), json!("target"));
    }
    if gemma4 {
        x.insert("drafter_backbone".into(), json!("gemma4"));
    }
    if !layer_types.is_empty() {
        x.insert("drafter_layer_types".into(), json!(layer_types));
    }
    // Which block layers attend causally (z-lab/dflash Qwen3DFlashAttention):
    // an explicit `is_causal` applies to every layer; absent, a
    // `sliding_attention` layer is causal and a full one bidirectional. The
    // sliding-window DFlash drafters (Qwen3.6-27B, Qwen3.5-35B-A3B) were
    // trained that way, and running their sliding layers bidirectionally lets
    // every block row read the rows after it. DSpark's reference attends the
    // block bidirectionally throughout.
    if kind == "dflash" {
        let n = cfg.num_hidden_layers as usize;
        let causal: Vec<bool> = match c.get("is_causal").and_then(|v| v.as_bool()) {
            Some(all) => vec![all; n],
            None => (0..n)
                .map(|i| layer_types.get(i).is_some_and(|t| t == "sliding_attention"))
                .collect(),
        };
        // Always written for DFlash, all-false included: a bundle WITHOUT the
        // key predates it, and the loader then applies the sliding rule
        // itself — so the key's absence must never mean "all bidirectional".
        x.insert("drafter_causal_layers".into(), json!(causal));
    }
    // Logit / embedding scalings the reference applies when configured
    // (`_draft_value`: dflash_config first, then top level). None of the
    // published drafters sets them; refused rather than silently ignored.
    for k in ["input_embedding_scale", "output_multiplier"] {
        let v = dc
            .and_then(|d| d.get(k))
            .or_else(|| c.get(k))
            .and_then(|v| v.as_f64());
        if v.is_some_and(|v| v != 1.0) {
            bail!(
                "config.json: {k} {} is not implemented for block drafters",
                v.unwrap()
            );
        }
    }
    // The reference caps the block logits (compute_logits: cap * tanh(x / cap);
    // the z-lab Gemma 4 drafters carry the target's 30). Recorded for the
    // runtime, which caps the rows its selector / Markov chain / host argmax
    // read (a plain argmax is unchanged by a monotonic cap).
    let softcap = dc
        .and_then(|d| d.get("final_logit_softcapping"))
        .or_else(|| c.get("final_logit_softcapping"))
        .and_then(|v| v.as_f64())
        .filter(|v| *v > 0.0);
    if let Some(cap) = softcap {
        x.insert("drafter_logit_softcap".into(), json!(cap));
    }
    // DFlash 2 (grouped dynamic conv + candidate selector): recorded for
    // the runtime that implements them; DFlash 1 behaviour when absent.
    for k in [
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ] {
        if let Some(v) = u64_at(c, dc, k) {
            x.insert(k.into(), json!(v));
        }
    }
    // A reduced draft vocabulary: the drafter's own lm_head has
    // `draft_vocab_size` rows and `d2t` maps each back to a target id; the
    // embedding stays the full `vocab_size`. Only a plain DFlash head takes
    // it — DSpark's Markov head and DFlash 2's selector are indexed by the
    // full vocabulary.
    if let Some(dv) = c.get("draft_vocab_size").and_then(|v| v.as_u64()) {
        if dv < cfg.vocab_size as u64 {
            if kind != "dflash" || x.contains_key("selector_rank") {
                bail!("a reduced draft vocabulary ({dv} of {}) is supported for DFlash (1) drafters only", cfg.vocab_size);
            }
            x.insert("draft_vocab_size".into(), json!(dv));
        }
    }
    Ok(cfg)
}

/// Refuse a llama/qwen3-backbone block drafter whose decoder layers lack the
/// per-head q/k norms. The runtime's `arch_from_config` maps every dflash /
/// dspark sidecar to Qwen3, and its drafter manifest requires
/// `attention.q_norm` / `attention.k_norm` on every layer, so a norm-free
/// (true Llama) backbone would convert cleanly and then fail to load with a
/// missing-tensor error. `canonical` is the converter's mapped tensor names
/// (`layers.N.self_attn.q_norm.weight`). Gemma 4 backbones map through their
/// own tensor names and always carry the norms, so they are not checked here.
pub fn check_block_drafter_qk_norms<'a>(
    cfg: &ArchConfig,
    canonical: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    if cfg
        .extra_config
        .get("drafter_backbone")
        .and_then(|v| v.as_str())
        == Some("gemma4")
    {
        return Ok(());
    }
    let have: std::collections::HashSet<&str> = canonical.into_iter().collect();
    let missing: Vec<String> = (0..cfg.num_hidden_layers)
        .flat_map(|l| ["q_norm", "k_norm"].map(|n| format!("layers.{l}.self_attn.{n}.weight")))
        .filter(|n| !have.contains(n.as_str()))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let shown = missing
        .iter()
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let more = if missing.len() > 4 {
        format!(" and {} more", missing.len() - 4)
    } else {
        String::new()
    };
    bail!(
        "block drafter layers lack per-head q/k norms (missing {shown}{more}): the runtime runs \
         dflash / dspark drafters through its Qwen3 encoder, which needs them on every layer — \
         a norm-free Llama drafter backbone is not supported"
    )
}

impl HfMapper for DflashHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "dflash"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        let mut cfg = drafter_config_from_hf(c, "dflash")?;
        // DFlash 2 (`DFlash2DraftModel`): two grouped dynamic convolutions per
        // layer and a candidate selector over the block's top-k. A checkpoint
        // missing any of their four parameters would load as DFlash 1 and
        // draft with arithmetic it was not trained with, so all four are
        // required, and `dflash_version` 2 tells the loader to demand the
        // tensors they size.
        let v2 = c
            .get("architectures")
            .and_then(|v| v.as_array())
            .is_some_and(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .any(|a| a.starts_with("DFlash2"))
            });
        if v2 {
            let x = &cfg.extra_config;
            let get = |k: &str| x.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
            let missing: Vec<&str> = [
                "conv_kernel_size",
                "conv_group_size",
                "selector_rank",
                "selector_top_k",
            ]
            .into_iter()
            .filter(|k| get(k) == 0)
            .collect();
            if !missing.is_empty() {
                bail!("DFlash2 config.json: {missing:?} missing or zero");
            }
            let hidden = cfg.hidden_size as u64;
            let gs = get("conv_group_size");
            if hidden % gs != 0 {
                bail!("DFlash2 config.json: conv_group_size {gs} does not divide hidden_size {hidden}");
            }
            if get("selector_top_k") > 64 {
                bail!(
                    "DFlash2 config.json: selector_top_k {} > 64 (the selector keeps at most 64 candidates)",
                    get("selector_top_k")
                );
            }
            cfg.extra_config
                .insert("dflash_version".into(), serde_json::json!(2));
        }
        Ok(cfg)
    }
    fn value_transform(&self, canonical: &str) -> Option<ValueTransform> {
        fc_transform(canonical)
    }
}

impl HfMapper for DsparkHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "dspark"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        drafter_config_from_hf(c, "dspark")
    }
    fn value_transform(&self, canonical: &str) -> Option<ValueTransform> {
        fc_transform(canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> serde_json::Value {
        json!({
            "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 5,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 9728, "vocab_size": 151936, "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0, "tie_word_embeddings": true
        })
    }

    #[test]
    fn dflash_config_reads_nested_fields_and_windows() {
        let mut c = base();
        c["architectures"] = json!(["DFlashDraftModel"]);
        c["dflash_config"] = json!({"block_size": 16, "mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]});
        c["layer_types"] = json!([
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ]);
        c["sliding_window"] = json!(4096);
        assert!(matches!(
            hf_mapper_for_architectures(&c).map(|m| m.canonical_arch()),
            Some("dflash")
        ));
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert!(
            !cfg.tie_word_embeddings,
            "never synthesize a head for a drafter"
        );
        assert_eq!(cfg.sliding_window, 4096);
        assert_eq!(cfg.sliding_window_pattern, 5);
        assert_eq!(cfg.extra_config["block_size"], json!(16));
        assert_eq!(cfg.extra_config["logits_start"], json!(1));
        assert_eq!(
            cfg.extra_config["target_layer_ids"],
            json!([1, 9, 17, 25, 33])
        );
        assert_eq!(cfg.extra_config["speculator_kind"], json!("dflash"));
        // No `is_causal`: the sliding layers attend causally, the full one
        // bidirectionally (the reference's rule).
        assert_eq!(
            cfg.extra_config["drafter_causal_layers"],
            json!([true, true, true, true, false])
        );
        // All-sliding (DFlash 2 heads): a period no layer reaches.
        c["layer_types"] = json!(vec!["sliding_attention"; 5]);
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.sliding_window_pattern, 6);
        // An explicit is_causal=false overrides the sliding rule, and is
        // written out (all false) rather than left absent.
        c["is_causal"] = json!(false);
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(
            cfg.extra_config["drafter_causal_layers"],
            json!(vec![false; 5])
        );
    }

    #[test]
    fn dflash2_requires_its_conv_and_selector_geometry() {
        let mut c = base();
        c["architectures"] = json!(["DFlash2DraftModel"]);
        c["is_causal"] = json!(false);
        c["dflash_config"] = json!({
            "block_size": 8, "mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33],
            "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256, "selector_top_k": 16
        });
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["dflash_version"], json!(2));
        assert_eq!(cfg.extra_config["conv_group_size"], json!(16));
        assert_eq!(cfg.extra_config["selector_top_k"], json!(16));
        c["dflash_config"]["selector_rank"] = serde_json::Value::Null;
        assert!(
            DflashHfMapper.config_from_hf(&c).is_err(),
            "selector_rank missing"
        );
        c["dflash_config"]["selector_rank"] = json!(256);
        c["dflash_config"]["conv_group_size"] = json!(7);
        assert!(
            DflashHfMapper.config_from_hf(&c).is_err(),
            "group does not divide hidden"
        );
        // A DFlash 1 checkpoint never gets the version tag.
        c["architectures"] = json!(["DFlashDraftModel"]);
        c["dflash_config"]["conv_group_size"] = json!(16);
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert!(!cfg.extra_config.contains_key("dflash_version"));
        // Configured logit scalings are refused, not ignored.
        c["dflash_config"]["output_multiplier"] = json!(0.5);
        assert!(
            DflashHfMapper.config_from_hf(&c).is_err(),
            "output_multiplier"
        );
    }

    #[test]
    fn dspark_config_is_top_level_and_anchor_at_zero() {
        let mut c = base();
        c["architectures"] = json!(["Qwen3DSparkModel"]);
        c["block_size"] = json!(7);
        c["mask_token_id"] = json!(151669);
        c["target_layer_ids"] = json!([1, 9, 17, 25, 33]);
        c["markov_rank"] = json!(256);
        c["enable_confidence_head"] = json!(true);
        c["sliding_window"] = serde_json::Value::Null;
        assert!(matches!(
            hf_mapper_for_architectures(&c).map(|m| m.canonical_arch()),
            Some("dspark")
        ));
        let cfg = DsparkHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["logits_start"], json!(0));
        assert_eq!(cfg.extra_config["markov_rank"], json!(256));
        assert_eq!(cfg.extra_config["confidence_head"], json!(true));
        assert_eq!(cfg.sliding_window_pattern, 0);
    }

    #[test]
    fn specforge_dspark_drafts_from_row_one_through_the_target_head() {
        let mut c = base();
        c["architectures"] = json!(["DSparkDraftModel"]);
        c["block_size"] = json!(7);
        c["dspark_config"] = json!({"markov_rank": 256, "markov_head_type": "vanilla", "mask_token_id": 1,
                                    "target_layer_ids": [5, 19, 33, 47, 61]});
        c["dflash_config"] = c["dspark_config"].clone();
        c["enable_confidence_head"] = json!(true);
        c["rope_parameters"] = json!({"rope_theta": 1.0e7, "rope_type": "yarn", "factor": 32.0,
                                      "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 8192});
        let cfg = DsparkHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["logits_start"], json!(1));
        assert_eq!(cfg.extra_config["drafter_head"], json!("target"));
        assert_eq!(cfg.extra_config["markov_rank"], json!(256));
        assert_eq!(cfg.rope_scaling_type, "yarn");
        assert_eq!(cfg.rope_yarn_beta_fast, 32.0);
        assert!(cfg.rope_yarn_truncate);
        c["dspark_config"]["markov_head_type"] = json!("gated");
        assert!(
            DsparkHfMapper.config_from_hf(&c).is_err(),
            "gated Markov head is not implemented"
        );
    }

    #[test]
    fn gemma4_dspark_records_its_backbone() {
        // deepseek-ai/dspark_gemma4_12b_block7: `Gemma4DSparkModel`, a
        // gemma4_text decoder (global K=V layers, proportional rope) with
        // DeepSpec's mixed rope_parameters (top-level keys beside the
        // per-layer-type dicts).
        let c: serde_json::Value = serde_json::from_str(r#"{"architectures": ["Gemma4DSparkModel"], "attention_bias": false, "attention_dropout": 0.0, "attention_k_eq_v": true, "block_size": 7, "bos_token_id": 2, "confidence_head_with_markov": true, "dtype": "bfloat16", "enable_confidence_head": true, "enable_moe_block": false, "eos_token_id": 1, "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 3840, "hidden_size_per_layer_input": 0, "initializer_range": 0.02, "intermediate_size": 15360, "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"], "markov_head_type": "vanilla", "markov_rank": 256, "mask_token_id": 4, "max_position_embeddings": 262144, "model_type": "gemma4_text", "moe_intermediate_size": null, "num_anchors": 512, "num_attention_heads": 16, "num_experts": null, "num_global_key_value_heads": 1, "num_hidden_layers": 5, "num_key_value_heads": 8, "num_kv_shared_layers": 0, "num_target_layers": 48, "pad_token_id": 0, "rms_norm_eps": 1e-06, "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"}, "rope_theta": null, "rope_type": "default", "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}, "sliding_window": 1024, "target_layer_ids": [5, 17, 29, 41, 46], "target_model_type": "gemma4_unified", "target_text_model_type": "gemma4_unified_text", "tie_word_embeddings": false, "top_k_experts": null, "transformers_version": "5.10.2", "use_bidirectional_attention": "vision", "use_cache": true, "use_double_wide_mlp": false, "vocab_size": 262144, "vocab_size_per_layer_input": 262144}"#).unwrap();
        let m = hf_mapper_for_architectures(&c).expect("Gemma4DSparkModel maps to a drafter");
        let cfg = m.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["drafter_backbone"], json!("gemma4"));
        assert_eq!(cfg.extra_config["speculator_kind"], json!("dspark"));
        assert_eq!(cfg.extra_config["logits_start"], json!(0));
        assert_eq!(cfg.extra_config["markov_rank"], json!(256));
        assert_eq!(
            cfg.extra_config["target_layer_ids"],
            json!([5, 17, 29, 41, 46])
        );
        assert!(
            !cfg.extra_config.contains_key("drafter_head"),
            "carries its own head"
        );
        let mut q = c.clone();
        q["model_type"] = json!("gemma3");
        assert!(
            m.config_from_hf(&q).is_err(),
            "only llama / qwen3 / gemma4 backbones"
        );
    }

    #[test]
    fn speculators_dspark_keeps_its_own_head_and_drafts_from_the_anchor() {
        // RedHatAI/Qwen3-4B-speculator.dspark: `DSparkDraftModel` like SpecForge,
        // but its own embed / lm_head and sample_from_anchor = true (row 0).
        let c = json!({
            "architectures": ["DSparkDraftModel"], "speculators_model_type": "dspark",
            "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33], "block_size": 7, "mask_token_id": 151669,
            "draft_vocab_size": 151936, "sample_from_anchor": true, "sliding_window_non_causal": false,
            "markov_rank": 256, "markov_head_type": "vanilla", "enable_confidence_head": true,
            "confidence_head_with_markov": true,
            "transformer_layer_config": {
                "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 5, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9728, "vocab_size": 151936,
                "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                                "sliding_attention", "sliding_attention"],
                "sliding_window": 2048
            }
        });
        let cfg = DsparkHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["logits_start"], json!(0));
        assert!(
            !cfg.extra_config.contains_key("drafter_head"),
            "its own lm_head, not the target's"
        );
        assert_eq!(cfg.extra_config["markov_rank"], json!(256));
        assert_eq!(cfg.extra_config["confidence_head"], json!(true));
        assert_eq!(
            cfg.extra_config["target_layer_ids"],
            json!([0, 8, 16, 24, 32])
        );
    }

    #[test]
    fn speculators_block_drafter_is_flattened() {
        let mut c = json!({
            "architectures": ["DFlash2DraftModel"], "speculators_model_type": "dflash2",
            "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34], "block_size": 8, "mask_token_id": 151669,
            "draft_vocab_size": 151936, "sample_from_anchor": false, "sliding_window_non_causal": true,
            "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256, "selector_top_k": 16,
            "transformer_layer_config": {
                "model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 5, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936,
                "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                                "sliding_attention", "sliding_attention"],
                "sliding_window": 2048, "use_sliding_window": true
            }
        });
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(
            cfg.extra_config["target_layer_ids"],
            json!([1, 9, 17, 25, 33])
        );
        assert_eq!(cfg.extra_config["dflash_version"], json!(2));
        assert_eq!(cfg.extra_config["logits_start"], json!(1));
        assert_eq!(
            cfg.extra_config["drafter_causal_layers"],
            json!(vec![false; 5])
        );
        assert_eq!(cfg.rope_theta, 1.0e6);
        // sliding_window_non_causal false: the sliding layers are causal.
        c["sliding_window_non_causal"] = json!(false);
        c["architectures"] = json!(["DFlashDraftModel"]);
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(
            cfg.extra_config["drafter_causal_layers"],
            json!(vec![true; 5])
        );
        // A reduced draft vocabulary rides through for a DFlash (1) head...
        for k in [
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ] {
            c.as_object_mut().unwrap().remove(k);
        }
        c["speculators_model_type"] = json!("dflash");
        c["draft_vocab_size"] = json!(32000);
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["draft_vocab_size"], json!(32000));
        assert_eq!(
            cfg.vocab_size, 151936,
            "the embedding keeps the full vocabulary"
        );
        // ...and is refused where the full vocabulary indexes a head.
        c["architectures"] = json!(["DFlash2DraftModel"]);
        for (k, v) in [
            ("conv_kernel_size", 2),
            ("conv_group_size", 16),
            ("selector_rank", 256),
            ("selector_top_k", 16),
        ] {
            c[k] = json!(v);
        }
        assert!(
            DflashHfMapper.config_from_hf(&c).is_err(),
            "DFlash 2 with a reduced vocabulary"
        );
        c["draft_vocab_size"] = json!(200000);
        assert!(
            DflashHfMapper.config_from_hf(&c).is_err(),
            "larger than the vocabulary"
        );
    }

    fn layer_names(layers: u32, with_qk_norms: bool) -> Vec<String> {
        let mut v = vec!["fc.0.weight".to_string(), "hidden_norm.weight".to_string()];
        for l in 0..layers {
            for t in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                v.push(format!("layers.{l}.self_attn.{t}.weight"));
            }
            if with_qk_norms {
                v.push(format!("layers.{l}.self_attn.q_norm.weight"));
                v.push(format!("layers.{l}.self_attn.k_norm.weight"));
            }
        }
        v
    }

    #[test]
    fn llama_labelled_backbone_needs_qk_norms() {
        // RedHatAI/Qwen3-30B-A3B-speculator.dflash: transformer_layer_config
        // says `llama`, yet every layer ships self_attn.q_norm / k_norm.
        let mut c = base();
        c["model_type"] = json!("llama");
        c["dflash_config"] = json!({"mask_token_id": 1, "target_layer_ids": [1, 12]});
        let cfg = DflashHfMapper.config_from_hf(&c).unwrap();
        let names = layer_names(cfg.num_hidden_layers, true);
        check_block_drafter_qk_norms(&cfg, names.iter().map(String::as_str))
            .expect("Qwen-shaped llama-labelled layers are accepted");
        // A genuine Llama backbone (no q/k norms) would load as Qwen3 and fail
        // on its first missing norm: refused at conversion instead.
        let names = layer_names(cfg.num_hidden_layers, false);
        let err = check_block_drafter_qk_norms(&cfg, names.iter().map(String::as_str))
            .unwrap_err()
            .to_string();
        assert!(err.contains("layers.0.self_attn.q_norm.weight"), "{err}");
        assert!(err.contains("layers.0.self_attn.k_norm.weight"), "{err}");
        assert!(err.contains("and 6 more"), "{err}");
        assert!(
            err.contains("norm-free Llama drafter backbone is not supported"),
            "{err}"
        );
        // One layer short is enough to refuse.
        let names: Vec<String> = layer_names(cfg.num_hidden_layers, true)
            .into_iter()
            .filter(|n| n != "layers.4.self_attn.k_norm.weight")
            .collect();
        let err = check_block_drafter_qk_norms(&cfg, names.iter().map(String::as_str))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("missing layers.4.self_attn.k_norm.weight)"),
            "{err}"
        );
    }

    #[test]
    fn fc_blocks_carry_the_norm_invariant_scale() {
        assert_eq!(
            fc_transform("fc.0.weight"),
            Some(ValueTransform::DivBy(FC_SCALE_DIV))
        );
        assert_eq!(
            fc_transform("fc.7.weight"),
            Some(ValueTransform::DivBy(256))
        );
        assert!(fc_transform("fc.weight").is_none());
        assert!(fc_transform("hidden_norm.weight").is_none());
        assert!(fc_transform("layers.0.mlp.gate_proj.weight").is_none());
    }

    #[test]
    fn non_qwen3_backbone_and_missing_taps_are_refused() {
        let mut c = base();
        c["dflash_config"] = json!({"mask_token_id": 1});
        assert!(
            DflashHfMapper.config_from_hf(&c).is_err(),
            "no target_layer_ids"
        );
        c["dflash_config"] = json!({"mask_token_id": 1, "target_layer_ids": [1]});
        c["model_type"] = json!("gemma3");
        assert!(DflashHfMapper.config_from_hf(&c).is_err(), "gemma backbone");
        c["model_type"] = json!("llama");
        assert!(DflashHfMapper.config_from_hf(&c).is_ok(), "llama backbone");
        assert!(
            hf_mapper_for_architectures(&json!({"architectures": ["Qwen3ForCausalLM"]})).is_none()
        );
    }
}

// ── EAGLE-3 heads ────────────────────────────────────────────────────────────
//
// An EAGLE-3 drafter (SpecForge / EAGLE `LlamaForCausalLMEagle3`) is ONE
// llama-shaped decoder layer whose attention input is the concatenation of
// the normed token embedding and the normed feature `fc(concat of three
// target hidden states)` (so q/k/v project from 2·dim), a residual over the
// feature, its own output norm and a REDUCED-vocabulary lm_head
// (`draft_vocab_size` rows) with a `d2t` offset table back to target ids.
// The embedding is the target's (a copy when the checkpoint ships one).
// Sidecar arch `eagle3`: `midlayer.*` → `layers.0.*`, the q/k/v projections
// split by input column into `*_proj_emb` / `*_proj_hid`, `fc` split per tap
// (no scale fold — its output is also the residual), `d2t` carried as f32,
// `t2d` dropped (derivable).

pub struct Eagle3HfMapper;

pub fn is_eagle3_arch(arch: &str) -> bool {
    arch == "eagle3"
}

/// RedHat's `speculators` packaging (`speculators_model_type: "eagle3"`,
/// `Eagle3Speculator`): the layer geometry nests under
/// `transformer_layer_config`, the EAGLE fields sit at the top level, and
/// `eagle_aux_hidden_state_layer_ids` are hidden_states indices (vLLM captures
/// "before layer i"), where SGLang / SpecForge configs name the layer whose
/// OUTPUT is tapped (SGLang adds one). Flattened here into the SpecForge
/// form the rest of the converter and the runtime read.
fn flatten_speculators_eagle3(c: &serde_json::Value) -> Result<serde_json::Value> {
    let mut flat = c
        .get("transformer_layer_config")
        .cloned()
        .ok_or_else(|| anyhow!("speculators config without transformer_layer_config"))?;
    let o = flat
        .as_object_mut()
        .ok_or_else(|| anyhow!("transformer_layer_config is not an object"))?;
    for k in [
        "draft_vocab_size",
        "norm_before_residual",
        "norm_before_fc",
        "fc_norm",
        "target_hidden_size",
    ] {
        if let Some(v) = c.get(k).filter(|v| !v.is_null()) {
            o.insert(k.to_string(), v.clone());
        }
    }
    // speculators' `norm_output` is Eagle 3.1's "feed the POST-norm hidden back
    // across steps" — not the SpecForge flag of the same name (norm before the
    // lm_head, which speculators always does).
    if c.get("norm_output")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        bail!("speculators EAGLE-3 with norm_output (Eagle 3.1 post-norm chaining) is not supported yet");
    }
    if let Some(ids) = c
        .get("eagle_aux_hidden_state_layer_ids")
        .and_then(|v| v.as_array())
    {
        let ids: Vec<i64> = ids.iter().filter_map(|x| x.as_i64()).collect();
        if ids.iter().any(|&i| i < 1) {
            bail!("speculators aux layer ids {ids:?} name the embedding output; not supported");
        }
        let out: Vec<i64> = ids.iter().map(|i| i - 1).collect();
        o.insert(
            "eagle_aux_hidden_state_layer_ids".into(),
            serde_json::json!(out),
        );
    }
    Ok(flat)
}

fn eagle3_config_from_hf(c: &serde_json::Value) -> Result<ArchConfig> {
    let flat;
    let c = if c.get("speculators_model_type").and_then(|v| v.as_str()) == Some("eagle3") {
        flat = flatten_speculators_eagle3(c)?;
        &flat
    } else {
        c
    };
    let backbone = c.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    if backbone != "llama" {
        bail!("EAGLE-3 head model_type {backbone:?} is not llama — the engine runs the llama-shaped midlayer only");
    }
    let mut cfg = crate::llama::hf_generic_config(c)?;
    cfg.tie_word_embeddings = false;
    if cfg.num_hidden_layers != 1 {
        bail!(
            "EAGLE-3 head has {} layers; one midlayer expected",
            cfg.num_hidden_layers
        );
    }
    let draft_vocab = c
        .get("draft_vocab_size")
        .or_else(|| c.get("truncated_vocab_size"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("config.json: draft_vocab_size missing (the reduced lm_head)"))?;
    if c.get("fc_norm").and_then(|v| v.as_bool()).unwrap_or(false) {
        bail!("EAGLE-3 head with fc_norm (per-tap norms before fc) is not supported yet");
    }
    // norm_before_fc: one RMSNorm over the concatenated taps before `fc`
    // (RedHat's gpt-oss heads, weight `input_norm` [n_taps * hidden]);
    // norm_before_residual: the attention residual is the NORMED feature.
    // Both recorded for the runtime.
    let norm_before_fc = c
        .get("norm_before_fc")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let norm_before_residual = c
        .get("norm_before_residual")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // norm_output=false (no final norm before the lm_head): the runtime
    // always applies output_norm and the loader requires it, so such a head
    // would either fail to load or draft from mismatched logits.
    if !c
        .get("norm_output")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
    {
        bail!("EAGLE-3 head with norm_output=false is not supported yet (the runtime always normalizes before the lm_head)");
    }
    let target_hidden = c
        .get("target_hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(cfg.hidden_size as u64);
    if target_hidden != cfg.hidden_size as u64 {
        bail!(
            "EAGLE-3 head hidden {} != target hidden {}: cross-width heads are not supported",
            cfg.hidden_size,
            target_hidden
        );
    }
    use serde_json::json;
    let x = &mut cfg.extra_config;
    x.insert("speculator_kind".into(), json!("eagle3"));
    x.insert("draft_vocab_size".into(), json!(draft_vocab));
    if norm_before_residual {
        x.insert("norm_before_residual".into(), json!(true));
    }
    if norm_before_fc {
        x.insert("norm_before_fc".into(), json!(true));
    }
    x.insert(
        "norm_output".into(),
        json!(c
            .get("norm_output")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)),
    );
    // Explicit aux layers (layer ids whose OUTPUT is tapped); absent → the
    // runtime's default [1, L/2 - 1, L - 4] on the target's layer count.
    if let Some(ids) = c
        .get("eagle_config")
        .and_then(|e| e.get("eagle_aux_hidden_state_layer_ids"))
        .or_else(|| c.get("eagle_aux_hidden_state_layer_ids"))
        .and_then(|v| v.as_array())
    {
        let ids: Vec<u64> = ids.iter().filter_map(|x| x.as_u64()).collect();
        if ids.len() != 3 {
            bail!("EAGLE-3 aux layer ids must name 3 layers, got {ids:?}");
        }
        x.insert("aux_layer_ids".into(), json!(ids));
    }
    Ok(cfg)
}

impl HfMapper for Eagle3HfMapper {
    fn canonical_arch(&self) -> &'static str {
        "eagle3"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        eagle3_config_from_hf(c)
    }
}

/// Pick the EAGLE-3 mapper from `architectures` (`LlamaForCausalLMEagle3`,
/// `Eagle3LlamaForCausalLM`).
pub fn hf_mapper_for_eagle3(config: &serde_json::Value) -> Option<&'static dyn HfMapper> {
    let archs = config.get("architectures")?.as_array()?;
    let named = archs
        .iter()
        .filter_map(|v| v.as_str())
        .any(|a| a.contains("Eagle3") || a.contains("EAGLE3"));
    // nebius' EAGLE-3 heads (nebius/EAGLE3-gpt-oss-20b) say only
    // `LlamaForCausalLM`: a ONE-layer llama with a reduced draft vocabulary
    // is an EAGLE-3 head, never a language model.
    let unnamed_head = config.get("num_hidden_layers").and_then(|v| v.as_u64()) == Some(1)
        && (config.get("draft_vocab_size").is_some()
            || config.get("truncated_vocab_size").is_some());
    (named || unnamed_head).then_some(&Eagle3HfMapper)
}

#[cfg(test)]
mod eagle3_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn eagle3_config_reads_draft_vocab_and_aux_layers() {
        let mut c = json!({
            "architectures": ["Eagle3LlamaForCausalLM"], "model_type": "llama", "hidden_size": 2560,
            "num_hidden_layers": 1, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 9728, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "draft_vocab_size": 32000, "tie_word_embeddings": false
        });
        assert!(matches!(
            hf_mapper_for_eagle3(&c).map(|m| m.canonical_arch()),
            Some("eagle3")
        ));
        let cfg = Eagle3HfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["draft_vocab_size"], json!(32000));
        assert!(!cfg.extra_config.contains_key("aux_layer_ids"));
        c["eagle_config"] = json!({"eagle_aux_hidden_state_layer_ids": [1, 15, 28]});
        let cfg = Eagle3HfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["aux_layer_ids"], json!([1, 15, 28]));
        c["num_hidden_layers"] = json!(2);
        assert!(
            Eagle3HfMapper.config_from_hf(&c).is_err(),
            "one midlayer only"
        );
        assert!(hf_mapper_for_eagle3(&json!({"architectures": ["LlamaForCausalLM"]})).is_none());
    }

    #[test]
    fn eagle3_norm_before_fc_is_recorded() {
        // RedHatAI/gpt-oss-20b-speculator.eagle3: one RMSNorm over the
        // concatenated taps before fc (weight `input_norm` [3H]).
        let c = json!({
            "architectures": ["Eagle3DraftModel"], "speculators_model_type": "eagle3",
            "draft_vocab_size": 64000, "norm_before_fc": true, "norm_before_residual": true,
            "transformer_layer_config": {
                "model_type": "llama", "hidden_size": 2880, "num_hidden_layers": 1, "num_attention_heads": 64,
                "num_key_value_heads": 8, "head_dim": 64, "intermediate_size": 2880, "vocab_size": 201088,
                "rms_norm_eps": 1e-5, "rope_theta": 10000.0
            }
        });
        let cfg = Eagle3HfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["norm_before_fc"], json!(true));
        assert_eq!(cfg.extra_config["norm_before_residual"], json!(true));
    }

    #[test]
    fn unnamed_one_layer_head_is_eagle3() {
        // nebius/EAGLE3-gpt-oss-20b: architectures says only LlamaForCausalLM,
        // the reduced vocabulary is `truncated_vocab_size`, the rope is YaRN.
        let c = json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 1,
            "truncated_vocab_size": 64000, "draft_vocab_size": 64000, "hidden_size": 2880,
            "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64, "intermediate_size": 11520,
            "vocab_size": 201088, "rms_norm_eps": 1e-5, "rope_theta": 150000,
            "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0,
                             "original_max_position_embeddings": 4096, "rope_type": "yarn", "truncate": false}
        });
        assert!(hf_mapper_for_eagle3(&c).is_some());
        let cfg = Eagle3HfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.extra_config["draft_vocab_size"], json!(64000));
        assert_eq!(cfg.rope_scaling_type, "yarn");
        // A real one-layer LM with no draft vocabulary is not a head.
        let mut lm = c.clone();
        lm.as_object_mut().unwrap().remove("truncated_vocab_size");
        lm.as_object_mut().unwrap().remove("draft_vocab_size");
        assert!(hf_mapper_for_eagle3(&lm).is_none());
    }

    #[test]
    fn speculators_eagle3_is_flattened_with_hidden_state_taps() {
        let c = json!({
            "architectures": ["Eagle3Speculator"], "speculators_model_type": "eagle3",
            "draft_vocab_size": 32000, "norm_before_residual": true,
            "eagle_aux_hidden_state_layer_ids": [2, 18, 33],
            "transformer_layer_config": {
                "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 1,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "intermediate_size": 12288, "vocab_size": 151936, "rms_norm_eps": 1e-6,
                "rope_theta": 1000000
            }
        });
        assert!(hf_mapper_for_eagle3(&c).is_some());
        let cfg = Eagle3HfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.extra_config["draft_vocab_size"], json!(32000));
        assert_eq!(cfg.extra_config["norm_before_residual"], json!(true));
        // hidden_states indices [2, 18, 33] -> the loader's output-layer form.
        assert_eq!(cfg.extra_config["aux_layer_ids"], json!([1, 17, 32]));
        let mut bad = c.clone();
        bad["norm_output"] = json!(true);
        assert!(
            Eagle3HfMapper.config_from_hf(&bad).is_err(),
            "Eagle 3.1 post-norm chaining"
        );
    }
}
