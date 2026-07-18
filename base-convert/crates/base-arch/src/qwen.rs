//! Qwen2 / Qwen3 GGUF → canonical `.base` mapping.
//!
//! Qwen shares Llama's tensor naming convention; only the config metadata
//! key prefix differs (`qwen2.*` / `qwen3.*`).

use crate::llama::map_llama_style;
use crate::{ArchConfig, GgufMapper};
use anyhow::{Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct QwenMapper;
pub struct QwenHfMapper;
pub struct QwenMoeHfMapper;
pub struct Qwen35HfMapper;
pub struct Qwen35MoeHfMapper;

/// Qwen3.5 / 3.6 shared HF config extraction.
///
/// Qwen3.5 (`Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`) is a
/// natively-multimodal model whose language tower lives under a nested
/// `text_config` (model_type `qwen3_5_text`). It reuses the Qwen3-Next
/// hybrid decoder: most layers are Gated-DeltaNet linear-attention blocks,
/// with every `full_attention_interval`-th layer being a (gated) softmax
/// attention block. RoPE parameters (theta, partial-rotary factor, mRoPE
/// sections) live under `rope_parameters`, not at the top level.
///
/// This reads the standard Llama-shaped fields from the text config, then
/// patches in the RoPE + hybrid-linear-attention fields the generic reader
/// doesn't know about.
fn qwen35_config_from_hf(c: &serde_json::Value) -> Result<crate::ArchConfig> {
    // A text-only checkpoint may hoist the text params to the top level;
    // the multimodal wrapper nests them under `text_config`. Prefer the
    // nested object when present.
    let tc = c.get("text_config").unwrap_or(c);
    let mut config = crate::llama::hf_generic_config(tc)?;

    let u32_v =
        |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64()).map(|n| n as u32);
    let f32_v =
        |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64()).map(|f| f as f32);
    let bool_v = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_bool());

    // RoPE lives under `rope_parameters` on Qwen3.5 (not top-level
    // `rope_theta`), so hf_generic_config's default (10000) is wrong —
    // override it here along with the partial-rotary + mRoPE fields.
    if let Some(rp) = tc.get("rope_parameters") {
        if let Some(theta) = f32_v(rp, "rope_theta") {
            config.rope_theta = theta;
        }
        if let Some(f) = f32_v(rp, "partial_rotary_factor") {
            config.partial_rotary_factor = f;
        }
        if let Some(sec) = rp.get("mrope_section").and_then(|v| v.as_array()) {
            config.mrope_section = sec
                .iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect();
        }
        config.mrope_interleaved = bool_v(rp, "mrope_interleaved").unwrap_or(false);
    }

    // Hybrid layer schedule + Gated-DeltaNet ("linear attention") shapes.
    if let Some(lt) = tc.get("layer_types").and_then(|v| v.as_array()) {
        config.layer_types = lt
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
    }
    config.full_attention_interval = u32_v(tc, "full_attention_interval").unwrap_or(0);
    config.linear_num_key_heads = u32_v(tc, "linear_num_key_heads").unwrap_or(0);
    config.linear_num_value_heads = u32_v(tc, "linear_num_value_heads").unwrap_or(0);
    config.linear_key_head_dim = u32_v(tc, "linear_key_head_dim").unwrap_or(0);
    config.linear_value_head_dim = u32_v(tc, "linear_value_head_dim").unwrap_or(0);
    config.linear_conv_kernel_dim = u32_v(tc, "linear_conv_kernel_dim").unwrap_or(0);
    config.attn_output_gate = bool_v(tc, "attn_output_gate").unwrap_or(false);

    Ok(config)
}

/// Qwen3.5 / 3.6 RMSNorm gamma shift.
///
/// Qwen3NextRMSNorm stores zero-centered gamma and applies `(1 + weight)` at
/// inference (same convention as Gemma 3). Bake the +1 in here so the runtime
/// uses the plain rmsnorm kernel. This covers EVERY norm — input_layernorm,
/// post_attention_layernorm, the attention q_norm/k_norm, and the final
/// `model.norm` — EXCEPT the Gated-DeltaNet output norm
/// (`linear_attn.norm.weight`), which is a Qwen3NextRMSNormGated and uses plain
/// `weight` with no offset.
fn qwen35_norm_shift(canonical: &str) -> f32 {
    if canonical.ends_with("norm.weight") && !canonical.ends_with(".linear_attn.norm.weight") {
        1.0
    } else {
        0.0
    }
}

impl crate::HfMapper for Qwen35HfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen35"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        qwen35_config_from_hf(c)
    }
    fn norm_shift(&self, canonical: &str) -> f32 {
        qwen35_norm_shift(canonical)
    }
}

impl crate::HfMapper for Qwen35MoeHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen35moe"
    }
    fn norm_shift(&self, canonical: &str) -> f32 {
        qwen35_norm_shift(canonical)
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        // The real Qwen3.5/3.6-MoE text_config carries NO `intermediate_size`
        // (the HF config class deletes the attribute) — the dense-FFN slot is
        // taken by the shared expert, whose width lives in
        // `shared_expert_intermediate_size`. hf_generic_config hard-requires
        // `intermediate_size`, so patch the shared-expert width in before
        // delegating; the runtime then reads ffn_dim = shared-expert width
        // for the `ffn_*_shexp` SwiGLU stream (routed experts use
        // `moe_intermediate_size`).
        let mut patched = c.clone();
        {
            let tc_mut = if patched.get("text_config").is_some() {
                patched.get_mut("text_config").unwrap()
            } else {
                &mut patched
            };
            if tc_mut.get("intermediate_size").is_none() {
                let shexp = tc_mut
                    .get("shared_expert_intermediate_size")
                    .and_then(|v| v.as_u64());
                if let (Some(sh), Some(obj)) = (shexp, tc_mut.as_object_mut()) {
                    obj.insert("intermediate_size".into(), serde_json::json!(sh));
                }
            }
        }
        let mut config = qwen35_config_from_hf(&patched)?;
        // MoE topology lives alongside the text params (nested under
        // `text_config` for the multimodal wrapper). Keep `intermediate_size`
        // as the dense/shared-FFN width and read the per-expert width into
        // `moe_intermediate_size` — same distinction as Qwen3-MoE.
        let tc = c.get("text_config").unwrap_or(c);
        let u32_v = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let bool_v = |k: &str| tc.get(k).and_then(|v| v.as_bool());
        if let Some(n) = u32_v("num_experts") {
            config.num_experts = n;
        }
        if let Some(n) = u32_v("num_experts_per_tok") {
            config.num_experts_per_tok = n;
        }
        if let Some(n) = u32_v("moe_intermediate_size") {
            config.moe_intermediate_size = n;
        }
        // Shared expert (Qwen3NextSparseMoeBlock): always present on the
        // released 35B-A3B checkpoints. Signal it to the runtime so the
        // qwen3_5 encoder emits the ffn_*_shexp stream + scalar gate.
        if u32_v("shared_expert_intermediate_size").unwrap_or(0) > 0 {
            config.num_shared_experts = 1;
        }
        config.norm_topk_prob = bool_v("norm_topk_prob").unwrap_or(true);
        Ok(config)
    }
}

impl crate::HfMapper for QwenHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        crate::llama::hf_generic_config(c)
    }
}

impl crate::HfMapper for QwenMoeHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen3_moe"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        // Qwen3-MoE: keep `intermediate_size` as the dense shared-FFN
        // width (0 if no shared expert) and place the per-expert FFN
        // under `moe_intermediate_size` so the runtime can distinguish.
        // Pre-fix code overwrote `intermediate_size` with the per-expert
        // value, which made the .base header indistinguishable from a
        // tiny dense Qwen and broke MoE dispatch.
        let mut config = crate::llama::hf_generic_config(c)?;
        let u32_v = |k: &str| c.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let bool_v = |k: &str| c.get(k).and_then(|v| v.as_bool());
        if let Some(n) = u32_v("num_experts") {
            config.num_experts = n;
        }
        if let Some(n) = u32_v("num_experts_per_tok") {
            config.num_experts_per_tok = n;
        }
        if let Some(n) = u32_v("moe_intermediate_size") {
            config.moe_intermediate_size = n;
        }
        if let Some(b) = bool_v("norm_topk_prob") {
            config.norm_topk_prob = b;
        }
        Ok(config)
    }
}

impl GgufMapper for QwenMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        // Try most-specific first. qwen35 = Qwen 3.5 (later rev of Qwen3).
        let prefix = if m.keys().any(|k| k.starts_with("qwen36.")) {
            "qwen36"
        } else if m.keys().any(|k| k.starts_with("qwen35.")) {
            "qwen35"
        } else if m.keys().any(|k| k.starts_with("qwen3.")) {
            "qwen3"
        } else {
            "qwen2"
        };

        let u32_key = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("missing metadata key: {k}"))
        };
        let f32_key = |k: &str| m.get(k).and_then(|v| v.as_f32());

        let hidden_size = u32_key(&format!("{prefix}.embedding_length"))?;
        let num_hidden_layers = u32_key(&format!("{prefix}.block_count"))?;
        let num_attention_heads = u32_key(&format!("{prefix}.attention.head_count"))?;
        let num_kv_heads = u32_key(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads);
        let intermediate_size = u32_key(&format!("{prefix}.feed_forward_length"))?;
        let vocab_size = u32_key(&format!("{prefix}.vocab_size"))
            .or_else(|_| {
                m.get("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        KvValue::Array(a) => Some(a.len() as u32),
                        _ => None,
                    })
                    .context("no vocab_size and no tokenizer.ggml.tokens")
            })?;
        let head_dim = u32_key(&format!("{prefix}.attention.key_length"))
            .unwrap_or(hidden_size / num_attention_heads);

        let rope_theta = f32_key(&format!("{prefix}.rope.freq_base")).unwrap_or(10_000.0);
        let rope_scale =
            f32_key(&format!("{prefix}.rope.scaling.factor")).unwrap_or(1.0);
        let rms_norm_eps =
            f32_key(&format!("{prefix}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

        Ok(ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta,
            rope_scale,
            rms_norm_eps,
            tie_word_embeddings: false,
            ..ArchConfig::default()
        })
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        map_llama_style(n)
    }
}

/// Qwen2/3 MoE variant. Differs from dense Qwen in that per-layer FFN is
/// an expert stack (`ffn_{gate,up,down}_exps.weight`) + a router
/// (`ffn_gate_inp.weight`) instead of a single MLP.
pub struct QwenMoeMapper;

impl GgufMapper for QwenMoeMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen_moe"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        let prefix = if m.keys().any(|k| k.starts_with("qwen36moe.")) {
            "qwen36moe"
        } else if m.keys().any(|k| k.starts_with("qwen35moe.")) {
            "qwen35moe"
        } else if m.keys().any(|k| k.starts_with("qwen3moe.")) {
            "qwen3moe"
        } else {
            "qwen2moe"
        };

        let u32_key = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("missing metadata key: {k}"))
        };
        let f32_key = |k: &str| m.get(k).and_then(|v| v.as_f32());

        let hidden_size = u32_key(&format!("{prefix}.embedding_length"))?;
        let num_hidden_layers = u32_key(&format!("{prefix}.block_count"))?;
        let num_attention_heads = u32_key(&format!("{prefix}.attention.head_count"))?;
        let num_kv_heads = u32_key(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads);
        // For MoE, GGUF carries two FFN widths:
        //   `feed_forward_length`        = nominal/dense width (HF `intermediate_size`)
        //   `expert_feed_forward_length` = per-expert width (HF `moe_intermediate_size`)
        // Keep them distinct; conflating them broke header layout and
        // (latently) any scratch buffer sized off cfg.ffn_dim.
        let intermediate_size = u32_key(&format!("{prefix}.feed_forward_length"))?;
        let vocab_size = u32_key(&format!("{prefix}.vocab_size")).or_else(|_| {
            m.get("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    KvValue::Array(a) => Some(a.len() as u32),
                    _ => None,
                })
                .context("no vocab_size and no tokenizer.ggml.tokens")
        })?;
        let head_dim = u32_key(&format!("{prefix}.attention.key_length"))
            .unwrap_or(hidden_size / num_attention_heads);

        let rope_theta = f32_key(&format!("{prefix}.rope.freq_base")).unwrap_or(10_000.0);
        let rope_scale = f32_key(&format!("{prefix}.rope.scaling.factor")).unwrap_or(1.0);
        let rms_norm_eps =
            f32_key(&format!("{prefix}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

        // MoE topology — runtime requires these to dispatch experts.
        // Without them the .base header drops MoE keys (gated on
        // num_experts > 0 in to_config_map) and runtime reads n_experts=0,
        // collapsing forward to a degenerate path.
        let num_experts = u32_key(&format!("{prefix}.expert_count")).unwrap_or(0);
        let num_experts_per_tok = u32_key(&format!("{prefix}.expert_used_count")).unwrap_or(0);
        let moe_intermediate_size =
            u32_key(&format!("{prefix}.expert_feed_forward_length")).unwrap_or(0);

        // Qwen MoE family normalizes top-k routing weights; GGUF doesn't
        // carry this flag so we set it from arch family. Gemma is the
        // false case (handled separately in gemma.rs).
        let config = ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta,
            rope_scale,
            rms_norm_eps,
            tie_word_embeddings: false,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob: true,
            ..ArchConfig::default()
        };
        Ok(config)
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        map_llama_style(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HfMapper;
    use base_readers::gguf::KvValue;
    use serde_json::json;

    /// Equivalent HF and GGUF metadata for Qwen3-30B-A3B should produce
    /// identical `ArchConfig` for the load-bearing fields. This is the
    /// catch-net for the "GGUF path drops a key the HF path reads" bug
    /// class — pre-fix, the GGUF mapper was a stub that emitted
    /// `num_experts=0`, `intermediate_size=expert_ffn` (conflated dense
    /// and expert widths) and `norm_topk_prob=false`, while the HF
    /// mapper read the right values. End-to-end output was degenerate
    /// single-token loops; the runtime's MoE config check silently
    /// accepted n_experts=0 and ran a degenerate forward.
    #[test]
    fn qwen3_moe_30b_a3b_hf_gguf_parity() {
        let hf_json = json!({
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "vocab_size": 151936,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": false,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "norm_topk_prob": true,
        });
        let hf = QwenMoeHfMapper.config_from_hf(&hf_json).unwrap();

        let mut gguf = std::collections::BTreeMap::new();
        gguf.insert("qwen3moe.embedding_length".into(), KvValue::U32(2048));
        gguf.insert("qwen3moe.block_count".into(), KvValue::U32(48));
        gguf.insert("qwen3moe.attention.head_count".into(), KvValue::U32(32));
        gguf.insert("qwen3moe.attention.head_count_kv".into(), KvValue::U32(4));
        gguf.insert("qwen3moe.attention.key_length".into(), KvValue::U32(128));
        gguf.insert("qwen3moe.feed_forward_length".into(), KvValue::U32(6144));
        gguf.insert(
            "qwen3moe.expert_feed_forward_length".into(),
            KvValue::U32(768),
        );
        gguf.insert("qwen3moe.vocab_size".into(), KvValue::U32(151936));
        gguf.insert("qwen3moe.rope.freq_base".into(), KvValue::F32(1_000_000.0));
        gguf.insert(
            "qwen3moe.attention.layer_norm_rms_epsilon".into(),
            KvValue::F32(1e-06),
        );
        gguf.insert("qwen3moe.expert_count".into(), KvValue::U32(128));
        gguf.insert("qwen3moe.expert_used_count".into(), KvValue::U32(8));
        let g = QwenMoeMapper.config_from_gguf(&gguf).unwrap();

        // Compare load-bearing fields. Cosmetic ones (max_position_embeddings,
        // bos/eos_token_id) are read by HF from optional keys but not by
        // the GGUF Qwen3-MoE path; they don't affect runtime correctness
        // and are intentionally outside this assertion.
        assert_eq!(hf.hidden_size, g.hidden_size);
        assert_eq!(hf.num_hidden_layers, g.num_hidden_layers);
        assert_eq!(hf.num_attention_heads, g.num_attention_heads);
        assert_eq!(hf.num_kv_heads, g.num_kv_heads);
        assert_eq!(hf.head_dim, g.head_dim);
        assert_eq!(hf.intermediate_size, g.intermediate_size);
        assert_eq!(hf.vocab_size, g.vocab_size);
        assert_eq!(hf.rope_theta, g.rope_theta);
        assert_eq!(hf.rms_norm_eps, g.rms_norm_eps);
        assert_eq!(hf.tie_word_embeddings, g.tie_word_embeddings);
        // These four fields must agree between the GGUF and HF paths
        // (a mismatch means the GGUF path emitted zero / wrong values).
        assert_eq!(hf.num_experts, 128);
        assert_eq!(g.num_experts, 128);
        assert_eq!(hf.num_experts_per_tok, g.num_experts_per_tok);
        assert_eq!(hf.moe_intermediate_size, g.moe_intermediate_size);
        assert_eq!(hf.intermediate_size, 6144);
        assert_eq!(g.intermediate_size, 6144);
        assert_eq!(hf.norm_topk_prob, g.norm_topk_prob);
        assert!(hf.norm_topk_prob);
    }

    /// Qwen3NextRMSNorm is zero-centered (`1 + weight`), so base-convert bakes
    /// +1 into every norm EXCEPT the Gated-DeltaNet output norm
    /// (`linear_attn.norm`, a Qwen3NextRMSNormGated that uses plain weight).
    #[test]
    fn qwen35_norm_shift_predicate() {
        use super::qwen35_norm_shift;
        // Regular norms shift by +1.
        for s in [
            "layers.0.input_norm.weight",
            "layers.3.post_attn_norm.weight",
            "layers.3.self_attn.q_norm.weight",
            "layers.3.self_attn.k_norm.weight",
            "final_norm.weight",
        ] {
            assert_eq!(qwen35_norm_shift(s), 1.0, "should shift: {s}");
        }
        // The gated GDN norm and everything non-norm do NOT shift.
        for s in [
            "layers.0.linear_attn.norm.weight",
            "layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.mlp.gate_proj.weight",
            "embed_tokens.weight",
        ] {
            assert_eq!(qwen35_norm_shift(s), 0.0, "should not shift: {s}");
        }
    }

    /// Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (identical config shape): the REAL
    /// checkpoint's text_config has NO `intermediate_size` (the HF config
    /// class deletes it — the dense-FFN slot is the shared expert), GQA GDN
    /// (nv=32 ≠ nk=16), 256 experts top-8 with fused gate_up, a shared
    /// expert of width `shared_expert_intermediate_size`, and an MTP head
    /// (tensors skipped at map time). The mapper must not bail on the
    /// missing `intermediate_size` and must surface every MoE + GDN field.
    #[test]
    fn qwen35_moe_35b_a3b_config_from_hf() {
        let cfg = json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "attn_output_gate": true,
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 2048,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "moe_intermediate_size": 512,
                "mtp_num_hidden_layers": 1,
                "num_attention_heads": 16,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-06,
                "shared_expert_intermediate_size": 512,
                "vocab_size": 248320,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            }
        });
        let c = Qwen35MoeHfMapper.config_from_hf(&cfg).unwrap();

        // intermediate_size backfilled from the shared-expert width (the
        // runtime's ffn_dim = shared-expert SwiGLU width).
        assert_eq!(c.intermediate_size, 512);
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 512);
        assert_eq!(c.num_shared_experts, 1);
        assert!(c.norm_topk_prob);

        // GQA GDN: value heads ≠ key heads.
        assert_eq!(c.linear_num_key_heads, 16);
        assert_eq!(c.linear_num_value_heads, 32);
        assert_eq!(c.rope_theta, 10_000_000.0);
        assert_eq!(c.partial_rotary_factor, 0.25);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_kv_heads, 2);

        // Round-trip: the runtime reads these back from the .base header.
        let m = c.to_config_map();
        assert_eq!(m["num_experts"], json!(256));
        assert_eq!(m["num_shared_experts"], json!(1));
        assert_eq!(m["moe_intermediate_size"], json!(512));
        assert_eq!(m["intermediate_size"], json!(512));
        assert_eq!(m["linear_num_value_heads"], json!(32));
    }

    /// Qwen3.5-2B-Base: the multimodal wrapper nests the LM params under
    /// `text_config`, RoPE under `rope_parameters`, and declares a hybrid
    /// Gated-DeltaNet / full-attention layer schedule. The mapper must dig
    /// into `text_config`, override rope_theta from `rope_parameters`
    /// (top-level default 10000 is wrong — real value is 1e7), and capture
    /// every linear-attention shape + the layer_types schedule.
    #[test]
    fn qwen35_2b_hybrid_config_from_hf() {
        let cfg = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text",
                "attn_output_gate": true,
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": true,
                "vocab_size": 248320,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_type": "default",
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            }
        });
        let c = Qwen35HfMapper.config_from_hf(&cfg).unwrap();

        // Core Llama-shaped fields read from the nested text_config.
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.num_attention_heads, 8);
        assert_eq!(c.num_kv_heads, 2);
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.intermediate_size, 6144);
        assert_eq!(c.vocab_size, 248320);
        assert_eq!(c.rms_norm_eps, 1e-6);
        assert!(c.tie_word_embeddings);
        assert_eq!(c.eos_token_id, 248044);

        // RoPE pulled from rope_parameters (NOT the 10000 default).
        assert_eq!(c.rope_theta, 10_000_000.0);
        assert_eq!(c.partial_rotary_factor, 0.25);
        assert_eq!(c.mrope_section, vec![11, 11, 10]);
        assert!(c.mrope_interleaved);

        // Hybrid schedule + Gated-DeltaNet shapes.
        assert_eq!(c.full_attention_interval, 4);
        assert_eq!(c.layer_types.len(), 24);
        assert_eq!(c.layer_types[0], "linear_attention");
        assert_eq!(c.layer_types[3], "full_attention");
        assert_eq!(c.linear_num_key_heads, 16);
        assert_eq!(c.linear_num_value_heads, 16);
        assert_eq!(c.linear_key_head_dim, 128);
        assert_eq!(c.linear_value_head_dim, 128);
        assert_eq!(c.linear_conv_kernel_dim, 4);
        assert!(c.attn_output_gate);

        // Every hybrid field must survive the round-trip into the .base
        // header config map (the runtime reads them back from here).
        let m = c.to_config_map();
        assert_eq!(m["rope_theta"], json!(10_000_000.0));
        assert_eq!(m["partial_rotary_factor"], json!(0.25));
        assert_eq!(m["full_attention_interval"], json!(4));
        assert_eq!(m["linear_num_key_heads"], json!(16));
        assert_eq!(m["linear_conv_kernel_dim"], json!(4));
        assert_eq!(m["attn_output_gate"], json!(true));
        assert_eq!(m["mrope_section"], json!([11, 11, 10]));
        assert!(m.contains_key("layer_types"));
    }
}
