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
}
