//! Llama GGUF → canonical `.base` mapping.
//!
//! GGUF Llama tensor naming conventions (what we map from):
//! ```text
//!   token_embd.weight            → embed_tokens.weight
//!   output.weight                → lm_head.weight (absent if tied)
//!   output_norm.weight           → final_norm.weight
//!   rope_freqs.weight            → dropped (we recompute at runtime)
//!   blk.N.attn_norm.weight       → layers.N.input_norm.weight
//!   blk.N.attn_q.weight          → layers.N.self_attn.q_proj.weight
//!   blk.N.attn_k.weight          → layers.N.self_attn.k_proj.weight
//!   blk.N.attn_v.weight          → layers.N.self_attn.v_proj.weight
//!   blk.N.attn_output.weight     → layers.N.self_attn.o_proj.weight
//!   blk.N.ffn_norm.weight        → layers.N.post_attn_norm.weight
//!   blk.N.ffn_gate.weight        → layers.N.mlp.gate_proj.weight
//!   blk.N.ffn_up.weight          → layers.N.mlp.up_proj.weight
//!   blk.N.ffn_down.weight        → layers.N.mlp.down_proj.weight
//! ```

use crate::{ArchConfig, GgufMapper};
use anyhow::{Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct LlamaMapper;
pub struct LlamaHfMapper;

impl crate::HfMapper for LlamaHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "llama"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        hf_generic_config(c)
    }
    fn rope_permute_heads(&self, canonical: &str, cfg: &crate::ArchConfig) -> Option<u32> {
        // Weights only: llama/mistral attention carries no q/k bias.
        if canonical.ends_with("self_attn.q_proj.weight") {
            Some(cfg.num_attention_heads)
        } else if canonical.ends_with("self_attn.k_proj.weight") {
            Some(cfg.num_kv_heads)
        } else {
            None
        }
    }
}

pub(crate) fn hf_generic_config(c: &serde_json::Value) -> Result<crate::ArchConfig> {
    let u32_key = |k: &str| {
        c.get(k)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .with_context(|| format!("config.json missing {k}"))
    };
    let f32_key = |k: &str| c.get(k).and_then(|v| v.as_f64()).map(|f| f as f32);
    let bool_key = |k: &str| c.get(k).and_then(|v| v.as_bool());

    let hidden_size = u32_key("hidden_size")?;
    let num_hidden_layers = u32_key("num_hidden_layers")?;
    let num_attention_heads = u32_key("num_attention_heads")?;
    let num_kv_heads =
        u32_key("num_key_value_heads").unwrap_or(num_attention_heads);
    // Gemma 3n stores `intermediate_size` as a per-layer array; other archs
    // store a single u32. Accept both and fall back to the first array entry
    // (the runtime then mirrors the array via per_layer_ffn).
    let intermediate_size = match c.get("intermediate_size") {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .map(|x| x as u32)
            .with_context(|| "config.json `intermediate_size` not a u32")?,
        Some(serde_json::Value::Array(arr)) => arr
            .first()
            .and_then(|v| v.as_u64())
            .map(|x| x as u32)
            .with_context(|| "config.json `intermediate_size` array empty or non-numeric")?,
        Some(other) => anyhow::bail!(
            "config.json `intermediate_size` must be u32 or array, got {other:?}"
        ),
        None => anyhow::bail!("config.json missing intermediate_size"),
    };
    let vocab_size = u32_key("vocab_size")?;
    let head_dim = u32_key("head_dim").unwrap_or(hidden_size / num_attention_heads);
    let rope_theta = f32_key("rope_theta").unwrap_or(10_000.0);
    let rope_scaling = c.get("rope_scaling");
    let rope_scale = rope_scaling
        .and_then(|v| v.get("factor"))
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(1.0);
    // `rope_type` (current HF) with `type` (older configs) as fallback.
    let rope_scaling_type = rope_scaling
        .and_then(|v| v.get("rope_type").or_else(|| v.get("type")))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let rs_f32 = |k: &str| {
        rope_scaling
            .and_then(|v| v.get(k))
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(0.0)
    };
    let rope_low_freq_factor = rs_f32("low_freq_factor");
    let rope_high_freq_factor = rs_f32("high_freq_factor");
    let rope_original_max_pos = rope_scaling
        .and_then(|v| v.get("original_max_position_embeddings"))
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
        .unwrap_or(0);
    let rms_norm_eps = f32_key("rms_norm_eps").unwrap_or(1e-6);
    let tie_word_embeddings = bool_key("tie_word_embeddings").unwrap_or(false);

    // Optional fields populated when present (Qwen3-MoE, Mixtral, …).
    let max_position_embeddings = u32_key("max_position_embeddings").unwrap_or(0);
    // bos_token_id / eos_token_id may be a scalar (most archs) or an
    // array of multiple ids (Gemma 3: `eos_token_id: [1, 106]` covers
    // <eos> + <end_of_turn>; Llama-3 instruct: `[128001, 128008,
    // 128009]` covers <|end_of_text|> + <|eom_id|> + <|eot_id|>). Take
    // the FIRST element as the primary `eos_token_id` and put the rest
    // into `eos_token_ids` so the runtime can register multi-EOS.
    let token_id_all = |k: &str| -> Vec<u32> {
        let Some(v) = c.get(k) else { return Vec::new() };
        if let Some(n) = v.as_u64() {
            return vec![n as u32];
        }
        v.as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
            .unwrap_or_default()
    };
    let bos_ids = token_id_all("bos_token_id");
    let eos_ids = token_id_all("eos_token_id");
    let bos_token_id = bos_ids.first().copied().unwrap_or(0);
    let eos_token_id = eos_ids.first().copied().unwrap_or(0);
    let eos_token_ids: Vec<u32> = if eos_ids.len() > 1 { eos_ids[1..].to_vec() } else { Vec::new() };

    Ok(crate::ArchConfig {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        rope_theta,
        rope_scale,
        rope_scaling_type,
        rope_low_freq_factor,
        rope_high_freq_factor,
        rope_original_max_pos,
        rms_norm_eps,
        tie_word_embeddings,
        max_position_embeddings,
        bos_token_id,
        eos_token_id,
        eos_token_ids,
        ..crate::ArchConfig::default()
    })
}

impl GgufMapper for LlamaMapper {
    fn canonical_arch(&self) -> &'static str {
        "llama"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        let u32_key = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("missing metadata key: {k}"))
        };
        let f32_key = |k: &str| m.get(k).and_then(|v| v.as_f32());

        let hidden_size = u32_key("llama.embedding_length")?;
        let num_hidden_layers = u32_key("llama.block_count")?;
        let num_attention_heads = u32_key("llama.attention.head_count")?;
        let num_kv_heads = u32_key("llama.attention.head_count_kv").unwrap_or(num_attention_heads);
        let intermediate_size = u32_key("llama.feed_forward_length")?;
        let vocab_size = u32_key("llama.vocab_size")
            .or_else(|_| {
                // Some GGUFs don't store it explicitly; derive from tokenizer.
                m.get("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        KvValue::Array(a) => Some(a.len() as u32),
                        _ => None,
                    })
                    .context("no vocab_size and no tokenizer.ggml.tokens")
            })?;
        let head_dim = u32_key("llama.attention.key_length")
            .unwrap_or(hidden_size / num_attention_heads);

        let rope_theta = f32_key("llama.rope.freq_base").unwrap_or(10_000.0);
        let rope_scale = f32_key("llama.rope.scaling.factor").unwrap_or(1.0);
        let rms_norm_eps = f32_key("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let tie_word_embeddings = m
            .get("llama.tie_lm_head")
            .and_then(|v| match v {
                KvValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

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
            tie_word_embeddings,
            ..ArchConfig::default()
        })
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        map_llama_style(n)
    }
}

/// Llama / Qwen / most transformer GGUF models use this same layout —
/// only the config-key prefix differs. Extracting the mapping here lets
/// QwenMapper reuse it verbatim.
pub fn map_llama_style(n: &str) -> Option<String> {
    match n {
        "token_embd.weight" => Some("embed_tokens.weight".into()),
        "output.weight" => Some("lm_head.weight".into()),
        "output_norm.weight" => Some("final_norm.weight".into()),
        "rope_freqs.weight" => None, // drop — recomputed at runtime
        _ => {
            // blk.N.*
            let rest = n.strip_prefix("blk.")?;
            let (layer_str, rest) = rest.split_once('.')?;
            let layer: u32 = layer_str.parse().ok()?;
            let canonical_suffix = match rest {
                "attn_norm.weight" => "input_norm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                // Pre-fused qkv (Qwen3.5-style hybrid) — stored as one
                // [in, q_out + k_out + v_out] tensor. Runtime splits.
                "attn_qkv.weight" => "self_attn.qkv_proj.weight",
                // Attention gate (scalar-ish gating used in some hybrids).
                "attn_gate.weight" => "self_attn.gate.weight",
                "ffn_norm.weight" => "post_attn_norm.weight",
                // Gemma / Qwen3.5 post-attention norm variant.
                "post_attention_norm.weight" => "post_attn_norm.weight",
                "post_ffw_norm.weight" => "post_mlp_norm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                // MoE expert stacks — shape [in, out, num_experts] in GGUF.
                "ffn_gate_exps.weight" => "mlp.experts.gate_proj.weight",
                "ffn_up_exps.weight" => "mlp.experts.up_proj.weight",
                "ffn_down_exps.weight" => "mlp.experts.down_proj.weight",
                "ffn_gate_inp.weight" => "mlp.router.weight",
                // MoE shared experts (some variants).
                "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
                "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
                "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
                // SSM (Mamba / Mamba2 / hybrid SSM+attn like Qwen3.5 and
                // Qwen3.6). The `ssm_a` tensor is the state-transition
                // matrix; loader enforces f32 + cpu region via the
                // SSM_A_MATRIX flag set at convert time.
                "ssm_a" => "ssm.a_log",
                "ssm_d" => "ssm.d",
                "ssm_alpha.weight" => "ssm.alpha_proj.weight",
                "ssm_beta.weight" => "ssm.beta_proj.weight",
                "ssm_in.weight" => "ssm.in_proj.weight",
                "ssm_out.weight" => "ssm.out_proj.weight",
                "ssm_x.weight" => "ssm.x_proj.weight",
                "ssm_conv1d.weight" => "ssm.conv1d.weight",
                "ssm_conv1d.bias" => "ssm.conv1d.bias",
                "ssm_dt.weight" => "ssm.dt_proj.weight",
                "ssm_dt.bias" => "ssm.dt_bias",
                "ssm_norm.weight" => "ssm.norm.weight",
                _ => return None, // unknown — better to fail loud than drop
            };
            Some(format!("layers.{layer}.{canonical_suffix}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_core_llama_names() {
        assert_eq!(
            map_llama_style("token_embd.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            map_llama_style("blk.7.attn_q.weight"),
            Some("layers.7.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            map_llama_style("blk.0.ffn_down.weight"),
            Some("layers.0.mlp.down_proj.weight".to_string())
        );
        assert_eq!(
            map_llama_style("output_norm.weight"),
            Some("final_norm.weight".to_string())
        );
    }

    #[test]
    fn drops_rope_freqs() {
        assert_eq!(map_llama_style("rope_freqs.weight"), None);
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(map_llama_style("weird.custom.tensor"), None);
        assert_eq!(map_llama_style("blk.0.weird"), None);
    }

    /// `eos_token_id` is a scalar in most archs but an array on Gemma 3
    /// (`[1, 106]` covers both `<eos>` and `<end_of_turn>`). The
    /// previous `as_u64()` extraction returned None for arrays and the
    /// .base header silently landed with `eos_token_id = 0`. Take the
    /// first array element instead.
    #[test]
    fn eos_token_id_accepts_array() {
        let cfg = serde_json::json!({
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "intermediate_size": 6912,
            "vocab_size": 262144,
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6,
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
        });
        let c = hf_generic_config(&cfg).unwrap();
        assert_eq!(c.bos_token_id, 2, "bos_token_id scalar form still works");
        assert_eq!(c.eos_token_id, 1, "eos_token_id array → first element");
        assert_eq!(c.eos_token_ids, vec![106u32], "trailing eos ids land in eos_token_ids");
    }

    /// Llama-3 instruct: `eos_token_id: [128001, 128008, 128009]`. Primary
    /// `<|end_of_text|>` lands in `eos_token_id`; `<|eom_id|>` and
    /// `<|eot_id|>` go into `eos_token_ids` for runtime multi-EOS.
    #[test]
    fn eos_token_id_three_element_array() {
        let cfg = serde_json::json!({
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "rope_theta": 500_000.0,
            "rms_norm_eps": 1e-5,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
        });
        let c = hf_generic_config(&cfg).unwrap();
        assert_eq!(c.bos_token_id, 128000);
        assert_eq!(c.eos_token_id, 128001);
        assert_eq!(c.eos_token_ids, vec![128008u32, 128009u32]);
    }

    #[test]
    fn eos_token_id_scalar_form_unchanged() {
        let cfg = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "rope_theta": 500_000.0,
            "rms_norm_eps": 1e-5,
            "bos_token_id": 128000,
            "eos_token_id": 128009,
        });
        let c = hf_generic_config(&cfg).unwrap();
        assert_eq!(c.bos_token_id, 128000);
        assert_eq!(c.eos_token_id, 128009);
    }
}
