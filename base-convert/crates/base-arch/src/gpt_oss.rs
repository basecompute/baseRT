//! OpenAI gpt-oss (gpt-oss-20b / gpt-oss-120b) — HF `model_type = "gpt_oss"`.
//!
//! Decoder-only MoE transformer with a few departures from the Llama /
//! Qwen family the runtime has to know about (all carried in the header
//! `config`, never inferred from the arch string):
//!
//!   - alternating sliding-window (128) / full attention layers
//!     (`layer_types`), mirrored into `swa_layers` + `sliding_window`;
//!   - learned per-head attention **sinks** (an extra softmax logit per
//!     head that absorbs probability mass; tensor
//!     `self_attn.sinks`), biased q/k/v/o projections;
//!   - YaRN RoPE (factor 32 over 4096 original positions, beta_fast 32,
//!     beta_slow 1) — the betas are emitted so the runtime can derive
//!     the per-pair divisors and the attention-temperature `mscale`;
//!   - MoE: 32 routed experts, top-4, biased linear router, softmax over
//!     the selected experts (== `norm_topk_prob`), experts with biases on
//!     every projection and a clamped SwiGLU
//!     `(up + 1) * gate * sigmoid(1.702 * gate)` with
//!     `gate <= limit`, `|up| <= limit` (`swiglu_limit`, 7.0);
//!   - expert weights shipped as MXFP4 (E2M1 + E8M0 block scales), which
//!     the converter transplants verbatim (see `convert_gpt_oss` in
//!     base-convert).
//!
//! Tensor names are already canonical HF names; the converter's gpt-oss
//! path maps them itself (the fused `gate_up_proj` stays fused and
//! row-interleaved exactly as the checkpoint stores it — the runtime's
//! gpt-oss expert kernel reads gate = row 2j, up = row 2j+1).

use anyhow::Result;

use crate::{ArchConfig, HfMapper};

pub struct GptOssHfMapper;

impl HfMapper for GptOssHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "gpt_oss"
    }

    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        let mut config = crate::llama::hf_generic_config(c)?;
        let u32_v = |k: &str| c.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_v = |k: &str| c.get(k).and_then(|v| v.as_f64()).map(|f| f as f32);

        // MoE topology. `num_local_experts` is the HF key; `experts_per_token`
        // duplicates `num_experts_per_tok` on gpt-oss configs.
        config.num_experts = u32_v("num_local_experts").unwrap_or(0);
        config.num_experts_per_tok = u32_v("num_experts_per_tok")
            .or_else(|| u32_v("experts_per_token"))
            .unwrap_or(4);
        // Per-expert FFN width == intermediate_size (there is no dense FFN).
        config.moe_intermediate_size = config.intermediate_size;
        // Router: top-k over the raw logits, then softmax over the k winners —
        // algebraically the renormalized full softmax.
        config.norm_topk_prob = true;
        config.num_shared_experts = 0;

        // Attention schedule: `layer_types` lists "sliding_attention" |
        // "full_attention" per layer (gpt-oss-20b: alternating, sliding first).
        config.sliding_window = u32_v("sliding_window").unwrap_or(128);
        if let Some(arr) = c.get("layer_types").and_then(|v| v.as_array()) {
            config.swa_layers = arr
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s == "sliding_attention")
                        .unwrap_or(false)
                })
                .collect();
            config.per_layer_attn = config
                .swa_layers
                .iter()
                .map(|&b| if b { "sliding" } else { "global" }.to_string())
                .collect();
        } else {
            // Default schedule from the reference implementation: even layers
            // sliding, odd layers full.
            config.swa_layers = (0..config.num_hidden_layers).map(|i| i % 2 == 0).collect();
            config.per_layer_attn = config
                .swa_layers
                .iter()
                .map(|&b| if b { "sliding" } else { "global" }.to_string())
                .collect();
        }

        // YaRN betas (HF `rope_scaling.beta_fast` / `beta_slow`); the factor and
        // original_max_position_embeddings are already parsed generically.
        if let Some(rs) = c.get("rope_scaling") {
            config.rope_yarn_beta_fast = rs
                .get("beta_fast")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32)
                .unwrap_or(32.0);
            config.rope_yarn_beta_slow = rs
                .get("beta_slow")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32)
                .unwrap_or(1.0);
            config.rope_yarn_truncate =
                rs.get("truncate").and_then(|v| v.as_bool()).unwrap_or(true);
            if config.rope_original_max_pos == 0 {
                config.rope_original_max_pos = u32_v("initial_context_length").unwrap_or(4096);
            }
        }

        // Clamped SwiGLU constants.
        config.swiglu_limit = f32_v("swiglu_limit").unwrap_or(7.0);
        config.swiglu_alpha = 1.702;
        config.attention_sinks = true;
        config.attention_bias = c
            .get("attention_bias")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gpt_oss_20b_config() {
        let c: serde_json::Value = serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 2880, "num_hidden_layers": 4, "num_attention_heads": 64,
            "num_key_value_heads": 8, "head_dim": 64, "intermediate_size": 2880,
            "vocab_size": 201088, "rope_theta": 150000, "rms_norm_eps": 1e-5,
            "num_local_experts": 32, "num_experts_per_tok": 4, "sliding_window": 128,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                             "original_max_position_embeddings": 4096, "truncate": true},
            "eos_token_id": 200002, "max_position_embeddings": 131072
        });
        let cfg = GptOssHfMapper.config_from_hf(&c).unwrap();
        assert_eq!(cfg.num_experts, 32);
        assert_eq!(cfg.num_experts_per_tok, 4);
        assert_eq!(cfg.moe_intermediate_size, 2880);
        assert!(cfg.norm_topk_prob);
        assert_eq!(cfg.swa_layers, vec![true, false, true, false]);
        assert_eq!(cfg.sliding_window, 128);
        assert_eq!(cfg.rope_scaling_type, "yarn");
        assert_eq!(cfg.rope_scale, 32.0);
        assert_eq!(cfg.rope_original_max_pos, 4096);
        assert_eq!(cfg.rope_yarn_beta_fast, 32.0);
        assert_eq!(cfg.rope_yarn_beta_slow, 1.0);
        assert!(cfg.rope_yarn_truncate);
        assert_eq!(cfg.swiglu_limit, 7.0);
        let m = cfg.to_config_map();
        assert_eq!(m["num_experts"], 32);
        assert_eq!(m["rope_yarn_beta_fast"], 32.0);
        assert_eq!(m["swiglu_limit"], 7.0);
        assert_eq!(m["attention_sinks"], true);
    }
}
