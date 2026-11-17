//! GLM 5.2 (`glm-dsa`) GGUF → canonical `.base` mapping.
//!
//! GLM 5.2 is a DeepSeek-V3.2-style decoder:
//!   * Multi-head Latent Attention (MLA): compressed query (`attn_q_a` →
//!     `attn_q_a_norm` → `attn_q_b`) and compressed KV
//!     (`attn_kv_a_mqa` → `attn_kv_a_norm`, up-projected by `attn_k_b` /
//!     `attn_v_b`), with a decoupled `qk_rope_head_dim`-wide RoPE.
//!   * DeepSeek Sparse Attention (DSA) "lightning indexer"
//!     (`indexer.*`) selecting the top-`indexer_top_k` keys per query.
//!   * Sigmoid-gated MoE (256 experts, top-8) with a bias-corrected
//!     selection (`exp_probs_b`), routed-weight normalization + scaling,
//!     and one always-on shared expert. The first
//!     `leading_dense_block_count` layers use a dense SwiGLU FFN.
//!   * A Multi-Token-Prediction (`nextn`) head — dropped at convert time.
//!
//! Tensor names for the shared pieces (norms, o_proj, router, expert
//! stacks, shared expert, dense FFN, embeddings, output) follow the
//! Llama/Qwen convention, so we delegate those to `map_llama_style` and
//! only special-case the MLA + indexer tensors here.

use crate::llama::map_llama_style;
use crate::{ArchConfig, GgufMapper, HfMapper};
use anyhow::{bail, Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct GlmDsaMapper;

impl GgufMapper for GlmDsaMapper {
    fn canonical_arch(&self) -> &'static str {
        // Must contain "glm" — the runtime's arch_from_config() matches
        // on `strstr(arch, "glm")` to select the MLA + DSA encoder.
        "glm_dsa"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        let prefix = "glm-dsa";

        let u32_key = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("missing metadata key: {k}"))
        };
        let u32_opt = |k: &str| m.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| m.get(k).and_then(|v| v.as_f32());
        let bool_opt = |k: &str| m.get(k).and_then(|v| v.as_bool());

        let hidden_size = u32_key(&format!("{prefix}.embedding_length"))?;
        // GGUF `block_count` includes the trailing Multi-Token-Prediction
        // (MTP / nextn) block, which llama.cpp loads but never runs. The
        // real transformer stack is `block_count - nextn_predict_layers`.
        let block_count = u32_key(&format!("{prefix}.block_count"))?;
        let nextn_predict_layers = u32_opt(&format!("{prefix}.nextn_predict_layers")).unwrap_or(0);
        let num_hidden_layers = block_count.saturating_sub(nextn_predict_layers);
        let num_attention_heads = u32_key(&format!("{prefix}.attention.head_count"))?;
        let num_kv_heads =
            u32_opt(&format!("{prefix}.attention.head_count_kv")).unwrap_or(num_attention_heads);
        // Dense-FFN width (used by the leading dense layers).
        let intermediate_size = u32_key(&format!("{prefix}.feed_forward_length"))?;
        let vocab_size = u32_key(&format!("{prefix}.vocab_size")).or_else(|_| {
            m.get("tokenizer.ggml.tokens")
                .and_then(|v| match v {
                    KvValue::Array(a) => Some(a.len() as u32),
                    _ => None,
                })
                .context("no vocab_size and no tokenizer.ggml.tokens")
        })?;

        let rope_theta = f32_key(&format!("{prefix}.rope.freq_base")).unwrap_or(10_000.0);
        let rope_scale = f32_key(&format!("{prefix}.rope.scaling.factor")).unwrap_or(1.0);
        let rms_norm_eps =
            f32_key(&format!("{prefix}.attention.layer_norm_rms_epsilon")).unwrap_or(1e-6);

        // ── MLA geometry ────────────────────────────────────────────
        let q_lora_rank = u32_opt(&format!("{prefix}.attention.q_lora_rank")).unwrap_or(0);
        let kv_lora_rank = u32_opt(&format!("{prefix}.attention.kv_lora_rank")).unwrap_or(0);
        // Decoupled-RoPE width (the only rotated part of Q/K).
        let qk_rope_head_dim = u32_opt(&format!("{prefix}.rope.dimension_count")).unwrap_or(0);
        // Per-head expanded Q/K dim (`key_length_mla`) = nope + rope.
        let key_length_mla = u32_opt(&format!("{prefix}.attention.key_length_mla")).unwrap_or(0);
        let qk_nope_head_dim = key_length_mla.saturating_sub(qk_rope_head_dim);
        // Per-head value dim after v_b up-projection.
        let v_head_dim = u32_opt(&format!("{prefix}.attention.value_length_mla")).unwrap_or(0);
        // Generic head_dim slot: the full per-head Q/K dim. The MLA
        // encoder reads the specific fields above; this keeps the
        // header's derived q_dim/kv_dim sane for any generic consumer.
        let head_dim = if key_length_mla > 0 {
            key_length_mla
        } else {
            hidden_size / num_attention_heads
        };

        // ── MoE topology ────────────────────────────────────────────
        let num_experts = u32_opt(&format!("{prefix}.expert_count")).unwrap_or(0);
        let num_experts_per_tok = u32_opt(&format!("{prefix}.expert_used_count")).unwrap_or(0);
        let moe_intermediate_size =
            u32_opt(&format!("{prefix}.expert_feed_forward_length")).unwrap_or(0);
        let num_shared_experts = u32_opt(&format!("{prefix}.expert_shared_count")).unwrap_or(0);
        // GGUF `expert_gating_func`: 1 = softmax, 2 = sigmoid (DeepSeek/GLM).
        // Runtime `expert_gating`: 0 = softmax, 1 = sigmoid.
        let expert_gating_func = u32_opt(&format!("{prefix}.expert_gating_func")).unwrap_or(1);
        let expert_gating = if expert_gating_func == 2 { 1 } else { 0 };
        let routed_scaling_factor =
            f32_key(&format!("{prefix}.expert_weights_scale")).unwrap_or(0.0);
        let norm_topk_prob = bool_opt(&format!("{prefix}.expert_weights_norm")).unwrap_or(true);
        let first_k_dense_replace =
            u32_opt(&format!("{prefix}.leading_dense_block_count")).unwrap_or(0);

        // ── DSA lightning indexer ───────────────────────────────────
        let indexer_head_count =
            u32_opt(&format!("{prefix}.attention.indexer.head_count")).unwrap_or(0);
        let indexer_key_length =
            u32_opt(&format!("{prefix}.attention.indexer.key_length")).unwrap_or(0);
        let indexer_top_k = u32_opt(&format!("{prefix}.attention.indexer.top_k")).unwrap_or(0);

        // Token ids (best-effort; runtime also harvests from the tokenizer).
        let bos_token_id = u32_opt("tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos_token_id = u32_opt("tokenizer.ggml.eos_token_id").unwrap_or(0);
        let max_position_embeddings = u32_opt(&format!("{prefix}.context_length")).unwrap_or(0);

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
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            norm_topk_prob,
            num_shared_experts,
            max_position_embeddings,
            bos_token_id,
            eos_token_id,
            // MLA + sparse-attention + GLM-MoE extras.
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            routed_scaling_factor,
            expert_gating,
            first_k_dense_replace,
            nextn_predict_layers,
            indexer_head_count,
            indexer_key_length,
            indexer_top_k,
            ..ArchConfig::default()
        })
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        // Drop the Multi-Token-Prediction (nextn) head — not used for
        // standard next-token decoding. These tensors live on the trailing
        // MTP block; dropping the `nextn.*` projections is enough to skip
        // the head (the block's ordinary attn/ffn tensors are ignored by
        // the runtime, which only iterates the first `num_hidden_layers`).
        if n.contains(".nextn.") {
            return None;
        }

        // MLA + DSA-indexer tensors that map_llama_style doesn't know.
        if let Some(rest) = n.strip_prefix("blk.") {
            if let Some((layer_str, suffix)) = rest.split_once('.') {
                if let Ok(layer) = layer_str.parse::<u32>() {
                    let canonical_suffix = match suffix {
                        // ── MLA attention ──
                        "attn_q_a.weight" => Some("self_attn.q_a_proj.weight"),
                        "attn_q_a_norm.weight" => Some("self_attn.q_a_layernorm.weight"),
                        "attn_q_b.weight" => Some("self_attn.q_b_proj.weight"),
                        "attn_kv_a_mqa.weight" => Some("self_attn.kv_a_proj_with_mqa.weight"),
                        "attn_kv_a_norm.weight" => Some("self_attn.kv_a_layernorm.weight"),
                        "attn_k_b.weight" => Some("self_attn.k_b_proj.weight"),
                        "attn_v_b.weight" => Some("self_attn.v_b_proj.weight"),
                        // ── DSA lightning indexer ──
                        "indexer.attn_q_b.weight" => Some("self_attn.indexer.q_b_proj.weight"),
                        "indexer.attn_k.weight" => Some("self_attn.indexer.k_proj.weight"),
                        "indexer.k_norm.weight" => Some("self_attn.indexer.k_norm.weight"),
                        "indexer.k_norm.bias" => Some("self_attn.indexer.k_norm.bias"),
                        "indexer.proj.weight" => Some("self_attn.indexer.weights_proj.weight"),
                        // ── MoE bias-correction (sigmoid top-k selection) ──
                        "exp_probs_b.bias" => Some("mlp.gate.e_score_correction_bias"),
                        _ => None,
                    };
                    if let Some(s) = canonical_suffix {
                        return Some(format!("layers.{layer}.{s}"));
                    }
                }
            }
        }

        // Everything else (norms, o_proj, router, expert stacks, shared
        // expert, dense FFN, embeddings, output) is Llama-shaped.
        map_llama_style(n)
    }
}

/// HF/MLX `config.json` (`model_type: glm_moe_dsa`) → ArchConfig.
///
/// Mirrors [`GlmDsaMapper::config_from_gguf`] so a GGUF-sourced and an
/// MLX-sourced `.base` carry equivalent headers. Differences from the
/// GGUF path, on purpose:
///   * `num_hidden_layers` is already ex-MTP in HF configs (78; the
///     MLX export additionally sets `num_nextn_predict_layers: 0`
///     because it drops the MTP weights entirely).
///   * `num_kv_heads` is forced to 1 (the MQA latent). The HF config
///     says `num_key_value_heads: 64`, which describes the *expanded*
///     per-head view, but the GGUF path (and the runtime's MLA KV
///     sizing) use 1 — keep the headers consistent.
///   * The DSA full/shared layer pattern (`indexer_types` + friends)
///     only exists here; GGUF metadata doesn't carry it.
impl HfMapper for GlmDsaMapper {
    fn canonical_arch(&self) -> &'static str {
        // Same string as the GGUF side — the runtime dispatches on
        // `strstr(arch, "glm")`.
        "glm_dsa"
    }

    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        let u32_key = |k: &str| {
            c.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("config.json missing key: {k}"))
        };
        let u32_opt = |k: &str| c.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_opt = |k: &str| c.get(k).and_then(|v| v.as_f64()).map(|f| f as f32);
        let bool_opt = |k: &str| c.get(k).and_then(|v| v.as_bool());

        // The runtime's sigmoid+bias router kernel has no expert-group
        // masking; GLM 5.2 ships n_group = 1 so the MLX reference's
        // group step is a no-op. Refuse anything else loudly.
        let n_group = u32_opt("n_group").unwrap_or(1);
        if n_group > 1 {
            bail!(
                "glm_moe_dsa with n_group = {n_group} (grouped expert routing) is not \
                 supported — the runtime router has no group masking"
            );
        }
        if let Some(func) = c.get("scoring_func").and_then(|v| v.as_str()) {
            if func != "sigmoid" {
                bail!("glm_moe_dsa scoring_func {func:?} not supported (expected \"sigmoid\")");
            }
        }

        let qk_nope_head_dim = u32_key("qk_nope_head_dim")?;
        let qk_rope_head_dim = u32_key("qk_rope_head_dim")?;

        // `rope_theta` lives under `rope_parameters` on GLM 5.2.
        let rope_theta = c
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .or_else(|| f32_opt("rope_theta"))
            .unwrap_or(10_000.0);

        // `eos_token_id` is a list on GLM 5.2; first entry is primary,
        // the rest register as additional stop ids.
        let (eos_token_id, eos_token_ids) = match c.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> = a
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect();
                (
                    ids.first().copied().unwrap_or(0),
                    ids.get(1..).unwrap_or(&[]).to_vec(),
                )
            }
            Some(v) => (v.as_u64().unwrap_or(0) as u32, vec![]),
            None => (0, vec![]),
        };

        let indexer_layer_types: Vec<String> = c
            .get("indexer_types")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(ArchConfig {
            hidden_size: u32_key("hidden_size")?,
            num_hidden_layers: u32_key("num_hidden_layers")?,
            num_attention_heads: u32_key("num_attention_heads")?,
            num_kv_heads: 1,
            head_dim: qk_nope_head_dim + qk_rope_head_dim,
            intermediate_size: u32_key("intermediate_size")?,
            vocab_size: u32_key("vocab_size")?,
            rope_theta,
            rope_scale: 1.0,
            rms_norm_eps: f32_opt("rms_norm_eps").unwrap_or(1e-5),
            tie_word_embeddings: bool_opt("tie_word_embeddings").unwrap_or(false),
            num_experts: u32_opt("n_routed_experts").unwrap_or(0),
            num_experts_per_tok: u32_opt("num_experts_per_tok").unwrap_or(0),
            moe_intermediate_size: u32_opt("moe_intermediate_size").unwrap_or(0),
            norm_topk_prob: bool_opt("norm_topk_prob").unwrap_or(true),
            num_shared_experts: u32_opt("n_shared_experts").unwrap_or(0),
            max_position_embeddings: u32_opt("max_position_embeddings").unwrap_or(0),
            bos_token_id: 0,
            eos_token_id,
            eos_token_ids,
            // MLA geometry.
            q_lora_rank: u32_opt("q_lora_rank").unwrap_or(0),
            kv_lora_rank: u32_opt("kv_lora_rank").unwrap_or(0),
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim: u32_key("v_head_dim")?,
            routed_scaling_factor: f32_opt("routed_scaling_factor").unwrap_or(0.0),
            expert_gating: 1, // sigmoid (validated above)
            first_k_dense_replace: u32_opt("first_k_dense_replace").unwrap_or(0),
            nextn_predict_layers: u32_opt("num_nextn_predict_layers").unwrap_or(0),
            // DSA lightning indexer.
            indexer_head_count: u32_opt("index_n_heads").unwrap_or(0),
            indexer_key_length: u32_opt("index_head_dim").unwrap_or(0),
            indexer_top_k: u32_opt("index_topk").unwrap_or(0),
            indexer_layer_types,
            index_topk_freq: u32_opt("index_topk_freq").unwrap_or(0),
            index_skip_topk_offset: u32_opt("index_skip_topk_offset").unwrap_or(0),
            indexer_rope_interleave: bool_opt("indexer_rope_interleave").unwrap_or(false),
            index_share_for_mtp_iteration: bool_opt("index_share_for_mtp_iteration")
                .unwrap_or(false),
            ..ArchConfig::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glm_metadata() -> BTreeMap<String, KvValue> {
        let mut m = BTreeMap::new();
        let u = |v: u32| KvValue::U32(v);
        let f = |v: f32| KvValue::F32(v);
        m.insert("glm-dsa.embedding_length".into(), u(6144));
        m.insert("glm-dsa.block_count".into(), u(79));
        m.insert("glm-dsa.attention.head_count".into(), u(64));
        m.insert("glm-dsa.attention.head_count_kv".into(), u(1));
        m.insert("glm-dsa.feed_forward_length".into(), u(12288));
        m.insert("glm-dsa.vocab_size".into(), u(154880));
        m.insert("glm-dsa.attention.layer_norm_rms_epsilon".into(), f(1e-6));
        m.insert("glm-dsa.rope.freq_base".into(), f(8_000_000.0));
        m.insert("glm-dsa.rope.dimension_count".into(), u(64));
        m.insert("glm-dsa.attention.q_lora_rank".into(), u(2048));
        m.insert("glm-dsa.attention.kv_lora_rank".into(), u(512));
        m.insert("glm-dsa.attention.key_length".into(), u(576));
        m.insert("glm-dsa.attention.value_length".into(), u(512));
        m.insert("glm-dsa.attention.key_length_mla".into(), u(256));
        m.insert("glm-dsa.attention.value_length_mla".into(), u(256));
        m.insert("glm-dsa.expert_count".into(), u(256));
        m.insert("glm-dsa.expert_used_count".into(), u(8));
        m.insert("glm-dsa.expert_shared_count".into(), u(1));
        m.insert("glm-dsa.expert_feed_forward_length".into(), u(2048));
        m.insert("glm-dsa.expert_gating_func".into(), u(2));
        m.insert("glm-dsa.expert_weights_scale".into(), f(2.5));
        m.insert("glm-dsa.expert_weights_norm".into(), KvValue::Bool(true));
        m.insert("glm-dsa.leading_dense_block_count".into(), u(3));
        m.insert("glm-dsa.nextn_predict_layers".into(), u(1));
        m.insert("glm-dsa.attention.indexer.head_count".into(), u(32));
        m.insert("glm-dsa.attention.indexer.key_length".into(), u(128));
        m.insert("glm-dsa.attention.indexer.top_k".into(), u(2048));
        m
    }

    #[test]
    fn glm_dsa_config_from_gguf() {
        let c = GlmDsaMapper.config_from_gguf(&glm_metadata()).unwrap();
        assert_eq!(c.hidden_size, 6144);
        // block_count(79) - nextn_predict_layers(1) = 78 real layers.
        assert_eq!(c.num_hidden_layers, 78);
        assert_eq!(c.num_attention_heads, 64);
        assert_eq!(c.num_kv_heads, 1);
        assert_eq!(c.intermediate_size, 12288);
        assert_eq!(c.vocab_size, 154880);
        assert_eq!(c.rope_theta, 8_000_000.0);
        // MLA geometry.
        assert_eq!(c.q_lora_rank, 2048);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.qk_rope_head_dim, 64);
        assert_eq!(c.qk_nope_head_dim, 192); // 256 - 64
        assert_eq!(c.v_head_dim, 256);
        assert_eq!(c.head_dim, 256);
        // MoE.
        assert_eq!(c.num_experts, 256);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.num_shared_experts, 1);
        assert_eq!(c.moe_intermediate_size, 2048);
        assert_eq!(c.expert_gating, 1); // sigmoid
        assert_eq!(c.routed_scaling_factor, 2.5);
        assert!(c.norm_topk_prob);
        assert_eq!(c.first_k_dense_replace, 3);
        assert_eq!(c.nextn_predict_layers, 1);
        // Indexer.
        assert_eq!(c.indexer_head_count, 32);
        assert_eq!(c.indexer_key_length, 128);
        assert_eq!(c.indexer_top_k, 2048);
    }

    #[test]
    fn glm_dsa_config_round_trips_through_header() {
        let c = GlmDsaMapper.config_from_gguf(&glm_metadata()).unwrap();
        let m = c.to_config_map();
        use serde_json::json;
        assert_eq!(m["q_lora_rank"], json!(2048));
        assert_eq!(m["kv_lora_rank"], json!(512));
        assert_eq!(m["qk_nope_head_dim"], json!(192));
        assert_eq!(m["qk_rope_head_dim"], json!(64));
        assert_eq!(m["v_head_dim"], json!(256));
        assert_eq!(m["routed_scaling_factor"], json!(2.5));
        assert_eq!(m["expert_gating"], json!(1));
        assert_eq!(m["first_k_dense_replace"], json!(3));
        assert_eq!(m["num_experts"], json!(256));
        assert_eq!(m["num_shared_experts"], json!(1));
        assert_eq!(m["moe_intermediate_size"], json!(2048));
        assert_eq!(m["indexer_head_count"], json!(32));
        assert_eq!(m["indexer_top_k"], json!(2048));
    }

    #[test]
    fn glm_dsa_maps_mla_and_indexer_tensors() {
        let map = |n: &str| GlmDsaMapper.map_tensor_name(n);
        // MLA.
        assert_eq!(
            map("blk.5.attn_q_a.weight").as_deref(),
            Some("layers.5.self_attn.q_a_proj.weight")
        );
        assert_eq!(
            map("blk.5.attn_q_a_norm.weight").as_deref(),
            Some("layers.5.self_attn.q_a_layernorm.weight")
        );
        assert_eq!(
            map("blk.5.attn_kv_a_mqa.weight").as_deref(),
            Some("layers.5.self_attn.kv_a_proj_with_mqa.weight")
        );
        assert_eq!(
            map("blk.5.attn_k_b.weight").as_deref(),
            Some("layers.5.self_attn.k_b_proj.weight")
        );
        assert_eq!(
            map("blk.5.attn_v_b.weight").as_deref(),
            Some("layers.5.self_attn.v_b_proj.weight")
        );
        // Indexer.
        assert_eq!(
            map("blk.5.indexer.attn_q_b.weight").as_deref(),
            Some("layers.5.self_attn.indexer.q_b_proj.weight")
        );
        assert_eq!(
            map("blk.5.indexer.k_norm.bias").as_deref(),
            Some("layers.5.self_attn.indexer.k_norm.bias")
        );
        assert_eq!(
            map("blk.5.indexer.proj.weight").as_deref(),
            Some("layers.5.self_attn.indexer.weights_proj.weight")
        );
        // MoE bias.
        assert_eq!(
            map("blk.5.exp_probs_b.bias").as_deref(),
            Some("layers.5.mlp.gate.e_score_correction_bias")
        );
    }

    #[test]
    fn glm_dsa_config_from_hf_mlx_checkpoint() {
        // Mirrors mlx-community/GLM-5.2-4bit's config.json (trimmed).
        let c = serde_json::json!({
            "model_type": "glm_moe_dsa",
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "head_dim": 192,
            "qk_head_dim": 256,
            "intermediate_size": 12288,
            "vocab_size": 154880,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "first_k_dense_replace": 3,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
            "max_position_embeddings": 1048576,
            "eos_token_id": [154820, 154827, 154829],
            "tie_word_embeddings": false,
            "num_nextn_predict_layers": 0,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "indexer_rope_interleave": true,
            "index_share_for_mtp_iteration": true,
            "indexer_types": ["full", "full", "full", "shared"],
        });
        let cfg = HfMapper::config_from_hf(&GlmDsaMapper, &c).unwrap();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 78);
        assert_eq!(cfg.num_kv_heads, 1); // forced MQA-latent, not the HF 64
        assert_eq!(cfg.head_dim, 256); // nope 192 + rope 64
        assert_eq!(cfg.rope_theta, 8_000_000.0);
        assert_eq!(cfg.rms_norm_eps, 1e-5);
        assert_eq!(cfg.q_lora_rank, 2048);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.v_head_dim, 256);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.num_shared_experts, 1);
        assert_eq!(cfg.expert_gating, 1);
        assert_eq!(cfg.routed_scaling_factor, 2.5);
        assert_eq!(cfg.first_k_dense_replace, 3);
        assert_eq!(cfg.nextn_predict_layers, 0);
        assert_eq!(cfg.eos_token_id, 154820);
        assert_eq!(cfg.eos_token_ids, vec![154827, 154829]);
        assert_eq!(cfg.indexer_head_count, 32);
        assert_eq!(cfg.indexer_key_length, 128);
        assert_eq!(cfg.indexer_top_k, 2048);
        assert_eq!(cfg.indexer_layer_types.len(), 4);
        assert_eq!(cfg.index_topk_freq, 4);
        assert_eq!(cfg.index_skip_topk_offset, 3);
        assert!(cfg.indexer_rope_interleave);
        assert!(cfg.index_share_for_mtp_iteration);
        // The pattern fields round-trip through the header map.
        let m = cfg.to_config_map();
        assert_eq!(m["indexer_layer_types"][3], serde_json::json!("shared"));
        assert_eq!(m["index_topk_freq"], serde_json::json!(4));
        assert_eq!(m["indexer_rope_interleave"], serde_json::json!(true));
    }

    #[test]
    fn glm_dsa_config_from_hf_rejects_grouped_routing() {
        let c = serde_json::json!({
            "hidden_size": 6144, "num_hidden_layers": 78,
            "num_attention_heads": 64, "intermediate_size": 12288,
            "vocab_size": 154880, "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64, "v_head_dim": 256,
            "n_group": 8, "scoring_func": "sigmoid",
        });
        assert!(HfMapper::config_from_hf(&GlmDsaMapper, &c).is_err());
    }

    #[test]
    fn glm_dsa_delegates_shared_tensors_and_drops_mtp() {
        let map = |n: &str| GlmDsaMapper.map_tensor_name(n);
        // Shared (Llama-shaped) tensors delegate to map_llama_style.
        assert_eq!(
            map("blk.5.attn_output.weight").as_deref(),
            Some("layers.5.self_attn.o_proj.weight")
        );
        assert_eq!(
            map("blk.5.ffn_gate_inp.weight").as_deref(),
            Some("layers.5.mlp.router.weight")
        );
        assert_eq!(
            map("blk.5.ffn_down_exps.weight").as_deref(),
            Some("layers.5.mlp.experts.down_proj.weight")
        );
        assert_eq!(
            map("blk.5.ffn_up_shexp.weight").as_deref(),
            Some("layers.5.mlp.shared_expert.up_proj.weight")
        );
        assert_eq!(
            map("blk.1.ffn_gate.weight").as_deref(),
            Some("layers.1.mlp.gate_proj.weight")
        );
        assert_eq!(
            map("token_embd.weight").as_deref(),
            Some("embed_tokens.weight")
        );
        // MTP / nextn head dropped.
        assert_eq!(map("blk.79.nextn.eh_proj.weight"), None);
        assert_eq!(map("blk.79.nextn.enorm.weight"), None);
    }
}
