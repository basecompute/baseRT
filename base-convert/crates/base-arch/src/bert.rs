//! BERT-family GGUF mapper. Currently scoped to nomic-bert (the
//! embedding model that ships with `general.architecture = "nomic-bert"`
//! in the wild). Modern BERT variants are LLaMA-flavored: SwiGLU FFN,
//! RoPE, fused QKV — but with bidirectional attention and per-layer
//! LayerNorm bias on top of the weight.
//!
//! The runtime expects the BERT-specific norms to keep their GGUF
//! names (`token_embd_norm`, `attn_output_norm`, `layer_output_norm`)
//! because the layer-index semantics shift between the source name and
//! the runtime query (`blk.N.layer_output_norm` → `layers.{N+1}.attention_norm`).
//! Resolving that cross-layer aliasing is the runtime's job; the
//! converter just makes the bytes available with predictable names.

use crate::{ArchConfig, GgufMapper};
use anyhow::{Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct NomicBertMapper;
pub struct NomicBertHfMapper;

impl crate::HfMapper for NomicBertHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "nomic-bert"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        // Nomic uses GPT-2-style key names (n_embd / n_head / n_inner)
        // rather than the LLaMA convention (hidden_size / num_attention_heads).
        let u32_key = |k: &str| c.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| c.get(k).and_then(|v| v.as_f64()).map(|f| f as f32);
        let hidden_size = u32_key("n_embd")
            .or_else(|| u32_key("hidden_size"))
            .context("nomic-bert: missing n_embd / hidden_size")?;
        let num_hidden_layers = u32_key("n_layer")
            .or_else(|| u32_key("num_hidden_layers"))
            .context("nomic-bert: missing n_layer / num_hidden_layers")?;
        let num_attention_heads = u32_key("n_head")
            .or_else(|| u32_key("num_attention_heads"))
            .context("nomic-bert: missing n_head / num_attention_heads")?;
        let intermediate_size = u32_key("n_inner")
            .or_else(|| u32_key("intermediate_size"))
            .unwrap_or(hidden_size * 4);
        let head_dim = u32_key("head_dim").unwrap_or(hidden_size / num_attention_heads);
        let vocab_size = u32_key("vocab_size").context("nomic-bert: missing vocab_size")?;
        let rope_theta = f32_key("rotary_emb_base").unwrap_or(10_000.0);
        let rms_norm_eps = f32_key("layer_norm_epsilon").unwrap_or(1e-12);
        let max_position_embeddings = u32_key("max_trained_positions")
            .or_else(|| u32_key("max_position_embeddings"))
            .unwrap_or(0);
        // BERT-family tokenizers don't ship `bos_token_id` / `eos_token_id`
        // in config.json — they use `cls_token` (`[CLS]`) and `sep_token`
        // (`[SEP]`) which sit at canonical ids 101 / 102 across
        // bert-base-uncased, multilingual-bert, and nomic-embed-text-v1.5
        // (verified). The runtime expects bos / eos to flag the
        // beginning / end of an embedding-input sequence; without these,
        // bos / eos default to 1 / 2 (irrelevant ids in the BERT vocab)
        // and the embedding-prepare path produces degenerate inputs.
        // Read explicit `cls_token_id` / `sep_token_id` when present;
        // otherwise fall back to the BERT convention.
        let bos_token_id = u32_key("cls_token_id").unwrap_or(101);
        let eos_token_id = u32_key("sep_token_id").unwrap_or(102);
        Ok(ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads: num_attention_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta,
            rope_scale: 1.0,
            rms_norm_eps,
            tie_word_embeddings: false,
            max_position_embeddings,
            bos_token_id,
            eos_token_id,
            ..ArchConfig::default()
        })
    }
}

impl GgufMapper for NomicBertMapper {
    fn canonical_arch(&self) -> &'static str {
        "nomic-bert"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        let prefix = "nomic-bert";
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
        let head_dim = u32_key(&format!("{prefix}.attention.key_length"))
            .unwrap_or(hidden_size / num_attention_heads);
        let vocab_size = m
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                KvValue::Array(a) => Some(a.len() as u32),
                _ => None,
            })
            .context("nomic-bert: tokenizer.ggml.tokens missing")?;
        let rope_theta = f32_key(&format!("{prefix}.rope.freq_base")).unwrap_or(10_000.0);
        // nomic-bert's epsilon key uses `_epsilon` (LayerNorm) rather
        // than `_rms_epsilon` (RMSNorm) — keep the same struct field.
        let rms_norm_eps = f32_key(&format!("{prefix}.attention.layer_norm_epsilon"))
            .unwrap_or(1e-12);
        let max_position_embeddings = u32_key(&format!("{prefix}.context_length")).unwrap_or(0);
        let bos_token_id = u32_key("tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos_token_id = u32_key("tokenizer.ggml.eos_token_id").unwrap_or(0);

        Ok(ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta,
            rope_scale: 1.0,
            rms_norm_eps,
            tie_word_embeddings: false,
            max_position_embeddings,
            bos_token_id,
            eos_token_id,
            ..ArchConfig::default()
        })
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        match n {
            "token_embd.weight" => Some("embed_tokens.weight".into()),
            // BERT-specific globals — kept verbatim. The runtime reads
            // these directly via the file→file self-map.
            "token_embd_norm.weight" | "token_embd_norm.bias" | "token_types.weight" => {
                Some(n.to_string())
            }
            // Drop unsupported pooling / classifier heads if they ever
            // appear (none in the current nomic-bert GGUF).
            "rope_freqs.weight" => None,
            _ => {
                let rest = n.strip_prefix("blk.")?;
                let (layer_str, suffix) = rest.split_once('.')?;
                let layer: u32 = layer_str.parse().ok()?;
                // BERT-specific suffixes keep their GGUF name; runtime
                // (src/core/models/bert.cpp) queries the same names
                // verbatim with a `layers.N.` prefix.
                let canonical = match suffix {
                    "attn_qkv.weight" => "self_attn.qkv_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    // LayerNorm pairs (weight + bias) — names preserved
                    // verbatim. The runtime resolves the cross-layer
                    // semantics (`layer_output_norm` of blk.N becomes
                    // the input norm for layer N+1) directly.
                    "attn_output_norm.weight"
                    | "attn_output_norm.bias"
                    | "layer_output_norm.weight"
                    | "layer_output_norm.bias" => suffix,
                    _ => return None,
                };
                Some(format!("layers.{layer}.{canonical}"))
            }
        }
    }
}
