//! Gemma 3 / Gemma 4 GGUF mapper.
//!
//! Gemma shares the Llama-style tensor naming convention (blk.N.attn_q,
//! etc.) with a few variations:
//! - `blk.N.attn_q_norm` / `blk.N.attn_k_norm` for the per-head QK norms
//!   (map to `self_attn.q_norm` / `self_attn.k_norm`)
//! - `blk.N.post_attention_norm` for the post-attention norm
//! - Config keys are prefixed with `gemma.` / `gemma2.` / `gemma3.` /
//!   `gemma4.`

use crate::llama::map_llama_style;
use crate::{ArchConfig, GgufMapper};
use anyhow::{Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct Gemma3Mapper;
pub struct Gemma4Mapper;
pub struct Gemma3HfMapper;
pub struct Gemma4HfMapper;

impl crate::HfMapper for Gemma3HfMapper {
    fn canonical_arch(&self) -> &'static str {
        "gemma3"
    }

    /// Gemma 3 stores zero-centered RMSNorm gamma and applies
    /// `rmsnorm(x) * (1 + weight)` at inference. Bake the +1 into
    /// every `*norm.weight` tensor at convert time so the runtime
    /// rmsnorm kernel (which computes `rmsnorm(x) * weight`) produces
    /// the right result. Mirrors llama.cpp's `Gemma3Model.norm_shift`.
    /// Gemma 4 dropped `add_unit_offset`, so its mapper keeps the
    /// trait default of 0.0.
    fn norm_shift(&self, canonical: &str) -> f32 {
        if canonical.ends_with("norm.weight") {
            1.0
        } else {
            0.0
        }
    }

    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        // Gemma HF often nests the text config under "text_config". Prefer
        // that sub-object when present.
        let source = c.get("text_config").unwrap_or(c);
        let mut config = crate::llama::hf_generic_config(source)?;
        config.tie_word_embeddings = source
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Gemma 3 sliding-window fields. The runtime needs these to
        // dispatch SWA layers correctly. Source keys mirror
        // gemma-3-1b-it/config.json verbatim.
        let u32_key = |k: &str| source.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| source.get(k).and_then(|v| v.as_f64()).map(|f| f as f32);
        config.sliding_window = u32_key("sliding_window").unwrap_or(0);
        config.sliding_window_pattern = u32_key("sliding_window_pattern").unwrap_or(0);
        config.rope_local_theta = f32_key("rope_local_base_freq").unwrap_or(0.0);
        config.logit_softcap = f32_key("final_logit_softcapping").unwrap_or(0.0);
        // `f_attention_scale = 1 / sqrt(query_pre_attn_scalar)` — applied
        // by llama.cpp's `gemma3.cpp` to Q before attention (line 74).
        // Across the published Gemma 3 family, `query_pre_attn_scalar`
        // equals `head_dim` for 1b/4b/12b (all 256), so the runtime's
        // `1 / sqrt(head_dim)` fallback lands on the same value; only
        // gemma-3-27b diverges (head_dim=128, qpas=168) and needs the
        // explicit override. Emit the explicit scale when the HF config
        // carries it and it differs from the runtime default; skip the
        // no-op write when they coincide so existing 1b bundles keep
        // `attention_scale=0` in the header (no header churn).
        // Reads via `as_f64()` so a float-typed qpas (e.g. 168.0) is
        // accepted — Gemma 3 stores it as int today, but the HF schema
        // for derived configs sometimes stringifies as a float.
        let qpas = source
            .get("query_pre_attn_scalar")
            .and_then(|v| v.as_f64())
            .map(|f| f as u32)
            .filter(|&n| n > 0);
        if let Some(qpas) = qpas {
            if qpas != config.head_dim {
                config.attention_scale = 1.0 / (qpas as f32).sqrt();
            }
        }
        Ok(config)
    }
}

impl crate::HfMapper for Gemma4HfMapper {
    fn canonical_arch(&self) -> &'static str {
        "gemma4"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        // Gemma-4 multimodal configs nest text under "text_config". For
        // the primary model weights we use that sub-config; audio/vision
        // tower configs are siblings handled by the mmproj path.
        let source = c.get("text_config").unwrap_or(c);
        let mut config = crate::llama::hf_generic_config(source)?;
        config.tie_word_embeddings = source
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Gemma-4-specific fields — keys matched against the actual HF
        // text_config (gemma-4-e2b-it-4bit/config.json). The earlier
        // pass guessed `query_pre_attn_scalar` and `rope_local_base_freq`
        // — neither key exists; `query_pre_attn_scalar` is a separate
        // attention-scale attribute (=1.0 for Gemma 4) and the SWA rope
        // theta lives under the nested `rope_parameters` block.
        let u32_key = |k: &str| source.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| source.get(k).and_then(|v| v.as_f64()).map(|f| f as f32);

        // llama.cpp hardcodes f_attention_scale = 1.0 for gemma4 (not
        // 1/sqrt(head_dim) like other archs). Matches the GGUF mapper.
        config.attention_scale = 1.0;
        config.n_embd_per_layer = u32_key("hidden_size_per_layer_input").unwrap_or(0);
        // Gemma 3n stores `intermediate_size` as a per-layer u32 array
        // when FFN width varies layer-to-layer. hf_generic_config above
        // already extracted the first entry as the scalar fallback;
        // mirror the full array into per_layer_ffn so the runtime sees
        // each layer's actual width. Older Gemma 4 variants (E2B, E4B)
        // also use this when per-layer widths vary.
        if let Some(arr) = source.get("intermediate_size").and_then(|v| v.as_array()) {
            config.per_layer_ffn = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect();
        }
        // Gemma 4: per-layer head_dim differs between Global and SWA.
        // `head_dim` (HF) = SWA head dim (256 on E2B/E4B);
        // `global_head_dim` (HF) = Global head dim (512 on E2B/E4B).
        config.head_dim_swa = u32_key("head_dim").unwrap_or(config.head_dim);
        config.head_dim_global =
            u32_key("global_head_dim").unwrap_or(config.head_dim_swa);
        let num_kv_shared = u32_key("num_kv_shared_layers").unwrap_or(0);
        config.n_layer_kv_from_start = if num_kv_shared > 0 && num_kv_shared < config.num_hidden_layers {
            config.num_hidden_layers - num_kv_shared
        } else {
            config.num_hidden_layers
        };
        config.logit_softcap = f32_key("final_logit_softcapping").unwrap_or(0.0);
        config.sliding_window = u32_key("sliding_window").unwrap_or(0);

        // Rope theta and partial-rotary factor are nested under
        // `rope_parameters.{full_attention, sliding_attention}` on
        // Gemma 4. `hf_generic_config` already pulled the top-level
        // `rope_theta` (defaulted to 10000.0 since the key isn't there)
        // — overwrite from the nested block.
        if let Some(rope_params) = source.get("rope_parameters") {
            if let Some(full) = rope_params.get("full_attention") {
                if let Some(theta) = full.get("rope_theta").and_then(|v| v.as_f64()) {
                    config.rope_theta = theta as f32;
                }
                if let Some(prf) =
                    full.get("partial_rotary_factor").and_then(|v| v.as_f64())
                {
                    config.global_rope_partial_factor = prf as f32;
                }
            }
            if let Some(sw) = rope_params.get("sliding_attention") {
                if let Some(theta) = sw.get("rope_theta").and_then(|v| v.as_f64()) {
                    config.rope_local_theta = theta as f32;
                }
            }
        }
        // Per-layer SWA pattern. HF stores `layer_types` (string array
        // of "sliding_attention" | "full_attention") or, on older
        // checkpoints, `sliding_window_pattern` (bool array). Mirror it
        // into both `swa_layers` (bitfield consumed by the runtime
        // gemma4 layer helpers) and `per_layer_attn` (string list kept
        // for header parity with the GGUF mapper, which extracts the
        // same data from `gemma4.attention.sliding_window_pattern`).
        if let Some(arr) = source.get("layer_types").and_then(|v| v.as_array()) {
            config.swa_layers = arr
                .iter()
                .map(|v| v.as_str().map(|s| s == "sliding_attention").unwrap_or(false))
                .collect();
            config.per_layer_attn = arr
                .iter()
                .map(|v| {
                    if v.as_str() == Some("sliding_attention") {
                        "sliding".to_string()
                    } else {
                        "global".to_string()
                    }
                })
                .collect();
        } else if let Some(arr) = source
            .get("sliding_window_pattern")
            .and_then(|v| v.as_array())
        {
            config.swa_layers = arr
                .iter()
                .map(|v| v.as_bool().unwrap_or(false))
                .collect();
            config.per_layer_attn = config
                .swa_layers
                .iter()
                .map(|&b| if b { "sliding" } else { "global" }.to_string())
                .collect();
        }
        if config.head_dim_global > 0 || config.head_dim_swa > 0 {
            config.head_dim = config.head_dim_global.max(config.head_dim_swa);
        }

        if let Some(global_kv) = u32_key("num_global_key_value_heads") {
            if !config.swa_layers.is_empty() {
                let swa_kv = config.num_kv_heads;
                config.n_kv_heads_per_layer = config
                    .swa_layers
                    .iter()
                    .map(|&is_swa| if is_swa { swa_kv } else { global_kv })
                    .collect();
            }
        }

        // MoE fields (Gemma 4 26B-A4B). Mirrors the GGUF Gemma 4 path
        // (which reads via `gemma4.expert_*`); HF stores these as the
        // standard Mixtral-style keys. Without these reads the .base
        // header silently drops num_experts / moe_intermediate_size from
        // HF-source bundles even though the GGUF path emits them.
        if let Some(n) = u32_key("num_experts") {
            config.num_experts = n;
        }
        // Gemma 4 names this `top_k_experts` (not the Mixtral/Qwen
        // `num_experts_per_tok`). Accept both — silently leaving it 0
        // makes the router never select any experts and the routed FFN
        // contributes nothing, so generation drifts even though shared
        // FFN works.
        if let Some(n) = u32_key("top_k_experts").or_else(|| u32_key("num_experts_per_tok")) {
            config.num_experts_per_tok = n;
        }
        if let Some(n) = u32_key("moe_intermediate_size") {
            config.moe_intermediate_size = n;
        }
        // Gemma normalizes top-k routing weights = false. Match the
        // GGUF default (norm_topk_prob isn't a GGUF metadata key).
        Ok(config)
    }
}

fn extract_config(m: &BTreeMap<String, KvValue>, prefix: &str) -> Result<ArchConfig> {
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
    // Gemma 4 26B-A4B stores `head_count_kv` as a per-layer Array
    // (typically 8 for SWA layers, 2 for Global). Other variants
    // store a single u32. Keep both: a uniform fallback and a
    // per-layer vector serialized through ArchConfig.
    let kv_key = format!("{prefix}.attention.head_count_kv");
    let (num_kv_heads, n_kv_heads_per_layer) = match m.get(&kv_key) {
        Some(KvValue::Array(arr)) => {
            let per_layer: Vec<u32> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect();
            let default_kv = per_layer.iter().max().copied().unwrap_or(num_attention_heads);
            (default_kv, per_layer)
        }
        Some(_) => (
            u32_key(&kv_key).unwrap_or(num_attention_heads),
            Vec::new(),
        ),
        None => (num_attention_heads, Vec::new()),
    };

    // Gemma-4 E2B stores `feed_forward_length` as a per-layer Array;
    // other variants store a single u32. Handle both.
    let ffn_key = format!("{prefix}.feed_forward_length");
    let (intermediate_size, per_layer_ffn) = match m.get(&ffn_key) {
        Some(KvValue::Array(arr)) => {
            let per_layer: Vec<u32> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect();
            let default_size = per_layer.first().copied().unwrap_or(0);
            (default_size, per_layer)
        }
        Some(_) => (u32_key(&ffn_key)?, Vec::new()),
        None => anyhow::bail!("missing metadata key: {ffn_key}"),
    };

    // Per-layer attention pattern (Gemma-4 E2B: sliding_window_pattern
    // Array of layer kinds). Coarsely map to "global" | "sliding".
    let per_layer_attn = m
        .get(&format!("{prefix}.attention.sliding_window_pattern"))
        .and_then(|v| match v {
            KvValue::Array(a) => Some(
                a.iter()
                    .map(|v| match v {
                        KvValue::Bool(true) => "sliding".to_string(),
                        KvValue::Bool(false) => "global".to_string(),
                        KvValue::U8(0) | KvValue::U32(0) => "global".to_string(),
                        _ => "sliding".to_string(),
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_default();
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

    // MoE fields (Gemma 4 26B-A4B / future MoE variants). 0 = dense.
    let num_experts = u32_key(&format!("{prefix}.expert_count")).unwrap_or(0);
    let num_experts_per_tok = u32_key(&format!("{prefix}.expert_used_count")).unwrap_or(0);
    let moe_intermediate_size =
        u32_key(&format!("{prefix}.expert_feed_forward_length")).unwrap_or(0);

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
        tie_word_embeddings: true, // Gemma typically ties embeddings
        per_layer_ffn,
        per_layer_attn,
        n_kv_heads_per_layer,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        // Gemma routes top-k experts via softmax without renormalizing
        // the top-k subset (unlike Qwen, which sets this to true).
        norm_topk_prob: false,
        max_position_embeddings: 0,
        bos_token_id: 0,
        eos_token_id: 0,
        ..ArchConfig::default()
    })
}

fn map_gemma_name(n: &str) -> Option<String> {
    // Gemma 4 PLE (per-layer-embedding) globals are passed through with
    // their GGUF names — the runtime reads `per_layer_token_embd.weight`,
    // `per_layer_model_proj.weight`, `per_layer_proj_norm.weight` directly.
    if matches!(
        n,
        "per_layer_token_embd.weight"
            | "per_layer_model_proj.weight"
            | "per_layer_proj_norm.weight"
    ) {
        return Some(n.to_string());
    }
    // Gemma 4 ships `rope_freqs.weight` as the partial-rotary divisor
    // mask for global-attention layers. The Llama mapper drops it (Llama
    // recomputes rope at runtime); Gemma 4 needs it preserved so the
    // global-rope synthesis can use the per-pair factors verbatim.
    if n == "rope_freqs.weight" {
        return Some(n.to_string());
    }

    if let Some(rest) = n.strip_prefix("blk.") {
        if let Some((layer_str, suffix)) = rest.split_once('.') {
            if let Ok(layer) = layer_str.parse::<u32>() {
                let canonical = match suffix {
                    // Gemma's pre-residual post-attention norm is a SEPARATE
                    // tensor from the pre-FFN `ffn_norm` that Llama renames
                    // to `post_attn_norm`. Keeping the GGUF name verbatim
                    // avoids the canonical-name collision that left one of
                    // the two norms unreadable at runtime.
                    "post_attention_norm.weight" => Some("post_attention_norm.weight"),
                    "post_ffw_norm.weight" => Some("post_ffw_norm.weight"),
                    // Gemma 4 PLE per-layer tensors (E2B / E4B / 26B-A4B).
                    "inp_gate.weight" => Some("per_layer_inp_gate.weight"),
                    "proj.weight" => Some("per_layer_proj.weight"),
                    "post_norm.weight" => Some("per_layer_post_norm.weight"),
                    "layer_output_scale.weight" => Some("layer_out_scale.weight"),
                    // Gemma 4 26B-A4B MoE: shared-vs-routed FFN wrappers
                    // and the fused gate+up MoE expert tensor. The runtime
                    // reads these names verbatim — see gemma4.cpp's MoE
                    // forward pass and moe_helpers.cpp.
                    "post_ffw_norm_1.weight" => Some("post_ffw_norm_1.weight"),
                    "post_ffw_norm_2.weight" => Some("post_ffw_norm_2.weight"),
                    "pre_ffw_norm_2.weight" => Some("pre_ffw_norm_2.weight"),
                    "ffn_gate_up_exps.weight" => Some("ffn_gate_up_exps.weight"),
                    // Per-layer router scale (F32 [dim]) — folded into
                    // the MoE rmsnorm at runtime.
                    "ffn_gate_inp.scale" => Some("ffn_gate_inp.scale"),
                    // Per-expert post-multiplier on the down output
                    // (F32 [n_experts]). Gemma-4-26B-A4B-specific.
                    "ffn_down_exps.scale" => Some("ffn_down_exps.scale"),
                    _ => None,
                };
                if let Some(c) = canonical {
                    return Some(format!("layers.{layer}.{c}"));
                }
            }
        }
    }
    map_llama_style(n)
}

impl GgufMapper for Gemma3Mapper {
    fn canonical_arch(&self) -> &'static str {
        "gemma3"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        // Prefer gemma3 keys, fall back to gemma2 then gemma.
        let prefix = if m.keys().any(|k| k.starts_with("gemma3.")) {
            "gemma3"
        } else if m.keys().any(|k| k.starts_with("gemma2.")) {
            "gemma2"
        } else {
            "gemma"
        };
        extract_config(m, prefix)
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        map_gemma_name(n)
    }
}

impl GgufMapper for Gemma4Mapper {
    fn canonical_arch(&self) -> &'static str {
        "gemma4"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        let mut config = extract_config(m, "gemma4")?;

        // ── Gemma-4-specific config fields ───────────────────────────
        // The runtime requires these for the gemma4 model path; without
        // them BaseWeightStore reports zeros and `gemma4.cpp` aborts on
        // `cfg.attention_scale must be > 0`.
        let u32_key = |k: &str| m.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| m.get(k).and_then(|v| v.as_f32());

        // llama.cpp hardcodes f_attention_scale = 1.0 for gemma4 (not
        // 1/sqrt(head_dim) like other archs).
        config.attention_scale = 1.0;

        config.n_embd_per_layer =
            u32_key("gemma4.embedding_length_per_layer_input").unwrap_or(0);

        config.head_dim_global =
            u32_key("gemma4.attention.key_length").unwrap_or(config.head_dim);
        config.head_dim_swa =
            u32_key("gemma4.attention.key_length_swa").unwrap_or(config.head_dim_global);

        // shared_kv_layers = number of LATE layers that reuse an earlier
        // layer's KV cache. n_layer_kv_from_start is therefore
        // (n_layers - shared_kv_layers).
        let n_shared_kv = u32_key("gemma4.attention.shared_kv_layers").unwrap_or(0);
        config.n_layer_kv_from_start =
            if n_shared_kv > 0 && n_shared_kv < config.num_hidden_layers {
                config.num_hidden_layers - n_shared_kv
            } else {
                config.num_hidden_layers
            };

        config.logit_softcap =
            f32_key("gemma4.final_logit_softcapping").unwrap_or(0.0);
        config.sliding_window =
            u32_key("gemma4.attention.sliding_window").unwrap_or(0);
        config.rope_local_theta = f32_key("gemma4.rope.local.freq_base")
            .or_else(|| f32_key("gemma4.rope.freq_base_swa"))
            .unwrap_or(0.0);

        // Partial-rotary factor for global-attention layers. HF stores it
        // as a fraction (`rope_parameters.full_attention.partial_rotary_factor`,
        // typically 0.5 on E2B/E4B and 1.0 on 26B-A4B); GGUF stores the
        // rotated dimension count directly. Derive the fraction so the
        // header is symmetric across sources — avoids the runtime
        // synthesizing wrong partial RoPE when `rope_freqs.weight` is
        // absent (MLX-source bundles skip that tensor).
        if let Some(rope_dim) = u32_key("gemma4.rope.dimension_count") {
            if config.head_dim_global > 0 {
                config.global_rope_partial_factor =
                    rope_dim as f32 / config.head_dim_global as f32;
            }
        }

        // Per-layer SWA bitfield (Gemma 4 stores a bool array under
        // `attention.sliding_window_pattern` rather than the periodic
        // u32 Gemma 3 uses).
        if let Some(KvValue::Array(arr)) = m.get("gemma4.attention.sliding_window_pattern") {
            let layers: Vec<bool> = arr
                .iter()
                .map(|v| match v {
                    KvValue::Bool(b) => *b,
                    KvValue::U8(n) => *n != 0,
                    KvValue::U32(n) => *n != 0,
                    _ => false,
                })
                .collect();
            if !layers.is_empty() {
                config.swa_layers = layers;
            }
        }

        // When head_dim varies per layer, head_dim is the max for
        // scratch-buffer sizing (matches what the runtime expects).
        if config.head_dim_global > 0 || config.head_dim_swa > 0 {
            config.head_dim = config.head_dim_global.max(config.head_dim_swa);
        }

        Ok(config)
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        map_gemma_name(n)
    }
}

/// Map a Gemma 4 multimodal tower / projector tensor name (HuggingFace
/// safetensors layout) to its baseRT runtime-canonical form. Returns
/// `None` when the name doesn't match any known Gemma 4 mmproj tensor.
///
/// Why this lives in the converter: the runtime previously carried ~280
/// lines of `BaseNameRule` tables to do this remap at load time, which
/// pushed Gemma-4-specific architecture into arch-agnostic loader code
/// and would collide the moment a second multimodal arch (PaliGemma,
/// Llama 3.2 Vision) hit the same path. Doing the rename once at
/// conversion keeps the runtime arch-agnostic — `BaseWeightStore` just
/// self-maps the canonical names as it does for the LM body.
pub fn map_gemma4_mmproj_name(n: &str) -> Option<String> {
    // ── Vision tower ────────────────────────────────────────────────
    if let Some(rest) = n.strip_prefix("vision_tower.") {
        // Patch embedder + factorized positional embedding.
        match rest {
            "patch_embedder.input_proj.weight" => return Some("vision.patch_embed.weight".into()),
            "patch_embedder.input_proj.input_max" => return Some("vision.patch_embed.input_max".into()),
            "patch_embedder.input_proj.input_min" => return Some("vision.patch_embed.input_min".into()),
            "patch_embedder.input_proj.output_max" => return Some("vision.patch_embed.output_max".into()),
            "patch_embedder.input_proj.output_min" => return Some("vision.patch_embed.output_min".into()),
            "patch_embedder.position_embedding_table" => return Some("vision.pos_embed.weight".into()),
            _ => {}
        }
        // encoder.layers.{n}.<suffix>
        if let Some(layer_rest) = rest.strip_prefix("encoder.layers.") {
            if let Some((idx_str, suffix)) = layer_rest.split_once('.') {
                if idx_str.parse::<u32>().is_ok() {
                    let canonical_suffix = vision_layer_suffix(suffix)?;
                    return Some(format!("vision.layers.{idx_str}.{canonical_suffix}"));
                }
            }
        }
        return None;
    }
    // ── Vision projection head (lives outside vision_tower in HF) ──
    if n == "embed_vision.embedding_projection.weight" {
        return Some("vision.projection.weight".into());
    }

    // ── Audio tower ─────────────────────────────────────────────────
    if let Some(rest) = n.strip_prefix("audio_tower.") {
        // SubSampleConvProjection (front-end).
        match rest {
            "subsample_conv_projection.layer0.conv.weight" => return Some("audio.sscp.layer0.conv.weight".into()),
            "subsample_conv_projection.layer0.norm.weight" => return Some("audio.sscp.layer0.norm.weight".into()),
            "subsample_conv_projection.layer1.conv.weight" => return Some("audio.sscp.layer1.conv.weight".into()),
            "subsample_conv_projection.layer1.norm.weight" => return Some("audio.sscp.layer1.norm.weight".into()),
            "subsample_conv_projection.input_proj_linear.weight" => return Some("audio.sscp.proj.weight".into()),
            "output_proj.weight" => return Some("audio.output_proj.weight".into()),
            "output_proj.bias" => return Some("audio.output_proj.bias".into()),
            _ => {}
        }
        // layers.{n}.<suffix>
        if let Some(layer_rest) = rest.strip_prefix("layers.") {
            if let Some((idx_str, suffix)) = layer_rest.split_once('.') {
                if idx_str.parse::<u32>().is_ok() {
                    let canonical_suffix = audio_layer_suffix(suffix)?;
                    return Some(format!("audio.layers.{idx_str}.{canonical_suffix}"));
                }
            }
        }
        return None;
    }
    // ── Audio projection head ───────────────────────────────────────
    if n == "embed_audio.embedding_projection.weight" {
        return Some("audio.projection.weight".into());
    }

    None
}

/// Vision-layer suffix rename: HF `<sub>.<part>` → runtime `<canonical>`.
/// Returns `None` if no rule matches (caller passes the tensor through
/// verbatim, which the runtime will then ignore).
fn vision_layer_suffix(s: &str) -> Option<String> {
    // Norms.
    match s {
        "input_layernorm.weight" => return Some("attention_norm.weight".into()),
        "pre_feedforward_layernorm.weight" => return Some("ffn_norm.weight".into()),
        "post_attention_layernorm.weight" => return Some("post_attention_norm.weight".into()),
        "post_feedforward_layernorm.weight" => return Some("post_ffw_norm.weight".into()),
        _ => {}
    }
    // Self-attention: q/k/v/o proj and qk-norm.
    if let Some(out) = strip_clipped_proj(s, "self_attn", &[
        ("q_proj", "attention.q"),
        ("k_proj", "attention.k"),
        ("v_proj", "attention.v"),
        ("o_proj", "attention.output"),
    ]) {
        return Some(out);
    }
    if s == "self_attn.q_norm.weight" {
        return Some("attention.q_norm.weight".into());
    }
    if s == "self_attn.k_norm.weight" {
        return Some("attention.k_norm.weight".into());
    }
    // MLP (GeGLU): gate / up / down with clipped-linear bounds.
    if let Some(out) = strip_clipped_proj(s, "mlp", &[
        ("gate_proj", "ffn.gate"),
        ("up_proj", "ffn.up"),
        ("down_proj", "ffn.down"),
    ]) {
        return Some(out);
    }
    None
}

/// Audio-layer suffix rename. Mirrors `vision_layer_suffix` but for the
/// Conformer block names Gemma 4's audio tower uses.
fn audio_layer_suffix(s: &str) -> Option<String> {
    // Feed-forward macroblock 1 / 2 (clipped-linear linears + pre/post norm).
    for (idx, prefix) in [(1u32, "feed_forward1"), (2u32, "feed_forward2")] {
        if let Some(rest) = s.strip_prefix(&format!("{prefix}.")) {
            match rest {
                "pre_layer_norm.weight" => return Some(format!("ffw{idx}.pre_norm.weight")),
                "post_layer_norm.weight" => return Some(format!("ffw{idx}.post_norm.weight")),
                _ => {}
            }
            if let Some(out) = clipped_proj_suffix(rest, "ffw_layer_1", &format!("ffw{idx}.up")) {
                return Some(out);
            }
            if let Some(out) = clipped_proj_suffix(rest, "ffw_layer_2", &format!("ffw{idx}.down")) {
                return Some(out);
            }
        }
    }
    // Self-attention (incl. relative-position k-proj + per-dim scale).
    if let Some(rest) = s.strip_prefix("self_attn.") {
        if let Some(out) = clipped_proj_suffix(rest, "q_proj", "attn.q") {
            return Some(out);
        }
        if let Some(out) = clipped_proj_suffix(rest, "k_proj", "attn.k") {
            return Some(out);
        }
        if let Some(out) = clipped_proj_suffix(rest, "v_proj", "attn.v") {
            return Some(out);
        }
        if let Some(out) = clipped_proj_suffix(rest, "post", "attn.output") {
            return Some(out);
        }
        match rest {
            "per_dim_scale" => return Some("attn.per_dim_scale".into()),
            "relative_k_proj.weight" => return Some("attn.rel_pos_proj.weight".into()),
            _ => {}
        }
    }
    // Block-level norms (already canonical, just route through).
    match s {
        "norm_pre_attn.weight" => return Some("norm_pre_attn.weight".into()),
        "norm_post_attn.weight" => return Some("norm_post_attn.weight".into()),
        "norm_out.weight" => return Some("norm_out.weight".into()),
        _ => {}
    }
    // LightConv1d block.
    if let Some(rest) = s.strip_prefix("lconv1d.") {
        match rest {
            "pre_layer_norm.weight" => return Some("lconv.pre_norm.weight".into()),
            "depthwise_conv1d.weight" => return Some("lconv.dw_conv.weight".into()),
            "conv_norm.weight" => return Some("lconv.conv_norm.weight".into()),
            _ => {}
        }
        if let Some(out) = clipped_proj_suffix(rest, "linear_start", "lconv.linear_start") {
            return Some(out);
        }
        if let Some(out) = clipped_proj_suffix(rest, "linear_end", "lconv.linear_end") {
            return Some(out);
        }
    }
    None
}

/// HF clipped-linear modules ship as
/// `<group>.<proj>.<linear|input_max|input_min|output_max|output_min>`
/// where the runtime expects
/// `<canonical>.{weight,input_max,input_min,output_max,output_min}`.
/// Walks a list of `(hf_proj, runtime_canonical)` pairs and returns the
/// first match. Returns `None` if no proj matches.
fn strip_clipped_proj(s: &str, group: &str, projs: &[(&str, &str)]) -> Option<String> {
    let group_rest = s.strip_prefix(&format!("{group}."))?;
    for (hf_proj, canonical) in projs {
        if let Some(out) = clipped_proj_suffix(group_rest, hf_proj, canonical) {
            return Some(out);
        }
    }
    None
}

/// `<hf_proj>.linear.weight` → `<canonical>.weight`,
/// `<hf_proj>.{input,output}_{min,max}` → `<canonical>.{input,output}_{min,max}`.
fn clipped_proj_suffix(s: &str, hf_proj: &str, canonical: &str) -> Option<String> {
    let rest = s.strip_prefix(&format!("{hf_proj}."))?;
    match rest {
        "linear.weight" => Some(format!("{canonical}.weight")),
        "input_max" => Some(format!("{canonical}.input_max")),
        "input_min" => Some(format!("{canonical}.input_min")),
        "output_max" => Some(format!("{canonical}.output_max")),
        "output_min" => Some(format!("{canonical}.output_min")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HfMapper;
    use base_readers::gguf::KvValue;
    use serde_json::json;

    /// Equivalent HF and GGUF metadata for Gemma 4 26B-A4B should produce
    /// identical `ArchConfig` for the load-bearing fields. Sister test to
    /// the Qwen3-MoE parity check; covers the asymmetries surfaced
    /// during the Gemma drift investigation:
    ///   - HF read `partial_rotary_factor`; GGUF didn't (now derived from
    ///     `gemma4.rope.dimension_count / head_dim_global`).
    ///
    /// This catches header-level divergence between sources but does NOT
    /// catch numerical drift caused by re-quant precision misalignment
    /// (Q5_0/Q8_0 32-element source blocks regrouped into 64-element
    /// base_q4 groups). That class needs an end-to-end inference
    /// baseline test, separate work.
    #[test]
    fn gemma4_26b_a4b_hf_gguf_parity() {
        let hf_json = json!({
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "global_head_dim": 512,
                "intermediate_size": 2112,
                "vocab_size": 262144,
                "rms_norm_eps": 1e-06,
                "tie_word_embeddings": true,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 704,
                "sliding_window": 1024,
                "final_logit_softcapping": 30.0,
                "rope_parameters": {
                    "full_attention": {
                        "rope_theta": 1_000_000.0,
                        "partial_rotary_factor": 1.0,
                    },
                    "sliding_attention": { "rope_theta": 10_000.0 },
                },
                // 5 sliding then 1 global, repeated.
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention","full_attention",
                ],
            },
        });
        let hf = Gemma4HfMapper.config_from_hf(&hf_json).unwrap();

        let mut g = std::collections::BTreeMap::new();
        g.insert("gemma4.embedding_length".into(), KvValue::U32(2816));
        g.insert("gemma4.block_count".into(), KvValue::U32(30));
        g.insert("gemma4.attention.head_count".into(), KvValue::U32(16));
        // GGUF stores per-layer KV head counts as Array on 26B-A4B.
        g.insert(
            "gemma4.attention.head_count_kv".into(),
            KvValue::Array(
                std::iter::repeat([
                    KvValue::U32(8),
                    KvValue::U32(8),
                    KvValue::U32(8),
                    KvValue::U32(8),
                    KvValue::U32(8),
                    KvValue::U32(2),
                ])
                .take(5)
                .flatten()
                .collect(),
            ),
        );
        g.insert("gemma4.feed_forward_length".into(), KvValue::U32(2112));
        g.insert(
            "gemma4.expert_feed_forward_length".into(),
            KvValue::U32(704),
        );
        g.insert("gemma4.attention.key_length".into(), KvValue::U32(512));
        g.insert("gemma4.attention.value_length".into(), KvValue::U32(512));
        g.insert("gemma4.attention.key_length_swa".into(), KvValue::U32(256));
        g.insert("gemma4.attention.value_length_swa".into(), KvValue::U32(256));
        g.insert("gemma4.vocab_size".into(), KvValue::U32(262144));
        g.insert("gemma4.rope.freq_base".into(), KvValue::F32(1_000_000.0));
        g.insert("gemma4.rope.freq_base_swa".into(), KvValue::F32(10_000.0));
        g.insert("gemma4.rope.dimension_count".into(), KvValue::U32(512));
        g.insert("gemma4.rope.dimension_count_swa".into(), KvValue::U32(256));
        g.insert(
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            KvValue::F32(1e-06),
        );
        g.insert("gemma4.expert_count".into(), KvValue::U32(128));
        g.insert("gemma4.expert_used_count".into(), KvValue::U32(8));
        g.insert(
            "gemma4.final_logit_softcapping".into(),
            KvValue::F32(30.0),
        );
        g.insert("gemma4.attention.sliding_window".into(), KvValue::U32(1024));
        g.insert("gemma4.attention.shared_kv_layers".into(), KvValue::U32(0));
        g.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            KvValue::Array(
                std::iter::repeat([
                    KvValue::Bool(true),
                    KvValue::Bool(true),
                    KvValue::Bool(true),
                    KvValue::Bool(true),
                    KvValue::Bool(true),
                    KvValue::Bool(false),
                ])
                .take(5)
                .flatten()
                .collect(),
            ),
        );
        let gg = Gemma4Mapper.config_from_gguf(&g).unwrap();

        // Core dims.
        assert_eq!(hf.hidden_size, gg.hidden_size);
        assert_eq!(hf.num_hidden_layers, gg.num_hidden_layers);
        assert_eq!(hf.num_attention_heads, gg.num_attention_heads);
        assert_eq!(hf.num_kv_heads, gg.num_kv_heads);
        assert_eq!(hf.intermediate_size, gg.intermediate_size);
        assert_eq!(hf.vocab_size, gg.vocab_size);
        assert_eq!(hf.tie_word_embeddings, gg.tie_word_embeddings);

        // Gemma-4-specific.
        assert_eq!(hf.head_dim_global, gg.head_dim_global);
        assert_eq!(hf.head_dim_swa, gg.head_dim_swa);
        assert_eq!(hf.attention_scale, gg.attention_scale);
        assert_eq!(hf.attention_scale, 1.0);
        assert_eq!(hf.logit_softcap, gg.logit_softcap);
        assert_eq!(hf.sliding_window, gg.sliding_window);
        assert_eq!(hf.rope_local_theta, gg.rope_local_theta);
        assert_eq!(hf.rope_theta, gg.rope_theta);
        assert_eq!(hf.n_layer_kv_from_start, gg.n_layer_kv_from_start);

        // The asymmetry the Gemma probe surfaced: HF reads
        // partial_rotary_factor from the nested HF rope_parameters
        // block; GGUF used to emit 0.0 (silent default). Now derived
        // from `gemma4.rope.dimension_count / head_dim_global`.
        assert_eq!(hf.global_rope_partial_factor, gg.global_rope_partial_factor);
        assert_eq!(hf.global_rope_partial_factor, 1.0);

        // Per-layer SWA mask: both should reflect the 5-sliding-then-1-global pattern.
        assert_eq!(hf.swa_layers, gg.swa_layers);

        // MoE topology.
        assert_eq!(hf.num_experts, gg.num_experts);
        assert_eq!(hf.num_experts_per_tok, gg.num_experts_per_tok);
        assert_eq!(hf.moe_intermediate_size, gg.moe_intermediate_size);
        // Gemma is the false case (Qwen is true).
        assert_eq!(hf.norm_topk_prob, gg.norm_topk_prob);
        assert!(!hf.norm_topk_prob);
    }

    /// `Gemma3HfMapper` must add +1 to every `*norm.weight` so the
    /// runtime's plain rmsnorm kernel computes `rmsnorm(x) * (1 + w)`.
    /// Mirrors the suffix predicate in `convert_hf_to_gguf.py`'s
    /// `Gemma3Model.norm_shift`. Embeddings, projections, and biases
    /// (1-D non-norm) are unaffected.
    #[test]
    fn gemma3_norm_shift_predicate() {
        let m = Gemma3HfMapper;
        for s in [
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.pre_feedforward_layernorm.weight",
            "layers.0.post_feedforward_layernorm.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.25.input_norm.weight",
            "final_norm.weight",
            "model.norm.weight",
        ] {
            assert_eq!(m.norm_shift(s), 1.0, "should shift: {s}");
        }
        for s in [
            "embed_tokens.weight",
            "lm_head.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.input_layernorm.bias",  // hypothetical; norms have no bias in gemma3
        ] {
            assert_eq!(m.norm_shift(s), 0.0, "should not shift: {s}");
        }
    }

    /// `Gemma4HfMapper` must NOT shift — Gemma 4 dropped
    /// `add_unit_offset` (mirrors `Gemma4Model.norm_shift = 0.0`).
    #[test]
    fn gemma4_norm_shift_is_zero() {
        let m = Gemma4HfMapper;
        for s in [
            "layers.0.input_layernorm.weight",
            "layers.0.self_attn.q_norm.weight",
            "final_norm.weight",
        ] {
            assert_eq!(m.norm_shift(s), 0.0, "gemma4 must not shift: {s}");
        }
    }

    /// gemma-3-1b: query_pre_attn_scalar == head_dim == 256, so the
    /// derived `1/sqrt(head_dim)` already matches and `attention_scale`
    /// is left at 0 (no header churn for existing bundles).
    #[test]
    fn gemma3_attention_scale_skipped_when_equal_to_head_dim() {
        let cfg = serde_json::json!({
            "hidden_size": 1152, "num_hidden_layers": 26,
            "num_attention_heads": 4, "num_key_value_heads": 1,
            "head_dim": 256, "intermediate_size": 6912,
            "vocab_size": 262144, "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6, "query_pre_attn_scalar": 256,
        });
        let c = Gemma3HfMapper.config_from_hf(&cfg).unwrap();
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.attention_scale, 0.0, "no override when qpas == head_dim");
    }

    /// gemma-3-27b is the only published Gemma 3 variant where
    /// `query_pre_attn_scalar` (168) differs from `head_dim` (128).
    /// The mapper must emit an explicit attention_scale = 1/sqrt(qpas)
    /// so the runtime doesn't fall back to the wrong 1/sqrt(head_dim).
    /// Config snippet matches google/gemma-3-27b-it/config.json.
    #[test]
    fn gemma3_27b_attention_scale_emitted() {
        let cfg = serde_json::json!({
            "hidden_size": 5376, "num_hidden_layers": 62,
            "num_attention_heads": 32, "num_key_value_heads": 16,
            "head_dim": 128, "intermediate_size": 21504,
            "vocab_size": 262144, "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6, "query_pre_attn_scalar": 168,
        });
        let c = Gemma3HfMapper.config_from_hf(&cfg).unwrap();
        let want = 1.0_f32 / (168.0_f32).sqrt();
        assert!((c.attention_scale - want).abs() < 1e-6, "want {}, got {}", want, c.attention_scale);
    }

    /// Defensive: HF sometimes stores numeric scalars as floats. Make
    /// sure a float-typed `query_pre_attn_scalar` (e.g. 168.0) is read
    /// the same way as the integer-typed form.
    #[test]
    fn gemma3_attention_scale_accepts_float_qpas() {
        let cfg = serde_json::json!({
            "hidden_size": 5376, "num_hidden_layers": 62,
            "num_attention_heads": 32, "num_key_value_heads": 16,
            "head_dim": 128, "intermediate_size": 21504,
            "vocab_size": 262144, "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6, "query_pre_attn_scalar": 168.0,
        });
        let c = Gemma3HfMapper.config_from_hf(&cfg).unwrap();
        let want = 1.0_f32 / (168.0_f32).sqrt();
        assert!((c.attention_scale - want).abs() < 1e-6, "want {}, got {}", want, c.attention_scale);
    }
}
