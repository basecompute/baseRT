//! Whisper (OpenAI encoder-decoder speech) HF mapper.
//!
//! Converts `model_type: "whisper"` HF safetensors repos
//! (openai/whisper-tiny … large-v3-turbo, incl. the `.en` variants) to
//! `.base` per the whisper bundle contract:
//!
//! - Tensor names are renamed to the engine-native whisper convention
//!   (`encoder.blocks.N.attn.query.weight`, `decoder.blocks.N.cross_attn.…`).
//! - Default (no profile) bundles ship every tensor as f16. With a quant
//!   profile (`profiles/whisper-q8.json` / `whisper-q4.json`) ONLY the
//!   2-D linear projection weights (`is_quantizable_linear`) carry the
//!   canonical packed scheme; conv1/conv2, positional embeddings,
//!   token_embedding, norms and biases stay f16 always — the convert
//!   path in base-convert enforces that; this module only does
//!   naming/config/token-metadata + the eligibility predicate.
//! - Linear weights stay `[out, in]` row-major, conv weights
//!   `[out, in, k]` C-order. No qkv fusion, no transpose.
//! - `lm_head`/`proj_out` is tied to `decoder.token_embedding.weight`
//!   and is NOT stored (mapped to None; the header sets TIED_EMBEDDINGS).
//! - Special-token IDs are derived from the repo's `tokenizer.json`
//!   `added_tokens` — never from a hardcoded vocab-size table. That table
//!   lives only in this module's tests (the historic bug this fixes:
//!   large-v3's `<|yue|>` shifts every id after 50357 by one).

use crate::{ArchConfig, HfMapper};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub struct WhisperHfMapper;

impl HfMapper for WhisperHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "whisper"
    }

    fn config_from_hf(&self, c: &Value) -> Result<ArchConfig> {
        let u32_key = |k: &str| {
            c.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("whisper: config.json missing {k}"))
        };
        let d_model = u32_key("d_model")?;
        let decoder_layers = u32_key("decoder_layers")?;
        let decoder_attention_heads = u32_key("decoder_attention_heads")?;
        let decoder_ffn_dim = u32_key("decoder_ffn_dim")?;
        let encoder_layers = u32_key("encoder_layers")?;
        let encoder_attention_heads = u32_key("encoder_attention_heads")?;
        let encoder_ffn_dim = u32_key("encoder_ffn_dim")?;
        let num_mel_bins = u32_key("num_mel_bins")?;
        let max_source_positions = u32_key("max_source_positions")?;
        let max_target_positions = u32_key("max_target_positions")?;
        let vocab_size = u32_key("vocab_size")?;
        if decoder_attention_heads == 0 || d_model % decoder_attention_heads != 0 {
            bail!(
                "whisper: d_model={d_model} not divisible by decoder_attention_heads={decoder_attention_heads}"
            );
        }
        let bos_token_id = c.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let eos_token_id = c.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        Ok(ArchConfig {
            hidden_size: d_model,
            num_hidden_layers: decoder_layers,
            num_attention_heads: decoder_attention_heads,
            num_kv_heads: decoder_attention_heads,
            head_dim: d_model / decoder_attention_heads,
            intermediate_size: decoder_ffn_dim,
            vocab_size,
            rope_theta: 0.0, // whisper has no RoPE (learned absolute positions)
            rope_scale: 1.0,
            // Whisper uses LayerNorm with eps 1e-5; the shared struct
            // field carries it and to_config_map emits it as `norm_eps`.
            rms_norm_eps: 1e-5,
            tie_word_embeddings: true,
            max_position_embeddings: max_target_positions,
            bos_token_id,
            eos_token_id,
            encoder_layers,
            encoder_attention_heads,
            encoder_ffn_dim,
            decoder_ffn_dim,
            num_mel_bins,
            max_source_positions,
            max_target_positions,
            ..ArchConfig::default()
        })
    }
}

/// Map an HF whisper safetensors tensor name to the engine-native bundle
/// name. Returns None for tensors that must be dropped (`proj_out` /
/// `lm_head` — tied to `decoder.token_embedding.weight`; the loader
/// resolves the alias, so storing it would double the largest tensor).
pub fn map_hf_tensor_name(name: &str) -> Option<String> {
    // Some exports wrap everything in `model.`; others don't. Accept both.
    let n = name.strip_prefix("model.").unwrap_or(name);

    // lm_head is tied to the token embedding — never stored.
    if n == "proj_out.weight" || n == "lm_head.weight" {
        return None;
    }

    match n {
        "encoder.conv1.weight" => return Some("encoder.conv1.weight".into()),
        "encoder.conv1.bias" => return Some("encoder.conv1.bias".into()),
        "encoder.conv2.weight" => return Some("encoder.conv2.weight".into()),
        "encoder.conv2.bias" => return Some("encoder.conv2.bias".into()),
        "encoder.embed_positions.weight" => return Some("encoder.positional_embedding".into()),
        "encoder.layer_norm.weight" => return Some("encoder.ln_post.weight".into()),
        "encoder.layer_norm.bias" => return Some("encoder.ln_post.bias".into()),
        "decoder.embed_tokens.weight" => return Some("decoder.token_embedding.weight".into()),
        "decoder.embed_positions.weight" => return Some("decoder.positional_embedding".into()),
        "decoder.layer_norm.weight" => return Some("decoder.ln.weight".into()),
        "decoder.layer_norm.bias" => return Some("decoder.ln.bias".into()),
        _ => {}
    }

    // Per-layer tensors: {encoder,decoder}.layers.N.<suffix>
    let (side, rest) = if let Some(r) = n.strip_prefix("encoder.layers.") {
        ("encoder", r)
    } else {
        ("decoder", n.strip_prefix("decoder.layers.")?)
    };
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;

    let canonical = match suffix {
        "self_attn.q_proj.weight" => "attn.query.weight",
        "self_attn.q_proj.bias" => "attn.query.bias",
        "self_attn.k_proj.weight" => "attn.key.weight",
        // Whisper's key projection has no bias; if a checkpoint ships an
        // all-zero one anyway, drop it (the engine never reads it).
        "self_attn.k_proj.bias" => return None,
        "self_attn.v_proj.weight" => "attn.value.weight",
        "self_attn.v_proj.bias" => "attn.value.bias",
        "self_attn.out_proj.weight" => "attn.out.weight",
        "self_attn.out_proj.bias" => "attn.out.bias",
        "self_attn_layer_norm.weight" => "attn_ln.weight",
        "self_attn_layer_norm.bias" => "attn_ln.bias",
        "encoder_attn.q_proj.weight" => "cross_attn.query.weight",
        "encoder_attn.q_proj.bias" => "cross_attn.query.bias",
        "encoder_attn.k_proj.weight" => "cross_attn.key.weight",
        "encoder_attn.k_proj.bias" => return None,
        "encoder_attn.v_proj.weight" => "cross_attn.value.weight",
        "encoder_attn.v_proj.bias" => "cross_attn.value.bias",
        "encoder_attn.out_proj.weight" => "cross_attn.out.weight",
        "encoder_attn.out_proj.bias" => "cross_attn.out.bias",
        "encoder_attn_layer_norm.weight" => "cross_attn_ln.weight",
        "encoder_attn_layer_norm.bias" => "cross_attn_ln.bias",
        "fc1.weight" => "mlp.0.weight",
        "fc1.bias" => "mlp.0.bias",
        "fc2.weight" => "mlp.2.weight",
        "fc2.bias" => "mlp.2.bias",
        "final_layer_norm.weight" => "mlp_ln.weight",
        "final_layer_norm.bias" => "mlp_ln.bias",
        _ => return None,
    };
    // Cross-attention only exists on the decoder side.
    if side == "encoder" && canonical.starts_with("cross_attn") {
        return None;
    }
    Some(format!("{side}.blocks.{layer}.{canonical}"))
}

/// True for canonical whisper tensor names that are quant-eligible:
/// the 2-D linear projection weights of the transformer blocks
/// (`attn`/`cross_attn` query/key/value/out and `mlp.0`/`mlp.2`,
/// encoder and decoder). Everything else — conv1/conv2 (3-D),
/// positional embeddings, `token_embedding`, every norm, every bias —
/// must stay f16: the engine's whisper conv/norm/embedding kernels
/// read half, and quantizing per-channel norm gains wrecks the model.
pub fn is_quantizable_linear(canonical: &str) -> bool {
    let Some(rest) = canonical
        .strip_prefix("encoder.blocks.")
        .or_else(|| canonical.strip_prefix("decoder.blocks."))
    else {
        return false;
    };
    let Some((_layer, suffix)) = rest.split_once('.') else {
        return false;
    };
    matches!(
        suffix,
        "attn.query.weight"
            | "attn.key.weight"
            | "attn.value.weight"
            | "attn.out.weight"
            | "cross_attn.query.weight"
            | "cross_attn.key.weight"
            | "cross_attn.value.weight"
            | "cross_attn.out.weight"
            | "mlp.0.weight"
            | "mlp.2.weight"
    )
}

/// Every tensor the whisper bundle contract requires, derived from the
/// encoder/decoder depths in `cfg`. Conversion fails loud if any of
/// these is missing from the mapped source set.
pub fn required_tensor_names(cfg: &ArchConfig) -> Vec<String> {
    let mut v: Vec<String> = vec![
        "encoder.conv1.weight".into(),
        "encoder.conv1.bias".into(),
        "encoder.conv2.weight".into(),
        "encoder.conv2.bias".into(),
        "encoder.positional_embedding".into(),
        "encoder.ln_post.weight".into(),
        "encoder.ln_post.bias".into(),
        "decoder.token_embedding.weight".into(),
        "decoder.positional_embedding".into(),
        "decoder.ln.weight".into(),
        "decoder.ln.bias".into(),
    ];
    let attn_block = |v: &mut Vec<String>, prefix: &str, attn: &str| {
        for t in [
            "query.weight",
            "query.bias",
            "key.weight",
            "value.weight",
            "value.bias",
            "out.weight",
            "out.bias",
        ] {
            v.push(format!("{prefix}.{attn}.{t}"));
        }
        v.push(format!("{prefix}.{attn}_ln.weight"));
        v.push(format!("{prefix}.{attn}_ln.bias"));
    };
    let mlp_block = |v: &mut Vec<String>, prefix: &str| {
        for t in [
            "mlp.0.weight",
            "mlp.0.bias",
            "mlp.2.weight",
            "mlp.2.bias",
            "mlp_ln.weight",
            "mlp_ln.bias",
        ] {
            v.push(format!("{prefix}.{t}"));
        }
    };
    for l in 0..cfg.encoder_layers {
        let p = format!("encoder.blocks.{l}");
        attn_block(&mut v, &p, "attn");
        mlp_block(&mut v, &p);
    }
    for l in 0..cfg.num_hidden_layers {
        let p = format!("decoder.blocks.{l}");
        attn_block(&mut v, &p, "attn");
        attn_block(&mut v, &p, "cross_attn");
        mlp_block(&mut v, &p);
    }
    v
}

/// Whether the SOURCE checkpoint supports the translation task.
///
/// large-v3-turbo was distilled from large-v3 WITHOUT translation training:
/// its vocab still carries `<|translate|>`, but selecting it emits
/// source-language text. Detection heuristic (from config.json shape):
/// turbo is the only published whisper checkpoint that pairs a 32-layer
/// encoder with a 4-layer decoder — tiny/base/small/medium use equal
/// encoder/decoder depths (4/6/12/24) and large v1–v3 pair 32/32. The
/// resulting `whisper.supports_translate` metadata is the authoritative
/// runtime signal (the engine falls back to the same heuristic for bundles
/// converted before this key existed).
pub fn supports_translate(cfg: &ArchConfig) -> bool {
    !(cfg.encoder_layers == 32 && cfg.num_hidden_layers == 4)
}

/// True for `<|xx|>` / `<|xxx|>` ISO-language added tokens (`<|en|>`,
/// `<|yue|>`), false for every other special (`<|translate|>`,
/// timestamps `<|0.00|>`, …).
fn is_lang_token(content: &str) -> bool {
    let Some(inner) = content
        .strip_prefix("<|")
        .and_then(|s| s.strip_suffix("|>"))
    else {
        return false;
    };
    (inner.len() == 2 || inner.len() == 3) && inner.bytes().all(|b| b.is_ascii_lowercase())
}

/// Derive the flat `whisper.*` metadata keys from a repo's
/// `tokenizer.json`. Exact IDs come from `added_tokens` — never from a
/// vocab-size table (large-v3's extra `<|yue|>` lang token shifts every
/// id after 50357 by one; hardcoded tables mis-map it).
///
/// Hard requirements: `<|startoftranscript|>`, `<|endoftext|>` and
/// `<|notimestamps|>` must exist (every whisper vocab has them).
/// `<|translate|>` / `<|transcribe|>` / `<|startofprev|>` /
/// `<|nospeech|>` (or the older `<|nocaptions|>` spelling) are emitted
/// when present — English-only vocabs may omit some.
pub fn token_metadata_from_tokenizer(
    tokenizer_json: &Value,
    vocab_size: i64,
) -> Result<BTreeMap<String, Value>> {
    let added = tokenizer_json
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .context("whisper: tokenizer.json has no added_tokens array")?;

    let mut by_content: BTreeMap<&str, i64> = BTreeMap::new();
    for t in added {
        let (Some(content), Some(id)) = (
            t.get("content").and_then(|v| v.as_str()),
            t.get("id").and_then(|v| v.as_i64()),
        ) else {
            continue;
        };
        by_content.insert(content, id);
    }

    let sot = *by_content.get("<|startoftranscript|>").context(
        "whisper: tokenizer.json added_tokens lacks <|startoftranscript|> — \
         not a whisper tokenizer",
    )?;
    let eot = *by_content
        .get("<|endoftext|>")
        .context("whisper: tokenizer.json added_tokens lacks <|endoftext|>")?;
    let no_timestamps = *by_content
        .get("<|notimestamps|>")
        .context("whisper: tokenizer.json added_tokens lacks <|notimestamps|>")?;

    // Language tokens sit in a contiguous run starting at sot+1 on
    // multilingual vocabs (99 on v1/v2, 100 on v3 with `<|yue|>`).
    // English-only vocabs have none. Count the contiguous run so a
    // stray lang-shaped token elsewhere can't inflate the count.
    let lang_ids: std::collections::BTreeSet<i64> = added
        .iter()
        .filter_map(|t| {
            let content = t.get("content")?.as_str()?;
            let id = t.get("id")?.as_i64()?;
            is_lang_token(content).then_some(id)
        })
        .collect();
    let mut num_languages: i64 = 0;
    while lang_ids.contains(&(sot + 1 + num_languages)) {
        num_languages += 1;
    }
    // The tokenizer alone cannot discriminate English-only checkpoints: the
    // `.en` vocabs physically contain the language tokens in added_tokens
    // even though the model was never trained on them. The vocab size is the
    // authoritative signal (51864 = English-only; 51865/51866 multilingual),
    // matching the engine's GGML-side WhisperTokenMap::from_vocab_size.
    if vocab_size < 51865 {
        num_languages = 0;
    }
    let multilingual = num_languages > 0;

    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: i64| {
        m.insert(format!("whisper.{k}"), Value::from(v));
    };
    put("eot_token_id", eot);
    put("sot_token_id", sot);
    put("no_timestamps_token_id", no_timestamps);
    put("timestamp_begin_token_id", no_timestamps + 1);
    // The runtime's WhisperTokenMap::from_metadata hard-requires every one
    // of these keys (a missing id has no safe default), so a tokenizer that
    // lacks any of them must fail HERE at convert time — not produce a
    // bundle that later refuses to load. Every published whisper tokenizer
    // (including the .en ones) carries all four.
    let translate = *by_content
        .get("<|translate|>")
        .context("whisper: tokenizer.json added_tokens lacks <|translate|>")?;
    let transcribe = *by_content
        .get("<|transcribe|>")
        .context("whisper: tokenizer.json added_tokens lacks <|transcribe|>")?;
    let sot_prev = *by_content
        .get("<|startofprev|>")
        .context("whisper: tokenizer.json added_tokens lacks <|startofprev|>")?;
    // Newer vocabs (large-v3) spell it `<|nospeech|>`; older ones
    // (tiny…large-v2) use `<|nocaptions|>`. Same slot, same semantics.
    let no_speech = *by_content
        .get("<|nospeech|>")
        .or_else(|| by_content.get("<|nocaptions|>"))
        .context("whisper: tokenizer.json added_tokens lacks <|nospeech|>/<|nocaptions|>")?;
    put("translate_token_id", translate);
    put("transcribe_token_id", transcribe);
    put("sot_prev_token_id", sot_prev);
    put("no_speech_token_id", no_speech);
    put("num_languages", num_languages);
    put("lang_token_offset", if multilingual { sot + 1 } else { 0 });
    put("is_multilingual", multilingual as i64);
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HfMapper;
    use serde_json::json;

    // ── Name mapping ─────────────────────────────────────────────────

    /// Every HF tensor pattern in the contract's table maps to its
    /// engine-native name; layer indices survive; tied lm_head drops.
    #[test]
    fn hf_name_mapping_table() {
        let cases: &[(&str, &str)] = &[
            ("model.encoder.conv1.weight", "encoder.conv1.weight"),
            ("model.encoder.conv1.bias", "encoder.conv1.bias"),
            ("model.encoder.conv2.weight", "encoder.conv2.weight"),
            ("model.encoder.conv2.bias", "encoder.conv2.bias"),
            (
                "model.encoder.embed_positions.weight",
                "encoder.positional_embedding",
            ),
            ("model.encoder.layer_norm.weight", "encoder.ln_post.weight"),
            ("model.encoder.layer_norm.bias", "encoder.ln_post.bias"),
            (
                "model.encoder.layers.0.self_attn.q_proj.weight",
                "encoder.blocks.0.attn.query.weight",
            ),
            (
                "model.encoder.layers.0.self_attn.q_proj.bias",
                "encoder.blocks.0.attn.query.bias",
            ),
            (
                "model.encoder.layers.3.self_attn.k_proj.weight",
                "encoder.blocks.3.attn.key.weight",
            ),
            (
                "model.encoder.layers.1.self_attn.v_proj.weight",
                "encoder.blocks.1.attn.value.weight",
            ),
            (
                "model.encoder.layers.1.self_attn.v_proj.bias",
                "encoder.blocks.1.attn.value.bias",
            ),
            (
                "model.encoder.layers.2.self_attn.out_proj.weight",
                "encoder.blocks.2.attn.out.weight",
            ),
            (
                "model.encoder.layers.2.self_attn.out_proj.bias",
                "encoder.blocks.2.attn.out.bias",
            ),
            (
                "model.encoder.layers.0.self_attn_layer_norm.weight",
                "encoder.blocks.0.attn_ln.weight",
            ),
            (
                "model.encoder.layers.0.self_attn_layer_norm.bias",
                "encoder.blocks.0.attn_ln.bias",
            ),
            (
                "model.encoder.layers.0.fc1.weight",
                "encoder.blocks.0.mlp.0.weight",
            ),
            (
                "model.encoder.layers.0.fc1.bias",
                "encoder.blocks.0.mlp.0.bias",
            ),
            (
                "model.encoder.layers.0.fc2.weight",
                "encoder.blocks.0.mlp.2.weight",
            ),
            (
                "model.encoder.layers.0.fc2.bias",
                "encoder.blocks.0.mlp.2.bias",
            ),
            (
                "model.encoder.layers.0.final_layer_norm.weight",
                "encoder.blocks.0.mlp_ln.weight",
            ),
            (
                "model.encoder.layers.0.final_layer_norm.bias",
                "encoder.blocks.0.mlp_ln.bias",
            ),
            (
                "model.decoder.embed_tokens.weight",
                "decoder.token_embedding.weight",
            ),
            (
                "model.decoder.embed_positions.weight",
                "decoder.positional_embedding",
            ),
            ("model.decoder.layer_norm.weight", "decoder.ln.weight"),
            ("model.decoder.layer_norm.bias", "decoder.ln.bias"),
            (
                "model.decoder.layers.0.self_attn.q_proj.weight",
                "decoder.blocks.0.attn.query.weight",
            ),
            (
                "model.decoder.layers.0.self_attn.k_proj.weight",
                "decoder.blocks.0.attn.key.weight",
            ),
            (
                "model.decoder.layers.0.self_attn.v_proj.bias",
                "decoder.blocks.0.attn.value.bias",
            ),
            (
                "model.decoder.layers.0.self_attn.out_proj.weight",
                "decoder.blocks.0.attn.out.weight",
            ),
            (
                "model.decoder.layers.0.self_attn_layer_norm.weight",
                "decoder.blocks.0.attn_ln.weight",
            ),
            (
                "model.decoder.layers.31.encoder_attn.q_proj.weight",
                "decoder.blocks.31.cross_attn.query.weight",
            ),
            (
                "model.decoder.layers.31.encoder_attn.q_proj.bias",
                "decoder.blocks.31.cross_attn.query.bias",
            ),
            (
                "model.decoder.layers.5.encoder_attn.k_proj.weight",
                "decoder.blocks.5.cross_attn.key.weight",
            ),
            (
                "model.decoder.layers.5.encoder_attn.v_proj.weight",
                "decoder.blocks.5.cross_attn.value.weight",
            ),
            (
                "model.decoder.layers.5.encoder_attn.v_proj.bias",
                "decoder.blocks.5.cross_attn.value.bias",
            ),
            (
                "model.decoder.layers.5.encoder_attn.out_proj.weight",
                "decoder.blocks.5.cross_attn.out.weight",
            ),
            (
                "model.decoder.layers.5.encoder_attn.out_proj.bias",
                "decoder.blocks.5.cross_attn.out.bias",
            ),
            (
                "model.decoder.layers.5.encoder_attn_layer_norm.weight",
                "decoder.blocks.5.cross_attn_ln.weight",
            ),
            (
                "model.decoder.layers.5.encoder_attn_layer_norm.bias",
                "decoder.blocks.5.cross_attn_ln.bias",
            ),
            (
                "model.decoder.layers.2.fc1.weight",
                "decoder.blocks.2.mlp.0.weight",
            ),
            (
                "model.decoder.layers.2.fc2.weight",
                "decoder.blocks.2.mlp.2.weight",
            ),
            (
                "model.decoder.layers.2.final_layer_norm.weight",
                "decoder.blocks.2.mlp_ln.weight",
            ),
        ];
        for (src, want) in cases {
            assert_eq!(
                map_hf_tensor_name(src).as_deref(),
                Some(*want),
                "mapping for {src}"
            );
            // The un-prefixed form (no `model.`) maps identically.
            let bare = src.strip_prefix("model.").unwrap();
            assert_eq!(
                map_hf_tensor_name(bare).as_deref(),
                Some(*want),
                "bare mapping for {bare}"
            );
        }
    }

    /// Tied lm_head and k-bias tensors drop; unknown names drop.
    #[test]
    fn dropped_tensors() {
        for n in [
            "proj_out.weight",
            "lm_head.weight",
            "model.proj_out.weight",
            "model.encoder.layers.0.self_attn.k_proj.bias",
            "model.decoder.layers.0.encoder_attn.k_proj.bias",
            // cross-attention on the encoder side is not a thing
            "model.encoder.layers.0.encoder_attn.q_proj.weight",
            "model.unknown.tensor",
            "model.decoder.layers.x.fc1.weight",
        ] {
            assert_eq!(map_hf_tensor_name(n), None, "should drop {n}");
        }
    }

    // ── Config extraction ────────────────────────────────────────────

    fn tiny_config() -> Value {
        json!({
            "model_type": "whisper",
            "d_model": 384,
            "decoder_layers": 4,
            "decoder_attention_heads": 6,
            "decoder_ffn_dim": 1536,
            "encoder_layers": 4,
            "encoder_attention_heads": 6,
            "encoder_ffn_dim": 1536,
            "num_mel_bins": 80,
            "max_source_positions": 1500,
            "max_target_positions": 448,
            "vocab_size": 51865,
            "bos_token_id": 50257,
            "eos_token_id": 50257
        })
    }

    #[test]
    fn config_extraction_tiny() {
        let cfg = WhisperHfMapper.config_from_hf(&tiny_config()).unwrap();
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.num_attention_heads, 6);
        assert_eq!(cfg.num_kv_heads, 6);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.intermediate_size, 1536);
        assert_eq!(cfg.encoder_layers, 4);
        assert_eq!(cfg.encoder_attention_heads, 6);
        assert_eq!(cfg.encoder_ffn_dim, 1536);
        assert_eq!(cfg.decoder_ffn_dim, 1536);
        assert_eq!(cfg.num_mel_bins, 80);
        assert_eq!(cfg.max_source_positions, 1500);
        assert_eq!(cfg.max_target_positions, 448);
        assert_eq!(cfg.max_position_embeddings, 448);
        assert_eq!(cfg.vocab_size, 51865);
        assert!(cfg.tie_word_embeddings);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-12);

        // The header config map carries the whisper contract keys.
        let m = cfg.to_config_map();
        assert_eq!(m["model_type"], json!("whisper"));
        assert_eq!(m["d_model"], json!(384));
        assert_eq!(m["encoder_layers"], json!(4));
        assert_eq!(m["encoder_attention_heads"], json!(6));
        assert_eq!(m["encoder_ffn_dim"], json!(1536));
        assert_eq!(m["decoder_ffn_dim"], json!(1536));
        assert_eq!(m["num_mel_bins"], json!(80));
        assert_eq!(m["max_source_positions"], json!(1500));
        assert_eq!(m["max_target_positions"], json!(448));
        assert_eq!(m["norm_eps"], json!(1e-5f32));
    }

    #[test]
    fn config_missing_key_fails_loud() {
        let mut c = tiny_config();
        c.as_object_mut().unwrap().remove("num_mel_bins");
        let err = WhisperHfMapper.config_from_hf(&c).unwrap_err();
        assert!(err.to_string().contains("num_mel_bins"), "err: {err}");
    }

    // ── Token-id derivation ──────────────────────────────────────────

    fn added(tokens: &[(i64, &str)]) -> Value {
        json!({
            "added_tokens": tokens
                .iter()
                .map(|(id, c)| json!({"id": id, "content": c}))
                .collect::<Vec<_>>()
        })
    }

    /// Real multilingual v1/v2-style tokenizer.json (whisper-tiny).
    ///
    /// The fixture is the genuine `added_tokens` array from
    /// openai/whisper-tiny, trimmed to ids <= 50364 — that drops the 1500
    /// timestamp tokens the parser never inspects and takes the file from
    /// 2.4 MB to 5 KB. It is compiled in rather than read from a path so
    /// this runs everywhere: it previously pointed at one developer's home
    /// directory and returned early on every other machine, which made the
    /// only real-tokenizer coverage green-by-skip in CI.
    #[test]
    fn token_ids_from_real_whisper_tiny_tokenizer() {
        let bytes = include_bytes!("../tests/fixtures/whisper-tiny-added-tokens.json");
        let tj: Value = serde_json::from_slice(bytes).unwrap();
        let m = token_metadata_from_tokenizer(&tj, 51865).unwrap();
        assert_eq!(m["whisper.eot_token_id"], json!(50257));
        assert_eq!(m["whisper.sot_token_id"], json!(50258));
        assert_eq!(m["whisper.translate_token_id"], json!(50358));
        assert_eq!(m["whisper.transcribe_token_id"], json!(50359));
        assert_eq!(m["whisper.sot_prev_token_id"], json!(50361));
        // tiny uses the older `<|nocaptions|>` spelling for the same slot.
        assert_eq!(m["whisper.no_speech_token_id"], json!(50362));
        assert_eq!(m["whisper.no_timestamps_token_id"], json!(50363));
        assert_eq!(m["whisper.timestamp_begin_token_id"], json!(50364));
        assert_eq!(m["whisper.num_languages"], json!(99));
        assert_eq!(m["whisper.lang_token_offset"], json!(50259));
        assert_eq!(m["whisper.is_multilingual"], json!(1));
    }

    /// Synthetic large-v3-style vocab: 100 languages (`<|yue|>` appended
    /// at 50358) shift translate/transcribe/… up by one vs v1/v2. The
    /// hardcoded-off-vocab-size bug this converter replaces mis-mapped
    /// exactly these ids.
    #[test]
    fn token_ids_large_v3_style() {
        let mut toks: Vec<(i64, String)> = vec![
            (50257, "<|endoftext|>".into()),
            (50258, "<|startoftranscript|>".into()),
        ];
        // 99 two-letter-style langs at 50259..=50357 + `<|yue|>` at 50358.
        for i in 0..99i64 {
            let a = (b'a' + (i / 26) as u8) as char;
            let b = (b'a' + (i % 26) as u8) as char;
            toks.push((50259 + i, format!("<|{a}{b}|>")));
        }
        toks.push((50358, "<|yue|>".into()));
        toks.extend([
            (50359, "<|translate|>".into()),
            (50360, "<|transcribe|>".into()),
            (50361, "<|startoflm|>".into()),
            (50362, "<|startofprev|>".into()),
            (50363, "<|nospeech|>".into()),
            (50364, "<|notimestamps|>".into()),
            (50365, "<|0.00|>".into()),
        ]);
        let fixture = added(
            &toks
                .iter()
                .map(|(id, c)| (*id, c.as_str()))
                .collect::<Vec<_>>(),
        );
        let m = token_metadata_from_tokenizer(&fixture, 51866).unwrap();
        assert_eq!(m["whisper.eot_token_id"], json!(50257));
        assert_eq!(m["whisper.sot_token_id"], json!(50258));
        assert_eq!(m["whisper.num_languages"], json!(100));
        assert_eq!(m["whisper.lang_token_offset"], json!(50259));
        assert_eq!(m["whisper.translate_token_id"], json!(50359));
        assert_eq!(m["whisper.transcribe_token_id"], json!(50360));
        assert_eq!(m["whisper.sot_prev_token_id"], json!(50362));
        assert_eq!(m["whisper.no_speech_token_id"], json!(50363));
        assert_eq!(m["whisper.no_timestamps_token_id"], json!(50364));
        assert_eq!(m["whisper.timestamp_begin_token_id"], json!(50365));
        assert_eq!(m["whisper.is_multilingual"], json!(1));
    }

    /// English-only vocab: no lang tokens at all. transcribe/translate
    /// still emitted when present; multilingual = 0, offset = 0.
    #[test]
    fn token_ids_en_only() {
        let fixture = added(&[
            (50256, "<|endoftext|>"),
            (50257, "<|startoftranscript|>"),
            (50358, "<|translate|>"),
            (50359, "<|transcribe|>"),
            (50360, "<|startoflm|>"),
            (50361, "<|startofprev|>"),
            (50362, "<|nocaptions|>"),
            (50363, "<|notimestamps|>"),
            (50364, "<|0.00|>"),
        ]);
        let m = token_metadata_from_tokenizer(&fixture, 51864).unwrap();
        assert_eq!(m["whisper.eot_token_id"], json!(50256));
        assert_eq!(m["whisper.sot_token_id"], json!(50257));
        assert_eq!(m["whisper.num_languages"], json!(0));
        assert_eq!(m["whisper.lang_token_offset"], json!(0));
        assert_eq!(m["whisper.is_multilingual"], json!(0));
        assert_eq!(m["whisper.translate_token_id"], json!(50358));
        assert_eq!(m["whisper.transcribe_token_id"], json!(50359));
        assert_eq!(m["whisper.no_speech_token_id"], json!(50362));
        assert_eq!(m["whisper.no_timestamps_token_id"], json!(50363));
        assert_eq!(m["whisper.timestamp_begin_token_id"], json!(50364));
    }

    /// `.en` checkpoints: the tokenizer PHYSICALLY contains the language
    /// tokens (they sit in the vocab), but vocab_size 51864 is authoritative
    /// — the model was never trained on them. The override must force
    /// English-only metadata or `task=translate` and language selection
    /// would be accepted on models that can't honor them.
    #[test]
    fn token_ids_en_only_with_lang_tokens_in_vocab() {
        let mut toks: Vec<(i64, String)> = vec![
            (50256, "<|endoftext|>".into()),
            (50257, "<|startoftranscript|>".into()),
            (50358, "<|translate|>".into()),
            (50359, "<|transcribe|>".into()),
            (50361, "<|startofprev|>".into()),
            (50362, "<|nocaptions|>".into()),
            (50363, "<|notimestamps|>".into()),
        ];
        // 99 real-looking language tokens in the contiguous run at sot+1,
        // exactly as the actual `.en` tokenizer.json files ship them.
        for (i, code) in ["en", "zh", "de", "es", "ru"].iter().enumerate() {
            toks.push((50258 + i as i64, format!("<|{code}|>")));
        }
        let borrowed: Vec<(i64, &str)> = toks.iter().map(|(i, s)| (*i, s.as_str())).collect();
        let fixture = added(&borrowed);
        let m = token_metadata_from_tokenizer(&fixture, 51864).unwrap();
        assert_eq!(m["whisper.num_languages"], json!(0));
        assert_eq!(m["whisper.lang_token_offset"], json!(0));
        assert_eq!(m["whisper.is_multilingual"], json!(0));
    }

    /// A tokenizer without `<|startoftranscript|>` is not a whisper
    /// tokenizer — hard error, never silent defaults.
    #[test]
    fn missing_sot_fails_loud() {
        let fixture = added(&[(50256, "<|endoftext|>"), (50363, "<|notimestamps|>")]);
        let err = token_metadata_from_tokenizer(&fixture, 51866).unwrap_err();
        assert!(
            err.to_string().contains("<|startoftranscript|>"),
            "err: {err}"
        );
    }

    /// Timestamp tokens and multi-char specials must not count as
    /// languages; `<|yue|>` must.
    #[test]
    fn lang_token_pattern() {
        assert!(is_lang_token("<|en|>"));
        assert!(is_lang_token("<|zh|>"));
        assert!(is_lang_token("<|yue|>"));
        assert!(!is_lang_token("<|0.00|>"));
        assert!(!is_lang_token("<|translate|>"));
        assert!(!is_lang_token("<|EN|>"));
        assert!(!is_lang_token("<|e|>"));
        assert!(!is_lang_token("<|abcd|>"));
        assert!(!is_lang_token("en"));
    }

    // ── Turbo translate-capability heuristic ─────────────────────────

    /// 32-encoder/4-decoder (large-v3-turbo's distillation shape) drops
    /// translate; every other published depth pairing keeps it.
    #[test]
    fn turbo_supports_translate_heuristic() {
        let mk = |enc: u32, dec: u32| ArchConfig {
            encoder_layers: enc,
            num_hidden_layers: dec,
            ..ArchConfig::default()
        };
        assert!(!supports_translate(&mk(32, 4)), "large-v3-turbo must drop translate");
        for (enc, dec) in [(4u32, 4u32), (6, 6), (12, 12), (24, 24), (32, 32)] {
            assert!(supports_translate(&mk(enc, dec)), "{enc}/{dec} keeps translate");
        }
    }

    // ── Required-tensor inventory ────────────────────────────────────

    /// tiny (4 enc + 4 dec layers) must require exactly the 167 tensors
    /// the HF checkpoint ships (all of which map; nothing else).
    #[test]
    fn required_tensor_count_tiny() {
        let cfg = WhisperHfMapper.config_from_hf(&tiny_config()).unwrap();
        let req = required_tensor_names(&cfg);
        // 7 encoder globals + 4·15 encoder-layer + 4 decoder globals +
        // 4·(15 self + 9 cross) decoder-layer = 167.
        assert_eq!(req.len(), 167);
        // No duplicates.
        let set: std::collections::BTreeSet<_> = req.iter().collect();
        assert_eq!(set.len(), req.len());
    }

    // ── Quant eligibility ────────────────────────────────────────────

    /// Exactly the block linear projections are quant-eligible; conv,
    /// embeddings, norms and every bias must stay f16.
    #[test]
    fn quant_eligibility_split() {
        for name in [
            "encoder.blocks.0.attn.query.weight",
            "encoder.blocks.3.attn.key.weight",
            "encoder.blocks.1.attn.value.weight",
            "encoder.blocks.2.attn.out.weight",
            "encoder.blocks.0.mlp.0.weight",
            "encoder.blocks.0.mlp.2.weight",
            "decoder.blocks.31.cross_attn.query.weight",
            "decoder.blocks.5.cross_attn.key.weight",
            "decoder.blocks.5.cross_attn.value.weight",
            "decoder.blocks.5.cross_attn.out.weight",
            "decoder.blocks.2.mlp.0.weight",
            "decoder.blocks.2.mlp.2.weight",
            "decoder.blocks.0.attn.query.weight",
        ] {
            assert!(is_quantizable_linear(name), "{name} should be eligible");
        }
        for name in [
            "encoder.conv1.weight",
            "encoder.conv2.weight",
            "encoder.conv1.bias",
            "encoder.positional_embedding",
            "decoder.positional_embedding",
            "decoder.token_embedding.weight",
            "encoder.ln_post.weight",
            "decoder.ln.weight",
            "encoder.blocks.0.attn_ln.weight",
            "decoder.blocks.5.cross_attn_ln.weight",
            "encoder.blocks.0.mlp_ln.weight",
            "encoder.blocks.0.attn.query.bias",
            "encoder.blocks.0.attn.value.bias",
            "encoder.blocks.0.attn.out.bias",
            "encoder.blocks.0.mlp.0.bias",
            "encoder.blocks.0.mlp.2.bias",
        ] {
            assert!(!is_quantizable_linear(name), "{name} must stay f16");
        }
    }
}
