//! Muse Glimmer HF mapper.
//!
//! `MuseGlimmerForConditionalGeneration` (model_type `muse_glimmer`) is a
//! dense decoder with a perception (ViT) tower. The language tower nests
//! under `text_config` (model_type `muse_glimmer_text`) and is shaped like
//! Gemma 3 — four zero-centered RMSNorms per layer, SwiGLU FFN, a final
//! logit softcap — with four Muse-Glimmer-specific twists:
//!
//!   1. Attention layers alternate local(sliding, 2048) x3 → global, and
//!      the GLOBAL layers are NoPE: `layer_rope_theta[i] == 0` disables
//!      rotary entirely there (the reference passes
//!      `position_embeddings=None` for those layers). Local layers use the
//!      ordinary theta from `rope_parameters`.
//!   2. Q and K pass through a SCALELESS (weightless) RMSNorm, after which
//!      Q is multiplied by `qk_scale_factor` — on top of, not instead of,
//!      the usual `1/sqrt(head_dim)`. There are therefore no `q_norm` /
//!      `k_norm` tensors in the checkpoint.
//!   3. Attention output is gated: `attn_out *= sigmoid(gate_proj(x))`
//!      BEFORE `o_proj`, where `gate_proj` is a SEPARATE `[n_heads*head_dim,
//!      hidden]` projection over the layer input (not a doubled `q_proj`
//!      like Qwen3.5).
//!   4. The post-attention / post-FFN norms use `post_norm_eps` (1e-8),
//!      distinct from `rms_norm_eps` (1e-5) on the other two norms.
//!
//! The token embedding carries a weightless RMSNorm applied right after
//! lookup; `row_rms_normalize` bakes it into the matrix at convert time
//! (see the trait docs for why that is exact).

use crate::{ArchConfig, GgufMapper, HfMapper};
use anyhow::{Context, Result};
use base_readers::gguf::KvValue;
use std::collections::BTreeMap;

pub struct MuseGlimmerHfMapper;
pub struct MuseGlimmerGgufMapper;

/// Canonical names of the four per-layer norms that use Muse Glimmer's
/// zero-centered `(1 + weight)` RMSNorm formulation, AFTER the HF→canonical
/// rename in base-convert's `hf_rename`. The final `norm.weight` →
/// `final_norm.weight` is a PLAIN weighted RMSNorm in the reference
/// (`MuseGlimmerRMSNorm`, not `MuseGlimmerTextCenteredRMSNorm`) and must
/// NOT be shifted — nor may any `vision.*` LayerNorm.
const CENTERED_NORM_SUFFIXES: [&str; 4] = [
    ".input_norm.weight",
    ".post_attention_norm.weight",
    ".post_attn_norm.weight",
    ".post_ffw_norm.weight",
];

impl HfMapper for MuseGlimmerHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "muse_glimmer"
    }

    /// Bake the +1 unit offset into the zero-centered per-layer RMSNorm
    /// gammas so the runtime's plain `rmsnorm(x) * weight` kernel yields
    /// the reference's `rmsnorm(x) * (1 + weight)`. Same treatment Gemma 3
    /// gets, but scoped to the four per-layer norms: unlike Gemma 3, Muse
    /// Glimmer's FINAL norm is not zero-centered, so a blanket
    /// `ends_with("norm.weight")` rule would corrupt it.
    fn norm_shift(&self, canonical: &str) -> f32 {
        if !canonical.starts_with("layers.") {
            return 0.0;
        }
        if CENTERED_NORM_SUFFIXES
            .iter()
            .any(|s| canonical.ends_with(s))
        {
            1.0
        } else {
            0.0
        }
    }

    /// Fold the weightless embedding norm into the embedding matrix.
    /// Uses `rms_norm_eps`, matching `MuseGlimmerTextNormedEmbedding`
    /// (constructed with `config.rms_norm_eps`).
    fn row_rms_normalize(&self, canonical: &str, cfg: &ArchConfig) -> Option<f32> {
        if canonical == "embed_tokens.weight" {
            // Baking is only valid while the embedding is untied from the
            // output head — otherwise the LM head would inherit the norm.
            if cfg.tie_word_embeddings {
                return None;
            }
            return Some(cfg.rms_norm_eps);
        }
        None
    }

    fn config_from_hf(&self, c: &serde_json::Value) -> Result<ArchConfig> {
        // The multimodal wrapper nests the language tower under
        // `text_config`; a text-only checkpoint hoists it to the top level.
        let tc = c.get("text_config").unwrap_or(c);
        let mut config = crate::llama::hf_generic_config(tc)?;

        let f32_v =
            |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_f64()).map(|f| f as f32);
        let u32_v =
            |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_u64()).map(|n| n as u32);

        // Muse Glimmer keeps RoPE under `rope_parameters`, so
        // hf_generic_config's top-level `rope_theta` default (10000) is
        // wrong — override it.
        if let Some(rp) = tc.get("rope_parameters") {
            if let Some(theta) = f32_v(rp, "rope_theta") {
                config.rope_theta = theta;
            }
        }

        config.tie_word_embeddings = tc
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Sliding-window schedule. `layer_types` is authoritative; the
        // reference default is "full_attention on every 4th layer counted
        // BACKWARD from the last, sliding otherwise", which we reproduce
        // when the key is absent so a trimmed config still converts.
        let n_layers = config.num_hidden_layers as usize;
        config.sliding_window = u32_v(tc, "sliding_window").unwrap_or(0);
        let layer_types: Vec<String> = match tc.get("layer_types").and_then(|v| v.as_array()) {
            Some(lt) => lt
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            None => default_layer_types(n_layers),
        };
        config.swa_layers = layer_types
            .iter()
            .map(|t| t == "sliding_attention")
            .collect();

        // NoPE mask: `layer_rope_theta[i] == 0` means the layer runs with
        // no rotary at all. The reference default puts NoPE on exactly the
        // full-attention layers, so fall back to that when the key is
        // absent rather than silently giving every layer RoPE.
        config.nope_layers = match tc.get("layer_rope_theta").and_then(|v| v.as_array()) {
            Some(arr) => arr
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) == 0.0)
                .collect(),
            None => layer_types.iter().map(|t| t == "full_attention").collect(),
        };

        config.logit_softcap = f32_v(tc, "final_logit_softcapping").unwrap_or(0.0);
        config.qk_scale_factor = f32_v(tc, "qk_scale_factor").unwrap_or(0.0);
        config.output_multiplier = f32_v(tc, "output_multiplier").unwrap_or(0.0);
        config.post_norm_eps = f32_v(tc, "post_norm_eps").unwrap_or(0.0);
        // Gated attention via a dedicated `self_attn.gate_proj`.
        config.attn_output_gate = true;

        Ok(config)
    }
}

/// The reference's default attention schedule when a config/GGUF omits an
/// explicit one: `full_attention` on every 4th layer counted BACKWARD from
/// the last, `sliding_attention` everywhere else. Shared by the HF and
/// GGUF paths so a trimmed config and a metadata-poor GGUF land on the
/// same schedule.
fn default_layer_types(n_layers: usize) -> Vec<String> {
    (0..n_layers)
        .map(|i| {
            if (n_layers - 1 - i) % 4 == 0 {
                "full_attention".to_string()
            } else {
                "sliding_attention".to_string()
            }
        })
        .collect()
}

// ── GGUF path ───────────────────────────────────────────────────────
//
// `general.architecture` is the HYPHENATED "muse-glimmer", so every
// per-arch metadata key is prefixed `muse-glimmer.` — see
// `source_mapper_for_gguf`. `canonical_arch()` returns the UNDERSCORED
// "muse_glimmer" so a GGUF-sourced and an HF-sourced bundle carry the
// identical `arch` field and select the same runtime model class.

/// Published Muse Glimmer 30B architecture constants. llama.cpp bakes
/// most of these into its graph builder rather than exporting them, so a
/// GGUF that predates an exporter change lands here. Same treatment
/// Gemma 3's sliding-window pattern / local theta get in `gemma.rs`.
const REF_QK_SCALE_FACTOR: f32 = 3.87;
const REF_LOGIT_SOFTCAP: f32 = 20.0;
const REF_POST_NORM_EPS: f32 = 1e-8;
const REF_SLIDING_WINDOW: u32 = 2048;

impl GgufMapper for MuseGlimmerGgufMapper {
    fn canonical_arch(&self) -> &'static str {
        "muse_glimmer"
    }

    fn config_from_gguf(&self, m: &BTreeMap<String, KvValue>) -> Result<ArchConfig> {
        // Prefer the hyphenated prefix llama.cpp actually writes; accept
        // the underscored one so a hand-rolled exporter still converts.
        let prefix = if m.keys().any(|k| k.starts_with("muse-glimmer.")) {
            "muse-glimmer"
        } else {
            "muse_glimmer"
        };
        let u32_req = |k: &str| {
            m.get(k)
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .with_context(|| format!("missing metadata key: {k}"))
        };
        let u32_key = |k: &str| m.get(k).and_then(|v| v.as_u64()).map(|n| n as u32);
        let f32_key = |k: &str| m.get(k).and_then(|v| v.as_f32());
        // First key that exists wins — lets us accept more than one
        // plausible spelling without a combinatorial explosion of ifs.
        let f32_any = |suffixes: &[&str]| -> Option<f32> {
            suffixes
                .iter()
                .find_map(|s| f32_key(&format!("{prefix}.{s}")))
        };
        let u32_any = |suffixes: &[&str]| -> Option<u32> {
            suffixes
                .iter()
                .find_map(|s| u32_key(&format!("{prefix}.{s}")))
        };

        let hidden_size = u32_req(&format!("{prefix}.embedding_length"))?;
        let num_hidden_layers = u32_req(&format!("{prefix}.block_count"))?;
        let num_attention_heads = u32_req(&format!("{prefix}.attention.head_count"))?;
        let num_kv_heads = u32_key(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_attention_heads);
        let intermediate_size = u32_req(&format!("{prefix}.feed_forward_length"))?;
        let vocab_size = u32_key(&format!("{prefix}.vocab_size"))
            .or_else(|| match m.get("tokenizer.ggml.tokens") {
                Some(KvValue::Array(a)) => Some(a.len() as u32),
                _ => None,
            })
            .context("no vocab_size and no tokenizer.ggml.tokens")?;
        let head_dim = u32_key(&format!("{prefix}.attention.key_length"))
            .unwrap_or(hidden_size / num_attention_heads);

        let mut config = ArchConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rope_theta: f32_key(&format!("{prefix}.rope.freq_base")).unwrap_or(500_000.0),
            rope_scale: f32_key(&format!("{prefix}.rope.scaling.factor")).unwrap_or(1.0),
            rms_norm_eps: f32_key(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            // The released checkpoint keeps `lm_head` separate; the GGUF
            // ships `output.weight` accordingly.
            tie_word_embeddings: m
                .get(&format!("{prefix}.tie_lm_head"))
                .and_then(|v| match v {
                    KvValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(false),
            max_position_embeddings: u32_key(&format!("{prefix}.context_length")).unwrap_or(0),
            bos_token_id: m
                .get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .unwrap_or(0),
            eos_token_id: m
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
                .unwrap_or(0),
            // Gated attention via a dedicated `blk.N.attn_gate.weight`.
            attn_output_gate: true,
            ..ArchConfig::default()
        };

        // ── Muse-Glimmer-specific scales ────────────────────────────
        // A zero here is NOT harmless: `qk_scale_factor` multiplies the
        // attention scale (muse_glimmer.cpp only applies it when > 0), so
        // dropping it silently shrinks every logit by ~4x. Prefer an
        // exported key; otherwise fall back to the published constant.
        config.qk_scale_factor = f32_any(&["attention.qk_scale_factor", "qk_scale_factor"])
            .unwrap_or(REF_QK_SCALE_FACTOR);
        config.logit_softcap = f32_any(&["final_logit_softcapping", "logit_softcap"])
            .unwrap_or(REF_LOGIT_SOFTCAP);
        config.post_norm_eps =
            f32_any(&["attention.post_norm_epsilon", "post_norm_eps"]).unwrap_or(REF_POST_NORM_EPS);
        // llama.cpp exports HF's `output_multiplier` under its own generic
        // name `logit_scale` (verified against the KV dump of
        // meta-models/Muse-Glimmer-30B-GGUF: `muse-glimmer.logit_scale =
        // 0.1961161345243454`); there is no `output_multiplier` key. The
        // HF spelling stays in the probe list for a hand-authored GGUF.
        //
        // The derivation below is the fallback: the value is exactly
        // 1/sqrt(hidden_size/256) on the released checkpoint (6656/256 = 26
        // → 0.19611613…), so it beats hardcoding a width-specific number.
        config.output_multiplier = f32_any(&["logit_scale", "output_multiplier"]).unwrap_or_else(|| {
            let d = hidden_size as f32 / 256.0;
            if d > 0.0 {
                1.0 / d.sqrt()
            } else {
                0.0
            }
        });

        // ── Local/global + NoPE schedules ───────────────────────────
        config.sliding_window = u32_any(&["attention.sliding_window"]).unwrap_or(REF_SLIDING_WINDOW);
        let n_layers = num_hidden_layers as usize;
        // A bool array under `attention.sliding_window_pattern` is the
        // shape Gemma 4 uses and the natural one for this arch; fall back
        // to the reference's local x3 → global cycle.
        let swa_from_kv = match m.get(&format!("{prefix}.attention.sliding_window_pattern")) {
            Some(KvValue::Array(arr)) if !arr.is_empty() => Some(
                arr.iter()
                    .map(|v| match v {
                        KvValue::Bool(b) => *b,
                        other => other.as_u64().unwrap_or(0) != 0,
                    })
                    .collect::<Vec<bool>>(),
            ),
            _ => None,
        };
        let layer_types = default_layer_types(n_layers);
        config.swa_layers = swa_from_kv
            .unwrap_or_else(|| layer_types.iter().map(|t| t == "sliding_attention").collect());

        // NoPE mask. HF encodes it as `layer_rope_theta[i] == 0`; if a
        // per-layer theta array ever lands in GGUF metadata, honour it.
        // Otherwise the reference default is "NoPE on exactly the
        // full-attention layers", which is the complement of `swa_layers`.
        let nope_from_kv = match m.get(&format!("{prefix}.rope.layer_freq_base")) {
            Some(KvValue::Array(arr)) if !arr.is_empty() => Some(
                arr.iter()
                    .map(|v| v.as_f32().unwrap_or(0.0) == 0.0)
                    .collect::<Vec<bool>>(),
            ),
            _ => None,
        };
        config.nope_layers =
            nope_from_kv.unwrap_or_else(|| config.swa_layers.iter().map(|s| !s).collect());

        // llama.cpp does NOT fold Muse Glimmer's weightless embedding norm
        // into `token_embd`: the GGUF rows equal the HF RAW embedding
        // (verified — per-row RMS 0.0625 on both, cosine 0.997 after Q4_K
        // dequant). Passthrough cannot fold it either, because the rows
        // arrive as packed super-blocks and folding means dequantizing, the
        // exact thing passthrough exists to avoid. So the runtime applies it.
        //
        // The HF path leaves this 0 — `row_rms_normalize` already baked it in
        // there, which is exact and costs nothing at inference. Without this
        // signal a GGUF-sourced bundle enters the stack ~16x too small and
        // degenerates into single-token repetition.
        config.embed_norm_eps = config.rms_norm_eps;

        // `tokenizer.ggml.eos_token_id` alone is not enough to stop this
        // model. Its chat format ends an assistant turn with `<|eot|>`
        // (200008), exported separately as `tokenizer.ggml.eot_token_id`,
        // while `eos_token_id` (200001) is `<|end_of_text|>` and only ends
        // the whole document. The HF path picks both up by merging
        // generation_config.json; without the eot id here the model answers
        // correctly and then keeps opening fresh turns until the token cap.
        if let Some(eot) = m
            .get("tokenizer.ggml.eot_token_id")
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
        {
            if eot != 0 && eot != config.eos_token_id && !config.eos_token_ids.contains(&eot) {
                config.eos_token_ids.push(eot);
            }
        }

        Ok(config)
    }

    fn map_tensor_name(&self, n: &str) -> Option<String> {
        map_gguf_name(n)
    }

    /// Muse Glimmer runs the NeoX (split-half) rope kernel — the runtime's
    /// `arch_muse_glimmer()` sets `rope_standalone = "rope_neox_f16"` and
    /// `muse_glimmer.cpp` passes `is_neox=true` — and the HF path leaves
    /// `q_proj`/`k_proj` rows untouched (no `rope_permute_heads` impl on
    /// `MuseGlimmerHfMapper`). llama.cpp's exporter, however, applies the
    /// Llama interleaved-pair permute to `attn_q` / `attn_k`. The GGUF
    /// path must therefore undo it, or Q and K get their trained rotary
    /// frequencies assigned to the wrong dim-pairs.
    ///
    /// Everything else — `attn_v`, `attn_output`, `attn_gate`, the FFN —
    /// is rotation-free and must NOT be touched.
    fn rope_unpermute_heads(&self, canonical: &str, cfg: &ArchConfig) -> Option<u32> {
        if canonical.ends_with(".self_attn.q_proj.weight") {
            Some(cfg.num_attention_heads)
        } else if canonical.ends_with(".self_attn.k_proj.weight") {
            Some(cfg.num_kv_heads)
        } else {
            None
        }
    }
}

/// Map a Muse Glimmer GGUF tensor name to the canonical `.base` name —
/// the SAME name `to_canonical_name` produces for the HF checkpoint, so
/// the runtime's `BaseWeightStore` alias table resolves both identically.
///
/// Deliberately NOT `map_llama_style`: three of its rules are wrong here.
///
///   - `attn_gate.weight` → `self_attn.gate.weight` in the Llama table,
///     but the alias table only knows `self_attn.gate_proj.weight` →
///     `attention.gate.weight` (the Muse Glimmer separate-projection
///     output gate). The Llama name would load as a null buffer.
///   - `post_attention_norm.weight` → `post_attn_norm.weight` in the
///     Llama table, which COLLIDES with `ffn_norm` → `post_attn_norm`.
///     Muse Glimmer has four distinct per-layer norms, so, exactly like
///     Gemma, `post_attention_norm` and `post_ffw_norm` stay verbatim
///     (`muse_glimmer.cpp` reads `arch.post_attn_norm_key =
///     "post_attention_norm"` / `post_ffn_norm_key = "post_ffw_norm"`)
///     and only `ffn_norm` — the PRE-FFN norm — becomes
///     `post_attn_norm.weight` (aliased to runtime `ffn_norm.weight`).
pub fn map_gguf_name(n: &str) -> Option<String> {
    match n {
        "token_embd.weight" => return Some("embed_tokens.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        "output_norm.weight" => return Some("final_norm.weight".into()),
        // RoPE is recomputed at runtime from `rope_theta`; Muse Glimmer
        // has no partial rotary, so the divisor mask carries no
        // information Gemma 4 would need preserved.
        "rope_freqs.weight" => return None,
        _ => {}
    }

    let rest = n.strip_prefix("blk.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;
    let canonical = match suffix {
        "attn_norm.weight" => "input_norm.weight",
        // Pre-FFN norm → the canonical the alias table maps to the
        // runtime's `ffn_norm.weight`.
        "ffn_norm.weight" => "post_attn_norm.weight",
        // Kept verbatim — read directly by name, see the doc comment.
        "post_attention_norm.weight" => "post_attention_norm.weight",
        "post_ffw_norm.weight" => "post_ffw_norm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        // Separate `[n_heads*head_dim, hidden]` sigmoid gate over the
        // layer input, applied to the attention output before o_proj.
        "attn_gate.weight" => "self_attn.gate_proj.weight",
        // The QK norm is WEIGHTLESS on this arch, so these should not
        // exist. Map them rather than failing the conversion if an
        // exporter emits all-ones tensors: `muse_glimmer.cpp` uses
        // `emit_head_rmsnorm_noweight` unconditionally and never reads
        // them, so they are inert ballast at worst.
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        // Unknown — fail loud rather than silently dropping a weight.
        _ => return None,
    };
    Some(format!("layers.{layer}.{canonical}"))
}

/// Map a Muse Glimmer perception-tower tensor name (leading `model.`
/// already stripped) to its canonical `.base` mmproj name.
///
/// Targets the same `vision.*` vocabulary the Gemma 4 tower uses so the
/// runtime's weight lookup stays arch-agnostic. Returns `None` when no rule
/// matches; the caller then passes the name through verbatim.
///
/// Note the tower is a plain LayerNorm ViT with SEPARATE q/k/v projections
/// that all carry biases — unlike Qwen3.5's fused, bias-carrying QKV — so
/// each maps individually.
pub fn map_mmproj_name(n: &str) -> Option<String> {
    // ── Projector (siblings of the tower) ───────────────────────────
    match n {
        "vision_adapter.fc1.weight" => return Some("vision.adapter.fc1.weight".into()),
        "vision_adapter.fc2.weight" => return Some("vision.adapter.fc2.weight".into()),
        "vision_projection.weight" => return Some("vision.projection.weight".into()),
        _ => {}
    }

    let rest = n.strip_prefix("vision_tower.")?;

    // ── Patch embedder + learned position table ─────────────────────
    // `patch_embedding` is a Linear over flattened
    // [patch_temporal * 3 * patch^2] pixels (1176 → 1536), bias-free.
    // `position_embedding_table` is a [pos_h*pos_w, dim] grid that the
    // runtime bilinearly resamples to each image's patch grid.
    match rest {
        "patch_embedder.patch_embedding.weight" => return Some("vision.patch_embed.weight".into()),
        "patch_embedder.position_embedding_table.weight" => {
            return Some("vision.pos_embed.weight".into())
        }
        "ln_pre.weight" => return Some("vision.ln_pre.weight".into()),
        "ln_pre.bias" => return Some("vision.ln_pre.bias".into()),
        "ln_post.weight" => return Some("vision.ln_post.weight".into()),
        "ln_post.bias" => return Some("vision.ln_post.bias".into()),
        _ => {}
    }

    // ── Encoder blocks ──────────────────────────────────────────────
    let layer_rest = rest.strip_prefix("layers.")?;
    let (idx, suffix) = layer_rest.split_once('.')?;
    idx.parse::<u32>().ok()?;
    let canonical_suffix = vision_layer_suffix(suffix)?;
    Some(format!("vision.layers.{idx}.{canonical_suffix}"))
}

/// Map a perception-tower tensor name as it appears in an **mmproj GGUF**
/// (`general.type = mmproj`, `clip.projector_type = muse-glimmer`) to the
/// same canonical `vision.*` vocabulary [`map_mmproj_name`] produces from
/// the HF checkpoint. Both sources must land on identical names or the
/// runtime's weight lookup would see two different towers.
///
/// The GGUF tower uses llama.cpp's `clip` vocabulary (`v.blk.N.*`, `v.*`,
/// `mm.N.*`) rather than HF's (`vision_tower.layers.N.*`), so this is a
/// separate table, not a pre-pass on [`map_mmproj_name`].
///
/// The projector is three stacked linears, named positionally in GGUF. The
/// widths pin the correspondence and are asserted by the tests below:
///   mm.0 [6144 -> 4096]  adapter.fc1   (6144 = dim * spatial_merge^2)
///   mm.1 [4096 -> 4096]  adapter.fc2
///   mm.2 [4096 -> 6656]  projection    (6656 = text embedding_length)
///
/// Returns `None` for anything unrecognized — the caller must fail loudly
/// rather than drop a tower weight, since a silently missing projector row
/// produces plausible-looking garbage rather than an error.
pub fn map_mmproj_gguf_name(n: &str) -> Option<String> {
    // ── Projector ───────────────────────────────────────────────────
    match n {
        "mm.0.weight" => return Some("vision.adapter.fc1.weight".into()),
        "mm.1.weight" => return Some("vision.adapter.fc2.weight".into()),
        "mm.2.weight" => return Some("vision.projection.weight".into()),
        _ => {}
    }

    // ── Tower-level tensors ─────────────────────────────────────────
    match n {
        "v.patch_embd.weight" => return Some("vision.patch_embed.weight".into()),
        "v.position_embd.weight" => return Some("vision.pos_embed.weight".into()),
        "v.pre_ln.weight" => return Some("vision.ln_pre.weight".into()),
        "v.pre_ln.bias" => return Some("vision.ln_pre.bias".into()),
        "v.post_ln.weight" => return Some("vision.ln_post.weight".into()),
        "v.post_ln.bias" => return Some("vision.ln_post.bias".into()),
        _ => {}
    }

    // ── Encoder blocks: v.blk.<i>.<stem>.<weight|bias> ──────────────
    let rest = n.strip_prefix("v.blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<u32>().ok()?;
    let (stem, tail) = match suffix.rsplit_once('.') {
        Some((stem, tail @ ("weight" | "bias"))) => (stem, tail),
        _ => return None,
    };
    // ln1/ln2 are pre-attention and pre-FFN respectively, matching HF's
    // norm1/norm2 — NOT a post-norm arrangement.
    let canonical_stem = match stem {
        "ln1" => "attention_norm",
        "ln2" => "ffn_norm",
        "attn_q" => "attention.q",
        "attn_k" => "attention.k",
        "attn_v" => "attention.v",
        "attn_out" => "attention.output",
        "ffn_up" => "ffn.up",
        "ffn_down" => "ffn.down",
        _ => return None,
    };
    Some(format!("vision.layers.{idx}.{canonical_stem}.{tail}"))
}

/// Encoder-block suffix rename. `.weight` / `.bias` are carried through
/// together since every linear and norm in this tower has both (except the
/// patch embedder, handled above).
fn vision_layer_suffix(s: &str) -> Option<String> {
    let (stem, tail) = match s.rsplit_once('.') {
        Some((stem, tail @ ("weight" | "bias"))) => (stem, tail),
        _ => return None,
    };
    let canonical_stem = match stem {
        "norm1" => "attention_norm",
        "norm2" => "ffn_norm",
        "attn.q_proj" => "attention.q",
        "attn.k_proj" => "attention.k",
        "attn.v_proj" => "attention.v",
        // `attn.proj` is the output projection (HF ViT naming), NOT a
        // gate — the perception tower has no gating.
        "attn.proj" => "attention.output",
        "mlp.fc1" => "ffn.up",
        "mlp.fc2" => "ffn.down",
        _ => return None,
    };
    Some(format!("{canonical_stem}.{tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed mirror of meta-models/Muse-Glimmer-30B/config.json — the
    /// 4-layer form so the schedules stay readable. Layer types follow the
    /// released 52-layer file: sliding x3 then full, with NoPE on the full
    /// layer.
    fn cfg() -> serde_json::Value {
        serde_json::json!({
            "model_type": "muse_glimmer",
            "image_token_id": 200092,
            "video_token_id": 200091,
            "out_hidden_size": 6144,
            "projector_hidden_size": 4096,
            "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 6656,
                "intermediate_size": 19968,
                "num_hidden_layers": 4,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "vocab_size": 202048,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-5,
                "post_norm_eps": 1e-8,
                "sliding_window": 2048,
                "tie_word_embeddings": false,
                "final_logit_softcapping": 20.0,
                "qk_scale_factor": 3.87,
                "output_multiplier": 0.19611613513818404,
                "bos_token_id": 200000,
                "eos_token_id": 200001,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
                ],
                "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0],
                "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" }
            }
        })
    }

    #[test]
    fn reads_text_config_and_rope_parameters() {
        let c = MuseGlimmerHfMapper.config_from_hf(&cfg()).unwrap();
        assert_eq!(c.hidden_size, 6656);
        assert_eq!(c.num_hidden_layers, 4);
        assert_eq!(c.num_kv_heads, 2);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.vocab_size, 202048);
        // Must come from rope_parameters, NOT the 10000 default.
        assert_eq!(c.rope_theta, 500_000.0);
        assert!(!c.tie_word_embeddings);
    }

    #[test]
    fn extracts_muse_glimmer_scales() {
        let c = MuseGlimmerHfMapper.config_from_hf(&cfg()).unwrap();
        assert_eq!(c.qk_scale_factor, 3.87);
        assert_eq!(c.logit_softcap, 20.0);
        assert_eq!(c.post_norm_eps, 1e-8);
        assert!((c.output_multiplier - 0.196_116_14).abs() < 1e-7);
        assert!(c.attn_output_gate);
        assert_eq!(c.sliding_window, 2048);
    }

    /// The local/local/local/global schedule and the NoPE mask must line
    /// up: every global layer is NoPE, every sliding layer keeps RoPE.
    #[test]
    fn swa_and_nope_schedules_are_complementary() {
        let c = MuseGlimmerHfMapper.config_from_hf(&cfg()).unwrap();
        assert_eq!(c.swa_layers, vec![true, true, true, false]);
        assert_eq!(c.nope_layers, vec![false, false, false, true]);
    }

    /// A config without `layer_types` / `layer_rope_theta` must reproduce
    /// the reference's "every 4th layer counted backward from the last"
    /// default rather than defaulting everything to full-attention+RoPE.
    #[test]
    fn derives_default_schedules_when_keys_absent() {
        let mut v = cfg();
        let tc = v.get_mut("text_config").unwrap();
        tc.as_object_mut().unwrap().remove("layer_types");
        tc.as_object_mut().unwrap().remove("layer_rope_theta");
        tc["num_hidden_layers"] = serde_json::json!(8);

        let c = MuseGlimmerHfMapper.config_from_hf(&v).unwrap();
        // Counting backward from layer 7: 7, 3 are global/NoPE.
        assert_eq!(
            c.swa_layers,
            vec![true, true, true, false, true, true, true, false]
        );
        assert_eq!(
            c.nope_layers,
            vec![false, false, false, true, false, false, false, true]
        );
    }

    /// Only the four zero-centered per-layer norms get the +1 offset.
    /// Shifting `final_norm.weight` (a plain weighted RMSNorm here, unlike
    /// Gemma 3) or a vision LayerNorm would corrupt the forward pass.
    #[test]
    fn norm_shift_scoped_to_centered_per_layer_norms() {
        let m = MuseGlimmerHfMapper;
        for s in [
            "layers.0.input_norm.weight",
            "layers.7.post_attention_norm.weight",
            "layers.7.post_attn_norm.weight",
            "layers.51.post_ffw_norm.weight",
        ] {
            assert_eq!(m.norm_shift(s), 1.0, "should shift: {s}");
        }
        for s in [
            "final_norm.weight",
            "embed_tokens.weight",
            "lm_head.weight",
            "vision.layers.0.attention_norm.weight",
            "vision.ln_post.weight",
            "layers.0.attention.q.weight",
        ] {
            assert_eq!(m.norm_shift(s), 0.0, "should not shift: {s}");
        }
    }

    /// The embedding norm is baked into the matrix; nothing else is, and
    /// the bake is suppressed if the checkpoint ever ties the head.
    #[test]
    fn row_rms_normalize_targets_embedding_only() {
        let m = MuseGlimmerHfMapper;
        let mut c = m.config_from_hf(&cfg()).unwrap();
        assert_eq!(m.row_rms_normalize("embed_tokens.weight", &c), Some(1e-5));
        assert_eq!(m.row_rms_normalize("lm_head.weight", &c), None);
        assert_eq!(m.row_rms_normalize("final_norm.weight", &c), None);

        c.tie_word_embeddings = true;
        assert_eq!(m.row_rms_normalize("embed_tokens.weight", &c), None);
    }

    #[test]
    fn maps_perception_tower_tensors() {
        let cases = [
            (
                "vision_tower.patch_embedder.patch_embedding.weight",
                "vision.patch_embed.weight",
            ),
            (
                "vision_tower.patch_embedder.position_embedding_table.weight",
                "vision.pos_embed.weight",
            ),
            ("vision_tower.ln_pre.weight", "vision.ln_pre.weight"),
            ("vision_tower.ln_post.bias", "vision.ln_post.bias"),
            (
                "vision_tower.layers.0.norm1.weight",
                "vision.layers.0.attention_norm.weight",
            ),
            (
                "vision_tower.layers.49.norm2.bias",
                "vision.layers.49.ffn_norm.bias",
            ),
            (
                "vision_tower.layers.3.attn.q_proj.bias",
                "vision.layers.3.attention.q.bias",
            ),
            (
                "vision_tower.layers.3.attn.k_proj.weight",
                "vision.layers.3.attention.k.weight",
            ),
            (
                "vision_tower.layers.3.attn.v_proj.weight",
                "vision.layers.3.attention.v.weight",
            ),
            (
                "vision_tower.layers.3.attn.proj.weight",
                "vision.layers.3.attention.output.weight",
            ),
            (
                "vision_tower.layers.7.mlp.fc1.weight",
                "vision.layers.7.ffn.up.weight",
            ),
            (
                "vision_tower.layers.7.mlp.fc2.bias",
                "vision.layers.7.ffn.down.bias",
            ),
            ("vision_adapter.fc1.weight", "vision.adapter.fc1.weight"),
            ("vision_adapter.fc2.weight", "vision.adapter.fc2.weight"),
            ("vision_projection.weight", "vision.projection.weight"),
        ];
        for (src, want) in cases {
            assert_eq!(map_mmproj_name(src).as_deref(), Some(want), "mapping {src}");
        }
    }

    #[test]
    fn mmproj_gguf_names_map_to_canonical() {
        let cases = [
            // Tower-level.
            ("v.patch_embd.weight", "vision.patch_embed.weight"),
            ("v.position_embd.weight", "vision.pos_embed.weight"),
            ("v.pre_ln.weight", "vision.ln_pre.weight"),
            ("v.pre_ln.bias", "vision.ln_pre.bias"),
            ("v.post_ln.weight", "vision.ln_post.weight"),
            ("v.post_ln.bias", "vision.ln_post.bias"),
            // Encoder blocks — every stem, both tails.
            ("v.blk.0.ln1.weight", "vision.layers.0.attention_norm.weight"),
            ("v.blk.0.ln1.bias", "vision.layers.0.attention_norm.bias"),
            ("v.blk.2.ln2.weight", "vision.layers.2.ffn_norm.weight"),
            ("v.blk.3.attn_q.weight", "vision.layers.3.attention.q.weight"),
            ("v.blk.3.attn_k.bias", "vision.layers.3.attention.k.bias"),
            ("v.blk.3.attn_v.weight", "vision.layers.3.attention.v.weight"),
            ("v.blk.3.attn_out.weight", "vision.layers.3.attention.output.weight"),
            ("v.blk.7.ffn_up.weight", "vision.layers.7.ffn.up.weight"),
            ("v.blk.7.ffn_down.bias", "vision.layers.7.ffn.down.bias"),
            // Projector.
            ("mm.0.weight", "vision.adapter.fc1.weight"),
            ("mm.1.weight", "vision.adapter.fc2.weight"),
            ("mm.2.weight", "vision.projection.weight"),
        ];
        for (src, want) in cases {
            assert_eq!(
                map_mmproj_gguf_name(src).as_deref(),
                Some(want),
                "gguf mapping {src}"
            );
        }

        // Unknown names fail loudly rather than passing through: a dropped
        // tower weight yields plausible garbage, not an error.
        for bad in [
            "v.blk.0.attn_gate.weight", // tower has no gating
            "v.blk.0.ln1.scale",        // not weight/bias
            "v.blk.x.ln1.weight",       // non-numeric index
            "mm.3.weight",              // projector is exactly three linears
            "blk.0.attn_q.weight",      // text tensor, wrong table
        ] {
            assert_eq!(map_mmproj_gguf_name(bad), None, "should reject {bad}");
        }
    }

    /// The HF and GGUF towers MUST land on identical canonical names —
    /// otherwise a bundle converted from GGUF and one converted from
    /// safetensors present different weights to the same runtime lookup.
    #[test]
    fn mmproj_hf_and_gguf_agree() {
        let pairs = [
            ("vision_tower.layers.5.norm1.weight", "v.blk.5.ln1.weight"),
            ("vision_tower.layers.5.norm2.bias", "v.blk.5.ln2.bias"),
            ("vision_tower.layers.5.attn.q_proj.weight", "v.blk.5.attn_q.weight"),
            ("vision_tower.layers.5.attn.k_proj.bias", "v.blk.5.attn_k.bias"),
            ("vision_tower.layers.5.attn.v_proj.weight", "v.blk.5.attn_v.weight"),
            ("vision_tower.layers.5.attn.proj.weight", "v.blk.5.attn_out.weight"),
            ("vision_tower.layers.5.mlp.fc1.weight", "v.blk.5.ffn_up.weight"),
            ("vision_tower.layers.5.mlp.fc2.bias", "v.blk.5.ffn_down.bias"),
            ("vision_tower.ln_pre.weight", "v.pre_ln.weight"),
            ("vision_tower.ln_post.bias", "v.post_ln.bias"),
            (
                "vision_tower.patch_embedder.patch_embedding.weight",
                "v.patch_embd.weight",
            ),
            (
                "vision_tower.patch_embedder.position_embedding_table.weight",
                "v.position_embd.weight",
            ),
            ("vision_adapter.fc1.weight", "mm.0.weight"),
            ("vision_adapter.fc2.weight", "mm.1.weight"),
            ("vision_projection.weight", "mm.2.weight"),
        ];
        for (hf, gguf) in pairs {
            let a = map_mmproj_name(hf);
            let b = map_mmproj_gguf_name(gguf);
            assert!(a.is_some(), "hf mapping missing for {hf}");
            assert_eq!(a, b, "hf {hf} and gguf {gguf} must agree");
        }
    }

    // ── GGUF path ───────────────────────────────────────────────────

    /// Trimmed `muse-glimmer.*` metadata — note the HYPHEN, which is what
    /// `general.architecture` (and therefore every key prefix) actually
    /// carries in the llama.cpp-produced file.
    fn gguf_meta() -> BTreeMap<String, KvValue> {
        let mut m = BTreeMap::new();
        m.insert("muse-glimmer.embedding_length".into(), KvValue::U32(6656));
        m.insert("muse-glimmer.block_count".into(), KvValue::U32(4));
        m.insert("muse-glimmer.attention.head_count".into(), KvValue::U32(32));
        m.insert("muse-glimmer.attention.head_count_kv".into(), KvValue::U32(2));
        m.insert("muse-glimmer.feed_forward_length".into(), KvValue::U32(19968));
        m.insert("muse-glimmer.vocab_size".into(), KvValue::U32(202048));
        m.insert("muse-glimmer.attention.key_length".into(), KvValue::U32(128));
        m.insert("muse-glimmer.rope.freq_base".into(), KvValue::F32(500_000.0));
        m.insert("muse-glimmer.context_length".into(), KvValue::U32(131072));
        m.insert(
            "muse-glimmer.attention.layer_norm_rms_epsilon".into(),
            KvValue::F32(1e-5),
        );
        m.insert("tokenizer.ggml.bos_token_id".into(), KvValue::U32(200000));
        m.insert("tokenizer.ggml.eos_token_id".into(), KvValue::U32(200001));
        m
    }

    /// The hyphenated arch string must resolve, and it must resolve to a
    /// mapper whose canonical arch is the UNDERSCORED form so GGUF- and
    /// HF-sourced bundles carry the same `arch` header field.
    #[test]
    fn hyphenated_gguf_arch_maps_to_underscored_canonical() {
        let m = crate::source_mapper_for_gguf("muse-glimmer")
            .expect("hyphenated `muse-glimmer` must resolve");
        assert_eq!(m.canonical_arch(), "muse_glimmer");
        assert_eq!(
            crate::source_mapper_for_gguf("muse_glimmer")
                .map(|m| m.canonical_arch()),
            Some("muse_glimmer"),
            "underscored spelling accepted too"
        );
    }

    #[test]
    fn gguf_config_matches_hf_config() {
        let g = MuseGlimmerGgufMapper.config_from_gguf(&gguf_meta()).unwrap();
        let h = MuseGlimmerHfMapper.config_from_hf(&cfg()).unwrap();
        assert_eq!(g.hidden_size, h.hidden_size);
        assert_eq!(g.num_hidden_layers, h.num_hidden_layers);
        assert_eq!(g.num_attention_heads, h.num_attention_heads);
        assert_eq!(g.num_kv_heads, h.num_kv_heads);
        assert_eq!(g.head_dim, h.head_dim);
        assert_eq!(g.intermediate_size, h.intermediate_size);
        assert_eq!(g.vocab_size, h.vocab_size);
        assert_eq!(g.rope_theta, h.rope_theta);
        assert_eq!(g.sliding_window, h.sliding_window);
        assert_eq!(g.swa_layers, h.swa_layers);
        assert_eq!(g.nope_layers, h.nope_layers);
        assert!(g.attn_output_gate);
        assert!(!g.tie_word_embeddings);
    }

    /// A zero `qk_scale_factor` silently shrinks every attention logit, so
    /// the mapper falls back to the published constants when llama.cpp
    /// doesn't export them. `output_multiplier` is DERIVED from
    /// hidden_size rather than hardcoded.
    #[test]
    fn gguf_falls_back_to_reference_scales() {
        let g = MuseGlimmerGgufMapper.config_from_gguf(&gguf_meta()).unwrap();
        assert_eq!(g.qk_scale_factor, 3.87);
        assert_eq!(g.logit_softcap, 20.0);
        assert_eq!(g.post_norm_eps, 1e-8);
        // 1/sqrt(6656/256) = 1/sqrt(26).
        assert!((g.output_multiplier - 0.196_116_14).abs() < 1e-6);
    }

    /// Exported keys must win over the fallbacks.
    #[test]
    fn gguf_prefers_exported_scales() {
        let mut m = gguf_meta();
        m.insert(
            "muse-glimmer.attention.qk_scale_factor".into(),
            KvValue::F32(2.5),
        );
        m.insert("muse-glimmer.output_multiplier".into(), KvValue::F32(0.5));
        m.insert(
            "muse-glimmer.final_logit_softcapping".into(),
            KvValue::F32(30.0),
        );
        m.insert("muse-glimmer.attention.sliding_window".into(), KvValue::U32(1024));
        let g = MuseGlimmerGgufMapper.config_from_gguf(&m).unwrap();
        assert_eq!(g.qk_scale_factor, 2.5);
        assert_eq!(g.output_multiplier, 0.5);
        assert_eq!(g.logit_softcap, 30.0);
        assert_eq!(g.sliding_window, 1024);
    }

    /// Every GGUF tensor name must land on the same canonical name the HF
    /// path produces — that is the whole contract between the two paths.
    #[test]
    fn gguf_names_match_hf_canonicals() {
        let cases = [
            ("token_embd.weight", "embed_tokens.weight"),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "final_norm.weight"),
            ("blk.0.attn_norm.weight", "layers.0.input_norm.weight"),
            // Pre-FFN norm → the alias the runtime reads as `ffn_norm`.
            ("blk.3.ffn_norm.weight", "layers.3.post_attn_norm.weight"),
            // These two are read VERBATIM by muse_glimmer.cpp.
            (
                "blk.7.post_attention_norm.weight",
                "layers.7.post_attention_norm.weight",
            ),
            ("blk.7.post_ffw_norm.weight", "layers.7.post_ffw_norm.weight"),
            ("blk.1.attn_q.weight", "layers.1.self_attn.q_proj.weight"),
            ("blk.1.attn_k.weight", "layers.1.self_attn.k_proj.weight"),
            ("blk.1.attn_v.weight", "layers.1.self_attn.v_proj.weight"),
            (
                "blk.1.attn_output.weight",
                "layers.1.self_attn.o_proj.weight",
            ),
            // NOT `self_attn.gate.weight` — see map_gguf_name's docs.
            ("blk.1.attn_gate.weight", "layers.1.self_attn.gate_proj.weight"),
            ("blk.1.attn_q_norm.weight", "layers.1.self_attn.q_norm.weight"),
            ("blk.1.attn_k_norm.weight", "layers.1.self_attn.k_norm.weight"),
            ("blk.51.ffn_gate.weight", "layers.51.mlp.gate_proj.weight"),
            ("blk.51.ffn_up.weight", "layers.51.mlp.up_proj.weight"),
            ("blk.51.ffn_down.weight", "layers.51.mlp.down_proj.weight"),
        ];
        for (src, want) in cases {
            assert_eq!(map_gguf_name(src).as_deref(), Some(want), "mapping {src}");
        }
        assert_eq!(map_gguf_name("rope_freqs.weight"), None, "dropped");
        assert_eq!(map_gguf_name("blk.0.something_new.weight"), None, "unknown");
        assert_eq!(map_gguf_name("blk.abc.attn_q.weight"), None);
    }

    /// The four per-layer norms must map to four DISTINCT canonical names.
    /// The Llama table collapses `ffn_norm` and `post_attention_norm` onto
    /// `post_attn_norm`, which would leave one of them unreadable.
    #[test]
    fn four_per_layer_norms_stay_distinct() {
        let names: Vec<String> = [
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.post_ffw_norm.weight",
        ]
        .iter()
        .map(|n| map_gguf_name(n).unwrap())
        .collect();
        let mut uniq = names.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 4, "norm canonicals collided: {names:?}");
    }

    /// Only `attn_q` / `attn_k` get the inverse rope permute, with the
    /// right head count each (kv heads for K under GQA). Everything else
    /// — v, o, the attention gate, the FFN — must be left alone.
    #[test]
    fn rope_unpermute_targets_q_and_k_only() {
        let c = MuseGlimmerGgufMapper.config_from_gguf(&gguf_meta()).unwrap();
        let m = MuseGlimmerGgufMapper;
        assert_eq!(
            m.rope_unpermute_heads("layers.0.self_attn.q_proj.weight", &c),
            Some(32)
        );
        assert_eq!(
            m.rope_unpermute_heads("layers.0.self_attn.k_proj.weight", &c),
            Some(2)
        );
        for n in [
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.gate_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert_eq!(m.rope_unpermute_heads(n, &c), None, "must not permute {n}");
        }
    }

    /// Llama must keep its GGUF rows untouched: it runs the INTERLEAVED
    /// `rope_f16` kernel and llama.cpp already exported in that layout, so
    /// a stray un-permute would corrupt it.
    #[test]
    fn other_gguf_archs_do_not_unpermute() {
        let c = ArchConfig {
            num_attention_heads: 32,
            num_kv_heads: 8,
            ..ArchConfig::default()
        };
        for arch in ["llama", "qwen3", "gemma3", "gemma4", "nomic-bert"] {
            let m = crate::source_mapper_for_gguf(arch).unwrap();
            assert_eq!(
                m.rope_unpermute_heads("layers.0.self_attn.q_proj.weight", &c),
                None,
                "{arch} must not un-permute"
            );
        }
    }

    /// Language-tower tensors must not be swallowed by the vision mapper,
    /// and unknown tower members fall through to verbatim pass-through.
    #[test]
    fn ignores_non_tower_tensors() {
        for n in [
            "layers.0.self_attn.q_proj.weight",
            "embed_tokens.weight",
            "vision_tower.layers.0.attn.unknown.weight",
            "vision_tower.layers.abc.norm1.weight",
        ] {
            assert_eq!(map_mmproj_name(n), None, "should not map: {n}");
        }
    }
}
