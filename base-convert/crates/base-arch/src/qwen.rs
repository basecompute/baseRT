//! Qwen2 / Qwen3 GGUF → canonical `.base` mapping.
//!
//! Qwen shares Llama's tensor naming convention; only the config metadata
//! key prefix differs (`qwen2.*` / `qwen3.*`).

use crate::decl::{FoldGroup, TensorClass};
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
    // Multimodal wrappers keep bos/eos at the wrapper level (see
    // backfill_wrapper_token_ids — this is the bug that silently dropped
    // gemma-3-4b's BOS).
    crate::backfill_wrapper_token_ids(&mut config, c);

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
    // The MTP head's two pre-fc norms are Qwen3NextRMSNorm too (zero-centered
    // gamma, HF values near 0) but their names end in `_embedding` /
    // `_hidden`, not `norm.weight`.
    let mtp_pre_fc = canonical.ends_with("pre_fc_norm_embedding.weight")
        || canonical.ends_with("pre_fc_norm_hidden.weight");
    if mtp_pre_fc
        || (canonical.ends_with("norm.weight") && !canonical.ends_with(".linear_attn.norm.weight"))
    {
        1.0
    } else {
        0.0
    }
}

/// MLX sources of Qwen3.5 / 3.6 / 3.8 carry NO norm shift: mlx-lm's
/// `qwen3_next` converter already stores every Qwen3NextRMSNorm as
/// `1 + weight` so its runtime can use the plain `nn.RMSNorm` (measured on
/// `mlx-community/Qwen3.8-27B-4bit`: `input_layernorm` mean 0.97,
/// `q_norm` 1.36, `model.norm` 1.94 — and `linear_attn.norm`, plain in HF
/// too, 0.87). Those are exactly the values the `.base` runtime wants, so
/// the HF +1 above must not be applied again (BAS-959: it was, every norm
/// sat near 2, the bundle decoded garbage and the MTP head accepted
/// nothing).
const QWEN35_MLX_NORM_SHIFT: f32 = 0.0;

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
    fn mlx_norm_shift(&self, _canonical: &str) -> f32 {
        QWEN35_MLX_NORM_SHIFT
    }
    fn tensor_class(&self, canonical: &str, shape: &[u64]) -> TensorClass {
        qwen_decl::class(canonical, shape, QwenBlocks::GDN)
    }
    fn fold_groups(&self, tensors: &[(String, Vec<u64>)], cfg: &ArchConfig) -> Vec<FoldGroup> {
        qwen_decl::fold_groups(tensors, cfg, &|n| self.norm_shift(n), QwenBlocks::GDN)
    }
}

impl crate::HfMapper for Qwen35MoeHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen35moe"
    }
    fn norm_shift(&self, canonical: &str) -> f32 {
        qwen35_norm_shift(canonical)
    }
    fn mlx_norm_shift(&self, _canonical: &str) -> f32 {
        QWEN35_MLX_NORM_SHIFT
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
    fn tensor_class(&self, canonical: &str, shape: &[u64]) -> TensorClass {
        qwen_decl::class(canonical, shape, QwenBlocks::GDN_MOE)
    }
    fn fold_groups(&self, tensors: &[(String, Vec<u64>)], cfg: &ArchConfig) -> Vec<FoldGroup> {
        qwen_decl::fold_groups(tensors, cfg, &|n| self.norm_shift(n), QwenBlocks::GDN_MOE)
    }
}

/// Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`, model_type
/// `qwen4_exp`, text tower `qwen4_exp_text`): the Qwen3.5-MoE hybrid
/// (Gated-DeltaNet + periodic gated full attention, 512-expert top-10 MoE
/// with a gated shared expert) plus three new pieces the runtime reads from
/// the header config: hyper-connections (`hc_count` residual streams mixed
/// by a low-rank gated residual per block), one per-layer n-gram embedding
/// (`ple_*` / `ngram_*`) and the QSA sparse-attention indexer (`indexer_*`).
/// The n-gram hash tables (layer multipliers, per-head vocab sizes and
/// offsets) are derived here exactly as the modeling code derives them
/// (splitmix64 from `seed`, the first primes after `ngram_vocab_size_base`)
/// — the checkpoint's i64 copies are dropped at convert (an i64 tensor has
/// no .base dtype) and these values were checked equal to them.
pub struct Qwen4ExpHfMapper;

fn qwen4_splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(0x9E3779B97F4A7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D049BB133111EB);
    v ^ (v >> 31)
}

fn qwen4_is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v % 2 == 0 {
        return v == 2;
    }
    let mut d = 3u64;
    while d * d <= v {
        if v % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// (layer multipliers [ngram_size], per-head vocab sizes, per-head offsets,
/// padded table rows) for the PLE layer `ple_layer_index`.
pub fn qwen4_ngram_tables(
    vocab_size: u64,
    ngram_size: u64,
    heads_per_ngram: u64,
    vocab_base: u64,
    divisible_by: u64,
    seed: u64,
    ple_layer_index: u64,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, u64) {
    let max_long: u64 = (1u64 << 63) - 1;
    let multiplier_max = max_long / vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = seed.wrapping_add(10007u64.wrapping_mul(ple_layer_index));
    let mut mults = Vec::new();
    for i in 0..ngram_size {
        let v = base_seed.wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(i + 1));
        mults.push(2 * (qwen4_splitmix64(v) % half_bound) + 1);
    }
    let ngram_heads = (ngram_size - 1) * heads_per_ngram;
    let mut sizes = Vec::new();
    let mut offsets = Vec::new();
    let mut total = 0u64;
    for h in 0..ngram_heads {
        let global = ple_layer_index * ngram_heads + h;
        // the (global+1)-th prime strictly after vocab_base - 1
        let mut p = vocab_base - 1;
        for _ in 0..(global + 1) {
            p += 1;
            while !qwen4_is_prime(p) {
                p += 1;
            }
        }
        sizes.push(p);
        offsets.push(total);
        total += p;
    }
    let padded = total.div_ceil(divisible_by) * divisible_by;
    (mults, sizes, offsets, padded)
}

impl crate::HfMapper for Qwen4ExpHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen4_exp"
    }
    fn norm_shift(&self, canonical: &str) -> f32 {
        // Every Qwen4ExpTextRMSNorm is zero-centered (1 + weight): the
        // hyper-connection norms (`hc_norm`), the attention q/k norms, the
        // indexer q/k layernorms, the three PLE norms (`norm_key` /
        // `norm_query` / `norm_conv`) and the MTP head's pre-fc norms. The
        // Gated-DeltaNet output norm (`linear_attn.norm`) is a plain-weight
        // RMSNormGated, as on Qwen3.5.
        if canonical.ends_with(".norm_key.weight")
            || canonical.ends_with(".norm_query.weight")
            || canonical.ends_with(".norm_conv.weight")
        {
            1.0
        } else {
            qwen35_norm_shift(canonical)
        }
    }
    fn mlx_norm_shift(&self, _canonical: &str) -> f32 {
        QWEN35_MLX_NORM_SHIFT
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        use serde_json::json;
        let mut config = Qwen35MoeHfMapper.config_from_hf(c)?;
        let tc = c.get("text_config").unwrap_or(c);
        let u64_v = |k: &str| tc.get(k).and_then(|v| v.as_u64());
        let hc_count = u64_v("hc_count").unwrap_or(1);
        let hc_lowrank = u64_v("hc_lowrank").unwrap_or(0);
        let ple_ids: Vec<u64> = tc
            .get("ple_layer_ids")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        let ngram_size = u64_v("ngram_size").unwrap_or(3);
        let heads_per_ngram = u64_v("heads_per_ngram").unwrap_or(8);
        let vocab_base = u64_v("ngram_vocab_size_base").unwrap_or(20_000_000);
        let divisible_by = u64_v("make_ngram_vocab_size_divisible_by").unwrap_or(128);
        let seed = u64_v("seed").unwrap_or(1234);
        let x = &mut config.extra_config;
        x.insert("hc_count".into(), json!(hc_count));
        x.insert("hc_lowrank".into(), json!(hc_lowrank));
        x.insert(
            "output_gate_type".into(),
            json!(tc.get("output_gate_type").cloned().unwrap_or(json!("silu"))),
        );
        // Indexer (QSA) geometry.
        for k in [
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        ] {
            if let Some(v) = u64_v(k) {
                x.insert(k.into(), json!(v));
            }
        }
        // The text config's own eos (the PLE n-gram history resets on it).
        if let Some(e) = tc.get("eos_token_id") {
            let eos = if e.is_array() {
                e.as_array()
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_u64())
            } else {
                e.as_u64()
            };
            if let Some(eos) = eos {
                x.insert("ple_eos_token_id".into(), json!(eos));
            }
        }
        if !ple_ids.is_empty() {
            x.insert("ple_layer_ids".into(), json!(ple_ids));
            x.insert(
                "ple_embed_dim".into(),
                json!(u64_v("ple_embed_dim").unwrap_or(config.hidden_size as u64)),
            );
            x.insert(
                "ple_conv_kernel_size".into(),
                json!(u64_v("ple_conv_kernel_size").unwrap_or(4)),
            );
            x.insert("ngram_size".into(), json!(ngram_size));
            x.insert("heads_per_ngram".into(), json!(heads_per_ngram));
            x.insert("ngram_vocab_size_base".into(), json!(vocab_base));
            x.insert(
                "split_ngram_parts".into(),
                json!(u64_v("split_ngram_parts").unwrap_or(128)),
            );
            // One table set per PLE layer, in ple_layer_ids order.
            let mut mults_all = Vec::new();
            let mut sizes_all = Vec::new();
            let mut offs_all = Vec::new();
            let mut padded_all = Vec::new();
            for (i, _) in ple_ids.iter().enumerate() {
                let (m, s, o, p) = qwen4_ngram_tables(
                    config.vocab_size as u64,
                    ngram_size,
                    heads_per_ngram,
                    vocab_base,
                    divisible_by,
                    seed,
                    i as u64,
                );
                mults_all.push(json!(m));
                sizes_all.push(json!(s));
                offs_all.push(json!(o));
                padded_all.push(json!(p));
            }
            x.insert("ngram_layer_multipliers".into(), json!(mults_all));
            x.insert("ngram_head_vocab_sizes".into(), json!(sizes_all));
            x.insert("ngram_head_offsets".into(), json!(offs_all));
            x.insert("ngram_padded_vocab".into(), json!(padded_all));
        }
        Ok(config)
    }
    fn tensor_class(&self, canonical: &str, shape: &[u64]) -> TensorClass {
        qwen_decl::class(canonical, shape, QwenBlocks::FLASH_NEXT)
    }
    fn fold_groups(&self, tensors: &[(String, Vec<u64>)], cfg: &ArchConfig) -> Vec<FoldGroup> {
        qwen_decl::fold_groups(
            tensors,
            cfg,
            &|n| self.norm_shift(n),
            QwenBlocks::FLASH_NEXT,
        )
    }
}

/// Which blocks a Qwen family has, for the shared declaration below.
#[derive(Copy, Clone)]
pub struct QwenBlocks {
    /// Gated-DeltaNet linear-attention blocks (Qwen3.5+).
    pub gdn: bool,
    /// Routed experts with a gated shared expert.
    pub moe: bool,
    /// Qwen3.8-Flash-Next: hyper-connections (no plain norm in front of a
    /// block), the PLE n-gram embedding, the QSA indexer.
    pub flash_next: bool,
}

impl QwenBlocks {
    pub const DENSE: QwenBlocks = QwenBlocks {
        gdn: false,
        moe: false,
        flash_next: false,
    };
    pub const MOE: QwenBlocks = QwenBlocks {
        gdn: false,
        moe: true,
        flash_next: false,
    };
    pub const GDN: QwenBlocks = QwenBlocks {
        gdn: true,
        moe: false,
        flash_next: false,
    };
    pub const GDN_MOE: QwenBlocks = QwenBlocks {
        gdn: true,
        moe: true,
        flash_next: false,
    };
    pub const FLASH_NEXT: QwenBlocks = QwenBlocks {
        gdn: true,
        moe: true,
        flash_next: true,
    };
}

/// §4.8 classes and fold groups of the Qwen families.
pub mod qwen_decl {
    use super::QwenBlocks;
    use crate::decl::{dense, moe, tail, Declare, FoldGroup, TensorClass, TensorSet};
    use crate::ArchConfig;

    /// Gated-DeltaNet GEMMs, `in_proj_a` / `in_proj_b` (the per-head decay
    /// and beta gates, `[n_heads, dim]`) included: small, but the encoder
    /// reads them through `dispatch_gemm` (prefill and multi-row decode),
    /// so they are `linear` whatever their size and the block's `attn`
    /// fold may list them. The conv, the gated norm, `A_log` and `dt_bias`
    /// are the recurrence's own: `other`.
    pub const GDN_LINEARS: [&str; 5] = [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.out_proj.weight",
    ];
    /// The gated shared expert's SwiGLU; `mlp.shared_expert_gate` (the
    /// scalar sigmoid gate) and `mlp.gate` (the router) are `other`.
    pub const SHEXP_LINEARS: [&str; 3] = [
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
    ];
    /// Flash-Next: the PLE key / value projections and the hyper-connection
    /// `W_down` (`input_mix_weight_down`, `[hc_lowrank, hc·dim]`) / `W_up`
    /// (`input_mix_weight_up`, `[hc·dim, hc_lowrank]`) are ordinary GEMMs
    /// (§14.2 item 1); `block_inject_weight` (`W_inj`, `[hc, dim]`), the
    /// `hc_norm` gains and the indexer projection are `other`.
    pub const FLASH_NEXT_LINEAR_SUFFIXES: [&str; 4] = [
        "ple.key_proj.weight",
        "ple.value_proj.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
    ];
    pub const NGRAM_TABLE_SUFFIX: &str = "ple.ple_embedding.ngram_embedding.weight";

    pub fn class(canonical: &str, shape: &[u64], blocks: QwenBlocks) -> TensorClass {
        if blocks.flash_next && canonical.ends_with(NGRAM_TABLE_SUFFIX) {
            return TensorClass::Embedding;
        }
        if blocks.moe && moe::is_stack(canonical, shape) {
            return TensorClass::ExpertStack;
        }
        let t = tail(canonical);
        if blocks.gdn && GDN_LINEARS.contains(&t) {
            return TensorClass::Linear;
        }
        if blocks.moe && SHEXP_LINEARS.contains(&t) {
            return TensorClass::Linear;
        }
        if blocks.flash_next && FLASH_NEXT_LINEAR_SUFFIXES.iter().any(|s| t.ends_with(s)) {
            return TensorClass::Linear;
        }
        dense::class(canonical, shape)
    }

    /// `gdn_out`: the Gated-DeltaNet output norm (`RMSNormGated`, gain of
    /// `head_dim`, applied per head after the normalization and the
    /// elementwise `silu(z)` gate) absorbs `1/s`, `out_proj` takes `s` on
    /// its `n_heads·head_dim` columns — `s` is tied across heads (the fold
    /// plan tiles it with the gain's period).
    pub fn gdn_out(d: &mut Declare, layer: usize) {
        let p = format!("layers.{layer}.linear_attn.");
        d.gamma(
            "gdn_out",
            Some(layer),
            format!("{p}norm.weight"),
            &[format!("{p}out_proj.weight")],
        );
    }

    pub fn fold_groups(
        tensors: &[(String, Vec<u64>)],
        cfg: &ArchConfig,
        norm_shift: &dyn Fn(&str) -> f32,
        blocks: QwenBlocks,
    ) -> Vec<FoldGroup> {
        let set = TensorSet::new(tensors);
        let mut d = Declare::new(&set, norm_shift);
        for layer in set.layers() {
            if blocks.flash_next {
                // Hyper-connections: a block's input is the gated mix of
                // the residual streams' `hc_norm` outputs, which also feed
                // `W_inj`, the indexer, the router and `shared_expert_gate`
                // — `other` consumers, so the input-side groups (q/k/v,
                // in_proj_qkv/z/a/b, gate_up) declare no fold (§14.7 c).
                // Only the output-side folds remain.
                let p = format!("layers.{layer}.self_attn.");
                d.rows(
                    "o",
                    Some(layer),
                    format!("{p}v_proj.weight"),
                    format!("{p}o_proj.weight"),
                    crate::decl::UpstreamRows::All,
                );
            } else {
                // A full-attention block: `attn` (q/k/v) and `o`.
                dense::attn_groups(&mut d, layer);
                // A Gated-DeltaNet block: the same `input_norm` feeds the
                // four GDN projections, all `linear` — one `attn` group.
                let p = format!("layers.{layer}.");
                d.gamma(
                    "attn",
                    Some(layer),
                    format!("{p}input_norm.weight"),
                    &[
                        format!("{p}linear_attn.in_proj_qkv.weight"),
                        format!("{p}linear_attn.in_proj_z.weight"),
                        format!("{p}linear_attn.in_proj_a.weight"),
                        format!("{p}linear_attn.in_proj_b.weight"),
                    ],
                );
                dense::mlp_groups(&mut d, layer);
            }
            if blocks.gdn {
                gdn_out(&mut d, layer);
            }
            if blocks.moe {
                // No input-side `moe` group: `post_attn_norm` also feeds the
                // router and `shared_expert_gate` (`other`).
                moe::down_group(
                    &mut d,
                    layer,
                    "ffn_gate_up_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                );
                dense::shared_expert_down(&mut d, layer);
            }
        }
        if !blocks.flash_next {
            // Flash-Next has no `final_norm`: the head reads the final
            // hyper-connection mixer's output.
            dense::head_group(&mut d, cfg);
        }
        d.finish()
    }
}

impl crate::HfMapper for QwenHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen"
    }
    fn config_from_hf(&self, c: &serde_json::Value) -> Result<crate::ArchConfig> {
        crate::llama::hf_generic_config(c)
    }
    fn tensor_class(&self, canonical: &str, shape: &[u64]) -> TensorClass {
        qwen_decl::class(canonical, shape, QwenBlocks::DENSE)
    }
    fn fold_groups(&self, tensors: &[(String, Vec<u64>)], cfg: &ArchConfig) -> Vec<FoldGroup> {
        qwen_decl::fold_groups(tensors, cfg, &|n| self.norm_shift(n), QwenBlocks::DENSE)
    }
}

impl crate::HfMapper for QwenMoeHfMapper {
    fn canonical_arch(&self) -> &'static str {
        "qwen3_moe"
    }
    fn tensor_class(&self, canonical: &str, shape: &[u64]) -> TensorClass {
        qwen_decl::class(canonical, shape, QwenBlocks::MOE)
    }
    fn fold_groups(&self, tensors: &[(String, Vec<u64>)], cfg: &ArchConfig) -> Vec<FoldGroup> {
        qwen_decl::fold_groups(tensors, cfg, &|n| self.norm_shift(n), QwenBlocks::MOE)
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
        let num_kv_heads =
            u32_key(&format!("{prefix}.attention.head_count_kv")).unwrap_or(num_attention_heads);
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
        let num_kv_heads =
            u32_key(&format!("{prefix}.attention.head_count_kv")).unwrap_or(num_attention_heads);
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
            // The MTP head's norms, under the speculator prefix.
            "mtp.layers.0.input_norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ] {
            assert_eq!(qwen35_norm_shift(s), 1.0, "should shift: {s}");
        }
        // The gated GDN norm and everything non-norm do NOT shift.
        for s in [
            "layers.0.linear_attn.norm.weight",
            "layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.mlp.gate_proj.weight",
            "embed_tokens.weight",
            "mtp.fc_embed.weight",
            "mtp.fc_hidden.weight",
        ] {
            assert_eq!(qwen35_norm_shift(s), 0.0, "should not shift: {s}");
        }
    }

    /// An MLX source already stores `1 + weight` (mlx-lm bakes the offset),
    /// so the MLX rule is "never shift" for every tensor the HF rule shifts
    /// — dense and MoE mappers alike (BAS-959).
    #[test]
    fn qwen35_mlx_sources_are_never_shifted() {
        use crate::HfMapper;
        let shifted_by_hf = [
            "layers.0.input_norm.weight",
            "layers.3.post_attn_norm.weight",
            "layers.3.self_attn.q_norm.weight",
            "layers.3.self_attn.k_norm.weight",
            "final_norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "layers.0.linear_attn.norm.weight",
            "embed_tokens.weight",
        ];
        for m in [
            &Qwen35HfMapper as &dyn HfMapper,
            &Qwen35MoeHfMapper as &dyn HfMapper,
        ] {
            for s in shifted_by_hf {
                assert_eq!(
                    m.mlx_norm_shift(s),
                    0.0,
                    "{}: MLX source must not shift {s}",
                    m.canonical_arch()
                );
            }
            // The HF rule is untouched.
            assert_eq!(m.norm_shift("layers.0.input_norm.weight"), 1.0);
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
