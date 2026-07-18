//! Per-architecture tensor-name normalization + config extraction.
//!
//! Maps source tensor names (GGUF `blk.0.attn_q.weight`, MLX
//! `model.layers.0.self_attn.q_proj.weight`) to the canonical `.base`
//! naming (HF convention). Also extracts arch-specific config fields
//! from source metadata so the `.base` header's `config` is populated
//! consistently regardless of source format.

pub mod bert;
pub mod gemma;
pub mod llama;
pub mod qwen;
pub mod tokenizer;

/// What a converter needs to produce one layer's worth of canonical
/// tensor names + fuse/route them into the `.base` writer.
#[derive(Debug, Clone)]
pub struct LayerNames {
    pub layer: u32,
    pub input_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_o: String,
    pub post_attn_norm: String,
    pub mlp_gate: String,
    pub mlp_up: String,
    pub mlp_down: String,
}

/// Dispatch table: which arch module to use for a given GGUF arch string.
pub fn source_mapper_for_gguf(arch: &str) -> Option<&'static dyn GgufMapper> {
    match arch {
        "llama" => Some(&llama::LlamaMapper),
        "qwen2" | "qwen3" | "qwen35" | "qwen36" => Some(&qwen::QwenMapper),
        "qwen2moe" | "qwen3moe" | "qwen35moe" | "qwen36moe" => Some(&qwen::QwenMoeMapper),
        "gemma" | "gemma2" | "gemma3" => Some(&gemma::Gemma3Mapper),
        "gemma4" => Some(&gemma::Gemma4Mapper),
        "nomic-bert" => Some(&bert::NomicBertMapper),
        _ => None,
    }
}

/// HF config.json model_type → mapper. HF tensor names already follow
/// canonical convention, so the mapper only needs to extract ArchConfig
/// from config.json — no tensor renaming.
pub trait HfMapper: Sync {
    fn canonical_arch(&self) -> &'static str;
    fn config_from_hf(&self, config: &serde_json::Value) -> anyhow::Result<ArchConfig>;

    /// Per-element offset added to a 1-D norm-weight tensor at HF→.base
    /// conversion time. Mirrors `convert_hf_to_gguf.py::norm_shift`.
    /// Gemma 3 stores zero-centered RMSNorm gamma and applies the
    /// canonical `(1 + weight)` formulation at inference time; baked
    /// into the tensor up front so the runtime can use the plain
    /// rmsnorm kernel. Default: no shift.
    fn norm_shift(&self, _canonical: &str) -> f32 {
        0.0
    }

    /// RoPE row-permutation head count for a canonical tensor at HF→.base
    /// conversion time, or None when the tensor needs no permutation.
    ///
    /// HF llama-family checkpoints store `q_proj` / `k_proj` in the
    /// transformers "split-half" rotary layout (rotate_half); the runtime
    /// rope kernels — and GGUF sources — use Meta's original interleaved
    /// pair layout. Mirrors `convert_hf_to_gguf.py::LlamaModel.permute`:
    /// out_row[h*HD + 2j + k] = in_row[h*HD + k*HD/2 + j]. Skipping this
    /// keeps attention internally consistent (Q and K scramble identically,
    /// so relative-position structure survives) but assigns every dim-pair
    /// the wrong trained frequency — retrieval collapses as context grows.
    fn rope_permute_heads(&self, _canonical: &str, _cfg: &ArchConfig) -> Option<u32> {
        None
    }
}

pub fn hf_mapper_for_model_type(model_type: &str) -> Option<&'static dyn HfMapper> {
    match model_type {
        // Mistral is Llama-shaped (RMSNorm, RoPE, SwiGLU, GQA), its HF tensor
        // names are already canonical Llama names, and it reuses the Llama mapper
        // (canonical_arch="llama" → llama model class). Validated end-to-end on
        // Mistral-7B-Instruct-v0.3 and Ministral-8B-Instruct-2410.
        //
        // Phi-3 is NOT enabled. The converter side is ready (SplittingProvider
        // splits its fused qkv_proj/gate_up_proj; weights convert; HD=96
        // attention is fine — regression-tested) and the generation_config eos
        // merge below makes it stop cleanly. BUT Phi-3 chat output degenerates
        // (raw completions are coherent; chat floods/repeats and is incoherent at
        // both Q4 and Q8, temp 0 and 0.7) — a chat-path issue (same class as
        // SmolLM2) that's not yet root-caused. Re-enable "phi3" once that's
        // fixed. (Phi-3.5 additionally needs LongRoPE — engine is linear-only.)
        "llama" | "mistral" => Some(&llama::LlamaHfMapper),
        "qwen2" | "qwen3" => Some(&qwen::QwenHfMapper),
        "qwen2_moe" | "qwen3_moe" => Some(&qwen::QwenMoeHfMapper),
        // Qwen3.5 / 3.6: hybrid Gated-DeltaNet + full-attention decoder
        // (reuses the Qwen3-Next design). The top-level HF model_type is
        // `qwen3_5` (multimodal wrapper `Qwen3_5ForConditionalGeneration`)
        // with the text tower under `text_config.model_type = qwen3_5_text`.
        // Both resolve here so a text-only checkpoint (top-level
        // `qwen3_5_text`) and the multimodal wrapper convert identically.
        "qwen3_5" | "qwen3_5_text" | "qwen35" => Some(&qwen::Qwen35HfMapper),
        "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen35_moe" => Some(&qwen::Qwen35MoeHfMapper),
        "nomic_bert" | "nomic-bert" => Some(&bert::NomicBertHfMapper),
        "gemma" | "gemma2" | "gemma3" | "gemma3_text" => Some(&gemma::Gemma3HfMapper),
        // gemma3n is a distinct arch (AltUp/Laurel/per-layer-FFN); the
        // existing local fixture historically named "gemma-4-e2b" was
        // actually google/gemma-3n-E2B-it. The canonical Gemma 4 lives
        // under google/gemma-4-{E2B,E4B}-it and uses model_type=gemma4
        // with text_config.model_type=gemma4_text.
        "gemma4" | "gemma4_text" => Some(&gemma::Gemma4HfMapper),
        _ => None,
    }
}

/// Every HF `model_type` value [`hf_mapper_for_model_type`] accepts. Kept in
/// lockstep with that match so a pre-flight support check (and its error
/// message) has a single source of truth for what convert-on-pull supports.
pub const SUPPORTED_HF_MODEL_TYPES: &[&str] = &[
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen2_moe",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_text",
    "qwen35",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen35_moe",
    "nomic_bert",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma4",
    "gemma4_text",
];

pub trait GgufMapper: Sync {
    /// Canonical arch name stored in the `.base` header's `arch` field.
    fn canonical_arch(&self) -> &'static str;

    /// Extract required config fields from GGUF metadata.
    fn config_from_gguf(
        &self,
        metadata: &std::collections::BTreeMap<String, base_readers::gguf::KvValue>,
    ) -> anyhow::Result<ArchConfig>;

    /// Map a GGUF source tensor name to a canonical `.base` tensor name.
    /// Returns None for tensors that should be dropped (e.g.,
    /// `rope_freqs.weight` — we precompute RoPE elsewhere or recompute
    /// at runtime).
    fn map_tensor_name(&self, gguf_name: &str) -> Option<String>;
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArchConfig {
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub rope_theta: f32,
    pub rope_scale: f32,
    /// HF `rope_scaling.rope_type` (empty = none). The runtime applies the
    /// llama3 piecewise divisor formula only for "llama3" (or, for legacy
    /// headers with no type, llama-arch + factor > 1); "linear" gets the
    /// uniform divisor; anything else is skipped with a warning.
    pub rope_scaling_type: String,
    pub rope_low_freq_factor: f32,
    pub rope_high_freq_factor: f32,
    pub rope_original_max_pos: u32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    /// Per-layer FFN widths when the model declares heterogeneous FFN
    /// (Gemma-4 E2B). Empty = all layers use `intermediate_size`.
    pub per_layer_ffn: Vec<u32>,
    /// Per-layer attention kind: "global" | "sliding". Empty = all
    /// layers use the same attention (indicated by config top-level).
    pub per_layer_attn: Vec<String>,
    /// Per-layer KV-head counts (Gemma 4 26B-A4B / 31B). Empty = all
    /// layers use `num_kv_heads`. Values commonly differ between SWA
    /// (more heads) and Global (fewer heads) layers.
    pub n_kv_heads_per_layer: Vec<u32>,
    /// MoE: total routed experts (0 = dense model).
    pub num_experts: u32,
    /// MoE: top-k experts per token.
    pub num_experts_per_tok: u32,
    /// MoE: per-expert FFN width.  When the model is MoE, runtime
    /// callers compute expert FFN with this and use `intermediate_size`
    /// for the dense shared FFN if present.
    pub moe_intermediate_size: u32,
    /// MoE: 1 if router top-k weights are renormalized to sum to 1
    /// (Qwen), 0 if left as-is (Gemma).
    pub norm_topk_prob: bool,
    /// MoE: number of always-on shared experts running in parallel to the
    /// routed ones (Qwen3.5/3.6-MoE: 1, width `intermediate_size`, plus a
    /// per-token scalar sigmoid gate). 0 = no shared expert.
    pub num_shared_experts: u32,
    /// Maximum positional embedding length.  Pulled from
    /// `max_position_embeddings` (HF) or `context_length` (GGUF).
    pub max_position_embeddings: u32,
    /// Token id used to begin a sequence.  0 = unset.
    pub bos_token_id: u32,
    /// Token id used to end a sequence.  0 = unset.
    pub eos_token_id: u32,
    /// Additional end-of-generation token ids when the HF config ships
    /// `eos_token_id` as an array (Llama-3 instruct: `[128001, 128008,
    /// 128009]`). The primary `eos_token_id` above takes the first
    /// element; the rest land here. Runtime registers each via
    /// `Tokenizer::add_eos_id` so generation honors any of them.
    pub eos_token_ids: Vec<u32>,

    // ── Gemma-4-specific fields (zero/empty for other archs) ─────────
    /// Explicit attention scale used at Q·K^T (Gemma 4 uses 1.0 instead
    /// of the standard 1/sqrt(head_dim)). 0.0 = derive 1/sqrt(head_dim)
    /// at runtime.
    pub attention_scale: f32,
    /// Per-layer-embedding (PLE) input width. 0 = no PLE.
    pub n_embd_per_layer: u32,
    /// First N layers own KV; layers in [n_layer_kv_from_start, n_layers)
    /// reuse an earlier layer's KV cache. 0 = all layers own KV.
    pub n_layer_kv_from_start: u32,
    /// Final logit softcap: y = cap * tanh(x/cap). 0 = disabled.
    pub logit_softcap: f32,
    /// head_dim used for global-attention layers (may differ from
    /// head_dim_swa on Gemma 4). 0 = uniform, use head_dim.
    pub head_dim_global: u32,
    /// head_dim used for sliding-window-attention layers. 0 = uniform.
    pub head_dim_swa: u32,
    /// SWA window size in tokens. 0 = full attention everywhere.
    pub sliding_window: u32,
    /// Sliding-window pattern period (Gemma 3: 6 = every 6th layer is
    /// global). 0 = use `swa_layers` bitfield instead.
    pub sliding_window_pattern: u32,
    /// Per-layer SWA mask (true = sliding-window layer). When non-empty
    /// it overrides `sliding_window_pattern`. Length = num_hidden_layers.
    pub swa_layers: Vec<bool>,
    /// RoPE theta for sliding-window layers (0 = same as `rope_theta`).
    pub rope_local_theta: f32,
    /// Partial-rotary factor for global (full-attention) layers on
    /// Gemma 4. Only the first `factor * head_dim_global / 2` rope
    /// pairs rotate; the remainder stay unchanged. 0 or 1 = full
    /// rotation (no partial). GGUF encodes this via the
    /// `rope_freqs.weight` divisor mask; HF stores it as
    /// `rope_parameters.full_attention.partial_rotary_factor`.
    pub global_rope_partial_factor: f32,

    // ── Qwen3.5 / 3.6 hybrid-linear-attention fields ─────────────────
    // (all zero/empty for non-hybrid archs). Qwen3.5 interleaves
    // Gated-DeltaNet linear-attention layers with periodic full
    // (softmax) attention layers, reusing the Qwen3-Next decoder.
    /// Per-layer attention kind, one entry per layer:
    /// "linear_attention" (Gated DeltaNet) | "full_attention".
    /// Empty = not a hybrid model. Length = num_hidden_layers.
    pub layer_types: Vec<String>,
    /// Every Nth layer is full attention (the rest are linear). Mirror
    /// of `full_attention_interval` in the HF config; 0 = not hybrid.
    /// Redundant with `layer_types` but kept for a cheap runtime check.
    pub full_attention_interval: u32,
    /// Gated-DeltaNet: number of key ("k") heads. 0 = not hybrid.
    pub linear_num_key_heads: u32,
    /// Gated-DeltaNet: number of value ("v") heads.
    pub linear_num_value_heads: u32,
    /// Gated-DeltaNet: per-head key/query dimension.
    pub linear_key_head_dim: u32,
    /// Gated-DeltaNet: per-head value dimension.
    pub linear_value_head_dim: u32,
    /// Gated-DeltaNet: causal depthwise short-conv kernel width (e.g. 4).
    pub linear_conv_kernel_dim: u32,
    /// Full-attention layers apply an output (sigmoid) gate to the
    /// attention output before o_proj (`attn_output_gate`). Qwen3.5=true.
    pub attn_output_gate: bool,
    /// Partial rotary factor for the full-attention layers: only the
    /// first `factor * head_dim` dims are rotated. Qwen3.5 = 0.25.
    /// 0 or 1 = full rotation.
    pub partial_rotary_factor: f32,
    /// Multimodal-RoPE section split (Qwen3.5: [11, 11, 10] over
    /// temporal/height/width). Empty = plain 1-D RoPE. For text-only
    /// inference all positions collapse so the runtime may treat this
    /// as ordinary 1-D RoPE.
    pub mrope_section: Vec<u32>,
    /// mRoPE interleaves the section frequencies rather than
    /// concatenating them (Qwen3.5 = true).
    pub mrope_interleaved: bool,
}

impl ArchConfig {
    /// Convert to the key/value map that populates the `.base` header's
    /// open-namespace `config` section.
    pub fn to_config_map(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        use serde_json::json;
        let mut m = std::collections::BTreeMap::new();
        m.insert("hidden_size".into(), json!(self.hidden_size));
        m.insert("num_hidden_layers".into(), json!(self.num_hidden_layers));
        m.insert(
            "num_attention_heads".into(),
            json!(self.num_attention_heads),
        );
        m.insert("num_key_value_heads".into(), json!(self.num_kv_heads));
        m.insert("head_dim".into(), json!(self.head_dim));
        m.insert("intermediate_size".into(), json!(self.intermediate_size));
        m.insert("vocab_size".into(), json!(self.vocab_size));
        m.insert("rope_theta".into(), json!(self.rope_theta));
        m.insert("rope_scaling_factor".into(), json!(self.rope_scale));
        if !self.rope_scaling_type.is_empty() {
            m.insert("rope_scaling_type".into(), json!(self.rope_scaling_type));
        }
        if self.rope_low_freq_factor > 0.0 {
            m.insert(
                "rope_scaling_low_freq_factor".into(),
                json!(self.rope_low_freq_factor),
            );
        }
        if self.rope_high_freq_factor > 0.0 {
            m.insert(
                "rope_scaling_high_freq_factor".into(),
                json!(self.rope_high_freq_factor),
            );
        }
        if self.rope_original_max_pos > 0 {
            m.insert(
                "rope_scaling_original_max_position_embeddings".into(),
                json!(self.rope_original_max_pos),
            );
        }
        m.insert("rms_norm_eps".into(), json!(self.rms_norm_eps));
        m.insert(
            "tie_word_embeddings".into(),
            json!(self.tie_word_embeddings),
        );
        if !self.per_layer_ffn.is_empty() {
            m.insert("per_layer_ffn".into(), json!(self.per_layer_ffn));
        }
        if !self.per_layer_attn.is_empty() {
            m.insert("per_layer_attn".into(), json!(self.per_layer_attn));
        }
        if !self.n_kv_heads_per_layer.is_empty() {
            m.insert(
                "n_kv_heads_per_layer".into(),
                json!(self.n_kv_heads_per_layer),
            );
        }
        if self.num_experts > 0 {
            m.insert("num_experts".into(), json!(self.num_experts));
            m.insert("num_experts_per_tok".into(), json!(self.num_experts_per_tok));
            m.insert("moe_intermediate_size".into(), json!(self.moe_intermediate_size));
            m.insert("norm_topk_prob".into(), json!(self.norm_topk_prob));
            if self.num_shared_experts > 0 {
                m.insert(
                    "num_shared_experts".into(),
                    json!(self.num_shared_experts),
                );
            }
        }
        if self.max_position_embeddings > 0 {
            m.insert(
                "max_position_embeddings".into(),
                json!(self.max_position_embeddings),
            );
        }
        if self.bos_token_id > 0 {
            m.insert("bos_token_id".into(), json!(self.bos_token_id));
        }
        if self.eos_token_id > 0 {
            m.insert("eos_token_id".into(), json!(self.eos_token_id));
        }
        if !self.eos_token_ids.is_empty() {
            m.insert("eos_token_ids".into(), json!(self.eos_token_ids));
        }
        // Gemma-4-specific fields — only emit when set so other archs'
        // headers stay tidy.
        if self.attention_scale > 0.0 {
            m.insert("attention_scale".into(), json!(self.attention_scale));
        }
        if self.n_embd_per_layer > 0 {
            m.insert("n_embd_per_layer".into(), json!(self.n_embd_per_layer));
        }
        if self.n_layer_kv_from_start > 0 {
            m.insert(
                "n_layer_kv_from_start".into(),
                json!(self.n_layer_kv_from_start),
            );
        }
        if self.logit_softcap > 0.0 {
            m.insert("logit_softcap".into(), json!(self.logit_softcap));
        }
        if self.head_dim_global > 0 {
            m.insert("head_dim_global".into(), json!(self.head_dim_global));
        }
        if self.head_dim_swa > 0 {
            m.insert("head_dim_swa".into(), json!(self.head_dim_swa));
        }
        if self.sliding_window > 0 {
            m.insert("sliding_window".into(), json!(self.sliding_window));
        }
        if self.sliding_window_pattern > 0 {
            m.insert(
                "sliding_window_pattern".into(),
                json!(self.sliding_window_pattern),
            );
        }
        if !self.swa_layers.is_empty() {
            m.insert("swa_layers".into(), json!(self.swa_layers));
        }
        if self.global_rope_partial_factor > 0.0 {
            m.insert(
                "global_rope_partial_factor".into(),
                json!(self.global_rope_partial_factor),
            );
        }
        if self.rope_local_theta > 0.0 {
            m.insert("rope_local_theta".into(), json!(self.rope_local_theta));
        }
        // Qwen3.5 / 3.6 hybrid-linear-attention fields — only emit when
        // set so non-hybrid archs' headers stay tidy.
        if !self.layer_types.is_empty() {
            m.insert("layer_types".into(), json!(self.layer_types));
        }
        if self.full_attention_interval > 0 {
            m.insert(
                "full_attention_interval".into(),
                json!(self.full_attention_interval),
            );
        }
        if self.linear_num_key_heads > 0 {
            m.insert(
                "linear_num_key_heads".into(),
                json!(self.linear_num_key_heads),
            );
            m.insert(
                "linear_num_value_heads".into(),
                json!(self.linear_num_value_heads),
            );
            m.insert("linear_key_head_dim".into(), json!(self.linear_key_head_dim));
            m.insert(
                "linear_value_head_dim".into(),
                json!(self.linear_value_head_dim),
            );
            m.insert(
                "linear_conv_kernel_dim".into(),
                json!(self.linear_conv_kernel_dim),
            );
        }
        if self.attn_output_gate {
            m.insert("attn_output_gate".into(), json!(self.attn_output_gate));
        }
        if self.partial_rotary_factor > 0.0 {
            m.insert(
                "partial_rotary_factor".into(),
                json!(self.partial_rotary_factor),
            );
        }
        if !self.mrope_section.is_empty() {
            m.insert("mrope_section".into(), json!(self.mrope_section));
            m.insert("mrope_interleaved".into(), json!(self.mrope_interleaved));
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SUPPORTED_HF_MODEL_TYPES` must list exactly what the dispatch match
    /// accepts — every advertised type resolves, and nothing else creeps in.
    #[test]
    fn supported_list_matches_dispatch() {
        for mt in SUPPORTED_HF_MODEL_TYPES {
            assert!(
                hf_mapper_for_model_type(mt).is_some(),
                "advertised model_type {mt:?} has no mapper"
            );
        }
        assert!(hf_mapper_for_model_type("not-a-real-arch").is_none());
    }
}
