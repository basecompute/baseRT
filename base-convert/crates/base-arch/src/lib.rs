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
}

pub fn hf_mapper_for_model_type(model_type: &str) -> Option<&'static dyn HfMapper> {
    match model_type {
        "llama" => Some(&llama::LlamaHfMapper),
        "qwen2" | "qwen3" => Some(&qwen::QwenHfMapper),
        "qwen2_moe" | "qwen3_moe" => Some(&qwen::QwenMoeHfMapper),
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
        m
    }
}
