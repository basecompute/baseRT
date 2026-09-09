//! Per-architecture tensor-name normalization + config extraction.
//!
//! Maps source tensor names (GGUF `blk.0.attn_q.weight`, MLX
//! `model.layers.0.self_attn.q_proj.weight`) to the canonical `.base`
//! naming (HF convention). Also extracts arch-specific config fields
//! from source metadata so the `.base` header's `config` is populated
//! consistently regardless of source format.

pub mod bert;
pub mod gemma;
pub mod glm;
pub mod gpt_oss;
pub mod llama;
pub mod muse_glimmer;
pub mod nemotron;
pub mod qwen;
pub mod tokenizer;
pub mod whisper;

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
        "nemotron_h" | "nemotron_h_moe" => Some(&nemotron::NemotronHMapper),
        "nomic-bert" => Some(&bert::NomicBertMapper),
        // GLM 5.2 — DeepSeek-V3.2-style MLA + sparse-attention MoE.
        "glm-dsa" => Some(&glm::GlmDsaMapper),
        // Muse Glimmer. `general.architecture` in the llama.cpp-produced
        // GGUF is the HYPHENATED "muse-glimmer" (llama.cpp's arch registry
        // spells multi-word archs with hyphens: "nomic-bert",
        // "deepseek2"...), while the HF `model_type` — and therefore the
        // canonical `.base` `arch` field — is the UNDERSCORED
        // "muse_glimmer". Both spellings resolve here so a hand-edited or
        // future exporter that emits the underscore form still converts.
        "muse-glimmer" | "muse_glimmer" => Some(&muse_glimmer::MuseGlimmerGgufMapper),
        _ => None,
    }
}

/// HF config.json model_type → mapper. HF tensor names already follow
/// canonical convention, so the mapper only needs to extract ArchConfig
/// from config.json — no tensor renaming.
/// An element-wise reparameterization undone at convert time.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ValueTransform {
    /// `x -> -exp(x)`. Mamba-2 checkpoints store the state-transition
    /// matrix as `A_log` and materialize `A = -exp(A_log)` in the model
    /// code; the value the scan kernel wants is `A`, which must be
    /// negative for the recurrence to decay.
    NegExp,
}

impl ValueTransform {
    pub fn apply(self, values: &mut [f32]) {
        match self {
            ValueTransform::NegExp => {
                for v in values.iter_mut() {
                    *v = -v.exp();
                }
            }
        }
    }
}

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

    /// Element-wise transform to apply to a tensor's values on the way
    /// from HF to `.base`, or `None` to copy them through.
    ///
    /// Some architectures store a *reparameterized* weight that the
    /// reference implementation undoes at load or at run time, while
    /// the GGUF conversion bakes the undo in. Baking it at convert time
    /// keeps the runtime kernel simple and keeps bundles from the two
    /// sources interchangeable.
    fn value_transform(&self, _canonical: &str) -> Option<ValueTransform> {
        None
    }

    /// Write tensor `shape` fastest-varying dim first (GGUF's `ne`
    /// order, `[in, out]`) instead of HF's C order (`[out, in]`).
    ///
    /// The two describe the *same bytes* — an HF `[out, in]` matrix is
    /// stored with `in` contiguous, exactly like a GGUF `[in, out]` one
    /// — so this changes how the header reports a tensor, not the
    /// buffer behind it. Set it on architectures whose GGUF-converted
    /// bundles are already in circulation, so a bundle built from either
    /// source describes itself identically. Default: keep HF order.
    fn shape_fastest_first(&self) -> bool {
        false
    }
    /// RMS-normalize each ROW of a 2-D tensor at HF→.base conversion
    /// time, returning the epsilon to use, or None to leave it alone.
    ///
    /// Muse Glimmer applies a weightless RMSNorm to the token embedding
    /// immediately after lookup (`MuseGlimmerTextNormedEmbedding`). Since
    /// a lookup returns exactly one row and the norm mixes nothing across
    /// rows, it is mathematically identical to normalizing every row of
    /// the embedding matrix up front — so the runtime needs no extra op.
    /// (The reference keeps them separate only because its DFlash drafter
    /// needs the un-normed embedding.) Safe only while the embedding is
    /// NOT tied to `lm_head`, which holds for Muse Glimmer
    /// (`tie_word_embeddings=false`). Default: no normalization.
    fn row_rms_normalize(&self, _canonical: &str, _cfg: &ArchConfig) -> Option<f32> {
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
        // Phi-3-mini (4k, standard RoPE) is Llama-shaped and reuses the Llama
        // mapper: the SplittingProvider (base-convert) slices its fused
        // self_attn.qkv_proj / mlp.gate_up_proj into the canonical split names
        // the mapper consumes, HD=96 is fine (regression-tested), and the
        // generation_config eos merge makes it stop cleanly on <|end|> (32007).
        // The chat-path flood that used to gate it (same class as SmolLM2) is
        // resolved by two landed tokenizer fixes — the added-token lstrip/rstrip
        // handling in split_special_tokens (Phi-3's `<|end|>` rstrip absorbs the
        // trailing newline; tokenizer.cpp) and the GPT2* add_bos default
        // (tokenizer_defaults.h). REVALIDATED on microsoft/Phi-3-mini-4k-instruct
        // (converted Q8): chat output is coherent across probes, no flood.
        // Phi-3.5 stays OUT — it needs LongRoPE, which the engine (linear scaling
        // only) does not implement.
        "llama" | "mistral" | "phi3" => Some(&llama::LlamaHfMapper),
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
        // Muse Glimmer: dense SWA/global decoder with a perception (ViT)
        // tower. The multimodal wrapper is `muse_glimmer`
        // (`MuseGlimmerForConditionalGeneration`) with the text tower under
        // `text_config.model_type = muse_glimmer_text`; both resolve here so
        // a text-only checkpoint and the multimodal wrapper convert alike.
        "muse_glimmer" | "muse_glimmer_text" => Some(&muse_glimmer::MuseGlimmerHfMapper),
        "gemma" | "gemma2" | "gemma3" | "gemma3_text" => Some(&gemma::Gemma3HfMapper),
        // gemma3n is a distinct arch (AltUp/Laurel/per-layer-FFN); the
        // existing local fixture historically named "gemma-4-e2b" was
        // actually google/gemma-3n-E2B-it. The canonical Gemma 4 lives
        // under google/gemma-4-{E2B,E4B}-it and uses model_type=gemma4
        // with text_config.model_type=gemma4_text.
        // gemma4_unified (gemma-4-12B-it): the encoder-free multimodal
        // variant — a standard gemma4 text stack under
        // `model.language_model.*` plus ~10 small modality-projection
        // tensors (embed_vision/embed_audio/vision_embedder) that the
        // text conversion skips like any other non-text tower. Config is
        // gemma4-shaped (uniform head_dim, rope_parameters, layer_types).
        "gemma4" | "gemma4_text" | "gemma4_unified" => Some(&gemma::Gemma4HfMapper),
        // Nemotron-H hybrid (Mamba-2 + attention + MoE). The HF
        // `model_type` is `nemotron_h` for both the dense and MoE
        // builds — the block schedule comes from
        // `hybrid_override_pattern`, and the MoE keys are simply absent
        // on a dense checkpoint — so one mapper covers both. Canonical
        // arch stays `nemotron_h_moe` to match the GGUF path.
        "nemotron_h" | "nemotron_h_moe" => Some(&nemotron::NemotronHHfMapper),
        // Whisper encoder-decoder speech models (openai/whisper-*). HF
        // safetensors only — whisper GGML files are not GGUF, so the GGUF
        // dispatch table stays untouched. Tensor renaming is
        // whisper-specific (`whisper::map_hf_tensor_name`); the convert
        // path emits everything as f16 (the engine's fused whisper
        // kernels are f16-only in v1).
        "whisper" => Some(&whisper::WhisperHfMapper),
        // GLM 5.2 — DeepSeek-V3.2-style MLA + DSA MoE. HF/MLX
        // checkpoints declare `model_type: glm_moe_dsa`.
        "glm_moe_dsa" => Some(&glm::GlmDsaMapper),
        // OpenAI gpt-oss (MXFP4 MoE, attention sinks, YaRN, alternating SWA).
        "gpt_oss" => Some(&gpt_oss::GptOssHfMapper),
        _ => None,
    }
}

/// Every HF `model_type` value [`hf_mapper_for_model_type`] accepts. Kept in
/// lockstep with that match so a pre-flight support check (and its error
/// message) has a single source of truth for what convert-on-pull supports.
pub const SUPPORTED_HF_MODEL_TYPES: &[&str] = &[
    "llama",
    "mistral",
    "phi3",
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
    "gemma4_unified",
    "muse_glimmer",
    "muse_glimmer_text",
    "whisper",
    "glm_moe_dsa",
    "gpt_oss",
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

    /// RoPE row-permutation head count for a canonical tensor on the GGUF
    /// path, or None when the tensor's rows are already in the layout this
    /// runtime's rope kernel expects. The INVERSE of
    /// [`HfMapper::rope_permute_heads`]: that one converts HF split-half
    /// rows INTO the interleaved-pair layout, this one converts GGUF
    /// interleaved-pair rows BACK to split-half.
    ///
    /// Background. There are two rotary row layouts in the wild:
    ///
    ///   - "interleaved pair" (Meta original / GPT-J): rope rotates the
    ///     element pairs `(2i, 2i+1)`. Runtime kernel: `rope_f16`.
    ///   - "split half" (NeoX / HF `rotate_half`): rope pairs element `i`
    ///     with `i + head_dim/2`. Runtime kernel: `rope_neox_f16`.
    ///
    /// `convert_hf_to_gguf.py::LlamaModel.permute` rewrites `q_proj` /
    /// `k_proj` from split-half into interleaved-pair when it exports an
    /// HF checkpoint, per head:
    ///
    /// ```text
    ///   gguf_row[h*HD + 2j + k] = hf_row[h*HD + k*HD/2 + j]
    /// ```
    ///
    /// so the inverse this hook drives is
    ///
    /// ```text
    ///   hf_row[h*HD + k*HD/2 + j] = gguf_row[h*HD + 2j + k]
    /// ```
    ///
    /// Whether a GGUF needs the inverse depends on which kernel the arch
    /// runs, NOT on the source format:
    ///
    ///   - llama/mistral run `rope_f16` (interleaved). llama.cpp permutes
    ///     on export, the HF path permutes at convert time, so both
    ///     sources agree and the GGUF needs NOTHING here. Returning
    ///     `Some(..)` for llama would actively break it.
    ///   - gemma/qwen/nomic-bert run `rope_neox_f16`, and llama.cpp does
    ///     NOT permute those archs on export — again nothing to do.
    ///   - Muse Glimmer runs `rope_neox_f16` (split-half) but its
    ///     llama.cpp exporter DOES apply the Llama permute. That is the
    ///     one combination that needs undoing, and it is why this hook
    ///     exists.
    ///
    /// The permutation moves whole ROWS (output features), never elements
    /// within a row, so it can be applied to packed k-quant blocks by
    /// reordering row-sized byte runs — no dequantization, fully lossless.
    /// The caller must still check that each row is a whole number of
    /// blocks (`in_features % block_elems == 0`).
    fn rope_unpermute_heads(&self, _canonical: &str, _cfg: &ArchConfig) -> Option<u32> {
        None
    }
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

    // ── gpt-oss fields (zero/false for other archs) ──────────────────
    /// YaRN correction-range betas (HF `rope_scaling.beta_fast` /
    /// `beta_slow`). Only meaningful when `rope_scaling_type == "yarn"`.
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    /// YaRN `truncate` (HF `rope_scaling.truncate`, default true): floor/ceil
    /// the correction range. gpt-oss ships `false` (continuous ramp bounds).
    pub rope_yarn_truncate: bool,
    /// Clamped-SwiGLU limit (`gate <= limit`, `|up| <= limit`) and the
    /// swish alpha (`gate * sigmoid(alpha * gate)`). 0 = plain SwiGLU.
    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
    /// Learned per-head attention sinks (`self_attn.sinks`).
    pub attention_sinks: bool,
    /// q/k/v/o projections carry biases.
    pub attention_bias: bool,

    // ── Nemotron-H / Mamba-2 SSM fields (zero for non-SSM archs) ─────
    // Nemotron-H interleaves Mamba-2 SSM blocks, GQA attention blocks
    // and (MoE) FFN blocks — the schedule rides in `layer_types`
    // ("mamba" | "attention" | "moe" | "mlp").
    /// Mamba-2 SSM state size per head (`d_state`). 0 = no SSM.
    pub ssm_state_size: u32,
    /// Causal depthwise conv kernel width in the SSM mixer (`d_conv`).
    pub ssm_conv_kernel: u32,
    /// Number of B/C groups (`n_groups`).
    pub ssm_num_groups: u32,
    /// SSM inner width (`d_inner` = heads × head dim).
    pub ssm_inner_size: u32,
    /// Number of SSM heads (GGUF stores this in `ssm.time_step_rank`
    /// for Mamba-2 checkpoints).
    pub ssm_num_heads: u32,

    // ── MoE routing extensions (DeepSeek-style routers) ──────────────
    /// Routed-expert scaling factor applied after top-k
    /// renormalization (Nemotron 3 Nano: 2.5). 0 = none.
    pub expert_weights_scale: f32,
    // ── Whisper encoder-decoder fields (zero/empty for other archs) ──
    // The decoder half reuses the standard fields above (hidden_size /
    // num_hidden_layers / num_attention_heads / intermediate_size /
    // max_position_embeddings); the encoder half is described here.
    // ── Muse Glimmer fields (zero/empty for other archs) ─────────────
    /// Multiplier applied to Q AFTER the scaleless (weightless) QK-norm,
    /// on top of the standard `1/sqrt(head_dim)` attention scaling
    /// (`qk_scale_factor`, 3.87 on the released checkpoint). 0 = not a
    /// Muse-Glimmer-style model. Distinct from `attention_scale`, which
    /// REPLACES the `1/sqrt(head_dim)` term rather than scaling it.
    pub qk_scale_factor: f32,
    /// Scale applied to the logits BEFORE the final tanh softcap
    /// (`output_multiplier`; `1/sqrt(hidden_size/256)` on the released
    /// checkpoint). 0 = no multiplier.
    pub output_multiplier: f32,
    /// Epsilon for the post-attention / post-FFN norms, which differs
    /// from `rms_norm_eps` on Muse Glimmer (1e-8 vs 1e-5). 0 = reuse
    /// `rms_norm_eps` for every norm.
    pub post_norm_eps: f32,
    /// Epsilon for a weightless RMSNorm the RUNTIME must apply to the token
    /// embedding after lookup. 0 = the norm is already folded into the
    /// embedding rows (the HF path does this via `row_rms_normalize`, which is
    /// exact) or the arch has no such norm.
    ///
    /// The GGUF path cannot fold it: the rows arrive as packed k-quant
    /// super-blocks, and folding would require dequantizing — the very thing
    /// passthrough exists to avoid. So a GGUF-sourced Muse Glimmer bundle sets
    /// this and pays for one extra kernel per prefill instead.
    pub embed_norm_eps: f32,

    /// Per-layer NoPE mask (true = layer applies NO rotary embedding).
    /// Derived from `layer_rope_theta[i] == 0`. Empty = every layer
    /// gets RoPE. Length = num_hidden_layers.
    pub nope_layers: Vec<bool>,

    /// Encoder transformer depth (`encoder_layers`). 0 = not an
    /// encoder-decoder model.
    pub encoder_layers: u32,
    /// Encoder attention head count (`encoder_attention_heads`).
    pub encoder_attention_heads: u32,
    /// Encoder FFN width (`encoder_ffn_dim`).
    pub encoder_ffn_dim: u32,
    /// Decoder FFN width (`decoder_ffn_dim`). Mirrors
    /// `intermediate_size`; emitted under its HF name so the runtime's
    /// whisper config parser reads the contract key directly.
    pub decoder_ffn_dim: u32,
    /// Mel filterbank bin count (`num_mel_bins`, 80 or 128). The engine
    /// computes the Slaney filterbank from this — it is not stored.
    pub num_mel_bins: u32,
    /// Encoder positional-embedding length (`max_source_positions`, 1500).
    pub max_source_positions: u32,
    /// Decoder positional-embedding length (`max_target_positions`, 448).
    pub max_target_positions: u32,

    // ── GLM-DSA / DeepSeek-V3.2-style MLA + sparse-attention fields ───
    // (all zero for other archs). GLM 5.2 uses Multi-head Latent
    // Attention (compressed q/kv latents + decoupled RoPE) plus a
    // DeepSeek Sparse Attention "lightning indexer", with a sigmoid-gated
    // MoE (bias-corrected top-k, routed weight scaling, shared expert)
    // and the first `first_k_dense_replace` layers dense.
    /// MLA query compression rank (`attn_q_a` output width). 0 = not MLA.
    pub q_lora_rank: u32,
    /// MLA key/value compression rank (`attn_kv_a_mqa` kv part). 0 = not MLA.
    pub kv_lora_rank: u32,
    /// Per-head non-positional Q/K dim (the part that attends in latent
    /// space via k_b). GLM 5.2 = 192.
    pub qk_nope_head_dim: u32,
    /// Per-head decoupled-RoPE Q/K dim (the only rotated part). GLM 5.2 = 64.
    pub qk_rope_head_dim: u32,
    /// Per-head value dim after v_b up-projection. GLM 5.2 = 256.
    pub v_head_dim: u32,
    /// Routed-expert output scaling (DeepSeek `routed_scaling_factor`).
    /// GLM 5.2 = 2.5. 0 = no scaling.
    pub routed_scaling_factor: f32,
    /// Expert gating function: 0 = softmax (default), 1 = sigmoid
    /// (GLM/DeepSeek; GGUF `expert_gating_func = 2`).
    pub expert_gating: u32,
    /// First N layers use a dense SwiGLU FFN instead of MoE
    /// (`leading_dense_block_count` / HF `first_k_dense_replace`).
    /// GLM 5.2 = 3. 0 = all MoE layers.
    pub first_k_dense_replace: u32,
    /// Multi-Token-Prediction (nextn) head layer count. Tensors are
    /// dropped at convert time; kept for the header record. GLM 5.2 = 1.
    pub nextn_predict_layers: u32,
    /// DSA lightning-indexer head count. GLM 5.2 = 32. 0 = no indexer.
    pub indexer_head_count: u32,
    /// DSA indexer per-head key dim. GLM 5.2 = 128.
    pub indexer_key_length: u32,
    /// DSA indexer top-k keys selected per query. GLM 5.2 = 2048.
    pub indexer_top_k: u32,
    /// Per-layer indexer kind: "full" (layer computes its own top-k
    /// selection and carries indexer weights) or "shared" (layer reuses
    /// the most recent full layer's selection; NO indexer weights).
    /// GLM 5.2: 21 full / 57 shared. Empty = every layer is full (the
    /// GGUF metadata doesn't carry the pattern). HF `indexer_types`.
    pub indexer_layer_types: Vec<String>,
    /// Selection-reuse period for shared indexer layers (HF
    /// `index_topk_freq`). GLM 5.2 = 4. 0 = unset.
    pub index_topk_freq: u32,
    /// Offset into the reuse period (HF `index_skip_topk_offset`).
    /// GLM 5.2 = 3. Only meaningful when `index_topk_freq > 0`.
    pub index_skip_topk_offset: u32,
    /// Indexer RoPE layout: true = interleaved/traditional (GPT-J
    /// adjacent-pair — what mlx-lm applies via `traditional=True`),
    /// false = NeoX half-split. HF `indexer_rope_interleave`.
    /// GLM 5.2 = true. NOTE: llama.cpp's deepseek32 reference uses NeoX
    /// for its indexer; GLM's config + the MLX implementation say
    /// interleaved. Trust this flag, not the deepseek32 source.
    pub indexer_rope_interleave: bool,
    /// Whether MTP iterations reuse the same top-k selection (HF
    /// `index_share_for_mtp_iteration`). Recorded for the header; only
    /// relevant once MTP lands.
    pub index_share_for_mtp_iteration: bool,
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
            m.insert(
                "num_experts_per_tok".into(),
                json!(self.num_experts_per_tok),
            );
            m.insert(
                "moe_intermediate_size".into(),
                json!(self.moe_intermediate_size),
            );
            m.insert("norm_topk_prob".into(), json!(self.norm_topk_prob));
            if self.num_shared_experts > 0 {
                m.insert("num_shared_experts".into(), json!(self.num_shared_experts));
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
            m.insert(
                "linear_key_head_dim".into(),
                json!(self.linear_key_head_dim),
            );
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
        // Muse Glimmer scales / epsilons / NoPE schedule — emitted only
        // when set so other archs' headers stay unchanged.
        if self.qk_scale_factor > 0.0 {
            m.insert("qk_scale_factor".into(), json!(self.qk_scale_factor));
        }
        if self.output_multiplier > 0.0 {
            m.insert("output_multiplier".into(), json!(self.output_multiplier));
        }
        if self.post_norm_eps > 0.0 {
            m.insert("post_norm_eps".into(), json!(self.post_norm_eps));
        }
        if !self.nope_layers.is_empty() {
            m.insert("nope_layers".into(), json!(self.nope_layers));
        }
        if self.embed_norm_eps > 0.0 {
            m.insert("embed_norm_eps".into(), json!(self.embed_norm_eps));
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
        // gpt-oss fields — emitted only when set.
        if self.rope_yarn_beta_fast > 0.0 {
            m.insert(
                "rope_yarn_beta_fast".into(),
                json!(self.rope_yarn_beta_fast),
            );
            m.insert(
                "rope_yarn_beta_slow".into(),
                json!(self.rope_yarn_beta_slow),
            );
            m.insert("rope_yarn_truncate".into(), json!(self.rope_yarn_truncate));
        }
        if self.swiglu_limit > 0.0 {
            m.insert("swiglu_limit".into(), json!(self.swiglu_limit));
            m.insert("swiglu_alpha".into(), json!(self.swiglu_alpha));
        }
        if self.attention_sinks {
            m.insert("attention_sinks".into(), json!(true));
        }
        if self.attention_bias {
            m.insert("attention_bias".into(), json!(true));
        }
        // Nemotron-H / Mamba-2 SSM fields — only emit when set.
        if self.ssm_inner_size > 0 {
            m.insert("ssm_state_size".into(), json!(self.ssm_state_size));
            m.insert("ssm_conv_kernel".into(), json!(self.ssm_conv_kernel));
            m.insert("ssm_num_groups".into(), json!(self.ssm_num_groups));
            m.insert("ssm_inner_size".into(), json!(self.ssm_inner_size));
            m.insert("ssm_num_heads".into(), json!(self.ssm_num_heads));
        }
        if self.expert_gating > 0 {
            m.insert("expert_gating".into(), json!(self.expert_gating));
        }
        if self.expert_weights_scale > 0.0 {
            m.insert(
                "expert_weights_scale".into(),
                json!(self.expert_weights_scale),
            );
        }
        // Whisper encoder-decoder fields — only emit when the encoder
        // half is populated so decoder-only archs' headers stay tidy.
        if self.encoder_layers > 0 {
            m.insert("model_type".into(), json!("whisper"));
            m.insert("encoder_layers".into(), json!(self.encoder_layers));
            m.insert(
                "encoder_attention_heads".into(),
                json!(self.encoder_attention_heads),
            );
            m.insert("encoder_ffn_dim".into(), json!(self.encoder_ffn_dim));
            m.insert("decoder_ffn_dim".into(), json!(self.decoder_ffn_dim));
            // Whisper's encoder width equals the decoder width; emit the
            // HF `d_model` key the runtime's whisper config parser reads.
            m.insert("d_model".into(), json!(self.hidden_size));
            m.insert("num_mel_bins".into(), json!(self.num_mel_bins));
            m.insert(
                "max_source_positions".into(),
                json!(self.max_source_positions),
            );
            m.insert(
                "max_target_positions".into(),
                json!(self.max_target_positions),
            );
            // Whisper uses LayerNorm, not RMSNorm; the contract key is
            // `norm_eps` (rms_norm_eps above carries the same value for
            // struct-level uniformity).
            m.insert("norm_eps".into(), json!(self.rms_norm_eps));
        }
        // GLM-DSA / MLA + sparse-attention fields — only emit when set so
        // other archs' headers stay tidy. The runtime reads these back in
        // BaseWeightStore::extract_config.
        if self.q_lora_rank > 0 {
            m.insert("q_lora_rank".into(), json!(self.q_lora_rank));
        }
        if self.kv_lora_rank > 0 {
            m.insert("kv_lora_rank".into(), json!(self.kv_lora_rank));
        }
        if self.qk_nope_head_dim > 0 {
            m.insert("qk_nope_head_dim".into(), json!(self.qk_nope_head_dim));
        }
        if self.qk_rope_head_dim > 0 {
            m.insert("qk_rope_head_dim".into(), json!(self.qk_rope_head_dim));
        }
        if self.v_head_dim > 0 {
            m.insert("v_head_dim".into(), json!(self.v_head_dim));
        }
        if self.routed_scaling_factor > 0.0 {
            m.insert(
                "routed_scaling_factor".into(),
                json!(self.routed_scaling_factor),
            );
        }
        // expert_gating is meaningful even when 0 (softmax) for MoE models,
        // but we only emit non-default sigmoid to keep other headers tidy.
        if self.expert_gating > 0 {
            m.insert("expert_gating".into(), json!(self.expert_gating));
        }
        if self.first_k_dense_replace > 0 {
            m.insert(
                "first_k_dense_replace".into(),
                json!(self.first_k_dense_replace),
            );
        }
        if self.nextn_predict_layers > 0 {
            m.insert(
                "nextn_predict_layers".into(),
                json!(self.nextn_predict_layers),
            );
        }
        if self.indexer_head_count > 0 {
            m.insert("indexer_head_count".into(), json!(self.indexer_head_count));
            m.insert("indexer_key_length".into(), json!(self.indexer_key_length));
            m.insert("indexer_top_k".into(), json!(self.indexer_top_k));
        }
        // DSA full/shared layer pattern — present only on sources that
        // declare it (the MLX/HF config.json; GGUF metadata doesn't).
        if !self.indexer_layer_types.is_empty() {
            m.insert(
                "indexer_layer_types".into(),
                json!(self.indexer_layer_types),
            );
            m.insert("index_topk_freq".into(), json!(self.index_topk_freq));
            m.insert(
                "index_skip_topk_offset".into(),
                json!(self.index_skip_topk_offset),
            );
            m.insert(
                "indexer_rope_interleave".into(),
                json!(self.indexer_rope_interleave),
            );
            m.insert(
                "index_share_for_mtp_iteration".into(),
                json!(self.index_share_for_mtp_iteration),
            );
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
