//! Raw FFI bindings for the BaseRT C API.
//!
//! These are unsafe, low-level bindings. Prefer the `baseRT` crate for safe usage.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_float, c_int, c_void};

/// Opaque model handle.
pub type baseRT_model_t = *mut c_void;

/// Model configuration extracted from weight file metadata.
///
/// This mirrors the FULL `BaseRTModelConfig` in include/baseRT/types.h. The
/// struct is returned BY VALUE from `baseRT_get_config`, so it MUST match the C
/// layout exactly — a truncated copy makes the C side write past the Rust
/// allocation (memory corruption). Earlier revisions of this binding dropped
/// `sliding_window` and every field after the encoder block; do not reintroduce
/// that. New fields in C are appended at the end (`#[repr(C)]` keeps offsets).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BaseRTModelConfig {
    // Decoder (or decoder-only LLM) parameters
    pub dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub ffn_dim: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub norm_eps: c_float,
    pub rope_theta: c_float,
    pub sliding_window_pattern: u32,
    pub sliding_window: u32,
    pub rope_local_theta: c_float,
    pub architecture: [c_char; 32],

    // Encoder parameters (0 = decoder-only model)
    pub enc_n_layers: u32,
    pub enc_n_heads: u32,
    pub enc_dim: u32,
    pub enc_ffn_dim: u32,
    pub n_mels: u32,
    pub enc_max_seq_len: u32,

    // Gemma 4-specific fields
    pub n_embd_per_layer: u32,
    pub n_layer_kv_from_start: u32,
    pub logit_softcap: c_float,
    pub attention_scale: c_float,
    pub head_dim_swa: u32,
    pub head_dim_global: u32,
    pub global_rope_partial_factor: c_float,
    pub swa_layers: [u8; 64],
    pub ffn_dims: [u32; 128],
    pub n_kv_heads_per_layer: [u32; 128],

    // Qwen3.5 / 3.6 hybrid linear-attention (Gated DeltaNet)
    pub attn_output_gate: u8,
    pub _qwen35_pad: [u8; 3],
    pub partial_rotary_factor: c_float,
    pub full_attention_interval: u32,
    pub linear_attn_layers: [u8; 64],
    pub gdn_num_k_heads: u32,
    pub gdn_num_v_heads: u32,
    pub gdn_key_head_dim: u32,
    pub gdn_value_head_dim: u32,
    pub gdn_conv_kernel: u32,

    // Nemotron-H hybrid Mamba-2 SSM
    pub ssm_state_size: u32,
    pub ssm_conv_kernel: u32,
    pub ssm_num_groups: u32,
    pub ssm_inner_size: u32,
    pub ssm_num_heads: u32,

    // Mixture-of-Experts (0 = dense)
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub n_experts_shared: u32,
    pub expert_ffn_dim: u32,
    pub expert_gating: u8,
    pub norm_topk_prob: u8,
    pub _moe_pad: [u8; 2],
    pub expert_weights_scale: c_float,

    // Vision tower (all zero = none)
    pub vision_n_layers: u32,
    pub vision_dim: u32,
    pub vision_n_heads: u32,
    pub vision_head_dim: u32,
    pub vision_ffn_dim: u32,
    pub vision_patch_size: u32,
    pub vision_image_size: u32,
    pub vision_pooling_kernel: u32,
    pub vision_soft_tokens: u32,
    pub vision_norm_eps: c_float,
    pub vision_rope_theta: c_float,
    pub vision_pos_embed_size: u32,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,

    // Vision-tower family selector + Qwen3-VL-style extras
    pub vision_arch: u32,
    pub vision_spatial_merge: u32,
    pub vision_temporal_patch: u32,
    pub vision_out_dim: u32,

    // Audio tower (all zero = none)
    pub audio_n_layers: u32,
    pub audio_dim: u32,
    pub audio_n_heads: u32,
    pub audio_head_dim: u32,
    pub audio_ffn_dim: u32,
    pub audio_output_proj_dim: u32,
    pub audio_chunk_size: u32,
    pub audio_left_context: u32,
    pub audio_conv_kernel: u32,
    pub audio_soft_tokens: u32,
    pub audio_logit_softcap: c_float,
    pub audio_norm_eps: c_float,
    pub audio_gradient_clip: c_float,
    pub audio_residual_weight: c_float,
    pub audio_ms_per_token: c_float,
    pub audio_sscp_channels: [u32; 2],
    pub audio_token_id: u32,
    pub boa_token_id: u32,
    pub eoa_token_id: u32,
    pub mrope_section: [u32; 3],
    pub mrope_interleaved: u8,
    pub _mrope_pad: [u8; 3],
    pub rope_scaling_factor: c_float,
    pub rope_low_freq_factor: c_float,
    pub rope_high_freq_factor: c_float,
    pub rope_orig_max_pos: u32,
    pub rope_scaling_type: u32,
    // Muse Glimmer (0 / empty = not applicable)
    pub qk_scale_factor: c_float,
    pub output_multiplier: c_float,
    pub post_norm_eps: c_float,
    pub nope_layers: [u8; 64],
    pub embed_norm_eps: c_float,
    pub vision_window_layers: [u8; 64],
    pub vision_window_size: u32,
    pub vision_pos_embed_h: u32,
    pub vision_pos_embed_w: u32,
    pub vision_adapter_dim: u32,
    pub video_token_id: u32,
    // GLM 5.2 / glm-dsa
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub routed_scaling_factor: c_float,
    pub first_k_dense_replace: u32,
    pub nextn_predict_layers: u32,
    pub indexer_head_count: u32,
    pub indexer_key_length: u32,
    pub indexer_top_k: u32,
    // gpt-oss
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
    pub attention_sinks: u8,
    pub attention_bias: u8,
    pub rope_yarn_truncate: u8,
    pub _gptoss_pad: u8,
}

/// Transcription result statistics.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BaseRTTranscribeStats {
    pub n_tokens: c_int,
    pub audio_ms: c_float,
    pub encode_ms: c_float,
    pub decode_ms: c_float,
    pub total_ms: c_float,
}

/// One segment of the last transcription (verbose_json surface).
/// `text` is borrowed from the engine — valid until the next transcription
/// or model free.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BaseRTTranscribeSegment {
    pub start_ms: c_int,
    pub end_ms: c_int,
    pub text: *const c_char,
    pub avg_logprob: c_float,
    pub no_speech_prob: c_float,
    pub compression_ratio: c_float,
    pub temperature: c_float,
}

impl Default for BaseRTTranscribeSegment {
    fn default() -> Self {
        Self {
            start_ms: 0,
            end_ms: 0,
            text: std::ptr::null(),
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            compression_ratio: 0.0,
            temperature: 0.0,
        }
    }
}

/// Sampling configuration for text generation.
///
/// Extended in baseRT 0.2 with OpenAI-compat presence / frequency penalties,
/// a deterministic-sample `seed`, and a per-token `logit_bias` map. New
/// fields are appended; older callers using `Default::default()` keep the
/// classic five values plus zeroed extensions (disabled).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BaseRTSamplingConfig {
    pub temperature: c_float,
    pub top_k: c_int,
    pub top_p: c_float,
    pub min_p: c_float,
    pub repeat_penalty: c_float,
    pub presence_penalty: c_float,
    pub frequency_penalty: c_float,
    pub seed: u32,
    pub n_logit_bias: i32,
    pub logit_bias_tokens: *const i32,
    pub logit_bias_values: *const c_float,
}

impl Default for BaseRTSamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            n_logit_bias: 0,
            logit_bias_tokens: std::ptr::null(),
            logit_bias_values: std::ptr::null(),
        }
    }
}

/// Generation result statistics.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BaseRTGenerationStats {
    pub prompt_tokens: c_int,
    pub generated_tokens: c_int,
    pub prefill_time_ms: c_float,
    pub decode_time_ms: c_float,
    pub prefill_tokens_per_sec: c_float,
    pub decode_tokens_per_sec: c_float,
}

/// Opaque per-sequence handle (independent KV state on the model's shared
/// paged-KV pool). Mirrors `baseRT_sequence_t` in baseRT.h.
pub type baseRT_sequence_t = *mut c_void;

/// Result of a prefix-cache lookup (`baseRT_prefix_match`). `blocks` points
/// into engine-owned storage valid until `baseRT_prefix_unlock(handle)` /
/// `baseRT_prefix_release(handle)`. `handle == 0` means miss / cache
/// disabled — nothing to unlock.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BaseRTPrefixMatch {
    pub matched_tokens: c_int,
    pub n_blocks: c_int,
    pub blocks: *const c_int,
    pub handle: u64,
}

/// Callback for streaming token output. Return `false` to stop generation.
pub type baseRT_token_callback = Option<
    unsafe extern "C" fn(token_id: u32, text: *const c_char, user_data: *mut c_void) -> bool,
>;

/// Callback for streaming transcription segments. Return `false` to stop transcription.
pub type baseRT_segment_callback = Option<
    unsafe extern "C" fn(
        start_ms: c_int,
        end_ms: c_int,
        text: *const c_char,
        user_data: *mut c_void,
    ) -> bool,
>;

extern "C" {
    // === Model lifecycle ===

    pub fn baseRT_load_model(
        model_path: *const c_char,
        kernel_library_path: *const c_char,
        max_context: c_int,
    ) -> baseRT_model_t;

    pub fn baseRT_free_model(model: baseRT_model_t);
    /// Tokenizer-only load: metadata + vocab, no GPU tensors. Only the
    /// tokenizer surface is valid on the returned handle.
    pub fn baseRT_load_tokenizer_only(model_path: *const c_char) -> baseRT_model_t;

    // === Model info ===

    pub fn baseRT_get_config(model: baseRT_model_t) -> BaseRTModelConfig;
    pub fn baseRT_model_config_sizeof() -> usize;
    pub fn baseRT_model_memory(model: baseRT_model_t) -> usize;
    /// Device memory budget in bytes (Metal: recommended working set; CUDA:
    /// total device memory). 0 when no supported device is present.
    pub fn baseRT_device_memory_budget() -> usize;
    /// A `max_context` sized for this device and this bundle. Metadata-only,
    /// so it is safe to call before loading. 0 = could not size.
    pub fn baseRT_suggest_max_context(
        model_path: *const c_char,
        max_batch: c_int,
        kv_bits: c_int,
        paged_kv: c_int,
    ) -> c_int;
    /// As above for a set of models that will be resident at the same time:
    /// weights and KV pools add up, and the trained-window cap is the
    /// shortest among them.
    pub fn baseRT_suggest_max_context_multi(
        model_paths: *const *const c_char,
        n_models: c_int,
        max_batch: c_int,
        kv_bits: c_int,
        paged_kv: c_int,
    ) -> c_int;
    /// As above, plus the speculators loading beside those models:
    /// `speculator_embedded[i]` non-zero is the target bundle's embedded head
    /// (its KV pool only); zero (or a null array) a sidecar / draft model
    /// (weights and KV pool), even when also served. `speculator_target[i]`
    /// >= 0 is the index of the model a sidecar's baseRT_load_drafter reopens
    /// for its embedding / lm_head (charged where the backend copies wrapped
    /// weights); -1 (or a null array) for none. Speculators never cap the
    /// window.
    pub fn baseRT_suggest_max_context_spec(
        model_paths: *const *const c_char,
        n_models: c_int,
        speculator_paths: *const *const c_char,
        speculator_embedded: *const c_int,
        speculator_target: *const c_int,
        n_speculators: c_int,
        max_batch: c_int,
        kv_bits: c_int,
        paged_kv: c_int,
    ) -> c_int;
    /// 1 = the window fits the co-resident set, 0 = it does not, -1 = unknown.
    pub fn baseRT_context_window_fits(
        model_paths: *const *const c_char,
        n_models: c_int,
        max_batch: c_int,
        kv_bits: c_int,
        paged_kv: c_int,
        window: c_int,
    ) -> c_int;
    pub fn baseRT_get_error() -> *const c_char;

    // === Tokenization ===

    pub fn baseRT_encode(
        model: baseRT_model_t,
        text: *const c_char,
        out_tokens: *mut u32,
        max_tokens: c_int,
    ) -> c_int;
    /// Special-token strings in `text` are ordinary text (client content),
    /// never control ids. See the header.
    pub fn baseRT_encode_plain(
        model: baseRT_model_t,
        text: *const c_char,
        out_tokens: *mut u32,
        max_tokens: c_int,
    ) -> c_int;
    /// `n_pieces` texts encoded as one: piece `i` parses marker strings
    /// when `plain[i]` is zero and keeps them as text otherwise, with the
    /// pre-tokenizer seeing the concatenation. See the header.
    pub fn baseRT_encode_pieces(
        model: baseRT_model_t,
        texts: *const *const c_char,
        plain: *const c_int,
        n_pieces: c_int,
        out_tokens: *mut u32,
        max_tokens: c_int,
    ) -> c_int;

    pub fn baseRT_decode_token(model: baseRT_model_t, token_id: u32) -> *const c_char;
    /// Length-preserving stateless decode (byte-level tokens can contain
    /// NUL); returns the true byte length, which may exceed `max_bytes`.
    pub fn baseRT_decode_token_raw(
        model: baseRT_model_t,
        token_id: u32,
        out: *mut c_char,
        max_bytes: c_int,
    ) -> c_int;

    // === Generation ===

    pub fn baseRT_generate(
        model: baseRT_model_t,
        prompt_tokens: *const u32,
        n_prompt: c_int,
        max_tokens: c_int,
        sampling: BaseRTSamplingConfig,
        callback: baseRT_token_callback,
        user_data: *mut c_void,
    ) -> BaseRTGenerationStats;

    pub fn baseRT_generate_continue(
        model: baseRT_model_t,
        new_tokens: *const u32,
        n_new: c_int,
        max_tokens: c_int,
        sampling: BaseRTSamplingConfig,
        callback: baseRT_token_callback,
        user_data: *mut c_void,
    ) -> BaseRTGenerationStats;

    // === Low-level API ===

    pub fn baseRT_prefill(model: baseRT_model_t, tokens: *const u32, n_tokens: c_int) -> u32;

    pub fn baseRT_decode_step(model: baseRT_model_t, token_id: u32, position: c_int) -> u32;

    pub fn baseRT_chain_decode(
        model: baseRT_model_t,
        first_token: u32,
        start_position: c_int,
        count: c_int,
        out_tokens: *mut u32,
    ) -> c_int;

    pub fn baseRT_get_position(model: baseRT_model_t) -> c_int;

    pub fn baseRT_set_speculation(model: baseRT_model_t, enabled: bool);

    pub fn baseRT_reset(model: baseRT_model_t);

    // === Whisper transcription ===

    pub fn baseRT_transcribe(
        model: baseRT_model_t,
        wav_path: *const c_char,
        language: *const c_char,
        stats_out: *mut BaseRTTranscribeStats,
    ) -> *const c_char;

    pub fn baseRT_transcribe_pcm(
        model: baseRT_model_t,
        samples: *const c_float,
        n_samples: c_int,
        language: *const c_char,
        stats_out: *mut BaseRTTranscribeStats,
    ) -> *const c_char;

    pub fn baseRT_set_timestamps(model: baseRT_model_t, enabled: bool);

    pub fn baseRT_set_task(model: baseRT_model_t, task: *const c_char) -> bool;

    pub fn baseRT_set_initial_prompt(model: baseRT_model_t, text: *const c_char);

    pub fn baseRT_set_condition_on_previous_text(model: baseRT_model_t, enabled: bool);

    pub fn baseRT_transcribe_language(model: baseRT_model_t) -> *const c_char;

    pub fn baseRT_transcribe_audio_duration_ms(model: baseRT_model_t) -> c_int;

    pub fn baseRT_transcribe_segment_count(model: baseRT_model_t) -> c_int;

    pub fn baseRT_transcribe_segment(
        model: baseRT_model_t,
        index: c_int,
        out: *mut BaseRTTranscribeSegment,
    ) -> bool;

    pub fn baseRT_is_whisper(model: baseRT_model_t) -> bool;

    // === Streaming transcription ===

    pub fn baseRT_transcribe_pcm_stream(
        model: baseRT_model_t,
        samples: *const c_float,
        n_samples: c_int,
        language: *const c_char,
        stats_out: *mut BaseRTTranscribeStats,
        callback: baseRT_segment_callback,
        user_data: *mut c_void,
    ) -> *const c_char;

    pub fn baseRT_transcribe_stream(
        model: baseRT_model_t,
        wav_path: *const c_char,
        language: *const c_char,
        stats_out: *mut BaseRTTranscribeStats,
        callback: baseRT_segment_callback,
        user_data: *mut c_void,
    ) -> *const c_char;

    // === Embeddings ===

    pub fn baseRT_embed(
        model: baseRT_model_t,
        tokens: *const u32,
        n_tokens: c_int,
        out_embedding: *mut c_float,
        max_dims: c_int,
    ) -> c_int;

    pub fn baseRT_embed_text(
        model: baseRT_model_t,
        text: *const c_char,
        out_embedding: *mut c_float,
        max_dims: c_int,
    ) -> c_int;

    pub fn baseRT_embedding_dim(model: baseRT_model_t) -> c_int;
    pub fn baseRT_is_embedding_model(model: baseRT_model_t) -> bool;

    // === Chat templates ===

    pub fn baseRT_format_chat(
        model: baseRT_model_t,
        system_prompt: *const c_char,
        user_message: *const c_char,
    ) -> *const c_char;

    pub fn baseRT_chat_template(model: baseRT_model_t) -> *const c_char;

    /// Raw Jinja `chat_template` source folded in from the HF tokenizer
    /// (empty if the bundle carries none).
    pub fn baseRT_chat_template_jinja(model: baseRT_model_t) -> *const c_char;

    /// Create a grammar from a JSON Schema (converted to GBNF internally);
    /// NULL on error. Used to validate that a codec's tool-call grammar
    /// compiles in the engine (the runtime constrained-decode path).
    pub fn baseRT_grammar_create_from_schema(
        model: baseRT_model_t,
        json_schema: *const c_char,
    ) -> *mut std::os::raw::c_void;

    /// Create a grammar from an xgrammar STRUCTURAL TAG document (JSON, not a
    /// grammar string); NULL on error. The tag's content formats reach past
    /// JSON Schema — `sequence`, `const_string`, `regex`, `or` — which is how
    /// a dialect whose call body is not JSON (Gemma 4's `call:fn{k:v}`) still
    /// gets an enforceable `tool_choice`.
    pub fn baseRT_grammar_create_from_structural_tag(
        model: baseRT_model_t,
        tag_json: *const c_char,
    ) -> *mut std::os::raw::c_void;

    /// Create a grammar from a GBNF string; NULL on parse error.
    pub fn baseRT_grammar_create(
        model: baseRT_model_t,
        gbnf: *const c_char,
    ) -> *mut std::os::raw::c_void;

    /// Step a grammar by one token; 0 = the grammar REJECTS it. Lets a test
    /// assert what a compiled grammar actually admits, not merely that it
    /// compiled — the difference between "the tag parses" and "the tag
    /// permits the calls this dialect must be able to make".
    pub fn baseRT_grammar_accept_token(
        grammar: *mut std::os::raw::c_void,
        token_id: u32,
    ) -> std::os::raw::c_int;

    /// Reset a grammar's acceptance state to its initial stacks, so one
    /// handle can check several candidate strings.
    pub fn baseRT_grammar_reset(grammar: *mut std::os::raw::c_void);

    /// Free a grammar.
    pub fn baseRT_grammar_free(grammar: *mut std::os::raw::c_void);

    /// Number of added/special tokens the model declares.
    pub fn baseRT_special_token_count(model: baseRT_model_t) -> c_int;

    /// The `index`-th special token string; writes its id to `id_out`.
    pub fn baseRT_special_token(
        model: baseRT_model_t,
        index: c_int,
        id_out: *mut u32,
    ) -> *const c_char;

    // === Token counting ===

    pub fn baseRT_token_count(model: baseRT_model_t, text: *const c_char) -> c_int;

    // === Model inspection ===

    pub fn baseRT_tensor_count(model: baseRT_model_t) -> c_int;
    pub fn baseRT_tensor_name(model: baseRT_model_t, index: c_int) -> *const c_char;
    pub fn baseRT_tensor_dtype(model: baseRT_model_t, index: c_int) -> u32;
    pub fn baseRT_tensor_raw_dtype(model: baseRT_model_t, index: c_int) -> *const c_char;

    // === Profiling ===

    pub fn baseRT_profile_decode_step(
        model: baseRT_model_t,
        token_id: u32,
        position: c_int,
        timing_out: *mut c_float,
        max_entries: c_int,
    ) -> c_int;

    pub fn baseRT_profile_label(model: baseRT_model_t, index: c_int) -> *const c_char;

    // === GPU sampling ===

    pub fn baseRT_gpu_temperature_scale(model: baseRT_model_t, temperature: c_float);

    pub fn baseRT_gpu_repetition_penalty(
        model: baseRT_model_t,
        token_ids: *const u32,
        n_tokens: c_int,
        penalty: c_float,
    );

    // === Pre-load configuration (call BEFORE baseRT_load_model) ===

    pub fn baseRT_set_paged_kv(enable: c_int);
    pub fn baseRT_set_baked_decode(enable: c_int);
    pub fn baseRT_set_max_batch_size(n: c_int);
    pub fn baseRT_set_prefix_cache(enable: c_int);
    pub fn baseRT_set_prefill_chunk(n: c_int);
    pub fn baseRT_set_kv_bits(bits: c_int);
    pub fn baseRT_set_verbose(on: c_int);

    // === Error / capability introspection ===

    pub fn baseRT_get_error_code() -> c_int;
    pub fn baseRT_strerror(code: c_int) -> *const c_char;
    pub fn baseRT_capabilities(model: baseRT_model_t) -> u32;
    pub fn baseRT_eos_token_id(model: baseRT_model_t) -> u32;
    pub fn baseRT_is_eos_token(model: baseRT_model_t, token: u32) -> i32;
    pub fn baseRT_weights_identity(model: baseRT_model_t) -> u64;
    pub fn baseRT_kv_bits_effective(model: baseRT_model_t) -> c_int;
    pub fn baseRT_bos_id(model: baseRT_model_t) -> u32;

    // === Multi-sequence generation (paged-KV only) ===

    pub fn baseRT_sequence_create(model: baseRT_model_t) -> baseRT_sequence_t;
    pub fn baseRT_sequence_free(seq: baseRT_sequence_t);
    pub fn baseRT_sequence_rollback(seq: baseRT_sequence_t, length: c_int) -> c_int;

    // === Batched prefill/decode ===

    pub fn baseRT_batch_step(
        model: baseRT_model_t,
        seqs: *mut baseRT_sequence_t,
        n_seqs: c_int,
        in_tokens: *const u32,
        in_token_counts: *const c_int,
        out_tokens: *mut u32,
    ) -> c_int;
    pub fn baseRT_batch_step_fused(
        model: baseRT_model_t,
        seqs: *mut baseRT_sequence_t,
        n_seqs: c_int,
        in_tokens: *const u32,
        in_token_counts: *const c_int,
        out_tokens: *mut u32,
    ) -> c_int;
    pub fn baseRT_batch_step_fused_logits(
        model: baseRT_model_t,
        seqs: *mut baseRT_sequence_t,
        n_seqs: c_int,
        in_tokens: *const u32,
        in_token_counts: *const c_int,
    ) -> c_int;
    pub fn baseRT_batch_warmup(model: baseRT_model_t, max_batch: c_int) -> c_int;
    pub fn baseRT_max_prefill_chunk(model: baseRT_model_t) -> c_int;

    // === Host-side logits-row operations ===

    pub fn baseRT_read_batch_logits(
        model: baseRT_model_t,
        n_seqs: c_int,
        out_logits_f16: *mut c_void,
    ) -> c_int;
    pub fn baseRT_batch_logits_stride(model: baseRT_model_t) -> usize;
    pub fn baseRT_mask_logits_row(
        model: baseRT_model_t,
        row: *mut c_void,
        bitmask: *const i32,
    ) -> c_int;
    pub fn baseRT_sample_logits_row(
        model: baseRT_model_t,
        row: *const c_void,
        cfg: *const BaseRTSamplingConfig,
        prev_tokens: *const u32,
        n_prev: c_int,
        repeat_window: c_int,
        seed_offset: u32,
    ) -> u32;
    pub fn baseRT_argmax_logits_row(model: baseRT_model_t, row: *const c_void) -> u32;
    pub fn baseRT_logits_row_logprobs(
        model: baseRT_model_t,
        row: *const c_void,
        token: u32,
        top_k: c_int,
        out_token_logprob: *mut c_float,
        out_ids: *mut u32,
        out_logprobs: *mut c_float,
    ) -> c_int;

    // === Prefix cache — scheduler-driven primitives ===

    pub fn baseRT_prefix_match(
        model: baseRT_model_t,
        tokens: *const u32,
        n_tokens: c_int,
    ) -> BaseRTPrefixMatch;
    pub fn baseRT_sequence_seed_prefix(
        seq: baseRT_sequence_t,
        blocks: *const c_int,
        n_blocks: c_int,
        n_tokens: c_int,
    ) -> c_int;
    pub fn baseRT_page_size(model: baseRT_model_t) -> c_int;
    pub fn baseRT_prefix_insert(
        model: baseRT_model_t,
        tokens: *const u32,
        n_tokens: c_int,
        seq: baseRT_sequence_t,
    ) -> c_int;
    pub fn baseRT_prefix_unlock(model: baseRT_model_t, handle: u64);
    pub fn baseRT_prefix_release(model: baseRT_model_t, handle: u64);
    pub fn baseRT_prefix_discard(model: baseRT_model_t, handle: u64);
    pub fn baseRT_prefix_evict(model: baseRT_model_t, n_blocks: c_int) -> c_int;
    pub fn baseRT_prefix_cache_stats(
        model: baseRT_model_t,
        out_hits: *mut u64,
        out_misses: *mut u64,
        out_reused_tokens: *mut u64,
        out_blocks_cached: *mut c_int,
    );

    // === Recurrent (GDN) state snapshots — hybrid models only ===

    pub fn baseRT_gdn_snapshot_size(model: baseRT_model_t) -> c_int;
    pub fn baseRT_sequence_gdn_capture(seq: baseRT_sequence_t, out: *mut u8, cap: c_int) -> c_int;
    pub fn baseRT_sequence_gdn_restore(
        seq: baseRT_sequence_t,
        blob: *const u8,
        len: c_int,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    // -----------------------------------------------------------------------
    // Struct size tests — must match the C struct layout in types.h
    // -----------------------------------------------------------------------

    #[test]
    fn model_config_size() {
        // Full struct: decoder + encoder + Gemma4 + Qwen3.5-GDN + MoE +
        // vision + audio.
        // Must be far larger than the old 112-byte truncation; returning a
        // 112-byte struct by value from baseRT_get_config corrupted memory.
        assert!(
            mem::size_of::<BaseRTModelConfig>() > 1000,
            "config struct unexpectedly small ({})",
            mem::size_of::<BaseRTModelConfig>()
        );
        assert_eq!(mem::size_of::<BaseRTModelConfig>(), 1792);
    }

    #[test]
    fn model_config_size_matches_library() {
        // The authoritative drift check: compare this hand-written mirror
        // against sizeof(BaseRTModelConfig) as compiled into libbaseRT. A
        // mismatch means every field after the divergence point decodes as
        // garbage through baseRT_get_config.
        assert_eq!(
            unsafe { baseRT_model_config_sizeof() },
            mem::size_of::<BaseRTModelConfig>(),
            "BaseRTModelConfig mirror drifted from include/baseRT/types.h"
        );
    }

    #[test]
    fn model_config_alignment() {
        assert_eq!(mem::align_of::<BaseRTModelConfig>(), 4);
    }

    #[test]
    fn sampling_config_size() {
        // 5 floats/ints (20) + 2 floats (presence/freq, 28) + uint32 seed (32)
        // + int32 n_logit_bias (36) + 4-byte pad to 8-byte align (40)
        // + 2 pointers (8 bytes each on 64-bit) = 56 bytes
        assert_eq!(mem::size_of::<BaseRTSamplingConfig>(), 56);
    }

    #[test]
    fn sampling_config_alignment() {
        // Two pointer fields force 8-byte alignment on 64-bit.
        assert_eq!(mem::align_of::<BaseRTSamplingConfig>(), 8);
    }

    #[test]
    fn generation_stats_size() {
        // int + int + float + float + float + float = 24 bytes
        assert_eq!(mem::size_of::<BaseRTGenerationStats>(), 24);
    }

    #[test]
    fn generation_stats_alignment() {
        assert_eq!(mem::align_of::<BaseRTGenerationStats>(), 4);
    }

    #[test]
    fn transcribe_stats_size() {
        // int + float + float + float + float = 20 bytes
        assert_eq!(mem::size_of::<BaseRTTranscribeStats>(), 20);
    }

    #[test]
    fn transcribe_stats_alignment() {
        assert_eq!(mem::align_of::<BaseRTTranscribeStats>(), 4);
    }

    // -----------------------------------------------------------------------
    // Field offset tests — verify C-compatible field ordering
    // -----------------------------------------------------------------------

    #[test]
    fn model_config_field_offsets() {
        // SAFETY: zeroed bytes are a valid bit pattern for this all-POD struct.
        let base: BaseRTModelConfig = unsafe { mem::zeroed() };
        let base_ptr = &base as *const _ as usize;

        assert_eq!(&base.dim as *const _ as usize - base_ptr, 0);
        assert_eq!(&base.n_layers as *const _ as usize - base_ptr, 4);
        assert_eq!(&base.norm_eps as *const _ as usize - base_ptr, 40);
        assert_eq!(&base.rope_theta as *const _ as usize - base_ptr, 44);
        assert_eq!(
            &base.sliding_window_pattern as *const _ as usize - base_ptr,
            48
        );
        // The field the old binding dropped — everything after shifts by 4.
        assert_eq!(&base.sliding_window as *const _ as usize - base_ptr, 52);
        assert_eq!(&base.rope_local_theta as *const _ as usize - base_ptr, 56);
        assert_eq!(&base.architecture as *const _ as usize - base_ptr, 60);
        assert_eq!(&base.enc_n_layers as *const _ as usize - base_ptr, 92);
        assert_eq!(&base.enc_max_seq_len as *const _ as usize - base_ptr, 112);
        // Spot-check the tail blocks to confirm the full layout.
        assert_eq!(&base.swa_layers as *const _ as usize - base_ptr, 144);
        assert_eq!(&base.ffn_dims as *const _ as usize - base_ptr, 208);
        assert_eq!(&base.attn_output_gate as *const _ as usize - base_ptr, 1232);
        assert_eq!(
            &base.linear_attn_layers as *const _ as usize - base_ptr,
            1244
        );
        assert_eq!(&base.gdn_num_k_heads as *const _ as usize - base_ptr, 1308);
        // Nemotron-H SSM block, inserted between gdn_* and the MoE block —
        // 20 bytes that shifted every field below it.
        assert_eq!(&base.ssm_state_size as *const _ as usize - base_ptr, 1328);
        assert_eq!(&base.ssm_num_heads as *const _ as usize - base_ptr, 1344);
        assert_eq!(&base.n_experts as *const _ as usize - base_ptr, 1348);
        assert_eq!(
            &base.expert_weights_scale as *const _ as usize - base_ptr,
            1368
        );
        assert_eq!(&base.vision_n_layers as *const _ as usize - base_ptr, 1372);
        assert_eq!(&base.vision_arch as *const _ as usize - base_ptr, 1432);
        assert_eq!(&base.audio_n_layers as *const _ as usize - base_ptr, 1448);
        assert_eq!(&base.eoa_token_id as *const _ as usize - base_ptr, 1524);
        assert_eq!(&base.mrope_section as *const _ as usize - base_ptr, 1528);
        assert_eq!(
            &base.mrope_interleaved as *const _ as usize - base_ptr,
            1540
        );
        assert_eq!(
            &base.rope_scaling_factor as *const _ as usize - base_ptr,
            1544
        );
        assert_eq!(
            &base.rope_orig_max_pos as *const _ as usize - base_ptr,
            1556
        );
        assert_eq!(
            &base.rope_scaling_type as *const _ as usize - base_ptr,
            1560
        );
        assert_eq!(&base.qk_scale_factor as *const _ as usize - base_ptr, 1564);
        assert_eq!(&base.nope_layers as *const _ as usize - base_ptr, 1576);
        assert_eq!(&base.embed_norm_eps as *const _ as usize - base_ptr, 1640);
        assert_eq!(
            &base.vision_window_layers as *const _ as usize - base_ptr,
            1644
        );
        assert_eq!(&base.video_token_id as *const _ as usize - base_ptr, 1724);
        assert_eq!(&base.q_lora_rank as *const _ as usize - base_ptr, 1728);
        assert_eq!(&base.indexer_top_k as *const _ as usize - base_ptr, 1768);
        assert_eq!(
            &base.rope_yarn_beta_fast as *const _ as usize - base_ptr,
            1772
        );
    }

    #[test]
    fn sampling_config_field_offsets() {
        let base = BaseRTSamplingConfig::default();
        let base_ptr = &base as *const _ as usize;

        assert_eq!(&base.temperature as *const _ as usize - base_ptr, 0);
        assert_eq!(&base.top_k as *const _ as usize - base_ptr, 4);
        assert_eq!(&base.top_p as *const _ as usize - base_ptr, 8);
        assert_eq!(&base.min_p as *const _ as usize - base_ptr, 12);
        assert_eq!(&base.repeat_penalty as *const _ as usize - base_ptr, 16);
        assert_eq!(&base.presence_penalty as *const _ as usize - base_ptr, 20);
        assert_eq!(&base.frequency_penalty as *const _ as usize - base_ptr, 24);
        assert_eq!(&base.seed as *const _ as usize - base_ptr, 28);
        assert_eq!(&base.n_logit_bias as *const _ as usize - base_ptr, 32);
        // 4-byte padding before pointer alignment to 8.
        assert_eq!(&base.logit_bias_tokens as *const _ as usize - base_ptr, 40);
        assert_eq!(&base.logit_bias_values as *const _ as usize - base_ptr, 48);
    }

    #[test]
    fn generation_stats_field_offsets() {
        let base = BaseRTGenerationStats::default();
        let base_ptr = &base as *const _ as usize;

        assert_eq!(&base.prompt_tokens as *const _ as usize - base_ptr, 0);
        assert_eq!(&base.generated_tokens as *const _ as usize - base_ptr, 4);
        assert_eq!(&base.prefill_time_ms as *const _ as usize - base_ptr, 8);
        assert_eq!(&base.decode_time_ms as *const _ as usize - base_ptr, 12);
        assert_eq!(
            &base.prefill_tokens_per_sec as *const _ as usize - base_ptr,
            16
        );
        assert_eq!(
            &base.decode_tokens_per_sec as *const _ as usize - base_ptr,
            20
        );
    }

    // -----------------------------------------------------------------------
    // Default trait tests
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_default_values() {
        let cfg = BaseRTSamplingConfig::default();
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 40);
        assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(cfg.min_p, 0.0);
        assert_eq!(cfg.repeat_penalty, 1.0);
        assert_eq!(cfg.presence_penalty, 0.0);
        assert_eq!(cfg.frequency_penalty, 0.0);
        assert_eq!(cfg.seed, 0);
        assert_eq!(cfg.n_logit_bias, 0);
        assert!(cfg.logit_bias_tokens.is_null());
        assert!(cfg.logit_bias_values.is_null());
    }

    #[test]
    fn prefix_match_layout() {
        // int matched_tokens (4) + int n_blocks (8) + const int *blocks
        // (16) + uint64_t handle (24). Returned BY VALUE from
        // baseRT_prefix_match, so the mirror must match the C layout
        // exactly.
        assert_eq!(mem::size_of::<BaseRTPrefixMatch>(), 24);
        assert_eq!(mem::offset_of!(BaseRTPrefixMatch, matched_tokens), 0);
        assert_eq!(mem::offset_of!(BaseRTPrefixMatch, n_blocks), 4);
        assert_eq!(mem::offset_of!(BaseRTPrefixMatch, blocks), 8);
        assert_eq!(mem::offset_of!(BaseRTPrefixMatch, handle), 16);
    }

    #[test]
    fn transcribe_stats_default_values() {
        let stats = BaseRTTranscribeStats::default();
        assert_eq!(stats.n_tokens, 0);
        assert_eq!(stats.audio_ms, 0.0);
        assert_eq!(stats.encode_ms, 0.0);
        assert_eq!(stats.decode_ms, 0.0);
        assert_eq!(stats.total_ms, 0.0);
    }

    #[test]
    fn generation_stats_default_values() {
        let stats = BaseRTGenerationStats::default();
        assert_eq!(stats.prompt_tokens, 0);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.prefill_time_ms, 0.0);
        assert_eq!(stats.decode_time_ms, 0.0);
        assert_eq!(stats.prefill_tokens_per_sec, 0.0);
        assert_eq!(stats.decode_tokens_per_sec, 0.0);
    }
}
