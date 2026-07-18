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

    // Mixture-of-Experts (0 = dense)
    pub n_experts: u32,
    pub n_experts_used: u32,
    pub n_experts_shared: u32,
    pub expert_ffn_dim: u32,
    pub expert_gating: u8,
    pub norm_topk_prob: u8,
    pub _moe_pad: [u8; 2],

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

/// Callback for streaming token output. Return `false` to stop generation.
pub type baseRT_token_callback =
    Option<unsafe extern "C" fn(token_id: u32, text: *const c_char, user_data: *mut c_void) -> bool>;

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

    // === Model info ===

    pub fn baseRT_get_config(model: baseRT_model_t) -> BaseRTModelConfig;
    pub fn baseRT_model_config_sizeof() -> usize;
    pub fn baseRT_model_memory(model: baseRT_model_t) -> usize;
    pub fn baseRT_get_error() -> *const c_char;

    // === Tokenization ===

    pub fn baseRT_encode(
        model: baseRT_model_t,
        text: *const c_char,
        out_tokens: *mut u32,
        max_tokens: c_int,
    ) -> c_int;

    pub fn baseRT_decode_token(model: baseRT_model_t, token_id: u32) -> *const c_char;

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

    // === Chat templates ===

    pub fn baseRT_format_chat(
        model: baseRT_model_t,
        system_prompt: *const c_char,
        user_message: *const c_char,
    ) -> *const c_char;

    pub fn baseRT_chat_template(model: baseRT_model_t) -> *const c_char;

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
        assert_eq!(mem::size_of::<BaseRTModelConfig>(), 1540);
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
        assert_eq!(&base.sliding_window_pattern as *const _ as usize - base_ptr, 48);
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
        assert_eq!(&base.linear_attn_layers as *const _ as usize - base_ptr, 1244);
        assert_eq!(&base.gdn_num_k_heads as *const _ as usize - base_ptr, 1308);
        assert_eq!(&base.n_experts as *const _ as usize - base_ptr, 1328);
        assert_eq!(&base.vision_n_layers as *const _ as usize - base_ptr, 1348);
        assert_eq!(&base.vision_arch as *const _ as usize - base_ptr, 1408);
        assert_eq!(&base.audio_n_layers as *const _ as usize - base_ptr, 1424);
        assert_eq!(&base.eoa_token_id as *const _ as usize - base_ptr, 1500);
        assert_eq!(&base.mrope_section as *const _ as usize - base_ptr, 1504);
        assert_eq!(&base.mrope_interleaved as *const _ as usize - base_ptr, 1516);
        assert_eq!(&base.rope_scaling_factor as *const _ as usize - base_ptr, 1520);
        assert_eq!(&base.rope_orig_max_pos as *const _ as usize - base_ptr, 1532);
        assert_eq!(&base.rope_scaling_type as *const _ as usize - base_ptr, 1536);
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
        assert_eq!(&base.prefill_tokens_per_sec as *const _ as usize - base_ptr, 16);
        assert_eq!(&base.decode_tokens_per_sec as *const _ as usize - base_ptr, 20);
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
