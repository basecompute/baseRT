#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Coarse error-code categories returned by `baseRT_get_error_code()`.
///
/// `baseRT_get_error()` always carries the full human-readable message;
/// the code lets callers branch on failure mode without parsing strings.
/// Categories are intentionally coarse — adding members is a minor
/// version bump, repurposing a value is a major bump. Untriaged failures
/// land in `BASERT_ERR_UNKNOWN`.
typedef enum {
    BASERT_OK = 0,
    BASERT_ERR_UNKNOWN = 1,
    BASERT_ERR_INVALID_ARGUMENT = 2,  // caller-side bug (null, out-of-range, bad shape)
    BASERT_ERR_FILE_NOT_FOUND = 3,    // model_path / kernel_library_path missing
    BASERT_ERR_INVALID_FORMAT = 4,    // file present but failed format checks
    BASERT_ERR_UNSUPPORTED = 5,       // recognized but not yet implemented
    BASERT_ERR_OUT_OF_MEMORY = 6,
    BASERT_ERR_GPU = 7,                // Metal device / pipeline failure
    BASERT_ERR_GENERATION_FAILED = 8,  // mid-run failure (NaN, dispatch error, …)
} BaseRTErrorCode;

/// Model configuration extracted from weight file metadata.
typedef struct {
    // Decoder (or decoder-only LLM) parameters
    uint32_t dim;                     // embedding dimension
    uint32_t n_layers;                // transformer layers (decoder layers for enc-dec)
    uint32_t n_heads;                 // query heads
    uint32_t n_kv_heads;              // key/value heads (GQA)
    uint32_t head_dim;                // dim per head
    uint32_t q_dim;                   // total Q projection output (n_heads * head_dim)
    uint32_t kv_dim;                  // total KV projection output (n_kv_heads * head_dim)
    uint32_t ffn_dim;                 // feed-forward intermediate dimension
    uint32_t vocab_size;              // vocabulary size
    uint32_t max_seq_len;             // maximum context length
    float norm_eps;                   // RMSNorm/LayerNorm epsilon
    float rope_theta;                 // RoPE base frequency (0 = no RoPE)
    uint32_t sliding_window_pattern;  // 0 = all global or use swa_layers bitfield, 6 = every 6th layer global (Gemma 3)
    uint32_t sliding_window;          // 0 = full context, >0 = sliding window size for SWA layers
    float rope_local_theta;           // RoPE theta for local/sliding-window layers (0 = same as rope_theta)
    char architecture[32];            // "llama", "qwen3", "gemma", "gemma3", "gemma4", "whisper"

    // Encoder parameters (0 = decoder-only model)
    uint32_t enc_n_layers;     // encoder transformer layers
    uint32_t enc_n_heads;      // encoder attention heads
    uint32_t enc_dim;          // encoder embedding dimension (may equal dim)
    uint32_t enc_ffn_dim;      // encoder feed-forward dimension
    uint32_t n_mels;           // mel spectrogram bins (80/128, 0 = not audio model)
    uint32_t enc_max_seq_len;  // encoder max positions (1500 for 30s Whisper)

    // Gemma 4-specific fields (0 / empty = not applicable)
    uint32_t n_embd_per_layer;       // per-layer embedding dim (PLE), 0 = no PLE
    uint32_t n_layer_kv_from_start;  // first N layers own KV; [n_layer_kv_from_start, n_layers) reuse earlier cache. 0
                                     // = all own KV
    float logit_softcap;             // final logit softcap: (x/cap).tanh()*cap, 0 = disabled
    float attention_scale;           // explicit attention scale (0 = derive 1/sqrt(head_dim))
    uint32_t head_dim_swa;           // per-layer head_dim for SWA layers (0 = uniform, use head_dim)
    uint32_t head_dim_global;        // per-layer head_dim for global (full-attention) layers
    // Partial-rotary factor for global (full-attention) layers. 0 or 1 = full rotation.
    // Gemma 4 26B-A4B / 31B: 0.25 — only the first 25% of head_dim_global pairs rotate,
    // the remainder stay unchanged. On GGUF this is encoded via `rope_freqs.weight`
    // (factors of 1.0 for the rotating pairs, ~1e30 for the rest). MLX safetensors don't
    // ship that tensor, so we parse this field from the HF config and synthesize the
    // divisor buffer at forward-pass time.
    float global_rope_partial_factor;
    // Per-layer SWA pattern as packed bitfield (bit=1 means SWA/local, bit=0 means global).
    // Supports up to 512 layers (64 bytes). Bit index i = layer i.
    // If sliding_window_pattern is nonzero, this bitfield is ignored (legacy Gemma 3 path).
    uint8_t swa_layers[64];

    // Per-layer FFN dimensions. Non-zero entries override cfg.ffn_dim for that layer.
    // E2B uses 6144 for SWA layers and 12288 for global layers.
    // When all zeros, the uniform cfg.ffn_dim applies to all layers.
    uint32_t ffn_dims[128];

    // Per-layer number of KV heads. Non-zero entries override cfg.n_kv_heads.
    // Gemma 4 26B-A4B: 8 for SWA layers, 2 for Global layers.
    // Gemma 4 31B:    16 for SWA layers, 4 for Global layers.
    // E4B / E2B: all zeros (uniform cfg.n_kv_heads applies).
    uint32_t n_kv_heads_per_layer[128];

    // Mixture-of-Experts (0 = dense model).
    // Gemma 4 26B-A4B: n_experts=128, n_experts_used=8, n_experts_shared=1 (via dense ffn.*), expert_gating=0
    // (softmax), norm_topk_prob=0 Qwen3.6-35B-A3B (qwen35moe): n_experts=128, n_experts_used=8, n_experts_shared=0,
    // expert_gating=0 (softmax), norm_topk_prob=1
    uint32_t n_experts;         // total routed experts; 0 = dense model (MoE disabled)
    uint32_t n_experts_used;    // top-k experts per token
    uint32_t n_experts_shared;  // shared (always-on) experts; Gemma 4 = 1 via dense ffn.*, Qwen = 0
    uint32_t expert_ffn_dim;    // per-expert FFN intermediate dim
    uint8_t expert_gating;      // 0 = softmax, 1 = sigmoid
    uint8_t norm_topk_prob;     // 1 = renormalize top-k weights to sum to 1 (Qwen), 0 = leave as-is (Gemma)
    uint8_t _moe_pad[2];        // align to 4 bytes

    // Vision tower (Gemma 4, PaliGemma, Llava-style multimodal).
    // All zero = no vision tower.
    uint32_t vision_n_layers;        // ViT encoder layers
    uint32_t vision_dim;             // ViT hidden size (e.g. 768)
    uint32_t vision_n_heads;         // ViT attention heads
    uint32_t vision_head_dim;        // per-head dim
    uint32_t vision_ffn_dim;         // ViT FFN intermediate
    uint32_t vision_patch_size;      // patch side length in pixels (e.g. 16)
    uint32_t vision_image_size;      // canonical input image side (e.g. 896)
    uint32_t vision_pooling_kernel;  // square pooling kernel after encoder (1 = no pool)
    uint32_t vision_soft_tokens;     // image tokens emitted per image (e.g. 280)
    float vision_norm_eps;
    float vision_rope_theta;         // 0 = no RoPE in ViT
    uint32_t vision_pos_embed_size;  // learned positional embedding size (0 = none)
    uint32_t image_token_id;         // text-side token id used as a placeholder for image features
    uint32_t boi_token_id;           // begin-of-image token (0 = none)
    uint32_t eoi_token_id;           // end-of-image token (0 = none)

    // Audio tower (Gemma 4 Conformer encoder). All zero = no audio tower.
    uint32_t audio_n_layers;          // Conformer blocks (e.g. 12)
    uint32_t audio_dim;               // hidden size (e.g. 1024)
    uint32_t audio_n_heads;           // attention heads (e.g. 8)
    uint32_t audio_head_dim;          // per-head dim (audio_dim / audio_n_heads)
    uint32_t audio_ffn_dim;           // FFN intermediate (4 * audio_dim)
    uint32_t audio_output_proj_dim;   // output proj to this dim (e.g. 1536), 0 = no proj
    uint32_t audio_chunk_size;        // chunked attention block size (e.g. 12)
    uint32_t audio_left_context;      // left context frames for attention (e.g. 13)
    uint32_t audio_conv_kernel;       // LightConv1d kernel size (e.g. 5)
    uint32_t audio_soft_tokens;       // max audio tokens per clip (e.g. 750)
    float audio_logit_softcap;        // attention logit cap (e.g. 50.0)
    float audio_norm_eps;             // RMSNorm epsilon
    float audio_gradient_clip;        // gradient clipping bound (e.g. 1e10)
    float audio_residual_weight;      // FFW residual scale (e.g. 0.5)
    float audio_ms_per_token;         // ms per output token (e.g. 40.0)
    uint32_t audio_sscp_channels[2];  // subsampling conv channels (e.g. {128, 32})
    uint32_t audio_token_id;          // placeholder token for audio features
    uint32_t boa_token_id;            // begin-of-audio token (0 = none)
    uint32_t eoa_token_id;            // end-of-audio token (0 = none)
} BaseRTModelConfig;

/// Transcription result statistics.
typedef struct {
    int n_tokens;     // output tokens generated
    float audio_ms;   // mel spectrogram computation time
    float encode_ms;  // encoder forward pass time
    float decode_ms;  // decoder generation time
    float total_ms;   // total wall time
} BaseRTTranscribeStats;

/// Sampling configuration for text generation.
///
/// Fields are appended at the end; older callers that brace-init only the
/// first five values still build (new fields zero-init to "disabled"). The
/// extended fields below mirror OpenAI's chat-completion params:
///
///   * presence_penalty  — subtract this from logits of tokens that have
///     appeared at least once in the trailing history window. Disabled at 0.
///   * frequency_penalty — subtract `freq * count(token)` for each token's
///     occurrence count in the trailing window. Disabled at 0.
///   * seed              — RNG seed for sampling. 0 = wall-clock-seeded
///     (non-deterministic); non-zero re-seeds the thread-local RNG before
///     generation begins.
///   * logit_bias_*      — additive per-token logit bias. Parallel arrays:
///     `logit_bias_tokens[i]` gets `+ logit_bias_values[i]` added to its
///     logit each step. Pointers must remain valid for the duration of the
///     generate call; `n_logit_bias = 0` disables.
typedef struct {
    float temperature;  // 0 = greedy
    int top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;
    uint32_t seed;
    int32_t n_logit_bias;
    const int32_t *logit_bias_tokens;
    const float *logit_bias_values;
} BaseRTSamplingConfig;

/// Generation result statistics.
typedef struct {
    int prompt_tokens;
    int generated_tokens;
    float prefill_time_ms;
    float decode_time_ms;
    float prefill_tokens_per_sec;
    float decode_tokens_per_sec;
} BaseRTGenerationStats;

#ifdef __cplusplus
}
#endif
