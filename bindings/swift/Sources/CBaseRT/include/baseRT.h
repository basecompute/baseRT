#pragma once

/// BaseRT C API — LLM inference for Apple Silicon.
///
/// Usage:
///   baseRT_model_t model = baseRT_load_model("model.base", "baseRT.metallib", 0);
///   uint32_t tokens[1024];
///   int n = baseRT_encode(model, "Hello, world!", tokens, 1024);
///   baseRT_generate(model, tokens, n, 256, sampling, callback, NULL);
///   baseRT_free_model(model);
///
/// ── API stability ──────────────────────────────────────────────────
/// This header is the supported surface. Anything in `src/` is internal
/// and may change in any release. Within a major version, we promise:
///   * No symbol is removed; no signature changes.
///   * `BaseRT*` struct layouts are stable. New fields may be appended
///     at the end of a struct (size grows; old offsets remain valid).
///     If you statically link, recompile after upgrading.
///   * `BaseRTErrorCode` may gain values in a minor release; never
///     repurposes an existing value.
/// Pre-1.0 (BASERT_VERSION_MAJOR == 0) the above is intent, not contract.
///
/// ── Error handling ─────────────────────────────────────────────────
/// Most functions report failure by returning NULL / 0 / -1 (see each
/// function's doc). On failure, call `baseRT_get_error()` for a human-
/// readable message and `baseRT_get_error_code()` for a category code.
/// Both reset only when the next API call succeeds; they're thread-local.
///
/// ── Threading model ────────────────────────────────────────────────
///   * Error state (`baseRT_get_error()`, `baseRT_get_error_code()`),
///     `baseRT_decode_token`, and any function documented as returning
///     a "static string" or "valid until next call" use thread-local
///     storage. They are safe to call from multiple threads, but the
///     returned pointer is only valid on the calling thread until the
///     next call (on that thread) that mutates the same buffer. Copy
///     before crossing thread boundaries.
///   * `baseRT_set_kv_bits()` writes a process-wide global and must be
///     called from a single thread before any `baseRT_load_model()`.
///   * A `baseRT_model_t` is single-owner. The runtime does not
///     serialize concurrent calls on the same handle — the caller is
///     responsible for one-thread-at-a-time access. Concurrent calls
///     on *different* handles are safe.
///   * Callback `text` pointers (token / segment callbacks) point to
///     thread-local buffers owned by the runtime. They are valid for
///     the duration of the callback only; copy if you need to keep
///     them.
///
/// ── Ownership ──────────────────────────────────────────────────────
/// Every function returning a handle has a matching `_free` (model,
/// grammar). Every function returning `const char *` returns into
/// runtime-owned storage — do not free, do not retain past the next
/// call. Output buffers passed by pointer are caller-allocated and
/// caller-freed.

#include "types.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// === Versioning ===

#define BASERT_VERSION_MAJOR 0
#define BASERT_VERSION_MINOR 2
#define BASERT_VERSION_PATCH 1

/// Compile-time version, packed as `(MAJOR<<16) | (MINOR<<8) | PATCH`.
/// Useful for `#if BASERT_VERSION >= 0x000200` feature checks.
#define BASERT_VERSION ((BASERT_VERSION_MAJOR << 16) | (BASERT_VERSION_MINOR << 8) | BASERT_VERSION_PATCH)

/// Runtime-resolved version string ("0.2.0"). Matches the linked
/// library; useful for diagnostics when a binding loads a different
/// `.dylib` than it was compiled against.
const char *baseRT_version_string(void);

/// Opaque model handle.
typedef void *baseRT_model_t;

// === Model lifecycle ===

/// Load a model from a `.base` bundle (or whisper.cpp GGML file).
/// Other source formats (GGUF, HF safetensors, MLX safetensors) must be
/// converted offline first via `basert convert`.
/// kernel_library_path: path to the compiled GPU kernel library (on the Metal
///   backend, baseRT.metallib), or NULL to auto-detect. Auto-detect order: a
///   kernel library next to the build, then a copy embedded in the loaded
///   binary itself (single-file distributions ship the shared library with the
///   kernels linked in, so NULL just works). Named generically so non-Metal
///   backends (CUDA/ROCm, future) can reuse the same parameter.
/// max_context: maximum context window (0 = use model default, capped at 4096).
/// Returns NULL on failure.
baseRT_model_t baseRT_load_model(const char *model_path, const char *kernel_library_path, int max_context);

/// Load a model with per-call options instead of the process-wide
/// baseRT_set_* pre-load setters. `opts` may be NULL (identical to
/// baseRT_load_model); otherwise set opts->struct_size =
/// sizeof(BaseRTLoadOptions) and zero any field you want left at its default
/// (see BaseRTLoadOptions in types.h). Loads through this entry point are
/// serialized against each other; the legacy globals are untouched from the
/// caller's point of view (saved, applied for the load, restored). A
/// concurrent plain baseRT_load_model on another thread races the applied
/// values exactly as it would race the legacy setters — serialize loads if
/// you mix the two forms.
baseRT_model_t baseRT_load_model_ex(const char *model_path, const char *kernel_library_path, int max_context,
                                    const BaseRTLoadOptions *opts);

/// Capability flags reported by baseRT_capabilities(). A scheduler should
/// branch on these instead of probing entry points for BASERT_ERR_UNSUPPORTED
/// or inferring support from BaseRTModelConfig fields.
enum {
    BASERT_CAP_PAGED_KV = 1u << 0,      ///< loaded with paged KV (sequences, prefix seeding)
    BASERT_CAP_SEQUENCES = 1u << 1,     ///< baseRT_sequence_create works: continuous batching
    BASERT_CAP_HOST_LOGITS = 1u << 2,   ///< baseRT_batch_step_fused_logits + read_batch_logits work
    BASERT_CAP_PREFIX_CACHE = 1u << 3,  ///< RadixCache prefix reuse is active
    BASERT_CAP_GDN_SNAPSHOT = 1u << 4,  ///< hybrid-GDN recurrent-state snapshot/restore
};

/// What this loaded model supports, as BASERT_CAP_* flags. Pure query: no GPU
/// work, no probe sequences, never touches the error state. Values are fixed
/// at load time. When a capability is absent and the reason matters (e.g. an
/// operator-facing "continuous batching disabled because ..." message), call
/// the gated entry point once and read baseRT_get_error().
uint32_t baseRT_capabilities(baseRT_model_t model);

/// Override the KV cache element width for the next baseRT_load_model call.
///   bits = 0  → auto (per-model default; Q8_0 when head_dim%32==0)
///   bits = 8  → force Q8_0 K and V slabs (1.88x smaller; tiny precision cost)
///   bits = 16 → force F16 K and V slabs (memory-heavier; full precision)
/// Process-wide; persists across loads. Must be called before
/// baseRT_load_model. Other values are ignored.
void baseRT_set_kv_bits(int bits);

/// Enable engine diagnostics (RoPE/tokenizer/GPU/architecture dumps, the
/// per-token dispatch-command count, "Warming up"). Off by default so end
/// users see only model output.
void baseRT_set_verbose(int on);

/// Toggle paged-KV mode for the next baseRT_load_model call.
///   enable = 0 → contiguous KV cache (default; existing layout)
///   enable = 1 → paged KV cache + block-table dispatch
/// Paged mode allocates KV in fixed-size blocks (per-model page size: 16
/// default, 8 for models with head_dim>=256) and addresses each layer's
/// slab through a CSR block table. Required foundation for multi-sequence
/// continuous batching and prefix caching. Process-wide; persists across
/// loads. Must be called before baseRT_load_model.
void baseRT_set_paged_kv(int enable);

/// Override the maximum batch size for baseRT_batch_decode_step.
/// Sizes scratch.logits as [B, vocab] at load time so the engine's output
/// GEMM can emit one logit row per sequence. Default 1 (single-seq);
/// callers driving continuous batching should set this to the expected
/// max in-flight sequence count before baseRT_load_model. Process-wide;
/// persists across loads. Capped at prefill_chunk at load time.
void baseRT_set_max_batch_size(int n);

/// Enable the prefix cache: shares the KV of common prompt prefixes across
/// requests (via a radix tree over the paged block pool), so the scheduler
/// can skip re-prefilling a shared system prompt / chat history. No effect
/// unless --paged-kv is also on. Process-wide; persists across loads. Must be
/// called before baseRT_load_model. Drive it via the baseRT_prefix_* API.
void baseRT_set_prefix_cache(int enable);

/// Override the prefill chunk size (tokens per prefill GEMM batch).
///   n = 0  → per-chip default (recommended)
///   n >= 16 → clamp to this; shrinks the batch_* scratch footprint
///     (~linearly) so an oversized model fits a tighter GPU working-set
///     budget, at a prefill-throughput cost. Values outside [16, chip max]
///     are ignored. Process-wide; read at load. Set before baseRT_load_model.
void baseRT_set_prefill_chunk(int n);

/// Paged-weights load policy (issue #113): per-tensor pinned weight buffers
/// for models past the OS wired-page budget.
///   mode = 0 → auto (default): normal load, retry paged on GPU-OOM
///   mode = 1 → force paged-weights on the first attempt (oversized/testing)
///   mode = 2 → disable the retry (fail hard on OOM, pre-#113 behavior)
/// Process-wide; read at load. Set before baseRT_load_model.
void baseRT_set_paged_weights(int mode);

/// Toggle the baked-decode fast path (DispatchTable replay).
///   enable = 1 → on (default): replay the baked table when eligible
///   enable = 0 → force the live fused decode path (baked-vs-live A/B)
/// Process-wide; read per decode step.
void baseRT_set_baked_decode(int enable);

/// CUDA decode-replay strategy for fused batch decode ticks.
///   mode = 2 → stream capture (default): capture the live walk as a CUDA
///     graph on a shape's second sighting; replay is numerically identical
///     to the live tick it captured.
///   mode = 1 → cmd-table replay (experiment; measured a net loss on GB10
///     serving — see batch_step_fused_common).
///   mode = 0 → neither: fused ticks run live (no capture, no cmd-table
///     replay).
/// Governs ONLY the CUDA fused-tick replay strategy. The baked DispatchTable
/// fast path for pure-decode ticks is a separate mechanism with its own
/// switch — baseRT_set_baked_decode(0) — matching the two knobs' historical
/// independence. Process-wide; the fused decode paths latch the mode on
/// first use, so set it before the first decode step. Other values are
/// ignored. No effect on the Metal backend. (Replaces the
/// BASERT_FUSED_BUCKETS / BASERT_STREAM_CAPTURE env gates.)
void baseRT_set_decode_replay(int mode);

/// GPU-wait timeout in milliseconds (Metal backend).
///   ms > 0   → (default: 300000, i.e. 5 min) return a loggable error if a
///     committed command buffer doesn't reach a terminal status in time. The
///     CPU unblocks; the wedged GPU work is NOT torn down (only a GPU
///     reset/reboot reclaims it). Chosen far above any legitimate single
///     command buffer so only a real wedge trips it.
///   ms = 0   → block indefinitely in the GPU wait (opt out of the bound).
/// No-op on non-Metal backends. Process-wide; read once per wait.
void baseRT_set_gpu_wait_timeout_ms(double ms);

/// Free all resources associated with a model.
void baseRT_free_model(baseRT_model_t model);

// === Model info ===

/// Get model configuration.
BaseRTModelConfig baseRT_get_config(baseRT_model_t model);

/// sizeof(BaseRTModelConfig) as compiled into the library. Language
/// bindings that mirror the struct by hand (Python/Rust/Node) compare
/// this against their mirror's size at load time so layout drift fails
/// loudly instead of decoding garbage fields.
size_t baseRT_model_config_sizeof(void);

/// Get total GPU memory used by model (bytes).
size_t baseRT_model_memory(baseRT_model_t model);

/// Get last error message (thread-local). The string is valid until
/// the next API call from the same thread that fails or that explicitly
/// resets the error state. Returns "" when there is no pending error.
const char *baseRT_get_error(void);

/// Coarse error-category code companion to `baseRT_get_error()`.
/// Returns `BASERT_OK` when there is no pending error. Resets when the
/// next API call succeeds. Thread-local — see header preamble.
BaseRTErrorCode baseRT_get_error_code(void);

/// Human-readable name of an error code (e.g. "FILE_NOT_FOUND").
/// The returned string is static and does not need to be freed.
const char *baseRT_strerror(BaseRTErrorCode code);

// === Tokenization ===

/// Encode text to token IDs. Returns number of tokens written.
int baseRT_encode(baseRT_model_t model, const char *text, uint32_t *out_tokens, int max_tokens);

/// Decode a single token ID to text. Returns static string (do not free).
///
/// Advances the tokenizer's incremental-decode state — call this from
/// generation callbacks where the token is part of the streaming output.
const char *baseRT_decode_token(baseRT_model_t model, uint32_t token_id);

/// Stateless variant of `baseRT_decode_token`. Decodes a single token id
/// without touching the tokenizer's incremental state — safe to call any
/// number of times from inside a token callback (e.g. when rendering
/// `top_logprobs` alternatives for `/v1/chat/completions`). The returned
/// string lives in a thread-local buffer that is overwritten on each call.
const char *baseRT_decode_token_static(baseRT_model_t model, uint32_t token_id);

/// Length-preserving variant of `baseRT_decode_token_static` for callers
/// that need the token's EXACT raw bytes. Byte-level BPE / byte-fallback
/// tokens can decode to bytes containing 0x00, which the C-string variants
/// above silently truncate at. Writes up to `max_bytes` into `out` (no NUL
/// terminator appended) and returns the token's FULL byte length — if the
/// return value exceeds `max_bytes`, call again with a larger buffer.
/// Stateless; does not touch the incremental-decode state. Returns 0 for
/// tokens that decode to nothing (e.g. filtered special tokens), <0 on
/// invalid arguments.
int baseRT_decode_token_raw(baseRT_model_t model, uint32_t token_id, char *out, int max_bytes);

// === Generation ===

/// Callback for streaming token output.
/// Return false to stop generation.
typedef bool (*baseRT_token_callback)(uint32_t token_id, const char *text, void *user_data);

/// Generate tokens from a prompt.
/// Returns generation statistics.
BaseRTGenerationStats baseRT_generate(baseRT_model_t model, const uint32_t *prompt_tokens, int n_prompt, int max_tokens,
                                      BaseRTSamplingConfig sampling, baseRT_token_callback callback, void *user_data);

/// Generate tokens from a prompt WITH serial RadixCache prefix reuse.
///
/// Semantically identical to `baseRT_generate` (single default sequence, same
/// greedy/sampled output), but when the model was loaded with BOTH `--paged-kv`
/// and `--prefix-cache` it reuses the longest cached whole-block prompt prefix:
/// it resets the default sequence, matches the prompt against the RadixCache,
/// seeds the shared prefix blocks into the default sequence, prefills ONLY the
/// divergent suffix `[matched_tokens, n_prompt)`, then inserts the full prompt
/// back into the cache for later reuse on generation end.
///
/// Greedy output is BIT-IDENTICAL to a cold `baseRT_generate` (the seeded blocks
/// hold the same KV a cold prefill would have produced). When the prefix cache
/// or paged-KV is disabled this is an exact passthrough to `baseRT_generate`.
///
/// Attention-KV (non-hybrid) models only — hybrid-GDN models fall back to the
/// plain path (their recurrent state is not block-shareable).
BaseRTGenerationStats baseRT_generate_cached(baseRT_model_t model, const uint32_t *prompt_tokens, int n_prompt,
                                             int max_tokens, BaseRTSamplingConfig sampling,
                                             baseRT_token_callback callback, void *user_data);

// === Multi-sequence generation (paged-KV only) ===

/// Opaque per-sequence handle.
///
/// A sequence is an independent KV-cache state that shares the model's paged-KV
/// block pool with other sequences. Many sequences can coexist on one model:
/// each consumes only the blocks it actually needs, not a full pre-allocated
/// max_context slab. The pool size is the per-process memory cap; sequences
/// are lightweight (an indptr + block list + GPU block table per sequence).
typedef struct baseRT_sequence_s *baseRT_sequence_t;

/// Allocate a fresh sequence handle on the model's shared paged-KV pool.
/// The model must be loaded with --paged-kv (`baseRT_set_paged_kv(1)`) —
/// returns NULL with BASERT_ERR_UNSUPPORTED otherwise.
///
/// Limitation: sequences are not safe to use concurrently on a single model
/// (dispatch state is shared). Schedule them sequentially — one
/// `baseRT_sequence_generate` call at a time per model.
baseRT_sequence_t baseRT_sequence_create(baseRT_model_t model);

/// Generate tokens for `seq` starting from `prompt_tokens`. Resets the
/// sequence's KV state before prefill (use `baseRT_sequence_generate_continue`
/// to append to an existing state).
BaseRTGenerationStats baseRT_sequence_generate(baseRT_sequence_t seq, const uint32_t *prompt_tokens, int n_prompt,
                                               int max_tokens, BaseRTSamplingConfig sampling,
                                               baseRT_token_callback callback, void *user_data);

/// Continue generation on `seq` from its current KV state. Appends `new_tokens`
/// without resetting. Mirrors `baseRT_generate_continue` for the multi-seq API.
BaseRTGenerationStats baseRT_sequence_generate_continue(baseRT_sequence_t seq, const uint32_t *new_tokens, int n_new,
                                                        int max_tokens, BaseRTSamplingConfig sampling,
                                                        baseRT_token_callback callback, void *user_data);

/// Release the sequence's blocks back to the pool and free the handle.
void baseRT_sequence_free(baseRT_sequence_t seq);

// baseRT_batch_step_fused_pads / baseRT_batch_warmup /
// baseRT_sequence_rollback are declared once, below with the rest of the
// batched-decode API (their doc blocks had already started drifting apart).

/// Batched decode: drives ONE batched decode step across N sequences. Each
/// sequence writes its `new_tokens[i]` to its own KV cache slot, and attention
/// reads each sequence's KV via its own block table. Throughput comes from
/// batching the otherwise-sequential per-sequence dispatches into one pass.
///
/// Requires the model to be loaded with `--paged-kv`. The N sequences must
/// all belong to the same model handle.
///
/// `n_seqs` must be > 0 and <= `baseRT_set_max_batch_size(n)` (default 1).
/// Set the cap BEFORE baseRT_load_model so scratch.logits can be sized for
/// the B-row output GEMM.
///
/// Output: `out_tokens[i]` receives the argmax token for seq i. The engine
/// dispatches GEMM with M=B at the output projection (one logit row per
/// seq), then argmax_f16_batched (one threadgroup per row) to write all B
/// argmax results to scratch.token_ids[0..B-1], which the API copies into
/// `out_tokens`.
///
/// Returns BASERT_OK on success, an error code otherwise. Errors are reported
/// via baseRT_get_error().
int baseRT_batch_decode_step(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *new_tokens,
                             uint32_t *out_tokens);

/// Multi-step batched decode loop. Calls baseRT_batch_decode_step
/// repeatedly, feeding each step's argmax back as the next step's input
/// for each sequence. Up to `max_steps` per seq; lanes that hit `eos_token`
/// (or whose user callback returns false) retire early and drop out of
/// subsequent batched dispatches, keeping the remaining lanes packed.
///
/// Inputs:
///   seqs[i]         : per-seq handle (assumed prefilled; lengths may differ).
///   first_tokens[i] : the token to feed seq i on step 0 (typically the
///                     last argmax from each seq's prefill).
///   max_steps       : per-seq cap on generated tokens.
///   eos_token       : stop generation for a seq when its argmax equals this.
///                     Pass UINT32_MAX (or any token > vocab_size) to disable.
///   out_tokens      : [n_seqs * max_steps] flat row-major buffer; row i
///                     receives seq i's decoded tokens (length out_lengths[i]).
///   out_lengths     : [n_seqs] per-seq actual decoded length (<= max_steps).
///
/// Returns BASERT_OK on success. The KV state of each seq advances by
/// `out_lengths[i]` positions and is left in a usable state for follow-up
/// calls (sequence_generate_continue, etc.).
int baseRT_batch_decode_loop(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *first_tokens,
                             int max_steps, uint32_t eos_token, uint32_t *out_tokens, int *out_lengths);

/// Mixed prefill+decode batch step. Each sequence i ingests
/// `in_token_counts[i]` new tokens from `in_tokens` (a flat row-major
/// buffer of `sum(in_token_counts)` tokens) and contributes one new argmax
/// output to `out_tokens[i]`. `in_token_counts[i] == 1` is a decode step;
/// `in_token_counts[i] > 1` is a prefill chunk that ingests the prompt
/// continuation before sampling.
///
/// The batch is partitioned into a **prefill subset** (L_i > 1) and a
/// **decode subset** (L_i == 1). Prefill sequences are advanced one at a time
/// through the single-sequence paged path (chunked by `max_prefill_chunk` if
/// needed); the decode subset is then advanced through one batched step. One
/// API call advances both kinds of sequences in the same scheduler tick.
///
/// Prefill and decode lanes are advanced in the same call but are not fused
/// into a single kernel pass; each runs its own dispatch within the step.
///
/// Requires `--paged-kv`. The N sequences must belong to the same model.
/// Decode subset count must be <= `baseRT_set_max_batch_size(n)` (default 1).
///
/// Returns BASERT_OK on success, an error code otherwise. On error,
/// already-advanced seqs are left in whatever state the underlying
/// sub-dispatches left them (the prefill subset advances first, so a
/// decode-subset failure does NOT rollback prefilled seqs).
int baseRT_batch_step(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *in_tokens,
                      const int *in_token_counts, uint32_t *out_tokens);

/// Fused-path variant of baseRT_batch_step: drives ONE unified forward pass
/// across all sequences (instead of the sequential sub-dispatch in
/// baseRT_batch_step). Each seq i contributes `in_token_counts[i]` rows to a
/// packed [sum_L, dim] residual stream with variable-length attention. After
/// the last layer the output stage gathers the last row per sequence, runs a
/// B-row output projection, and takes the argmax per sequence.
///
/// Same C API contract as baseRT_batch_step (B seqs, flat in_tokens,
/// in_token_counts[B], one argmax per seq via out_tokens[B]); same
/// requirements (--paged-kv, all seqs from this model, n_seqs <= max).
/// **NOT supported on every architecture** -- batched VARLEN routing is
/// currently available for Qwen3. Other architectures return a runtime
/// UNSUPPORTED error.
///
/// Returns BASERT_OK on success, BASERT_ERR_UNSUPPORTED if VARLEN attention
/// can't be dispatched at the given head_dim/seq_len (falls back to
/// baseRT_batch_step for those cases).
int baseRT_batch_step_fused(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *in_tokens,
                            const int *in_token_counts, uint32_t *out_tokens);

/// Fused batch step with per-sequence trailing PAD counts (serving grid
/// alignment): counts[i] includes pads[i] throwaway tokens whose rows are
/// computed but whose argmax row is skipped (the output token comes from the
/// last REAL row). The caller must roll each padded sequence's KV back by
/// pads[i] after the call (baseRT_sequence_rollback). Greedy only.
///
/// Backend note: the shape-padding fast path exists for CUDA-graph capture;
/// on backends without stream capture (Metal) the pads are STRIPPED before
/// dispatch — semantics identical (no pad KV is written, so the caller's
/// rollback is a no-op), no wasted compute.
int baseRT_batch_step_fused_pads(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *in_tokens,
                                 const int *in_token_counts, const int *pads, uint32_t *out_tokens);

/// Warm the batched-decode fast paths for batch sizes up to `max_batch`:
/// each B runs throwaway pure-decode ticks so shape-keyed caches (baked
/// dispatch tables, captured graphs, PSO/plan builds) are built at startup
/// instead of on the first real requests — the same boot-time warmup vLLM
/// performs. Requires --paged-kv; call after load, before serving.
int baseRT_batch_warmup(baseRT_model_t model, int max_batch);

/// Max prompt tokens the fused (varlen) prefill can process in one packed batch.
/// The continuous-batching engine caps per-tick admitted prompt tokens by this
/// so a burst of long prompts doesn't overflow the packed prefill. 0 on null.
int baseRT_max_prefill_chunk(baseRT_model_t model);

/// Roll a sequence's KV state back to `length` tokens, returning any blocks
/// past that point to the pool. `length` must be <= the current length; 0
/// resets the sequence to empty. Used by the serving engine's shape-padding
/// dummy lanes (their KV is discarded after every tick).
int baseRT_sequence_rollback(baseRT_sequence_t seq, int length);

/// Multi-step autoregressive driver for baseRT_batch_step_fused. Step 0
/// ingests the mixed-length input from `first_in_tokens` / `first_in_token_counts`
/// (one row per seq, total length `sum(first_in_token_counts)`). Subsequent
/// steps are all-decode L_i = 1 (each lane feeds back its own argmax). Lanes
/// that hit `eos_token` retire early and drop out of subsequent dispatches,
/// keeping the remaining lanes packed.
///
/// Outputs:
///   out_tokens   : [n_seqs * max_steps] flat row-major buffer; row i
///                  receives seq i's decoded tokens (length out_lengths[i]).
///   out_lengths  : [n_seqs] per-seq actual decoded length (<= max_steps).
///
/// Requires `--paged-kv`. Same architecture support as
/// `baseRT_batch_step_fused` (Qwen3, Gemma, Llama 3.2).
int baseRT_batch_step_fused_loop(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs,
                                 const uint32_t *first_in_tokens, const int *first_in_token_counts, int max_steps,
                                 uint32_t eos_token, uint32_t *out_tokens, int *out_lengths);

/// Host-sampling variant of baseRT_batch_step_fused: runs the same unified
/// forward pass but SKIPS the GPU argmax, leaving the per-seq logits ([B, vocab]
/// f16) in the engine's logits scratch. The caller then reads them back with
/// baseRT_read_batch_logits and runs per-sequence sampling / grammar / penalties
/// on the host. Same args/contract as baseRT_batch_step_fused minus out_tokens.
/// Used by the continuous-batching engine for per-request sampling without a
/// per-row GPU sampling kernel.
int baseRT_batch_step_fused_logits(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *in_tokens,
                                   const int *in_token_counts);

/// Read back the [n_seqs, vocab] f16 logits left by the most recent
/// baseRT_batch_step_fused_logits into `out_logits_f16` (n_seqs * vocab halves,
/// row-major). Pure UMA copy, no dispatch. Returns vocab_size, or <0 on error.
/// `n_seqs` must match the batch of the preceding step and be <= max_batch_size.
int baseRT_read_batch_logits(baseRT_model_t model, int n_seqs, void *out_logits_f16);

// === Host-side logits-row operations ===
//
// Companions to baseRT_read_batch_logits for schedulers that sample on the
// host: they encapsulate the engine's logits element type and row layout so
// the caller never casts raw buffers or re-implements dtype-sensitive math
// (which must stay bit-compatible with the engine's own greedy/sampled paths).
// A "row" below is one sequence's logits inside the buffer written by
// baseRT_read_batch_logits: row `s` starts at byte offset
// `s * baseRT_batch_logits_stride(model)`.

/// Bytes between consecutive sequence rows in the baseRT_read_batch_logits
/// output buffer (also the size of one row). Size the readback buffer as
/// `n_seqs * stride` bytes. Returns 0 on a null model.
size_t baseRT_batch_logits_stride(baseRT_model_t model);

/// Apply a grammar bitmask (from baseRT_grammar_fill_bitmask; a SET bit =
/// allowed token) to one logits row IN PLACE: every disallowed token's logit
/// becomes -inf, so subsequent sampling and logprob reads on the row see the
/// constrained distribution. Returns BASERT_OK or an error code.
int baseRT_mask_logits_row(baseRT_model_t model, void *row, const int32_t *bitmask);

/// Run the full CPU sampling pipeline (temperature / top-k / top-p / min-p /
/// repetition + presence + frequency penalties / logit_bias) over one logits
/// row. `prev_tokens`/`n_prev` feed the repetition penalties;
/// `repeat_window` bounds how many trailing prev_tokens are penalized (0 =
/// all). When cfg->seed != 0 the sampling RNG is reseeded with
/// cfg->seed + seed_offset first — pass the per-sequence generated-token
/// count as seed_offset for deterministic per-lane streams under batch
/// interleaving. Returns the sampled token id (0 with the error state set on
/// invalid arguments).
uint32_t baseRT_sample_logits_row(baseRT_model_t model, const void *row, const BaseRTSamplingConfig *cfg,
                                  const uint32_t *prev_tokens, int n_prev, int repeat_window, uint32_t seed_offset);

/// Lowest-index argmax over one logits row — the exact tie-break of the GPU
/// argmax and the CPU greedy fast path, which sampling with top_k=1 does NOT
/// guarantee (equal-logit ties are common with f16 logits). Use this for
/// greedy lanes in a host-sampled batch. Returns 0 with the error state set
/// on invalid arguments.
uint32_t baseRT_argmax_logits_row(baseRT_model_t model, const void *row);

/// Log-softmax over one logits row: writes the chosen token's logprob to
/// *out_token_logprob and the top `top_k` alternatives (ids + logprobs,
/// descending) to out_ids/out_logprobs, which must hold top_k entries.
/// top_k = 0 computes only the chosen token's logprob. Returns the number of
/// alternatives written, or < 0 with the error state set on invalid
/// arguments.
int baseRT_logits_row_logprobs(baseRT_model_t model, const void *row, uint32_t token, int top_k,
                               float *out_token_logprob, uint32_t *out_ids, float *out_logprobs);

// === Prefix cache — scheduler-driven primitives ===
//
// A scheduler (e.g. the continuous-batching BatchEngine) reuses the KV of a
// shared prompt prefix across requests. Per request:
//   1. m = baseRT_prefix_match(model, prompt, n);   // finds the longest cached
//                                                    // block-aligned prefix,
//                                                    // increfs+locks its blocks
//   2. seq = baseRT_sequence_create(model);
//      baseRT_sequence_seed_prefix(seq, m.blocks, m.n_blocks, m.matched_tokens);
//   3. prefill ONLY prompt[m.matched_tokens:] via baseRT_batch_step_fused;
//      decode as usual (attention gathers over shared + new blocks).
//   4. on finish: baseRT_prefix_insert(model, prompt, n, seq); // publish for reuse
//                 baseRT_prefix_unlock(model, m.handle);       // release the lock
// All calls must run under the same exclusive model access as the forward pass
// (the BatchEngine holds its model lock around the whole tick). No-ops / empty
// matches when the prefix cache is disabled.

/// Result of a prefix-cache lookup. `blocks` points into engine-owned storage
/// that stays valid until the matching baseRT_prefix_unlock(handle). The
/// matched blocks have been incref'd for the new sequence's ownership and the
/// matched trie node locked against eviction.
typedef struct {
    int matched_tokens;  ///< block-aligned count of reusable prompt tokens (0 = miss)
    int n_blocks;        ///< number of shared blocks (matched_tokens / page_size)
    const int *blocks;   ///< shared block IDs; valid until baseRT_prefix_unlock(handle)
    uint64_t handle;     ///< pass to baseRT_prefix_unlock; 0 = no match / cache disabled
} BaseRTPrefixMatch;

/// Look up the longest cached block-aligned prefix of `tokens`. Always leaves
/// at least one prompt token to prefill (never matches the entire prompt).
/// On a hit (matched_tokens>0): increfs each shared block for the caller's new
/// sequence and locks the prefix against eviction; release with
/// baseRT_prefix_unlock(handle). On a miss / disabled cache: returns all-zero
/// (handle=0) and there is nothing to unlock.
BaseRTPrefixMatch baseRT_prefix_match(baseRT_model_t model, const uint32_t *tokens, int n_tokens);

/// Seed a freshly-created, empty sequence with the shared blocks from a match
/// so it reuses their KV instead of re-prefilling. `n_tokens` must equal
/// `n_blocks * page_size`. An empty match (n_blocks == 0 && n_tokens == 0) is
/// a successful no-op; any other zero/null combination is rejected with
/// BASERT_ERR_INVALID_ARGUMENT (nothing was adopted — release the match).
/// Returns BASERT_OK, or an error if the model isn't paged / the sequence
/// isn't empty.
int baseRT_sequence_seed_prefix(baseRT_sequence_t seq, const int *blocks, int n_blocks, int n_tokens);

/// Paged-KV block (page) size in tokens for this model, or 0 when the model
/// was not loaded with --paged-kv. This is the block-alignment granularity for
/// baseRT_prefix_match/_seed_prefix (matched_tokens == matched_blocks *
/// page_size); the continuous-batching hybrid-GDN prefix-reuse path uses it to
/// pick the block-aligned GDN snapshot boundary at admit.
int baseRT_page_size(baseRT_model_t model);

/// Publish `seq`'s KV blocks for the block-aligned prefix of `tokens` into the
/// prefix cache so later requests can reuse them. Idempotent for an already-
/// cached prefix (no double refcount). No-op when the cache is disabled.
/// Returns BASERT_OK or an error code.
int baseRT_prefix_insert(baseRT_model_t model, const uint32_t *tokens, int n_tokens, baseRT_sequence_t seq);

/// Release the lock a baseRT_prefix_match took on a prefix and free the match's
/// bookkeeping. Call exactly once per non-zero handle, after the sequence that
/// reused the prefix has been inserted/retired. No-op for handle==0.
///
/// Use this ONLY when the match's blocks WERE seeded into a sequence
/// (baseRT_sequence_seed_prefix): the sequence owns those blocks and drops the
/// match's ownership incref when it resets/frees. If the match was NOT seeded
/// (you decided not to reuse it), call baseRT_prefix_release instead — unlock
/// alone would leak the increfed blocks.
void baseRT_prefix_unlock(baseRT_model_t model, uint64_t handle);

/// Abandon a baseRT_prefix_match WITHOUT seeding it: drops the ownership incref
/// on each matched block (which no sequence adopted) AND releases the trie lock.
/// Call exactly once per non-zero handle when you matched a prefix but chose not
/// to seed it (e.g. a boundary mismatch). No-op for handle==0.
void baseRT_prefix_release(baseRT_model_t model, uint64_t handle);

/// Evict least-recently-used UNLOCKED cached prefixes until at least `n_blocks`
/// block-frees have been performed back to the pool. Returns the number freed
/// (may be < n_blocks if the remaining prefixes are all locked by live
/// sequences). Call when a prefill hits pool exhaustion, then retry the step.
/// No-op (returns 0) when the cache is disabled.
int baseRT_prefix_evict(baseRT_model_t model, int n_blocks);

/// Persist the prefix cache (trie + every cached block's KV) to `path` so a
/// later process can reload the hot prefixes instead of cold-prefilling them.
/// The file is tagged with a model fingerprint (KV shapes + model path); load
/// rejects a file written by a different model. No-op (returns BASERT_OK) when
/// the prefix cache is disabled / empty. Call when no prefix matches are
/// outstanding (e.g. at shutdown). Returns BASERT_OK or an error code.
int baseRT_prefix_cache_save(baseRT_model_t model, const char *path);

/// Load a prefix cache previously written by baseRT_prefix_cache_save, REPLACING
/// the current in-memory cache. Validates magic / version / page_size / model
/// fingerprint; on any mismatch, missing file, or corruption the cache is left
/// empty and an error is returned (so a stale/foreign file never scatters wrong
/// KV). Requires --paged-kv with the prefix cache enabled. Call before serving
/// (no outstanding matches). Returns BASERT_OK or an error code.
int baseRT_prefix_cache_load(baseRT_model_t model, const char *path);

/// Lifetime prefix-cache stats (any out-pointer may be NULL). `hits`/`misses`
/// count baseRT_prefix_match calls that did / didn't reuse >=1 block;
/// `reused_tokens` is the running total of prompt tokens served from cache;
/// `blocks_cached` is the current number of blocks held by the trie.
void baseRT_prefix_cache_stats(baseRT_model_t model, uint64_t *out_hits, uint64_t *out_misses,
                               uint64_t *out_reused_tokens, int *out_blocks_cached);

// === Grammar-constrained decoding ===

/// Opaque grammar handle.
typedef void *baseRT_grammar_t;

/// Create a grammar from a GBNF grammar string.
/// Returns NULL on parse error (check baseRT_get_error()).
baseRT_grammar_t baseRT_grammar_create(baseRT_model_t model, const char *gbnf);

/// Create a grammar from a JSON Schema string.
/// Converts the schema to GBNF internally.
/// Returns NULL on error.
baseRT_grammar_t baseRT_grammar_create_from_schema(baseRT_model_t model, const char *json_schema);

/// Create a grammar for generic JSON output (any valid JSON object/array).
baseRT_grammar_t baseRT_grammar_create_json(baseRT_model_t model);

/// Free a grammar.
void baseRT_grammar_free(baseRT_grammar_t grammar);

/// Reset a grammar's acceptance state back to its initial (post-create)
/// stacks. Lets the caller reuse one grammar handle across multiple
/// independent decodes — e.g. the server's n>1 loop, which otherwise
/// would feed the second sample through a terminated grammar (garbage).
void baseRT_grammar_reset(baseRT_grammar_t grammar);

/// Grammar stepping for the continuous-batching server (xgrammar backend).
/// The server applies the bitmask to a lane's logits row on the host, then
/// accepts the sampled token to advance the grammar. A legacy-NPDA grammar
/// reports `bitmask_size == 0` — the caller must keep it on the serial path.
///   baseRT_grammar_bitmask_size : packed int32 words in the token bitmask
///     (0 = not an xgrammar grammar; use the serial decode path instead).
///   baseRT_grammar_fill_bitmask : fill `out_bitmask` (bitmask_size words) for
///     the CURRENT grammar state; a set bit = allowed token. 1 on success.
///   baseRT_grammar_accept_token : advance the grammar by one token. 1 on ok.
///   baseRT_grammar_is_terminated: 1 once the grammar reaches an end state.
///   baseRT_grammar_is_completed : 1 once a full match is accepted (a
///     structured value is complete). Decoding should stop on terminated OR
///     completed — matching the serial grammar loop.
int baseRT_grammar_bitmask_size(baseRT_grammar_t grammar);
int baseRT_grammar_fill_bitmask(baseRT_grammar_t grammar, int32_t *out_bitmask);
int baseRT_grammar_accept_token(baseRT_grammar_t grammar, uint32_t token_id);
int baseRT_grammar_is_terminated(baseRT_grammar_t grammar);
int baseRT_grammar_is_completed(baseRT_grammar_t grammar);

/// Generate tokens with grammar constraint.
/// Grammar masks invalid tokens at each step, guaranteeing output conforms to the grammar.
BaseRTGenerationStats baseRT_generate_grammar(baseRT_model_t model, const uint32_t *prompt_tokens, int n_prompt,
                                              int max_tokens, BaseRTSamplingConfig sampling, baseRT_grammar_t grammar,
                                              baseRT_token_callback callback, void *user_data);

/// Continue generation with grammar constraint from current KV cache state.
BaseRTGenerationStats baseRT_generate_grammar_continue(baseRT_model_t model, const uint32_t *new_tokens, int n_new,
                                                       int max_tokens, BaseRTSamplingConfig sampling,
                                                       baseRT_grammar_t grammar, baseRT_token_callback callback,
                                                       void *user_data);

// === GPU sampling ===

/// Run a profiled decode step — returns per-layer GPU timing.
/// Runs each layer in its own command buffer for accurate GPU timing.
/// Much slower than normal decode — use only for profiling.
/// timing_out: array of (n_layers + 3) floats [embedding, norm, layer0..N-1, logit, argmax]
/// Returns number of timing entries written, or -1 on a failed step
/// (details via baseRT_get_error).
int baseRT_profile_decode_step(baseRT_model_t model, uint32_t token_id, int position, float *timing_out,
                               int max_entries);

/// Get the kernel label for a profiled entry index.
const char *baseRT_profile_label(baseRT_model_t model, int index);

/// Apply temperature scaling to logits buffer on GPU (in-place).
void baseRT_gpu_temperature_scale(baseRT_model_t model, float temperature);

/// Apply repetition penalty on GPU (in-place on logits).
void baseRT_gpu_repetition_penalty(baseRT_model_t model, const uint32_t *token_ids, int n_tokens, float penalty);

// === Model inspection ===

/// Get number of tensors in model.
int baseRT_tensor_count(baseRT_model_t model);

/// Get tensor name by index. Returns static string.
const char *baseRT_tensor_name(baseRT_model_t model, int index);

/// Get tensor dtype code by index.
uint32_t baseRT_tensor_dtype(baseRT_model_t model, int index);

/// Get the canonical `.base` tensor dtype string at index
/// (e.g. "f16", "bf16", "f32", "base4", "base8", "base_q2"…"base_q8").
/// Returns empty string out of range.
const char *baseRT_tensor_raw_dtype(baseRT_model_t model, int index);

/// Whether the loaded model carries an mmproj sub-bundle (vision/audio
/// tower weights). Returns 0/1.
int baseRT_has_mmproj(baseRT_model_t model);

/// `header.mmproj.arch` tag (e.g. "gemma4_mm"). Returns empty string when
/// the model has no mmproj.
const char *baseRT_mmproj_arch(baseRT_model_t model);

// === Per-phase prefill profiling ===
// Run a single prefill chunk with each major op phase wrapped in its own
// command-buffer sync. Returns the number of unique phase labels recorded.
// Use baseRT_prefill_profile_label / total_ms / count to read back the
// per-label totals (each label is hit once per layer for per-layer
// phases, so `count` is typically n_layers).
int baseRT_profile_prefill(baseRT_model_t model, const uint32_t *tokens, int n_tokens);
int baseRT_prefill_profile_phase_count(baseRT_model_t model);
const char *baseRT_prefill_profile_label(baseRT_model_t model, int index);
float baseRT_prefill_profile_total_ms(baseRT_model_t model, int index);
int baseRT_prefill_profile_count(baseRT_model_t model, int index);

// === Calibration mode ===
// Run prefill in calibration mode: every linear-layer activation gets a
// per-input-channel absmax reduction whose result is keyed by canonical
// tensor name. Output is a JSON sidecar matching the AwqProfile schema
// consumed by `basert convert --awq-profile <path>`.
//
// Usage:
//   baseRT_calibrate_begin(model, "<fingerprint>");
//   for each calibration chunk:
//       baseRT_calibrate_prefill(model, tokens, n_tokens);
//   baseRT_calibrate_save(model, "awq_profile.json");
//
// `fingerprint` may be NULL; when non-null, it is stored in the sidecar's
// `source_fingerprint` field. The converter rejects a profile whose
// fingerprint does not match the source weights at convert time.
int baseRT_calibrate_begin(baseRT_model_t model, const char *fingerprint);
int baseRT_calibrate_prefill(baseRT_model_t model, const uint32_t *tokens, int n_tokens);
int baseRT_calibrate_save(baseRT_model_t model, const char *output_path);
void baseRT_calibrate_end(baseRT_model_t model);

/// Number of tensors under the mmproj sub-bundle. Returns 0 for non-MM bundles.
int baseRT_mmproj_tensor_count(baseRT_model_t model);

/// Tensor name (HF-canonical) at the given mmproj index. Returns empty
/// string out of range.
const char *baseRT_mmproj_tensor_name(baseRT_model_t model, int index);

/// Raw on-disk dtype string for the mmproj tensor at the given index
/// ("base4", "f16", "bf16", "f32", …). Returns empty string out of range.
const char *baseRT_mmproj_tensor_raw_dtype(baseRT_model_t model, int index);

// === Low-level API (for benchmarking) ===

/// Run prefill on tokens. Populates KV cache.
/// Returns the first generated token (argmax of prefill logits).
uint32_t baseRT_prefill(baseRT_model_t model, const uint32_t *tokens, int n_tokens);

/// Read the post-prefill / post-decode logits buffer (predicting the
/// next token after the most recent step) as float into `out`. Source
/// storage is f16 on GPU; this widens to f32 on copy.
/// Returns the number of logits written (== vocab_size on success, 0
/// on error).
int baseRT_read_logits(baseRT_model_t model, float *out, int max_logits);

/// Multimodal prefill: run vision tower on image, then prefill tokens with
/// image features spliced at positions where tokens[i] == config.image_token_id.
/// The number of image placeholder tokens in the stream must equal the image's
/// pooled token count (see baseRT_image_num_tokens).
/// Returns the first generated token, or 0 on error (check baseRT_get_error).
uint32_t baseRT_prefill_image(baseRT_model_t model, const uint32_t *tokens, int n_tokens, const char *image_path);

/// Returns the number of image placeholder tokens produced by the vision tower
/// for an image at `image_path`, or 0 on error. This is the value the caller
/// must use when expanding `<|image|>` placeholders in the prompt.
int baseRT_image_num_tokens(baseRT_model_t model, const char *image_path);

/// Audio prefill: run Conformer audio encoder on PCM samples, splice features
/// into prompt at audio_token_id positions. PCM must be 16kHz mono float32.
/// Returns the first generated token, or 0 on error.
uint32_t baseRT_prefill_audio(baseRT_model_t model, const uint32_t *tokens, int n_tokens, const float *pcm_samples,
                              int n_samples);

/// Returns the number of audio placeholder tokens for the given audio length.
int baseRT_audio_num_tokens(baseRT_model_t model, int n_samples);

/// Run one decode step. Returns sampled token ID.
uint32_t baseRT_decode_step(baseRT_model_t model, uint32_t token_id, int position);

/// Chain decode: generate multiple tokens in one GPU submission.
/// Returns number of tokens generated. Tokens written to out_tokens.
int baseRT_chain_decode(baseRT_model_t model, uint32_t first_token, int start_position, int count,
                        uint32_t *out_tokens);

/// Get current KV cache position (number of tokens processed).
int baseRT_get_position(baseRT_model_t model);

/// Enable/disable speculative decoding (n-gram prediction).
/// Disabled by default. Only affects greedy (temperature=0) mode.
void baseRT_set_speculation(baseRT_model_t model, bool enabled);

/// Read the current speculation flag for the given handle (default: false).
/// Used by the server to scope per-request `speculation: true/false` body
/// overrides without losing the model's prior setting.
bool baseRT_get_speculation(baseRT_model_t model);

/// Reset KV cache and internal state.
void baseRT_reset(baseRT_model_t model);

/// Persist the current KV cache state to `path`. Saves only the
/// `current_length` prefix (not the unused tail), so the file size grows
/// linearly with how much was prefilled+decoded. Returns 0 on success and
/// a negative error code on failure; check `baseRT_get_error` for details.
/// Hybrid linear-attention models (Qwen 3.5/3.6) are REJECTED: the format
/// holds attention KV only, not the Gated-DeltaNet recurrent state.
int baseRT_save_state(baseRT_model_t model, const char *path);

/// Inverse of `baseRT_save_state`. The cache must have been allocated for
/// a model with matching shape; mismatched files are rejected. After load,
/// `baseRT_get_position` reflects the restored token count.
/// Hybrid linear-attention models (Qwen 3.5/3.6) are REJECTED: the file
/// holds attention KV only, and restoring it without the matching
/// Gated-DeltaNet recurrent state would yield a corrupt hybrid state.
int baseRT_load_state(baseRT_model_t model, const char *path);

/// Install a LoRA adapter on this model. The adapter file is a `.base`
/// bundle with tensors named `lora.<canonical>.A` and `lora.<canonical>.B`
/// plus metadata `lora.rank` (int) and `lora.alpha` (float). After load,
/// every forward pass that runs a GEMM with a tensor_name registered in
/// the adapter has a post-GEMM low-rank delta applied (`y += B @ A @ x`).
///
/// Calling `baseRT_lora_load` again replaces the active adapter (no
/// stacking). Returns 0 on success, negative on failure (see
/// `baseRT_get_error`).
int baseRT_lora_load(baseRT_model_t model, const char *path);

/// Detach the active adapter (if any) and free its GPU buffers. No-op if
/// none is loaded.
void baseRT_lora_unload(baseRT_model_t model);

/// Returns the loaded adapter's id (the path it was loaded from), or an
/// empty string when no adapter is active. Valid until the next lora_load
/// / lora_unload / model_free on this handle.
const char *baseRT_lora_id(baseRT_model_t model);

/// Truncate KV cache to `to_position` tokens (drop everything after).
/// Used by the server to roll back generation tokens before reusing the
/// shared chat-template prefix from a prior request — keeps the cached
/// prefill of the common prefix while discarding the prior turn's
/// user-message tail and assistant reply.
/// Hybrid linear-attention models (Qwen 3.5/3.6): the recurrent state
/// cannot be rewound to an arbitrary position. This call keeps its "KV
/// length == to_position" promise only when `to_position` exactly matches
/// the recurrent-state snapshot (see `baseRT_set_prefill_snapshot`); any
/// other position degrades to a FULL reset (equivalent to `baseRT_reset`)
/// — the caller must then prefill the entire prompt again. Use
/// `baseRT_try_rollback` to detect what happened, or to resume from a
/// snapshot that sits before the requested position.
void baseRT_rollback(baseRT_model_t model, int to_position);

/// Rollback that reports the position actually achieved. Non-hybrid
/// models land on `min(to_position, current KV length)` — a target past
/// the cache end cannot be "achieved" by a rollback and is clamped so
/// callers prefilling from the returned position never skip tokens. Hybrid linear-attention models can only resume
/// from their recurrent-state snapshot (see
/// `baseRT_set_prefill_snapshot`): when the snapshot sits at or before
/// `to_position` the state is restored there and the SNAPSHOT position is
/// returned — the caller must prefill the prompt from that position
/// onward. When the snapshot lies past `to_position` (divergent history)
/// the call returns -1 and leaves the model state UNTOUCHED — fall back
/// to `baseRT_reset` + a full prefill. `to_position == 0` always succeeds
/// as a full reset.
int baseRT_try_rollback(baseRT_model_t model, int to_position);

/// Hybrid linear-attention models only (no-op otherwise): ask prompt
/// prefills to capture the reuse snapshot once absolute KV position `pos`
/// has been processed, instead of at the prompt end. Chat servers pass
/// the rendered-history boundary (the prompt minus the generation
/// scaffold): the scaffold tokens never reappear in the next request's
/// render, so a prompt-end snapshot would never match, while the history
/// boundary is exactly where the next request's shared prefix ends.
/// PERSISTENT: stays armed until replaced by the next call (so n>1
/// multi-choice requests re-snapshot the same boundary on every
/// full-prefill choice); pass -1 to clear. Out-of-range values fall back
/// to the prompt-end snapshot. Standalone `baseRT_prefill[_image/_audio]`
/// calls always snapshot at their prompt end (hints apply to
/// generate/generate_continue prefills only).
void baseRT_set_prefill_snapshot(baseRT_model_t model, int pos);

/// Portable GDN reuse-snapshot blob (hybrid linear-attention models only).
/// The engine keeps a single most-recent boundary snapshot; a server-side
/// keyed store keeps several (one per distinct prior prompt) and loads the
/// best prefix match back before baseRT_try_rollback restores it. All three
/// are no-ops / return 0 / -1 on non-hybrid models.
///   baseRT_gdn_snapshot_size    : fixed blob byte length for this model
///     (0 if not a hybrid model). Allocate this much for _capture.
///   baseRT_gdn_snapshot_capture : serialize the CURRENT snapshot (the one a
///     just-completed request's prompt prefill recorded) into `out` (capacity
///     `cap`). Returns bytes written, or -1 if there is no snapshot / cap is
///     too small / not hybrid.
///   baseRT_gdn_snapshot_load    : deserialize `blob` back into the engine's
///     snapshot slot (NOT live state — a following baseRT_try_rollback applies
///     it). Returns the snapshot's KV position, or -1 on a length/model
///     mismatch.
int baseRT_gdn_snapshot_size(baseRT_model_t model);
int baseRT_gdn_snapshot_capture(baseRT_model_t model, uint8_t *out, int cap);
int baseRT_gdn_snapshot_load(baseRT_model_t model, const uint8_t *blob, int len);

/// Per-sequence GDN snapshot (F6 M4: batched continuous-batching prefix reuse).
/// Capture/restore a CB sequence's OWN Gated-DeltaNet lane (its per-lane pool
/// slot) directly to/from a blob — distinct from the model-level snapshot APIs
/// above, which serve the single-sequence path via the lane-0 shadow. The blob
/// uses the same wire format and `baseRT_gdn_snapshot_size` byte length.
///
///   baseRT_sequence_gdn_capture : serialize the sequence's lane state,
///     stamping its current KV length as the resume position. Call it when the
///     lane's state is at the intended (block-aligned) boundary. Returns bytes
///     written, or -1 (not hybrid / bad slot / cap too small).
///   baseRT_sequence_gdn_restore : deserialize `blob` into the sequence's lane
///     LIVE state. Returns the encoded position (the caller then sets the
///     sequence's KV length and prefills the suffix), or -1 on a mismatch.
int baseRT_sequence_gdn_capture(baseRT_sequence_t seq, uint8_t *out, int cap);
int baseRT_sequence_gdn_restore(baseRT_sequence_t seq, const uint8_t *blob, int len);

/// Generate tokens continuing from current KV cache state (no reset).
/// Use for multi-turn chat: prefill new tokens only, then decode.
BaseRTGenerationStats baseRT_generate_continue(baseRT_model_t model, const uint32_t *new_tokens, int n_new,
                                               int max_tokens, BaseRTSamplingConfig sampling,
                                               baseRT_token_callback callback, void *user_data);

// === Embeddings ===

/// Compute text embeddings from token IDs using the model's hidden states.
/// Runs a forward pass and mean-pools the final hidden layer.
/// out_embedding: pre-allocated float array of size at least `dim` (from model config).
/// Returns embedding dimension on success, 0 on failure.
int baseRT_embed(baseRT_model_t model, const uint32_t *tokens, int n_tokens, float *out_embedding, int max_dims);

/// Convenience: embed a text string directly (tokenizes internally).
int baseRT_embed_text(baseRT_model_t model, const char *text, float *out_embedding, int max_dims);

/// Get the embedding dimension for a model.
int baseRT_embedding_dim(baseRT_model_t model);

// === Chat templates ===

/// Format a chat prompt using the model's native template.
/// Returns formatted string (valid until next call or model free).
/// messages: array of role/content pairs as "role\0content\0role\0content\0..." with double null terminator.
const char *baseRT_format_chat(baseRT_model_t model, const char *system_prompt, const char *user_message);

/// Get the chat template name for the loaded model ("chatml", "llama3", "gemma", or "unknown").
const char *baseRT_chat_template(baseRT_model_t model);

/// Raw Jinja chat template from `tokenizer.chat_template` in the .base file —
/// the HF `chat_template.jinja` the converter folded in. Empty string when
/// the bundle has no chat template metadata.
/// Returned pointer valid until the next call on the same thread, or until
/// the model is freed.
const char *baseRT_chat_template_jinja(baseRT_model_t model);

/// BOS / EOS token strings (what minja substitutes for `{{ bos_token }}`
/// and `{{ eos_token }}` in HF chat templates).
const char *baseRT_bos_token(baseRT_model_t model);

/// BOS token id, for callers that need to prepend BOS to raw token
/// sequences (e.g. perplexity windows on BOS-sensitive models).
uint32_t baseRT_bos_id(baseRT_model_t model);
const char *baseRT_eos_token(baseRT_model_t model);

/// Primary end-of-sequence token id (the one the continuous-batching engine and
/// other token-id consumers stop on). Returns 0 on a null handle.
uint32_t baseRT_eos_token_id(baseRT_model_t model);

// (baseRT_max_prefill_chunk is declared once, with the batched-decode API.)

// === Token counting ===

/// Count tokens in text without allocating an output buffer.
int baseRT_token_count(baseRT_model_t model, const char *text);

// === Whisper transcription ===

/// Callback for streaming transcription segments.
/// Called once per segment as it is decoded.
/// start_ms/end_ms: timestamp range in milliseconds.
/// text: segment text (valid only during callback).
/// Return false to stop transcription early.
typedef bool (*baseRT_segment_callback)(int start_ms, int end_ms, const char *text, void *user_data);

/// Transcribe audio from raw float32 PCM samples (16kHz, mono).
/// Returns transcribed text (valid until next call or model free).
/// stats_out: optional, receives timing statistics.
const char *baseRT_transcribe_pcm(baseRT_model_t model, const float *samples, int n_samples,
                                  const char *language,  // "en", "auto", etc. (NULL = "en")
                                  BaseRTTranscribeStats *stats_out);

/// Transcribe with per-segment streaming callback.
/// Same as baseRT_transcribe_pcm but calls segment_callback for each decoded segment.
const char *baseRT_transcribe_pcm_stream(baseRT_model_t model, const float *samples, int n_samples,
                                         const char *language, BaseRTTranscribeStats *stats_out,
                                         baseRT_segment_callback callback, void *user_data);

/// Transcribe audio from a WAV file (resampled to 16kHz internally).
const char *baseRT_transcribe(baseRT_model_t model, const char *wav_path, const char *language,
                              BaseRTTranscribeStats *stats_out);

/// Transcribe WAV file with per-segment streaming callback.
const char *baseRT_transcribe_stream(baseRT_model_t model, const char *wav_path, const char *language,
                                     BaseRTTranscribeStats *stats_out, baseRT_segment_callback callback,
                                     void *user_data);

/// Enable/disable timestamp generation for Whisper transcription.
/// When enabled (default): produces [start --> end] text segments with seeking.
/// When disabled: faster greedy decode, plain text output.
void baseRT_set_timestamps(baseRT_model_t model, bool enabled);

/// Set the Whisper task: "transcribe" (default) or "translate" (any-to-English).
/// NULL resets to "transcribe". Returns false (with baseRT_get_error set) on an
/// unknown task string, or on "translate" with an English-only model.
bool baseRT_set_task(baseRT_model_t model, const char *task);

/// Set an initial prompt for Whisper transcription (reference `initial_prompt`):
/// tokenized and fed after <|startofprev|> ahead of the first window's prompt,
/// biasing the decode (names, spellings, style). NULL or "" clears it.
/// On models whose tokenizer cannot encode text (legacy GGML whisper files
/// without BPE merges) the prompt is ignored with a warning at transcribe time.
void baseRT_set_initial_prompt(baseRT_model_t model, const char *text);

/// condition_on_previous_text (default true, reference semantics): feed each
/// window the previous windows' decoded text after <|startofprev|>. Disable if
/// the model gets stuck in repetition loops on your audio.
void baseRT_set_condition_on_previous_text(baseRT_model_t model, bool enabled);

/// Language code of the LAST transcription: the detected code when the request
/// language was NULL/""/"auto", otherwise the requested code. Empty string if
/// no transcription has run. Valid until the next transcription or model free.
const char *baseRT_transcribe_language(baseRT_model_t model);

/// Duration of the LAST transcription's source audio, in milliseconds
/// (n_samples at 16 kHz — the OpenAI verbose_json `duration` field is this
/// value in fractional seconds). 0 if no transcription has run.
int baseRT_transcribe_audio_duration_ms(baseRT_model_t model);

/// Per-segment metadata for the LAST transcription (verbose_json surface).
/// avg_logprob / no_speech_prob / compression_ratio / temperature are
/// window-level values applied to every segment decoded in that 30 s window
/// (the reference computes them per-decode-result; window-level is this
/// engine's documented approximation for chain-decoded tokens).
typedef struct {
    int start_ms;
    int end_ms;
    const char *text;  // valid until the next transcription or model free
    float avg_logprob;
    float no_speech_prob;
    float compression_ratio;
    float temperature;
} BaseRTTranscribeSegment;

/// Number of segments produced by the last transcription (0 if none).
int baseRT_transcribe_segment_count(baseRT_model_t model);

/// Fetch one segment of the last transcription. Returns false on bad index.
bool baseRT_transcribe_segment(baseRT_model_t model, int index, BaseRTTranscribeSegment *out);

/// Check if loaded model is a Whisper model.
bool baseRT_is_whisper(baseRT_model_t model);

#ifdef __cplusplus
}
#endif
