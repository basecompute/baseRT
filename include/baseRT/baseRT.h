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
#define BASERT_VERSION_MINOR 4
#define BASERT_VERSION_PATCH 1

/// Compile-time version, packed as `(MAJOR<<16) | (MINOR<<8) | PATCH`.
/// Useful for `#if BASERT_VERSION >= 0x000200` feature checks.
#define BASERT_VERSION ((BASERT_VERSION_MAJOR << 16) | (BASERT_VERSION_MINOR << 8) | BASERT_VERSION_PATCH)

/// Runtime-resolved version string ("0.1.0"). Matches the linked
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

/// Override the KV cache element width for the next baseRT_load_model call.
///   bits = 0  → auto (per-model default; Q8_0 when head_dim%32==0)
///   bits = 8  → force Q8_0 K and V slabs (1.88x smaller; tiny precision cost)
///   bits = 16 → force F16 K and V slabs (memory-heavier; full precision)
/// Process-wide; persists across loads. Must be called before
/// baseRT_load_model. Other values are ignored.
void baseRT_set_kv_bits(int bits);

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

/// Free all resources associated with a model.
void baseRT_free_model(baseRT_model_t model);

// === Model info ===

/// Get model configuration.
BaseRTModelConfig baseRT_get_config(baseRT_model_t model);

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

// === Generation ===

/// Callback for streaming token output.
/// Return false to stop generation.
typedef bool (*baseRT_token_callback)(uint32_t token_id, const char *text, void *user_data);

/// Generate tokens from a prompt.
/// Returns generation statistics.
BaseRTGenerationStats baseRT_generate(baseRT_model_t model, const uint32_t *prompt_tokens, int n_prompt, int max_tokens,
                                      BaseRTSamplingConfig sampling, baseRT_token_callback callback, void *user_data);

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
/// `n_blocks * page_size`. Returns BASERT_OK, or an error if the model isn't
/// paged / the sequence isn't empty.
int baseRT_sequence_seed_prefix(baseRT_sequence_t seq, const int *blocks, int n_blocks, int n_tokens);

/// Publish `seq`'s KV blocks for the block-aligned prefix of `tokens` into the
/// prefix cache so later requests can reuse them. Idempotent for an already-
/// cached prefix (no double refcount). No-op when the cache is disabled.
/// Returns BASERT_OK or an error code.
int baseRT_prefix_insert(baseRT_model_t model, const uint32_t *tokens, int n_tokens, baseRT_sequence_t seq);

/// Release the lock a baseRT_prefix_match took on a prefix and free the match's
/// bookkeeping. Call exactly once per non-zero handle, after the sequence that
/// reused the prefix has been inserted/retired. No-op for handle==0.
void baseRT_prefix_unlock(baseRT_model_t model, uint64_t handle);

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
/// Returns number of timing entries written.
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
int baseRT_save_state(baseRT_model_t model, const char *path);

/// Inverse of `baseRT_save_state`. The cache must have been allocated for
/// a model with matching shape; mismatched files are rejected. After load,
/// `baseRT_get_position` reflects the restored token count.
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
void baseRT_rollback(baseRT_model_t model, int to_position);

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
const char *baseRT_eos_token(baseRT_model_t model);

/// Primary end-of-sequence token id (the one the continuous-batching engine and
/// other token-id consumers stop on). Returns 0 on a null handle.
uint32_t baseRT_eos_token_id(baseRT_model_t model);

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

/// Check if loaded model is a Whisper model.
bool baseRT_is_whisper(baseRT_model_t model);

#ifdef __cplusplus
}
#endif
