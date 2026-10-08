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
#define BASERT_VERSION_MINOR 3
#define BASERT_VERSION_PATCH 0

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
///   backend, baseRT.metallib; on CUDA, baseRT_cuda_kernels.fatbin; on ROCm,
///   baseRT_rocm_kernels.hsaco), or NULL to auto-detect. Auto-detect order:
///   $BASERT_KERNEL_LIB (a file, or a directory holding the artifact), a
///   kernel library next to the build, next to the
///   executable, the current directory, then a copy embedded in the loaded
///   binary itself (single-file distributions ship the shared library with the
///   kernels linked in, so NULL just works). Named generically so non-Metal
///   backends (CUDA/ROCm, future) can reuse the same parameter.
/// max_context: maximum context window. 0 = the model's trained window, capped
///   at the chip's max prefill chunk — a shape default that ignores how much
///   memory this device actually has. A serving front end should instead pass
///   `baseRT_suggest_max_context()`, which derives the window from the device
///   budget, or a number the operator chose.
/// Returns NULL on failure.
baseRT_model_t baseRT_load_model(const char *model_path, const char *kernel_library_path, int max_context);

/// Load ONLY the bundle's tokenizer: metadata parse + vocab/merges, no
/// tensor upload, no KV/scratch allocation (a serving daemon tokenizes in
/// its own process while a worker owns the GPU state — basertd §3's codec
/// placement). The returned handle supports exactly the tokenizer surface:
/// baseRT_encode, baseRT_decode_token / _static / _raw, baseRT_token_count,
/// baseRT_eos_token_id, baseRT_bos_id / baseRT_bos_token / baseRT_eos_token,
/// baseRT_get_config, and baseRT_free_model. Calling generation or
/// sequence APIs on it is undefined. NULL on failure (see
/// baseRT_get_error).
baseRT_model_t baseRT_load_tokenizer_only(const char *model_path);

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
/// Hot-swap the kernel library on an already-loaded model, WITHOUT reloading
/// weights. Weights are mmap-backed Metal buffers independent of the kernel
/// library, so only the compiled pipelines + baked decode dispatch tables are
/// rebuilt against the new metallib (sub-second). Built for the automated
/// kernel-tuning loop: edit a `.metal`, `make shaders`, reload — no ~400 GB
/// model reload per iteration. `metallib_path` NULL uses the default sidecar
/// search (`build/baseRT.metallib`). Picks up `.metal`-only changes; a
/// C++/param/dispatch change still needs a rebuilt binary + restart.
/// Returns BASERT_OK, or an error code (leaves the model on the OLD library on
/// load failure). Not supported for whisper models. NOT thread-safe against
/// in-flight inference — serialize the caller.
///
/// INVALIDATES ALL DECODED STATE. Everything in a KV or recurrent slab was
/// produced by the old kernels, so resuming on it would mix kernel versions —
/// the one thing a tuning comparison must not do. On success the default
/// sequence's KV, recurrent state and prefix cache are reset for you (position
/// returns to 0). Caller-owned sequences (baseRT_sequence_create) are NOT
/// reachable from here and must be destroyed and recreated; using one across a
/// reload is undefined.
int baseRT_reload_metallib(baseRT_model_t model, const char *metallib_path);

/// A JSON capability descriptor for this loaded model (schema field
/// `descriptor_version`): workload, modalities, serving, state, adaptation,
/// and tick-op support, each capability either `true` or a stable reason
/// string explaining why it is unavailable on this architecture + backend +
/// load-options triple. Computed once at load (verdicts frozen, like the
/// multi-row overcommit verdict) — immutable, race-free, never touches the
/// error state. The pointer is owned by the model handle and valid until
/// baseRT_free_model. Supersedes the five-bit mask above for every question
/// it cannot answer; the mask stays for cheap scheduler branching.
const char *baseRT_capability_descriptor(baseRT_model_t model);

/// Override the KV cache element width for the next baseRT_load_model call.
///   bits = 0  → auto (per-model default; Q8_0 when head_dim%32==0)
///   bits = 8  → force Q8_0 K and V slabs (1.88x smaller; tiny precision cost)
///   bits = 16 → force F16 K and V slabs (memory-heavier; full precision)
/// Process-wide; persists across loads. Must be called before
/// baseRT_load_model. Other values are ignored.
void baseRT_set_kv_bits(int bits);

/// The KV cache's element width as ACTUALLY allocated for `model`, in the
/// `baseRT_set_kv_bits` encoding: 16 (f16), 8 (Q8_0), 4 (Q4_0), 84 (K at
/// Q8_0, V at Q4_0), or 0 for a model with no KV cache. Differs from what
/// was set whenever the loader overrode it — hybrid GDN / SSM, gpt-oss and
/// MLA models force f16; a paged pool is uniform (84 → 8); a head_dim that
/// is not a multiple of 32 cannot take the quantized blocks — and the load
/// printed one `[baseRT] <path>: --kv-bits N requested, ...` line saying
/// why. Serve this rather than the setting when reporting KV precision.
int baseRT_kv_bits_effective(baseRT_model_t model);

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

/// The device memory budget automatic context sizing plans within, in bytes,
/// without needing a loaded model. NOT installed RAM: on Metal this is 85% of
/// the unified-memory working set the OS recommends (itself a fraction of RAM
/// the OS picks per machine, roughly two thirds to three quarters) — the share
/// the engine keeps its pinned weights within, since past it Metal evicts from
/// the pinned set and reads slow down or go corrupt; on CUDA the device's total
/// memory. 0 when no supported device is present.
///
/// This is the number `baseRT_suggest_max_context*` and
/// `baseRT_context_window_fits` size against. It is a planning line, not a
/// hard limit: a load whose working set lands between it and the full working
/// set still succeeds, with the tail of its weights left to page-cache LRU (and
/// the load warns once the working set passes the full budget).
size_t baseRT_device_memory_budget(void);

/// Floor on a window derived by `baseRT_suggest_max_context`. Policy, not a
/// hardware limit: an explicit max_context is honoured below it. There is no
/// corresponding ceiling — a derived window is bounded by the device's memory
/// and by the model's trained window, and by nothing else.
enum {
    BASERT_AUTO_CONTEXT_MIN = 4096,
};

/// Suggest a `max_context` for `model_path` that fits this device: the memory
/// budget above, less the weights and activation/scratch headroom, divided by
/// what one token of KV costs across `max_batch` concurrent decode lanes. The
/// result is a multiple of 1024, at least BASERT_AUTO_CONTEXT_MIN, and never
/// above the model's trained window — a device with the memory for a model's
/// whole window gets the whole window.
///
/// Reads the bundle's metadata only — no tensor upload, no allocation — so it
/// is cheap enough to call before `baseRT_load_model`. A serving front end
/// calls it once at startup instead of shipping a fixed default that is too
/// small on a workstation and too large on a laptop.
///
/// max_batch: concurrent sequences the KV pool must hold (0 or 1 = single).
///            Only counted for models the load will actually page, since a
///            contiguous cache holds one history however wide the batch is.
/// kv_bits:   the value that will be passed to `baseRT_set_kv_bits` (0 = auto,
///            or 4 / 8 / 16 / 84), since KV precision changes the answer.
/// paged_kv:  non-zero if the load will enable paged KV (`baseRT_set_paged_kv`
///            or BaseRTLoadOptions::paged_kv). It changes both the per-layer
///            shape and whether lanes multiply, and the two caches differ by
///            several times on the hybrid decoders.
/// Returns 0 if the bundle cannot be read, the device budget is unknown, or the
/// bundle declares no trained window; the caller keeps its own default then.
int baseRT_suggest_max_context(const char *model_path, int max_batch, int kv_bits, int paged_kv);

/// As above for a set of models that will be resident AT THE SAME TIME, such
/// as a server's eagerly loaded set. They share one budget, so their weights
/// add up, their KV pools add up, and the window returned is the one that fits
/// all of them at once, capped at the SHORTEST trained window among them.
///
/// This is not the same as calling the single-model form on each and taking
/// the minimum: that answers "what fits this model alone", and two models that
/// each fit alone can exceed the budget together.
int baseRT_suggest_max_context_multi(const char *const *model_paths, int n_models, int max_batch, int kv_bits,
                                     int paged_kv);

/// As `baseRT_suggest_max_context_multi`, plus the speculators that will load
/// beside those models. A drafter runs at its target's window and lane count
/// with its own paged KV pool, so it spends the same budget: leaving it out
/// sizes the targets' pools into memory the drafter then cannot get.
/// `speculator_embedded[i]` non-zero marks speculator i as its target
/// bundle's own embedded head (header.speculator, e.g. MTP): its KV pool only,
/// its weights being part of the bundle. Zero — or a null array — is a
/// separately loaded bundle (a DFlash / DSpark / EAGLE-3 sidecar or a draft
/// model): its weights and its KV pool, even if the same file is also one of
/// `model_paths` (a draft model is loaded again beside the served copy).
/// `speculator_target[i]` >= 0 marks speculator i as a sidecar loaded with
/// `baseRT_load_drafter` against `model_paths[speculator_target[i]]`: that
/// loader reopens the target's bundle for its token embedding / lm_head, and a
/// backend that copies wrapped weights (CUDA's UMA path) holds those rows a
/// second time, so they are charged too. -1 — or a null array — is a
/// speculator with no such view (a draft model loads on its own); it is
/// ignored for an embedded head, whose view of its own bundle is always charged.
/// Speculators never cap the window, one that cannot be read is skipped, and
/// none is charged unless `paged_kv` (their loaders require a paged target).
int baseRT_suggest_max_context_spec(const char *const *model_paths, int n_models, const char *const *speculator_paths,
                                    const int *speculator_embedded, const int *speculator_target, int n_speculators,
                                    int max_batch, int kv_bits, int paged_kv);

/// Does `window` fit `model_paths` co-resident on this device, under the same
/// budget the suggestion above derives from? 1 = yes, 0 = no, -1 = cannot tell
/// (a bundle would not open, no trained window, unknown budget).
///
/// Distinct from comparing against `baseRT_suggest_max_context_multi`: that
/// clamps its answer into the policy floor, so a set that fits NO tokens at all
/// still reports the floor and would compare equal to it. Ask this instead
/// before admitting a model into a window that was chosen without it.
int baseRT_context_window_fits(const char *const *model_paths, int n_models, int max_batch, int kv_bits, int paged_kv,
                               int window);
/// KV-cache allocation strategy reported by BaseRTMemoryStats.
typedef enum BaseRTKVCacheLayout {
    BASERT_KV_CACHE_NONE = 0,
    BASERT_KV_CACHE_CONTIGUOUS = 1,
    BASERT_KV_CACHE_PAGED = 2,
} BaseRTKVCacheLayout;

/// Runtime-owned memory counters for an idle model handle. Capacity is the
/// memory reserved for the KV cache; used bytes are the logical occupied
/// portion (contiguous cache) or occupied physical blocks (paged cache).
/// Neither value is process RSS, device-global usage, nor a peak.
typedef struct BaseRTMemoryStats {
    uint64_t runtime_allocated_bytes;
    uint64_t kv_cache_capacity_bytes;
    uint64_t kv_cache_used_bytes;
    uint64_t kv_cache_blocks_total;
    uint64_t kv_cache_blocks_used;
    /// Tokens resident in the KV. Exact on a contiguous cache. On a PAGED cache
    /// this is block-granular — `kv_cache_blocks_used * page_size`, so each
    /// sequence's partially-filled last block counts whole — because the pool
    /// tracks blocks, not per-sequence lengths, and the default lane's own
    /// length reads 0 whenever batched sequences own the live state.
    uint64_t kv_cache_tokens_used;
    BaseRTKVCacheLayout kv_cache_layout;
    uint32_t reserved;
} BaseRTMemoryStats;

/// Read current runtime and KV-cache memory counters. This is a boundary
/// observation: call only while no inference operation is mutating `model`.
/// Returns false for a null model or output pointer.
bool baseRT_model_memory_stats(baseRT_model_t model, BaseRTMemoryStats *out_stats);

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
///
/// A substring equal to one of the model's special tokens (`<|im_end|>`,
/// `<think>`, `<tool_call>`, …) becomes that single control id — what a
/// rendered chat prompt needs, since its framing IS those markers.
int baseRT_encode(baseRT_model_t model, const char *text, uint32_t *out_tokens, int max_tokens);

/// Encode text with special-token strings treated as ordinary text.
///
/// The same substrings tokenize to their BPE pieces instead of control ids,
/// so a message body, tool result or document that QUOTES a marker cannot
/// forge a turn boundary or open a reasoning block. Use this for client
/// content and `baseRT_encode` for a dialect's own framing. No automatic
/// BOS-dedup either: a leading BOS string here is content. Returns number
/// of tokens written.
int baseRT_encode_plain(baseRT_model_t model, const char *text, uint32_t *out_tokens, int max_tokens);

/// Encode a sequence of text pieces as ONE text.
///
/// Piece `i` is framing when `plain[i]` is zero (marker strings parse to
/// control ids, as in `baseRT_encode`) and client content otherwise (they
/// stay ordinary text, as in `baseRT_encode_plain`). Unlike encoding the
/// pieces one by one, the pre-tokenizer and BPE see the concatenation, so a
/// boundary between two pieces tokenizes exactly as it does in a whole-text
/// encode of the same characters (a content piece starting with a newline
/// after a framing newline still merges into the `\n\n` token) and the
/// automatic BOS is handled once at the front. A chat codec renders a turn
/// with this: the role line and the wire around a message are framing, the
/// message is content. Returns number of tokens written.
int baseRT_encode_pieces(baseRT_model_t model, const char *const *texts, const int *plain, int n_pieces,
                         uint32_t *out_tokens, int max_tokens);

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

/// A caller-owned incremental-decode stream: the same UTF-8 assembly and
/// channel-protocol normalization `baseRT_decode_token` applies (gpt-oss
/// Harmony and Muse framing become reasoning spans / ChatML tool-call
/// blocks), but with state private to this stream. One per continuously
/// batched lane; the model's own stream is untouched.
typedef struct baseRT_decode_stream *baseRT_decode_stream_t;
baseRT_decode_stream_t baseRT_decode_stream_create(baseRT_model_t model);
void baseRT_decode_stream_reset(baseRT_decode_stream_t stream);

/// Reset a stream for a lane whose generation resumes after `prompt`, deriving
/// the channel resume state from its last framing token. Prefer this over
/// `baseRT_decode_stream_reset` on any lane that may carry a RAW (unframed)
/// prompt: the plain reset assumes a chat prompt, which ends inside a
/// `<|start|>assistant` header, and on a Muse vocabulary an unframed prompt
/// left in that state has every generated token buffered as protocol until a
/// 128-byte valve trips — a short completion returns nothing at all. Harmony
/// has its own fail-open path and is unaffected either way. `prompt` NULL or
/// `n_prompt` 0 behaves exactly like `baseRT_decode_stream_reset`.
void baseRT_decode_stream_reset_for_prompt(baseRT_decode_stream_t stream, const uint32_t *prompt, int n_prompt);
/// Returns a pointer into the stream's own buffer, valid until the next
/// call on the same stream.
const char *baseRT_decode_stream_token(baseRT_decode_stream_t stream, uint32_t token_id);
void baseRT_decode_stream_free(baseRT_decode_stream_t stream);

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
/// dummy lanes (their KV is discarded after every tick) and by speculative
/// verification (drop the rejected drafts). Recurrent-state models: a
/// hybrid-GDN lane rolls back only to a length inside its last
/// baseRT_batch_step_fused_logits_rows feed (the recurrence is rebuilt from
/// that feed's captured rows — deterministic, and bit-identical to a feed
/// that stopped there under bit-exact routing (BASERT_SPEC_BITEXACT=1);
/// the default speculative routing picks its small-M GEMM kernel by row
/// count, so the replayed prefix carries the longer feed's last-ulp
/// rounding instead); `length == current` is a no-op commit on EVERY
/// lane (it drops a hybrid-GDN lane's capture); any shorter target on a
/// Mamba-2 lane, or outside the captured feed on a hybrid-GDN lane,
/// returns BASERT_ERR_UNSUPPORTED — a no-op rollback is not a probe of
/// rollback support.
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

/// Per-row sampling spec for baseRT_batch_step_fused_sample. temperature <= 0
/// samples greedily (argmax) for that row; otherwise Gumbel-max over the
/// top-k / top-p / min-p survivors with `seed` driving the noise stream
/// (same seed + same logits = same token).
///
/// Penalties ARE applied for a sampled row, over `history` — the caller's
/// recent-token window for that row — with the model's penalty-exemption mask
/// honoured, so a sampled row no longer has to leave the fused path to get
/// one. Logit bias and grammar still take the host logits path.
typedef struct BaseRTRowSampling {
    float temperature;
    float top_p;              ///< 1.0 = disabled
    float min_p;              ///< 0.0 = disabled
    int32_t top_k;            ///< <= 0 = disabled
    uint32_t seed;            ///< Gumbel noise seed for this row
    float repeat_penalty;     ///< 1.0 (or 0.0) = disabled
    float presence_penalty;   ///< 0.0 = disabled
    float frequency_penalty;  ///< 0.0 = disabled
    /// Recent tokens this row's penalties are computed over, most recent last.
    /// NULL / 0 disables the penalties for the row regardless of the values
    /// above: with no history there is nothing to penalize.
    const uint32_t *history;
    uint32_t history_count;
} BaseRTRowSampling;

/// Fused batch step that samples every output row ON THE GPU per `rows[i]`
/// (greedy rows argmax, sampled rows Gumbel-max) and returns one token per
/// seq in `out_tokens` — no full-vocab logits readback and no per-row host
/// sampling. Same forward as baseRT_batch_step_fused_logits (so every
/// architecture that serves host logits serves this), followed by one
/// gumbel_topk_f16 dispatch per sampled row and one batched argmax.
/// Returns BASERT_ERR_UNSUPPORTED when the backend lacks the sampling
/// kernels (caller falls back to the logits path).
int baseRT_batch_step_fused_sample(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *in_tokens,
                                   const int *in_token_counts, const BaseRTRowSampling *rows, uint32_t *out_tokens);

/// The sampling pass of baseRT_batch_step_fused_sample on its own: samples
/// logits rows [first_row, first_row + n_rows) that the last fused forward
/// left in the host logits buffer (baseRT_batch_step_fused_logits_rows /
/// _rows_taps — a speculative verify's rows), `rows[i]` driving row
/// first_row + i, one token per row into `out_tokens`. The pass noises the
/// rows in place: read them back first (baseRT_read_batch_logits) if the raw
/// values are still needed. BASERT_ERR_UNSUPPORTED when the backend lacks the
/// sampling kernels or the penalty windows do not fit — the rows are untouched
/// then and the caller samples them on the host.
int baseRT_sample_logits_rows(baseRT_model_t model, int first_row, int n_rows, const BaseRTRowSampling *rows,
                              uint32_t *out_tokens);

/// Multi-row variant for speculative verification (basertd P2.4): seq i
/// contributes in_token_counts[i] input rows and leaves its LAST
/// out_rows[i] rows' logits (1 <= out_rows[i] <= in_token_counts[i]) in the
/// host logits buffer, ascending, seqs in order — sum(out_rows) rows total,
/// read back with baseRT_read_batch_logits(sum(out_rows)). The host checks
/// each drafted position against the target's logits from ONE forward and
/// rolls the sequence back to the accepted length. Served on the fused
/// VARLEN path and on the staged decomposition (MoE, hybrid GDN, residency
/// overcommit) alike; Mamba-2 hybrids return BASERT_ERR_UNSUPPORTED (no
/// per-lane SSM rollback). On the staged route a lane's rows must fit in
/// its last prefill chunk; a hybrid-GDN lane's whole feed must be one
/// chunk of at most 31 tokens, and the lane's next call must be either
/// baseRT_sequence_rollback (to any length inside the feed) or a plain step
/// (which commits the feed).
/// sum(out_rows) is bounded by the logits scratch (max_batch_size × 8).
int baseRT_batch_step_fused_logits_rows(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs,
                                        const uint32_t *in_tokens, const int *in_token_counts, const int *out_rows);

/// Hidden-state taps for speculative drafters (MTP heads, EAGLE-3, DFlash
/// condition on the target's residual stream). Same step as
/// baseRT_batch_step_fused_logits_rows, plus: for each of the `n_tap_layers`
/// requested hidden_states indices in `tap_layers` (0 = embedding output,
/// k = residual stream after decoder layer k-1, n_layers = final pre-norm
/// hidden), the LAST tap_rows[i] fed rows of lane i are copied into the
/// engine's tap buffer, rows in lane order, one contiguous [rows, dim] f16
/// block per tap (read back with baseRT_read_tap_rows, or consumed on the
/// device by an in-engine drafter). At most 8 distinct tap layers and
/// baseRT_tap_rows_capacity() tapped rows per call; a lane's tapped rows
/// must fit in its last prefill chunk. A 1-token lane with tap_rows 1 runs
/// its step serially (the batched decode tables cannot tap). tap_rows may
/// name 0 rows for lanes that need no taps. Taps are emitted by the llama
/// family, Gemma 4, Qwen3.5, gpt-oss and GLM-DSA encoders; a tapped step on any other
/// architecture returns BASERT_ERR_UNSUPPORTED (BASERT_ERR_OUT_OF_MEMORY
/// when the tap buffer cannot be allocated).
/// Greedy speculative verification without a logits readback: the same
/// multi-row forward as baseRT_batch_step_fused_logits_rows_taps (taps
/// optional: n_tap_layers = 0), every requested row argmaxed on the GPU;
/// `out_tokens` receives sum(out_rows) tokens in lane order. Row i of a
/// lane is the target's greedy choice after the lane's i-th fed token —
/// accept draft i while it equals row i. BASERT_ERR_UNSUPPORTED on bundles
/// that verify through host logits (MoE / hybrid / non-llama output
/// stages): fall back to the logits variant.
int baseRT_batch_step_fused_rows_argmax(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs,
                                        const uint32_t *in_tokens, const int *in_token_counts, const int *out_rows,
                                        const int *tap_layers, int n_tap_layers, const int *tap_rows,
                                        uint32_t *out_tokens);

int baseRT_batch_step_fused_logits_rows_taps(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs,
                                             const uint32_t *in_tokens, const int *in_token_counts, const int *out_rows,
                                             const int *tap_layers, int n_tap_layers, const int *tap_rows);

/// Rows one tapped call may cover (min(max_prefill_chunk, 1024)); <0 on a
/// null model.
int baseRT_tap_rows_capacity(baseRT_model_t model);

/// Tapped step with an INPUT OVERRIDE: `input_f16` ([sum(in_token_counts),
/// dim] f16, rows in feed order) replaces the embedding lookup — the
/// drafters' input is a projection of the target's hidden state, not a
/// token. Token ids still drive positions and KV. Rows are bounded by
/// baseRT_tap_rows_capacity(). Feeding a step's own tap-0 rows back through
/// this call reproduces its logits bit for bit.
int baseRT_batch_step_fused_logits_rows_taps_input(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs,
                                                   const uint32_t *in_tokens, const int *in_token_counts,
                                                   const int *out_rows, const int *tap_layers, int n_tap_layers,
                                                   const int *tap_rows, const void *input_f16);

// ── Speculator heads (bundle `header.speculator`) ──────────────────────

/// 1 when the loaded bundle carries a speculator sub-bundle (an MTP head or
/// a converted drafter), 0 otherwise.
int baseRT_has_speculator(baseRT_model_t model);

/// The speculator's `kind` ("mtp", ...) into `out`; returns its length, 0
/// when the bundle has no speculator, <0 on error.
int baseRT_speculator_kind(baseRT_model_t model, char *out, int cap);

/// Whether a loaded speculator head can actually run the MTP prologue: the
/// prologue is an encoder's embedding stage — the Qwen3.5 hybrid encoder's
/// (`qwen35_mtp_prologue`) or GLM-DSA's nextn one (`glm_mtp_prologue`) — so
/// `baseRT_mtp_chain` and the per-step path both refuse a head on any other
/// architecture. A scheduler should ask this
/// before granting an MTP strategy rather than registering one whose every
/// proposal then falls back. 1 = supported, 0 = not, <0 = not a head.
int baseRT_speculator_head_supported(baseRT_model_t head);

/// Whether this bundle's encoder can emit hidden-state taps. Every block
/// drafter and EAGLE-3 sidecar consumes target taps, so a scheduler should
/// ask before granting one of those strategies: without taps the first
/// tapped forward fails with BASERT_ERR_UNSUPPORTED, which the tick reports
/// as a generation failure rather than falling back to plain decoding.
/// 1 = taps available, 0 = not, <0 = bad argument.
int baseRT_model_supports_taps(baseRT_model_t model);

/// Load the bundle's speculator head as its own model handle: the head's
/// decoder layer(s) and norms from the speculator section, the target's
/// embeddings and lm_head shared. It is a paged model of the target's
/// context with its own sequences (baseRT_sequence_create), stepped with
/// the tapped/override steps above; free it with baseRT_free_model before
/// the target. NULL (with an error) when the bundle has no loadable head.
baseRT_model_t baseRT_load_speculator_head(baseRT_model_t target);

/// MTP head input rows (host convenience for tests and drivers): out[r] =
/// fc_embed · rmsnorm(embed(next_tokens[r])) + fc_hidden · rmsnorm(hidden[r])
/// (GLM-DSA nextn heads: eh_proj · [enorm(embed) | hnorm(output_norm(hidden))])
/// — `hidden_f16` the target's final pre-norm hidden ([rows, dim], tap
/// n_layers) at each row's position, `out_f16` [rows, dim] to feed the head
/// as its input override. The head predicts the token AFTER next_tokens[r].
int baseRT_mtp_prologue(baseRT_model_t head, const uint32_t *next_tokens, int rows, const void *hidden_f16,
                        void *out_f16);

/// The whole MTP-head proposal of `n_seqs` head lanes in ONE encoding (a
/// GLM-DSA head runs it as staged per-step forwards, same contract):
/// each lane's committed rows (`counts[i]` of them, 1..15; `tokens` the
/// NEXT token of every row packed lane by lane, `hidden_f16` the target's
/// final pre-norm hidden of every row, [sum counts][dim] f16) go through
/// the prologue and one head forward, then `n_draft - 1` chained steps of
/// one row per lane draft from the head's own output hidden and previous
/// draft, without a host round trip between steps. `out_drafts` receives
/// n_seqs * n_draft ids, lane-major (`out_drafts[i * n_draft + s]` = lane
/// i's draft s; draft 0 is the head's prediction after the last committed
/// row). Each lane's head sequence ends `counts[i] + n_draft - 1`
/// positions longer; roll the chained positions back before the next
/// commit. Requires the head's --paged-kv context; n_draft 1..15.
int baseRT_mtp_chain(baseRT_model_t head, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *tokens,
                     const int *counts, const void *hidden_f16, int n_draft, uint32_t *out_drafts);

/// Copy tap `tap_index` (its position in the call's tap_layers) of the most
/// recent tapped step into `out_f16`: n_rows * dim halves, row-major, rows
/// in lane order as fed. Bounded by that step: `tap_index` below its tap
/// count and `n_rows` at most its tapped rows (BASERT_ERR_INVALID_ARGUMENT
/// otherwise). Returns dim, or <0 on error.
int baseRT_read_tap_rows(baseRT_model_t model, int tap_index, int n_rows, void *out_f16);

/// Block drafters (DFlash / DSpark): load a drafter SIDECAR bundle (arch
/// "dflash" / "dspark", converted from the drafter checkpoint) against
/// `target`. The drafter shares the target's token embedding and lm_head
/// (resolved from the target's bundle when the sidecar lacks them), runs at
/// the target's paged/lane shape, and is stepped with baseRT_drafter_step
/// on its own sequences (baseRT_sequence_create). Free it with
/// baseRT_free_model before the target. NULL with an error when the sidecar
/// does not fit the target (dim / vocab / taps).
baseRT_model_t baseRT_load_drafter(baseRT_model_t target, const char *path);

/// Drafter geometry: `kind` ("dflash" | "dspark"), the block width, the
/// mask token the block is padded with, the first block row whose logits
/// are a draft (DFlash 1: row 0 is the anchor; DSpark 0), and the TARGET
/// hidden_states indices to tap (target_layer_ids + 1) in the order the
/// drafter's context expects them. Returns the tap count, or <0.
int baseRT_drafter_info(baseRT_model_t drafter, char *kind, int kind_cap, int *block_size, uint32_t *mask_token_id,
                        int *logits_start, int *tap_layers, int tap_cap);

/// One drafter step on `seq`: append `n_ctx` context rows — the target's
/// tapped hidden rows at the next `n_ctx` positions, `ctx_taps_f16` laid
/// out [n_taps][n_ctx][dim] (tap-major, the baseRT_drafter_info order;
/// each tap block is exactly what baseRT_read_tap_rows returns) — then,
/// when `block_len > 0`, run the block `block[0..block_len)` (the last
/// committed token followed by mask tokens) after the context with
/// bidirectional block attention and leave its `block_len` logits rows for
/// baseRT_read_batch_logits (row j = block position j; drafts are rows
/// logits_start.. — the argmax of row j predicts the token at block
/// position j for DSpark, or the token replacing mask j for DFlash). The
/// block's KV is dropped; the sequence ends at its context length. Either
/// half may be empty (context-only ingestion of a prompt slice, or a block
/// on an up-to-date context). n_ctx <= baseRT_tap_rows_capacity(drafter);
/// `block_len` is 0 or exactly the block width (the drafters are trained
/// on a fixed-width bidirectional block — a narrower block is refused).
int baseRT_drafter_step(baseRT_model_t drafter, baseRT_sequence_t seq, const void *ctx_taps_f16, int n_ctx,
                        const uint32_t *block, int block_len);

/// DSpark heads carried by the sidecar: `markov_rank` (0 = no Markov head)
/// and whether a confidence head is present.
int baseRT_drafter_heads(baseRT_model_t drafter, int *markov_rank, int *has_confidence);

/// DSpark Markov head, one block row: add `markov.w2 · markov.w1[prev]` to
/// row `row` of the last block step's logits (in place) and return its
/// argmax in `out_token` — the draft for that row given the token before
/// it (`prev` = the pending token for the first drafted row, then the
/// previous draft). Sequential: call per row in order after
/// baseRT_drafter_step.
int baseRT_drafter_markov_argmax(baseRT_model_t drafter, int row, uint32_t prev, uint32_t *out_token);

/// DSpark confidence head over the last block step's rows 0..rows: the
/// probability that block row r's draft survives verification, given the
/// token before it (`prev_tokens[r]`). Adaptive block length keeps drafts
/// while the running product stays above a threshold. Returns `rows`, or
/// <0 (BASERT_ERR_UNSUPPORTED without a confidence head).
int baseRT_drafter_confidence(baseRT_model_t drafter, const uint32_t *prev_tokens, int rows, float *out_conf);

/// EAGLE-3 heads (a sidecar of arch "eagle3", loaded with baseRT_load_drafter):
/// one llama layer drafting the next token from (the current token, a
/// feature of the target's hidden states at three layers). `tap_layers`
/// are the TARGET hidden_states indices to tap (3), `draft_vocab` the
/// reduced vocabulary the head predicts over. Returns the tap count.
int baseRT_eagle3_info(baseRT_model_t head, int *draft_vocab, int *tap_layers, int tap_cap);

/// One head step over `rows` positions from the sequence's length:
/// `tokens[r]` is the token that FOLLOWED position r (the token the row
/// conditions on), and exactly one of `taps_f16` ([3][rows][dim], the
/// target's tapped rows at those positions, baseRT_read_tap_rows layout) or
/// `hidden_f16` ([rows][dim], the head's own output hidden — chaining) is
/// the feature. Afterwards baseRT_eagle3_draft gives row r's draft (the
/// argmax over the draft vocabulary mapped to a target id) and
/// baseRT_eagle3_read the raw rows.
int baseRT_eagle3_step(baseRT_model_t head, baseRT_sequence_t seq, const uint32_t *tokens, int rows,
                       const void *taps_f16, const void *hidden_f16);
int baseRT_eagle3_read(baseRT_model_t head, int rows, void *logits_f16, void *hidden_f16);
int baseRT_eagle3_draft(baseRT_model_t head, int row, uint32_t *out_target_token);

/// The whole proposal of `n_seqs` head sequences in one call: each lane's
/// `counts[i]` committed rows (tokens packed lane by lane; `taps_f16` the
/// target's tapped rows for ALL packed rows, [3][sum rows][dim]) as one
/// forward, then `n_draft - 1` chained one-row steps per lane on the
/// device (the previous draft is the token, the head's own output hidden
/// the feature) with no host round trip between them. `out_drafts[i *
/// n_draft + s]` is lane i's draft s (target ids). The head sequences end
/// `counts[i] + n_draft - 1` positions longer; roll back to drop the
/// chained positions. n_draft in 1..15.
int baseRT_eagle3_chain(baseRT_model_t head, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *tokens,
                        const int *counts, const void *taps_f16, int n_draft, uint32_t *out_drafts);

/// Media variant (basertd P2.7): the packed prompt's `image_token_id`
/// placeholders (exactly `n_rows` of them, in order) take rows from the
/// engine-owned features buffer produced by the media tower (`feats_buffer`
/// is a `baseRT::Buffer*` obtained through the tick media surface). Fused
/// VARLEN live path only (hybrid/MoE/overcommit return UNSUPPORTED). Host
/// sampled: read the logits back with baseRT_read_batch_logits.
int baseRT_batch_step_fused_logits_media(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs,
                                         const uint32_t *in_tokens, const int *in_token_counts, void *feats_buffer,
                                         int n_rows, uint32_t image_token_id, int grid_h, int grid_w);

/// Read back the [n_rows, vocab] f16 logits left by the most recent
/// baseRT_batch_step_fused_logits(_rows) into `out_logits_f16` (n_rows * vocab
/// halves, row-major). Pure UMA copy, no dispatch. Returns vocab_size, or <0
/// on error. `n_rows` is the preceding step's batch (one row per seq) or its
/// sum(out_rows), bounded by the logits scratch (max_batch_size × 8 rows).
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

/// Drop a baseRT_prefix_match taken only to LOOK at what the cache holds (a
/// metadata probe, e.g. basertd's space_match): releases the pin like
/// baseRT_prefix_release but settles nothing, so the probe counts as neither
/// a hit nor a miss in baseRT_prefix_cache_stats. A match that was a reuse
/// decision and was declined belongs in baseRT_prefix_release. No-op for
/// handle==0.
void baseRT_prefix_discard(baseRT_model_t model, uint64_t handle);

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

/// Lifetime prefix-cache stats (any out-pointer may be NULL), all monotonic.
/// A baseRT_prefix_match that found nothing counts as a miss at once; one that
/// found blocks settles later, as a hit at baseRT_prefix_unlock (its blocks
/// were seeded) or as a miss at baseRT_prefix_release (abandoned); one
/// dropped by baseRT_prefix_discard (a probe) is in neither, as is an
/// outstanding match. `reused_tokens` is the running total of
/// prompt tokens served from cache by settled hits;
/// `blocks_cached` is the current number of blocks held by the trie.
void baseRT_prefix_cache_stats(baseRT_model_t model, uint64_t *out_hits, uint64_t *out_misses,
                               uint64_t *out_reused_tokens, int *out_blocks_cached);

// === Boundary-snapshot blobs (hybrid GDN prefix reuse) ===
//
// A hybrid (Gated-DeltaNet) model's cached KV blocks are only half of the
// state a warm seed needs: the recurrence at the SAME position is the other
// half. These calls attach an opaque per-boundary blob (a
// baseRT_sequence_gdn_capture snapshot) to a cached prefix at an exact
// block-aligned depth, discover the deepest attached boundary along a
// prompt's match path, and fetch the blob back for a
// baseRT_sequence_gdn_restore after seeding the KV. The cache stores bytes;
// validity-at-position is the caller's contract. Attachments ride node
// eviction (a dropped prefix drops its blobs), live under their own byte
// budget (BASERT_PREFIX_BLOB_BUDGET_MB, default 512, oldest evicted first),
// and are NOT persisted by baseRT_prefix_cache_save.

/// Attach `blob` at exactly the block-aligned depth `n_tokens` of a prefix
/// already published with baseRT_prefix_insert (insert first, then attach).
/// Replaces an existing attachment at that depth. Best-effort: a disabled
/// cache, an uncovered/misaligned path, or a blob over budget is a no-op.
/// Returns BASERT_OK (attached or clean no-op) or an argument error.
int baseRT_prefix_attach_blob(baseRT_model_t model, const uint32_t *tokens, int n_tokens, const void *blob,
                              int blob_len);

/// The deepest attached-blob depth (in tokens) along the exact-match path of
/// `tokens`, capped at `n_tokens`. 0 = none / cache disabled. Read-only (no
/// lock, no LRU update): use it to cap a hybrid match at a boundary a seed
/// can actually restore.
int baseRT_prefix_blob_match(baseRT_model_t model, const uint32_t *tokens, int n_tokens);

/// Copy the blob attached at exactly depth `n_tokens` into `out` (capacity
/// `cap`). Returns the blob's byte length, or -1 when no blob is attached at
/// that exact depth / the buffer is too small / the cache is disabled.
int baseRT_prefix_blob_fetch(baseRT_model_t model, const uint32_t *tokens, int n_tokens, void *out, int cap);

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

/// Create a grammar from an xgrammar STRUCTURAL TAG (a JSON document, not a
/// grammar string).
///
/// A structural tag constrains only the parts of the output that matter: free
/// text until a trigger appears, then the matching tag's schema until its end
/// tag, then free text again. This is what lets a model choose whether to call
/// a tool while guaranteeing any call it makes is well-formed and delimited —
/// which a whole-output schema grammar cannot express.
///
/// Returns NULL on error (malformed tag, or a schema inside it that cannot be
/// compiled); the caller should decode unconstrained rather than fail.
baseRT_grammar_t baseRT_grammar_create_from_structural_tag(baseRT_model_t model, const char *tag_json);

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
///     the CURRENT grammar state; a set bit = allowed token. Returns 1 when
///     the mask excludes some token and is to be applied, 0 when there is
///     nothing to apply: every token may follow (free text in a structural
///     tag, where the model's markers are text as they are unconstrained),
///     or the grammar has no matcher. Not an error indicator.
///   baseRT_grammar_accept_token : advance the grammar by one token. 1 on ok.
///   baseRT_grammar_is_terminated: 1 once the grammar reaches an end state.
///   baseRT_grammar_is_completed : 1 once a full match is accepted (a
///     structured value is complete). Decoding should stop on terminated OR
///     completed — matching the serial grammar loop. A STRUCTURAL-TAG
///     grammar never reports completed: free text is admitted around (and,
///     unless at_least_one, instead of) its tagged regions, so "could end
///     here" holds at every free-text step and is no reason to stop — those
///     lanes end on EOS (or terminated).
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

/// `baseRT_generate_resume` with a grammar constraint (see there): prefill
/// `[n_cached, n_prompt)` over the kept KV, seed history from the whole prompt.
BaseRTGenerationStats baseRT_generate_grammar_resume(baseRT_model_t model, const uint32_t *prompt_tokens, int n_prompt,
                                                     int n_cached, int max_tokens, BaseRTSamplingConfig sampling,
                                                     baseRT_grammar_t grammar, baseRT_token_callback callback,
                                                     void *user_data);

// === GPU sampling ===

/// Run a profiled decode step — returns per-layer GPU timing.
/// Runs each layer in its own command buffer for accurate GPU timing.
/// Much slower than normal decode — use only for profiling.
/// timing_out: array of (n_layers + 3) floats [embedding, norm, layer0..N-1, logit, argmax]
/// Returns number of timing entries written, or -1 on a failed step
/// (details via baseRT_get_error).
/// Per-kernel profile of one baked BATCHED decode round (see the .cpp
/// comment). Requires a prior baked batched round at this B. Labels via
/// baseRT_profile_batched_label(model, n_seqs, index).
int baseRT_profile_batched_decode(baseRT_model_t model, baseRT_sequence_t *seqs, int n_seqs, const uint32_t *in_tokens,
                                  float *timing_out, int max_entries);
const char *baseRT_profile_batched_label(baseRT_model_t model, int n_seqs, int index);

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

/// Byte size of the tensor's storage blob at index (weights only — MLX
/// companion `.scales`/`.biases` tensors are separate entries with their
/// own sizes). Returns 0 out of range.
size_t baseRT_tensor_nbytes(baseRT_model_t model, int index);

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

/// Sliding-window teacher-forced perplexity of `tokens[0..n_tokens)` on this
/// model. For each start position (advanced by `stride`, up to `max_positions`;
/// <=0 means unbounded), prefills the preceding `window` tokens with a fresh KV
/// cache and accumulates -log P(true next token); PPL = exp(mean NLL). Leaves
/// the model's KV cache reset. Writes exp(mean NLL) to `*out_ppl` and the number
/// of scored positions to `*out_positions` (either may be NULL). This is the
/// exact loop the `baseRT_ppl` tool runs, exposed so a resident server can gate
/// accuracy without a second model load. Returns BASERT_OK or an error code.
/// NOT thread-safe against concurrent inference on the same model.
int baseRT_perplexity(baseRT_model_t model, const uint32_t *tokens, int n_tokens, int window, int stride,
                      int max_positions, double *out_ppl, int *out_positions);

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
/// While an adapter is active, decode skips the baked dispatch-table replay
/// (the table carries no delta dispatches, and one recorded WITH them would go
/// stale on unload) and takes the immediate encode instead — correct output at
/// reduced throughput. Speculative decode is likewise disabled for the
/// duration.
///
/// The batched multi-sequence entry points DO apply the adapter:
/// `baseRT_batch_decode_step`, `baseRT_batch_step` and the fused/logits
/// variants each install it for the pass, bypass baked replay and command
/// bucketing while it is active, and emit the delta GEMMs per layer. (This
/// paragraph previously said the opposite, which would lead a caller to
/// disable batching or write a serial fallback it does not need.)
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

/// Resume generation over a KV the caller kept: the KV holds exactly the first
/// `n_cached` tokens of `prompt_tokens` (e.g. after baseRT_try_rollback to a
/// shared prefix), and only `[n_cached, n_prompt)` is prefilled. Unlike
/// `baseRT_generate_continue`, which sees only the tokens it is given, the
/// channel decoder and the sampling history (repetition / presence /
/// frequency penalties) are seeded from the WHOLE prompt, so the run samples
/// as a cold `baseRT_generate(prompt_tokens, n_prompt)` would. When the KV
/// length is not `n_cached`, or `n_cached` leaves no token to prefill, it runs
/// the prompt cold instead. `prompt_tokens` in the returned stats counts the
/// whole prompt.
BaseRTGenerationStats baseRT_generate_resume(baseRT_model_t model, const uint32_t *prompt_tokens, int n_prompt,
                                             int n_cached, int max_tokens, BaseRTSamplingConfig sampling,
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

/// True for a BERT-style encoder-only embedding model (no autoregressive
/// decode, no paged KV) — the serving worker uses this to allow loading such a
/// model despite it having no paged decode KV.
bool baseRT_is_embedding_model(baseRT_model_t model);

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

/// Number of added/special tokens the model declares (chat scaffold,
/// reasoning/tool channel markers, etc.). Pairs with
/// `baseRT_special_token` so a codec can learn a model's markers from the
/// model itself rather than assuming a hardcoded dialect.
int baseRT_special_token_count(baseRT_model_t model);

/// The `index`-th special token's string; writes its token id (or
/// UINT32_MAX if the string is not a single vocab entry) to `id_out`.
/// Returns "" for an out-of-range index. Returned pointer valid until the
/// next call on the same thread.
const char *baseRT_special_token(baseRT_model_t model, int index, uint32_t *id_out);

/// BOS / EOS token strings (what minja substitutes for `{{ bos_token }}`
/// and `{{ eos_token }}` in HF chat templates).
const char *baseRT_bos_token(baseRT_model_t model);

/// BOS token id, for callers that need to prepend BOS to raw token
/// sequences (e.g. perplexity windows on BOS-sensitive models).
uint32_t baseRT_bos_id(baseRT_model_t model);

/// 1 when the tokenizer prepends BOS to encoded text (the model was trained
/// with a leading BOS), 0 when it does not (byte-level BPE families such as
/// GPT-2 / o200k). baseRT_bos_id can still name a token in the 0 case;
/// callers that synthesize a sequence start (perplexity windows) must key
/// on this, not on bos_id being valid.
int baseRT_add_bos(baseRT_model_t model);
const char *baseRT_eos_token(baseRT_model_t model);

/// Primary end-of-sequence token id (the one the continuous-batching engine and
/// other token-id consumers stop on). Returns 0 on a null handle.
uint32_t baseRT_eos_token_id(baseRT_model_t model);

/// Content-derived weights/bundle identity: FNV-1a-64 over the model
/// file's first MiB, stamped at load (basertd §4 materialization
/// identity input — two fine-tunes of one architecture never share a
/// version_tag). 0 on a null handle or unidentified weights.
uint64_t baseRT_weights_identity(baseRT_model_t model);

/// Whether `token` is ANY of the model's stop tokens — the primary
/// `eos_token_id` plus the extra ids the loader registers (`<|eot|>`,
/// `<|eot_id|>`, `<end_of_turn>`, gpt-oss's `<|call|>` and `<|endoftext|>`,
/// …). Multi-EOS models (Muse Glimmer:
/// `<|eot|>` ends the turn while `eos_token_id` only ends the document)
/// need this; comparing against `baseRT_eos_token_id` alone never
/// terminates their chat turns. Returns 0 on a null handle.
int32_t baseRT_is_eos_token(baseRT_model_t model, uint32_t token);

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
