// baseRT_tick.h — the basertd engine ABI additions (basertd §10 items 1–3, 8).
//
// Normative reference: docs/design/basertd-tick-schema.md (draft 7).
// This header is the in-process ABI between a Rust engine-worker and the
// linked libbaseRT: raw pointers and C strings are legal here; nothing in
// this header crosses a process boundary (the IPC frame format is a
// separate, serialized schema — schema doc §4).
//
// Layout rules (normative):
//   - Every struct is 8-byte aligned with EXPLICIT padding; no field relies
//     on implicit compiler padding. Layouts are identical across the
//     compilers we target; basertd-abi's layout probe test enforces parity
//     with the Rust #[repr(C)] mirrors.
//   - Top-level structs carry `struct_size` and grow append-only.
//   - Every record array is a BaseRTArray; decoding follows the four
//     normative rules in schema §1.1 (zero-init + min(elem_size, local)
//     copy; writer zero-pads; elem_size < min-prefix rejects the plan;
//     8-byte alignment and overflow/bounds checks before any element read).

#ifndef BASERT_TICK_H
#define BASERT_TICK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// From baseRT.h; duplicated guard so this header is self-sufficient.
typedef void *baseRT_model_t;
typedef struct baseRT_sequence_s *baseRT_sequence_t;

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

typedef int32_t baseRT_status;

enum {
    BASERT_STATUS_OK = 0,

    // --- plan rejection class: tick refused, no state change (§1.5) ---
    BASERT_REJECT_BAD_STRIDE = -100,          // elem_size < record min-prefix
    BASERT_REJECT_BAD_ALIGNMENT = -101,       // array data not 8-byte aligned
    BASERT_REJECT_BOUNDS = -102,              // count*elem_size overflow / arena OOB
    BASERT_REJECT_BAD_REF_GENERATION = -103,  // stale ring/buffer generation
    BASERT_REJECT_UNKNOWN_LANE = -104,
    BASERT_REJECT_DUPLICATE_LANE = -105,       // lane_tag twice in one array
    BASERT_REJECT_ILLEGAL_COMBINATION = -106,  // admit+retire, commit on non-HOST, ...
    BASERT_REJECT_BUDGET = -107,               // budget exceeds engine capacity
    BASERT_REJECT_STALE_SEED = -108,           // seed handle expired/consumed
    BASERT_REJECT_HOST_RULES = -110,           // HOST lane: max_new_tokens != 1,
                                               // strategy without HOST_COMPATIBLE,
                                               // decode with uncommitted logits
    BASERT_REJECT_TRANSFER_LOCKED = -111,      // sequence has in-flight state op
    BASERT_REJECT_STALE_NONCE = -112,          // commit's logits_nonce != emitted row
    BASERT_NEEDS_REPLAN = -113,                // shed envelope cannot serve the plan
    BASERT_REJECT_BAD_STRUCT = -114,           // struct_size below v1 minimum

    // --- synchronous op errors ---
    BASERT_EBUSY = -120,  // (sequence, space) already claimed
    BASERT_ERR_BUFFER_TOO_SMALL = -121,
    BASERT_ERR_STALE_SIZING = -122,       // sizing_gen no longer current
    BASERT_ERR_OUT_OF_BOUNDARY = -123,    // trim off a snapshot boundary
    BASERT_ERR_ENVELOPE_MISMATCH = -124,  // restore/promote envelope vs target
    BASERT_ERR_CHECKSUM = -125,
    BASERT_ERR_IDENTITY_MISMATCH = -126,  // compatibility identity mismatch
    // Named OP_UNSUPPORTED (not ERR_UNSUPPORTED) because baseRT.h's
    // BaseRTErrorCode already claims that identifier; the two headers
    // compile together inside the engine. Wire value is unchanged.
    BASERT_ERR_OP_UNSUPPORTED = -127,        // per-kind rule (e.g. lossy blob demote)
    BASERT_ERR_REGISTRATION_REFUSED = -128,  // fail-closed strategy registration
    BASERT_ERR_UNKNOWN_HANDLE = -129,        // unknown op/seed/sequence handle
    BASERT_ERR_SEED_UNSERVABLE = -130,       // acquire: a space cannot serve prefix
    BASERT_ERR_RING_LAYOUT = -131,           // rings_attach: header/size mismatch

    // --- admission-result codes (schema §2.2): per-lane outcomes reported
    //     in admit_results while the tick executes — deliberately OUTSIDE
    //     the plan-rejection range, because a lane whose configuration
    //     matches no certificate is REJECTED in that tick's admit_results,
    //     never a whole-plan refusal ---
    BASERT_REJECT_CERT_UNMATCHED = -140,  // no certificate at/above floor

    // --- fatal ---
    BASERT_ERR_FATAL = -1,  // process presumed dead after this
};

// Per-lane fault codes (BaseRTLaneFault.code — schema §1.5: "lane stops
// with typed code; tick continues"). Distinct vocabulary from
// baseRT_status: these are recoverable, per-lane, and never imply plan
// rejection or state rollback.
enum {
    BASERT_FAULT_UNSPECIFIED = 0,
    BASERT_FAULT_GRAMMAR_OVERFLOW = 1,  // grammar stack overflow
    BASERT_FAULT_REF_SPAN_OOB = 2,      // ref span out of bounds mid-lane
    BASERT_FAULT_STRATEGY = 3,          // speculation-strategy fault
    BASERT_FAULT_OOM_TENTATIVE = 4,     // tentative-namespace OOM (rollback taken)
    BASERT_FAULT_INTERNAL = 5,          // engine-internal, lane-recoverable
    BASERT_FAULT_MEDIA = 6,             // media bind/placeholder mismatch, split run
};

// ---------------------------------------------------------------------------
// Common value types
// ---------------------------------------------------------------------------

// Stride-aware record array (schema §1.1). `elem_size` is the WRITER's
// sizeof(record); `data` must be 8-byte aligned.
typedef struct BaseRTArray {
    const void *data;
    uint32_t count;
    uint32_t elem_size;
} BaseRTArray;  // 16 bytes

// Span of token ids in a shm ring: (ring_id, index, generation) triple plus
// length. Stale generations are detectable, never dereferenceable.
typedef struct BaseRTTokenRef {
    uint32_t ring_id;
    uint32_t index;  // first slot in the ring
    uint32_t count;  // number of token ids
    uint32_t _pad0;
    uint64_t generation;
} BaseRTTokenRef;  // 24 bytes

// One row in a logits shm ring. The 64-bit generation doubles as the
// row's logits_nonce (schema §1.3: one identity, no separate field).
typedef struct BaseRTRingRef {
    uint32_t ring_id;
    uint32_t index;
    uint64_t generation;
} BaseRTRingRef;  // 16 bytes

// Caller-owned host buffer. Caller allocates, caller frees, always; loaned
// to the engine from op submission to terminal state (schema §3.2).
typedef struct BaseRTBuf {
    void *ptr;
    uint64_t len;
} BaseRTBuf;  // 16 bytes

typedef struct BaseRTTokenRange {  // [start, end)
    uint64_t start;
    uint64_t end;
} BaseRTTokenRange;  // 16 bytes

// ---------------------------------------------------------------------------
// Enums (carried as fixed-width integers in records)
// ---------------------------------------------------------------------------

enum {  // BaseRTLaneAdmit.sampling
    BASERT_SAMPLING_GPU_GREEDY = 0,
    BASERT_SAMPLING_GPU_GUMBEL = 1,
    BASERT_SAMPLING_HOST = 2,
};

enum {  // determinism_class
    BASERT_DETERMINISM_BEST_EFFORT = 0,
    BASERT_DETERMINISM_DETERMINISTIC = 1,
};

// Exactness classes, ordered so `minimum_exactness` floors compare
// numerically (basertd §4: approximate < distribution-exact < path-invariant).
enum {
    BASERT_EXACTNESS_APPROXIMATE = 0,
    BASERT_EXACTNESS_DISTRIBUTION_EXACT = 1,
    BASERT_EXACTNESS_SEED_PATH_INVARIANT = 2,
};

enum {  // BaseRTTickPlan.flags
    BASERT_TICK_DRAIN = 1u << 0,
    BASERT_TICK_CAPTURE_OK = 1u << 1,
    // The plan carries a live `on_partial_emit` callback (appended fields —
    // struct_size-gated): the tick invokes it as lanes produce tokens, so
    // the host can stream them out MID-tick instead of waiting for the
    // tick's terminal emit. The terminal emit stays authoritative and
    // unchanged; partial emissions are display-path deltas.
    BASERT_TICK_PARTIAL_EMITS = 1u << 2,
};

enum {  // BaseRTTickEvents.tick_status
    BASERT_TICK_OK = 0,
    BASERT_TICK_SHED = 1,
    BASERT_TICK_LANE_ERRORS = 2,
    BASERT_TICK_FATAL = 3,
};

enum {                             // BaseRTAdmitResult.status
    BASERT_ADMIT_UNSPECIFIED = 0,  // absent (older writer)
    BASERT_ADMIT_ADMITTED = 1,
    BASERT_ADMIT_REJECTED = 2,
};

enum {  // BaseRTLaneEmit.finish
    BASERT_FINISH_NONE = 0,
    BASERT_FINISH_EOS = 1,
    BASERT_FINISH_LENGTH = 2,
    BASERT_FINISH_GRAMMAR = 3,
    BASERT_FINISH_CANCELLED = 4,
    BASERT_FINISH_ERROR = 5,
};

enum {  // BaseRTOpStatus.state
    BASERT_OP_PENDING = 0,
    BASERT_OP_RUNNING = 1,
    BASERT_OP_DONE = 2,
    BASERT_OP_FAILED = 3,
    BASERT_OP_CANCELLED = 4,
};

enum {  // demote/snapshot encoding
    BASERT_ENCODING_LOSSLESS = 0,
    BASERT_ENCODING_Q8 = 1,
    BASERT_ENCODING_Q4 = 2,
};

enum {  // state-space kinds (§7.2)
    BASERT_SPACE_KIND_UNSPECIFIED = 0,
    BASERT_SPACE_KIND_PAGED_TOKEN_KV = 1,
    BASERT_SPACE_KIND_RING_KV = 2,
    BASERT_SPACE_KIND_RECURRENT_BLOB = 3,
    BASERT_SPACE_KIND_DEPTH_PAGED_KV = 4,
    BASERT_SPACE_KIND_ENCODER_CACHE = 5,
};

enum {  // BaseRTStateSpaceDesc.flags
    // The space exists in the model (its state is real and evolves under
    // ticks) but NO state op serves it yet: seeds cannot pin it, snapshot/
    // demote/restore/trim refuse BASERT_ERR_OP_UNSUPPORTED. A seed or park
    // covering only the OTHER spaces would rebuild PARTIAL state, so the
    // engine refuses seed acquisition on such bundles and the daemon must
    // neither park nor resume them (basertd phase 2, P2.0). Cleared per
    // space as its ops land (P2.5 for RecurrentBlob).
    BASERT_SPACE_FLAG_OPS_UNAVAILABLE = 1u << 0,
    // The space's state is NOT served by the reuse cache: a cache-based
    // seed (baseRT_seed_acquire over published prefixes) can never cover
    // it, so a bundle carrying such a space admits warm ONLY through
    // adopted sequences (baseRT_seed_adopt over a restored sequence) — the
    // hybrid GDN blob in v1 (no boundary-snapshot cache yet; basertd
    // phase 2, P2.5). Cache seeds refuse SEED_UNSERVABLE on such bundles.
    BASERT_SPACE_FLAG_NO_PREFIX_CACHE = 1u << 1,
};

enum {  // BaseRTArtifact.role
    BASERT_ARTIFACT_UNSPECIFIED = 0,
    BASERT_ARTIFACT_DRAFT_WEIGHTS = 1,
    BASERT_ARTIFACT_AUX_HEAD = 2,
    BASERT_ARTIFACT_PROJECTION = 3,
    BASERT_ARTIFACT_DRAFT_TOKENIZER = 4,
};

enum {  // BaseRTTapSpec.tensor
    BASERT_TAP_TENSOR_UNSPECIFIED = 0,
    BASERT_TAP_TENSOR_HIDDEN = 1,
    BASERT_TAP_TENSOR_ATTN_OUT = 2,
    BASERT_TAP_TENSOR_EMBED = 3,
};

// Certificate applicability: sampling-mode bitset (schema §2.2).
enum {
    BASERT_CERT_MODE_GREEDY = 1u << 0,
    BASERT_CERT_MODE_GUMBEL = 1u << 1,
    BASERT_CERT_MODE_HOST = 1u << 2,
};

// Certificate parameter-domain. 0 matches nothing (fail-closed default).
enum {
    BASERT_CERT_DOMAIN_NONE = 0,
    BASERT_CERT_DOMAIN_GREEDY_ONLY = 1,   // temperature == 0 only
    BASERT_CERT_DOMAIN_PENALTY_FREE = 2,  // any temp, no penalties
    BASERT_CERT_DOMAIN_ANY = 3,
};

enum {                                        // strategy capability kinds:
    BASERT_STRATEGY_CAP_UNSPECIFIED = 0,      // an engine that does not
    BASERT_STRATEGY_CAP_PROPOSAL_LINEAR = 1,  // recognize a kind_id REFUSES
    BASERT_STRATEGY_CAP_PROPOSAL_BLOCK = 2,   // registration (fail-closed)
    BASERT_STRATEGY_CAP_PROPOSAL_TREE = 3,
    BASERT_STRATEGY_CAP_MASK_ANCESTOR = 4,    // tree ancestor-only masks
    BASERT_STRATEGY_CAP_HOST_COMPATIBLE = 5,  // verification consumes no sampler
};

enum {  // provenance taint reason bits
    BASERT_TAINT_QUANTIZED_DEMOTION = 1u << 0,
    BASERT_TAINT_DECOMPOSITION = 1u << 1,  // decomposition-changing restore
};

enum {  // resident tier (match results)
    BASERT_TIER_GPU = 0,
    BASERT_TIER_HOST = 1,
    BASERT_TIER_GONE = 2,  // metadata only; re-prefill
};

enum {  // BaseRTShedEntry.kind
    BASERT_SHED_UNSPECIFIED = 0,
    BASERT_SHED_PREFILL_CHUNK = 1,
    BASERT_SHED_CACHE_EVICTION = 2,
};

enum {  // fork flags
    BASERT_FORK_EAGER = 1u << 0,
};

enum {  // BaseRTStateEnvelope.checksum_kind
    // 0 is reserved-invalid: payload checksum validation on restore and
    // promote is UNCONDITIONAL (schema §3.2 — corruption is detected, not
    // trusted away); an envelope without a recognized checksum kind is a
    // typed refusal, never a skipped check.
    BASERT_CHECKSUM_XXH3_64 = 1,
    // FNV-1a-64 over the payload (the canonical portable checksum the
    // native engine writes; see the identity spec in basertd_tick.cpp).
    BASERT_CHECKSUM_FNV1A64 = 2,
};

// ---------------------------------------------------------------------------
// §1.2 Tick plan
// ---------------------------------------------------------------------------

typedef struct BaseRTSamplingParams {
    float temperature;
    float top_p;
    float min_p;
    uint32_t top_k;  // 0 = disabled
    float freq_penalty;
    float presence_penalty;
    float repeat_penalty;  // 0 = disabled (means 1.0)
    uint32_t flags;        // BASERT_SAMPLING_FLAG_* (was _pad0; 0 = none)
} BaseRTSamplingParams;    // 32 bytes

enum {  // BaseRTSamplingParams.flags
    // The lane does not finish on an end-of-sequence token: it decodes to
    // its max_new_tokens like any other token. A benchmark's fixed output
    // length (vLLM's `ignore_eos`); unlike a logit bias on the stop set it
    // leaves the lane on the GPU sampling and speculation paths, so what is
    // measured is the path real traffic takes.
    BASERT_SAMPLING_FLAG_IGNORE_EOS = 1u << 0,
};

// Shedding envelope: the engine may shed/evict ONLY within this envelope
// (schema §1.2). Shared verbatim by baseRT_cache_evict.
typedef struct BaseRTShedPolicy {
    BaseRTArray victim_lanes;          // of uint64_t lane_tag, eviction order
    uint64_t evictable_cache_classes;  // bitset over daemon-defined classes
    uint64_t protected_quota_bytes;
    uint64_t max_evict_bytes;
} BaseRTShedPolicy;  // 40 bytes

typedef struct BaseRTLaneAdmit {
    uint64_t lane_tag;  // opaque; never interpreted
    BaseRTTokenRef prompt;
    uint64_t seed_handle;               // leased via baseRT_seed_acquire; 0 = cold
    uint32_t sampling;                  // BASERT_SAMPLING_*
    uint32_t logit_bias_handle;         // 0 = none; created out-of-band (was _pad0)
    BaseRTSamplingParams params;        // used by GPU_GUMBEL; recorded regardless
    uint64_t rng_counter_base;          // counter-based RNG start (basertd §4)
    uint32_t grammar_handle;            // 0 = none; created out-of-band
    uint32_t strategy_slot;             // registered speculation slot; 0 = off
    uint8_t determinism_class;          // BASERT_DETERMINISM_*
    uint8_t minimum_exactness;          // floor for strategy certificates
    uint8_t allow_approximate;          // explicit consent; default 0 rejects
    uint8_t want_logprobs;              // 1 = emit each token's logprob (was _pad1)
    uint32_t required_cert_id;          // 0 = engine selects; else exact match
    uint8_t host_sampler_identity[32];  // HOST lanes only; zero = cert-ineligible
    // Appended (min-prefix stays 136, the v1 size): a preempted lane's
    // continuation resumes its grammar where the generation stopped. The
    // prompt's last `grammar_replay` tokens are what the grammar already
    // accepted; the engine replays them into the fresh matcher before the
    // lane's first draw (an admit whose replay the grammar refuses, or with
    // no grammar, is rejected). 0 = a fresh generation.
    uint32_t grammar_replay;
    // The prompt's last `decode_replay` tokens were GENERATED by this very
    // request before a preemption (the daemon's produced-so-far count). A
    // resumed lane whose seed stops short of them does not prefill them as
    // a slice: it feeds each as a one-token decode step whose output is
    // discarded — the forward the uninterrupted lane ran at that position —
    // so the continuation is schedule-exact (the §11 numeric floor is per
    // execution shape; a prefill slice is not the shape that produced
    // them). Prefill entries end where these begin; decode is legal from
    // there. 0 = a fresh generation.
    uint32_t decode_replay;
} BaseRTLaneAdmit;  // 144 bytes

// HOST-sampled feedback: EXACTLY ONE token — one logits row authorizes one
// autoregressive choice. No rng_counter field by design: the ENGINE advances
// the canonical position counter by exactly one per committed token.
typedef struct BaseRTLaneCommit {
    uint64_t lane_tag;
    uint32_t token_id;
    uint32_t _pad0;
    uint64_t logits_nonce;  // = emitted row's ring-ref generation
} BaseRTLaneCommit;         // 24 bytes

typedef struct BaseRTLanePrefill {
    uint64_t lane_tag;
    uint32_t token_offset;  // into the lane's admitted prompt
    uint32_t token_count;
} BaseRTLanePrefill;  // 16 bytes

typedef struct BaseRTLaneDecode {
    uint64_t lane_tag;
    uint16_t max_new_tokens;  // HOST lanes: MUST be 1
    // Tokens a speculative round MAY commit past max_new_tokens (0 = none):
    // when max_new_tokens is the tick's decode grant rather than the end of
    // the request, the scheduler sets this to the request's headroom, so the
    // lane's last round drafts a full block instead of a truncated one (or a
    // draft-less step). New rounds still start only while the grant lasts.
    uint16_t overshoot;
    uint32_t _pad1;
} BaseRTLaneDecode;  // 16 bytes

typedef struct BaseRTLaneRetire {
    uint64_t lane_tag;
    uint8_t publish_to_cache;
    uint8_t _pad0[7];
} BaseRTLaneRetire;  // 16 bytes

typedef struct BaseRTTickPlan {
    uint64_t struct_size;  // ABI guard, append-only growth
    uint64_t plan_seq;     // monotonic per bundle; echoed
    uint32_t flags;        // BASERT_TICK_*
    uint32_t _pad0;

    uint32_t prefill_token_budget;  // max prompt tokens this tick
    uint32_t max_decode_lanes;      // <= engine max_batch_size

    BaseRTArray admits;    // of BaseRTLaneAdmit
    BaseRTArray commits;   // of BaseRTLaneCommit
    BaseRTArray prefills;  // of BaseRTLanePrefill
    BaseRTArray decodes;   // of BaseRTLaneDecode
    BaseRTArray retires;   // of BaseRTLaneRetire

    BaseRTShedPolicy shed_policy;

    // Appended (struct_size-gated, BASERT_TICK_PARTIAL_EMITS): invoked from
    // WITHIN the tick, on the ticking thread, each time a lane commits
    // freshly produced tokens — decode rounds and accepted speculative
    // drafts alike. The pointer must stay valid for the duration of the
    // call; the callback must not re-enter the engine.
    void (*on_partial_emit)(void *user, uint64_t lane_tag, const uint32_t *tokens, uint32_t n_tokens);
    void *partial_emit_user;
} BaseRTTickPlan;  // 152 bytes legacy; 168 with the partial-emit tail

// ---------------------------------------------------------------------------
// §1.3 Tick events
// ---------------------------------------------------------------------------

typedef struct BaseRTAdmitResult {
    uint64_t lane_tag;
    uint32_t status;        // BASERT_ADMIT_*
    uint32_t reject_code;   // baseRT_status value when REJECTED
    uint32_t cert_id;       // certificate in effect (0 = none)
    uint8_t granted_class;  // BASERT_EXACTNESS_* actually granted
    uint8_t _pad0[3];
} BaseRTAdmitResult;  // 24 bytes

typedef struct BaseRTSpecStats {
    uint32_t proposed;
    uint32_t accepted;
    uint32_t cert_id_in_effect;  // echo of the admission result
    uint32_t _pad0;
} BaseRTSpecStats;  // 16 bytes

typedef struct BaseRTLaneEmit {
    uint64_t lane_tag;
    BaseRTTokenRef token_ref;  // committed tokens (shm)
    uint32_t n_tokens;
    uint32_t finish;           // BASERT_FINISH_*
    BaseRTRingRef logits_row;  // HOST lanes only; generation IS the
                               // logits_nonce (zeroed otherwise)
    BaseRTSpecStats spec;
} BaseRTLaneEmit;  // 72 bytes

typedef struct BaseRTLaneFault {
    uint64_t lane_tag;
    uint32_t code;  // typed, recoverable per-lane error
    uint32_t _pad0;
    uint64_t detail;
} BaseRTLaneFault;  // 24 bytes

typedef struct BaseRTShedEntry {
    uint64_t lane_tag;  // 0 for non-lane cache evictions
    uint32_t kind;      // BASERT_SHED_*
    uint32_t reason;
    uint64_t bytes;
    uint64_t tokens;
} BaseRTShedEntry;  // 32 bytes

typedef struct BaseRTShedReport {  // never silent (schema §1.4 rule 2)
    uint32_t prefill_chunks_dropped;
    uint32_t evictions_performed;
    uint64_t bytes_evicted;
    BaseRTArray entries;  // of BaseRTShedEntry, with reasons
} BaseRTShedReport;       // 32 bytes

typedef struct BaseRTOpComplete {
    uint64_t op;    // baseRT_op_t
    uint8_t state;  // BASERT_OP_* (terminal)
    uint8_t _pad0[3];
    uint32_t error;
    uint64_t bytes_moved;
} BaseRTOpComplete;  // 24 bytes

typedef struct BaseRTTickTimings {  // coarse, always-on (basertd §11.2)
    uint64_t wall_ns;
    uint64_t prefill_ns;
    uint64_t decode_ns;
    uint32_t graph_hits;
    uint32_t graph_misses;
} BaseRTTickTimings;  // 32 bytes

typedef struct BaseRTMemCounters {  // the budget manager's ledger
    uint64_t allocated_bytes;
    uint64_t host_retained_bytes;
    uint64_t pool_blocks_total;
    uint64_t pool_blocks_used;
    uint64_t pool_bytes_evictable;
    uint64_t tentative_bytes;  // speculation namespaces
} BaseRTMemCounters;           // 48 bytes

// OpenAI caps `top_logprobs` at 20; the per-token record carries that many
// alternatives inline (bounded, fixed stride — no variable-length side arrays).
#define BASERT_MAX_TOP_LOGPROBS 20

// One per-token logprob for a lane that admitted with `want_logprobs` (its admit
// byte holds 1 + top_logprobs count). Records appear in token-emission order per
// lane, so the daemon zips them with the lane's emitted tokens
// (`BaseRTLaneEmit.token_ref`) in order. `logprob` is the natural-log
// probability of the emitted token under the row it was sampled from (after any
// grammar mask / logit_bias). `top_ids[0..n_top]` / `top_logprobs[0..n_top]` are
// the highest-probability alternatives at that position, descending.
typedef struct BaseRTLaneLogprob {
    uint64_t lane_tag;
    float logprob;
    uint32_t n_top;  // number of valid entries in top_ids/top_logprobs (0..20)
    uint32_t top_ids[BASERT_MAX_TOP_LOGPROBS];
    float top_logprobs[BASERT_MAX_TOP_LOGPROBS];
} BaseRTLaneLogprob;  // 176 bytes

typedef struct BaseRTTickEvents {
    uint64_t struct_size;
    uint64_t plan_seq;     // echo
    uint32_t tick_status;  // BASERT_TICK_*
    uint32_t _pad0;

    BaseRTArray admit_results;  // of BaseRTAdmitResult
    BaseRTArray emits;          // of BaseRTLaneEmit
    BaseRTArray faults;         // of BaseRTLaneFault

    BaseRTShedReport shed;
    BaseRTArray op_completions;  // of BaseRTOpComplete

    BaseRTTickTimings timings;
    BaseRTMemCounters mem;
    BaseRTArray logprobs;  // of BaseRTLaneLogprob (want_logprobs lanes only)
} BaseRTTickEvents;        // 216 bytes

// One call per tick per execution bundle. Blocking; returns when the tick's
// GPU work is complete and events are final. `*events_out` is an immutable
// view into an engine-owned double buffer, valid until the next baseRT_tick
// or baseRT_free_model call on the same bundle.
baseRT_status baseRT_tick(baseRT_model_t bundle, const BaseRTTickPlan *plan, const BaseRTTickEvents **events_out);

// ---------------------------------------------------------------------------
// Ring attachment (basertd §10 item 5: caller-provided shm rings)
// ---------------------------------------------------------------------------
//
// The worker owns the shm mappings; the engine is handed base pointers.
// Ring memory layout (normative; basertd-shm's `SharedRing` is the
// reference implementation and the interop test):
//
//   [64-byte ring header]  magic u64 = BASERT_RING_MAGIC ("BRTRING1" LE)
//                          ring_id u32 | slot_bytes u32 | slots u32
//                          (rest reserved, zero)
//   [slots × slot_bytes]   each slot: generation u64 | payload_len u32 |
//                          pad u32 | payload
//
// Exactly one writer per ring. The writer publishes payload bytes first
// and the generation last, with release fences between; readers validate
// the generation before AND after copying the payload (a recycled slot
// is detected, torn bytes are discarded). Generation 0 is the
// never-written sentinel and never matches. Writer cursors start at
// (index 0, generation 1) and generations are globally monotonic per
// ring — `BaseRTTokenRef`/`BaseRTRingRef` generations come from here.

#define BASERT_RING_MAGIC 0x42525452494E4731ull /* "BRTRING1" LE */

enum {  // BaseRTRingDesc.kind
    BASERT_RING_KIND_UNSPECIFIED = 0,
    BASERT_RING_KIND_TOKENS = 1,  // payload = u32 token ids
    BASERT_RING_KIND_LOGITS = 2,  // payload = f32 logits row
};

enum {                                   // BaseRTRingDesc.role —
    BASERT_RING_ROLE_UNSPECIFIED = 0,    // from the ENGINE's side
    BASERT_RING_ROLE_ENGINE_READS = 1,   // agent-staged prompts
    BASERT_RING_ROLE_ENGINE_WRITES = 2,  // emits / logits rows
};

typedef struct BaseRTRingDesc {
    uint64_t struct_size;  // ABI guard, append-only
    uint32_t ring_id;
    uint32_t kind;  // BASERT_RING_KIND_*
    uint32_t role;  // BASERT_RING_ROLE_*
    uint32_t slots;
    uint32_t slot_bytes;  // 16-byte slot header + payload
    uint32_t _pad0;
    void *base;    // mapped ring memory (worker-owned
                   // mapping; ring header leads)
    uint64_t len;  // mapped bytes; >= 64 + slots*slot_bytes
} BaseRTRingDesc;  // 48 bytes

// Attach (or re-attach, replacing by ring_id) the bundle's rings. The
// engine validates each ring header against the descriptor, fail-closed
// with BASERT_ERR_RING_LAYOUT; on any failure NOTHING is attached (all
// or nothing, like plan validation). Mappings must outlive every
// subsequent call on the bundle until re-attach or baseRT_free_model —
// they are the worker's to unmap, never the engine's. Attaching resets
// the engine's writer cursors: attach exactly once per ring lifetime
// (fresh rings per connection, generation_base 1 — schema §4.2).
baseRT_status baseRT_rings_attach(baseRT_model_t bundle, const BaseRTRingDesc *descs, uint32_t count);

// ---------------------------------------------------------------------------
// §10 item 2: state-space discovery
// ---------------------------------------------------------------------------

typedef struct BaseRTStateSpaceDesc {
    uint32_t space_id;
    uint32_t kind;                      // BASERT_SPACE_KIND_*
    uint64_t version_tag;               // stale -> refuse restore -> re-prefill
    uint64_t bytes_per_token;           // paged kinds; 0 for blob kinds
    uint64_t blob_bytes;                // blob kinds; 0 for paged kinds
    uint32_t page_size_tokens;          // paged kinds; 0 otherwise
    uint32_t fork_cost_class;           // 0 unspecified; 1 refcount; 2 copy
    uint64_t fork_cost_bytes;           // what a lazy CoW copy will move
    uint32_t snapshot_cadence;          // 0 none; 1 tool-boundary; 2 interval
    uint32_t snapshot_interval_tokens;  // cadence == 2 only
    uint32_t placement;                 // 0 local (Stage-1 PP field)
    uint32_t flags;
    char name[32];       // NUL-terminated debug name
} BaseRTStateSpaceDesc;  // 96 bytes

// Engine-owned descriptor list; valid until the next call on the same bundle.
baseRT_status baseRT_state_spaces(baseRT_model_t bundle, BaseRTArray *out_descs);

// ---------------------------------------------------------------------------
// §2 Speculation strategy registration (at bundle load, before admission)
// ---------------------------------------------------------------------------

typedef struct BaseRTArtifact {
    uint32_t role;  // BASERT_ARTIFACT_*
    uint32_t _pad0;
    uint8_t content_hash[32];  // claim; the open-handle hash is identity
    uint64_t byte_size;
    const char *load_path;
} BaseRTArtifact;  // 56 bytes

typedef struct BaseRTTapSpec {
    uint32_t layer;
    uint32_t tensor;  // BASERT_TAP_TENSOR_*
    uint32_t dtype;   // engine dtype enum
    uint32_t layout;
} BaseRTTapSpec;  // 16 bytes

// Extensible typed capability descriptor. Unknown kind_id -> registration
// refused, fail-closed (schema §2 scope note).
typedef struct BaseRTCapabilityReq {
    uint32_t kind_id;  // BASERT_STRATEGY_CAP_* or newer
    uint32_t _pad0;
    BaseRTArray params;  // kind-defined records
} BaseRTCapabilityReq;   // 24 bytes

typedef struct BaseRTStrategyRegistration {
    uint64_t struct_size;
    const char *strategy_id;   // "prompt-lookup", "draft-model", ...
    const char *impl_version;  // semver
    uint8_t impl_hash[32];     // detects difference (not compat)
    uint8_t config_hash[32];   // canonical config (basertd §4 identity)

    BaseRTArray artifacts;     // of BaseRTArtifact
    BaseRTArray taps;          // of BaseRTTapSpec
    BaseRTArray capabilities;  // of BaseRTCapabilityReq
    BaseRTArray target_archs;  // of const char *

    uint32_t kernel_caps_required;  // bitset over DeviceCaps-style bits
    uint32_t _pad0;
    uint64_t est_state_bytes;  // drafter state, for the budget ledger

    uint8_t claimed_exactness;  // verified, never trusted (§2.3)
    uint8_t _pad1[3];
    uint32_t rng_contract_version;
} BaseRTStrategyRegistration;  // 176 bytes

typedef struct BaseRTExactnessCert {
    uint32_t cert_id;        // nonzero
    uint8_t exactness;       // BASERT_EXACTNESS_*
    uint8_t sampling_modes;  // bitset of BASERT_CERT_MODE_*
    uint8_t grammar_allowed;
    uint8_t logit_bias_allowed;
    uint32_t param_domain;  // BASERT_CERT_DOMAIN_*
    uint32_t verification_shape_class;
    BaseRTArray admissible_host_identities;  // of uint8_t[32]; empty = never HOST
} BaseRTExactnessCert;                       // 32 bytes

typedef struct BaseRTStrategyGrant {
    uint64_t struct_size;
    uint32_t strategy_slot;  // referenced by BaseRTLaneAdmit
    uint32_t _pad0;
    BaseRTArray certificates;  // of BaseRTExactnessCert; starts EMPTY
    BaseRTArray state_spaces;  // of BaseRTStateSpaceDesc (drafter state)
    uint64_t reserved_bytes;   // what the ledger was charged
} BaseRTStrategyGrant;         // 56 bytes

// Engine-owned grant; valid until the next registration or free on the
// bundle. Fails closed with a typed reason before the model admits work.
baseRT_status baseRT_strategy_register(baseRT_model_t bundle, const BaseRTStrategyRegistration *reg,
                                       const BaseRTStrategyGrant **grant_out);

// ---------------------------------------------------------------------------
// §3 Per-space state operations
// ---------------------------------------------------------------------------

typedef uint64_t baseRT_op_t;  // per-bundle, never reused

typedef struct BaseRTOpStatus {
    uint64_t struct_size;
    uint8_t state;  // BASERT_OP_*
    uint8_t _pad0[3];
    uint32_t error;        // typed, valid when FAILED
    uint64_t bytes_moved;  // progress, monotonic
} BaseRTOpStatus;          // 24 bytes

baseRT_status baseRT_op_poll(baseRT_model_t bundle, baseRT_op_t op, BaseRTOpStatus *out);
baseRT_status baseRT_op_cancel(baseRT_model_t bundle, baseRT_op_t op);  // request only

#define BASERT_ALL_SPACES 0xFFFFFFFFu

typedef struct BaseRTMatchCandidate {
    uint64_t prefix_len;            // tokens
    uint8_t provenance_digest[32];  // basertd §4 hash chain
    uint32_t taint_bits;            // BASERT_TAINT_*
    uint8_t resident_tier;          // BASERT_TIER_*
    uint8_t _pad0[3];
} BaseRTMatchCandidate;  // 48 bytes

typedef struct BaseRTSpaceMatch {
    uint32_t space_id;
    uint32_t _pad0;
    BaseRTArray candidates;  // of BaseRTMatchCandidate
} BaseRTSpaceMatch;          // 24 bytes

typedef struct BaseRTMatchResult {
    uint64_t struct_size;
    BaseRTArray spaces;  // of BaseRTSpaceMatch
} BaseRTMatchResult;     // 24 bytes

// match: synchronous, metadata only, NO pinning. Advisory: a candidate can
// vanish between match and acquire. Result is engine-owned, next-call
// lifetime on the same bundle.
baseRT_status baseRT_space_match(baseRT_model_t bundle,
                                 uint32_t space_id,  // or BASERT_ALL_SPACES
                                 BaseRTTokenRef span,
                                 BaseRTArray media_deps,  // of uint8_t[32] blob hashes
                                 const BaseRTMatchResult **out);

// seed acquire / release: the pinning step. Acquire re-validates the
// coordinator-chosen common prefix against every space and pins the
// underlying blocks/snapshots until the handle is consumed by exactly one
// BaseRTLaneAdmit or released. Unconsumed leases expire after the engine's
// declared TTL (invalidates the handle, not the data).
baseRT_status baseRT_seed_acquire(baseRT_model_t bundle, BaseRTTokenRef span, uint64_t prefix_len,
                                  uint8_t determinism_class,  // clean-variant filter
                                  uint64_t *seed_handle);
baseRT_status baseRT_seed_release(baseRT_model_t bundle, uint64_t seed_handle);
// The declared seed-lease TTL, in ticks: a lease acquired at tick count T
// survives every tick that starts at or before T + ticks. A holder that
// keeps a lease across ticks (a pinned prefix) renews within it.
baseRT_status baseRT_seed_lease_ticks(baseRT_model_t bundle, uint64_t *ticks);

// fork: synchronous, logical. Refcounts paged spaces; CoW-marks blob/ring
// spaces (copy at the child's first divergent step, or now under FORK_EAGER).
// EBUSY if any space of the parent has an in-flight op.
baseRT_status baseRT_seq_fork(baseRT_sequence_t parent, baseRT_sequence_t *child, uint32_t flags);

// export sizing: MANDATORY before snapshot/demote. The result is valid for a
// specific content-generation; the export submission carries sizing_gen back
// and is rejected synchronously with BASERT_ERR_STALE_SIZING if the
// (sequence, space) advanced. Undersized dst rejects synchronously with
// BASERT_ERR_BUFFER_TOO_SMALL — no op handle, no partial writes, ever.
baseRT_status baseRT_space_export_size(baseRT_sequence_t seq, uint32_t space_id, BaseRTTokenRange range,
                                       uint8_t encoding, uint64_t *required_bytes, uint64_t *sizing_gen);

// Engine-authored, self-describing envelope wrapping EVERY exported payload
// (snapshot AND demote). restore/promote accept raw bytes ONLY through this
// envelope: the engine validates it against the TARGET (identity, kind,
// bounds) and the payload against the checksum before publishing. There is
// no caller-asserted "expected identity". Mismatch is a typed refusal.
#define BASERT_STATE_ENVELOPE_MAGIC 0x4554415453545242ull /* "BRTSTATE" LE */

typedef struct BaseRTStateEnvelope {
    uint64_t magic;
    uint32_t envelope_version;
    uint32_t space_kind;          // BASERT_SPACE_KIND_*
    uint8_t compat_identity[32];  // basertd §4 identity 1, per space
    uint64_t version_tag;
    uint8_t provenance_digest[32];  // basertd §4 identity 2
    uint32_t taint_bits;            // BASERT_TAINT_* — taint is sticky
                                    // along the derivation chain
                                    // (basertd §4): the digest travels
                                    // WITH its reason bits, so a
                                    // lossless export of tainted state
                                    // can never restore as clean
    uint32_t _pad0;
    BaseRTTokenRange range;  // blob kinds: {boundary, boundary}
    uint8_t encoding;        // BASERT_ENCODING_*
    uint8_t _pad1[3];
    uint32_t checksum_kind;  // BASERT_CHECKSUM_*
    uint64_t payload_len;    // bytes following this envelope
    uint64_t content_checksum;
} BaseRTStateEnvelope;  // 136 bytes

// snapshot / restore: async, envelope-checked. dst/src are caller-owned but
// LOANED to the engine from submission to terminal state.
baseRT_status baseRT_space_snapshot(baseRT_sequence_t seq, uint32_t space_id, uint64_t boundary_pos, BaseRTBuf dst,
                                    uint64_t sizing_gen, baseRT_op_t *op);
baseRT_status baseRT_space_restore(baseRT_sequence_t seq, uint32_t space_id, BaseRTBuf src, baseRT_op_t *op);

// The newest DURABLE snapshot boundary <= cap for a space: the position at
// which an export equals state the prefix cache could have served, so an
// adopted (park) resume re-prefills exactly the gap a cache-warm resume
// re-prefills — byte-identical start state + identical feed schedule =
// bit-identical continuation. Blob spaces answer their cached capture
// position (0 = nothing durable yet — parking would not be resume-exact);
// the paged KV space answers the last whole-page length. Synchronous.
baseRT_status baseRT_space_snapshot_boundary(baseRT_sequence_t seq, uint32_t space_id, uint64_t cap,
                                             uint64_t *boundary_out);

// trim: synchronous metadata. Paged spaces: any length. Blob/ring spaces:
// snapshot boundaries only (BASERT_ERR_OUT_OF_BOUNDARY otherwise).
baseRT_status baseRT_space_trim(baseRT_sequence_t seq, uint32_t space_id, uint64_t new_len);

// demote / promote: async tier movement. A lossy encoding sets the
// determinism-taint reason bit in the promoted-back state's provenance.
// RecurrentBlob demote is LOSSLESS-only (BASERT_ERR_UNSUPPORTED otherwise).
baseRT_status baseRT_space_demote(baseRT_sequence_t seq, uint32_t space_id, BaseRTTokenRange range, uint8_t encoding,
                                  BaseRTBuf dst, uint64_t sizing_gen, baseRT_op_t *op);
baseRT_status baseRT_space_promote(baseRT_sequence_t seq, uint32_t space_id, BaseRTTokenRange range, BaseRTBuf src,
                                   baseRT_op_t *op);

// cache maintenance: policy-enveloped, like shedding. Pinned blocks (active
// ops, seed leases) are never eligible. Publication into the cache is the
// tick's retire.publish_to_cache, not here.
baseRT_status baseRT_cache_evict(baseRT_model_t bundle, const BaseRTShedPolicy *policy, uint64_t bytes_target,
                                 uint64_t *bytes_freed);

// The cache-entry key for enumerated eviction: the basertd §8 physical
// cache key — (compatibility key, provenance digest) — one record per
// physical variant to drop.
typedef struct BaseRTCacheKey {
    uint8_t compat_key[32];         // per-space §8 chain key
    uint8_t provenance_digest[32];  // variant selector
} BaseRTCacheKey;                   // 64 bytes

baseRT_status baseRT_cache_evict_entries(baseRT_model_t bundle,
                                         BaseRTArray entry_keys);  // of BaseRTCacheKey

// ---------------------------------------------------------------------------
// Tick-bundle sequence lifecycle (support surface for §3 state ops):
// restore targets need sequences that no lane owns yet, and callers
// driving state ops need the sequence a lane owns. In-process handles.
// ---------------------------------------------------------------------------

// The sequence a live lane owns (NULL if the lane is unknown).
baseRT_sequence_t baseRT_tick_lane_sequence(baseRT_model_t bundle, uint64_t lane_tag);

// Create a bare, empty sequence on the bundle's paged pool — a restore
// target. Freed with baseRT_tick_sequence_free (or bundle teardown).
baseRT_sequence_t baseRT_tick_sequence_create(baseRT_model_t bundle);
// Free a LIFECYCLE-OWNED bare sequence. Owned-only, fail-closed:
// unknown, cross-bundle, already-freed, or lane-owned handles are
// refused with BASERT_ERR_UNKNOWN_HANDLE (a lane's sequence is retired
// by its plan, never through this call).
baseRT_status baseRT_tick_sequence_free(baseRT_model_t bundle, baseRT_sequence_t seq);

// Compile a JSON-Schema grammar (xgrammar backend) into the bundle's tick
// registry for grammar-constrained decoding (§6). Returns a nonzero handle
// a lane's admit references via `grammar_handle`, or 0 on failure (bad
// schema, or a legacy grammar with no host bitmask — the batched path
// requires xgrammar). Free with baseRT_tick_grammar_free at the end of the
// constrained generation. The matcher's acceptance state is per-generation,
// so a handle is used by one lane at a time.
uint32_t baseRT_tick_grammar_create(baseRT_model_t bundle, const char *json_schema);

/// As `baseRT_tick_grammar_create`, but from an xgrammar STRUCTURAL TAG (see
/// `baseRT_grammar_create_from_structural_tag`): free text except inside the
/// tag's delimiters, where its schema holds. The handle is interchangeable
/// with a schema-created one — same registry, same host-bitmask path, same
/// free. 0 on failure, including a grammar with no host bitmask.
uint32_t baseRT_tick_grammar_create_structural(baseRT_model_t bundle, const char *tag_json);
// Free a grammar handle; BASERT_ERR_UNKNOWN_HANDLE if absent.
baseRT_status baseRT_tick_grammar_free(baseRT_model_t bundle, uint32_t handle);

// Register a per-request logit_bias map (OpenAI `logit_bias`) into the bundle's
// tick registry: `tokens[i]` gets `+values[i]` added to its logit before the
// argmax/sample in `sample_row`. Returns a nonzero handle a lane's admit
// references via `logit_bias_handle`, or 0 on failure (n <= 0, or a null
// array). A lane carrying a bias handle is kept off the GPU-argmax fast path
// (the bias must reach the host logits row). Free at the end of the generation.
uint32_t baseRT_tick_logit_bias_create(baseRT_model_t bundle, const int32_t *tokens, const float *values, int n);
// Free a logit_bias handle; BASERT_ERR_UNKNOWN_HANDLE if absent.
baseRT_status baseRT_tick_logit_bias_free(baseRT_model_t bundle, uint32_t handle);

// Adopt: lease a LIFECYCLE-OWNED bare sequence (a restore target whose
// state ops have run) as the seed of the next admission (basertd P2.5).
// `span` names the tokens the sequence's state covers (its length must
// equal the sequence's current length; the daemon validated the content
// against the artifact's stream digest — the engine records it as the
// lane's committed prefix). Unlike a cache seed the sequence is taken
// over WHOLE — every state space it holds (KV incl. a partial last block,
// recurrent blobs) — so it is the sound warm path for bundles whose
// spaces the cache cannot serve. The lease consumes the sequence at
// admission (the lane owns it from then on); an expired lease frees it.
baseRT_status baseRT_seed_adopt(baseRT_model_t bundle, baseRT_sequence_t seq, BaseRTTokenRef span,
                                uint64_t *seed_handle);

// ---------------------------------------------------------------------------
// Media (basertd §4 media, §7.2 EncoderCache — phase 2 P2.7). Images enter
// a lane's prompt as runs of the model's image placeholder token; the
// tower's features for a blob are an engine-owned EncoderCache entry
// addressed by a media handle; a bind attaches handles to a lane BEFORE
// its placeholders prefill (by lane tag, admitted or not yet — the admit
// adopts pending binds; a retire drops the lane's remaining binds).
// ---------------------------------------------------------------------------

typedef struct BaseRTMediaInfo {
    uint32_t n_tokens;        // placeholders this image expands to
    uint32_t image_token_id;  // the placeholder token
    uint32_t boi_token_id;    // begin-of-image wrapper (0 = none)
    uint32_t eoi_token_id;    // end-of-image wrapper (0 = none)
    uint64_t preprocess_fp;   // preprocessing-config fingerprint (basertd §4
                              // media identity: patch/merge/pool geometry)
} BaseRTMediaInfo;            // 24 bytes

// Preprocess only (no tower): how many placeholders `image_path` expands
// to and the wrapper token ids. UNSUPPORTED on bundles without a tower.
baseRT_status baseRT_media_probe(baseRT_model_t bundle, const char *image_path, BaseRTMediaInfo *out);

// Preprocess + tower: an EncoderCache entry (engine-owned features) for
// `image_path`. UNSUPPORTED on bundles whose tick path cannot splice
// media (hybrid-GDN: no per-lane M-RoPE in continuous batching yet; MoE).
baseRT_status baseRT_media_encode(baseRT_model_t bundle, const char *image_path, uint64_t *media_handle,
                                  BaseRTMediaInfo *out);
baseRT_status baseRT_media_release(baseRT_model_t bundle, uint64_t media_handle);

// Bind a media handle to a lane (by tag; before or after admit) at the
// prompt offset where its placeholder run starts. The lane's prefill
// chunk containing that run splices the features; a chunk that splits a
// run is a per-lane fault. Binds are consumed in offset order and
// released with the lane.
baseRT_status baseRT_media_bind(baseRT_model_t bundle, uint64_t lane_tag, uint64_t media_handle, uint32_t token_offset);

// Publish a bare sequence's block-aligned KV prefix into the reuse
// cache under `tokens` — the resume bridge: restore into a fresh
// sequence, publish, then re-enter through the ordinary seeded
// admission path. Same whole-block contract as retire's
// publish_to_cache (sub-block tails publish nothing and still
// succeed). Consumes the sequence on BASERT_STATUS_OK; on error the
// sequence is untouched and still owned by the caller.
baseRT_status baseRT_tick_sequence_publish(baseRT_model_t bundle, baseRT_sequence_t seq, const uint32_t *tokens,
                                           uint32_t n_tokens);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // BASERT_TICK_H
