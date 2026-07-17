/**
 * @baseRT/node — Node.js / TypeScript bindings for the BaseRT LLM inference
 * engine (Apple Silicon / Metal).
 *
 * Implemented on top of koffi (https://koffi.dev) — a maintained, prebuilt FFI
 * for Node that needs no node-gyp / native addon compilation and correctly
 * handles struct-by-value arguments/returns and C callbacks on Node 18–22.
 *
 *   import { BaseRTModel } from "@baseRT/node";
 *   const model = new BaseRTModel("model.base");
 *   const text = model.generateText(model.encode("Hello, world!"), 128);
 *   model.close();
 *
 * Generation calls are synchronous and block the calling thread for their
 * duration (local inference is CPU/GPU-bound); run them in a worker_thread if
 * you need the event loop free. The streaming `onToken` callback is invoked
 * synchronously per token from inside the call.
 */

import koffi from "koffi";
import * as fs from "fs";
import * as path from "path";

// ---------------------------------------------------------------------------
// Library resolution
// ---------------------------------------------------------------------------

// Platform-appropriate shared-library filenames, most likely first. Metal/
// macOS is the only shipping backend today, but selecting by platform means
// this loader needs no change when a Linux (CUDA/ROCm) or Windows build appears
// — the C ABI is identical, only the file name differs.
function libNames(): string[] {
  switch (process.platform) {
    case "darwin":
      return ["libbaseRT.dylib"];
    case "win32":
      return ["baseRT.dll", "libbaseRT.dll"];
    default:
      return ["libbaseRT.so"]; // linux / other unix
  }
}

function resolveLibPath(): string {
  if (process.env.BASERT_LIB_PATH) return process.env.BASERT_LIB_PATH;
  const here = __dirname; // dist/ or src/ at runtime
  const dirs = [
    path.resolve(here, "..", "..", "..", "build"), // bindings/node/{dist,src} -> repo/build
    path.resolve(here, "..", "..", "build"),
    path.resolve(process.cwd(), "build"),
    path.resolve(process.cwd()),
  ];
  const names = libNames();
  const tried: string[] = [];
  for (const d of dirs) {
    for (const name of names) {
      const c = path.join(d, name);
      tried.push(c);
      if (fs.existsSync(c)) return c;
    }
  }
  throw new Error(
    `Cannot find ${names.join(" / ")}. Set BASERT_LIB_PATH or build with \`make shared\`. Looked in:\n  ${tried.join(
      "\n  "
    )}`
  );
}

// ---------------------------------------------------------------------------
// C struct definitions (koffi computes offsets/size from the field list, so
// these track include/baseRT/types.h exactly — including the fields the old
// hand-written offset table dropped, e.g. `sliding_window`).
// ---------------------------------------------------------------------------

const ModelConfigC = koffi.struct("BaseRTModelConfig", {
  dim: "uint32",
  n_layers: "uint32",
  n_heads: "uint32",
  n_kv_heads: "uint32",
  head_dim: "uint32",
  q_dim: "uint32",
  kv_dim: "uint32",
  ffn_dim: "uint32",
  vocab_size: "uint32",
  max_seq_len: "uint32",
  norm_eps: "float",
  rope_theta: "float",
  sliding_window_pattern: "uint32",
  sliding_window: "uint32",
  rope_local_theta: "float",
  architecture: koffi.array("char", 32),
  // Encoder (Whisper)
  enc_n_layers: "uint32",
  enc_n_heads: "uint32",
  enc_dim: "uint32",
  enc_ffn_dim: "uint32",
  n_mels: "uint32",
  enc_max_seq_len: "uint32",
  // Gemma 4
  n_embd_per_layer: "uint32",
  n_layer_kv_from_start: "uint32",
  logit_softcap: "float",
  attention_scale: "float",
  head_dim_swa: "uint32",
  head_dim_global: "uint32",
  global_rope_partial_factor: "float",
  swa_layers: koffi.array("uint8", 64),
  ffn_dims: koffi.array("uint32", 128),
  n_kv_heads_per_layer: koffi.array("uint32", 128),
  // Qwen3.5 / 3.6 hybrid linear-attention (Gated DeltaNet)
  attn_output_gate: "uint8",
  _qwen35_pad: koffi.array("uint8", 3),
  partial_rotary_factor: "float",
  full_attention_interval: "uint32",
  linear_attn_layers: koffi.array("uint8", 64),
  gdn_num_k_heads: "uint32",
  gdn_num_v_heads: "uint32",
  gdn_key_head_dim: "uint32",
  gdn_value_head_dim: "uint32",
  gdn_conv_kernel: "uint32",
  // MoE
  n_experts: "uint32",
  n_experts_used: "uint32",
  n_experts_shared: "uint32",
  expert_ffn_dim: "uint32",
  expert_gating: "uint8",
  norm_topk_prob: "uint8",
  _moe_pad: koffi.array("uint8", 2),
  // Vision tower
  vision_n_layers: "uint32",
  vision_dim: "uint32",
  vision_n_heads: "uint32",
  vision_head_dim: "uint32",
  vision_ffn_dim: "uint32",
  vision_patch_size: "uint32",
  vision_image_size: "uint32",
  vision_pooling_kernel: "uint32",
  vision_soft_tokens: "uint32",
  vision_norm_eps: "float",
  vision_rope_theta: "float",
  vision_pos_embed_size: "uint32",
  image_token_id: "uint32",
  boi_token_id: "uint32",
  eoi_token_id: "uint32",
  // Vision-tower family selector + Qwen3-VL-style extras
  vision_arch: "uint32",
  vision_spatial_merge: "uint32",
  vision_temporal_patch: "uint32",
  vision_out_dim: "uint32",
  // Audio tower
  audio_n_layers: "uint32",
  audio_dim: "uint32",
  audio_n_heads: "uint32",
  audio_head_dim: "uint32",
  audio_ffn_dim: "uint32",
  audio_output_proj_dim: "uint32",
  audio_chunk_size: "uint32",
  audio_left_context: "uint32",
  audio_conv_kernel: "uint32",
  audio_soft_tokens: "uint32",
  audio_logit_softcap: "float",
  audio_norm_eps: "float",
  audio_gradient_clip: "float",
  audio_residual_weight: "float",
  audio_ms_per_token: "float",
  audio_sscp_channels: koffi.array("uint32", 2),
  audio_token_id: "uint32",
  boa_token_id: "uint32",
  eoa_token_id: "uint32",
  mrope_section: koffi.array("uint32", 3),
  mrope_interleaved: "uint8",
  _mrope_pad: koffi.array("uint8", 3),
});

const SamplingConfigC = koffi.struct("BaseRTSamplingConfig", {
  temperature: "float",
  top_k: "int",
  top_p: "float",
  min_p: "float",
  repeat_penalty: "float",
  presence_penalty: "float",
  frequency_penalty: "float",
  seed: "uint32",
  n_logit_bias: "int32",
  logit_bias_tokens: "int32 *",
  logit_bias_values: "float *",
});

const GenerationStatsC = koffi.struct("BaseRTGenerationStats", {
  prompt_tokens: "int",
  generated_tokens: "int",
  prefill_time_ms: "float",
  decode_time_ms: "float",
  prefill_tokens_per_sec: "float",
  decode_tokens_per_sec: "float",
});

const TranscribeStatsC = koffi.struct("BaseRTTranscribeStats", {
  n_tokens: "int",
  audio_ms: "float",
  encode_ms: "float",
  decode_ms: "float",
  total_ms: "float",
});

// bool (*)(uint32_t token_id, const char *text, void *user_data)
const TokenCallbackC = koffi.proto(
  "bool TokenCallback(uint32_t token_id, const char *text, void *user_data)"
);
// bool (*)(int start_ms, int end_ms, const char *text, void *user_data)
const SegmentCallbackC = koffi.proto(
  "bool SegmentCallback(int start_ms, int end_ms, const char *text, void *user_data)"
);

// ---------------------------------------------------------------------------
// Public TypeScript types
// ---------------------------------------------------------------------------

export interface ModelConfig {
  dim: number;
  nLayers: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  qDim: number;
  kvDim: number;
  ffnDim: number;
  vocabSize: number;
  maxSeqLen: number;
  normEps: number;
  ropeTheta: number;
  slidingWindowPattern: number;
  slidingWindow: number;
  ropeLocalTheta: number;
  architecture: string;
  // Encoder (Whisper); 0 for decoder-only models.
  encNLayers: number;
  encNHeads: number;
  encDim: number;
  encFfnDim: number;
  nMels: number;
  encMaxSeqLen: number;
  // Capability flags derived from the tower/MoE fields.
  nExperts: number;
  hasVision: boolean;
  hasAudio: boolean;
}

export interface SamplingConfig {
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  repeatPenalty?: number;
  /** OpenAI-style presence penalty. 0 = off. */
  presencePenalty?: number;
  /** OpenAI-style frequency penalty. 0 = off. */
  frequencyPenalty?: number;
  /** Sampling seed. 0 = wall-clock-seeded; non-zero = deterministic. */
  seed?: number;
  /** Additive per-token logit bias. Values must be in [-100, 100]. */
  logitBias?: Record<number, number>;
}

export interface GenerationStats {
  promptTokens: number;
  generatedTokens: number;
  prefillTimeMs: number;
  decodeTimeMs: number;
  prefillTokensPerSec: number;
  decodeTokensPerSec: number;
}

export interface TranscribeStats {
  nTokens: number;
  audioMs: number;
  encodeMs: number;
  decodeMs: number;
  totalMs: number;
}

export interface TranscribeResult {
  text: string;
  stats: TranscribeStats;
}

export type TokenCallback = (tokenId: number, text: string) => boolean;
export type SegmentCallback = (
  startMs: number,
  endMs: number,
  text: string
) => boolean;

// ---------------------------------------------------------------------------
// FFI binding (declared once, lazily, against the resolved library)
// ---------------------------------------------------------------------------

interface BaseRTLib {
  baseRT_version_string: () => string;
  baseRT_load_model: (
    modelPath: string,
    kernelLibraryPath: string | null,
    maxContext: number
  ) => unknown;
  baseRT_free_model: (model: unknown) => void;
  baseRT_get_config: (model: unknown) => Record<string, unknown>;
  baseRT_model_memory: (model: unknown) => number;
  baseRT_get_error: () => string;
  baseRT_set_kv_bits: (bits: number) => void;
  baseRT_set_paged_kv: (enable: number) => void;
  baseRT_set_max_batch_size: (n: number) => void;
  baseRT_encode: (
    model: unknown,
    text: string,
    outTokens: Uint32Array,
    maxTokens: number
  ) => number;
  baseRT_token_count: (model: unknown, text: string) => number;
  baseRT_decode_token: (model: unknown, tokenId: number) => string;
  baseRT_decode_token_static: (model: unknown, tokenId: number) => string;
  baseRT_generate: (
    model: unknown,
    promptTokens: Uint32Array,
    nPrompt: number,
    maxTokens: number,
    sampling: Record<string, unknown>,
    callback: unknown,
    userData: unknown
  ) => Record<string, number>;
  baseRT_generate_continue: (
    model: unknown,
    newTokens: Uint32Array,
    nNew: number,
    maxTokens: number,
    sampling: Record<string, unknown>,
    callback: unknown,
    userData: unknown
  ) => Record<string, number>;
  baseRT_prefill: (model: unknown, tokens: Uint32Array, n: number) => number;
  baseRT_prefill_image: (
    model: unknown,
    tokens: Uint32Array,
    n: number,
    imagePath: string
  ) => number;
  baseRT_image_num_tokens: (model: unknown, imagePath: string) => number;
  baseRT_prefill_audio: (
    model: unknown,
    tokens: Uint32Array,
    n: number,
    pcm: Float32Array,
    nSamples: number
  ) => number;
  baseRT_audio_num_tokens: (model: unknown, nSamples: number) => number;
  baseRT_decode_step: (
    model: unknown,
    tokenId: number,
    position: number
  ) => number;
  baseRT_chain_decode: (
    model: unknown,
    firstToken: number,
    startPosition: number,
    count: number,
    outTokens: Uint32Array
  ) => number;
  baseRT_get_position: (model: unknown) => number;
  baseRT_set_speculation: (model: unknown, enabled: boolean) => void;
  baseRT_reset: (model: unknown) => void;
  baseRT_embed: (
    model: unknown,
    tokens: Uint32Array,
    n: number,
    out: Float32Array,
    maxDims: number
  ) => number;
  baseRT_embed_text: (
    model: unknown,
    text: string,
    out: Float32Array,
    maxDims: number
  ) => number;
  baseRT_embedding_dim: (model: unknown) => number;
  baseRT_format_chat: (
    model: unknown,
    system: string,
    user: string
  ) => string;
  baseRT_chat_template: (model: unknown) => string;
  baseRT_is_whisper: (model: unknown) => boolean;
  baseRT_set_timestamps: (model: unknown, enabled: boolean) => void;
  baseRT_transcribe: (
    model: unknown,
    wavPath: string,
    language: string | null,
    statsOut: Record<string, unknown>
  ) => string;
  baseRT_transcribe_pcm: (
    model: unknown,
    samples: Float32Array,
    n: number,
    language: string | null,
    statsOut: Record<string, unknown>
  ) => string;
  baseRT_transcribe_stream: (
    model: unknown,
    wavPath: string,
    language: string | null,
    statsOut: Record<string, unknown>,
    callback: unknown,
    userData: unknown
  ) => string;
  baseRT_transcribe_pcm_stream: (
    model: unknown,
    samples: Float32Array,
    n: number,
    language: string | null,
    statsOut: Record<string, unknown>,
    callback: unknown,
    userData: unknown
  ) => string;
}

let _lib: BaseRTLib | null = null;

function lib(): BaseRTLib {
  if (_lib) return _lib;
  const k = koffi.load(resolveLibPath());
  const out = "_Out_";
  void out;
  _lib = {
    baseRT_version_string: k.func("const char *baseRT_version_string()"),
    baseRT_load_model: k.func(
      "void *baseRT_load_model(const char *, const char *, int)"
    ),
    baseRT_free_model: k.func("void baseRT_free_model(void *)"),
    baseRT_get_config: k.func("BaseRTModelConfig baseRT_get_config(void *)"),
    baseRT_model_memory: k.func("size_t baseRT_model_memory(void *)"),
    baseRT_get_error: k.func("const char *baseRT_get_error()"),
    baseRT_set_kv_bits: k.func("void baseRT_set_kv_bits(int)"),
    baseRT_set_paged_kv: k.func("void baseRT_set_paged_kv(int)"),
    baseRT_set_max_batch_size: k.func("void baseRT_set_max_batch_size(int)"),
    baseRT_encode: k.func(
      "int baseRT_encode(void *, const char *, _Out_ uint32_t *, int)"
    ),
    baseRT_token_count: k.func("int baseRT_token_count(void *, const char *)"),
    baseRT_decode_token: k.func(
      "const char *baseRT_decode_token(void *, uint32_t)"
    ),
    baseRT_decode_token_static: k.func(
      "const char *baseRT_decode_token_static(void *, uint32_t)"
    ),
    baseRT_generate: k.func(
      "BaseRTGenerationStats baseRT_generate(void *, uint32_t *, int, int, BaseRTSamplingConfig, TokenCallback *, void *)"
    ),
    baseRT_generate_continue: k.func(
      "BaseRTGenerationStats baseRT_generate_continue(void *, uint32_t *, int, int, BaseRTSamplingConfig, TokenCallback *, void *)"
    ),
    baseRT_prefill: k.func("uint32_t baseRT_prefill(void *, uint32_t *, int)"),
    baseRT_prefill_image: k.func(
      "uint32_t baseRT_prefill_image(void *, uint32_t *, int, const char *)"
    ),
    baseRT_image_num_tokens: k.func(
      "int baseRT_image_num_tokens(void *, const char *)"
    ),
    baseRT_prefill_audio: k.func(
      "uint32_t baseRT_prefill_audio(void *, uint32_t *, int, float *, int)"
    ),
    baseRT_audio_num_tokens: k.func(
      "int baseRT_audio_num_tokens(void *, int)"
    ),
    baseRT_decode_step: k.func(
      "uint32_t baseRT_decode_step(void *, uint32_t, int)"
    ),
    baseRT_chain_decode: k.func(
      "int baseRT_chain_decode(void *, uint32_t, int, int, _Out_ uint32_t *)"
    ),
    baseRT_get_position: k.func("int baseRT_get_position(void *)"),
    baseRT_set_speculation: k.func("void baseRT_set_speculation(void *, bool)"),
    baseRT_reset: k.func("void baseRT_reset(void *)"),
    baseRT_embed: k.func(
      "int baseRT_embed(void *, uint32_t *, int, _Out_ float *, int)"
    ),
    baseRT_embed_text: k.func(
      "int baseRT_embed_text(void *, const char *, _Out_ float *, int)"
    ),
    baseRT_embedding_dim: k.func("int baseRT_embedding_dim(void *)"),
    baseRT_format_chat: k.func(
      "const char *baseRT_format_chat(void *, const char *, const char *)"
    ),
    baseRT_chat_template: k.func(
      "const char *baseRT_chat_template(void *)"
    ),
    baseRT_is_whisper: k.func("bool baseRT_is_whisper(void *)"),
    baseRT_set_timestamps: k.func("void baseRT_set_timestamps(void *, bool)"),
    baseRT_transcribe: k.func(
      "const char *baseRT_transcribe(void *, const char *, const char *, _Out_ BaseRTTranscribeStats *)"
    ),
    baseRT_transcribe_pcm: k.func(
      "const char *baseRT_transcribe_pcm(void *, float *, int, const char *, _Out_ BaseRTTranscribeStats *)"
    ),
    baseRT_transcribe_stream: k.func(
      "const char *baseRT_transcribe_stream(void *, const char *, const char *, _Out_ BaseRTTranscribeStats *, SegmentCallback *, void *)"
    ),
    baseRT_transcribe_pcm_stream: k.func(
      "const char *baseRT_transcribe_pcm_stream(void *, float *, int, const char *, _Out_ BaseRTTranscribeStats *, SegmentCallback *, void *)"
    ),
  } as BaseRTLib;

  // ModelConfigC is mirrored by hand above; a size mismatch means the mirror
  // drifted from include/baseRT/types.h and every field after the divergence
  // point decodes as garbage. Fail at load, not at use. (Older libraries
  // predate the symbol; skip the check there.)
  try {
    const configSizeof = k.func("size_t baseRT_model_config_sizeof()");
    const cSize = Number(configSizeof());
    const jsSize = koffi.sizeof(ModelConfigC);
    if (cSize !== jsSize) {
      throw new Error(
        `BaseRTModelConfig layout drift: library says ${cSize} bytes, ` +
          `Node mirror is ${jsSize} bytes. Update ModelConfigC in ` +
          `bindings/node/src/index.ts to match include/baseRT/types.h.`
      );
    }
  } catch (e) {
    if (e instanceof Error && e.message.includes("layout drift")) throw e;
    // Symbol missing in an older library — no check possible.
  }
  return _lib;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Process-wide engine settings. Must be set BEFORE constructing a model. */
export const Engine = {
  /** Library version string (e.g. "0.2.0"). */
  version(): string {
    return lib().baseRT_version_string();
  },
  /** Force KV-cache element width: 0=auto, 8=Q8_0, 16=F16. */
  setKvBits(bits: number): void {
    lib().baseRT_set_kv_bits(bits);
  },
  /** Toggle paged-KV mode for subsequent loads. */
  setPagedKv(enable: boolean): void {
    lib().baseRT_set_paged_kv(enable ? 1 : 0);
  },
  /** Max in-flight batch size for batched decode (sizes logits scratch). */
  setMaxBatchSize(n: number): void {
    lib().baseRT_set_max_batch_size(n);
  },
  /** Last thread-local error string, or null. */
  lastError(): string | null {
    return lib().baseRT_get_error() || null;
  },
};

function cStringFromCharArray(arr: ArrayLike<number> | string): string {
  if (typeof arr === "string") {
    const nul = arr.indexOf("\0");
    return nul >= 0 ? arr.slice(0, nul) : arr;
  }
  let s = "";
  for (let i = 0; i < arr.length; i++) {
    const c = arr[i];
    if (c === 0) break;
    s += String.fromCharCode(c);
  }
  return s;
}

function toModelConfig(c: Record<string, any>): ModelConfig {
  return {
    dim: c.dim,
    nLayers: c.n_layers,
    nHeads: c.n_heads,
    nKvHeads: c.n_kv_heads,
    headDim: c.head_dim,
    qDim: c.q_dim,
    kvDim: c.kv_dim,
    ffnDim: c.ffn_dim,
    vocabSize: c.vocab_size,
    maxSeqLen: c.max_seq_len,
    normEps: c.norm_eps,
    ropeTheta: c.rope_theta,
    slidingWindowPattern: c.sliding_window_pattern,
    slidingWindow: c.sliding_window,
    ropeLocalTheta: c.rope_local_theta,
    architecture: cStringFromCharArray(c.architecture),
    encNLayers: c.enc_n_layers,
    encNHeads: c.enc_n_heads,
    encDim: c.enc_dim,
    encFfnDim: c.enc_ffn_dim,
    nMels: c.n_mels,
    encMaxSeqLen: c.enc_max_seq_len,
    nExperts: c.n_experts,
    hasVision: c.vision_n_layers > 0,
    hasAudio: c.audio_n_layers > 0,
  };
}

function toSamplingC(opts?: SamplingConfig): Record<string, unknown> {
  const s: Record<string, unknown> = {
    temperature: opts?.temperature ?? 0.0,
    top_k: opts?.topK ?? 40,
    top_p: opts?.topP ?? 0.9,
    min_p: opts?.minP ?? 0.0,
    repeat_penalty: opts?.repeatPenalty ?? 1.0,
    presence_penalty: opts?.presencePenalty ?? 0.0,
    frequency_penalty: opts?.frequencyPenalty ?? 0.0,
    seed: opts?.seed ?? 0,
    n_logit_bias: 0,
    logit_bias_tokens: null,
    logit_bias_values: null,
  };
  const lb = opts?.logitBias;
  if (lb && Object.keys(lb).length > 0) {
    const entries = Object.entries(lb);
    const toks = new Int32Array(entries.length);
    const vals = new Float32Array(entries.length);
    entries.forEach(([t, v], i) => {
      toks[i] = parseInt(t, 10);
      vals[i] = v;
    });
    s.n_logit_bias = entries.length;
    s.logit_bias_tokens = toks;
    s.logit_bias_values = vals;
  }
  return s;
}

function toGenStats(s: Record<string, number>): GenerationStats {
  return {
    promptTokens: s.prompt_tokens,
    generatedTokens: s.generated_tokens,
    prefillTimeMs: s.prefill_time_ms,
    decodeTimeMs: s.decode_time_ms,
    prefillTokensPerSec: s.prefill_tokens_per_sec,
    decodeTokensPerSec: s.decode_tokens_per_sec,
  };
}

function toTranscribeStats(s: Record<string, number>): TranscribeStats {
  return {
    nTokens: s.n_tokens,
    audioMs: s.audio_ms,
    encodeMs: s.encode_ms,
    decodeMs: s.decode_ms,
    totalMs: s.total_ms,
  };
}

// ---------------------------------------------------------------------------
// BaseRTModel
// ---------------------------------------------------------------------------

export class BaseRTModel {
  private _handle: unknown | null;
  private _l: BaseRTLib;

  /**
   * Load a BaseRT model from a `.base` bundle.
   *
   * @param modelPath         Path to the `.base` model file.
   * @param kernelLibraryPath Path to the compiled GPU kernel library (on Metal,
   *                          baseRT.metallib), or null to auto-detect — the
   *                          shipped single-file dylib carries it embedded.
   * @param maxContext        Maximum context window (0 = model default).
   */
  constructor(
    modelPath: string,
    kernelLibraryPath: string | null = null,
    maxContext = 0
  ) {
    this._l = lib();
    this._handle = this._l.baseRT_load_model(
      modelPath,
      kernelLibraryPath,
      maxContext
    );
    if (!this._handle) {
      throw new Error(
        `Failed to load model: ${this._l.baseRT_get_error() || "unknown error"}`
      );
    }
  }

  private get handle(): unknown {
    if (!this._handle) throw new Error("Model has been closed");
    return this._handle;
  }

  /** Free all GPU resources. Safe to call multiple times. */
  close(): void {
    if (this._handle) {
      this._l.baseRT_free_model(this._handle);
      this._handle = null;
    }
  }

  /** Support `using model = new BaseRTModel(...)`. */
  [Symbol.dispose](): void {
    this.close();
  }

  // -- Model info ------------------------------------------------------------

  get config(): ModelConfig {
    return toModelConfig(this._l.baseRT_get_config(this.handle));
  }

  get memoryUsage(): number {
    return this._l.baseRT_model_memory(this.handle);
  }

  get isWhisper(): boolean {
    return this._l.baseRT_is_whisper(this.handle);
  }

  get position(): number {
    return this._l.baseRT_get_position(this.handle);
  }

  // -- Tokenization ----------------------------------------------------------

  encode(text: string, maxTokens = 8192): number[] {
    const buf = new Uint32Array(maxTokens);
    const n = this._l.baseRT_encode(this.handle, text, buf, maxTokens);
    if (n < 0) {
      throw new Error(`encode failed: ${this._l.baseRT_get_error() || "?"}`);
    }
    return Array.from(buf.subarray(0, n));
  }

  tokenCount(text: string): number {
    return this._l.baseRT_token_count(this.handle, text);
  }

  decodeToken(tokenId: number): string {
    return this._l.baseRT_decode_token(this.handle, tokenId) || "";
  }

  // -- Generation ------------------------------------------------------------

  private _registerToken(
    onToken?: TokenCallback
  ): koffi.IKoffiRegisteredCallback | null {
    if (!onToken) return null;
    return koffi.register(
      (tokenId: number, text: string, _ud: unknown) =>
        onToken(tokenId, text || ""),
      koffi.pointer(TokenCallbackC)
    );
  }

  /**
   * Generate from a prompt. Blocks until generation completes; `onToken` is
   * invoked synchronously per token (return false to stop early).
   */
  generate(
    tokens: number[] | Uint32Array,
    maxTokens: number,
    sampling?: SamplingConfig,
    onToken?: TokenCallback
  ): GenerationStats {
    const arr = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    const cb = this._registerToken(onToken);
    try {
      const stats = this._l.baseRT_generate(
        this.handle,
        arr,
        arr.length,
        maxTokens,
        toSamplingC(sampling),
        cb,
        null
      );
      return toGenStats(stats);
    } finally {
      if (cb) koffi.unregister(cb);
    }
  }

  /** Continue from the current KV cache state (multi-turn). */
  generateContinue(
    tokens: number[] | Uint32Array,
    maxTokens: number,
    sampling?: SamplingConfig,
    onToken?: TokenCallback
  ): GenerationStats {
    const arr = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    const cb = this._registerToken(onToken);
    try {
      const stats = this._l.baseRT_generate_continue(
        this.handle,
        arr,
        arr.length,
        maxTokens,
        toSamplingC(sampling),
        cb,
        null
      );
      return toGenStats(stats);
    } finally {
      if (cb) koffi.unregister(cb);
    }
  }

  /** Generate and return the full decoded text. */
  generateText(
    tokens: number[] | Uint32Array,
    maxTokens: number,
    sampling?: SamplingConfig
  ): string {
    const pieces: string[] = [];
    this.generate(tokens, maxTokens, sampling, (_id, text) => {
      pieces.push(text);
      return true;
    });
    return pieces.join("");
  }

  // -- Low-level -------------------------------------------------------------

  prefill(tokens: number[] | Uint32Array): number {
    const arr = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    return this._l.baseRT_prefill(this.handle, arr, arr.length);
  }

  prefillImage(tokens: number[] | Uint32Array, imagePath: string): number {
    const arr = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    return this._l.baseRT_prefill_image(this.handle, arr, arr.length, imagePath);
  }

  imageNumTokens(imagePath: string): number {
    return this._l.baseRT_image_num_tokens(this.handle, imagePath);
  }

  prefillAudio(tokens: number[] | Uint32Array, pcm: Float32Array): number {
    const arr = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    return this._l.baseRT_prefill_audio(this.handle, arr, arr.length, pcm, pcm.length);
  }

  audioNumTokens(nSamples: number): number {
    return this._l.baseRT_audio_num_tokens(this.handle, nSamples);
  }

  decodeStep(tokenId: number, position: number): number {
    return this._l.baseRT_decode_step(this.handle, tokenId, position);
  }

  chainDecode(firstToken: number, startPosition: number, count: number): number[] {
    const buf = new Uint32Array(count);
    const n = this._l.baseRT_chain_decode(
      this.handle,
      firstToken,
      startPosition,
      count,
      buf
    );
    return Array.from(buf.subarray(0, Math.max(0, n)));
  }

  reset(): void {
    this._l.baseRT_reset(this.handle);
  }

  setSpeculation(enabled: boolean): void {
    this._l.baseRT_set_speculation(this.handle, enabled);
  }

  // -- Embeddings ------------------------------------------------------------

  get embeddingDim(): number {
    return this._l.baseRT_embedding_dim(this.handle);
  }

  embed(tokens: number[] | Uint32Array): number[] {
    const dim = this.embeddingDim;
    const arr = tokens instanceof Uint32Array ? tokens : Uint32Array.from(tokens);
    const out = new Float32Array(dim);
    const n = this._l.baseRT_embed(this.handle, arr, arr.length, out, dim);
    if (n <= 0) throw new Error(`embed failed: ${this._l.baseRT_get_error() || "?"}`);
    return Array.from(out.subarray(0, n));
  }

  embedText(text: string): number[] {
    const dim = this.embeddingDim;
    const out = new Float32Array(dim);
    const n = this._l.baseRT_embed_text(this.handle, text, out, dim);
    if (n <= 0) throw new Error(`embedText failed: ${this._l.baseRT_get_error() || "?"}`);
    return Array.from(out.subarray(0, n));
  }

  // -- Chat templates --------------------------------------------------------

  formatChat(system: string, user: string): string {
    return this._l.baseRT_format_chat(this.handle, system, user) || "";
  }

  get chatTemplate(): string {
    return this._l.baseRT_chat_template(this.handle) || "";
  }

  // -- Whisper ---------------------------------------------------------------

  setTimestamps(enabled: boolean): void {
    this._l.baseRT_set_timestamps(this.handle, enabled);
  }

  transcribe(wavPath: string, language?: string): TranscribeResult {
    const stats: Record<string, number> = {};
    const text = this._l.baseRT_transcribe(
      this.handle,
      wavPath,
      language ?? null,
      stats
    );
    return { text: text || "", stats: toTranscribeStats(stats) };
  }

  transcribePcm(samples: Float32Array, language?: string): TranscribeResult {
    const stats: Record<string, number> = {};
    const text = this._l.baseRT_transcribe_pcm(
      this.handle,
      samples,
      samples.length,
      language ?? null,
      stats
    );
    return { text: text || "", stats: toTranscribeStats(stats) };
  }

  transcribeStream(
    wavPath: string,
    language?: string,
    onSegment?: SegmentCallback
  ): TranscribeResult {
    const stats: Record<string, number> = {};
    const cb: koffi.IKoffiRegisteredCallback | null = onSegment
      ? koffi.register(
          (s: number, e: number, t: string, _ud: unknown) =>
            onSegment(s, e, t || ""),
          koffi.pointer(SegmentCallbackC)
        )
      : null;
    try {
      const text = this._l.baseRT_transcribe_stream(
        this.handle,
        wavPath,
        language ?? null,
        stats,
        cb,
        null
      );
      return { text: text || "", stats: toTranscribeStats(stats) };
    } finally {
      if (cb) koffi.unregister(cb);
    }
  }

  transcribePcmStream(
    samples: Float32Array,
    language?: string,
    onSegment?: SegmentCallback
  ): TranscribeResult {
    const stats: Record<string, number> = {};
    const cb: koffi.IKoffiRegisteredCallback | null = onSegment
      ? koffi.register(
          (s: number, e: number, t: string, _ud: unknown) =>
            onSegment(s, e, t || ""),
          koffi.pointer(SegmentCallbackC)
        )
      : null;
    try {
      const text = this._l.baseRT_transcribe_pcm_stream(
        this.handle,
        samples,
        samples.length,
        language ?? null,
        stats,
        cb,
        null
      );
      return { text: text || "", stats: toTranscribeStats(stats) };
    } finally {
      if (cb) koffi.unregister(cb);
    }
  }
}

export default BaseRTModel;

/**
 * Internal handles exposed for ABI tests only (verifying struct offsets match
 * include/baseRT/types.h). Not part of the public API — do not depend on these.
 */
export const __internal = {
  ModelConfigC,
  SamplingConfigC,
  GenerationStatsC,
  TranscribeStatsC,
  toSamplingC,
  toModelConfig,
};
