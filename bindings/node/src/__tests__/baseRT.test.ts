/**
 * Unit tests for @baseRT/node.
 *
 * These run WITHOUT a GPU, model file, or the native shared library: koffi's
 * struct/type definitions are pure (the dylib is only loaded lazily on first
 * model construction), so we can assert the FFI struct layout matches the C
 * ABI in include/baseRT/types.h, plus the pure JS helpers.
 */

import koffi from "koffi";
import { __internal, type SamplingConfig } from "../index";

const { ModelConfigC, SamplingConfigC, GenerationStatsC, TranscribeStatsC, toSamplingC } =
  __internal;

describe("C struct ABI (offsets match include/baseRT/types.h)", () => {
  test("BaseRTModelConfig includes sliding_window and places architecture at offset 60", () => {
    // The previous hand-written binding dropped the `sliding_window` field,
    // which mis-placed architecture (it read offset 56 instead of 60) and every
    // encoder field after it. Assert the corrected layout.
    expect(koffi.offsetof(ModelConfigC, "sliding_window_pattern")).toBe(48);
    expect(koffi.offsetof(ModelConfigC, "sliding_window")).toBe(52);
    expect(koffi.offsetof(ModelConfigC, "rope_local_theta")).toBe(56);
    expect(koffi.offsetof(ModelConfigC, "architecture")).toBe(60);
    expect(koffi.offsetof(ModelConfigC, "enc_n_layers")).toBe(92); // 60 + 32
  });

  test("BaseRTModelConfig is much larger than the old 112-byte assumption", () => {
    // Real struct carries Gemma4 + MoE + vision + audio fields — returning it
    // by value with a 112-byte type would corrupt caller memory.
    expect(koffi.sizeof(ModelConfigC)).toBeGreaterThan(1000);
  });

  test("Gemma4 / MoE / tower fields are present and ordered", () => {
    expect(koffi.offsetof(ModelConfigC, "swa_layers")).toBeGreaterThan(
      koffi.offsetof(ModelConfigC, "head_dim_global")
    );
    expect(koffi.offsetof(ModelConfigC, "n_experts")).toBeGreaterThan(
      koffi.offsetof(ModelConfigC, "ffn_dims")
    );
    expect(koffi.offsetof(ModelConfigC, "vision_n_layers")).toBeGreaterThan(
      koffi.offsetof(ModelConfigC, "_moe_pad")
    );
    expect(koffi.offsetof(ModelConfigC, "audio_n_layers")).toBeGreaterThan(
      koffi.offsetof(ModelConfigC, "eoi_token_id")
    );
  });

  test("BaseRTSamplingConfig matches the extended 0.2 layout", () => {
    expect(koffi.offsetof(SamplingConfigC, "temperature")).toBe(0);
    expect(koffi.offsetof(SamplingConfigC, "top_k")).toBe(4);
    expect(koffi.offsetof(SamplingConfigC, "seed")).toBe(28);
    expect(koffi.offsetof(SamplingConfigC, "n_logit_bias")).toBe(32);
    // pointer fields are 8-byte aligned after n_logit_bias (+ 4 bytes padding)
    expect(koffi.offsetof(SamplingConfigC, "logit_bias_tokens")).toBe(40);
    expect(koffi.offsetof(SamplingConfigC, "logit_bias_values")).toBe(48);
  });

  test("BaseRTGenerationStats is 24 bytes (6 x 4-byte fields)", () => {
    expect(koffi.sizeof(GenerationStatsC)).toBe(24);
  });

  test("BaseRTTranscribeStats is 20 bytes (5 x 4-byte fields)", () => {
    expect(koffi.sizeof(TranscribeStatsC)).toBe(20);
  });
});

describe("toSamplingC defaults", () => {
  test("greedy defaults when no options given", () => {
    const s = toSamplingC();
    expect(s.temperature).toBe(0.0);
    expect(s.top_k).toBe(40);
    expect(s.top_p).toBeCloseTo(0.9);
    expect(s.min_p).toBe(0.0);
    expect(s.repeat_penalty).toBe(1.0);
    expect(s.n_logit_bias).toBe(0);
    expect(s.logit_bias_tokens).toBeNull();
  });

  test("explicit zero values are respected (nullish coalescing, not ||)", () => {
    const s = toSamplingC({ topK: 0, topP: 0 } as SamplingConfig);
    expect(s.top_k).toBe(0);
    expect(s.top_p).toBe(0);
  });

  test("logitBias is packed into parallel typed arrays", () => {
    const s = toSamplingC({ logitBias: { 10: 1.5, 20: -2.0 } });
    expect(s.n_logit_bias).toBe(2);
    expect(s.logit_bias_tokens).toBeInstanceOf(Int32Array);
    expect(s.logit_bias_values).toBeInstanceOf(Float32Array);
    expect(Array.from(s.logit_bias_tokens as Int32Array)).toEqual([10, 20]);
    expect(Array.from(s.logit_bias_values as Float32Array)).toEqual([1.5, -2.0]);
  });

  test("OpenAI-compat penalties and seed pass through", () => {
    const s = toSamplingC({
      presencePenalty: 0.5,
      frequencyPenalty: 0.3,
      seed: 42,
    });
    expect(s.presence_penalty).toBe(0.5);
    expect(s.frequency_penalty).toBeCloseTo(0.3);
    expect(s.seed).toBe(42);
  });
});

describe("library loading is lazy (no native load at import)", () => {
  test("importing the module did not throw despite no dylib present", () => {
    // If koffi.load were eager this import would have failed in CI without a
    // built dylib. Reaching here proves it's deferred to model construction.
    expect(typeof __internal.toModelConfig).toBe("function");
  });
});
