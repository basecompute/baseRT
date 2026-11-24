/**
 * Unit tests for @baseRT/node.
 *
 * These run WITHOUT a GPU, model file, or the native shared library: koffi's
 * struct/type definitions are pure (the dylib is only loaded lazily on first
 * model construction), so we can assert the FFI struct layout matches the C
 * ABI in include/baseRT/types.h, plus the pure JS helpers. The last section
 * compiles a stand-in engine with `cc` to cover a library on the previous
 * BaseRTModelConfig layout (skipped where there is no C compiler).
 *
 * Everything lives in this one file: koffi registers struct names on its
 * shared native instance, so a second test file importing ../index would fail
 * with "Duplicate type name" under --runInBand.
 */

import { execFileSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import koffi from "koffi";
import { BaseRTModel, __internal, type SamplingConfig } from "../index";

const {
  ModelConfigC,
  SamplingConfigC,
  GenerationStatsC,
  TranscribeStatsC,
  TranscribeSegmentC,
  toSamplingC,
} = __internal;

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

  test("BaseRTModelConfig is 2144 bytes, 8-byte aligned, with the qwen4_exp tail", () => {
    // Exact sizeof(BaseRTModelConfig) from include/baseRT/types.h; must match
    // the Python and Rust mirror tests. The qwen4_exp block starts where the
    // struct ended before it (1792) and its uint64_t ngram_mult[4] raised the
    // alignment from 4 to 8 (2108 pads to 2112).
    expect(koffi.sizeof(ModelConfigC)).toBe(2144);
    expect(koffi.alignof(ModelConfigC)).toBe(8);
    expect(koffi.offsetof(ModelConfigC, "rope_yarn_beta_fast")).toBe(1772);
    expect(koffi.offsetof(ModelConfigC, "hc_count")).toBe(1792);
    expect(koffi.offsetof(ModelConfigC, "ngram_head_vocab")).toBe(1832);
    expect(koffi.offsetof(ModelConfigC, "ngram_head_off")).toBe(1960);
    expect(koffi.offsetof(ModelConfigC, "indexer_n_heads")).toBe(2088);
    expect(koffi.offsetof(ModelConfigC, "_q4_pad")).toBe(2104);
    expect(koffi.offsetof(ModelConfigC, "ngram_mult")).toBe(2112);
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

  test("BaseRTTranscribeSegment is 32 bytes with text at offset 8", () => {
    // 2 x int, 8-byte-aligned char*, 4 x float — matches the C struct in
    // include/baseRT/baseRT.h.
    expect(koffi.sizeof(TranscribeSegmentC)).toBe(32);
    expect(koffi.offsetof(TranscribeSegmentC, "text")).toBe(8);
    expect(koffi.offsetof(TranscribeSegmentC, "avg_logprob")).toBe(16);
    expect(koffi.offsetof(TranscribeSegmentC, "temperature")).toBe(28);
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

// ---------------------------------------------------------------------------
// Mixed-version engine: the binding and the installed libbaseRT disagree on
// sizeof(BaseRTModelConfig)
// ---------------------------------------------------------------------------

// Every C symbol the binding resolves at load: koffi fails k.func() on a
// missing one, so the stand-in exports each of them (never called here).
const BINDING_SOURCE = fs.readFileSync(path.join(__dirname, "..", "index.ts"), "utf8");
const ALL_SYMBOLS = Array.from(
  new Set(BINDING_SOURCE.match(/\bbaseRT_[A-Za-z0-9_]+\b/g) ?? [])
).sort();

// The previous layout (0.5.2, before the qwen4_exp block) ends where the first
// appended field begins.
const OLD_SIZE = koffi.offsetof(ModelConfigC, "hc_count");

const IMPLEMENTED = new Set([
  "baseRT_load_model",
  "baseRT_free_model",
  "baseRT_get_error",
  "baseRT_version_string",
  "baseRT_model_config_sizeof",
  "baseRT_get_config",
  "baseRT_get_config_into",
]);

function hasCC(): boolean {
  try {
    execFileSync("cc", ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

/** Compile an engine built against the previous header, with or without the sized getter. */
function buildOldEngine(dir: string, sizedGetter: boolean): string {
  const off = (f: string) => koffi.offsetof(ModelConfigC, f);
  const stubs = ALL_SYMBOLS.filter((s) => !IMPLEMENTED.has(s))
    .map((s) => `void ${s}(void) {}`)
    .join("\n");
  const sized = sizedGetter
    ? `
size_t baseRT_get_config_into(void *m, void *out, size_t out_size) {
    size_t n = out_size < OLD_SIZE ? out_size : OLD_SIZE;
    if (!m) { memset(out, 0, out_size); return OLD_SIZE; }
    memcpy(out, &g_cfg, n);
    if (out_size > n) memset((unsigned char *)out + n, 0, out_size - n);
    return OLD_SIZE;
}`
    : "";
  const src = `
#include <stddef.h>
#include <string.h>
#define OLD_SIZE ${OLD_SIZE}
typedef struct { unsigned char b[OLD_SIZE]; } OldConfig;
static OldConfig g_cfg;
static void fill(void) {
    unsigned int v;
    memset(&g_cfg, 0, sizeof g_cfg);
    v = 4096u;   memcpy(g_cfg.b + ${off("dim")}, &v, 4);
    v = 32u;     memcpy(g_cfg.b + ${off("n_layers")}, &v, 4);
    v = 151936u; memcpy(g_cfg.b + ${off("vocab_size")}, &v, 4);
    memcpy(g_cfg.b + ${off("architecture")}, "qwen3", 6);
    /* _gptoss_pad, the old struct's last byte: a copy that stops short shows here */
    g_cfg.b[OLD_SIZE - 1] = 0xA5;
}
void *baseRT_load_model(const char *p, const char *k, int c) { (void)p; (void)k; (void)c; fill(); return &g_cfg; }
void baseRT_free_model(void *m) { (void)m; }
const char *baseRT_get_error(void) { return "stand-in engine"; }
const char *baseRT_version_string(void) { return "0.5.2-stand-in"; }
size_t baseRT_model_config_sizeof(void) { return OLD_SIZE; }
OldConfig baseRT_get_config(void *m) { (void)m; return g_cfg; }
${sized}
${stubs}
`;
  const tag = sizedGetter ? "sized" : "legacy";
  const srcPath = path.join(dir, `engine-${tag}.c`);
  const ext = process.platform === "darwin" ? "dylib" : "so";
  const libPath = path.join(dir, `libbaseRT-prev-${tag}.${ext}`);
  fs.writeFileSync(srcPath, src);
  execFileSync("cc", ["-shared", "-fPIC", "-o", libPath, srcPath], { stdio: "pipe" });
  return libPath;
}

const describeIfCC = hasCC() ? describe : describe.skip;

describeIfCC("mixed-version engine (previous BaseRTModelConfig layout)", () => {
  let dir: string;
  const savedPath = process.env.BASERT_LIB_PATH;

  beforeAll(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "basert-abi-"));
  });

  afterEach(() => {
    if (savedPath === undefined) delete process.env.BASERT_LIB_PATH;
    else process.env.BASERT_LIB_PATH = savedPath;
    __internal.resetLib();
  });

  afterAll(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  test("the previous layout is really smaller than the mirror", () => {
    expect(OLD_SIZE).toBe(1792);
    expect(koffi.sizeof(ModelConfigC)).toBeGreaterThan(OLD_SIZE);
  });

  test("an engine with the sized getter loads and its shared fields read back; appended fields are zero", () => {
    process.env.BASERT_LIB_PATH = buildOldEngine(dir, true);
    __internal.resetLib();

    const m = new BaseRTModel("any.base");
    try {
      const cfg = m.config;
      expect(cfg.dim).toBe(4096);
      expect(cfg.nLayers).toBe(32);
      expect(cfg.vocabSize).toBe(151936);
      expect(cfg.architecture).toBe("qwen3");

      // Through the raw sized getter, into a full-size mirror.
      const l = (m as any)._l;
      const handle = (m as any)._handle;
      const sizeofInEngine = koffi.load(process.env.BASERT_LIB_PATH as string)
        .func("size_t baseRT_model_config_sizeof()");
      expect(Number(sizeofInEngine())).toBe(OLD_SIZE);
      const out: Record<string, any> = {};
      const libSize = Number(
        l.baseRT_get_config_into(handle, out, koffi.sizeof(ModelConfigC))
      );
      expect(libSize).toBe(OLD_SIZE);
      expect(out.dim).toBe(4096);
      expect(out._gptoss_pad).toBe(0xa5); // the old struct's last byte arrived
      // Everything the previous layout lacks reads as zero / empty.
      expect(out.hc_count).toBe(0);
      expect(out.indexer_n_heads).toBe(0);
      expect(out._q4_pad).toBe(0);
      expect(Array.from(out.ngram_head_vocab as ArrayLike<number>).every((v) => Number(v) === 0)).toBe(true);
      expect(Array.from(out.ngram_mult as ArrayLike<number | bigint>).map(Number)).toEqual([0, 0, 0, 0]);
    } finally {
      m.close();
    }
  });

  test("an engine without the sized getter and another size is still refused", () => {
    process.env.BASERT_LIB_PATH = buildOldEngine(dir, false);
    __internal.resetLib();
    expect(() => new BaseRTModel("any.base")).toThrow(/layout drift/);
    // The refused library was not cached as a half-checked table.
    expect(() => new BaseRTModel("any.base")).toThrow(/layout drift/);
  });
});
