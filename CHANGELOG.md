# Changelog

All notable changes to BaseRT. This project is pre-1.0, so minor versions may
include behavior changes. Format loosely based on
[Keep a Changelog](https://keepachangelog.com).

## [0.2.0] — 2026-07-31

**A second hardware backend.** BaseRT now runs on **NVIDIA GB10 (DGX Spark,
Linux/arm64)** alongside Apple Silicon — the same `.base` bundles, the same
`basert` CLI and OpenAI-compatible server, now on CUDA. This release also brings
the **Qwen3.5 / Qwen3.6 hybrid** model family and a **continuous-batching**
overhaul of `basert serve`.

### Added

- **CUDA backend — NVIDIA GB10 / DGX Spark.** Full inference on Linux/arm64 + CUDA:
  dense, MoE, and hybrid (Gated-DeltaNet) architectures, across prefill,
  single-stream decode, and continuous-batching serving. `install.sh` auto-detects
  the platform (macOS/arm64 → Metal, Linux/arm64 → CUDA) and `basert pull` resolves
  the right bundle for the host.
- **Qwen3.5 & Qwen3.6 hybrid models.** Support for the Gated-DeltaNet hybrid
  architecture (dense **and** MoE), including vision (Qwen3.5-VL), on both Metal and
  CUDA. GB10 continuous batching for the dense hybrids.
- **Continuous-batching serving.** `basert serve` now drives **tool calls**
  (streaming + non-streaming), **grammar-constrained decode**, **per-token
  logprobs**, and **`n>1` choices** through the continuous-batching engine, with a
  decode-priority scheduler and radix prefix-cache reuse (dense and hybrid).
- **CUDA-native model bundles.** `cuda-q8` and `cuda-q4mix` variants for 11 models
  (Qwen3-0.6B/30B-A3B, Qwen3.5-2B/35B-A3B, Qwen3.6-27B/35B, Llama-3.2-1B/3B,
  Gemma-3-1B, Gemma-4-E2B/26B), on Hugging Face and in the catalog.
- **Faster Metal GEMMs.** Large-tile prefill GEMM kernels for native bf16 / f16
  weights and Q8, plus batched Gated-DeltaNet decode/prefill kernels for the M1–M5
  families.
- **Tokenizer conformance.** An HF-exact conformance harness and a Unicode-category
  pretokenizer; Phi-3-mini un-quarantined after revalidation.

### Fixed

- **CUDA model resolution.** `basert pull`/`chat <id>` on NVIDIA GB10 now fetches
  the CUDA-native bundle instead of the universal one it couldn't load
  (`no function named 'gemv_q4'`). The 0.2.0 Linux/arm64 CUDA download was updated
  in place with this fix.
- **Chunked-decode correctness on CUDA.** Free generation past the decode chunk size
  no longer repeats an earlier block (an eager-dispatch write-after-read hazard);
  prefill and teacher-forced decode were always correct.
- **Large quantized tensors.** Weight tensors with more than ~512M elements no longer
  decode to garbage (64-bit byte-offset fix).
- **Gemma normalization precision.** The canonical post-attention / feed-forward norm
  weights are kept in floating point in the shipped bundles.
- **Hybrid prefix reuse.** Correct KV/GDN dtype handling and prefix-cache reuse for
  hybrid-GDN models under serving.
- **MoE quantization parity.** Qwen3.5 MoE quant rules matched to canonical tensor
  names so experts quantize as intended.

### Changed

- **Backend-aware model resolution.** CUDA hosts prefer CUDA-native bundles, Apple
  Silicon prefers Metal, with a universal fallback; catalog entries are
  backend-tagged and can be derived from a `.base` header.
- **Performance.** Metal decode/prefill tuning across the M1 Max / M4 Pro / M5 Pro
  families (large-tile GEMM, higher-occupancy Gemma-4 MoE gate/up).

### Install

```sh
# macOS (Apple Silicon, Metal) or Linux (arm64, CUDA) — auto-detected
curl -fsSL https://raw.githubusercontent.com/basecompute/baseRT/main/install.sh | sh
basert pull <model>          # resolves the right bundle for your host
basert serve <model>.base    # OpenAI-compatible server
```

## [0.1.7]

Maintenance release focused on the three most-reported serve/CLI issues, plus
converter and chat quality-of-life fixes.

### Fixed

- **Tool calls for Qwen 3.5 / 3.6** (#21) — structured `tool_calls` for the Qwen XML
  tool-call dialect, streaming and non-streaming; truncated generations never execute
  partial arguments; logprobs stay aligned across tool-call segments.
- **Deterministic HTTP 500 on some short CJK prompts** (#22) — fixed a byte-alphabet
  hole (soft hyphen `0xAD`) in GPT-2-style byte tokenization.
- **`--no-think` and chat-template handling** (#23) — `--no-think` across all
  thinking-capable architectures; render model-embedded chat templates; fixed a
  Gemma 4 channel-marker leak and KV-reuse correctness on divergent history.
- **MLX 3/5/6-bit conversion** — quantized MLX checkpoints at 3/5/6-bit unpack
  correctly (little-endian bitstream).
- **`basert pull` HF-cache duplication** — pulled models are no longer duplicated in
  the global Hugging Face cache.

### Added

- **Chat line editing** — cursor movement, kill ops, word delete, with UTF-8/CJK.
- **q6 MoE kernels** — Q6 expert GEMM + a `default-q6` conversion profile.
- **`baseRT_decode_token_raw()`** — length-preserving raw-byte token decode.

### Changed

- Chat / complete / bench banners show the model id (`org/repo · variant`).
- `basert` help output regrouped runtime-first.

## 0.1.6 and earlier

See the corresponding GitHub release notes.
