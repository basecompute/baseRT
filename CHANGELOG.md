# Changelog

All notable changes to BaseRT. This project is pre-1.0, so minor versions may
include behavior changes. Format loosely based on
[Keep a Changelog](https://keepachangelog.com).

## [0.5.1] — 2026-11-16

**`basert serve` uses speculative decoding by default.** A model drafts with
the drafter the catalog pairs with it, or with the MTP head its file carries;
`--speculate off` serves it plainly. The C API also reports GPU faults.

### Changed

- **Speculative decoding by default in `basert serve`.** A start that names no
  `--speculate` strategy works like `--speculate auto`, and downloads the
  paired drafter first when it is not installed (1.4 GB for Qwen3.8-27B, 458 MB
  for Qwen3.6-35B-A3B on a Mac). `--offline` downloads nothing. Greedy output
  can differ slightly from plain decoding unless `--spec-bitexact` is set. The
  automatically sized context window leaves room for the drafter.

### Added

- **GPU faults in the C API.** When a GPU command fails, `baseRT_prefill`,
  `baseRT_decode_step` and the logits reads fail with `BASERT_ERR_GPU` and
  leave the KV cache as it was, where they used to return success with the
  logits unwritten. `baseRT_gpu_error()` reports a fault, and forward passes
  refuse until `baseRT_clear_gpu_error()` is called.
- **More quantization formats for mixture-of-experts models on Macs.** Expert
  weights in GGUF `Q4_K`, `Q5_K` or `IQ4_XS`, or in `base_q5` with groups of
  64, run with their own kernels. `basert convert --kquant-passthrough` keeps
  `Q4_K`, `Q5_K` and `IQ4_XS` expert weights from a GGUF as they are.
- **AWQ in the converter.** `basert convert` runs AWQ's scale search itself and
  folds the scales into the model's weights, so the result needs no special
  kernel. New commands for quantization work: `basert overlay`, `merge`,
  `fakequant`, `awq-search`, `fingerprint` and `inventory`.

### Fixed

- BaseQuant-V1 models load with their projections fused, without the
  `k-quant fuse refused` warnings; output is unchanged.
- DGX Spark no longer refuses a model that fits: the free-memory check now
  counts the file cache Linux hands back on demand.
- Models with 6-bit mixture-of-experts weights are no longer marked as
  NVIDIA-only by the converter. Re-convert an older file to run it on a Mac.

The engine ABI is unchanged since 0.4.0 (superfluid 0.1.13 serves it).

## [0.5.0] — 2026-11-12

**BaseRT runs on AMD GPUs.** The engine, the `.base` files and `basert serve`
now run on Linux x86-64 with an AMD Ryzen AI Max (Strix Halo: Radeon 8060S or
8050S) or a Ryzen AI 300 / 400 with a Radeon 890M or 880M (Strix Point,
Gorgon Point).

### Added

- **AMD GPUs on Linux.** A new `basert-engine-linux-x64-rocm` bundle, taken by
  the install script, `pip install basert`, npm and Homebrew on Linux. Needs
  Ubuntu 24.04 or newer (glibc 2.39+), ROCm 10 from AMD's repository
  (`amdrocm-blas10.0-gfx1151` on Strix Halo, `-gfx1150` on Strix Point and
  Gorgon Point) and a recent kernel (Ubuntu 24.04's HWE kernel
  6.17.0-19.19~24.04.2 or later, its OEM kernel 6.14.0-1018 or later, or Linux
  6.18.4). `basert self update` works on Linux x86-64.
- **What runs on AMD:** dense and mixture-of-experts text models, the Qwen3.5
  and 3.6 hybrids, and gemma-4 with image input, with paged KV, prefix caching,
  batching and a quantized KV cache, served through superfluid.
- **Speculative decoding on AMD** with MTP heads, DFlash, DSpark and EAGLE-3.

### Changed

- **Long context on DGX Spark.** gemma-4-E2B and 26B-A4B prefill 8K–32K-token
  prompts 5 to 38 times as fast as in 0.4.7, and gemma-4 and Qwen3-8B decode
  1.6 to 4.6 times as fast at a 16K–32K context.
- gemma-4-26B-A4B decodes 6.7% faster on M5 Pro Macs, with the same output.

### Fixed

- Qwen3.6-35B-A3B prefill on M5 and M6 Macs uses the new expert kernels again
  (4–8% faster than the fallback).
- Qwen3.5 and 3.6 prompts on NVIDIA GPUs are processed in full-size chunks
  again, restoring 0.4.6's speed.
- gemma-4-26B-A4B serving no longer leaks GPU memory per generated token.
- A library user that exits with a Qwen3.5 or 3.6 model still loaded no longer
  aborts at exit on M4, M5 and M6 Macs.
- When several long requests outgrow the KV cache, the engine declines the step
  before changing anything instead of failing partway with an HTTP 500.
- Qwen3.5 and 3.6 loads on NVIDIA no longer print a `CUDA_ERROR_NOT_FOUND` line.
- A file that packs its quantization groups across rows is refused at load,
  naming the tensor, instead of giving wrong output.

## [0.4.7] — 2026-11-10

**Prefill with the Neural Engine on Apple Silicon**, and `basert` on PyPI.

### Added

- **Prefill on the Neural Engine.** On M4-, M5- and M6-generation Macs, a large
  dense model's feed-forward layers, and the Qwen3.5 / 3.6 / 3.8 hybrids'
  recurrent and attention projections, are split between the GPU and the
  Neural Engine during prefill. Qwen3.8-27B at Q4 prefills 1k-token prompts at
  about 700 tokens/s on an M5 Pro, from 531. The split computes in 8-bit
  integers, so prefill output is no longer bit-for-bit the GPU's; perplexity is
  unchanged within noise and decoding is untouched.
- Turn it off with `--no-ane` on `basert bench`, `chat` and `complete`,
  `baseRT_set_neural_engine(0)` before loading (`setNeuralEngine(false)` in the
  Swift and Node bindings), or superfluid's `--basert-neural-engine off`.
  `baseRT_prepare_neural_engine()` compiles a model's Neural Engine programs
  ahead of its first load.
- **`pip install basert`** installs the `basert` command, the prebuilt engine
  and the Python bindings (macOS 15+ on Apple Silicon, Linux arm64 with CUDA 13;
  Python 3.9+). Published since 0.4.6.

### Changed

- The hybrid models keep their recurrent state at every 64-, 256- and
  1024-token boundary, so a served prompt reuses more of a shared prefix.
- A DFlash drafter no longer holds up the first token.
- Faster prefill on 32-core and larger M5 GPUs, and faster speculative verify
  on 16- to 20-core M5 GPUs.

## [0.4.6] — 2026-11-06

### Added

- `basert serve <model> --speculate auto` drafts Qwen3.8-27B and
  Qwen3.6-35B-A3B with their DFlash2 drafters, downloading the drafter the first
  time (Apple Silicon; needs superfluid 0.1.10).

### Changed

- Faster DFlash2 speculative decoding on Apple Silicon (Qwen3.6-35B-A3B 179.0 →
  214.8 tokens/s at one request on an M5 Pro). `BASERT_SPEC_BITEXACT=1` keeps
  the earlier, bitwise-plain route.
- gemma-4-26B-A4B decodes faster on Apple Silicon (17.04 → 13.34 ms a token on
  an M5 Pro), and its MTP head accepts more drafts. Its token table is held at
  8 bits in memory, which changes plain-decode output slightly.

### Fixed

- `--speculate dflash:<path>.base` with the drafter as a separate word no
  longer reads the drafter as a second model to serve.

## [0.4.5] — 2026-11-05

### Added

- Speculative decoding on gemma-4-26B-A4B with Google's own assistant head,
  now carried in the catalog bundles (`--speculate mtp-head` or `auto`).
- `basert add-speculator` embeds a head converted with `--quantize-drafter`,
  not only f16 heads.
- Releases that come in sizes rather than quants (the BaseQuant repositories,
  `-XS` … `-XL`) pull their Medium by default; a size names a variant
  (`basert pull <id>:small`), and `basert pull <id> --dry-run` says which file
  it would take.

### Changed

- On Apple Silicon, mixture-of-experts models verify speculative drafts on the
  batched decode route (three rows at 1K context: 58 → 30 ms).

## [0.4.4] — 2026-11-03

### Added

- Speculative decoding for Qwen3.8-27B and Qwen3.6-35B-A3B on Apple Silicon
  with the DFlash 2 block drafters, paired in the catalog
  (`--speculate dflash:basecompute/Qwen3.8-27B-DFlash2`).

### Changed

- Gemma 4 prompts served from a paged KV cache prefill at `basert bench` speed.
- Under several concurrent requests, a request a gate takes out of
  speculation keeps drafting along with the others instead of decoding plainly.
- Bundles converted from GGUF declare tensor shapes in the same order as every
  other source. Older bundles keep working; re-converting restores fused QKV
  attention for models with narrower K/V projections.

### Fixed

- Logits past 2 GiB in a 16,384-token prefill chunk came out zero on Apple
  Silicon (32-bit tile offsets).
- Some served prompts of 512–4,096 tokens ran their first chunk without a
  causal mask on Qwen3.5 / 3.6 full-attention layers (present since 0.3).
- Qwen3.5-2B at Q4 repeated its first token on a single request.
- Vocabulary entries shaped like markers (`<?>`, `[:]`) are tokenized as
  ordinary text; only special added tokens are cut out.
- `basert self update --dry-run` names the superfluid the update would bring.

## [0.4.3] — 2026-10-29

### Added

- **`basert self update`** updates an install-script install: the newest
  release is unpacked under `~/.basert/versions/<v>/`, its SHA-256 and launcher
  checked, and made current in one step. `basert self use <v>` goes back,
  `self check` and `self list` report. Package-manager installs are updated by
  their package manager.
- Speculative decoding on Gemma 4 E2B and E4B with Google's assistant head,
  carried in the catalog bundles (`--speculate mtp-head` or `auto`).
  `basert add-speculator` and `basert convert --mtp-head` embed a head in a
  bundle of your own.

### Changed

- Gemma 4 draft verification runs on the GPU argmax and decode attention
  kernels on Apple Silicon.

### Fixed

- superfluid installed by Homebrew, npm, pip, uv or cargo now goes to BaseRT's
  data directory and survives package upgrades.

## [0.4.2] — 2026-10-27

### Added

- **Package registries.** `brew install basecompute/tap/basert`, `npm install
  -g @basecompute/basert` (the command and engine) or `npm install
  @basecompute/basert` (Node bindings), and the `baseRT` / `baseRT-sys` crates
  plus `cargo install basert-cli` on crates.io. For Apple Silicon on macOS 15+
  and Linux arm64 with an NVIDIA GPU and CUDA 13.
- The engine license covers Linux systems with an NVIDIA GPU.
- `basert pull <model>:<variant>` pulls a named variant; an installed variant is
  used without contacting Hugging Face.

### Changed

- Faster batched decode of GGUF k-quant bundles on M1 Max, M3 Ultra, M4 Max,
  M5 Max and M6, and faster k-quant prefill on M5.
- 8-bit group-64 bundles prefill up to 2.3× faster on M5.

### Fixed

- Gemma 4 requests that declare tools no longer spend seconds tokenizing.

## [0.4.1] — 2026-10-23

### Added

- `basert pull <id> --file <name>` downloads a `.base` by file name;
  `basert list --json` reports the file each install came from.

### Changed

- When superfluid is missing, `basert serve` installs its newest release that
  serves this engine.
- Batched decode of GGUF k-quant bundles is 1.5–3.5× faster on M1, M3, M4, M5
  and M6 Macs.
- A pull with no matching quant lists the repo's files and suggests `--file`.

### Fixed

- Gemma 4 decodes up to four times faster through `basert serve` with long
  prompts.

## [0.4.0] — 2026-10-22

### Added

- GGUF `Q3_K`, `IQ4_XS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS` and `IQ3_S` weights run
  natively on Macs, alongside `Q4_K`, `Q5_K` and `Q6_K`.
- `basert serve` installs and runs superfluid's own release.
- The engine library carries an ABI number (`BASERT_ABI_VERSION`,
  `baseRT_abi_version()`), published with each release as
  `basert-engine-abi-<N>.txt`.

### Fixed

- Qwen 3.5 and 3.6 mixture-of-experts models decode up to 12% faster on Mac.
- Qwen 3.8 bundles with GGUF weights no longer decode garbage on M5 Max.
- A bundle with weights this build cannot run is refused when it opens.

## [0.3.0] — 2026-10-08

**`basert serve` is a new server.** It keeps the command line and the
OpenAI-compatible API, and runs them on
[superfluid](https://github.com/basecompute/superfluid). This release also
carries the fixes prepared for 0.2.7, which was not published.

### Added

- **Serving.** Continuous batching and prefix reuse are always on;
  `--continuous-batching [N]` or `--max-batch-size N` serves several requests
  at once. Streamed tool calls arrive as partial deltas, `tool_choice` naming a
  function or `required` is honoured, `/v1/completions` accepts `suffix`, and
  `/v1/models` lists each model's capabilities.
- **Larger windows for hybrid models.** The paged KV cache charges only the
  attention layers, so Qwen 3.5 / 3.6 / 3.8 get context windows up to four
  times larger from the same memory.
- GLM 5.2 serves with continuous batching on Metal and speculates with its MTP
  head.
- Block drafters with a reduced draft vocabulary convert and serve;
  `basert complete --spec` turns speculation on for a bundle with its own head.

### Changed

- Flags the new server has no use for are accepted and noted on stderr.
- Server state lives under `$BASERT_STATE_DIR`, one directory per port.
- `/v1/models` reports the window as `meta.n_ctx`, `/metrics` series are named
  `superfluid_*`, and replies say `system_fingerprint: superfluid`.
- Text that looks like a chat marker inside a message is content, never
  framing.

### Fixed

- Gemma 4 26B-A4B misread every image (a vision-encoder step was skipped).
- Hybrid-model prefix reuse survives a replayed or branched conversation.
- Llama-family models no longer produce garbage in batch-invariant mode
  (`BASERT_BATCH_INVARIANT=1`).
- On NVIDIA, a LoRA adapter on Qwen 3.5's shared-expert gate is applied during
  generation, and two mixture-of-experts paths read mixed scale formats
  correctly.
- The GPU sampler treated `top_k: 0` as greedy.

## [0.2.6] — 2026-09-18

### Fixed

- Qwen3.x 27B and 35B-A3B models produced garbage after the first token on
  M1-family Macs (0.2.4 and 0.2.5).

### Changed

- A GPU dispatch the hardware would silently skip now stops with an error
  naming the kernel.

## [0.2.5] — 2026-09-17

### Added

- **Streaming tool calls.** With `stream: true`, a tool call arrives as an
  opening `delta.tool_calls` entry with its name, then `function.arguments`
  fragments. A call cut short by the token budget keeps
  `finish_reason: "length"`.
- `basert-benchmark-harness` reports telemetry from the measured run and gains
  `--headline-first`, `--isolated-workloads`, `--workload` and
  `--idle-baseline`.

### Changed

- Benchmark report schema: same-run telemetry is `basert-telemetry/4`
  (breaking for report consumers).

### Fixed

- `basert pull <org/model>` from Hugging Face failed with
  `no .safetensors shards` after downloading the whole model.

## [0.2.4] — 2026-09-09

**Three new model families, and a context window that sizes itself.**

### Added

- **Automatic context sizing.** `basert serve`, `chat` and `complete` derive
  the window from the GPU memory available, the weights and the KV cost per
  token, capped at the model's trained window. `--max-context N` pins one.
- **gpt-oss** (`basecompute/gpt-oss-20b`, bf16 / Q8 / Q4), with its MXFP4
  experts carried verbatim and Harmony tool calls parsed.
- **Nemotron 3 Nano** (`basecompute/NVIDIA-Nemotron-3-Nano-30B-A3B`), a
  Mamba-2 hybrid.
- **GLM 5.2**, with multi-head latent attention and the sparse indexer.

### Changed

- **`BaseRTModelConfig` changed layout** — recompile anything built against
  0.2.3 that reads it.
- The shared library reports its real version, and the macOS archive ships the
  symlink its install name needs.

### Fixed

- gpt-oss NaNs and runaway generation after a tool call; a stray end-of-turn
  token on large models under batching; LoRA adapters ignoring penalties,
  `logit_bias` and `seed`; a temporary file left by a failed conversion.

## [0.2.3] — 2026-08-19

**Model pulls that finish.**

### Added

- `basert pull` downloads over many connections at once and resumes per chunk
  after an interruption, across process restarts. A stalled transfer is cut
  off and retried, and a short one is reported instead of installed. Tunable
  with `BASERT_HF_CONNECTIONS`, `BASERT_HF_CHUNK_MB`,
  `BASERT_HF_READ_TIMEOUT_SECS` and `BASERT_HF_MAX_RETRIES`.
- Xet transfers behind `BASERT_HF_XET=1` (deduplicated, not resumable).

### Fixed

- Muse Glimmer's tool calls are parsed as tool calls.
- `--kquant-passthrough` embeddings on models other than Muse Glimmer.
- `developer` and `function` roles no longer break a model's native template.
- Chat templates using `is undefined` render instead of falling back to
  ChatML.

### Changed

- Building `base-convert` requires Rust 1.85+.

## [0.2.2] — 2026-08-12

**Muse Glimmer, and native GGUF k-quants.**

### Added

- Muse Glimmer 30B, text and vision (`basecompute/Muse-Glimmer-30B`).
- GGUF `Q4_K` / `Q5_K` / `Q6_K` (and `Q2_K`, `Q3_K`) run natively on Apple
  Silicon. `basert convert --kquant-passthrough` copies GGUF super-blocks into a
  `.base` verbatim, and `--mmproj` folds a companion `mmproj-*.gguf` vision
  tower into the same bundle.

### Changed

- Short prompts prefill 1.4–1.55× faster on M5; k-quant prefill 3.4× faster.

### Fixed

- M3 Ultra was identified as a base M3.
- The Python, Rust and Node mirrors of the model-config struct had fallen
  behind the C header; a test now holds them in step.
- Models converted from GGUF ignored the end-of-turn token.

## [0.2.1] — 2026-08-09

**Speech-to-text.** All twelve OpenAI Whisper variants, `tiny` to
`large-v3-turbo` including the `.en` builds, run as `.base` bundles on Metal
and CUDA.

### Added

- Whisper conversion with `whisper-q8` and `whisper-q4` profiles.
- `POST /v1/audio/transcriptions` and `/v1/audio/translations`, including
  `verbose_json`, and streaming transcription (`stream: true`).
- C API: `baseRT_load_model_ex` with `BaseRTLoadOptions`, Whisper task and
  prompt setters with per-segment results, batched-logits helpers, and
  `baseRT_capabilities`.

### Changed

- Faster shared-prefix serving and CUDA MoE decode; f32 accumulation on lossy
  Metal GEMM paths.

### Fixed

- `context_length_exceeded` reports the real token count.
- Image requests under-reported usage and never returned
  `finish_reason: "length"`.
- Gemma 4 audio embeddings, repetition penalty on some Apple GPUs, a KV-cache
  shift data race, and unbounded rate-limit buckets.

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
