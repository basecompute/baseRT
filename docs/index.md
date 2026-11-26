# BaseRT

**A fast LLM inference runtime for Apple Silicon, NVIDIA GB10 and AMD Strix.**

BaseRT runs large language models locally on Apple Silicon (Metal), NVIDIA
GB10 / DGX Spark (CUDA) and AMD Strix Halo / Strix Point (ROCm). Pull a model
from the BaseRT catalog or Hugging Face, chat with it, or serve it over
OpenAI-, Anthropic- and Ollama-compatible APIs — all through one CLI, `basert`.

## Why BaseRT

- **Self-contained engine.** One prebuilt bundle per platform: install it
  with one command or your package manager and run.
- **One CLI for everything.** `basert pull`, `basert chat`, `basert serve`,
  `basert convert` — model management and runtime in one front-end. basert
  keeps itself and Superfluid up to date (`basert self auto-update off` turns
  that off; `basert self update` updates by hand).
- **A local API server.** `basert serve` runs
  [Superfluid](https://github.com/basecompute/superfluid), our open-source
  server: OpenAI-, Anthropic- and Ollama-compatible endpoints, tool calls,
  continuous batching, prefix caching and speculative decoding.
- **Its own `.base` format.** Affine quantization (Q2–Q8), GGUF k-quants,
  optional AWQ calibration, signed bundles.
- **Bindings everywhere.** Python, Node, Rust, Swift over a stable C API.

## Get started

- **[Installation](getting-started/installation.md)** — install the engine
  and the `basert` CLI.
- **[Quickstart](getting-started/quickstart.md)** —
  pull a model and chat in under a minute.
- **[CLI reference](cli/reference.md)** — every command and
  flag.
- **[Server API](reference/server-api.md)** — the endpoints `basert serve`
  exposes.

## At a glance

```sh
curl -LsSf https://basecompute.co/install.sh | sh   # or: brew / pip / npm, see Installation

basert pull basecompute/Qwen3-4B                     # download from the catalog
basert chat basecompute/Qwen3-4B                     # interactive chat
basert serve basecompute/Qwen3-4B --api-key "$(uuidgen)"   # API server on 127.0.0.1:8453
basert self update                                   # the newest release now (basert also updates itself)
```

## How the pieces fit

| Piece | Role |
| --- | --- |
| `libbaseRT` | The engine (`.dylib` on macOS, `.so` on Linux). Prebuilt binary. |
| `basert-chat`, `basert-complete`, `basert-bench`, … | Runtime tools that link the engine. |
| `basert` | The CLI: model hub, converter, updater, and launcher for the tools. |
| Superfluid | The server `basert serve` runs, installed on first use. |
| `.base` files | The on-disk model format the runtime loads. |
| Bindings | Python / Node / Rust / Swift over the C API (`baseRT.h`). |

> [!NOTE]
> **Open ecosystem**
>
> The engine ships as a prebuilt binary; this repository — the CLI, format,
> headers, bindings, and docs — is open source (Apache-2.0), and so is
> [Superfluid](https://github.com/basecompute/superfluid). The engine is
> consumed as a [prebuilt release](reference/engine-releases.md), so you never
> need to build it yourself.

## Requirements

One of:

- Apple Silicon (M1 or later), macOS 15+.
- NVIDIA GB10 (DGX Spark), Linux arm64 with the CUDA 13 runtime.
- AMD Strix Halo (Ryzen AI Max) or Strix Point / Gorgon Point (Ryzen AI 300
  or 400 with a Radeon 890M or 880M), Linux x86-64 with the ROCm 10 runtime.

See [Installation](getting-started/installation.md#requirements) for the OS,
kernel and driver details. To build the `basert` CLI yourself you need Rust
1.85+; the bindings need Python 3.9+, Node 18+ or Swift 5.9+.
