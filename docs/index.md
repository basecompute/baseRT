# BaseRT

**A fast LLM inference runtime for Apple Silicon (Metal).**

BaseRT runs large language models locally on Apple Silicon. Pull a model from
HuggingFace, chat with it, or serve an OpenAI-compatible API — all through one
CLI, `basert`.

## Why BaseRT

- **Self-contained engine.** No GPU drivers, Python runtime, or extra
  components to install — drop in the binaries and run.
- **One CLI for everything.** `basert pull`, `basert chat`, `basert serve`,
  `basert convert` — model management and runtime in one front-end.
- **OpenAI-compatible server.** Chat, completions, embeddings, transcription,
  rerank, tool calls, continuous batching, paged-KV, prefix caching.
- **Its own `.base` format.** Affine quantization (Q2–Q8), optional AWQ
  calibration, signed bundles.
- **Bindings everywhere.** Python, Node, Rust, Swift over a stable C API.

## Get started

- **[Installation](getting-started/installation.md)** — get
  the engine + the `basert` CLI on your `PATH`.
- **[Quickstart](getting-started/quickstart.md)** —
  pull a model and chat in under a minute.
- **[CLI reference](cli/reference.md)** — every command and
  flag.
- **[Server API](reference/server-api.md)** — the
  OpenAI-compatible endpoints.

## At a glance

```sh
# install (see Installation for details)
export PATH="$PWD/build:$PWD/base-convert/target/release:$PATH"

basert pull Qwen/Qwen3-4B            # download + convert
basert chat Qwen/Qwen3-4B            # interactive chat
basert serve --model Qwen/Qwen3-4B --api-key "$(uuidgen)"   # OpenAI server
```

## How the pieces fit

| Piece | Role |
| --- | --- |
| `libbaseRT.dylib` | The engine (Metal kernels embedded). Prebuilt binary. |
| `basert-serve`, `basert-chat`, … | Runtime tools that link the engine. |
| `basert` | The CLI: model hub + converter + launcher for the tools. |
| `.base` files | The on-disk model format the runtime loads. |
| Bindings | Python / Node / Rust / Swift over the C API (`baseRT.h`). |

> [!NOTE]
> **Open ecosystem**
>
> The engine ships as a prebuilt binary; this repository — the CLI, format,
> headers, bindings, and docs — is open source (Apache-2.0). The engine is
> consumed as a [prebuilt release](reference/engine-releases.md), so you never
> need to build it yourself.

## Requirements

- Apple Silicon (M1 or later), macOS 14+.
- Rust 1.80+ to build the `basert` CLI.
- A binding toolchain as needed (Python 3.9+, Node 18+, Swift 5.9+).
