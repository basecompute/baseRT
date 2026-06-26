# BaseRT

**A fast LLM inference runtime for Apple Silicon (Metal).**

BaseRT runs large language models locally on Apple Silicon. Pull a model from
HuggingFace, chat with it, or serve an OpenAI-compatible API — all through one
CLI, `basert`. The engine is a single self-contained `libbaseRT.dylib` with the
Metal kernels embedded; there is no separate `.metallib` to ship.

## Why BaseRT

- **Single-binary engine.** `libbaseRT.dylib` carries the compiled Metal
  kernels. Drop it next to the `basert-*` tools and run.
- **One CLI for everything.** `basert pull`, `basert chat`, `basert serve`,
  `basert convert` — model management and runtime in one front-end.
- **OpenAI-compatible server.** Chat, completions, embeddings, transcription,
  rerank, tool calls, continuous batching, paged-KV, prefix caching.
- **Its own `.base` format.** Affine quantization (Q2–Q8), optional AWQ
  calibration, signed bundles.
- **Bindings everywhere.** Python, Node, Rust, Swift over a stable C API.

## Get started

<div class="grid cards" markdown>

- :material-download: **[Installation](getting-started/installation.md)** — get
  the engine + the `basert` CLI on your `PATH`.
- :material-rocket-launch: **[Quickstart](getting-started/quickstart.md)** —
  pull a model and chat in under a minute.
- :material-console: **[CLI reference](cli/reference.md)** — every command and
  flag.
- :material-api: **[Server API](reference/server-api.md)** — the
  OpenAI-compatible endpoints.

</div>

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

!!! note "Open core"
    The **engine binary is the product** and its source is private. This
    repository — the CLI, format, headers, bindings, and docs — is open
    (Apache-2.0). The engine is consumed as a
    [prebuilt release](reference/engine-releases.md).

## Requirements

- Apple Silicon (M1 or later), macOS 14+.
- Rust 1.80+ to build the `basert` CLI.
- A binding toolchain as needed (Python 3.9+, Node 18+, Swift 5.9+).
