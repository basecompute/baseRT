<div align="center">

# BaseRT

**Fastest LLM inference on Apple Silicon — now on NVIDIA GB10 (DGX Spark), too.**

Pull a model from HuggingFace, chat with it, or serve an OpenAI-compatible API — all from one CLI.

[![Docs](https://img.shields.io/badge/docs-docs.basecompute.co-5b8fa8)](https://docs.basecompute.co)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%C2%B7%20NVIDIA%20GB10-black)](#requirements)
[![Bindings](https://img.shields.io/badge/bindings-Python%20%C2%B7%20Node%20%C2%B7%20Rust%20%C2%B7%20Swift-444)](docs/bindings/index.md)

</div>

---

BaseRT runs large language models locally on Apple Silicon (Metal) and NVIDIA
GB10 / DGX Spark (CUDA), accelerated by hand-written GPU kernels. This repository
is the open ecosystem around the engine: the `basert` CLI (model hub +
converter), the `.base` model format, the public C API, and language bindings for
Python, Node, Rust, and Swift.

- **One CLI for everything** — `basert pull`, `chat`, `serve`, `convert`.
- **OpenAI-compatible server** — chat, completions, embeddings, transcription,
  tool calls, continuous batching, paged-KV, and prefix caching.
- **Multimodal** — text, vision, and audio on supported models.
- **Its own `.base` format** — affine quantization (Q2–Q8), optional AWQ
  calibration, and signed bundles.
- **Bindings everywhere** — Python, Node, Rust, and Swift over a stable C API.
- **A drop-in coding-agent backend** — see [below](#use-it-as-a-coding-agent).

> **Documentation:** **[docs.basecompute.co](https://docs.basecompute.co)** (source in [`docs/`](docs/index.md)).

## Requirements

One of:

- Apple Silicon (M1 or later), macOS 14+.
- NVIDIA GB10 (DGX Spark), Linux / arm64 with a CUDA driver.

## Quickstart

### 1. Install

```sh
curl -LsSf https://basecompute.co/install.sh | sh
```

This installs the prebuilt engine for your platform and the `basert` CLI to
`~/.basert` and adds it to your `PATH`. Restart your shell afterward (or
`export PATH="$HOME/.basert:$PATH"`). See
[Installation](docs/getting-started/installation.md) for building from source.

### 2. Pull a model

```sh
basert pull Qwen/Qwen3-4B    # download from HuggingFace + convert to .base
basert list                  # show installed models
```

Models are cached under `~/.cache/baseRT/models` (override with
`$BASERT_MODELS_DIR`). To pick a specific quant build, append `:default-q8` to
the id (or pass `--variant default-q8`).

### 3. Chat

```sh
basert chat Qwen/Qwen3-4B
```

### 4. Serve an OpenAI-compatible API

```sh
basert serve Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

### 5. Or call it from your language

```python
import baseRT

model = baseRT.Model("models/your-model.base")
print(model.generate_text("The capital of France is", max_tokens=64))
```

Node, Rust, and Swift bindings work the same way — see
[Bindings](docs/bindings/index.md).

## Use it as a coding agent

BaseRT runs as a local backend for the [pi](https://pi.dev) coding agent through
the **[pi-basert](https://github.com/basecompute/pi-basert)** extension, which
auto-discovers whatever models your server is running.

```sh
# 1. Serve a model (a dense model is best for tool calling)
basert serve basecompute/gemma-4-E2B-it

# 2. Install the extension and launch pi
pi install git:github.com/basecompute/pi-basert
pi
```

Inside pi, run `/model` to pick a served model. See the
[pi-basert README](https://github.com/basecompute/pi-basert) for details.

### Claude Code via cc-bridge

[cc-bridge](cc-bridge/) is a native Anthropic Messages API adapter that lets
[Claude Code](https://claude.com/claude-code) talk to a local `basert serve`
directly (`/v1/messages` ↔ `/v1/chat/completions`), with SSE streaming, thinking
and tool blocks, keep-alive pings, and cancellation propagation. Inbound
requests require a master key; the adapter binds to 127.0.0.1 by default.

```sh
# 1. Serve a model
basert serve basecompute/Qwen3.6-35B-A3B

# 2. Set up and run the adapter (see cc-bridge/README.md for all options)
cd cc-bridge
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python anthropic_adapter.py --master-key change-me \
  --model basecompute/Qwen3.6-35B-A3B

# 3. Point Claude Code at it
export ANTHROPIC_BASE_URL="http://localhost:4003"
export ANTHROPIC_AUTH_TOKEN="change-me"
export ANTHROPIC_MODEL="local"
claude
```

Full client setup and troubleshooting:
[docs/guides/connect-claude-code.md](docs/guides/connect-claude-code.md).

## The `basert` CLI

`basert` is a single front-end. Model-management commands run natively; runtime
commands are forwarded to the matching `basert-<cmd>` engine binary.

| Command | What it does |
| --- | --- |
| `basert pull <id>` | Download from HuggingFace (convert-on-pull) or the catalog |
| `basert list` | List installed models (`--remote` to include the catalog) |
| `basert convert <src>` | Convert a local GGUF / HF / MLX checkpoint to `.base` |
| `basert chat <model>` | Interactive chat |
| `basert serve <model>` | OpenAI-compatible HTTP server (`--model` repeats for multi-model) |
| `basert complete <model>` | One-shot completion (text / image / audio) |
| `basert bench <model>` | Throughput benchmark |
| `basert inspect <model>` | Dump a `.base` header + tensor inventory |
| `basert sign` / `verify` / `keygen` | ed25519 signing of `.base` bundles |

Full flag reference: [CLI reference](docs/cli/reference.md).

## What's in this repo

| Path | Contents |
| --- | --- |
| [`base-convert/`](base-convert/) | The `basert` CLI (Rust): model hub, converter, launcher — plus the [`.base` format](base-convert/FORMAT.md) and [quantization](base-convert/CANONICAL_QUANT_SPEC.md) specs and the generic quant profiles. |
| [`include/baseRT/`](include/baseRT/) | The stable public C API (`baseRT.h`, `types.h`). |
| [`bindings/`](bindings/) | Python, Node, Rust, and Swift. |
| [`benchmarks/`](benchmarks/) | Scripts and reference results. |
| [`docs/`](docs/index.md) | The documentation site. |

This repository — the CLI, `.base` format, public headers, and bindings — is
licensed under **Apache-2.0**. The engine ships as a separate prebuilt binary
under its own license; see
[Engine releases](docs/reference/engine-releases.md).

## Links

- **[Documentation](https://docs.basecompute.co)** · [Server API](docs/reference/server-api.md) · [`.base` format](base-convert/FORMAT.md)
- [Security policy](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The prebuilt engine
binary distributed via [Releases](https://github.com/basecompute/baseRT/releases)
is proprietary and ships under its own license.
