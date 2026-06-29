# BaseRT

**A fast LLM inference runtime for Apple Silicon (Metal).** Pull a model, chat
with it, or serve an OpenAI-compatible API — all from one CLI, `basert`.

The engine ships as a single self-contained `libbaseRT.dylib` (Metal kernels
embedded) plus the `basert-*` runtime tools. This repository is the open
ecosystem around it: the `basert` CLI (model hub + converter), the `.base`
model format, the public C API, and language bindings for Python, Node, Rust,
and Swift.

> **Full documentation:** [docs/](docs/), also published at
> [basecompute.co/docs](https://basecompute.co/docs).

---

## Quickstart

### 1. Install

```sh
curl -LsSf https://basecompute.co/install.sh | sh
```

This downloads the prebuilt engine + the `basert` CLI to `~/.basert` and adds it
to your `PATH`. Requires Apple Silicon (M1+) and macOS 14+. Restart your shell
afterward (or `export PATH="$HOME/.basert:$PATH"`).

<details>
<summary>Build from source instead</summary>

```sh
# Prebuilt engine (libbaseRT.dylib + basert-* tools), into build/
gh release download --repo basecompute/baseRT --pattern 'basert-engine-macos-arm64*.tar.gz'
mkdir -p build && tar -xzf basert-engine-macos-arm64*.tar.gz -C build

# The basert CLI (model hub + converter + launcher) — needs Rust 1.80+
cd base-convert && cargo build --release && cd ..

export PATH="$PWD/build:$PWD/base-convert/target/release:$PATH"
```
</details>

See [docs/getting-started/installation.md](docs/getting-started/installation.md)
for details.

### 2. Pull a model

```sh
basert pull Qwen/Qwen3-4B           # download from HuggingFace + convert to .base
basert list                          # show installed models
```

Models are cached under `~/.cache/baseRT/models` (`$BASERT_MODELS_DIR`).

### 3. Chat

```sh
basert chat Qwen/Qwen3-4B
```

### 4. Serve an OpenAI-compatible API

```sh
basert serve --model Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

### 5. Or call it from your language

```python
import baseRT
m = baseRT.Model("models/your-model.base")   # kernels embedded; no metallib needed
print(m.generate_text("The capital of France is", max_tokens=64))
```

Node, Rust, and Swift bindings work the same way — see
[docs/bindings/](docs/bindings/index.md).

---

## The `basert` CLI

`basert` is a single front-end. Model-management commands run natively; runtime
commands are forwarded to the matching `basert-<cmd>` engine binary:

| Command | What it does |
| --- | --- |
| `basert pull <id>` | Download from HuggingFace (convert-on-pull) or the catalog |
| `basert list` | List installed models (`--remote` to include the catalog) |
| `basert convert <src>` | Convert a local GGUF / HF / MLX checkpoint to `.base` |
| `basert chat <model>` | Interactive chat |
| `basert serve --model <m>` | OpenAI-compatible HTTP server |
| `basert complete <model>` | One-shot completion (text / image / audio) |
| `basert bench <model>` | Throughput benchmark |
| `basert inspect <model>` | Dump a `.base` header + tensor inventory |
| `basert sign` / `verify` / `keygen` | ed25519 signing of `.base` bundles |

Full flag reference: [docs/cli/reference.md](docs/cli/reference.md).

## What's in this repo

- **`base-convert/`** — the `basert` CLI (Rust): model hub (`base-hub`),
  converter, and launcher. Includes the `.base` format spec
  ([FORMAT.md](base-convert/FORMAT.md)), the quantization spec
  ([CANONICAL_QUANT_SPEC.md](base-convert/CANONICAL_QUANT_SPEC.md)), and the
  generic quant profiles.
- **`include/baseRT/`** — the stable public C API (`baseRT.h`, `types.h`).
- **`bindings/`** — Python, Node, Rust, Swift.
- **`benchmarks/`** — scripts + reference results.
- **`docs/`** — the documentation site.

Everything in this repository is open source (Apache-2.0). The engine ships as
a separate prebuilt binary under a proprietary license — see
[docs/reference/engine-releases.md](docs/reference/engine-releases.md).

## Links

- [Documentation](docs/index.md)
- [Server API](docs/reference/server-api.md)
- [`.base` format](base-convert/FORMAT.md)
- [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## License

This repository (the CLI, `.base` format, public headers, and bindings) is
licensed under Apache-2.0 — see `LICENSE` and `NOTICE`. The prebuilt engine
binary distributed via [Releases](https://github.com/basecompute/baseRT/releases)
is proprietary and ships under its own license.
