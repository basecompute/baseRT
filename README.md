# BaseRT

The open **`.base` model format**, the **`basert`** CLI (model hub + converter),
and the official **language bindings** for the **BaseRT** inference engine — an
LLM runtime for Apple Silicon (Metal).

This repository is the open ecosystem around BaseRT:

- **`base-convert/`** — the `basert` CLI: a model hub (`pull`/`list` from
  HuggingFace), an offline converter (GGUF / HuggingFace / MLX → `.base`, with
  affine quantization Q2–Q8 and optional AWQ calibration), and a launcher that
  forwards `basert serve`/`chat`/… to the engine runtime tools.
- **`include/baseRT/`** — the stable public C API (`baseRT.h`, `types.h`).
- **`bindings/`** — Python, Node, Rust, and Swift bindings over that API.
- **`benchmarks/`** — scripts to reproduce throughput numbers.

The **engine itself ships as a prebuilt binary** (`libbaseRT.dylib` + `basert-*`
CLI tools) via [GitHub Releases](../../releases) — see
[Getting the engine](#getting-the-engine). The compiled Metal kernels are
embedded in the dylib, so it is a single self-contained file.

## What this is / isn't

- **Is:** the format, CLI, headers, and bindings you build on top of the
  BaseRT engine. Apache-2.0.
- **Isn't:** the engine source. The runtime is distributed as a binary.

## Requirements

- Apple Silicon (M1 or later), macOS 14+.
- Rust 1.80+ (to build the `basert` CLI).
- A binding toolchain as needed (Python 3.9+, Node 18+, Swift 5.9+).

## Getting the engine

Download the latest engine release and unpack it where the CLI/bindings can find
it (default: a `build/` directory at the repo root, or set `BASERT_LIB_PATH`):

```sh
# fetch the latest release's macOS arm64 bundle
gh release download --repo prabod/baseRT --pattern 'basert-engine-macos-arm64*.tar.gz'
mkdir -p build && tar -xzf basert-engine-macos-arm64*.tar.gz -C build
# build/ now has libbaseRT.dylib (kernels embedded) + basert-* CLI tools + headers
```

## The `basert` CLI

Build it once and put both it and the engine tools on your `PATH`:

```sh
cd base-convert
cargo build --release          # produces target/release/basert
export PATH="$PWD/target/release:$PWD/../build:$PATH"
```

`basert` is a unified front-end. Model-management commands run natively; runtime
commands (`serve`, `chat`, `complete`, `bench`, …) are forwarded to the matching
`basert-<cmd>` engine binary from the release bundle.

### Pull or convert a model

```sh
# Pull straight from HuggingFace (downloads source + converts to .base):
basert pull Qwen/Qwen3-4B

# …or a pre-converted model from the catalog (downloaded directly, no convert):
basert pull basecompute/<name>

# …or convert a local GGUF / HF checkpoint:
basert convert <path-to-gguf-or-hf-dir> \
    --profile profiles/default-q4.json --output models/your-model.base

basert list                    # show installed models
```

Models live in a per-user cache (`$BASERT_MODELS_DIR`, default
`~/.cache/baseRT/models`). See [`base-convert/FORMAT.md`](base-convert/FORMAT.md)
for the `.base` container spec,
[`base-convert/CANONICAL_QUANT_SPEC.md`](base-convert/CANONICAL_QUANT_SPEC.md)
for the quantization schemes, and
[`base-convert/profiles/PROFILES.md`](base-convert/profiles/PROFILES.md) to write
your own precision policy.

## Run a model (Python)

```python
import baseRT
m = baseRT.Model("models/your-model.base")          # kernels embedded; no metallib needed
print(m.generate_text("The capital of France is", max_tokens=64))
```

Equivalent bindings exist for Node, Rust, and Swift — see [`bindings/`](bindings/).

## Run the OpenAI-compatible server

The engine bundle includes `basert-serve`, an OpenAI-compatible HTTP server.
`basert serve` forwards to it (or run `./build/basert-serve` directly — it
auto-detects its `baseRT.metallib` next to the executable):

```sh
basert serve --model models/your-model.base --api-key "$(uuidgen)" --port 8080
```

Then call it like any OpenAI endpoint:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"your-model.base","messages":[{"role":"user","content":"Hello!"}]}'
```

`basert serve --help` lists the flags (continuous batching, paged-KV, prefix
cache, rate limiting, etc.). See `SECURITY.md` before exposing it beyond
localhost. The other runtime tools — `basert chat`, `basert complete`,
`basert bench`, `basert inspect`, `basert transcribe` — work the same way.

## Benchmarks

`benchmarks/scripts/` reproduces throughput numbers against the engine binary;
example results are in `benchmarks/results/`. See [`benchmarks/README.md`](benchmarks/README.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues and focused PRs welcome —
new model mappings for the converter and binding improvements especially.

## Security

See [`SECURITY.md`](SECURITY.md). Report vulnerabilities privately.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
