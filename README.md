# base

The open **`.base` model format**, the **`base-convert`** tool, and the official
**language bindings** for the baseRT inference engine — an LLM runtime for Apple
Silicon (Metal).

This repository is the open ecosystem around baseRT:

- **`base-convert/`** — converts GGUF / HuggingFace / MLX checkpoints to the
  `.base` format, with affine quantization (Q2–Q8) and optional AWQ calibration.
- **`include/baseRT/`** — the stable public C API (`baseRT.h`, `types.h`).
- **`bindings/`** — Python, Node, Rust, and Swift bindings over that API.
- **`benchmarks/`** — scripts to reproduce throughput numbers.

The **engine itself ships as a prebuilt binary** (`libbaseRT.dylib` + CLI tools)
via [GitHub Releases](../../releases) — see [Getting the engine](#getting-the-engine).
The compiled Metal kernels are embedded in the dylib, so it is a single
self-contained file.

## What this is / isn't

- **Is:** the format, converter, headers, and bindings you build on top of the
  baseRT engine. Apache-2.0.
- **Isn't:** the engine source. The runtime is distributed as a binary.

## Requirements

- Apple Silicon (M1 or later), macOS 14+.
- Rust 1.80+ (to build `base-convert`).
- A binding toolchain as needed (Python 3.9+, Node 18+, Swift 5.9+).

## Getting the engine

Download the latest engine release and unpack it where your binding can find it
(default: a `build/` directory at the repo root, or set `BASERT_LIB_PATH`):

```sh
# example: fetch the latest release's macOS arm64 bundle
gh release download --repo prabod/base --pattern 'baseRT-engine-macos-arm64*.tar.gz'
mkdir -p build && tar -xzf baseRT-engine-macos-arm64*.tar.gz -C build
# build/ now has libbaseRT.dylib (kernels embedded) + CLI tools + headers
```

## Convert a model

```sh
cd base-convert
cargo build --release
./target/release/base-convert convert \
    --source <path-to-gguf-or-hf-dir> \
    --profile profiles/default-q4.json \
    --output ../models/your-model.base
```

See [`base-convert/FORMAT.md`](base-convert/FORMAT.md) for the `.base` container
spec and [`base-convert/CANONICAL_QUANT_SPEC.md`](base-convert/CANONICAL_QUANT_SPEC.md)
for the quantization schemes. Write your own precision policy with a profile —
see [`base-convert/profiles/PROFILES.md`](base-convert/profiles/PROFILES.md).

## Run a model (Python)

```python
import baseRT
m = baseRT.Model("models/your-model.base")          # kernels embedded; no metallib needed
print(m.generate_text("The capital of France is", max_tokens=64))
```

Equivalent bindings exist for Node, Rust, and Swift — see [`bindings/`](bindings/).

## Run the OpenAI-compatible server

The engine bundle includes `baseRT_serve`, an OpenAI-compatible HTTP server.
Run it from the repo root (the engine release unpacks into `build/`, where the
server auto-detects its `baseRT.metallib`):

```sh
./build/baseRT_serve --model models/your-model.base --api-key "$(uuidgen)" --port 8080
```

Then call it like any OpenAI endpoint:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"your-model.base","messages":[{"role":"user","content":"Hello!"}]}'
```

`baseRT_serve --help` lists the flags (continuous batching, paged-KV, prefix
cache, rate limiting, etc.). See `SECURITY.md` before exposing it beyond
localhost. The other bundled CLI tools — `baseRT_chat`, `baseRT_complete`,
`baseRT_bench`, `baseRT_inspect`, `baseRT_transcribe` — work the same way.

## Benchmarks

`benchmarks/scripts/` reproduces throughput numbers against the engine binary;
example results are in `benchmarks/results/`. See [`benchmarks/README.md`](benchmarks/README.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Issues and focused PRs welcome —
new model mappings for `base-convert` and binding improvements especially.

## Security

See [`SECURITY.md`](SECURITY.md). Report vulnerabilities privately.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
