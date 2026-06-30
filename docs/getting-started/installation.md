# Installation

## One-shot install (recommended)

```sh
curl -LsSf https://basecompute.co/install.sh | sh
```

This downloads the prebuilt engine bundle (the `libbaseRT.dylib`, the `basert`
CLI, and the `basert-*` runtime tools) into `~/.basert` and adds it to your
`PATH`. Restart your shell afterward, then:

```sh
basert pull Qwen/Qwen3-0.6B
basert chat Qwen/Qwen3-0.6B
```

Overrides (set as environment variables before running):

| Variable | Purpose | Default |
| --- | --- | --- |
| `BASERT_INSTALL_DIR` | Install location | `~/.basert` |
| `BASERT_VERSION` | Specific release tag (e.g. `v0.5.0`) | latest |
| `BASERT_NO_MODIFY_PATH` | Set to `1` to skip editing shell profiles | unset |

Requires Apple Silicon (M1+) and macOS 14+. The installer is macOS/arm64 only.

## Manual install

BaseRT has two parts you put on your `PATH`:

1. **The engine** — `libbaseRT.dylib` plus the `basert-*` runtime tools,
   distributed as a prebuilt release.
2. **The `basert` CLI** — built from this repository (`base-convert`).

### Requirements

- Apple Silicon (M1 or later), macOS 14+.
- Rust 1.80+ (to build the CLI).
- For bindings: Python 3.9+, Node 18+, or Swift 5.9+ as needed.
- [`gh`](https://cli.github.com) (optional, to download releases).

## 1. Get the engine

Each release ships a macOS arm64 bundle named
`basert-engine-macos-arm64-<version>.tar.gz` containing `libbaseRT.dylib` (Metal
kernels embedded), the `basert-*` tools, `baseRT.metallib`, and the public
headers.

```sh
gh release download --repo basecompute/baseRT --pattern 'basert-engine-macos-arm64*.tar.gz'
mkdir -p build && tar -xzf basert-engine-macos-arm64*.tar.gz -C build
# build/ now has libbaseRT.dylib + basert-serve, basert-chat, … + baseRT.metallib + include/
```

See [Engine releases](../reference/engine-releases.md) for what's in a bundle
and how it's produced.

## 2. Build the `basert` CLI

```sh
cd base-convert
cargo build --release        # produces target/release/basert
cd ..
```

## 3. Put both on your `PATH`

The launcher resolves `basert <cmd>` to a `basert-<cmd>` binary sitting next to
it or anywhere on `PATH`, so co-locating them (as a release tarball does) makes
everything Just Work:

```sh
export PATH="$PWD/build:$PWD/base-convert/target/release:$PATH"
```

Verify:

```sh
basert --help
basert serve --help
```

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `BASERT_MODELS_DIR` | Where pulled/converted models are cached | `~/.cache/baseRT/models` |
| `BASERT_LIB_PATH` / `BASERT_LIB_DIR` | Where bindings look for `libbaseRT.dylib` | `build/` at repo root |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | HuggingFace auth for gated/private repos | unset |

## Next steps

- [Quickstart](quickstart.md) — pull a model and chat.
- [Managing models](../guides/models.md) — the model hub and cache layout.
- [Bindings](../bindings/index.md) — call BaseRT from your language.
