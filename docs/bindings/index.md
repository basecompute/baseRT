# Bindings

BaseRT exposes a C API (`include/baseRT/baseRT.h`) and ships idiomatic
wrappers for four languages. All of them drive the same engine library:
`libbaseRT.dylib` on macOS (Metal kernels embedded, no separate `.metallib`
to manage) and `libbaseRT.so` on Linux.

| Language | Install | Import | Source |
| --- | --- | --- | --- |
| [Python](python.md) | `pip install basert` | `import baseRT` | [`bindings/python`](https://github.com/basecompute/baseRT/tree/main/bindings/python) |
| [Node](node.md) | `npm install @basecompute/basert` | `import { BaseRTModel } from "@basecompute/basert"` | [`bindings/node`](https://github.com/basecompute/baseRT/tree/main/bindings/node) |
| [Rust](rust.md) | `cargo add baseRT` | `use baseRT::…` | [`bindings/rust`](https://github.com/basecompute/baseRT/tree/main/bindings/rust) |
| [Swift](swift.md) | local Swift package | `import BaseRT` | [`bindings/swift`](https://github.com/basecompute/baseRT/tree/main/bindings/swift) |

The PyPI and npm packages include the prebuilt engine for your platform. The
Rust crates and the Swift package link an engine you have installed.

## Finding the engine

Each binding needs to locate the engine library. Without an override, the
bindings look in this order:

1. The engine inside the package (`pip install basert`, `npm install
   @basecompute/basert`) — Python and Node.
2. A source checkout's `build/` directory.
3. An install-script engine: `$BASERT_INSTALL_DIR`, default `~/.basert`.
4. Homebrew: `/opt/homebrew/opt/basert/libexec` on macOS,
   `/home/linuxbrew/.linuxbrew/opt/basert/libexec` on Linux.

To point a binding at a specific engine instead, set:

| Variable | Used by | Value |
| --- | --- | --- |
| `BASERT_LIB_PATH` | Python, Node | The library file itself |
| `BASERT_LIB_DIR` | Rust, Swift | The directory containing it |

The Swift package uses `BASERT_LIB_DIR`, or the checkout's `build/`, only.

An [engine release](../reference/engine-releases.md) unpacked into `build/`
at the repo root works for all four.

## Shared model

All bindings load the same `.base` files you produce with `basert pull` /
`basert convert`, and expose the same primitives: encode/decode, prefill,
single or streaming generation, multi-turn continuation from the KV cache,
embeddings, chat-template formatting, and (for Whisper models) transcription.
Python, Node and Swift add image and audio prefill for multimodal models. Node
and Swift also expose the process-wide engine settings, such as the KV-cache
width and the Neural Engine prefill split; set those before loading a model.

A model handle is single-owner: use one model from one thread at a time, and
don't call a model from inside its own token callback.
