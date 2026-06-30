# Bindings

BaseRT exposes a stable C API (`include/baseRT/baseRT.h`) and ships idiomatic
wrappers for four languages. All of them link the single `libbaseRT.dylib`
(Metal kernels embedded) — no separate `.metallib` to manage.

| Language | Package | Import | Source |
| --- | --- | --- | --- |
| [Python](python.md) | `baseRT` | `import baseRT` | [`bindings/python`](https://github.com/basecompute/baseRT/tree/main/bindings/python) |
| [Node](node.md) | `@baseRT/node` | `import { BaseRTModel }` | [`bindings/node`](https://github.com/basecompute/baseRT/tree/main/bindings/node) |
| [Rust](rust.md) | `baseRT` | `use baseRT::…` | [`bindings/rust`](https://github.com/basecompute/baseRT/tree/main/bindings/rust) |
| [Swift](swift.md) | `BaseRT` | `import BaseRT` | [`bindings/swift`](https://github.com/basecompute/baseRT/tree/main/bindings/swift) |

## Finding the engine

Each binding needs to locate `libbaseRT.dylib` at runtime. Unpack an
[engine release](../reference/engine-releases.md) into `build/` at the repo root
(the default search path) or set:

| Variable | Used by |
| --- | --- |
| `BASERT_LIB_PATH` | Python, Node |
| `BASERT_LIB_DIR` | Rust, Swift |

The dylib carries the Metal kernels and its own framework/C++ dependencies, so
it's a single redistributable artifact across every binding.

## Shared model

All bindings load the same `.base` files you produce with `basert pull` /
`basert convert`, and expose the same primitives: encode/decode, prefill, single
or streaming generation, embeddings, chat-template formatting, and (for
Whisper-class models) transcription.
