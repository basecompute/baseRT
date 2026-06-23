# Contributing to base

Thanks for your interest. This repo holds the open `.base` format, the
`base-convert` tool, and the language bindings. The engine itself is distributed
as a prebuilt binary and is not part of this repository.

## Good first contributions

- **New model mappings** for `base-convert` (a new architecture's tensor-name
  normalization / canonical-inference fixes in `base-convert/crates/base-arch`).
- **Binding improvements** — coverage, ergonomics, packaging.
- **Quantization profiles** — see `base-convert/profiles/PROFILES.md`.
- **Benchmark scripts / results** for additional hardware.

## Development

- `base-convert`: `cd base-convert && cargo build && cargo test`.
- Bindings: each has its own test step (e.g. `bindings/node` → `npm test`,
  `bindings/rust/baseRT-sys` → `cargo test`, `bindings/swift` → `swift test`).
  Tests that exercise the engine need the binary (see the README's
  "Getting the engine") and a `.base` model; pure unit tests do not.

## Conventions

- Conventional Commits (`feat(convert): ...`, `fix(bindings): ...`).
- Keep PRs focused; open an issue first for larger changes.
- Rust: `cargo fmt` / `cargo clippy`. C headers in `include/baseRT/` are the
  public API contract — changes there must stay ABI-compatible within a major
  version.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
