# Engine releases

The BaseRT engine is distributed as a prebuilt binary bundle under
[Releases](https://github.com/prabod/baseRT/releases). This repository — the
CLI, format, headers, bindings, and docs — is open; the engine binary is the
product. You never need to build the engine from source.

## What a release contains

`basert-engine-macos-arm64-<version>.tar.gz`:

- **`libbaseRT.dylib`** — the engine shared library, with the compiled Metal
  kernels **embedded** (a single self-contained file; no sidecar `.metallib`).
- **`baseRT.metallib`** — the standalone kernel library (for tools that link the
  static engine).
- **`basert-*` CLI tools** — `basert-serve`, `basert-chat`, `basert-complete`,
  `basert-bench`, `basert-inspect`, `basert-transcribe`.
- **`include/baseRT/`** — the public headers matching this release.

## Installing a release

```sh
gh release download --repo prabod/baseRT --pattern 'basert-engine-macos-arm64*.tar.gz'
mkdir -p build && tar -xzf basert-engine-macos-arm64*.tar.gz -C build
export PATH="$PWD/build:$PWD/base-convert/target/release:$PATH"
```

See [Installation](../getting-started/installation.md) for the full setup,
including building the `basert` CLI and putting everything on your `PATH`.

## Versioning

Releases are versioned (`<version>` in the archive name) and the bundled
`include/baseRT/` headers match the `libbaseRT.dylib` in the same release —
always use headers and dylib from the **same** release to keep the C ABI
consistent. The bindings in this repo target the latest engine release.
