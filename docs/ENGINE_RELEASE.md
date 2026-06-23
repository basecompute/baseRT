# Engine releases

The baseRT engine is built from a separate (private) source repository and
published here as a binary bundle under [Releases](../../releases). This repo —
the format, converter, and bindings — is open; the engine binary is the product.

## What a release contains

`baseRT-engine-macos-arm64-<version>.tar.gz`:

- `libbaseRT.dylib` — the engine shared library, with the compiled Metal
  kernels **embedded** (single self-contained file; no sidecar `.metallib`).
- CLI tools: `baseRT_serve`, `baseRT_chat`, `baseRT_complete`, `baseRT_bench`,
  `baseRT_inspect`, `baseRT_transcribe`.
- `include/baseRT/` — the public headers matching this release.

## How releases are produced

A workflow in the engine repo (`release-engine.yml`) builds the engine on an
Apple-Silicon runner and uploads the bundle to a Release in **this** repo via a
cross-repo token. The trigger is a version tag / GitHub release in the engine
repo. The bindings here are tested against the latest engine release in CI (see
`.github/workflows/ci.yml`).

Consumers never need the engine source — download the release, unpack into
`build/`, and use any binding.
