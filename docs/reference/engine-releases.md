# Engine releases

The BaseRT engine is distributed as prebuilt binary bundles under
[Releases](https://github.com/basecompute/baseRT/releases). This repository — the
CLI, format, headers, bindings, and docs — is open source (Apache-2.0). The
engine binary itself is **proprietary** (see the `LICENSE` included in each
bundle); it is provided ready to use, so you never need to build it.

Most people never handle a bundle directly: the install script, `pip install
basert`, `npm install -g @basecompute/basert` and `brew install
basecompute/tap/basert` each fetch the right one for the machine. See
[Installation](../getting-started/installation.md).

## What a release contains

Each release is tagged `v<version>` (for example `v0.5.1`) and carries one
bundle per platform, each with a `.sha256` file beside it:

| Asset | Platform |
| --- | --- |
| `basert-engine-macos-arm64-<version>.tar.gz` | Apple Silicon, macOS 15 or later (Metal) |
| `basert-engine-linux-arm64-cuda-<version>.tar.gz` | Linux arm64 with an NVIDIA GPU (GB10 / DGX Spark) and the CUDA 13 runtime (cuBLAS) |
| `basert-engine-linux-x64-rocm-<version>.tar.gz` | Linux x86-64 with an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU and the ROCm 10 runtime in `/opt/rocm/core-10.0` (from 0.5.0) |
| `basert-engine-abi-<N>.txt` | The engine's C ABI number, in the name and in the file (from 0.4.0) |

A bundle extracts flat — every file side by side, no top-level folder:

- **The engine library** with its versioned names: `libbaseRT.dylib` →
  `libbaseRT.0.dylib` → `libbaseRT.<version>.dylib` on macOS, and
  `libbaseRT.so` → `libbaseRT.so.0` → `libbaseRT.so.<version>` on Linux. On
  macOS the Metal kernels are embedded in the library, so a program linking it
  needs no other file.
- **Kernel files** the runtime tools load from beside themselves:
  `baseRT.metallib` and `baseRT_tensor.metallib` on macOS,
  `baseRT_cuda_kernels.fatbin` on CUDA, `baseRT_rocm_kernels.hsaco` on ROCm.
- **`basert`**, the CLI: model hub, converter, and launcher for the tools
  below.
- **Runtime tools** that `basert` forwards to: `basert-chat`,
  `basert-complete`, `basert-bench` and `basert-benchmark-harness` on every
  platform, `basert-inspect` on macOS and ROCm, and `basert-transcribe` on
  macOS.
- **`include/baseRT/`** — `baseRT.h` and `types.h`, the C API headers matching
  this library.
- **`LICENSE`** (the engine license) and **`NOTICE`**.

superfluid, the server `basert serve` runs, is not in the bundle. `basert`
installs a superfluid release that serves this engine on the first `basert
serve`, or when you run `basert superfluid install`.

## Installing a release

The install script picks the bundle for your platform, places it under
`~/.basert/versions/<version>/`, and puts `~/.basert` on your `PATH`:

```sh
curl -LsSf https://basecompute.co/install.sh | sh
```

`BASERT_VERSION=v0.5.0` installs a specific release instead of the latest.
Once installed, basert updates itself (from 0.5.5, see
[Updating](../getting-started/installation.md#updating)), `basert self update`
moves to the newest release now, and `basert self use <version>` goes back to
a kept one.

To unpack a bundle by hand, for example into a checkout's `build/` directory,
where the bindings look by default:

```sh
gh release download v0.5.1 --repo basecompute/baseRT \
  --pattern 'basert-engine-macos-arm64-0.5.1.tar.gz*'
shasum -a 256 -c basert-engine-macos-arm64-0.5.1.tar.gz.sha256
mkdir -p build && tar -xzf basert-engine-macos-arm64-0.5.1.tar.gz -C build
export PATH="$PWD/build:$PATH"
```

On Linux, use `sha256sum -c` in place of `shasum -a 256 -c`.

## Verifying a release

Each `.sha256` file holds the SHA-256 of its tarball in the format `shasum`
and `sha256sum` check (`<hash>  <file name>`). `basert self update` and an
automatic update check the same hash before they switch to a release, and
then check that the new launcher runs and reports the expected version.

## Versioning

Releases follow `MAJOR.MINOR.PATCH`, and the version is in every asset name.
The `include/baseRT/` headers in a bundle match the library in the same
bundle: `BASERT_VERSION_MAJOR`, `_MINOR` and `_PATCH` give the version at
compile time, and `baseRT_version_string()` gives the loaded library's at run
time.

Compatibility between a library and the code that drives it is tracked by a
separate number, the **engine ABI**:

- `BASERT_ABI_VERSION` in `baseRT.h` is the number the headers describe, and
  `baseRT_abi_version()` returns the loaded library's.
- It changes only when a function signature, struct layout or enum value
  changes incompatibly. A patch release never changes it. A release may add
  functions, or append fields to `BaseRTModelConfig` (read it with
  `baseRT_get_config_into()`, which takes a size), and keep the number.
- The current ABI is **1**, unchanged since 0.4.0.
- The `basert-engine-abi-<N>.txt` asset lets a tool pick a compatible engine
  from the release listing before downloading anything; superfluid does this.

A program that loads `libbaseRT` at run time should compare
`baseRT_abi_version()` with the `BASERT_ABI_VERSION` it was built against,
rather than compare version strings. The bindings in this repository target
the latest engine release.
