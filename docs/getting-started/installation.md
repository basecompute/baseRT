# Installation

BaseRT ships as a prebuilt bundle per platform: the engine library, the
`basert` CLI and the `basert-*` runtime tools. The server `basert serve` runs,
[Superfluid](https://github.com/basecompute/superfluid), is installed
separately and automatically (see [Superfluid](#superfluid-the-server)).

## Requirements

One of:

- **Apple Silicon (Metal).** An M1 or later Mac on macOS 15 or later.
- **NVIDIA (CUDA).** NVIDIA GB10 (DGX Spark): Linux on arm64 with the NVIDIA
  driver and the CUDA 13 runtime (cuBLAS).
- **AMD (ROCm).** Linux on x86-64 with one of these integrated GPUs:
  - Ryzen AI Max (Strix Halo, `gfx1151`): Radeon 8060S or 8050S.
  - Ryzen AI 300 or 400 (Strix Point and Gorgon Point, `gfx1150`): Radeon
    890M or 880M. The Radeon 860M and 840M are not supported.

  It needs Ubuntu 24.04 or newer, or another distribution with glibc 2.39 or
  later (such as Debian 13, current Fedora or Arch), and the ROCm 10 runtime
  from AMD's repository, installed in `/opt/rocm/core-10.0`. The kernel must
  be one AMD supports for these GPUs: on Ubuntu 24.04 the HWE kernel
  `6.17.0-19.19~24.04.2` or later or the OEM kernel `6.14.0-1018` or later,
  Ubuntu 26.04's own kernel, and Linux 6.18.4 or later elsewhere
  ([AMD's list](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)).

### ROCm on AMD

Add AMD's ROCm 10 repository, then install the runtime with the BLAS kernels
for your chip: `amdrocm-blas10.0-gfx1151` on Strix Halo,
`amdrocm-blas10.0-gfx1150` on Strix Point and Gorgon Point. AMD serves the
repository for Ubuntu 24.04 (`ubuntu2404`, as below) and 26.04 (`ubuntu2604`).

```sh
sudo curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /etc/apt/keyrings/amdrocm.asc
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.asc] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main' | sudo tee /etc/apt/sources.list.d/amdrocm.list
sudo apt update
sudo apt install amdrocm-blas10.0-gfx1151
sudo usermod -aG render "$USER"   # access to the GPU; log in again afterwards
```

The install script checks for the runtime, the BLAS kernels and access to
`/dev/kfd`, and prints the commands for your machine when something is
missing. It does not run `apt` itself.

### The NPU on AMD

From 0.5.3 the engine also uses the XDNA2 NPU of these machines beside the GPU
while it reads a prompt. This is optional: without any of the following the
engine prints one `[npu_ffn] the prefill stays on the GPU alone: ...` line when
it loads a model and runs on the GPU as before.

- **The NPU's kernel driver and firmware**, which make `/dev/accel/accel0`:
  the `amdxdna` driver (in Linux 6.14 and later; Ubuntu 26.04's kernel has it)
  and the `amdnpu` firmware from `linux-firmware`. The device belongs to the
  `render` group, like the GPU.
- **XRT, AMD's runtime for the NPU, with its NPU plugin.** On Ubuntu 26.04:
  `sudo apt install libxrt2 libxrt-npu2`.
- **An unlimited locked-memory limit**, because XRT locks a 64 MiB heap. Check
  with `ulimit -l`. For logins, put `* - memlock unlimited` in a file under
  `/etc/security/limits.d/` and log in again; for a systemd service, set
  `LimitMEMLOCK=infinity`. A session that does not pass through PAM keeps the
  limit of the service that started it.

On a machine with an NPU the install script checks these three and prints what
is missing, with the command or file for each. It changes none of them itself.

## One-shot install (recommended)

```sh
curl -LsSf https://basecompute.co/install.sh | sh
```

The script picks the bundle for this machine (macOS arm64, Linux arm64 with
CUDA, or Linux x86-64 with ROCm), unpacks it under `~/.basert/versions/<v>/`,
links it into `~/.basert`, installs Superfluid, and adds `~/.basert` to your
`PATH`. Restart your shell afterward (or `export PATH="$HOME/.basert:$PATH"`),
then:

```sh
basert pull basecompute/Qwen3-0.6B
basert chat basecompute/Qwen3-0.6B
```

Running the script again installs the target release the same way, and does
nothing when you are already on it.

Overrides (set as environment variables before running):

| Variable | Purpose | Default |
| --- | --- | --- |
| `BASERT_INSTALL_DIR` | Install location | `~/.basert` |
| `BASERT_VERSION` | Specific release tag (e.g. `v0.5.1`) | latest |
| `BASERT_FORCE` | Set to `1` to reinstall even if already up to date | unset |
| `BASERT_NO_MODIFY_PATH` | Set to `1` to skip editing shell profiles | unset |

## Package managers

Each release is also published to these registries, for the same three
platforms:

| Manager | Command | Installs |
| --- | --- | --- |
| Homebrew | `brew install basecompute/tap/basert` | CLI, engine, runtime tools |
| PyPI | `pip install basert` | CLI, engine, Python bindings (Python 3.9+) |
| npm | `npm install -g @basecompute/basert` | CLI, engine, Node.js bindings (Node 18+) |
| crates.io | `cargo install basert-cli` | CLI only (model hub and converter) |

Notes:

- `pip install basert` installs one wheel per platform; pip refuses it on a
  machine the engine cannot run on. `pipx install basert` and
  `uv tool install basert` work too.
- npm installs only the engine package for your platform as an optional
  dependency, so don't install with `--omit=optional`. In a project,
  `npm install @basecompute/basert` adds the Node.js bindings.
- `cargo install basert-cli` builds the CLI alone, without the engine or the
  runtime tools. Put a release bundle's directory on `PATH` and point
  `BASERT_LIB` at it (see [Environment variables](#environment-variables)).
- The Rust bindings (`baseRT`, `baseRT-sys` on crates.io) link the engine
  that Homebrew or the install script installed, or the one `BASERT_LIB_DIR`
  names. See [Bindings](../bindings/index.md).

## Updating

An install the script made updates itself:

```sh
basert self check            # is a newer release out?
basert self update           # install it, with the Superfluid it needs
basert self update --dry-run # say what would change, change nothing
basert self update --version 0.5.0   # go to a specific release, older ones too
```

Each release is unpacked beside the current one, checked (its SHA-256, and
that its launcher runs and is that version), and made current in one step, so
the CLI, the runtime tools and the engine always change together. The version
it replaces is kept:

```sh
basert self list             # versions installed, and which is in use
basert self use 0.5.0        # switch back to an installed version
```

When the installed Superfluid cannot serve the new engine, `self update`
installs the newest Superfluid release that can first; when none can yet,
nothing changes.

A command run in a terminal checks for a newer release once a day and prints
one line when there is one. Set `BASERT_NO_UPDATE_CHECK=1` to turn this off.

A copy Homebrew, npm, pip, pipx, uv or cargo installed belongs to that
manager. `basert self update` runs the manager's own update after asking
(`--yes` skips the question, and is needed without a terminal), for example
`brew upgrade basert` or `pip install --upgrade basert`.

## Superfluid, the server

`basert serve` runs [Superfluid](https://github.com/basecompute/superfluid),
an open-source inference server, with the engine beside `basert`. The install
script installs it. Otherwise the first `basert serve` installs the newest
Superfluid release that serves this engine (asking first in a terminal). To
do it ahead of time, or to check what is installed:

```sh
basert superfluid install    # --force installs again
basert superfluid status     # exit 0: an installed Superfluid can serve this engine
```

Superfluid lives in `superfluid/` beside `basert` for an install-script,
hand-extracted or source-built copy. A copy a package manager installed keeps
it in BaseRT's data directory instead (`~/Library/Application Support/baseRT/superfluid`
on macOS, `~/.local/share/baseRT/superfluid` on Linux), so upgrades and
uninstalls leave it alone. A Superfluid you installed yourself is never
touched.

Superfluid's own documentation is at
[docs.basecompute.co](https://docs.basecompute.co/latest/).

## Manual install

Each [GitHub release](https://github.com/basecompute/baseRT/releases) ships
one bundle per platform:

| Bundle | Platform |
| --- | --- |
| `basert-engine-macos-arm64-<version>.tar.gz` | macOS arm64 (Metal) |
| `basert-engine-linux-arm64-cuda-<version>.tar.gz` | Linux arm64 (NVIDIA CUDA) |
| `basert-engine-linux-x64-rocm-<version>.tar.gz` | Linux x86-64 (AMD ROCm) |

A bundle holds the engine library (`libbaseRT.dylib` or `libbaseRT.so`), the
`basert` CLI, the `basert-*` runtime tools, the kernel files, the public
headers (`include/baseRT/`), and its `LICENSE`. A `.sha256` file sits beside
each bundle.

### 1. Get the engine

```sh
gh release download --repo basecompute/baseRT --pattern 'basert-engine-macos-arm64-*.tar.gz'
mkdir -p build && tar -xzf basert-engine-macos-arm64-*.tar.gz -C build
```

Use the Linux bundle's name on Linux. See
[Engine releases](../reference/engine-releases.md) for what's in a bundle.

### 2. Build the `basert` CLI (optional)

The bundle already has the CLI. To build it from this repository instead
(Rust 1.85+):

```sh
cd base-convert
cargo build --release        # produces target/release/basert
cd ..
```

A CLI built here finds the engine through `BASERT_LIB`:

```sh
export BASERT_LIB="$PWD/build"
```

### 3. Put it on your `PATH`

The CLI runs `basert <cmd>` by finding a `basert-<cmd>` binary beside itself
or on `PATH`, so putting the bundle directory on `PATH` is enough:

```sh
export PATH="$PWD/build:$PATH"
# with a CLI built from source, put it first:
export PATH="$PWD/base-convert/target/release:$PWD/build:$PATH"
```

Verify:

```sh
basert --version
basert --help
basert superfluid status
```

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `BASERT_MODELS_DIR` | Where pulled and converted models are cached | `~/Library/Caches/baseRT/models` (macOS), `~/.cache/baseRT/models` (Linux) |
| `BASERT_STATE_DIR` | Where `basert serve` keeps its state (request log, uploaded files) | `serve/` beside the model cache |
| `BASERT_LIB` | Directory holding the engine library `basert serve` uses | beside `basert`, or its `../lib` |
| `BASERT_LIB_PATH` | Full path to the engine library, for the Python and Node bindings | the bundled or installed engine |
| `BASERT_LIB_DIR` | Directory holding the engine library, for the Rust and Swift bindings | the installed engine |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face auth for gated or private repos | the cached `hf` login |
| `BASERT_HF_CONNECTIONS` | Parallel range requests per download | `24` |
| `BASERT_HF_CHUNK_MB` | Bytes per range request (peak memory is connections × chunk) | `16` |
| `BASERT_HF_READ_TIMEOUT_SECS` | How long a download may stall before it is retried | `60` |
| `BASERT_HF_MAX_RETRIES` | Retries per chunk (`0` fails on the first error) | `5` |
| `BASERT_HF_XET` | Use Xet's CAS path for large files (deduplicated, but cannot resume) | unset |
| `BASERT_NO_UPDATE_CHECK` | Set to `1` to stop the daily check for a newer release | unset |
| `BASERT_SUPERFLUID_AUTO_INSTALL` | `0` forbids `basert serve` to download Superfluid; `1` downloads without asking | ask in a terminal |
| `BASERT_SUPERFLUID_DIR` | A directory of its own to run and install Superfluid in | see [above](#superfluid-the-server) |

## Next steps

- [Quickstart](quickstart.md) — pull a model and chat.
- [Managing models](../guides/models.md) — the model hub and cache layout.
- [Bindings](../bindings/index.md) — call BaseRT from your language.
