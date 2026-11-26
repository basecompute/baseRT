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

From 0.5.5, basert updates itself. Once a day, a command run in a terminal
looks for a newer release, in a process of its own so that the command does
not wait on it. A release is taken once it is three days old, so that early
reports land first, and only when a Superfluid release that old serves its
engine. The next command updates to it.

An install the script made updates in the foreground, with a progress bar for
the engine and for Superfluid, then runs the command on the new release:

```text
Updating basert 0.5.4 → 0.5.5 · release notes: https://github.com/basecompute/baseRT/releases/tag/v0.5.5
  basert 0.5.5       [==============================]  41.2 MiB / 41.2 MiB  …
  superfluid v0.2.2  [==============================]  18.0 MiB / 18.0 MiB  …
basert 0.5.5 is in use · `basert self use 0.5.4` goes back · `basert self auto-update off` turns automatic updates off
```

When it cannot update (offline, another `basert self` command under way, a
download or check that fails, Ctrl-C), it prints
`couldn't update basert to 0.5.5: …; staying on 0.5.4`, runs the command on
the version in use, and tries again after the next day's look. A server
already running keeps the engine it started with; starting it again uses the
new one.

Each release is unpacked beside the current one, checked (its SHA-256, and
that its launcher runs and is that version), and made current in one step, so
the CLI, the runtime tools and the engine always change together. The version
it replaces is kept for going back.

### Superfluid follows the engine

An update brings the Superfluid `basert serve` runs to the newest release
that serves the new engine; when none serves it yet, basert stays where it
is. When basert is not updated first (it is the newest, it is a package
manager's to update, or you went back to an older one), a newer Superfluid
that serves its engine comes by itself the same way, in the foreground. That
includes the Superfluid in basert's data directory for a Homebrew, npm, pip,
pipx, uv or cargo install:

```text
Updating superfluid v0.2.1 → v0.2.2 · release notes: https://github.com/basecompute/superfluid/releases/tag/v0.2.2
  superfluid v0.2.2  [==============================]  18.0 MiB / 18.0 MiB  …
superfluid v0.2.2 is in use · `basert superfluid install v0.2.1` goes back · `basert self auto-update off` turns automatic updates off
```

A Superfluid a server runs from is never replaced: it stays until no server
uses it, and the next command after that updates it. A Superfluid in a
`BASERT_SUPERFLUID_DIR` that basert did not install does not come forward by
itself; an update of basert still brings it to the newest release that serves
the new engine, as `basert self update` does, unless its directory holds more
than Superfluid.

### A package manager's install

- **Homebrew, a global npm, pipx, uv or cargo.** When the install is yours
  (no root needed to change it) and (except for cargo) no server runs on its
  engine, basert runs that manager's update in the background once the
  command is done, once per release and pinned to that release, and prints
  one line:
  `Updating basert 0.5.4 → 0.5.5 with Homebrew in the background (log: …)`.
  While a server runs on its engine, it gets the line below instead (except
  cargo, which replaces only the launcher and runs anyway).
  The commands are `brew upgrade basert` (only when the formula is that
  release), `npm install --global @basecompute/basert@<version>`,
  `pipx install --force basert==<version>`,
  `uv tool install basert==<version>` and
  `cargo install basert-cli --version <version>` (a build from source, a few
  minutes). Their output goes to `auto-update.log` in basert's state
  directory (`serve/` beside the model cache, or `BASERT_STATE_DIR`).
- **pip, a project's npm dependency, an install another user owns or this
  one cannot write, or a pinned Homebrew formula** is not basert's to change.
  Once a day a line names the command that updates it, with the release
  notes. For pip that is `pip install --upgrade basert` (as
  `python -m pip install --upgrade basert` with that environment's python);
  basert never runs it, since the environment may be a project's, pinned by a
  lock file, or the system's.

### Turning automatic updates off

Each of these updates, and each line naming a command, links the release
notes and says how to turn automatic updates off. Three switches do:

| Switch | Turns them off for |
| --- | --- |
| `basert self auto-update off` | every command, from now on (`on` turns them back on; with no argument it says which, and why when off) |
| `BASERT_NO_UPDATE=1` | a shell, a service, a script |
| `--no-update` on any command | that one run |

With these, the once-a-day line saying a newer release is out stays.
`BASERT_NO_UPDATE_CHECK=1` turns automatic updates off too, and the daily
look and the line with them. `basert self auto-update off` writes
`"autoUpdate": false` to `settings.json` in basert's config directory
(`~/Library/Application Support/baseRT` on macOS, `~/.config/baseRT` on
Linux; `BASERT_CONFIG_DIR` instead). A `settings.json` that cannot be read,
or whose `autoUpdate` is neither `true` nor `false`, leaves automatic updates
off, and `basert self auto-update` says why.

Nothing updates without a terminal on stderr (an app such as Superfluid
Desktop, a service), with `CI` set, as root (`sudo basert …`), or for a
command that only tells something (`basert self …`, `--help`, `--version`,
`list`, `inspect`, `verify`, `superfluid`, anything with `--json`).

### By hand, and going back

```sh
basert self check            # is a newer release out?
basert self update           # install the newest now, with the Superfluid it needs
basert self update --dry-run # say what would change, change nothing
basert self update --version 0.5.4   # go to a specific release, older ones too
```

`basert self update` goes to the newest release at once, three days old or
not. For an install the script made, releases sit side by side under
`~/.basert/versions/`:

```sh
basert self list             # versions installed, and which is in use
basert self use 0.5.4        # switch back to an installed version
```

After going back with `basert self use` or `basert self update --version`,
automatic updates leave that version in use until a release newer than the
one you left is out; `basert self update` goes forward now. The same holds
for Superfluid: `basert superfluid install <release>` (for example
`basert superfluid install v0.2.1`) goes back to that release, and
Superfluid's own automatic updates then leave it in place until a release
newer than the one it replaced is out. An update of basert itself still brings
it to the newest release that serves the new engine.

For a Homebrew, npm, pipx, uv or cargo install, `basert self update` runs the
manager's own update after asking (`--yes` skips the question, and is needed
without a terminal): `brew upgrade basert`,
`npm install --global @basecompute/basert@latest`, `pipx upgrade basert`,
`uv tool upgrade basert` or `cargo install basert-cli`. For pip it names
`pip install --upgrade basert` and runs nothing. `--version` is the
manager's to choose there.

## Superfluid, the server

`basert serve` runs [Superfluid](https://github.com/basecompute/superfluid),
an open-source inference server, with the engine beside `basert`. The install
script installs it. Otherwise the first `basert serve` installs the newest
Superfluid release that serves this engine (asking first in a terminal). To
do it ahead of time, to go back to an earlier release, or to check what is
installed:

```sh
basert superfluid install            # the newest release that serves this engine; --force installs again
basert superfluid install v0.2.1     # from 0.5.5: this release instead (one that serves this engine), an older one too
basert superfluid status             # exit 0: an installed Superfluid can serve this engine
```

basert keeps the Superfluid it installed up to date with its own updates (see
[Superfluid follows the engine](#superfluid-follows-the-engine)).

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
| `BASERT_STATE_DIR` | Where `basert serve` keeps its state (request log, uploaded files), and basert its update check and `auto-update.log` | `serve/` beside the model cache |
| `BASERT_LIB` | Directory holding the engine library `basert serve` uses | beside `basert`, or its `../lib` |
| `BASERT_LIB_PATH` | Full path to the engine library, for the Python and Node bindings | the bundled or installed engine |
| `BASERT_LIB_DIR` | Directory holding the engine library, for the Rust and Swift bindings | the installed engine |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face auth for gated or private repos | the cached `hf` login |
| `BASERT_HF_CONNECTIONS` | Parallel range requests per download | `24` |
| `BASERT_HF_CHUNK_MB` | Bytes per range request (peak memory is connections × chunk) | `16` |
| `BASERT_HF_READ_TIMEOUT_SECS` | How long a download may stall before it is retried | `60` |
| `BASERT_HF_MAX_RETRIES` | Retries per chunk (`0` fails on the first error) | `5` |
| `BASERT_HF_XET` | Use Xet's CAS path for large files (deduplicated, but cannot resume) | unset |
| `BASERT_NO_UPDATE` | Set to `1` to turn automatic updates off; the daily line saying a newer release is out stays (see [Updating](#updating)) | unset |
| `BASERT_NO_UPDATE_CHECK` | Set to `1` to stop the daily look for a newer release: no automatic update and no line | unset |
| `BASERT_CONFIG_DIR` | Where basert keeps `settings.json` (the `autoUpdate` setting) | `~/Library/Application Support/baseRT` (macOS), `~/.config/baseRT` (Linux) |
| `BASERT_SUPERFLUID_AUTO_INSTALL` | `0` forbids `basert serve` to download Superfluid; `1` downloads without asking | ask in a terminal |
| `BASERT_SUPERFLUID_DIR` | A directory of its own to run and install Superfluid in | see [above](#superfluid-the-server) |

## Next steps

- [Quickstart](quickstart.md) — pull a model and chat.
- [Managing models](../guides/models.md) — the model hub and cache layout.
- [Bindings](../bindings/index.md) — call BaseRT from your language.
