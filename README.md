<div align="center">

# BaseRT

**Fastest LLM inference on Apple Silicon — now on NVIDIA GB10 (DGX Spark) and AMD Strix Halo / Strix Point, too.**

Pull a model from HuggingFace, chat with it, or serve it over OpenAI-, Anthropic- and Ollama-compatible APIs — all from one CLI.

[![Docs](https://img.shields.io/badge/docs-basert.basecompute.co-5b8fa8)](https://basert.basecompute.co)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%C2%B7%20NVIDIA%20GB10%20%C2%B7%20AMD%20Strix-black)](#requirements)
[![Bindings](https://img.shields.io/badge/bindings-Python%20%C2%B7%20Node%20%C2%B7%20Rust%20%C2%B7%20Swift-444)](docs/bindings/index.md)

</div>

---

BaseRT runs large language models locally on Apple Silicon (Metal), NVIDIA
GB10 / DGX Spark (CUDA) and AMD Strix Halo / Strix Point (ROCm), accelerated by
hand-written GPU kernels. This repository is the open ecosystem around the
engine: the `basert` CLI (model hub + converter), the `.base` model format, the
public C API, and language bindings for Python, Node, Rust, and Swift.

- **One CLI for everything** — `basert pull`, `chat`, `serve`, `convert`.
- **A local API server** — `basert serve` runs
  [Superfluid](https://github.com/basecompute/superfluid): OpenAI-, Anthropic-
  and Ollama-compatible endpoints, tool calls, continuous batching, prefix
  caching and speculative decoding.
- **Multimodal** — text, vision, and audio on supported models.
- **Its own `.base` format** — affine quantization (Q2–Q8), optional AWQ
  calibration, and signed bundles.
- **Bindings everywhere** — Python, Node, Rust, and Swift over a stable C API.
- **A drop-in coding-agent backend** — see [below](#use-it-as-a-coding-agent).

> **Documentation:** **[basert.basecompute.co](https://basert.basecompute.co)** (source in [`docs/`](docs/index.md)). The server's API is documented at [docs.basecompute.co](https://docs.basecompute.co).

## Requirements

One of:

- Apple Silicon (M1 or later), macOS 15+.
- NVIDIA GB10 (DGX Spark), Linux / arm64 with a CUDA driver.
- AMD Strix Halo (Ryzen AI Max, gfx1151) or Strix Point (Ryzen AI 300,
  gfx1150), Linux / x86-64 on Ubuntu 24.04 or newer, with the ROCm 10 runtime
  from AMD's apt repository (below). The kernel must be one AMD supports for
  RDNA3.5: on Ubuntu 24.04 the HWE kernel `6.17.0-19.19~24.04.2` or later or
  the OEM kernel `6.14.0-1018` or later, Ubuntu 26.04's own kernel, and
  elsewhere Linux 6.18.4 or later
  ([AMD's list](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)).

On AMD, add AMD's ROCm 10 repository, then install the runtime with the BLAS
kernels for your chip: `amdrocm-blas10.0-gfx1151` on Strix Halo,
`amdrocm-blas10.0-gfx1150` on Strix Point. AMD serves the repository for
Ubuntu 24.04 (`ubuntu2404`, as below) and 26.04 (`ubuntu2604`). The installer
checks for all of this and prints the commands for your machine.

```sh
sudo curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /etc/apt/keyrings/amdrocm.asc
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.asc] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main' | sudo tee /etc/apt/sources.list.d/amdrocm.list
sudo apt update
sudo apt install amdrocm-blas10.0-gfx1151
sudo usermod -aG render "$USER"   # access to the GPU; log in again afterwards
```

From 0.5.3 the engine also uses the XDNA2 NPU beside the GPU to read prompts
faster, when the machine has the NPU's driver (`/dev/accel/accel0`), AMD's XRT
(`sudo apt install libxrt2 libxrt-npu2` on Ubuntu 26.04) and an unlimited
locked-memory limit. Without them it runs on the GPU alone; the installer says
which are missing ([details](docs/getting-started/installation.md#the-npu-on-amd)).

## Quickstart

### 1. Install

```sh
curl -LsSf https://basecompute.co/install.sh | sh
```

This installs the prebuilt engine for your platform and the `basert` CLI to
`~/.basert` and adds it to your `PATH`. Restart your shell afterward (or
`export PATH="$HOME/.basert:$PATH"`). Update later with `basert self update`.

Or use a package manager:

```sh
brew install basecompute/tap/basert
pip install basert              # also pipx / uv tool
npm install -g @basecompute/basert
```

See [Installation](docs/getting-started/installation.md) for the details and
for building from source.

### 2. Pull a model

```sh
basert pull Qwen/Qwen3-4B    # download from HuggingFace + convert to .base
basert list                  # show installed models
```

Models are cached under `~/Library/Caches/baseRT/models` on macOS and
`~/.cache/baseRT/models` on Linux (override with `$BASERT_MODELS_DIR`). To pick a specific quant build, append `:default-q8` to
the id (or pass `--variant default-q8`).

### 3. Chat

```sh
basert chat Qwen/Qwen3-4B
```

### 4. Serve an API

```sh
basert serve Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}]}'
```

### 5. Or call it from your language

```python
import baseRT

model = baseRT.Model("models/your-model.base")
print(model.generate_text("The capital of France is", max_tokens=64))
```

Node, Rust, and Swift bindings work the same way — see
[Bindings](docs/bindings/index.md).

## Use it as a coding agent

BaseRT runs as a local backend for the [pi](https://pi.dev) coding agent through
the **[pi-basert](https://github.com/basecompute/pi-basert)** extension, which
auto-discovers whatever models your server is running.

```sh
# 1. Serve a model (a dense model is best for tool calling)
basert serve basecompute/gemma-4-E2B-it

# 2. Install the extension and launch pi
pi install git:github.com/basecompute/pi-basert
pi
```

Inside pi, run `/model` to pick a served model. See the
[pi-basert README](https://github.com/basecompute/pi-basert) for details.

## The `basert` CLI

`basert` is a single front-end. Model-management commands run natively; `chat`,
`complete` and `bench` run the matching `basert-<cmd>` engine binary, and `serve`
runs Superfluid, installing it on first use.

| Command | What it does |
| --- | --- |
| `basert pull <id>` | Download from HuggingFace (convert-on-pull) or the catalog |
| `basert list` | List installed models (`--remote` to include the catalog) |
| `basert convert <src>` | Convert a local GGUF / HF / MLX checkpoint to `.base` |
| `basert chat <model>` | Interactive chat |
| `basert serve <model>` | HTTP server (Superfluid; `--model` repeats for multi-model) |
| `basert complete <model>` | One-shot completion (text / image / audio) |
| `basert bench <model>` | Throughput benchmark |
| `basert inspect <model>` | Dump a `.base` header + tensor inventory |
| `basert sign` / `verify` / `keygen` | ed25519 signing of `.base` bundles |
| `basert computearena` | Run a ComputeArena benchmark and save a signed report |
| `basert superfluid install` / `status` | Install or check the server `basert serve` runs |
| `basert self update` / `check` / `list` / `use` | Update basert, or switch between installed versions |

Full flag reference: [CLI reference](docs/cli/reference.md).

## Updates

basert updates itself. Once a day, a command run in a terminal looks for a newer release in the background, so nothing waits on it. A release is taken as soon as it is out and a superfluid release serves its engine. The command after that updates to it:

- **An install the install script made** (`~/.basert`) updates in the foreground, with a progress bar, then runs the command on the new release:

  ```text
  Updating basert 0.5.4 → 0.5.5 · release notes: https://github.com/basecompute/baseRT/releases/tag/v0.5.5
  basert 0.5.5 is in use · `basert self use 0.5.4` goes back · `basert self auto-update off` turns automatic updates off
  ```

  When it cannot (offline, another `basert self` under way, a failed download, Ctrl-C), it says `couldn't update basert to 0.5.5: …; staying on 0.5.4` and tries again after the next day's look. A server already running keeps the engine it started with.
- **superfluid follows the engine.** The superfluid `basert serve` runs moves to the newest release that serves the new engine in the same update. When basert is already the newest, a newer superfluid that serves its engine is installed the same way. A superfluid a server runs from is never replaced while it runs.
- **A Homebrew, global npm, pipx, uv or cargo install** that is yours runs that manager's update in the background, once per release, logging to `auto-update.log` in basert's state directory.
- **A pip install, a project's npm dependency, an install another user owns, or a pinned Homebrew formula** gets a line once a day with the command that updates it.

| Switch | What it turns off | For |
|---|---|---|
| `basert self auto-update off` (`on` again; no argument says which and why) | automatic updates | every command, from now on |
| `BASERT_NO_UPDATE=1` | the look, updates and the line | a shell, a service, a script |
| `--no-update` on any command | the look, updates and the line | that one run |

Nothing updates without a terminal (an app such as Superfluid Desktop, a service), with `CI` set, as root, or for a command that only reports (`--help`, `--version`, `list`, `inspect`, `verify`, anything with `--json`).

**The wait.** `basert self auto-update wait 3` takes a release only once it has been out three days (`0`, the default, takes it as soon as it is out). `BASERT_UPDATE_WAIT_DAYS=<days>` sets it for one shell or service.

**By hand.** `basert self update` installs the newest release now, `basert self check` says what is out, `basert self list` shows the releases installed side by side under `~/.basert/versions/`, and `basert self use <version>` goes back to one; automatic updates then leave it in use until a newer release is out.

## What's in this repo

| Path | Contents |
| --- | --- |
| [`base-convert/`](base-convert/) | The `basert` CLI (Rust): model hub, converter, launcher — plus the [`.base` format](base-convert/FORMAT.md) and [quantization](base-convert/CANONICAL_QUANT_SPEC.md) specs and the generic quant profiles. |
| [`include/baseRT/`](include/baseRT/) | The stable public C API (`baseRT.h`, `types.h`). |
| [`bindings/`](bindings/) | Python, Node, Rust, and Swift. |
| [`benchmarks/`](benchmarks/) | Scripts and reference results. |
| [`docs/`](docs/index.md) | The documentation site ([basert.basecompute.co](https://basert.basecompute.co)). |

This repository — the CLI, `.base` format, public headers, and bindings — is
licensed under **Apache-2.0**. The engine ships as a separate prebuilt binary
under its own license; see
[Engine releases](docs/reference/engine-releases.md).

## Links

- **[Documentation](https://basert.basecompute.co)** · [Server API](docs/reference/server-api.md) · [`.base` format](base-convert/FORMAT.md)
- [Security policy](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The prebuilt engine
binary distributed via [Releases](https://github.com/basecompute/baseRT/releases)
is proprietary and ships under its own license.
