# BaseRT

Fast local LLM inference on Apple Silicon (Metal), NVIDIA GB10 / DGX Spark
(CUDA) and AMD Strix Halo / Strix Point (ROCm). This package installs the
`basert` CLI, the prebuilt engine, and the Python bindings in one step.

```sh
pip install basert
```

Requires one of:

- Apple Silicon with macOS 15+;
- Linux/arm64 with an NVIDIA GPU and the CUDA 13 runtime (cuBLAS);
- Linux/x86-64 with an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU
  and the ROCm 10 runtime in `/opt/rocm/core-10.0`
  (`sudo apt install amdrocm-blas10.0-gfx1151`, or `-gfx1150`).

## CLI

```sh
basert pull Qwen/Qwen3-4B      # download from Hugging Face and convert to .base
basert chat Qwen/Qwen3-4B      # chat in the terminal
basert serve Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080   # OpenAI-compatible API
```

### Updates

basert does not change a Python environment itself: it may be a project's,
pinned by a lock file. Once a release is out, a `basert` command run on a
terminal prints, once a day, the command that updates it
(`python -m pip install --upgrade basert` with that environment's python)
and a link to the release notes. A pipx or uv tool install is updated in the
background, pinned to that release (`pipx install --force basert==<version>`
or `uv tool install basert==<version>`); `basert self update` runs
`pipx upgrade basert` or `uv tool upgrade basert` after asking.
`basert self auto-update wait <days>` (or `BASERT_UPDATE_WAIT_DAYS`) waits that
many days after a release first (default 0). `basert self auto-update off`
turns automatic updates off and leaves the line; `BASERT_NO_UPDATE=1` or
`--no-update` (one run) turn off the daily look and the line as well.

## Python

```python
import baseRT

with baseRT.Model("model.base") as model:
    print(model.generate_text("The capital of France is", max_tokens=64))
```

Docs: <https://docs.basecompute.co> · Source: <https://github.com/basecompute/baseRT>

## License

The Python bindings and the `basert` CLI launcher are Apache-2.0. The bundled
engine binaries (`baseRT/_engine/`) are under the BaseRT Engine license
included with the package.
