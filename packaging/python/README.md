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
