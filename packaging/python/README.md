# BaseRT

Fast local LLM inference on Apple Silicon (Metal) and NVIDIA GB10 / DGX Spark
(CUDA). This package installs the `basert` CLI, the prebuilt engine, and the
Python bindings in one step.

```sh
pip install basert
```

Requires Apple Silicon with macOS 15+, or Linux/arm64 with an NVIDIA GPU and
the CUDA 13 runtime (cuBLAS).

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
