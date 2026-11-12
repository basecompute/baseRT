# @basecompute/basert

Fast local LLM inference on Apple Silicon (Metal), NVIDIA GB10 / DGX Spark
(CUDA) and AMD Strix Halo / Strix Point (ROCm). This package installs the
`basert` CLI, the prebuilt engine, and the Node.js / TypeScript bindings.

```sh
npm install -g @basecompute/basert   # the CLI
npm install @basecompute/basert      # the bindings, in a project
```

Requires Node 18+, and one of:

- Apple Silicon with macOS 15+;
- Linux/arm64 with an NVIDIA GPU and the CUDA 13 runtime (cuBLAS);
- Linux/x64 with an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU and
  the ROCm 10 runtime in `/opt/rocm/core-10.0`
  (`sudo apt install amdrocm-blas10.0-gfx1151`, or `-gfx1150`).

npm installs only the engine for your platform
(`@basecompute/basert-darwin-arm64`, `@basecompute/basert-linux-arm64-cuda` or
`@basecompute/basert-linux-x64-rocm`), so don't install with
`--omit=optional`.

## CLI

```sh
basert pull Qwen/Qwen3-4B      # download from Hugging Face and convert to .base
basert chat Qwen/Qwen3-4B      # chat in the terminal
basert serve Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080   # OpenAI-compatible API
```

## Node.js

```typescript
import { BaseRTModel } from "@basecompute/basert";

const model = new BaseRTModel("model.base");
const text = model.generateText(model.encode("Hello, world!"), 128);
model.close();
```

Docs: <https://docs.basecompute.co> · Source: <https://github.com/basecompute/baseRT>

## License

The bindings and the `basert` CLI launcher are Apache-2.0. The engine packages
are under the BaseRT Engine license included in each.
