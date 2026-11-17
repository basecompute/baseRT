# Node

TypeScript / Node.js bindings over the BaseRT C API, built on the
[koffi](https://koffi.dev) FFI (no native addon to compile). Node 18+.

## Install

```sh
npm install @basecompute/basert      # the bindings, in a project
npm install -g @basecompute/basert   # the basert command
```

[`@basecompute/basert`](https://www.npmjs.com/package/@basecompute/basert)
carries the bindings and the `basert` command. The prebuilt engine comes as
one optional dependency per platform, and npm installs only the one that
matches the machine, so don't install with `--omit=optional`:

| Engine package | Platform |
| --- | --- |
| `@basecompute/basert-darwin-arm64` | Apple Silicon, macOS 15 or later |
| `@basecompute/basert-linux-arm64-cuda` | Linux arm64 with an NVIDIA GPU and the CUDA 13 runtime |
| `@basecompute/basert-linux-x64-rocm` | Linux x86-64 with an AMD Strix Halo or Strix Point GPU and the ROCm 10 runtime |

From a source checkout:

```sh
cd bindings/node
npm install
npm run build
export BASERT_LIB_PATH=/path/to/libbaseRT.dylib   # or libbaseRT.so on Linux
```

Without `BASERT_LIB_PATH`, the bindings look for the npm engine package, then
the checkout's `build/`, then `~/.basert` (the install script) and Homebrew.
See [Finding the engine](index.md#finding-the-engine).

## Generate text

```ts
import { BaseRTModel } from "@basecompute/basert";

const model = new BaseRTModel("models/your-model.base");
const tokens = model.encode("Hello, world!");

const stats = model.generate(tokens, 128, { temperature: 0.7 }, (id, text) => {
  process.stdout.write(text);   // streamed token callback
  return true;                  // false stops generation
});
console.log(`\n\nGenerated ${stats.generatedTokens} tokens`);
model.close();
```

`new BaseRTModel(path, kernelLibraryPath = null, maxContext = 0)` loads a
bundle; `maxContext = 0` takes the model's default window. Calls are
synchronous and block the calling thread; run them in a `worker_thread` to
keep the event loop free. `close()` frees the GPU resources — call it before
the program exits. `generateText(tokens, maxTokens, sampling)` returns the
whole text instead of streaming it.

## Continue a conversation

`generateContinue` keeps the KV cache, so a follow-up prompt doesn't reprocess
history:

```ts
const write = (_id: number, t: string) => { process.stdout.write(t); return true; };
model.generate(prompt1, 256, { temperature: 0.7 }, write);
model.generateContinue(prompt2, 256, { temperature: 0.7 }, write);
```

The sampling options are `temperature`, `topK`, `topP`, `minP`,
`repeatPenalty`, `presencePenalty`, `frequencyPenalty`, `seed` and
`logitBias`.

## Inspect, embed and transcribe

```ts
const model = new BaseRTModel("models/your-model.base");
console.log(model.config.architecture, model.memoryUsage);

const vec = model.embedText("hello world");      // embedding models
const prompt = model.formatChat("You are helpful.", "Hi!");
```

Also on the model: `tokenCount()`, `decodeToken()`, `prefill()`,
`decodeStep()`, `chainDecode()`, `reset()`, `prefillImage()` /
`imageNumTokens()`, `prefillAudio()` / `audioNumTokens()`, `embed()`,
`embeddingDim`, `chatTemplate`, `position`, `setSpeculation()`, and for
Whisper models `transcribe()`, `transcribePcm()`, `transcribeStream()`,
`transcribePcmStream()` and `transcribeSegments()`.

## Engine settings

`Engine` holds the process-wide settings; set them before constructing a
model:

```ts
import { Engine } from "@basecompute/basert";

console.log(Engine.version());
Engine.setKvBits(8);              // KV-cache width: 0 = auto, 8, 16
Engine.setNeuralEngine(false);    // prefill on the GPU alone (Apple Silicon)
```

See [`bindings/node`](https://github.com/basecompute/baseRT/tree/main/bindings/node)
for the full API and the struct-ABI tests.
