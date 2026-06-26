# @baseRT/node

Node.js / TypeScript bindings for [BaseRT](../../README.md) — an LLM inference
engine for Apple Silicon.

Built on [**koffi**](https://koffi.dev), a maintained, prebuilt FFI: no
node-gyp / native addon compilation, correct struct-by-value and C-callback
handling, and the native library is loaded lazily (only on first model
construction), so the module imports cleanly in test/CI environments without a
built engine.

## Prerequisites

- Node.js >= 18, macOS with Apple Silicon
- The prebuilt engine: `make shared` in the repo root → `build/libbaseRT.dylib`.
  The dylib carries the Metal kernels **embedded**, so there is no separate
  `baseRT.metallib` to locate at runtime.

## Installation

```bash
cd bindings/node
npm install
npm run build
npm test          # struct-ABI + helper tests; no engine/model needed
```

## Library Path

The bindings resolve `libbaseRT.dylib` in this order:

1. `BASERT_LIB_PATH` environment variable (full path to the `.dylib`)
2. `../../build/libbaseRT.dylib` relative to the package (when inside the repo)
3. `build/libbaseRT.dylib` / `libbaseRT.dylib` under the current directory

## Note on threading

Generation calls are **synchronous** and block the calling thread for their
duration (local inference is GPU-bound); the `onToken` callback fires
synchronously per token. Run a model in a `worker_thread` if you need the event
loop free or want concurrent sequences.

## Usage

### Text Generation

```typescript
import { BaseRTModel } from "@baseRT/node";

const model = new BaseRTModel("models/Qwen3-0.6B-Q4_0.base");

// Encode a prompt
const tokens = model.encode("The meaning of life is");

// Generate with streaming
const stats = model.generate(tokens, 128, { temperature: 0.7 }, (id, text) => {
  process.stdout.write(text);
  return true; // return false to stop early
});

console.log(`\n\nGenerated ${stats.generatedTokens} tokens`);
console.log(`Decode: ${stats.decodeTokensPerSec.toFixed(1)} tok/s`);

model.close();
```

### Multi-turn Chat

```typescript
const model = new BaseRTModel("models/Qwen3-0.6B-Q4_0.base");

// First turn
const prompt1 = model.encode("<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n");
model.generate(prompt1, 256, { temperature: 0.7 }, (id, text) => {
  process.stdout.write(text);
  return true;
});

// Continue from existing KV cache (no re-prefill of previous turns)
const prompt2 = model.encode("<|im_end|>\n<|im_start|>user\nTell me more.<|im_end|>\n<|im_start|>assistant\n");
model.generateContinue(prompt2, 256, { temperature: 0.7 }, (id, text) => {
  process.stdout.write(text);
  return true;
});

model.close();
```

### Audio Transcription (Whisper)

```typescript
const model = new BaseRTModel("models/whisper-base-en.bin");

// From WAV file
const { text, stats } = model.transcribe("audio.wav");
console.log(text);
console.log(`Transcribed in ${stats.totalMs.toFixed(0)}ms`);

// From raw PCM (16kHz mono float32)
const samples = new Float32Array(16000 * 5); // 5 seconds
const result = model.transcribePcm(samples, "en");
console.log(result.text);

// Disable timestamps for faster plain-text output
model.setTimestamps(false);
const fast = model.transcribe("audio.wav");
console.log(fast.text);

model.close();
```

### Model Inspection

```typescript
const model = new BaseRTModel("models/Qwen3-0.6B-Q4_0.base");

const cfg = model.config;
console.log(`Architecture: ${cfg.architecture}`);
console.log(`Parameters: ${cfg.dim}d, ${cfg.nLayers}L, ${cfg.nHeads}H`);
console.log(`Vocab: ${cfg.vocabSize}, Max context: ${cfg.maxSeqLen}`);
console.log(`GPU memory: ${(model.memoryUsage / 1e6).toFixed(1)} MB`);
console.log(`Is Whisper: ${model.isWhisper}`);

model.close();
```

### Sampling Configuration

All fields are optional with sensible defaults:

```typescript
const sampling = {
  temperature: 0.8,    // 0 = greedy (default)
  topK: 40,            // default: 40
  topP: 0.9,           // default: 0.9
  minP: 0.0,           // default: 0.0
  repeatPenalty: 1.1,   // default: 1.0 (no penalty)
};
```

### Low-level API

```typescript
const model = new BaseRTModel("models/Qwen3-0.6B-Q4_0.base");
const tokens = model.encode("Hello");

const firstToken = model.prefill(tokens);
console.log(`First token: ${model.decodeToken(firstToken)}`);

const nextToken = model.decodeStep(firstToken, tokens.length);
console.log(`Next token: ${model.decodeToken(nextToken)}`);

// Chain decode: batch multiple tokens in one GPU call
const chain = model.chainDecode(nextToken, tokens.length + 1, 10);
console.log(`Chain: ${chain.map(t => model.decodeToken(t)).join("")}`);

console.log(`Position: ${model.position}`);

model.reset();
model.close();
```

## API Reference

### `BaseRTModel`

| Method / Property | Description |
|---|---|
| `constructor(modelPath, kernelLibraryPath?, maxContext?)` | Load a model |
| `close()` | Free GPU resources |
| `config` | Model configuration (readonly) |
| `memoryUsage` | GPU memory in bytes (readonly) |
| `isWhisper` | Whether model is Whisper (readonly) |
| `encode(text)` | Tokenize text |
| `decodeToken(id)` | Token ID to string |
| `generate(tokens, max, sampling?, onToken?)` | Generate from prompt |
| `generateText(tokens, max, sampling?)` | Generate and return the full string |
| `generateContinue(tokens, max, sampling?, onToken?)` | Continue from KV cache |
| `embed(tokens)` / `embedText(text)` | Compute embeddings |
| `formatChat(system, user)` / `chatTemplate` | Native chat template |
| `tokenCount(text)` | Count tokens without allocating |
| `prefillImage(tokens, path)` / `prefillAudio(tokens, pcm)` | Multimodal prefill |
| `prefill(tokens)` | Low-level prefill |
| `decodeStep(tokenId, position)` | Single decode step |
| `chainDecode(firstToken, startPos, count)` | Batched decode |
| `position` | Current KV cache position |
| `setSpeculation(enabled)` | Toggle n-gram speculation |
| `setTimestamps(enabled)` | Toggle Whisper timestamps |
| `reset()` | Clear KV cache |
| `transcribe(wavPath, language?)` | Transcribe WAV file |
| `transcribePcm(samples, language?)` | Transcribe raw PCM |

### `Engine` (process-wide settings — set before constructing a model)

| Member | Description |
|---|---|
| `Engine.version()` | Runtime engine version string |
| `Engine.setKvBits(bits)` | KV-cache width: 0=auto, 8=Q8_0, 16=F16 |
| `Engine.setPagedKv(enable)` | Toggle paged-KV mode |
| `Engine.setMaxBatchSize(n)` | Max in-flight batch size |
