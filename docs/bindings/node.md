# Node

TypeScript / Node.js bindings (`@baseRT/node`) over the BaseRT C API via FFI.
Node 18+.

## Install

```sh
cd bindings/node
npm install
npm run build
```

Point at the engine dylib if needed:

```sh
export BASERT_LIB_PATH=/path/to/build
```

## Generate text

```ts
import { BaseRTModel } from "@baseRT/node";

const model = new BaseRTModel("models/your-model.base");
const tokens = model.encode("Hello, world!");

const stats = model.generate(tokens, 128, { temperature: 0.7 }, (id, text) => {
  process.stdout.write(text);   // streamed token callback
});
console.log(`\n\nGenerated ${stats.generatedTokens} tokens`);
```

## Continue a conversation

`generateContinue` keeps the KV cache, so a follow-up prompt doesn't reprocess
history:

```ts
model.generate(prompt1, 256, { temperature: 0.7 }, (id, t) => process.stdout.write(t));
model.generateContinue(prompt2, 256, { temperature: 0.7 }, (id, t) => process.stdout.write(t));
```

## Inspect & embed

```ts
const model = new BaseRTModel("models/your-model.base");
// model config, embeddings, tokenization helpers, and (for Whisper models)
// transcription are exposed on the model instance.
```

See [`bindings/node`](https://github.com/basecompute/baseRT/tree/main/bindings/node)
for the full API and the struct-ABI tests.
