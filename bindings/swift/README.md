# BaseRT Swift Bindings

A Swift package wrapping the BaseRT inference engine. The C headers
(`baseRT.h` / `types.h`) are vendored into the `CBaseRT` module, so the
package is self-contained and builds against the prebuilt engine.

## Requirements

- macOS 13+ / iOS 16+, Swift 5.9+
- The prebuilt engine: run downloading an engine release (see the top-level README "Getting the engine") in the repo root to produce
  `build/libbaseRT.dylib`. The dylib carries the Metal kernels **embedded**,
  so there is no separate `baseRT.metallib` to ship or locate at runtime.

## Setup

Point the package at the directory containing `libbaseRT.dylib` with the
`BASERT_LIB_DIR` environment variable (defaults to the repo's `build/`):

```bash
export BASERT_LIB_DIR=/path/to/baseRT/build
swift build
swift test                     # pure unit tests; no model needed
# optional end-to-end test:
BASERT_TEST_MODEL=/path/to/model.base swift test
```

Add it as a local dependency in your app's `Package.swift`:

```swift
.package(path: "/path/to/baseRT/bindings/swift")
```

The `BaseRT` target already links `-lbaseRT` and embeds an rpath to
`BASERT_LIB_DIR`, so consuming targets just depend on the product:

```swift
.target(
    name: "MyApp",
    dependencies: [.product(name: "BaseRT", package: "BaseRT")]
)
```

> Distributing a resolvable (non-path) package: replace the `unsafeFlags`
> linker settings in `Package.swift` with an XCFramework binary target
> wrapping `libbaseRT.dylib`.

## Usage

### Text Generation

```swift
import BaseRT

// Process-wide settings (optional) — must be set before loading any model:
//   BaseRTEngine.setPagedKV(true)
//   BaseRTEngine.setKVBits(8)
print("engine \(BaseRTEngine.version)")

// Load a model. kernelLibraryPath defaults to nil → the metallib embedded in
// libbaseRT.dylib is used (no sidecar file needed).
let model = try BaseRTModel(modelPath: "models/Qwen3-0.6B-Q4_0.base")

// Check model info
print("Architecture: \(model.config.architecture)")
print("Memory: \(model.memoryUsage / 1_048_576) MB")

// Encode and generate
let tokens = try model.encode(text: "Once upon a time")
let stats = model.generate(tokens: tokens, maxTokens: 256) { token in
    print(token.text, terminator: "")
    return true  // return false to stop early
}
print("\nDecode: \(stats.decodeTokensPerSec) tok/s")
```

### Sampling Configuration

```swift
let sampling = SamplingConfig(
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    minP: 0.05,
    repeatPenalty: 1.1
)

model.generate(tokens: tokens, maxTokens: 512, sampling: sampling) { token in
    print(token.text, terminator: "")
    return true
}
```

### Multi-turn Chat

```swift
// First turn
let turn1 = try model.encode(text: "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")
model.generate(tokens: turn1, maxTokens: 256) { token in
    print(token.text, terminator: "")
    return true
}

// Continue from existing KV cache
let turn2 = try model.encode(text: "<|im_end|>\n<|im_start|>user\nTell me more.<|im_end|>\n<|im_start|>assistant\n")
model.generateContinue(tokens: turn2, maxTokens: 256) { token in
    print(token.text, terminator: "")
    return true
}
```

### Async Streaming (Swift Concurrency)

```swift
let tokens = try model.encode(text: "Explain quantum computing")

for await token in model.stream(tokens: tokens, maxTokens: 256) {
    print(token.text, terminator: "")
}
```

### Whisper Transcription

```swift
let whisper = try BaseRTModel(modelPath: "models/whisper-base-en.bin")

// From a WAV file
let (text, stats) = try whisper.transcribe(wavPath: "audio.wav")
print(text)
print("Took \(stats.totalMs)ms")

// Disable timestamps for faster plain-text output
whisper.setTimestamps(enabled: false)
let (plainText, _) = try whisper.transcribe(wavPath: "audio.wav")
print(plainText)

// From raw PCM samples (16kHz, mono, Float32)
let samples: [Float] = loadAudioSamples()
let (pcmText, pcmStats) = try whisper.transcribePCM(samples: samples)
```

### Low-level API

```swift
// Manual prefill + decode loop
let firstToken = model.prefill(tokens: tokens)
var tok = firstToken
for pos in tokens.count..<(tokens.count + 100) {
    tok = model.decodeStep(tokenID: tok, position: pos)
    print(model.decodeToken(tok), terminator: "")
}

// Reset state for a new conversation
model.reset()
```

### Model Inspection

```swift
for i in 0..<model.tensorCount {
    if let name = model.tensorName(at: i) {
        print("\(name): dtype=\(model.tensorDtype(at: i))")
    }
}
```
