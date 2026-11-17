# Swift

A Swift package wrapping the BaseRT engine. The C headers (`baseRT.h` /
`types.h`) are vendored into the `CBaseRT` module; the `BaseRT` target links
`-lbaseRT` and embeds an rpath. Swift 5.9+. The manifest declares macOS 13+ /
iOS 16+, but the prebuilt engine needs an Apple Silicon Mac on macOS 15 or
later.

## Setup

Install the engine (Homebrew, the install script, or an
[engine release](../reference/engine-releases.md)) and point the package at
the directory containing `libbaseRT.dylib`:

```sh
export BASERT_LIB_DIR=/opt/homebrew/opt/basert/libexec   # Homebrew
# or: export BASERT_LIB_DIR=$HOME/.basert                 # install script
```

Without it, the package uses the checkout's `build/` directory. The package is
used from a local checkout (its linker flags keep it out of remote package
resolution):

```swift
dependencies: [.package(path: "/path/to/baseRT/bindings/swift")],
targets: [
    .target(name: "MyApp", dependencies: [.product(name: "BaseRT", package: "BaseRT")]),
]
```

## Usage

```swift
import BaseRT

// Optional engine-wide config before loading:
//   BaseRTEngine.setKVBits(8)
//   BaseRTEngine.setNeuralEngine(false)   // prefill on the GPU alone
print("engine \(BaseRTEngine.version)")

let model = try BaseRTModel(modelPath: "models/your-model.base")

let tokens = try model.encode(text: "Once upon a time")
let stats = try model.generate(tokens: tokens, maxTokens: 256) { token in
    print(token.text, terminator: "")
    return true   // keep generating
}
print("\n\(stats.generatedTokens) tokens")
```

Or as an `AsyncSequence`:

```swift
for await token in model.stream(tokens: tokens, maxTokens: 256) {
    print(token.text, terminator: "")
}
```

`generate` and `stream` take an optional `SamplingConfig(temperature:topK:topP:
minP:repeatPenalty:presencePenalty:frequencyPenalty:seed:logitBias:)`;
`generateContinue` / `streamContinue` continue from the KV cache. The model
also has `generateText`, `embed` / `embedText`, `formatChat`, `tokenCount`,
`prefillImage` / `prefillAudio`, `saveState` / `loadState`, LoRA adapters
(`loadLoRA`, `unloadLoRA`), and for Whisper models `transcribe` /
`transcribePCM`. Calls that touch the model throw
`BaseRTError.reentrantModelCall` if made from inside that model's token
callback.

See [`bindings/swift`](https://github.com/basecompute/baseRT/tree/main/bindings/swift)
for the full API.
