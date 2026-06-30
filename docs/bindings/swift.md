# Swift

A Swift package wrapping the BaseRT engine. The C headers (`baseRT.h` /
`types.h`) are vendored into the `CBaseRT` module; the `BaseRT` target links
`-lbaseRT` and embeds an rpath. Swift 5.9+.

## Setup

Make `libbaseRT.dylib` available and point the package at it:

```sh
export BASERT_LIB_DIR=/path/to/build
```

Add the package as a dependency:

```swift
dependencies: [.product(name: "BaseRT", package: "BaseRT")]
```

## Usage

```swift
import BaseRT

// Optional engine-wide config before loading:
//   BaseRTEngine.setPagedKV(true)
//   BaseRTEngine.setKVBits(8)
print("engine \(BaseRTEngine.version)")

let model = try BaseRTModel(modelPath: "models/your-model.base")

let tokens = try model.encode(text: "Once upon a time")
let stats = model.generate(tokens: tokens, maxTokens: 256) { token in
    print(token, terminator: "")
    return true   // keep generating
}
```

See [`bindings/swift`](https://github.com/basecompute/baseRT/tree/main/bindings/swift)
for the full API.
