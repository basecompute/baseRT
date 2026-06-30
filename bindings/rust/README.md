# BaseRT Rust Bindings

Rust bindings for the [BaseRT](../../) LLM inference engine (Apple Silicon / Metal).

Two crates are provided:

- **`baseRT-sys`** -- raw `#[repr(C)]` FFI bindings (unsafe).
- **`baseRT`** -- safe, idiomatic Rust wrapper with RAII, `Result` error handling, and closures for streaming.

## Prerequisites

Build the single shared engine so that `libbaseRT.dylib` exists:

```bash
cd /path/to/baseRT
make shared
```

By default the crates **dynamically link the one `libbaseRT.dylib`**, which
already carries the Metal kernels (embedded metallib) and its own framework /
C++ dependencies — one redistributable artifact, matching the Python / Node /
Swift bindings. Point the build at the directory containing it with
`BASERT_LIB_DIR` (defaults to `../../../build` relative to the sys crate):

```bash
export BASERT_LIB_DIR=/path/to/baseRT/build
```

The `baseRT` crate embeds an rpath to that directory, so its own examples and
tests run without `DYLD_LIBRARY_PATH`. A **downstream binary** that depends on
`baseRT` must make the dylib discoverable at runtime — either add an rpath in
its own `build.rs` (`cargo:rustc-link-arg=-Wl,-rpath,<dir>`), install the dylib
to a standard location, or set `DYLD_LIBRARY_PATH=$BASERT_LIB_DIR`.

To link the static engine archive instead (fully static, pulls in the
frameworks manually), enable the `static` feature:

```toml
baseRT = { path = "...", features = ["static"] }
```

## Usage

Add the dependency (path-based for now):

```toml
[dependencies]
baseRT = { path = "path/to/baseRT/bindings/rust/baseRT" }
```

Run the bundled example end-to-end:

```bash
BASERT_LIB_DIR=/path/to/baseRT/build \
  cargo run --example smoke -- /path/to/model.base
```

### Text generation

```rust
use baseRT::{Model, SamplingConfig};

fn main() -> baseRT::Result<()> {
    // metallib path = None → use the metallib embedded in libbaseRT.dylib.
    let model = Model::load("models/Qwen3-0.6B-Q4_0.base", None, 0)?;

    println!("Architecture: {}", model.architecture());
    println!("GPU memory:   {:.1} MB", model.memory() as f64 / 1e6);

    let tokens = model.encode("The meaning of life is")?;

    let stats = model.generate(&tokens, 128, SamplingConfig::greedy(), |_id, text| {
        print!("{text}");
        true // return false to stop early
    })?;

    println!("\n\nPrefill: {:.0} tok/s", stats.prefill_tokens_per_sec);
    println!("Decode:  {:.0} tok/s", stats.decode_tokens_per_sec);
    Ok(())
}
```

### Non-streaming generation

```rust
let (text, stats) = model.generate_text(&tokens, 128, SamplingConfig::greedy())?;
println!("{text}");
```

### Multi-turn chat (continue from KV cache)

```rust
let turn1 = model.encode("User: Hi!\nAssistant:")?;
model.generate(&turn1, 64, SamplingConfig::greedy(), |_, t| { print!("{t}"); true })?;

let turn2 = model.encode("\nUser: Tell me more.\nAssistant:")?;
model.generate_continue(&turn2, 64, SamplingConfig::greedy(), |_, t| { print!("{t}"); true })?;
```

### Whisper transcription

```rust
let model = Model::load("models/whisper-base-en.bin", None, 0)?;
assert!(model.is_whisper());

let (text, stats) = model.transcribe("audio.wav", Some("en"))?;
println!("{text}");
println!("Decode: {:.1} ms", stats.decode_ms);
```

### Model inspection

```rust
println!("Tensors: {}", model.tensor_count());
for (name, dtype) in model.tensors() {
    println!("  {name}: dtype={dtype}");
}
```

### Sampling configuration

```rust
// Greedy (default)
let cfg = SamplingConfig::greedy();

// Temperature sampling
let cfg = SamplingConfig::with_temperature(0.7);

// Full control
let cfg = SamplingConfig {
    temperature: 0.8,
    top_k: 50,
    top_p: 0.95,
    min_p: 0.05,
    repeat_penalty: 1.1,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    seed: 0,                 // 0 = wall-clock-seeded; non-zero = deterministic
    logit_bias: Vec::new(),  // e.g. vec![(token_id, +5.0), (banned_id, -100.0)]
};
```

### Low-level API

```rust
let first_token = model.prefill(&tokens);
let next = model.decode_step(first_token, tokens.len() as i32);

// Or batch decode
let batch = model.chain_decode(first_token, tokens.len() as i32, 16)?;
```

## Error handling

All fallible operations return `baseRT::Result<T>`. Errors come from:

- `Error::Api(msg)` -- the C API returned NULL or an error (message from `baseRT_get_error()`).
- `Error::InvalidString(e)` -- a Rust string contained an interior NUL byte.
