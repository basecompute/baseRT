# Rust

Two crates:

- **`baseRT-sys`** — raw `#[repr(C)]` FFI bindings (unsafe).
- **`baseRT`** — a safe, idiomatic wrapper with RAII, `Result` error handling,
  and closures for streaming.

## Setup

Make `libbaseRT.dylib` available (from an [engine release](../reference/engine-releases.md)
or a local build) and point the crates at it:

```sh
export BASERT_LIB_DIR=/path/to/build   # dir containing libbaseRT.dylib
```

The `baseRT` crate embeds an rpath to that directory, so its own examples and
tests run without `DYLD_LIBRARY_PATH`. A downstream binary that depends on
`baseRT` must make the dylib discoverable at runtime — add an rpath in its
`build.rs` (`cargo:rustc-link-arg=-Wl,-rpath,<dir>`), install the dylib to a
standard location, or set `DYLD_LIBRARY_PATH=$BASERT_LIB_DIR`.

Add it to your `Cargo.toml`:

```toml
baseRT = { path = "path/to/baseRT/bindings/rust/baseRT" }
```

To link the static engine archive instead (pulls in the frameworks manually),
enable the `static` feature:

```toml
baseRT = { path = "...", features = ["static"] }
```

## Usage

```rust
use baseRT::Model;

let model = Model::open("models/your-model.base")?;
let tokens = model.encode("Once upon a time")?;
let stats = model.generate(&tokens, 256, Default::default(), |_id, text| {
    print!("{text}");
    true // keep going
})?;
println!("\n{} tokens", stats.generated_tokens);
```

See [`bindings/rust`](https://github.com/prabod/baseRT/tree/main/bindings/rust)
for the full API, the `static` feature, and examples.
