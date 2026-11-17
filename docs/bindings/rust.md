# Rust

Two crates, both on crates.io:

- **`baseRT-sys`** — raw `#[repr(C)]` FFI bindings (unsafe).
- **`baseRT`** — a safe, idiomatic wrapper with RAII, `Result` error handling,
  and closures for streaming.

The crates link an engine you have installed; they don't carry one.

## Setup

Install the engine first, with Homebrew or the install script:

```sh
brew install basecompute/tap/basert
# or
curl -LsSf https://basecompute.co/install.sh | sh
```

Then add the crate:

```toml
[dependencies]
baseRT = "0.5"
```

The build looks for the engine library (`libbaseRT.dylib` on macOS,
`libbaseRT.so` on Linux) in `BASERT_LIB_DIR` when it is set, and otherwise in
the first of: a source checkout's `build/`, `$BASERT_INSTALL_DIR` (default
`~/.basert`), and Homebrew's `opt/basert/libexec`. To use an
[engine release](../reference/engine-releases.md) unpacked elsewhere:

```sh
export BASERT_LIB_DIR=/path/to/engine   # dir containing libbaseRT.dylib / .so
```

To build against a checkout instead of crates.io:

```toml
baseRT = { path = "path/to/baseRT/bindings/rust/baseRT" }
```

The `baseRT` crate embeds an rpath to the engine directory, so its own
examples and tests run without `DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH`. A
downstream binary that depends on `baseRT` must make the library discoverable
at runtime — add an rpath in its `build.rs`
(`cargo:rustc-link-arg=-Wl,-rpath,<dir>`), install the library to a standard
location, or set `DYLD_LIBRARY_PATH` (`LD_LIBRARY_PATH` on Linux).

A crate that loads `libbaseRT` itself at run time can depend on `baseRT-sys`
with the `no-link` feature: nothing is linked, the types stay usable, and the
`extern` functions must not be called. Either way, `baseRT-sys` passes the
engine's C headers to dependents' build scripts as `DEP_BASERT_INCLUDE`.

## Usage

```rust
use baseRT::{Model, SamplingConfig};

fn main() -> baseRT::Result<()> {
    // kernel library None → auto-detect; max_context 0 → the model default.
    let model = Model::load("models/your-model.base", None, 0)?;
    let tokens = model.encode("Once upon a time")?;

    let stats = model.generate(&tokens, 256, SamplingConfig::greedy(), |_id, text| {
        print!("{text}");
        true // return false to stop early
    })?;
    println!("\n{} tokens, {:.0} tok/s", stats.generated_tokens, stats.decode_tokens_per_sec);
    Ok(())
}
```

`SamplingConfig::with_temperature(0.7)` samples instead of decoding greedily;
the struct also has `top_k`, `top_p`, `min_p`, `repeat_penalty`,
`presence_penalty`, `frequency_penalty`, `seed` and `logit_bias`.
`generate_continue` continues from the KV cache for a follow-up turn, and
`generate_text` returns the whole text with the stats.

`Model` also has `config()`, `memory()`, `architecture()`, `prefill()`,
`decode_step()`, `chain_decode()`, `reset()`, `embed()` / `embed_text()`,
`format_chat()`, `chat_template()`, `token_count()`, `tensors()`, and for
Whisper models `transcribe()`, `transcribe_pcm()`, the `_stream` variants and
`transcribe_segments()`. The model is freed when it is dropped.

See [`bindings/rust`](https://github.com/basecompute/baseRT/tree/main/bindings/rust)
for the full API and examples.
