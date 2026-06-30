# BaseRT Python Bindings

Python ctypes bindings for the [BaseRT](../../) LLM inference engine, optimised for Apple Silicon via Metal.

## Building the shared library

BaseRT's Makefile produces a static library (`build/libbaseRT_engine.a`). For Python ctypes you need a shared library. Build it with:

```bash
# From the project root, after running `make`:
clang++ -dynamiclib -o build/libbaseRT.dylib \
    -Wl,-all_load build/libbaseRT_engine.a \
    -framework Metal -framework Foundation -framework MetalPerformanceShaders \
    -lc++
```

## Installation

```bash
cd bindings/python
pip install -e .
```

## Finding the library

The bindings look for `libbaseRT.dylib` in this order:

1. `BASERT_LIB_PATH` environment variable (full path to the `.dylib`).
2. `../../build/libbaseRT.dylib` relative to the package source.
3. `build/libbaseRT.dylib` in the current working directory.

## Usage

### Text generation

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    # Simple text generation
    text = model.generate_text("Once upon a time", max_tokens=128)
    print(text)
```

### Streaming generation

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    for token_text in model.stream("The capital of France is", max_tokens=64):
        print(token_text, end="", flush=True)
    print()
```

### Sampling parameters

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    text = model.generate_text(
        "Write a haiku about programming:",
        max_tokens=64,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        min_p=0.05,
        repeat_penalty=1.1,
    )
    print(text)
```

### Generation with callback

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    def on_token(token_id: int, text: str) -> bool:
        print(text, end="", flush=True)
        return True  # return False to stop early

    stats = model.generate("Hello, world!", max_tokens=128, callback=on_token)
    print(f"\n\nPrefill: {stats.prefill_tokens_per_sec:.0f} tok/s")
    print(f"Decode:  {stats.decode_tokens_per_sec:.0f} tok/s")
```

### Multi-turn chat

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    # First turn
    stats = model.generate("User: Hi!\nAssistant:", max_tokens=64, callback=print_cb)

    # Continue from existing KV cache (no re-prefill of previous turns)
    stats = model.generate_continue(
        "\nUser: Tell me a joke.\nAssistant:",
        max_tokens=128,
        callback=print_cb,
    )
```

### Model inspection

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    cfg = model.config
    print(f"Architecture: {cfg.architecture}")
    print(f"Parameters:   {cfg.dim}d / {cfg.n_layers}L / {cfg.n_heads}H")
    print(f"Vocab size:   {cfg.vocab_size}")
    print(f"Memory:       {model.memory_bytes / 1e6:.0f} MB")

    # List all tensors
    for t in model.tensors():
        print(f"  [{t.index}] {t.name}  dtype={t.dtype}")
```

### Tokenization

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    tokens = model.encode("Hello, world!")
    print(f"Token IDs: {tokens}")

    for tid in tokens:
        print(f"  {tid} -> {model.decode_token(tid)!r}")
```

### Whisper transcription

```python
import baseRT

with baseRT.Model("models/whisper-base-en.bin") as model:
    assert model.is_whisper

    # Transcribe from a WAV file
    text, stats = model.transcribe("audio.wav", language="en")
    print(text)
    print(f"Took {stats.total_ms:.0f} ms")

    # Disable timestamps for faster plain-text output
    model.set_timestamps(False)
    text, stats = model.transcribe("audio.wav")
    print(text)
```

### Low-level API

```python
import baseRT

with baseRT.Model("models/Qwen3-0.6B-Q4_0.base") as model:
    tokens = model.encode("Hello")
    first_token = model.prefill(tokens)
    print(f"First token: {model.decode_token(first_token)}")

    next_token = model.decode_step(first_token, len(tokens))
    print(f"Next token:  {model.decode_token(next_token)}")

    model.reset()
```
