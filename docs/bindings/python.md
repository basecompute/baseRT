# Python

`ctypes`-based bindings over the BaseRT C API. Python 3.9+.

## Install

```sh
pip install basert
```

The [`basert`](https://pypi.org/project/basert/) package installs the
`basert` command, the prebuilt engine and the `baseRT` Python module in one
step. Wheels are published for:

- Apple Silicon, macOS 15 or later;
- Linux arm64 with an NVIDIA GPU and the CUDA 13 runtime (cuBLAS);
- Linux x86-64 with an AMD Strix Halo (gfx1151) or Strix Point (gfx1150) GPU
  and the ROCm 10 runtime in `/opt/rocm/core-10.0`.

From a source checkout, install the module alone and give it an engine:

```sh
cd bindings/python
pip install -e .
export BASERT_LIB_PATH=/path/to/libbaseRT.dylib   # or libbaseRT.so on Linux
```

Without `BASERT_LIB_PATH`, the module looks for the engine bundled in the
wheel, then the checkout's `build/`, then `~/.basert` (the install script) and
Homebrew. See [Finding the engine](index.md#finding-the-engine).

## Generate text

```python
import baseRT

m = baseRT.Model("models/your-model.base")
print(m.generate_text("The capital of France is", max_tokens=64, temperature=0.7))
```

`Model(path, kernel_library_path=None, max_context=0)` loads a bundle;
`max_context=0` takes the model's default window. `m.close()` frees the GPU
resources, and `Model` also works as a context manager (`with
baseRT.Model(...) as m:`).

## Stream tokens

```python
for chunk in m.stream("Explain RoPE in one sentence.", max_tokens=128):
    print(chunk, end="", flush=True)
```

## Lower-level control

```python
def on_token(token_id: int, text: str) -> bool:
    print(text, end="", flush=True)
    return True                                 # False stops generation

tokens = m.encode("Once upon a time")
stats = m.generate(tokens, max_tokens=256, temperature=0.7,
                   top_k=40, top_p=0.9, callback=on_token)
print(stats.generated_tokens, stats.decode_tokens_per_sec)
```

`generate_continue()` takes the same arguments and continues from the current
KV cache, so a follow-up turn doesn't reprocess the history.

Other `Model` methods: `reset()`, `decode_token()`, `prefill()`,
`decode_step()`, `chain_decode()`, `prefill_image()` / `image_num_tokens()`,
`prefill_audio()` / `audio_num_tokens()`, `embed()` / `embed_text()`,
`format_chat(system, user)`, `token_count()`, `set_speculation()`,
`tensors()`, and the transcription methods below. Properties: `config`,
`memory_bytes`, `position`, `is_whisper`, `embedding_dim`, `chat_template`.
`baseRT.get_error()` returns the engine's last error message.

## Embeddings

```python
m = baseRT.Model("models/embedding-model.base")
vec = m.embed_text("hello world")
print(m.embedding_dim, len(vec))
```

## Transcription

```python
m = baseRT.Model("models/whisper-base.base")
text, stats = m.transcribe("audio.wav", language="en")
print(text)

for seg in m.transcribe_segments():          # per-segment detail of the last run
    print(seg.start_ms, seg.end_ms, seg.text)
```

`transcribe_pcm()` takes 16 kHz mono float samples instead of a file, and
`transcribe_stream()` / `transcribe_pcm_stream()` call back per segment.
`set_task("translate")`, `set_initial_prompt()`,
`set_condition_on_previous_text()` and `set_timestamps()` configure the next
run; `transcribe_language()` reports the detected language.

See [`bindings/python`](https://github.com/basecompute/baseRT/tree/main/bindings/python)
for the full surface and tests.
