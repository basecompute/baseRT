# Python

`ctypes`-based bindings over the BaseRT C API. Python 3.9+.

## Install

```sh
cd bindings/python
pip install -e .
```

Point the bindings at the engine dylib if it isn't in `build/`:

```sh
export BASERT_LIB_PATH=/path/to/build   # dir containing libbaseRT.dylib
```

## Generate text

```python
import baseRT

m = baseRT.Model("models/your-model.base")
print(m.generate_text("The capital of France is", max_tokens=64, temperature=0.7))
```

## Stream tokens

```python
for chunk in m.stream("Explain RoPE in one sentence.", max_tokens=128):
    print(chunk, end="", flush=True)
```

## Lower-level control

```python
tokens = m.encode("Once upon a time")
stats = m.generate(tokens, max_tokens=256, temperature=0.7,
                   top_k=40, top_p=0.9)        # streaming via callback under the hood
print(stats.generated_tokens, stats.prefill_tokens_per_sec)
```

Other `Model` methods: `config()`, `memory_bytes()`, `position()`, `reset()`,
`decode_token()`, `prefill()` / `prefill_image()`, `decode_step()`,
`embed()` / `embed_text()` / `embedding_dim()`,
`format_chat(system, user)` / `chat_template()`, `token_count()`,
`tensors()`, and `transcribe()` for Whisper-class models.

## Embeddings

```python
m = baseRT.Model("models/embedding-model.base")
vec = m.embed_text("hello world")
print(m.embedding_dim(), len(vec))
```

## Transcription

```python
m = baseRT.Model("models/whisper-base.base")
# pass PCM samples / audio per the transcribe() signature
```

See [`bindings/python`](https://github.com/basecompute/baseRT/tree/main/bindings/python)
for the full surface and tests.
