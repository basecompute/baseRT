# Quickstart

This walks you from zero to a running model. It assumes you've completed
[Installation](installation.md) and have `basert` and the engine tools on your
`PATH`.

## Pull a model

`basert pull` downloads a model from HuggingFace and converts it to `.base` on
the fly (or fetches a pre-converted model from the catalog):

```sh
basert pull Qwen/Qwen3-4B
```

Models are stored in `~/.cache/baseRT/models` (override with
`$BASERT_MODELS_DIR`). List what you have:

```sh
basert list                 # installed models
basert list --remote        # also show catalog models you haven't pulled
```

You can also pull a smaller model to start:

```sh
basert pull Qwen/Qwen3-0.6B
```

!!! tip "Gated or private repos"
    Set `HF_TOKEN` (or log in with the HuggingFace CLI) before pulling gated
    models. BaseRT uses the standard token chain.

## Chat

```sh
basert chat Qwen/Qwen3-4B
```

You're dropped into an interactive prompt. Type `/clear` to reset the
conversation, `quit` to exit. Common flags:

```sh
basert chat Qwen/Qwen3-4B --temp 0.7 --top-p 0.9 --max-tokens 512 --max-context 8192
```

See [Chat & completion](../guides/chat-complete.md) for all flags (sampling,
thinking mode, KV-cache width, paged-KV).

## One-shot completion

For scripting, `basert complete` runs a single prompt and prints the result:

```sh
basert complete Qwen/Qwen3-4B --prompt "Write a haiku about Metal GPUs." --max-tokens 64
```

## Serve an OpenAI-compatible API

```sh
basert serve --model Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080
```

Call it like any OpenAI endpoint:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3-4B",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
```

Stream tokens with `"stream": true`. See [Serving an API](../guides/serving.md)
and the [Server API reference](../reference/server-api.md) for endpoints,
batching, and deployment.

## From your language

```python
import baseRT

m = baseRT.Model("~/.cache/baseRT/models/Qwen/Qwen3-4B/default-q4/model.base")
for chunk in m.stream("Explain RoPE in one sentence.", max_tokens=128):
    print(chunk, end="", flush=True)
```

See [Bindings](../bindings/index.md) for Python, Node, Rust, and Swift.

## Convert a local checkpoint

Already have a GGUF / HF / MLX checkpoint? Convert it directly:

```sh
basert convert ./path/to/checkpoint \
    --profile base-convert/profiles/default-q4.json \
    --output models/my-model.base
```

More in [Converting models](../guides/converting.md).
