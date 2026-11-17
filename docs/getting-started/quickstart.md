# Quickstart

This walks you from zero to a running model. It assumes you've completed
[Installation](installation.md) and have `basert` on your `PATH`.

## Pull a model

`basert pull` downloads a pre-converted `.base` model from the BaseRT catalog:

```sh
basert pull basecompute/Qwen3-4B
```

A raw Hugging Face repo works too. Its weights are downloaded and converted to
`.base` on your machine:

```sh
basert pull Qwen/Qwen3-4B
```

Models are stored in `~/Library/Caches/baseRT/models` on macOS and
`~/.cache/baseRT/models` on Linux (override with `$BASERT_MODELS_DIR`). List
what you have:

```sh
basert list                 # installed models
basert list --remote        # also show catalog models you haven't pulled
```

You can also start with a smaller model:

```sh
basert pull basecompute/Qwen3-0.6B
```

`basert chat`, `complete`, `bench` and `serve` pull a model id on first use,
so `pull` is optional.

> [!TIP]
> **Gated or private repos**
>
> Set `HF_TOKEN` (or log in with the Hugging Face CLI) before pulling gated
> models. BaseRT uses the standard token chain.

### Variants and sizes

A model can come in several builds. Add `:<variant>` to the id to choose one,
named as `basert list` shows it:

```sh
basert pull basecompute/Qwen3-4B:default-q8
basert chat basecompute/Qwen3-4B:default-q8
basert chat basecompute/Qwen3-4B --variant default-q8   # same, as a flag
```

A variant can also be a quant (`:q8`). A release that comes in sizes rather
than quants (files named `-XS`, `-Small`, `-Medium`, `-Large`, `-XL`) takes
the Medium by default, and a size names a variant:

```sh
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1:small
```

## Chat

```sh
basert chat basecompute/Qwen3-4B
```

You're dropped into an interactive prompt. Type `/clear` to reset the
conversation, `quit` to exit. Common flags:

```sh
basert chat basecompute/Qwen3-4B --temp 0.7 --top-p 0.9 --max-tokens 512 --max-context 8192
```

`--max-context` is sized for your machine when you leave it out. See
[Chat & completion](../guides/chat-complete.md) for all flags (sampling,
thinking mode, KV-cache width, paged-KV).

## One-shot completion

For scripting, `basert complete` runs a single prompt and prints the result:

```sh
basert complete basecompute/Qwen3-4B --chat --prompt "Write a haiku about GPUs." --max-tokens 64
```

`--chat` wraps the prompt in the model's chat template, which instruct models
need.

## Serve an OpenAI-compatible API

`basert serve` runs [Superfluid](https://github.com/basecompute/superfluid),
our open-source server, with the BaseRT engine. The first start installs
Superfluid if it isn't there yet.

```sh
export API_KEY="$(uuidgen)"
basert serve basecompute/Qwen3-4B --api-key "$API_KEY"
```

The server listens on `127.0.0.1:8453` by default (`--host`, `--port` to
change it). Call it like any OpenAI endpoint:

```sh
curl http://127.0.0.1:8453/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{
        "model": "basecompute/Qwen3-4B",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
```

`GET /v1/models` lists the served model names. Stream tokens with
`"stream": true`.

`basert serve` uses speculative decoding by default: the drafter the catalog
pairs with the model, or the head its file carries, or none. A first start can
download that drafter. `--speculate off` serves without it, and `--offline`
downloads nothing.

See [Serving an API](../guides/serving.md) and the
[Server API reference](../reference/server-api.md) for endpoints, batching,
and deployment, and [Superfluid's docs](https://docs.basecompute.co/latest/)
for everything the server does.

## From your language

```python
import baseRT

with baseRT.Model("/path/to/model.base") as m:
    for chunk in m.stream("Explain RoPE in one sentence.", max_tokens=128):
        print(chunk, end="", flush=True)
```

`basert list --json` prints the path of each installed model. See
[Bindings](../bindings/index.md) for Python, Node, Rust, and Swift.

## Convert a local checkpoint

Already have a GGUF / Hugging Face / MLX checkpoint? Convert it directly:

```sh
basert convert ./path/to/checkpoint --target base-q4 --output models/my-model.base
```

More in [Converting models](../guides/converting.md).

## Stay up to date

```sh
basert self update
```

See [Updating](installation.md#updating).
