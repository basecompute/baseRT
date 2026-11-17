# Server API

`basert serve` runs [Superfluid](https://github.com/basecompute/superfluid),
which serves OpenAI-, Anthropic- and Ollama-compatible APIs on one listener,
`http://127.0.0.1:8453` by default. This page covers what is specific to
`basert serve`; the full request and response reference is in
[Superfluid's docs](https://docs.basecompute.co/latest/openai-server/). See
[Serving an API](../guides/serving.md) for launch flags.

## Authentication

If `--api-key` is set, every request except `GET /health` must send one of:

```
Authorization: Bearer <api-key>
X-Api-Key: <api-key>
```

This includes `/metrics` and the Ollama routes. A missing or wrong key is a
401 with code `invalid_api_key`. See [Security](security.md).

## Model ids

A model served by hub id is named by that id (`basecompute/Qwen3-4B`); a
`.base` file by its file name without the extension. `GET /v1/models` lists
the ids, with each model's context window and capabilities.

## Endpoints

| API | Base URL | Main routes | Reference |
| --- | --- | --- | --- |
| OpenAI | `http://127.0.0.1:8453/v1` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/audio/transcriptions`, `/v1/models`, `/v1/tokenize`, `/v1/files`, `/v1/batches` | [OpenAI-compatible server](https://docs.basecompute.co/latest/openai-server/), [Responses API](https://docs.basecompute.co/latest/responses-api/) |
| Anthropic | `http://127.0.0.1:8453` | `/v1/messages` | [Anthropic Messages API](https://docs.basecompute.co/latest/anthropic-api/) |
| Ollama | `http://127.0.0.1:8453` | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags` | [Ollama-compatible API](https://docs.basecompute.co/latest/ollama-api/) |
| Operations | `http://127.0.0.1:8453` | `/health`, `/props`, `/slots`, `/metrics` | [Observability](https://docs.basecompute.co/latest/observability/) |

`POST /v1/models/load`, `/v1/models/unload`, `/v1/lora/load` and
`/v1/lora/unload` change the server for every client. With authentication on,
they need the global `--api-key` (or an admin key under `--key-policy`). LoRA adapters are adapter
`.base` files made with `basert convert-lora`.

## Chat completions

```sh
curl http://127.0.0.1:8453/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{
        "model": "basecompute/Qwen3-4B",
        "messages": [
          {"role": "system", "content": "You are concise."},
          {"role": "user", "content": "What is RoPE?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
      }'
```

A request that omits a sampling parameter gets the model's default. A request
that names no `max_tokens` generates up to `--max-tokens`, or to the end of
the context window.

### Streaming

Set `"stream": true` to receive Server-Sent Events (`data: {…}` chunks ending
with `data: [DONE]`):

```sh
curl -N http://127.0.0.1:8453/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"basecompute/Qwen3-4B","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

### Tool calling

Pass `tools` (OpenAI function-calling schema); the model emits `tool_calls` in
the response, streamed as partial deltas when `stream` is set. See
[Tool calling](https://docs.basecompute.co/latest/tool-calling/).

## Using the SDKs

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8453/v1", api_key="$API_KEY")
resp = client.chat.completions.create(
    model="basecompute/Qwen3-4B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8453", api_key="$API_KEY")
msg = client.messages.create(
    model="basecompute/Qwen3-4B",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(msg.content)
```

## Under `basert serve`

- Embeddings, rerank and transcription need a model that serves them; other
  models answer 400 `unsupported_model`.
- `/v1/files` and `/v1/batches` are always on. Files are kept in the server's
  state directory; cap them with `--files-max-bytes` and `--files-expiry`.
- The `x-superfluid-qos` request header is ignored, so clients cannot pick
  their own scheduling class.
- `--request-timeout` bounds fill-in-the-middle completions only. Chat
  completions run to their token limit.
