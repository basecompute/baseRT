# Server API

`basert serve` exposes an OpenAI-compatible HTTP API. Point any OpenAI client at
the base URL and set the API key to your `--api-key` value. See
[Serving an API](../guides/serving.md) for launch flags.

## Authentication

If `--api-key` is set, every request must send:

```
Authorization: Bearer <api-key>
```

## Endpoints

| Method · Path | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat completions (streaming via `"stream": true`). Tool/function calling supported. |
| `POST /v1/completions` | Text completions. |
| `POST /v1/embeddings` | Embedding vectors. |
| `POST /v1/rerank` | Rerank documents against a query. |
| `POST /v1/audio/transcriptions` | Whisper-class transcription. |
| `POST /v1/tokenize` | Tokenize text (count/inspect tokens). |
| `GET  /v1/models` | List loaded models. |
| `POST /v1/models/load` · `POST /v1/models/unload` | Load/unload a model at runtime. |
| `POST /v1/lora/load` · `POST /v1/lora/unload` | Manage LoRA adapters. |
| `POST /v1/files` · `GET /v1/files/...` | File storage (requires `--files-dir`). |
| `POST /v1/batches` · `GET /v1/batches/...` | Batch jobs (requires `--files-dir`). |
| `GET  /health` | Liveness probe. |
| `GET  /metrics` · `GET /v1/metrics` | Server metrics. |

## Chat completions

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3-4B",
        "messages": [
          {"role": "system", "content": "You are concise."},
          {"role": "user", "content": "What is RoPE?"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
      }'
```

### Streaming

Set `"stream": true` to receive Server-Sent Events (`data: {…}` chunks ending
with `data: [DONE]`):

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-4B","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

### Tool calling

Pass `tools` (OpenAI function-calling schema); the model emits `tool_calls` in
the response (streamed incrementally when `stream` is set).

## Embeddings

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"my-embed-model","input":["hello world","second doc"]}'
```

## Using the OpenAI SDKs

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="$API_KEY")
resp = client.chat.completions.create(
    model="Qwen3-4B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

!!! note
    The exact request/response fields follow the OpenAI schema. Endpoints like
    `/v1/files`, `/v1/batches`, and LoRA management depend on server flags
    (`--files-dir`) and the loaded model's capabilities.
