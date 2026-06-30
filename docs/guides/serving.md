# Serving an API

`basert serve` starts an OpenAI-compatible HTTP server backed by the engine.

```sh
basert serve --model Qwen/Qwen3-4B --api-key "$(uuidgen)" --port 8080
```

You can load several models at once by repeating `--model`; clients pick one via
the `model` field in the request.

```sh
basert serve --model Qwen/Qwen3-4B --model Qwen/Qwen3-0.6B --api-key "$KEY"
```

See the full list of endpoints in the [Server API reference](../reference/server-api.md).

## Choosing a quant build

Pick which quant variant to serve with an inline `:variant` on the id, or the
`--variant` flag. Dense models ship `default-q4` and `default-q8`; MoE models are
q4-only.

```sh
basert serve basecompute/gemma-4-E2B-it:default-q8 --max-context 16000
# equivalently:
basert serve basecompute/gemma-4-E2B-it --variant default-q8 --max-context 16000
```

`basert list` shows installed variants; an uninstalled one is pulled on demand.

## Core flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model <path>` | — | Model file or hub id. Repeatable. |
| `--host <addr>` | `127.0.0.1` | Bind address. |
| `--port <N>` | `8080` | Bind port. |
| `--api-key <key>` | — | Require `Authorization: Bearer <key>`. |
| `--max-context <N>` | `4096` | Context window; sizes the KV cache up front. |
| `--max-tokens <N>` | `2048` | Default max generation tokens. |
| `--metallib <path>` | auto | Path to `baseRT.metallib` (auto-detected next to the binary). |

> [!TIP]
> **Set `--max-context` for long-context models**
>
> The KV cache is allocated up front from `--max-context`. Models trained for
> long context (Gemma 4 / Qwen3 MoE = 32k+) need this set explicitly; requests
> exceeding it are rejected with `context_length_exceeded`.

## Throughput: batching & caching

| Flag | Purpose |
| --- | --- |
| `--kv-bits 4\|8\|16` | KV cache element width (default: per-model auto). |
| `--paged-kv` | Paged KV cache + block-table dispatch. |
| `--max-batch-size <N>` | Max concurrent sequences for batched decode (default 1). |
| `--continuous-batching [N]` | Decode concurrent requests through one shared forward pass (implies `--paged-kv`; `N` = max in-flight lanes, default 8). |
| `--prefix-cache` | Share KV of common prompt prefixes across requests (implies `--paged-kv`; most effective with `--continuous-batching`). |
| `--prefix-cache-file <path>` | Persist the prefix cache per model (`<path>.<model_id>`), loaded on startup and saved on shutdown. |
| `--prefix-cache-save-interval <sec>` | Re-save the prefix cache periodically. |

A typical high-throughput configuration:

```sh
basert serve --model Qwen/Qwen3-4B \
  --continuous-batching 8 --prefix-cache \
  --max-context 8192 --api-key "$KEY"
```

## Operability

| Flag | Purpose |
| --- | --- |
| `--rate-limit <N>` | Requests per minute per client (0 = unlimited). |
| `--idle-timeout <N>` | Auto-unload idle models after N seconds (0 = disabled). |
| `--request-timeout <ms>` | Abort generation longer than N ms (0 = disabled). Recommended 60000–300000 for unattended agents. |
| `--drain-timeout <N>` | Seconds to wait for in-flight requests on `SIGUSR1` (default 60). |
| `--log-file <path>` | Redirect stderr (access log + diagnostics). |
| `--files-dir <path>` | Enable `/v1/files` (+ `/v1/batches`) rooted at this directory. |
| `--files-max-bytes <N>` | Reject uploads that would exceed total stored bytes. |
| `--files-expiry <N>` / `--files-sweep <N>` | Auto-remove old files; sweep cadence. |

### Rolling restarts

`SIGUSR1` stops accepting connections and drains in-flight requests (up to
`--drain-timeout`); pair it with an external supervisor for zero-downtime
restarts. `SIGINT`/`SIGTERM` exit immediately.

### Logs

`--log-file` writes the access log + diagnostics to a file. Use external
rotation — `logrotate copytruncate` (Linux) or `newsyslog` with the `F` flag
(macOS).

> [!CAUTION]
> **Before exposing beyond localhost**
>
> Always set `--api-key`, bind carefully with `--host`, and read
> [Security](../reference/security.md). The server is intended for trusted
> environments.

## Quick test

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-4B","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```
