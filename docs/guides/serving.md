# Serving an API

`basert serve` starts an HTTP server for baseRT models. The server is
[Superfluid](https://github.com/basecompute/superfluid), our open-source
serving daemon. It answers OpenAI-, Anthropic- and Ollama-compatible APIs on
one port.

```sh
basert serve basecompute/Qwen3-4B --api-key "$(uuidgen)"
```

The server listens on `http://127.0.0.1:8453`. A hub id is pulled (or
converted) on first use; a `.base` path is served as it is.

You can load several models at once by repeating `--model`; clients pick one
via the `model` field in the request. The first model is the default one.

```sh
basert serve basecompute/Qwen3-4B --model basecompute/Qwen3-0.6B --api-key "$KEY"
```

See what clients can call in the [Server API reference](../reference/server-api.md).

## How `basert serve` runs Superfluid

Since baseRT 0.4.0, basert ships no server of its own. `basert serve` runs a
Superfluid release, which loads the engine library installed beside basert:

1. It looks for Superfluid in `$BASERT_SUPERFLUID_DIR`, then in `superfluid/`
   beside `basert` (where the install script puts it), then in basert's data
   directory (`~/Library/Application Support/baseRT/superfluid` on macOS,
   `~/.local/share/baseRT/superfluid` on Linux). A basert from Homebrew, npm,
   pip or cargo looks in its data directory first and installs only there. It
   runs the first install that can serve this engine, whatever its release.
2. When none is installed, or none can serve this engine, it installs the
   newest Superfluid release that can, from Superfluid's GitHub releases, and
   checks its SHA-256. On a terminal it asks first; otherwise it installs
   without asking.

A Superfluid you installed yourself is not run in its place, and the server
keeps its state in basert's state directory (see [State](#state)), not in
`~/.superfluid`.

| Command | Purpose |
| --- | --- |
| `basert superfluid install` | Install the newest release that serves this engine ahead of time. Nothing to do when it is already there. |
| `basert superfluid install --force` | Install again, even when that release or a newer one is there. |
| `basert superfluid status` | Show the release basert was tested with, what is installed where, and whether it can serve this engine. Exits 0 when an install can serve, 1 when none is installed, 2 when none installed can serve. |
| `basert serve --version` | Print basert's version and the Superfluid a start would run. |
| `basert self update` | Update basert, and bring Superfluid to the newest release that serves the new engine. |

| Environment variable | Effect |
| --- | --- |
| `BASERT_SUPERFLUID_AUTO_INSTALL=0` | Never download Superfluid at a start; the command to run is printed instead. `=1` downloads without asking, even on a terminal. |
| `BASERT_SUPERFLUID_DIR` | A directory of Superfluid's own to run and install it in. One holding other files is never replaced. |
| `BASERT_SUPERFLUID_RELEASES` | Another release listing to choose from (a URL or a file). |
| `BASERT_SUPERFLUID_SOURCE` | A directory holding the tested release's tarball and checksum, read instead of GitHub (for machines without internet access). |
| `BASERT_STATE_DIR` | Where the server keeps its state (see [State](#state)). |
| `BASERT_LIB` | Another directory holding libbaseRT. |

### What a start hands Superfluid

`basert serve` translates its flags (below) into a `superfluid serve` command
line and always adds:

- `--runtime basert`: only baseRT models are served.
- `--offline`: the server fetches nothing. basert resolves and pulls hub ids
  itself before it starts the server.
- A fresh request log at each start (`--sessions-cache`), no KV parking
  (`--no-park`), no terminal monitor (`--no-tui`), and no `x-superfluid-qos`
  header from clients (`--no-http-qos-header`).
- Superfluid's own update check is off; basert updates it.

`--print-command` prints the command line a start would run, and exits.

Any other `--flag` is handed to `superfluid serve` as it is, so Superfluid's
own serve flags work on `basert serve` (for example `--key-policy`,
`--spec-bitexact`, or `--basert-neural-engine off`). Everything after `--` is
handed over untouched. `--runtime` is refused. See Superfluid's
[command-line reference](https://docs.basecompute.co/latest/cli/) for its
flags.

## Choosing a quant build

Pick which quant variant to serve with an inline `:variant` on the id, or the
`--variant` flag (it applies to ids without an inline variant). A variant can
be named as `basert list` shows it (`default-q8`), as a quant (`q8`), or as a
size (`small`) of a release that comes in sizes.

```sh
basert serve basecompute/gemma-4-E2B-it:default-q8
# equivalently:
basert serve basecompute/gemma-4-E2B-it --variant default-q8
```

`basert list` shows installed variants; `basert list --remote` adds the
catalog. An uninstalled variant is pulled on demand; an installed one is used
without contacting Hugging Face.

## Core flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `<model>`, `--model <model>` | — | `.base` file or hub id. `--model` is repeatable. |
| `--model-dir <path>` | — | Register every `*.base` under `<path>`, loaded on first request. Without `--model`, the first one is loaded at startup. |
| `--host <addr>` | `127.0.0.1` | Bind address. |
| `--port <N>` | `8453` | Bind port. |
| `--api-key <key>` | — | Require `Authorization: Bearer <key>` (or `X-Api-Key`). |
| `--max-context <N>` | auto | Context window in tokens; sizes the KV cache up front. |
| `--max-tokens <N>` | rest of the window | Cap on generated tokens for a request that names none. |
| `--verbose`, `-v` | off | Show engine diagnostics (load, GPU, dispatch table). |
| `--basert-lib <dir>` | beside `basert` | Directory holding the libbaseRT to serve with. |
| `--print-command` | — | Print the `superfluid serve` command line and exit. |

> [!TIP]
> **`--max-context` is sized for your machine**
>
> By default the window is what the GPU memory budget leaves after the
> weights and scratch headroom, spread over the configured lanes, capped at
> the model's trained window. Set it explicitly to keep memory for other work.
> Requests past it are rejected with `context_length_exceeded`. With
> `--model-dir` and no `--max-context`, Superfluid uses a fixed 8192-token
> window so models loaded later find room.

## Throughput: batching & caching

Paged KV and prefix reuse are always on: a request that repeats an earlier
prompt prefix is served from the cache. The server runs one request at a time
unless you ask for more.

| Flag | Purpose |
| --- | --- |
| `--continuous-batching [N]` | Serve up to `N` requests at once through one shared forward pass (default 8). Set `N` to at least your peak concurrent requests, or the surplus queues. |
| `--max-batch-size <N>` | Max concurrent sequences (default 1). Alias `--max-batch`. The larger of this and `--continuous-batching` wins. |
| `--prefill-budget <N>` | Max prompt tokens ingested per engine tick (64–4096). Lower it to trade first-token latency for smoother concurrent decode. |
| `--kv-bits 4\|8\|16\|84` | KV cache element width; `84` is mixed K=Q8/V=Q4 (default: per-model auto). |

A typical high-throughput configuration:

```sh
basert serve basecompute/Qwen3-4B \
  --continuous-batching 8 --max-context 8192 --api-key "$KEY"
```

## Speculative decoding

Since baseRT 0.5.1, `basert serve` decodes speculatively by default, as with
`--speculate auto`: it drafts with the drafter the catalog pairs with the
model, else with the MTP head the model's bundle carries, else decodes
plainly.

| Flag | Purpose |
| --- | --- |
| `--speculate auto` | The default. The paired drafter is pulled on first use: on a Mac, 1.4 GB for Qwen3.8-27B and 458 MB for Qwen3.6-35B-A3B (their DFlash2 drafters). |
| `--speculate off` | Decode without drafting. |
| `--speculate mtp-head` \| `prompt-lookup` \| `dflash:<path or hub id>` | Use this strategy. A drafter hub id is pulled like a model. |
| `--offline` | Pull nothing; serve the files already installed. |
| `--spec-bitexact` | Make speculative greedy output match plain decoding (slower). |

See Superfluid's
[speculative decoding](https://docs.basecompute.co/latest/speculative-decoding/)
page for how strategies are chosen and gated.

## Operability

| Flag | Purpose |
| --- | --- |
| `--rate-limit <N>` | Requests per minute per client IP (0 = unlimited). |
| `--idle-timeout <N>` | Unload a model unused for `N` seconds (0 = disabled). The default model stays loaded. |
| `--request-timeout <ms>` | Bounds fill-in-the-middle completions only; chat completions run to their token limit. |
| `--drain-timeout <N>` | Seconds to wait for in-flight requests on `SIGUSR1` (default 60). |
| `--log-file <path>` | Redirect stderr (access log + diagnostics) to this file. |
| `--sessions <path>` | Request-log directory (default: per port under the state directory). |
| `--files-max-bytes <N>` | Reject `/v1/files` uploads that would push total stored bytes above `N`. |
| `--files-expiry <N>` / `--files-sweep <N>` | Remove files older than `N` seconds; sweep cadence (default 300). |

### Rolling restarts

`SIGUSR1` stops accepting connections and drains in-flight requests (up to
`--drain-timeout`); pair it with an external supervisor for zero-downtime
restarts. `SIGINT`/`SIGTERM` exit immediately.

### Logs

`--log-file` writes the access log + diagnostics to a file. Use external
rotation — `logrotate copytruncate` (Linux) or `newsyslog` with the `F` flag
(macOS).

### State

The request log and uploaded files are kept under `$BASERT_STATE_DIR`, by
default the `serve/` directory beside the model cache, one directory per port.
The request log starts fresh at each start.

### Flags no longer used

These are accepted so existing command lines keep working, and do nothing:

- `--paged-kv`, `--prefix-cache`: paged KV and prefix reuse are always on.
- `--prefix-cache-file`, `--prefix-cache-save-interval`: the prefix cache is
  not kept on disk.
- `--files-dir`: `/v1/files` and `/v1/batches` are always on; files are kept
  in the server's state directory.
- `--media-dir`: images and audio arrive as `data:` URIs or uploads.
- `--metallib`, `--prefill-chunk`, `--paged-weights`,
  `--no-paged-weights-retry`, `--gpu-wait-timeout-ms`, `--decode-replay`,
  `--no-baked-decode`: engine settings this server does not expose.

> [!CAUTION]
> **Before exposing beyond localhost**
>
> Always set `--api-key`, bind carefully with `--host`, and read
> [Security](../reference/security.md). The server speaks no TLS and is
> intended for trusted environments.

## Quick test

```sh
curl http://127.0.0.1:8453/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"basecompute/Qwen3-4B","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```
