# cc-bridge — native Anthropic Messages adapter for baseRT

A thin protocol bridge that lets Claude Code talk to a local baseRT inference
server directly, with no LiteLLM or other intermediate layer.

```
Claude Code ──(/v1/messages)──▶ anthropic_adapter.py:4003 ──(/v1/chat/completions, stream)──▶ baseRT:7999
```

## Why it exists

- Claude Code speaks only the Anthropic Messages API (`/v1/messages`);
  `basert serve` exposes OpenAI-compatible endpoints (`/v1/chat/completions`).
- A LiteLLM bridge was tried first and failed in two measured ways (see
  `docs/proposals/native-anthropic-adapter.md`):
  1. **Non-streaming translation bug**: LiteLLM's openai→anthropic non-streaming
     conversion always returns empty `content` (upstream issue #27492, still
     open in 1.89.2 and 1.95.0).
  2. **Long prefill drops the connection**: Claude Code's requests carry a
     ~23.5k-token system prompt; baseRT needs ~35s to prefill it, and the
     stainless HTTP client gives up after ~36s of silence and retries the same
     prompt **non-streaming** — straight into bug 1.
- This adapter generates the Anthropic wire format natively instead, modeled on
  llama.cpp server's Anthropic implementation (MIT).

## Setup

```bash
cd cc-bridge
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
# prerequisite: baseRT is running
.venv/bin/python anthropic_adapter.py --master-key <key> --model <model-id>
```

`--master-key` and `--model` are required; everything else has a sensible
default:

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | bind address; use `0.0.0.0` only for cross-machine deployments, together with a strong key |
| `--port` | 4003 | listen port |
| `--upstream` | `http://127.0.0.1:7999/v1` | baseRT chat/completions endpoint |
| `--model` | — (**required**) | backend model id sent to baseRT |
| `--master-key` | — (**required**) | shared secret; every inbound request must present it via `x-api-key` or `Authorization: Bearer`, otherwise it is rejected with 401 |
| `--max-tokens-override` | 0 | max_tokens override, 0 = pass the client value through |
| `--debug-dir` | *(empty)* | request/response dump directory, empty = no dump; enabling it writes full prompts to disk |
| `--no-system-merge` | off | skip system-block merging (froggeric v21.3 baked into the model) |
| `--allow-classifier-bypass` | off | short-circuit harm-classifier probes; security implications in TECHNICAL.md §3.7 |
| `--log-level` | `info` | debug / info / warning / error |

Example with different port/upstream/model:

```bash
.venv/bin/python anthropic_adapter.py \
  --port 4004 --upstream http://other:8080/v1 --model Qwen3-0.6B \
  --master-key 'change-me'
```

Run `python3 anthropic_adapter.py --help` for the full flag reference.

Design details, protocol rules and known limitations: [TECHNICAL.md](TECHNICAL.md).

## Claude Code client configuration

```bash
export ANTHROPIC_BASE_URL="http://<host-ip>:4003"
export ANTHROPIC_AUTH_TOKEN="<your --master-key value>"
export ANTHROPIC_MODEL="local"
export ANTHROPIC_DEFAULT_SONNET_MODEL="local"
export ANTHROPIC_DEFAULT_OPUS_MODEL="local"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="local"
```

The model name is an alias the adapter echoes back as `local`; it does not need
to match any known model family.

## Verify

```bash
# non-streaming smoke test
curl -s -m 120 http://localhost:4003/v1/messages \
  -H "content-type: application/json" -H "x-api-key: <your-key>" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"local","max_tokens":64,"messages":[{"role":"user","content":"1+1=? Answer with a number only"}]}'
```

## Known limitations

- The model always thinks (cannot be disabled), and thinking tokens count
  toward the output budget; give a generous `--max-tokens-override` for heavy
  thinking models.
- Thinking blocks carry an empty signature (local models, no cryptographic
  verification; same as llama.cpp).
- Claude Code's system reminders (mid-array `role:"system"` messages) are
  merged into a single leading system message, because baseRT's Qwen template
  requires system to come first. With a chat template that handles mid-list
  system messages (e.g. froggeric v21.3), pass `--no-system-merge`.
- Text-only: multimodal and audio inputs are not supported.
