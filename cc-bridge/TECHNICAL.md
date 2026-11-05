# cc-bridge — technical notes

Native Anthropic Messages API adapter that lets Claude Code talk to baseRT
directly. This document records the motivation, design decisions, protocol
details, security posture, deployment notes and known limitations.

## 1. Motivation

Claude Code speaks only the Anthropic Messages API (`POST /v1/messages`, SSE
streaming, content blocks); `basert serve` exposes OpenAI-compatible endpoints
(`/v1/chat/completions`). A protocol bridge is required; the question is who
bridges and how.

### Why not LiteLLM (measured, 2026-08-08)

The LiteLLM bridge failed in two ways that compound into "Claude Code gets a
200 but displays nothing":

**Failure A: Claude Code gives up on slow streams and falls back to non-streaming**
- CC requests carry a ~23.5k-token system prompt (28 tool schemas + 3 system blocks)
- baseRT needs ~35.4s to prefill 23.5k tokens (663 t/s), emitting no bytes meanwhile
- CC's HTTP client (stainless) cuts the streaming connection after ~36s without a
  first byte and resends the same prompt **non-streaming** (packet capture: first
  attempt has `"stream": true`, the retry has no `stream` field)
- baseRT logs show the streaming request cut at ~35.8s (output truncated, no
  finish_reason)

**Failure B: LiteLLM's non-streaming translation drops all content**
- baseRT returns full content (`content` + `reasoning_content`) for non-streaming
- LiteLLM's anthropic translation yields `"content":[],"stop_reason":"end_turn"` — content swallowed
- Excluded via experiments: reproduces with and without upstream `reasoning_content`;
  identical on litellm 1.89.2 and 1.95.0
- Confirmed upstream: BerriAI/litellm#27492 (OPEN); root cause in
  `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`

**Conclusion**: the streaming path works fine through LiteLLM, but Failure A
pushes requests into Failure B's dead end. The community has the same idea
(leonmeijer/litellm_reasoning_proxy: force streaming + re-aggregate), but rather
than fix LiteLLM we generate the Anthropic protocol natively — following
llama.cpp server's approach.

## 2. Design

Modeled on llama.cpp server's Anthropic implementation (MIT):

```
Claude Code ──/v1/messages──▶ anthropic_adapter.py:4003 ──/v1/chat/completions, stream──▶ basert serve:7999
```

Key decisions:

| Decision | Rationale |
|---|---|
| Generate the Anthropic protocol natively | Avoids LiteLLM's translation bug; response shape fully under our control |
| SSE comment keep-alive pings | llama.cpp `sse_ping_interval` (default 30s): send `:\n\n` during silent periods so CC's 36s timeout never fires; we start pinging from request dispatch (every 20s), earlier than llama.cpp |
| Cancellation propagation inside the polling loop | llama.cpp `HTTP_POLLING_SECONDS = 1` + httplib `is_connection_closed` (non-blocking `recv(MSG_PEEK)` EOF probe): no extra threads/tasks |
| Upstream timeouts (connect 10s / read 600s) | baseRT is single-slot: when busy, connections queue server-side and response headers can be late; without a timeout the handler hangs forever. sock_read must exceed the longest silent gap in a stream (~35s prefill) |
| `--max-tokens-override` | CC sends small budgets (64/8192) for tool-loop and background calls; a heavy-thinking model burns the budget in thinking and truncates the body. The override replaces the client budget with a fixed value |
| Empty thinking-block signature | Same as llama.cpp (local models, no cryptographic verification) |
| Native non-streaming path | CC may fall back to non-streaming at any time; assembly shares the same block rules as streaming and is unaffected by the LiteLLM bug |

### Why "poll + probe" instead of write-failure detection

Cancelling a request = Claude Code closes the TCP connection. The adapter may
be blocked waiting for the next upstream chunk at that moment; the disconnect
is invisible until the next chunk arrives and the write to the client fails.
So we probe actively: on the 1s polling tick, `recv(1, MSG_PEEK)` on the client
socket — EOF means gone. This is more elegant than a watchdog task: the
detection rides on the existing loop's tick with zero extra machinery.

## 3. Protocol details

### 3.1 Request conversion (Anthropic → OpenAI)

- **system**: the top-level `system` param (string or text blocks) and every
  `role:"system"` entry inside `messages` (CC 2.x injects system reminders
  mid-array) are merged into a single leading system message. baseRT's Qwen
  chat template throws on a system message that is not first. With
  `--no-system-merge` this step is skipped (requires a template that handles
  mid-list system messages, e.g. froggeric v21.3 baked in via base-convert);
  see `misc/chat-template-fix.md`.
- **assistant messages**: `thinking` blocks → `reasoning_content`; `tool_use`
  blocks → OpenAI `tool_calls`
- **user messages**: `tool_result` blocks → `role:"tool"` messages, kept in
  their original position relative to text blocks
- **tools**: `input_schema` → `parameters`; anthropic-only fields are dropped
- **stream / max_tokens**: passed through (max_tokens replaceable by
  `--max-tokens-override`)
- **sampling params**: `temperature`, `top_p` passed through as-is;
  `stop_sequences` → `stop`

### 3.2 Streaming response (SSE lifecycle)

Block index rule (llama.cpp): **thinking (0) → text (0 or 1) → tool_use (n+)**

```
event: message_start                          ← on first chunk (id/model/usage)
event: content_block_start  index=0  thinking ← on first reasoning delta
event: content_block_delta  index=0  thinking_delta
event: content_block_start  index=1  text     ← on first text delta (1 if thinking, else 0)
event: content_block_delta  index=1  text_delta
event: content_block_start  index=n  tool_use ← tool call (id/name/input:{})
event: content_block_delta  index=n  input_json_delta
event: content_block_stop   index=…           ← close all open blocks in 0,1,n order
event: message_delta  stop_reason=end_turn|tool_use|max_tokens
event: message_stop
```

- Keep-alive: after 20s of silence, a single `:\n\n` line (SSE comment,
  treated as a heartbeat by clients)
- Anthropic streams do **not** emit `data: [DONE]` (same as llama.cpp)
- If the upstream stream ends before any content arrives, `message_start` is
  still emitted first, so the event sequence is always legal
- `finish_reason="length"` (openai's max_tokens truncation) is reported as
  `stop_reason: "max_tokens"`, never as a natural end

### 3.3 Non-streaming response

Assembles `thinking` (signature:"") + `text` + `tool_use` (input JSON-parsed,
`{}` on parse failure) blocks; `stop_reason` maps from `finish_reason`
(stop→end_turn, tool_calls→tool_use, length→max_tokens).

### 3.4 Cancellation propagation

- The main loop awaits `upstream_iter.__anext__()` under a 1s
  `asyncio.wait_for` (polling tick)
- On timeout → `client_gone()`: non-blocking `recv(1, MSG_PEEK)` on the client
  socket; EOF means the client left
- Client gone → `up.close()` aborts the upstream call → baseRT stops generating
- `ConnectionResetError` on writes to the client triggers the same upstream
  close
- Note: aiohttp's `StreamReader` has no `__anext__`; take
  `up.content.__aiter__()` (an `AsyncStreamIterator`) and wait_for on that

### 3.5 Timeout design

- Upstream connect: 10s; upstream read (silence between reads): 600s
  (must exceed the longest prefill, ~35s)
- While baseRT is single-slot, queued requests are covered by the connect
  timeout — they never hang forever

### 3.6 Token accounting (honest estimates)

`/v1/messages/count_tokens` never under-reports: Claude Code uses it for
context-budget management, and under-reporting would let the conversation
exceed the real context window. The endpoint and the streaming
`message_start.usage.input_tokens` both use `estimate_input_tokens()`, a
deliberate upper-bound heuristic (CJK characters ~1 token each, other text
~1 token per 3.5 chars, including tool schemas) — not a real tokenizer.
Streaming `message_delta.usage.output_tokens` uses the upstream usage when
available, otherwise the same estimate over the emitted text.

### 3.7 Classifier bypass (off by default)

Before executing Bash/WebFetch/WebSearch, Claude Code sends the conversation
transcript to `/v1/messages` under a separate model name (e.g.
`claude-sonnet-5`) for online safety judgment. Our backend is not a classifier:
in practice it misjudges harmless commands (`date && echo`, `find`,
`echo "test"`) as dangerous and emits `<block>yes</block>`, which makes CC
block normal tool calls.

`--allow-classifier-bypass` (env: none; CLI only) short-circuits requests that
match a heuristic (model name ≠ `local`, no tools, non-streaming, exactly one
user message) with a fixed `<block>no</block>` allow response.

**Security impact**: this disables CC's online harm detection — the operator
is fully responsible for what the agent executes. It is OFF by default; the
heuristic is a heuristic, not an identifier of the classifier, and may change
as CC evolves. Enable it only for local, offline, or trusted workloads.

**Alternatives**:
- A: set `CLAUDE_CODE_ATTRIBUTION_HEADER=0` in the CC config, exploiting a
  known bug (#64585) that breaks the classifier sub-request (CC-side, no
  adapter change)
- B: forward classifier requests to baseRT and accept the false-positive
  blocks (for environments where no security degradation is acceptable)
- C: deploy a dedicated harm-classifier fine-tune on baseRT

### 3.8 Authentication

`--master-key` is required. Every inbound `/v1/messages` and
`/v1/messages/count_tokens` request must present it via `x-api-key` or
`Authorization: Bearer`; anything else is rejected with an Anthropic-style
`authentication_error` 401 before the body is parsed. The same key is sent
upstream as a Bearer token — either set baseRT's `--api-key` to the same value,
or run baseRT on a trusted network. The adapter binds to `127.0.0.1` by
default; bind `0.0.0.0` only for cross-machine deployments.

## 4. Deployment & operations

```bash
cd cc-bridge
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python anthropic_adapter.py --master-key 'change-me' \
  --model basecompute/Qwen3.6-35B-A3B
```

Client (Claude Code) configuration: `docs/guides/connect-claude-code.md`.

**baseRT-side notes**: in single-slot deployments concurrent requests queue
(upstream timeouts prevent hangs but not throughput); consider
`--continuous-batching` for multiple clients.

## 5. Known limitations & open problems

| # | Problem | Status / mitigation |
|---|---|---|
| 1 | The model always thinks (cannot be disabled); thinking tokens count toward the output budget, so small budgets truncate the body | `--max-tokens-override` mitigates; proper fix needs separate thinking budget in baseRT (tracked in the `--chat-template`/thinking upstream issues) |
| 2 | baseRT single-slot queueing: one long request (35s prefill) stalls others | upstream timeouts prevent hangs; `--continuous-batching` recommended |
| 3 | Occasional GPU OOM on the first big request after baseRT restart (kIOGPUCommandBufferCallbackErrorOutOfMemory) | Retry passes (prefix-cache hit); likely a memory spike while the model loads |
| 4 | Empty thinking-block signature (CC accepts; local models have no verification) | Intentional, same as llama.cpp |
| 5 | Multimodal/audio inputs unsupported | text path first |
| 6 | CC system reminders merged to the front; positional semantics lost | baseRT Qwen template constraint; `--no-system-merge` when the template handles it |
| 7 | The max_tokens override also applies to CC background tasks (title generation etc.) and may return over-long replies | under observation; may differentiate by request shape |
| 8 | CC's online harm classifier is useless for non-Anthropic models — Qwen misjudges harmless commands as `<block>yes</block>`, blocking Bash/WebFetch | adapter can short-circuit these requests when `--allow-classifier-bypass` is enabled; security cost in §3.7 |

## 6. Verification record (2026-08-08)

- Non-streaming: thinking+text blocks complete (empty on the same path through
  LiteLLM)
- Streaming: lifecycle/block indices correct; tool calls `input_json_delta` +
  `stop_reason: tool_use`
- 36k-token prompt: completes fully; keep-alive pings work
- Cancellation propagation: client disconnect → upstream closed → marker logged
- CC end-to-end: multi-turn agentic tool loop (WebSearch) runs completely,
  replies display correctly
- Upstream busy (single-slot queueing): no more permanent hangs (connect/read
  timeouts)

## 7. References

- llama.cpp `tools/server/server-chat.cpp` (request conversion),
  `server-task.cpp` (to_json_anthropic / to_json_anthropic_stream),
  `server-context.cpp` (sse_ping_interval / HTTP_POLLING_SECONDS),
  `server-http.h` (is_connection_closed), `vendor/cpp-httplib/httplib.h`
- BerriAI/litellm#27492 (non-streaming content loss, OPEN)
- leonmeijer/litellm_reasoning_proxy (community solution to the same problem, MIT)
- froggeric/Qwen-Fixed-Chat-Templates v21.3 (vendored at
  `misc/chat_template-patched.jinja`, MIT)
