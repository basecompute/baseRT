# Design: Native Anthropic Messages adapter for baseRT (cc-bridge)

Status: implemented (2026-08-08), field-tested with Claude Code 2.1.226 over LAN.

## Problem

Claude Code speaks only the Anthropic Messages API (`POST /v1/messages`, SSE
streaming, content blocks). `basert serve` exposes OpenAI-compatible endpoints
(`/v1/chat/completions` etc.). Bridging is required; the question is where.

### Why LiteLLM as the bridge failed (measured, 2026-08-08)

Two independent faults, both reproduced end-to-end with byte-level captures:

1. **Non-streaming translation drops content.** LiteLLM's openai→anthropic
   non-streaming conversion always returns `"content":[]` regardless of the
   upstream payload (tested with the real baseRT response and a fake upstream
   with and without `reasoning_content`; litellm 1.89.2 and 1.95.0 identical).
   Upstream confirmation: BerriAI/litellm#27492 (OPEN). Streaming path is fine.

2. **Claude Code gives up on slow streaming and falls back to non-streaming.**
   CC's requests carry ~23.5k tokens (28 tool schemas + 3 system blocks);
   baseRT prefills that in ~35.4 s. CC's HTTP client (stainless) abandoned the
   stream after ~36 s of silence and resent the same prompt as a non-streaming
   request — landing on fault 1. Every request in the session did this.

Net effect: CC received `200 OK` messages with empty content and rendered
nothing ("worked for 1m 55s").

## Solution: native adapter, no middleware

Copy llama.cpp server's Anthropic pattern (MIT): a thin HTTP layer that owns
the Anthropic wire format end to end.

```
Claude Code ──/v1/messages──▶ cc-bridge (anthropic_adapter.py) ──/v1/chat/completions──▶ basert serve
```

Implementation (cc-bridge/anthropic_adapter.py, ~450 lines, Python/aiohttp):

- **Request**: convert Anthropic → OpenAI chat. System prompt (string or blocks)
  plus any `role:"system"` entries inside `messages` (CC 2.x system reminders)
  are merged into a single leading system message — baseRT's Qwen chat template
  raises if a system message is not first.
- **Streaming response**: message_start → content_block_start (thinking, idx 0)
  → thinking_delta / text_delta (text, idx 0/1) → tool_use blocks with
  input_json_delta → message_delta (stop_reason) → message_stop. baseRT already
  splits `reasoning_content`/`content`, so no chat-template reasoning detection
  is needed.
- **Keep-alive**: SSE comment pings (`:\n\n`) every 20 s while the stream is
  silent, so CC never times out during long prefills (llama.cpp's
  `sse_ping_interval`, default 30 s; we start pinging from request dispatch).
- **Non-streaming response**: assembled into thinking (empty `signature`, like
  llama.cpp) + text + tool_use blocks — immune to the LiteLLM bug.
- **`--max-tokens-override`**: CC sends small budgets (64/8192) for tool-loop
  and background calls; a heavy-thinking local model truncates on those. The
  flag replaces the client budget with a fixed one (e.g. 32000).
- **`--debug-dir`**: saves every request body and response (raw SSE for
  streams) for offline debugging.
- `/api/hello` (CC health check) and `/health` endpoints included.

## Acceptance

- [x] Non-streaming /v1/messages returns thinking + text blocks (litellm path returned empty).
- [x] Streaming lifecycle and block indices verified (thinking 0 → text 0/1 → tool_use n+).
- [x] Tool use streams `input_json_delta` with `stop_reason: tool_use`.
- [x] 36k-token prompt completes with pings keeping the connection alive.
- [x] Claude Code on LAN renders replies (repeated calls, no regression).

## Known limitations

- Model always reasons (cannot be disabled); thinking tokens count toward
  output budget — use `--max-tokens-override` generously.
- Thinking blocks carry empty signature (local models, no crypto verification).
- Multimodal/audio input not supported yet (text path only).
- baseRT occasionally OOMs on the first large request right after model load
  (GPU memory peak); a retry succeeds (prefix cache then serves it).
