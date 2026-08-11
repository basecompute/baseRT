# Connect Claude Code to baseRT via cc-bridge

This guide explains how to make Claude Code in your LAN talk to a local baseRT
inference server directly, without LiteLLM. The bridging component is
`cc-bridge/anthropic_adapter.py`, modeled on llama.cpp server's Anthropic
implementation (design rationale: `docs/proposals/native-anthropic-adapter.md`;
full protocol details: `cc-bridge/TECHNICAL.md`).

## Architecture

```
Claude Code ──(/v1/messages)──▶ cc-bridge :4003 ──(/v1/chat/completions, stream)──▶ baseRT :7999
```

- 4003: cc-bridge (this repository's `cc-bridge/`) — a pure HTTP translation
  layer; it touches no GPU and no model files
- 7999: `basert serve` itself

## Start

```bash
# prerequisite: baseRT is running (any address; point the adapter at it with --upstream)
cd cc-bridge
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python anthropic_adapter.py --master-key 'change-me' --model <model-id>
```

All flags are documented in `cc-bridge/README.md`. In particular
`--max-tokens-override` shields the backend from Claude Code's small budgets
(64/8192) — a heavy local thinking model needs a generous budget.

## Client configuration (Claude Code)

```bash
export ANTHROPIC_BASE_URL="http://<host-ip>:4003"   # host IP for LAN, localhost for same machine
export ANTHROPIC_AUTH_TOKEN="<same value as --master-key>"
export ANTHROPIC_MODEL="local"
export ANTHROPIC_DEFAULT_SONNET_MODEL="local"
export ANTHROPIC_DEFAULT_OPUS_MODEL="local"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="local"
claude
```

The `local` alias is arbitrary — the adapter echoes it back as `local` for
every request. Persist the block in the `env` section of `~/.claude/settings.json`.
**A new Claude Code session is required after changing these.**

## Verify

```bash
# smoke test (non-streaming)
curl -s -m 120 http://localhost:4003/v1/messages \
  -H "content-type: application/json" -H "x-api-key: <your-key>" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"local","max_tokens":64,"messages":[{"role":"user","content":"1+1=? Answer with a number only"}]}'
# expect: content contains {type:"thinking"} and {type:"text"} blocks, stop_reason: end_turn

# streaming
curl -sN -m 120 http://localhost:4003/v1/messages \
  -H "content-type: application/json" -H "x-api-key: <your-key>" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"local","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
# expect: message_start → content_block_start → thinking_delta → text_delta → message_delta → message_stop
```

End-to-end acceptance: run an agentic task through Claude Code that writes and
executes a file (multi-turn tool calls).

## Troubleshooting

- Request/response dump: start the adapter with `--debug-dir <dir>`; every
  request is saved as `-req.json` and the response as `-res.json` (streams:
  `-res.sse`), so you can compare byte-for-byte what CC sent and received.
- CC shows no reply but baseRT logs look fine: most likely content is being
  dropped in translation — check whether `text_delta` appears in `-res.sse`.
- Truncated replies: the model's thinking burns the output budget; raise
  `--max-tokens-override`.
- Health checks: CC probes `/api/hello`; the adapter implements it (200).
- 401s: your `ANTHROPIC_AUTH_TOKEN` must match the adapter's master key.

## Known limitations (measured)

- The model always thinks (cannot be disabled); thinking tokens count toward
  the output budget.
- Thinking blocks carry an empty signature (local models; same as llama.cpp).
- Claude Code's system reminders (mid-array system messages) are merged into
  the leading system message; pass `--no-system-merge` if the model's chat
  template handles mid-list system messages.
- Text-only; multimodal and audio inputs are not supported.
