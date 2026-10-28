# Chat template: mid-list system messages on Qwen3.5/3.6

## Problem

Claude Code (and other agentic coding tools) inject `role:"system"` messages
into the middle of the message array at runtime — context-compression
reminders, file-change notifications, and similar steering instructions.

The official Qwen3.5/3.6 chat template rejects this with a hard exception:

```
Qwen/Qwen3.6-35B-A3B, commit 7da1103:
{%- if message.role == "system" %}
    {%- if not loop.first %}
        {{- raise_exception('System message must be at the beginning.') }}
    {%- endif %}
{%- endif %}
```

When triggered, baseRT catches the exception and falls back to a generic ChatML
template. The request succeeds but loses

- model-specific formatting
- tool-call syntax
- reasoning delimiters

Any agentic workload that uses a Qwen3.5/3.6 model and injects system
instructions mid-conversation hits this.

## Community solutions

The most mature community response is
[froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v21.3, MIT — vendored reference copy at `chat_template-patched.jinja`), which
replaces the exception with a direct ChatML rendering of each system block in
its original position. It also adds several features for agentic workloads:

| Aspect | Official Qwen3.6 | froggeric v21.3 |
|---|---|---|
| Mid-list system message | `raise_exception` — crash | Rendered as an independent ChatML system block |
| Unknown message role | `raise_exception` — crash | Rendered as `[role]: content` |
| Missing user query | `raise_exception` — crash | Safe fallback, no crash |
| `developer` role | Not supported | Treated identically to `system` |
| Tool-call format | XML only | XML (default) or JSON, selectable via `_tool_format` |
| Consecutive tool failure detection | None | Detects error keywords (traceback, command not found, etc.); after two consecutive failures, injects a system warning and disables thinking to prevent error loops |
| Inline thinking control | None | Message-level `think_off` / `think_on` markers in system and user prompts (rendered via ChatML special tokens) |
| Tool argument truncation | None | `max_tool_arg_chars` prevents context-window overflow |
| Tool response truncation | None | `max_tool_response_chars` with `[TRUNCATED]` marker |
| Reasoning tag parsing | `<think>` only | Six variants: `<think>`, `<thinking>`, and whitespace-deformed forms |
| Auto-disable thinking with tools | None | `auto_disable_thinking_with_tools` parameter |
| Thinking during consecutive failures | Always on | Forced off after two consecutive tool errors |

## What this means for downstream bridges

`anthropic_adapter.py` contains system-block merging logic in
`convert_anthropic_to_oai()` (enabled by default, skippable with
`--no-system-merge`). Its purpose is to collapse Claude Code's top-level
`system` parameter and any mid-array system reminders into a single leading
system message so the official Qwen template does not crash. With a patched
template baked into the model via `base-convert`, that logic becomes
unnecessary.

## Recommendation

`basert serve` does not currently support loading an external chat template.
Adding a `--chat-template` flag (or an equivalent mechanism, as llama.cpp
provides) would resolve this at the engine level:

- The adapter's system-merging logic could be removed.
- Downstream bridges would not each need to work around a template limitation
  that only affects certain model versions.
- Users could deploy community-maintained templates optimized for agentic
  workloads without re-converting their models.

Tracked upstream as a GitHub issue (the engine-side feature request, opened
together with the cc-bridge PR); this document and the vendored template are
the reference material for it.
