"""AnthropicStreamer state machine: block indices, stop reasons, usage."""

import json

from anthropic_adapter import AnthropicStreamer, _fmt_sse


def _events(feed_chunks, finish=True):
    s = AnthropicStreamer("msg_x", "local", input_tokens=42)
    out = []
    for c in feed_chunks:
        out.extend(s.feed(c))
    if finish:
        out.extend(s.finish())
    return s, out


def _parse(events):
    return [(e.split("\n")[0].replace("event: ", ""),
             json.loads(e.split("\n", 1)[1][5:])) for e in events]


def test_normal_sequence_thinking_text_tool():
    s, ev = _events([
        {"choices": [{"delta": {"reasoning_content": "think"}}]},
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t1",
                                                "function": {"name": "bash",
                                                             "arguments": '{"c":"ls"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    parsed = _parse(ev)
    kinds = [(e, d.get("index")) for e, d in parsed if e == "content_block_start"]
    assert kinds == [("content_block_start", 0),  # thinking
                     ("content_block_start", 1),  # text
                     ("content_block_start", 2)]  # tool_use
    delta_kinds = [d["delta"]["type"] for e, d in parsed if e == "content_block_delta"]
    assert delta_kinds == ["thinking_delta", "text_delta", "input_json_delta"]
    msg_delta = [d for e, d in parsed if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


def test_length_maps_to_max_tokens():
    _, ev = _events([{"choices": [{"delta": {"content": "trunc"},
                                   "finish_reason": "length"}]}])
    msg_delta = [d for e, d in _parse(ev) if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "max_tokens"


def test_stop_maps_to_end_turn():
    _, ev = _events([{"choices": [{"delta": {"content": "x"},
                                   "finish_reason": "stop"}]}])
    msg_delta = [d for e, d in _parse(ev) if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


def test_empty_feed_still_finishes_cleanly():
    _, ev = _events([])
    parsed = _parse(ev)
    events = [e for e, _ in parsed]
    assert events == ["message_delta", "message_stop"]
    assert [d for e, d in parsed if e == "message_delta"][0]["usage"]["output_tokens"] == 0


def test_output_tokens_fallback_estimate_without_usage():
    s = AnthropicStreamer("msg_x", "local")
    ev = s.feed({"choices": [{"delta": {"content": "a" * 100}}]})
    assert ev  # start + delta
    s.finish()
    # no upstream usage came in, so the estimate must be non-zero
    _, finish_events = _events([{"choices": [{"delta": {"content": "a" * 100}}]}])
    msg_delta = [d for e, d in _parse(finish_events) if e == "message_delta"][0]
    assert msg_delta["usage"]["output_tokens"] >= 1


def test_usage_from_upstream_wins():
    _, ev = _events([
        {"usage": {"completion_tokens": 17}, "choices": [{"delta": {"content": "x"}}]},
    ])
    msg_delta = [d for e, d in _parse(ev) if e == "message_delta"][0]
    assert msg_delta["usage"]["output_tokens"] == 17


def test_initial_carries_input_tokens():
    s = AnthropicStreamer("msg_x", "local", input_tokens=42)
    parsed = _parse(s.initial())
    msg = parsed[0][1]["message"]
    assert msg["id"] == "msg_x"
    assert msg["model"] == "local"
    assert msg["usage"]["input_tokens"] == 42
    assert parsed[0][0] == "message_start"
