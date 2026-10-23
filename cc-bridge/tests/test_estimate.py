"""estimate_input_tokens: honest upper bound, monotonic, never zero."""

from anthropic_adapter import estimate_input_tokens


def test_never_zero():
    assert estimate_input_tokens({}) >= 1
    assert estimate_input_tokens({"messages": []}) >= 1


def test_ascii_estimate():
    body = {"messages": [{"role": "user", "content": "a" * 1000}]}
    est = estimate_input_tokens(body)
    # upper bound: 1000 ascii chars / 3.5 per token
    assert 1 <= est <= 1000


def test_cjk_heavier_than_ascii():
    body_cjk = {"messages": [{"role": "user", "content": "中" * 100}]}
    body_ascii = {"messages": [{"role": "user", "content": "x" * 100}]}
    assert estimate_input_tokens(body_cjk) > estimate_input_tokens(body_ascii)


def test_monotonic_with_more_content():
    small = {"messages": [{"role": "user", "content": "hello"}]}
    big = {"messages": [{"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "world"}]}
    assert estimate_input_tokens(big) > estimate_input_tokens(small)


def test_tools_schema_counts():
    no_tools = {"messages": [{"role": "user", "content": "hi"}]}
    with_tools = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "f",
                   "input_schema": {"type": "object",
                                    "properties": {"a": {"type": "string"},
                                                   "b": {"type": "integer"}}}}],
    }
    assert estimate_input_tokens(with_tools) > estimate_input_tokens(no_tools)


def test_tool_result_and_thinking_counted():
    base = {"messages": [{"role": "user", "content": "hi"}]}
    tool = {"messages": [{"role": "user",
                          "content": [{"type": "tool_result", "tool_use_id": "t",
                                       "content": "output " * 50}]}]}
    think = {"messages": [{"role": "assistant",
                           "content": [{"type": "thinking", "thinking": "deep " * 50}]}]}
    assert estimate_input_tokens(tool) > estimate_input_tokens(base)
    assert estimate_input_tokens(think) > estimate_input_tokens(base)
