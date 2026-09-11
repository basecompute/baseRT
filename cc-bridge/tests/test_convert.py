"""Golden cases for anthropic -> openai request conversion."""

from anthropic_adapter import convert_anthropic_to_oai


def test_system_string_merged_to_front():
    oai = convert_anthropic_to_oai({
        "system": "top-level",
        "messages": [
            {"role": "user", "content": "hi"},
        ],
    })
    assert oai["messages"] == [
        {"role": "system", "content": "top-level"},
        {"role": "user", "content": "hi"},
    ]


def test_system_text_blocks_and_midlist_system_merged():
    oai = convert_anthropic_to_oai({
        "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "system", "content": "mid reminder"},
            {"role": "user", "content": "second"},
        ],
    })
    roles = [m["role"] for m in oai["messages"]]
    assert roles == ["system", "user", "user"]
    assert oai["messages"][0]["content"] == "A\nB\nmid reminder"


def test_no_system_merge_keeps_positions():
    oai = convert_anthropic_to_oai({
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "system", "content": "mid"},
            {"role": "user", "content": "last"},
        ],
    }, merge_system=False)
    roles = [m["role"] for m in oai["messages"]]
    assert roles == ["user", "system", "user"]


def test_assistant_thinking_and_tool_use():
    oai = convert_anthropic_to_oai({
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me think"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "tu_1", "name": "bash",
                 "input": {"cmd": "ls"}},
            ],
        }],
    })
    msg = oai["messages"][0]
    assert msg["reasoning_content"] == "let me think"
    assert msg["content"] == "answer"
    assert msg["tool_calls"][0]["id"] == "tu_1"
    assert msg["tool_calls"][0]["function"]["name"] == "bash"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'


def test_user_tool_result_after_text():
    oai = convert_anthropic_to_oai({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "please run it"},
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"},
            ],
        }],
    })
    assert [m["role"] for m in oai["messages"]] == ["user", "tool"]
    assert oai["messages"][1]["tool_call_id"] == "tu_1"


def test_user_text_between_tool_results_keeps_order():
    oai = convert_anthropic_to_oai({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "a"},
                {"type": "text", "text": "now try b"},
                {"type": "tool_result", "tool_use_id": "tu_2", "content": "b"},
                {"type": "text", "text": "and c"},
            ],
        }],
    })
    assert [m["role"] for m in oai["messages"]] == ["tool", "user", "tool", "user"]
    assert oai["messages"][0]["tool_call_id"] == "tu_1"
    assert oai["messages"][2]["tool_call_id"] == "tu_2"


def test_consecutive_text_blocks_merged():
    oai = convert_anthropic_to_oai({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "one "},
                {"type": "text", "text": "two"},
            ],
        }],
    })
    assert len(oai["messages"]) == 1
    assert oai["messages"][0]["content"] == "one two"


def test_tools_mapping_and_tool_choice_none():
    oai = convert_anthropic_to_oai({
        "tools": [{"name": "f", "description": "d",
                   "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}],
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert oai["tools"][0]["function"]["name"] == "f"
    assert oai["tools"][0]["function"]["parameters"]["properties"]["x"]["type"] == "string"
    assert oai["tool_choice"] == "none"


def test_sampling_params_passthrough():
    oai = convert_anthropic_to_oai({
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["END", "STOP"],
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert oai["temperature"] == 0.7
    assert oai["top_p"] == 0.9
    assert oai["stop"] == ["END", "STOP"]


def test_defaults():
    oai = convert_anthropic_to_oai({
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert oai["stream"] is False
    assert oai["max_tokens"] == 2048
    assert "temperature" not in oai
    assert "stop" not in oai
