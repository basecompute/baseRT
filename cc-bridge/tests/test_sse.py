"""End-to-end SSE lifecycle through the real HTTP stack."""

import aiohttp
from aiohttp import web

from tests.helpers import split_sse, sse_bytes


async def test_stream_normal_lifecycle(make_bridge):
    async def upstream(request):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(sse_bytes(
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    server = await make_bridge(upstream_handler=upstream)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            assert resp.status == 200
            raw = await resp.read()

    events = split_sse(raw)
    kinds = [e for e, _ in events]
    assert kinds[0] == "message_start"
    assert "content_block_start" in kinds
    assert kinds[-1] == "message_stop"
    # anthropic streams must not pass [DONE] through
    assert b"[DONE]" not in raw
    # message_start carries the honest input estimate
    assert events[0][1]["message"]["usage"]["input_tokens"] >= 1

    # block indices: thinking 0, text 1 (thinking already started)
    starts = [d for e, d in events if e == "content_block_start"]
    assert [d["index"] for d in starts] == [0, 1]
    assert [d["content_block"]["type"] for d in starts] == ["thinking", "text"]


async def test_stream_message_start_leading_on_immediate_done(make_bridge):
    """Upstream sends [DONE] immediately: message_start must still lead."""
    async def upstream(request):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"data: [DONE]\n\n")
        return resp

    server = await make_bridge(upstream_handler=upstream)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            raw = await resp.read()

    events = split_sse(raw)
    kinds = [e for e, _ in events]
    assert kinds[0] == "message_start"
    assert kinds == ["message_start", "message_delta", "message_stop"]
    assert events[0][1]["message"]["usage"]["input_tokens"] >= 1


async def test_stream_message_start_leading_on_abrupt_close(make_bridge):
    """Upstream closes without any data: same guarantee."""
    async def upstream(request):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        return resp  # end the response body immediately

    server = await make_bridge(upstream_handler=upstream)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            raw = await resp.read()

    kinds = [e for e, _ in split_sse(raw)]
    assert kinds == ["message_start", "message_delta", "message_stop"]


async def test_stream_garbage_json_skipped(make_bridge):
    async def upstream(request):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"data: {not json\n\n")
        await resp.write(sse_bytes(
            {"choices": [{"delta": {"content": "still works"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    server = await make_bridge(upstream_handler=upstream)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            raw = await resp.read()

    events = split_sse(raw)
    kinds = [e for e, _ in events]
    assert kinds[0] == "message_start"
    text_deltas = [d["delta"].get("text") for e, d in events
                   if e == "content_block_delta" and d["delta"].get("type") == "text_delta"]
    assert text_deltas == ["still works"]


async def test_stream_truncation_reports_max_tokens(make_bridge):
    async def upstream(request):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(sse_bytes(
            {"choices": [{"delta": {"content": "half"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ))
        await resp.write(b"data: [DONE]\n\n")
        return resp

    server = await make_bridge(upstream_handler=upstream)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local", "stream": True,
                                "messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            raw = await resp.read()

    msg_delta = [d for e, d in split_sse(raw) if e == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "max_tokens"


async def test_classifier_bypass_only_when_enabled(make_bridge):
    """A classifier-shaped request is forwarded when the flag is off..."""
    async def upstream(request):
        body = await request.json()
        return aiohttp.web.json_response({
            "id": "chatcmpl-bypass",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "real answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    probe = {"model": "claude-sonnet-5",  # classifier model name
             "messages": [{"role": "user", "content": "should I run this?"}]}
    hdrs = {"x-api-key": "test-key"}

    server = await make_bridge(upstream_handler=upstream)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"), json=probe,
                          headers=hdrs) as resp:
            assert resp.status == 200
            body = await resp.json()
            # flag off: forwarded upstream, real answer
            assert body["content"][0]["text"] == "real answer"

    server = await make_bridge(upstream_handler=upstream,
                               allow_classifier_bypass=True)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"), json=probe,
                          headers=hdrs) as resp:
            assert resp.status == 200
            body = await resp.json()
            # flag on: short-circuited allow response
            assert body["content"][0]["text"] == "<block>no</block>"
            assert body["model"] == "local"

    # with tools present it is not a classifier probe, even with the flag on
    probe_tools = dict(probe, tools=[{"name": "f", "input_schema": {"type": "object"}}])
    server = await make_bridge(upstream_handler=upstream,
                               allow_classifier_bypass=True)
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"), json=probe_tools,
                          headers=hdrs) as resp:
            body = await resp.json()
            assert body["content"][0]["text"] == "real answer"
