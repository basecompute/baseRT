"""Shared fixtures: a started adapter server backed by a fake upstream.

The fake upstream speaks OpenAI chat/completions and is fully configurable
per test (default non-streaming JSON, or any custom SSE byte stream).
"""
import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import anthropic_adapter as adapter


async def _default_upstream(request):
    """Non-streaming chat.completion; streaming tests override the handler."""
    body = await request.json()
    if body.get("stream"):
        resp = web.StreamResponse(status=200,
                                  headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"choices":[{"index":0,"delta":'
                         b'{"role":"assistant","content":"hi"}}]}\n\n')
        await resp.write(b'data: {"choices":[{"index":0,"delta":{},'
                         b'"finish_reason":"stop"}]}\n\n')
        await resp.write(b"data: [DONE]\n\n")
        return resp
    return web.json_response({
        "id": "chatcmpl-test",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    })


async def _adapter_startup(app):
    app["session"] = aiohttp.ClientSession()


async def _adapter_shutdown(app):
    await app["session"].close()


@pytest.fixture
async def make_bridge():
    """Build and start (adapter server, fake upstream server).

    Returns an async factory; app overrides (e.g. allow_classifier_bypass,
    debug_dir) are passed as keyword arguments.
    """
    created = []

    async def build(upstream_handler=None, **app_overrides):
        up_app = web.Application()
        up_app.router.add_post("/v1/chat/completions",
                               upstream_handler or _default_upstream)
        up_server = TestServer(up_app)
        await up_server.start_server()
        created.append(up_server)

        app = web.Application()
        app["upstream"] = f"http://127.0.0.1:{up_server.port}/v1"
        app["master_key"] = "test-key"
        app["model"] = "backend-model"
        app["max_tokens_override"] = 0
        app["debug_dir"] = ""
        app["no_system_merge"] = False
        app["allow_classifier_bypass"] = False
        app.update(app_overrides)
        app.router.add_post("/v1/messages", adapter.handle_messages)
        app.router.add_post("/v1/messages/count_tokens", adapter.handle_count_tokens)
        app.on_startup.append(_adapter_startup)
        app.on_shutdown.append(_adapter_shutdown)

        server = TestServer(app)
        await server.start_server()
        created.append(server)
        return server

    yield build

    for s in reversed(created):
        await s.close()
