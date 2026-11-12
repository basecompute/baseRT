"""Inbound authentication: pure function + endpoint level."""

import aiohttp

from anthropic_adapter import verify_master_key


def test_verify_missing_key_rejected():
    assert verify_master_key({}, "k") is False
    assert verify_master_key({"Content-Type": "application/json"}, "k") is False


def test_verify_x_api_key():
    assert verify_master_key({"x-api-key": "k"}, "k") is True
    assert verify_master_key({"x-api-key": "wrong"}, "k") is False


def test_verify_bearer():
    assert verify_master_key({"Authorization": "Bearer k"}, "k") is True
    assert verify_master_key({"Authorization": "Bearer wrong"}, "k") is False
    assert verify_master_key({"Authorization": "Basic abc"}, "k") is False


def test_verify_empty_configured_key_fails_closed():
    assert verify_master_key({"x-api-key": "k"}, "") is False
    assert verify_master_key({"x-api-key": "k"}, None) is False


async def test_endpoint_401_without_credentials(make_bridge):
    server = await make_bridge()
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local",
                                "messages": [{"role": "user", "content": "hi"}]}) as resp:
            assert resp.status == 401
            body = await resp.json()
            assert body["error"]["type"] == "authentication_error"


async def test_endpoint_401_with_wrong_key(make_bridge):
    server = await make_bridge()
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "wrong"}) as resp:
            assert resp.status == 401


async def test_401_before_body_parsing(make_bridge):
    """Bad JSON without credentials must 401, not 400 (auth precedes parsing)."""
    server = await make_bridge()
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          data=b"{not json", headers={"content-type": "application/json"}) as resp:
            assert resp.status == 401


async def test_count_tokens_requires_auth(make_bridge):
    server = await make_bridge()
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages/count_tokens"),
                          json={"messages": [{"role": "user", "content": "hi"}]}) as resp:
            assert resp.status == 401
        async with s.post(server.make_url("/v1/messages/count_tokens"),
                          json={"messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["input_tokens"] >= 1


async def test_authenticated_request_reaches_upstream(make_bridge):
    server = await make_bridge()
    async with aiohttp.ClientSession() as s:
        async with s.post(server.make_url("/v1/messages"),
                          json={"model": "local",
                                "messages": [{"role": "user", "content": "hi"}]},
                          headers={"x-api-key": "test-key"}) as resp:
            assert resp.status == 200
            body = await resp.json()
            assert body["type"] == "message"
            assert body["content"][0]["type"] == "text"
            assert body["usage"]["input_tokens"] == 10  # from fake upstream usage
