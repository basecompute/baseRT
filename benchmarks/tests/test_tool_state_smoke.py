from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import tool_state_smoke  # type: ignore[import-not-found]  # noqa: E402


def _response(key: str, *, name: str = "lookup_key", finish_reason: str = "tool_calls") -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps({"key": key}),
                            },
                        }
                    ]
                },
            }
        ]
    }


class ToolResponseValidationTests(unittest.TestCase):
    def test_accepts_one_exact_lookup_key_call(self) -> None:
        result = tool_state_smoke.validate_response(_response("seq-0000"), "seq-0000")

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, ())
        self.assertEqual(result.arguments, {"key": "seq-0000"})

    def test_rejects_non_exact_key_value(self) -> None:
        result = tool_state_smoke.validate_response(
            _response("prefix-seq-0000-suffix"), "seq-0000"
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("expected arguments" in error for error in result.errors))

    def test_rejects_extra_argument_properties(self) -> None:
        payload = _response("seq-0000")
        payload["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = json.dumps({"key": "seq-0000", "extra": "stale-c4-0001"})

        result = tool_state_smoke.validate_response(payload, "seq-0000")

        self.assertFalse(result.ok)
        self.assertTrue(any("expected arguments" in error for error in result.errors))

    def test_rejects_missing_key_property(self) -> None:
        payload = _response("seq-0000")
        payload["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = "{}"

        result = tool_state_smoke.validate_response(payload, "seq-0000")

        self.assertFalse(result.ok)
        self.assertTrue(any("expected arguments" in error for error in result.errors))

    def test_rejects_wrong_function_name(self) -> None:
        result = tool_state_smoke.validate_response(
            _response("seq-0000", name="other_lookup"), "seq-0000"
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("function name" in error for error in result.errors))

    def test_rejects_multiple_tool_calls(self) -> None:
        payload = _response("seq-0000")
        payload["choices"][0]["message"]["tool_calls"].append(
            payload["choices"][0]["message"]["tool_calls"][0]
        )

        result = tool_state_smoke.validate_response(payload, "seq-0000")

        self.assertFalse(result.ok)
        self.assertTrue(any("exactly one tool call" in error for error in result.errors))

    def test_rejects_arguments_that_are_not_json(self) -> None:
        payload = _response("seq-0000")
        payload["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = '{"key":'

        result = tool_state_smoke.validate_response(payload, "seq-0000")

        self.assertFalse(result.ok)
        self.assertTrue(any("valid JSON" in error for error in result.errors))

    def test_rejects_json_null_arguments(self) -> None:
        payload = _response("seq-0000")
        payload["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = "null"

        result = tool_state_smoke.validate_response(payload, "seq-0000")

        self.assertFalse(result.ok)
        self.assertTrue(any("JSON object" in error for error in result.errors))

    def test_rejects_non_tool_finish_reason(self) -> None:
        result = tool_state_smoke.validate_response(
            _response("seq-0000", finish_reason="length"), "seq-0000"
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("finish_reason" in error for error in result.errors))

    def test_rejects_malformed_response_shape_without_raising(self) -> None:
        result = tool_state_smoke.validate_response({"choices": []}, "seq-0000")

        self.assertFalse(result.ok)
        self.assertTrue(result.errors)


class RequestTests(unittest.TestCase):
    def test_request_uses_lookup_schema_and_exact_sentinel(self) -> None:
        payload = tool_state_smoke.build_request("model.base", "seq-0007", 321)

        self.assertEqual(payload["model"], "model.base")
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(
            payload["messages"],
            [{"role": "user", "content": "Call lookup_key with key seq-0007."}],
        )
        function = payload["tools"][0]["function"]
        self.assertEqual(function["name"], "lookup_key")
        self.assertEqual(function["parameters"]["required"], ["key"])
        self.assertFalse(function["parameters"]["additionalProperties"])
        self.assertEqual(
            function["parameters"]["properties"]["key"], {"type": "string"}
        )

    def test_defaults_cover_sequential_and_concurrent_regressions(self) -> None:
        args = tool_state_smoke.build_parser().parse_args([])

        self.assertEqual(args.sequential, 10)
        self.assertEqual(args.concurrency, [2, 4])

    def test_rejects_nonfinite_timeout_and_malformed_base_url(self) -> None:
        parser = tool_state_smoke.build_parser()
        for value in ("nan", "inf"):
            with self.subTest(timeout=value), self.assertRaises(SystemExit):
                parser.parse_args(["--timeout", value])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--timeout=-inf"])
        for value in ("", "localhost:8080/v1", "ftp://example.com/v1"):
            with self.subTest(base_url=value), self.assertRaises(SystemExit):
                parser.parse_args(["--base-url", value])


class _FakeState:
    def __init__(self, corrupt: bool = False, http_error: bool = False) -> None:
        self.corrupt = corrupt
        self.http_error = http_error
        self.requests: list[dict] = []
        self.authorization: list[str | None] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()


@contextlib.contextmanager
def _fake_server(*, corrupt: bool = False, http_error: bool = False):
    state = _FakeState(corrupt=corrupt, http_error=http_error)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            with state.lock:
                state.requests.append(payload)
                state.authorization.append(self.headers.get("Authorization"))
                state.active += 1
                state.max_active = max(state.max_active, state.active)

            try:
                time.sleep(0.05)
                if state.http_error:
                    body = b'{"error":{"message":"synthetic failure"}}'
                    self.send_response(500)
                else:
                    prompt = payload["messages"][0]["content"]
                    key = prompt.removeprefix("Call lookup_key with key ").removesuffix(".")
                    if state.corrupt:
                        key = f"corrupt-{key}"
                    body = json.dumps(_response(key)).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with state.lock:
                    state.active -= 1

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class HarnessIntegrationTests(unittest.TestCase):
    def test_harness_passes_and_sends_concurrent_requests(self) -> None:
        with _fake_server() as (state, base_url), contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout:
            exit_code = tool_state_smoke.main(
                [
                    "--base-url",
                    base_url,
                    "--model",
                    "model.base",
                    "--api-key",
                    "test-key",
                    "--sequential",
                    "2",
                    "--concurrency",
                    "2",
                    "--timeout",
                    "2",
                    "--max-tokens",
                    "64",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(state.requests), 4)
        sentinels = [
            request["messages"][0]["content"].removeprefix(
                "Call lookup_key with key "
            ).removesuffix(".")
            for request in state.requests
        ]
        self.assertEqual(len(sentinels), len(set(sentinels)))
        self.assertGreaterEqual(state.max_active, 2)
        self.assertEqual(state.authorization, ["Bearer test-key"] * 4)
        self.assertIn('"ok": true', stdout.getvalue())

    def test_harness_returns_nonzero_with_machine_readable_failure(self) -> None:
        with _fake_server(corrupt=True) as (_state, base_url), contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout:
            exit_code = tool_state_smoke.main(
                [
                    "--base-url",
                    base_url,
                    "--model",
                    "model.base",
                    "--sequential",
                    "1",
                    "--concurrency",
                    "2",
                    "--timeout",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 1)
        failure_lines = [
            line.removeprefix("FAIL ")
            for line in stdout.getvalue().splitlines()
            if line.startswith("FAIL ")
        ]
        self.assertEqual(len(failure_lines), 3)
        failure = json.loads(failure_lines[0])
        self.assertEqual(failure["expected_key"], "seq-0000")
        self.assertFalse(failure["ok"])
        self.assertIn("errors", failure)

    def test_http_errors_are_reported_without_a_traceback(self) -> None:
        with _fake_server(http_error=True) as (_state, base_url), contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout:
            exit_code = tool_state_smoke.main(
                [
                    "--base-url",
                    base_url,
                    "--model",
                    "model.base",
                    "--sequential",
                    "1",
                    "--concurrency",
                    "2",
                    "--timeout",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("HTTP 500", stdout.getvalue())
        self.assertNotIn("Traceback", stdout.getvalue())


class WorkflowIntegrationTests(unittest.TestCase):
    def test_manual_workflow_can_opt_into_harness(self) -> None:
        workflow = (
            Path(__file__).resolve().parents[2]
            / ".github"
            / "workflows"
            / "serve-smoke.yml"
        ).read_text()

        self.assertIn("run_tool_state_smoke:", workflow)
        self.assertIn("type: boolean", workflow)
        self.assertIn("benchmarks/scripts/tool_state_smoke.py", workflow)
        self.assertIn("'basert-engine-macos-arm64*.tar.gz'", workflow)
        self.assertIn("./build/basert-serve models/model.base", workflow)
        self.assertNotIn("baseRT_serve", workflow)
        self.assertIn("--base-url http://127.0.0.1:8080/v1", workflow)
        self.assertIn("--model model.base", workflow)


if __name__ == "__main__":
    unittest.main()
