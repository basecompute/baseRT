#!/usr/bin/env python3
"""Exercise OpenAI-compatible tool calls across reused and concurrent requests."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_TOOL_NAME = "lookup_key"
_DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


@dataclass(frozen=True)
class Validation:
    errors: tuple[str, ...]
    finish_reason: Any = None
    tool_call_count: int | None = None
    function_name: Any = None
    arguments: Any = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class CaseResult:
    phase: str
    index: int
    expected_key: str
    ok: bool
    elapsed_ms: int
    http_status: int | None
    finish_reason: Any
    tool_call_count: int | None
    function_name: Any
    arguments: Any
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["errors"] = list(self.errors)
        return result


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    api_key: str | None
    sequential: int
    concurrency: tuple[int, ...]
    timeout: float
    max_tokens: int


def build_request(model: str, sentinel: str, max_tokens: int) -> dict[str, Any]:
    """Build the repeated-schema request used to expose cross-request state leaks."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Call {_TOOL_NAME} with key {sentinel}.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": "Lookup a key",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def validate_response(payload: Any, expected_key: str) -> Validation:
    """Validate one complete non-streaming chat-completion response."""
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return Validation(("response must be a JSON object",))

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return Validation(("response must contain at least one choice",))

    choice = choices[0]
    if not isinstance(choice, Mapping):
        return Validation(("choice must be a JSON object",))

    finish_reason = choice.get("finish_reason")
    if finish_reason != "tool_calls":
        errors.append(
            f"expected finish_reason 'tool_calls', got {finish_reason!r}"
        )

    message = choice.get("message")
    if not isinstance(message, Mapping):
        errors.append("choice.message must be a JSON object")
        return Validation(tuple(errors), finish_reason=finish_reason)

    tool_calls = message.get("tool_calls")
    call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    if not isinstance(tool_calls, list) or call_count != 1:
        errors.append(f"expected exactly one tool call, got {call_count}")
        return Validation(
            tuple(errors),
            finish_reason=finish_reason,
            tool_call_count=call_count,
        )

    call = tool_calls[0]
    if not isinstance(call, Mapping):
        errors.append("tool call must be a JSON object")
        return Validation(
            tuple(errors),
            finish_reason=finish_reason,
            tool_call_count=call_count,
        )

    function = call.get("function")
    if not isinstance(function, Mapping):
        errors.append("tool call function must be a JSON object")
        return Validation(
            tuple(errors),
            finish_reason=finish_reason,
            tool_call_count=call_count,
        )

    function_name = function.get("name")
    if function_name != _TOOL_NAME:
        errors.append(
            f"expected function name {_TOOL_NAME!r}, got {function_name!r}"
        )

    raw_arguments = function.get("arguments")
    arguments: Any = None
    arguments_decoded = False
    if not isinstance(raw_arguments, str):
        errors.append("tool arguments must be a JSON-encoded string")
    else:
        try:
            arguments = json.loads(raw_arguments)
            arguments_decoded = True
        except json.JSONDecodeError as exc:
            errors.append(f"tool arguments are not valid JSON: {exc.msg}")

    if arguments_decoded:
        if not isinstance(arguments, Mapping):
            errors.append("decoded tool arguments must be a JSON object")
        elif arguments != {"key": expected_key}:
            errors.append(
                f"expected arguments {{'key': {expected_key!r}}}, got {arguments!r}"
            )

    return Validation(
        tuple(errors),
        finish_reason=finish_reason,
        tool_call_count=call_count,
        function_name=function_name,
        arguments=arguments,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _http_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("must be an absolute HTTP or HTTPS URL")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that sequential shared-prefix and concurrent tool requests "
            "preserve isolated arguments."
        )
    )
    parser.add_argument(
        "--base-url",
        type=_http_base_url,
        default=os.environ.get("BASERT_BASE_URL", _DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API base URL (default: {_DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BASERT_MODEL", "model.base"),
        help="model name sent in requests (default: model.base)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BASERT_API_KEY"),
        help="Bearer token; defaults to BASERT_API_KEY when set",
    )
    parser.add_argument(
        "--sequential",
        type=_nonnegative_int,
        default=10,
        help="number of sequential requests (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        nargs="+",
        default=[2, 4],
        metavar="N",
        help="concurrent batch sizes (default: 2 4)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=300.0,
        help="per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=2048,
        help="maximum completion tokens per request (default: 2048)",
    )
    return parser


def _endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _run_case(config: Config, phase: str, index: int, sentinel: str) -> CaseResult:
    body = json.dumps(
        build_request(config.model, sentinel, config.max_tokens),
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        _endpoint(config.base_url), data=body, headers=headers, method="POST"
    )

    started = time.monotonic()
    status: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            status = response.status
            response_body = response.read()
        try:
            payload = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            validation = Validation((f"response is not valid JSON: {exc}",))
        else:
            validation = validate_response(payload, sentinel)
    except urllib.error.HTTPError as exc:
        status = exc.code
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        validation = Validation((f"HTTP {exc.code}: {detail}",))
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        validation = Validation((f"request failed: {exc}",))
    except OSError as exc:
        validation = Validation((f"request failed: {exc}",))

    elapsed_ms = round((time.monotonic() - started) * 1000)
    return CaseResult(
        phase=phase,
        index=index,
        expected_key=sentinel,
        ok=validation.ok,
        elapsed_ms=elapsed_ms,
        http_status=status,
        finish_reason=validation.finish_reason,
        tool_call_count=validation.tool_call_count,
        function_name=validation.function_name,
        arguments=validation.arguments,
        errors=validation.errors,
    )


def _run_concurrent(config: Config, size: int) -> list[CaseResult]:
    phase = f"concurrency-{size}"
    barrier = threading.Barrier(size)

    def worker(index: int) -> CaseResult:
        try:
            barrier.wait(timeout=config.timeout)
        except threading.BrokenBarrierError:
            return CaseResult(
                phase=phase,
                index=index,
                expected_key=f"c{size}-{index:04d}",
                ok=False,
                elapsed_ms=0,
                http_status=None,
                finish_reason=None,
                tool_call_count=None,
                function_name=None,
                arguments=None,
                errors=("concurrency barrier timed out",),
            )
        return _run_case(config, phase, index, f"c{size}-{index:04d}")

    with ThreadPoolExecutor(max_workers=size) as executor:
        futures = [executor.submit(worker, index) for index in range(size)]
        return [future.result() for future in futures]


def _print_phase(phase: str, results: Sequence[CaseResult]) -> None:
    failed = [result for result in results if not result.ok]
    for result in failed:
        print("FAIL " + json.dumps(result.to_dict(), sort_keys=True))
    passed = len(results) - len(failed)
    label = "PASS" if not failed else "FAILED"
    print(f"{label} {phase}: {passed}/{len(results)} requests passed")


def run(config: Config) -> list[CaseResult]:
    all_results: list[CaseResult] = []

    sequential_results = [
        _run_case(config, "sequential", index, f"seq-{index:04d}")
        for index in range(config.sequential)
    ]
    _print_phase("sequential", sequential_results)
    all_results.extend(sequential_results)

    for size in config.concurrency:
        concurrent_results = _run_concurrent(config, size)
        _print_phase(f"concurrency-{size}", concurrent_results)
        all_results.extend(concurrent_results)

    return all_results


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        sequential=args.sequential,
        concurrency=tuple(args.concurrency),
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    results = run(config)
    failed = sum(not result.ok for result in results)
    summary = {
        "ok": failed == 0,
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
