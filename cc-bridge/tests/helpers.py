"""Small test helpers shared by the test modules."""

import json


def split_sse(raw: bytes) -> list[tuple[str, dict]]:
    """Split a raw SSE byte stream into (event, data) pairs, skipping
    comment pings (lines with no data: field)."""
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        event, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data is None:
            continue
        events.append((event, json.loads(data)))
    return events


def sse_bytes(*chunks: dict) -> bytes:
    """Serialize openai-style chunks as SSE data lines."""
    out = b""
    for c in chunks:
        out += b"data: " + json.dumps(c).encode("utf-8") + b"\n\n"
    return out
