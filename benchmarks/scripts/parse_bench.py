#!/usr/bin/env python3
"""Parse basert-bench's table output into benchmark CSV rows."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from typing import Iterable, TextIO

FIELDNAMES = [
    "model",
    "size_mb",
    "engine",
    "test",
    "tok_per_sec",
    "stddev",
]
NUMBER = r"[0-9]+(?:\.[0-9]+)?"
SIZE_RE = re.compile(rf"^(?P<size>{NUMBER})\s+MiB$")
RATE_RE = re.compile(
    rf"^(?P<tok_per_sec>{NUMBER})\s*±\s*(?P<stddev>{NUMBER})$"
)
TEST_RE = re.compile(r"^(?:pp|tg)[0-9]+$")


def parse_output(output: str, expected_tests: Iterable[str]) -> list[dict[str, str]]:
    """Return expected benchmark rows in the requested order.

    Non-table lines and table headers are ignored. Every expected test must occur
    exactly once so a changed or truncated basert-bench table cannot silently
    produce an incomplete results file.
    """
    expected = list(expected_tests)
    if len(set(expected)) != len(expected):
        raise ValueError("expected test names must be unique")
    parsed: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not TEST_RE.fullmatch(cells[2]):
            continue
        size = SIZE_RE.fullmatch(cells[1])
        rate = RATE_RE.fullmatch(cells[3])
        if size is None or rate is None:
            continue
        if cells[2] in parsed:
            raise ValueError(f"duplicate benchmark row: {cells[2]}")
        parsed[cells[2]] = {
            "model": cells[0],
            "size_mb": size.group("size"),
            "engine": "basert",
            "test": cells[2],
            "tok_per_sec": rate.group("tok_per_sec"),
            "stddev": rate.group("stddev"),
        }

    for test in expected:
        if test not in parsed:
            raise ValueError(f"missing expected row: {test}")
    return [parsed[test] for test in expected]


def write_rows(rows: Iterable[dict[str, str]], output: TextIO) -> None:
    """Write headerless rows for run_bench.sh to append to its result file."""
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        action="append",
        required=True,
        metavar="TEST",
        help="expected test row (repeat for each row, for example pp128 and tg128)",
    )
    args = parser.parse_args()
    try:
        rows = parse_output(sys.stdin.read(), args.expect)
    except ValueError as error:
        parser.error(str(error))
    write_rows(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
