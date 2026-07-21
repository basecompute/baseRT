#!/usr/bin/env python3
"""Tests for parsing basert-bench's human-readable result table."""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from parse_bench import parse_output  # noqa: E402


SAMPLE_OUTPUT = """\
BaseRT benchmark
| model | size | test | tok/s |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | pp128 | 553.17 ± 1.74 |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | tg128 | 97.04 ± 0.46 |
"""


class ParseOutputTests(unittest.TestCase):
    def test_parses_model_label_size_and_throughput(self):
        rows = parse_output(SAMPLE_OUTPUT, ["pp128", "tg128"])

        self.assertEqual(
            rows,
            [
                {
                    "model": "Qwen/Qwen3.6-35B-A3B · default-q4",
                    "size_mb": "20889",
                    "engine": "basert",
                    "test": "pp128",
                    "tok_per_sec": "553.17",
                    "stddev": "1.74",
                },
                {
                    "model": "Qwen/Qwen3.6-35B-A3B · default-q4",
                    "size_mb": "20889",
                    "engine": "basert",
                    "test": "tg128",
                    "tok_per_sec": "97.04",
                    "stddev": "0.46",
                },
            ],
        )

    def test_rejects_output_missing_an_expected_row(self):
        output = """\
| model | size | test | tok/s |
| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | pp128 | 553.17 ± 1.74 |
"""

        with self.assertRaisesRegex(ValueError, "missing expected row: tg128"):
            parse_output(output, ["pp128", "tg128"])

    def test_rejects_duplicate_expected_rows(self):
        output = SAMPLE_OUTPUT + (
            "| Qwen/Qwen3.6-35B-A3B · default-q4 | 20889 MiB | "
            "tg128 | 98.00 ± 0.50 |\n"
        )

        with self.assertRaisesRegex(ValueError, "duplicate benchmark row: tg128"):
            parse_output(output, ["pp128", "tg128"])

    def test_rejects_duplicate_expectations(self):
        with self.assertRaisesRegex(ValueError, "expected test names must be unique"):
            parse_output(SAMPLE_OUTPUT, ["pp128", "pp128"])


if __name__ == "__main__":
    unittest.main()
