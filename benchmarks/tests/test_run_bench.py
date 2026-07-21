#!/usr/bin/env python3
"""Integration tests for the generic benchmark entrypoint."""

import csv
import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BENCH = REPO_ROOT / "benchmarks" / "scripts" / "run_bench.sh"


class RunBenchTests(unittest.TestCase):
    def test_specific_model_honors_sweep_environment_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            model = temporary / "model with spaces.base"
            model.touch()
            bench = temporary / "fake-basert-bench"
            log = temporary / "calls.txt"
            results = temporary / "results" / "bench.csv"
            bench.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import os
                    import sys

                    args = sys.argv[1:]
                    with open(os.environ["FAKE_BENCH_LOG"], "a") as handle:
                        handle.write(repr(args) + "\\n")
                    prompt = args[args.index("-p") + 1]
                    decode = args[args.index("-n") + 1]
                    print("| model | size | test | tok/s |")
                    print(f"| Model label with spaces · q4 | 123 MiB | pp{prompt} | 10.50 ± 0.25 |")
                    print(f"| Model label with spaces · q4 | 123 MiB | tg{decode} | 20.75 ± 0.50 |")
                    """
                )
            )
            bench.chmod(bench.stat().st_mode | stat.S_IXUSR)
            environment = os.environ.copy()
            environment.update(
                {
                    "BASERT": str(bench),
                    "RESULTS": str(results),
                    "PP_VALS": "128 512",
                    "TG_VAL": "64",
                    "REPS": "3",
                    "WARMUP": "2",
                    "FAKE_BENCH_LOG": str(log),
                }
            )

            completed = subprocess.run(
                ["bash", str(RUN_BENCH), str(model)],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            with results.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [(row["test"], row["model"]) for row in rows],
                [
                    ("pp128", "Model label with spaces · q4"),
                    ("tg64", "Model label with spaces · q4"),
                    ("pp512", "Model label with spaces · q4"),
                    ("tg64", "Model label with spaces · q4"),
                ],
            )
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 2)
            for prompt, call in zip(("128", "512"), calls):
                self.assertIn(repr(str(model)), call)
                self.assertIn(f"'-p', '{prompt}'", call)
                self.assertIn("'-n', '64'", call)
                self.assertIn("'-r', '3'", call)
                self.assertIn("'-w', '2'", call)


if __name__ == "__main__":
    unittest.main()
