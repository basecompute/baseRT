"""The `basert` command installed by `pip install basert`.

Copied into the wheel as baseRT/_cli.py by packaging/python/build-wheel.sh.
It replaces this process with the `basert` launcher bundled in _engine/ rather
than running it as a child, so signals, exit codes and the terminal pass
straight through, and the launcher sees its real path and finds the `basert-*`
runtime tools and kernel files sitting beside it.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    exe = Path(__file__).resolve().parent / "_engine" / "basert"
    if not os.access(exe, os.X_OK):
        sys.exit(
            f"basert: the bundled launcher is missing or not executable ({exe}). "
            "Reinstall with `pip install --force-reinstall basert`."
        )
    os.execv(exe, [str(exe), *sys.argv[1:]])
