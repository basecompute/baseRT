# The `basert` CLI

`basert` is a single front-end with three kinds of commands:

- **Native** (run by the CLI itself): `pull`, `list`, `convert`, `inspect`,
  `sign`, `verify`, `keygen`, `add-speculator`, `superfluid`, `self`. These
  are the model hub, the converter, and installation management.
- **Forwarded** (dispatched to the engine): `chat`, `complete`, `bench`,
  `transcribe`. These exec the matching `basert-<cmd>` runtime binary.
- **Served**: `serve` runs [Superfluid](https://github.com/basecompute/superfluid),
  the open-source server, with the engine beside `basert`. `computearena`
  runs the separately installed ComputeArena CLI with BaseRT selected.

```
basert <command> [args...]
```

## How forwarding works

When you run a forwarded command, `basert` first resolves any model id in the
arguments (see [Model arguments](#model-arguments)), then looks for the
matching engine binary and replaces the current process with it (`exec`).
Resolution order:

1. `basert-<cmd>` next to the `basert` executable (how a release ships).
2. `baseRT_<cmd>` next to it (local dev builds).
3. `basert-<cmd>`, then `baseRT_<cmd>`, on your `PATH`.

This is why a release bundle, which keeps the CLI and the tools in one
directory, works without any extra wiring. See
[Installation](../getting-started/installation.md).

## How `serve` works

`basert serve` reads the same command line the earlier built-in server took,
finds the engine library beside `basert` (or the one `BASERT_LIB` names),
finds a Superfluid that can serve it, and runs `superfluid serve` with the
model. When no installed Superfluid can serve the engine, it installs the
newest release that can, asking first in a terminal.
`basert superfluid install` does this ahead of time, and
`basert superfluid status` shows what is installed.

Flags `basert serve` does not know are handed to `superfluid serve` as they
are. `basert serve --print-command` prints the Superfluid command line a start
would run. See [Serving an API](../guides/serving.md) and
[Superfluid's docs](https://docs.basecompute.co/latest/).

## Updates

From 0.5.5, a command run in a terminal may first update basert (an install
the script made, then running on the new release) or the Superfluid it runs,
to a newer release the day's look found. A Homebrew, global npm, pipx, uv or
cargo install is updated by its manager in the background once the command is
done; for pip, a line names the command. A release is taken as soon as it is
out; `basert self auto-update wait <days>` waits that many days after it
first. `--no-update` on any command turns off the day's look, the update and
the line saying a release is out for one run, `BASERT_NO_UPDATE=1` for a
shell, and `basert self auto-update off` turns off the update for every
command (the line stays). Commands that only tell
something (`self`, `--help`, `--version`, `list`, `inspect`, `verify`,
`superfluid`, anything with `--json`) never update. See
[Updating](../getting-started/installation.md#updating).

## Model arguments

Anywhere a model is expected, you can pass a **hub id** (e.g.
`basecompute/Qwen3-4B`) or a **path** to a `.base` file. A hub id is resolved
from the local cache and pulled on first use. `chat`, `complete`, `bench` and
`serve` take the model positionally; `serve --model` adds more models.

Pick a build with an inline `:variant` (`basecompute/Qwen3-4B:default-q8`) or
the `--variant` flag, which applies to ids without one. `--offline` on
`basert serve` uses only what is installed and downloads nothing.

## Discoverability

`basert --help` lists the commands. Each command has its own help:

```sh
basert --help
basert serve --help
basert chat --help
basert self --help
```

See the [command reference](reference.md) for every flag.
