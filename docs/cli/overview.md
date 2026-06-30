# The `basert` CLI

`basert` is a single front-end with two kinds of commands:

- **Native** (run by the CLI itself): `pull`, `list`, `convert`, `inspect`,
  `sign`, `verify`, `keygen`. These are the model hub + converter.
- **Forwarded** (dispatched to the engine): `serve`, `chat`, `complete`,
  `bench`, `profile`, `transcribe`. These exec the matching `basert-<cmd>`
  runtime binary.

```
basert <command> [args...]
```

## How forwarding works

When you run a forwarded command, `basert` looks for the matching engine binary
and replaces the current process with it (`exec`). Resolution order:

1. `basert-<cmd>` next to the `basert` executable (how a release ships).
2. `basert-<cmd>` on your `PATH`.
3. `baseRT_<cmd>` (local dev builds).

This is why putting the engine bundle directory and the CLI on the same `PATH`
makes `basert serve`/`chat`/… work without any extra wiring. See
[Installation](../getting-started/installation.md).

## Model arguments

Anywhere a model is expected, you can pass a **hub id** (resolved from the cache,
e.g. `Qwen/Qwen3-4B`) or a **path** to a `.base` file. `chat`/`complete` take the
model positionally; `serve` takes it via `--model` (repeatable).

## Discoverability

`basert --help` lists native commands and the forwarded runtime tools. Each tool
has its own help:

```sh
basert --help
basert serve --help
basert chat --help
```

See the [command reference](reference.md) for every flag.
