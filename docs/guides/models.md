# Managing models

BaseRT has a built-in model hub. `basert pull` and `basert list` resolve models
from three sources, in priority order:

1. **Local** — already installed in the cache.
2. **Catalog** — pre-converted `.base` models hosted in the `basecompute`
   HuggingFace org, downloaded directly (no local conversion).
3. **HuggingFace** — any raw repo, downloaded and converted on the fly.

## Pulling

```sh
# A raw HuggingFace repo — source is downloaded and converted locally:
basert pull Qwen/Qwen3-4B

# A pre-converted catalog model — downloaded directly, no conversion:
basert pull basecompute/<name>

# Pin a revision / branch / tag:
basert pull Qwen/Qwen3-4B --revision main

# Choose the precision for convert-on-pull:
basert pull Qwen/Qwen3-4B --target base-q8
basert pull Qwen/Qwen3-4B --profile base-convert/profiles/default-q4.json

# See the plan without downloading:
basert pull Qwen/Qwen3-4B --dry-run

# Force a re-download / re-convert:
basert pull Qwen/Qwen3-4B --force
```

When no profile is given, convert-on-pull uses a generic default profile
(`default-q4`). Tuned, model-specific quality is delivered through the catalog as
pre-converted artifacts.

## How downloads behave

Model bundles are large — a 30B-class Q4 artifact is around 20 GB — so `basert
pull` fetches them over many connections at once, in fixed chunks, rather than
through a single stream.

- **An interrupted pull resumes.** Chunk completion is recorded beside the
  partial file as each chunk lands, so a crash, a `^C`, or a link that drops
  overnight continues from what is already on disk, including after the process
  has exited. Re-run the same `basert pull` command.
- **A stalled transfer fails instead of hanging.** A connection that stops
  delivering bytes without closing is cut off and retried.
- **A short transfer is an error.** A download smaller than the size the Hub
  advertised is reported rather than installed as a truncated model.

Defaults are 24 connections of 16 MB; peak memory is connections × chunk. Both
are tunable via the `BASERT_HF_*` variables in
[Installation](../getting-started/installation.md#environment-variables).

### Xet

Setting `BASERT_HF_XET=1` routes large files through Xet's content-addressed
store, which deduplicates chunks against models already fetched. It cannot
resume: an interrupted Xet transfer restarts from the beginning, which is why it
is not the default.

## Listing

```sh
basert list             # installed models (table)
basert list --remote    # also show catalog models not yet installed
basert list --json      # machine-readable
```

## Cache layout

Models live under `$BASERT_MODELS_DIR` (default `~/.cache/baseRT/models`):

```
~/.cache/baseRT/models/
  <org>/<model>/<variant>/model.base    ← the artifact the runtime loads
  <org>/<model>/<variant>/hub.json      ← provenance sidecar
  .src/<org>/<model>/<revision>/        ← raw HF snapshot staging (ignored by list)
```

`<variant>` encodes the quant profile (e.g. `default-q4`). The same directory is
read by the runtime, so any model you pull is immediately usable by `basert
chat`, `basert serve`, and the bindings.

## Using a pulled model

Anywhere a model is accepted, you can pass either a hub id (resolved from the
cache) or a path to a `.base` file:

```sh
basert chat  Qwen/Qwen3-4B
basert serve --model Qwen/Qwen3-4B
basert chat  ~/.cache/baseRT/models/Qwen/Qwen3-4B/default-q4/model.base
```

> [!NOTE]
> **`chat` vs `serve` argument style**
>
> `chat`/`complete` take the model **positionally**; `serve` takes it via
> `--model` (repeatable, to load several models at once).
