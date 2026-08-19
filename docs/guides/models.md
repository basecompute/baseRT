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

Model bundles are large — a 30B-class Q4 artifact is around 20 GB — so which
transport a file takes matters. The Hub decides per file, and `basert` follows:

- **Xet-backed repositories** (the default for new uploads, including the
  pre-converted `basecompute/` bundles) transfer over the deduplicated CAS path.
  This is the fast one, and there is nothing to enable.
- **Everything else** is fetched over several connections at once, in fixed
  chunks, rather than through a single stream.

Either way, a stalled transfer is cut off and retried rather than left hanging,
and a transfer shorter than the size the Hub advertised is reported as an error
instead of being installed as a truncated model.

### If a pull is interrupted

The two transports differ, and the difference is worth knowing before an
overnight pull:

- A **ranged** download resumes. Chunk completion is recorded beside the partial
  file as each chunk lands, so re-running the same `basert pull` continues from
  what is already on disk, including after the process has exited.
- A **Xet** download restarts from the beginning. It is fast enough that the
  window is much smaller — but it is a restart, not a resume.

On a link that drops often, `BASERT_HF_FORCE_RANGED=1` takes the ranged path
even for a Xet-backed file, trading throughput for the ability to pick up where
it stopped. The other `BASERT_HF_*` variables in
[Installation](../getting-started/installation.md#environment-variables) tune
connection count, chunk size, and the stall timeout.

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
