# Managing models

BaseRT has a built-in model hub. `basert pull` and `basert list` resolve models
from three sources, in priority order:

1. **Local**: already installed in the cache.
2. **Catalog**: pre-converted `.base` models hosted in the `basecompute`
   Hugging Face org, downloaded directly (no local conversion).
3. **Hugging Face**: any other repo. A repo that holds `.base` files is
   downloaded as is; a repo of source safetensors is downloaded and converted
   on the fly.

## Pulling

```sh
# A pre-converted catalog model: downloaded directly, no conversion
basert pull basecompute/Qwen3-8B

# A raw Hugging Face repo: source is downloaded and converted locally
basert pull Qwen/Qwen3-4B

# Pin a revision / branch / tag
basert pull Qwen/Qwen3-4B --revision main

# Choose the precision for convert-on-pull
basert pull Qwen/Qwen3-4B --target base-q8
basert pull Qwen/Qwen3-4B --profile base-convert/profiles/default-q4.json

# See the plan without downloading
basert pull Qwen/Qwen3-4B --dry-run

# Force a re-download / re-convert
basert pull Qwen/Qwen3-4B --force
```

Convert-on-pull uses `--target base-q4` when no target is given. The `base-q4`,
`base-q6` and `base-q8` targets use the generic `default-q4`, `default-q6` and
`default-q8` profiles; other targets are applied to every tensor. A `--profile`
overrides both. Tuned, model-specific quality is delivered through the catalog
as pre-converted files.

Gated or private repos need a Hugging Face token: `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`), or the token a Hugging Face CLI login stores under
`$HF_HOME`.

After a convert-on-pull the downloaded safetensors are deleted. Set
`BASERT_KEEP_HF_SOURCES=1` to keep them, for example to convert another quant
without downloading again.

## Variants

One model can be installed in several builds, called variants: `default-q4`,
`default-q8`, `default-f16`, and so on. `basert list` shows one row per
installed variant, and `basert list --json` gives each row's `variant` name.

Name a variant after the id with `:`. `basert pull` accepts a variant name, a
quant, or a size:

```sh
basert pull basecompute/gemma-4-E4B-it:default-q8   # a catalog variant
basert pull basecompute/gemma-4-E4B-it:q8           # the same, by quant
basert pull basecompute/whisper-tiny:default-f16
```

A variant already installed is used as it is, without contacting Hugging Face.

The runtime commands take the same form, or a `--variant` flag that applies to
every id without an inline `:variant`:

```sh
basert chat  basecompute/gemma-4-E4B-it:default-q8
basert serve basecompute/gemma-4-E4B-it --variant default-q8
```

With one variant installed, a bare id uses it. With several, a bare id is an
error that asks you to name one.

The catalog can list builds of a model for a specific GPU backend. `basert
pull` takes one this machine can run, and refuses a model published only for
another backend.

### Releases in sizes

Some releases name their files by size rather than by quant: `-XS`, `-Small`,
`-Medium`, `-Large` and `-XL` (the BaseQuant releases). A plain pull takes the
Medium:

```sh
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1           # the Medium
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1:small     # another size
basert serve basecompute/Qwen3.8-27B-BaseQuant-V1:xl
```

`--dry-run` says which file a pull would take. `--target` is refused on such a
release, since its files are not named by quant.

### Pulling a file by name

`--file` takes one `.base` from a repo by its file name, instead of picking one
by quant or size:

```sh
basert pull basecompute/Qwen3.8-27B-BaseQuant-V1 --file Qwen3.8-27B-BaseQuant-V1-Medium.base
```

A file the catalog lists installs under its catalog variant. Any other file
installs under a variant of its own, `file-<name>-<hash>`, and `basert list
--json` shows the file each install came from. A size and the matching
`--file` land in the same install.

When no file in a repo matches the quant you asked for, the error lists the
repo's files and suggests `--file`.

## Speculative-decoding drafters

The catalog also carries drafter models for speculative decoding, such as
`basecompute/Qwen3.8-27B-DFlash2` for `basecompute/Qwen3.8-27B`, and some
catalog files carry their own MTP head (the Gemma 4 and Qwen 3.5 / 3.6 / 3.8
bundles). Drafters are pulled like any other model:

```sh
basert pull basecompute/Qwen3.8-27B-DFlash2
```

You rarely need to: `basert serve` speculates by default and, on its first
start, downloads the drafter the catalog pairs with the model, if there is one. See
[Serving an API](serving.md).

## How downloads behave

Model files are large (a 30B-class Q4 file is around 20 GB), so `basert pull`
fetches them over many connections at once, in fixed chunks, rather than
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
basert list --json      # machine-readable, with each install's variant, path and file
```

The table shows each model's id, architecture, quant, size and status.

## The catalog

The catalog is a list of pre-converted models published in the
[baseRT repository](https://github.com/basecompute/baseRT/blob/main/base-convert/crates/base-hub/catalog.json).
`basert` keeps a copy in the cache and fetches it again once that copy is six
hours old, so new catalog models appear without a new release. A copy built into the binary
is the fallback. Set `BASERT_CATALOG_OFFLINE=1` to skip the network and use the
cached or built-in copy, or `BASERT_CATALOG_URL` to read another catalog.

## Cache layout

Models live under `$BASERT_MODELS_DIR`. The default is the platform cache
directory: `~/Library/Caches/baseRT/models` on macOS and
`~/.cache/baseRT/models` on Linux.

```
<models dir>/
  <org>/<model>/<variant>/model.base    ← the file the runtime loads
  <org>/<model>/<variant>/hub.json      ← provenance sidecar
  .src/hf/models--<org>--<model>/       ← download staging (ignored by list)
  .catalog-cache.json                   ← cached catalog
```

Any model you pull is immediately usable by `basert chat`, `basert serve`, and
the bindings.

## Using a pulled model

Anywhere a model is accepted, you can pass either a hub id (resolved from the
cache) or a path to a `.base` file:

```sh
basert chat  Qwen/Qwen3-4B
basert serve Qwen/Qwen3-4B
basert chat  ./models/my-model.base
```

A hub id that is not installed yet is pulled first, from the catalog when it
has a pre-converted copy (`Qwen/Qwen3-4B` is fetched and installed as
`basecompute/Qwen3-4B`), otherwise by converting the source repo.

> [!NOTE]
> **Several models in one server**
>
> `chat`, `complete`, `bench` and `serve` take the model **positionally**.
> `serve` also takes `--model` (repeatable) to load more models at once.
