# The `.base` format

`.base` is BaseRT's on-disk model container: a header describing the
architecture and tensor inventory, followed by the packed weight blob. Bundles
can be signed (ed25519) and carry provenance for the quant profile used at
convert time.

You produce `.base` files with [`basert convert`](../guides/converting.md) or
[`basert pull`](../guides/models.md), and inspect them with `basert inspect`.

## Canonical spec

The authoritative container specification lives in the repo:

- **[`base-convert/FORMAT.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/FORMAT.md)**
  — header schema, tensor table, slot layout, and the on-disk byte layout.

## Inspecting a bundle

```sh
basert inspect models/your-model.base
basert inspect models/your-model.base --verify-checksums   # verify per-tensor xxhash64
```

This prints the header (architecture, config), the tensor inventory (names,
dtypes, shapes, group sizes), and the multimodal sub-bundle slots when present.

## Header highlights

The header records, among other fields:

- `arch` — the model architecture (drives the runtime's forward path).
- `target_backend` — e.g. `metal`.
- `quant_profile` — the profile name used at convert time (audit trail).
- `calibration` — AWQ method/tokens/dataset when calibration was applied.
- Per-tensor entries — `dtype`, `group_size`, `scale_dtype`, `symmetric`,
  layout, and offsets.

See [Quantization](quantization.md) for what the per-tensor fields mean.
