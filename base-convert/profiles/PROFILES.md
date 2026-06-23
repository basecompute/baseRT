# Quantization profiles

A profile is a JSON file that tells `base-convert` what dtype to use for each
tensor, matched by glob pattern. Pass one with `--profile <file.json>`.

## Schema

```json
{
  "name": "my-profile",
  "arch": "*",
  "rules": [
    { "pattern": "model.embed_tokens.weight", "dtype": "f16" },
    { "pattern": "**.input_layernorm.weight", "dtype": "f16" },
    { "pattern": "**",                         "dtype": "base_q4" }
  ]
}
```

- `pattern` — glob over canonical tensor names; **first match wins**, so order
  from most-specific to the catch-all `**`.
- `dtype` — `f16`, `bf16`, `f32`, or a packed quant: `base_q2` … `base_q8`
  (see `CANONICAL_QUANT_SPEC.md` for the layouts).
- Optional `calibration` block enables AWQ:
  ```json
  "calibration": { "method": "awq", "tokens": 1024, "dataset": "wikitext-103" }
  ```
  AWQ needs a calibration corpus; download WikiText-103 and pass
  `--calib-file <wiki.train.raw>`.

## Bundled profiles

The `default-q4*` and `default-q8*` profiles are general-purpose starting points
(embeddings/norms kept high-precision, everything else Q4 or Q8). They work
across architectures (`"arch": "*"`).

## Writing your own

Per-model precision tuning — which layers tolerate low bits and which need
headroom — is model-specific and best found empirically. Start from a `default-*`
profile, lower bits on the bulk weights, keep embeddings/norms/router and any
sensitive projections higher, convert, and measure quality (perplexity or a task
metric) against size. Keep your tuned profiles in your own repo.
