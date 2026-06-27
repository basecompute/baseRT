# Quant profiles

A profile is a reusable JSON file that maps tensor-name globs to per-tensor quant
rules. `basert convert --profile <path>` applies it; the bundle records the
profile name for audit.

The generic profiles ship in
[`base-convert/profiles/`](https://github.com/basecompute/baseRT/tree/main/base-convert/profiles)
(`default-q4`, `default-q8`, and scale-dtype variants). Full guidance is in
[`profiles/PROFILES.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/profiles/PROFILES.md).

## Schema

```jsonc
{
  "name": "my-profile-v1",          // recorded in the bundle's quant_profile
  "arch": "llama",                   // checked against the model's arch
  "calibration": {                    // optional; omit for RTN-only
    "method": "awq",
    "tokens": 1024,
    "dataset": "wikitext-103"
  },
  "rules": [                          // first match wins, per tensor
    { "pattern": "model.embed_tokens.weight", "dtype": "bf16" },
    { "pattern": "model.layers.*.self_attn.{q,k,v,o}_proj.weight",
      "dtype": "base_q4", "scale_dtype": "bf16", "group_size": 64 },
    { "pattern": "lm_head.weight", "dtype": "base_q8" },
    { "pattern": "**.weight", "dtype": "base_q4" }    // catch-all
  ]
}
```

## Glob syntax

| Token | Matches |
| --- | --- |
| `*` | anything except `.` (within one name segment) |
| `**` | anything, including `.` (any number of segments) |
| `{a,b,c}` | alternation (expanded at load time) |

Rules are evaluated top-down; the **first** matching rule wins. Include a
catch-all (`**.weight`) so every tensor is covered, or pair the profile with
`--target` as the fallback.

## Per-rule fields

| Field | Required | Notes |
| --- | --- | --- |
| `pattern` | yes | Tensor-name glob. |
| `dtype` | yes | `base_q2`…`base_q8`, `bf16`, `f16`, `f32`. |
| `group_size` | no | Defaults to the dtype's canonical group size. |
| `scale_dtype` | no | `bf16` (default) / `f16` / `e8m0` / `e4m3` (q8 only). |
| `symmetric` | no | Default `false` (asymmetric). |

## Tips

- Keep norms, routers, and (often) embeddings at `bf16`/`f16` — they're small
  and precision-sensitive.
- Validate by converting a small model and running `basert inspect` to confirm
  the per-tensor dtypes resolved as intended.
