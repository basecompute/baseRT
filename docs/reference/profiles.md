# Quant profiles

A profile is a reusable JSON file that maps tensor-name globs to per-tensor quant
rules. `basert convert --profile <path>` applies it; the bundle records the
profile name in its `quant_profile` header field for audit.

The generic profiles ship in
[`base-convert/profiles/`](https://github.com/basecompute/baseRT/tree/main/base-convert/profiles):

| Profile | What it does |
| --- | --- |
| `default-q4` | Projections and other weights at `base_q4` (group 64, bf16 scales); embeddings, norms and biases at f16. |
| `default-q4-bf16` | The same rules as `default-q4`, with the bf16 scale dtype spelled out. |
| `default-q4-f16scale` | `default-q4` with f16 scales. |
| `default-q4-embq6` | `default-q4` with the token embeddings at `base_q6`. |
| `default-q6` | As `default-q4`, at `base_q6`. |
| `default-q8` | As `default-q4`, at `base_q8` (group 128). |
| `default-q8-f16scale` | `default-q8` with f16 scales. |
| `whisper-f16`, `whisper-q8`, `whisper-q4` | Whisper: everything f16, or the attention and MLP projections at q8 / q4 with the convolutional front end, embeddings, norms and biases at f16. |

The `default-*` profiles use `"arch": "*"` and work across architectures. A
`basert pull` that converts a Hugging Face checkpoint uses `default-q4` unless
you ask for another. Full guidance is in
[`profiles/PROFILES.md`](https://github.com/basecompute/baseRT/blob/main/base-convert/profiles/PROFILES.md).

## Schema

```jsonc
{
  "name": "my-profile-v1",           // recorded in the bundle's quant_profile
  "arch": "*",                        // the architecture it is written for; "*" for any
  "calibration": {                    // optional; recorded with the bundle
    "method": "awq",
    "tokens": 1024,
    "dataset": "wikitext-103"
  },
  "rules": [                          // first match wins, per tensor
    { "pattern": "embed_tokens.weight", "dtype": "f16" },
    { "pattern": "**.input_layernorm.weight", "dtype": "f16" },
    { "pattern": "**.{q,k,v,o}_proj.weight",
      "dtype": "base_q4", "scale_dtype": "bf16", "group_size": 64, "algo": "awq" },
    { "pattern": "lm_head.weight", "dtype": "base_q8" },
    { "pattern": "**.weight", "dtype": "base_q4" },   // catch-all
    { "pattern": "**.bias", "dtype": "f16" }
  ]
}
```

Patterns match the converter's canonical tensor names (`basert inventory
<checkpoint>` prints them for a Hugging Face checkpoint).

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
| `dtype` | yes | `base_q2`…`base_q8`, `mxfp4`, `nvfp4`, `bf16`, `f16`, `f32`. |
| `group_size` | no | Defaults to the dtype's canonical group size (32 for q2/q3, 64 for q4–q6, 128 for q8). |
| `scale_dtype` | no | `bf16` (default) / `f16` / `e8m0` / `e4m3` (q8 only). |
| `symmetric` | no | Default `false` (asymmetric). |
| `algo` | no | `rtn` (default, round to nearest) or `awq`. `awq` applies when the conversion also gets an activation sidecar (`--awq-profile`); otherwise the tensor falls back to `rtn` with a warning. |
| `awq_alpha` | no | A fixed AWQ alpha in `0`–`1` instead of the search. Only with `"algo": "awq"`. |

A profile that names an unknown `algo`, an out-of-range `awq_alpha`, or
`e4m3` scales on anything but `base_q8` is rejected when it loads. See
[Quantization](quantization.md#awq-calibration) for how AWQ runs.

## Tips

- Keep norms, routers, and (often) embeddings at `bf16`/`f16` — they're small
  and precision-sensitive.
- Validate by converting a small model and running `basert inspect` to confirm
  the per-tensor dtypes resolved as intended.
- Keep your own tuned profiles in your own repository and pass them by path.
