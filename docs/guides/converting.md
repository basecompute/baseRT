# Converting models

`basert convert` turns a local source checkpoint into a `.base` file. Sources
are **GGUF** files and **Hugging Face** or **MLX** safetensors directories. By
default every source is dequantized to f32 and re-packed through the canonical
quantization path; the [passthrough options](#keeping-source-quantization)
keep a source's own quantized weights instead.

To convert a Hugging Face repo without downloading it yourself, use
`basert pull <org/model>` (see [Managing models](models.md)).

## Supported architectures

Hugging Face and MLX checkpoints are recognized by the `model_type` in their
`config.json`:

| Family | `model_type` |
| --- | --- |
| Llama, Mistral | `llama`, `mistral` |
| Phi-3 | `phi3` (checkpoints with LongRoPE scaling, such as Phi-3.5, are refused) |
| Qwen 2, Qwen 3 | `qwen2`, `qwen3` |
| Qwen 2 / Qwen 3 MoE | `qwen2_moe`, `qwen3_moe` |
| Qwen 3.5, 3.6, 3.8 (hybrid, dense and MoE) | `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text` |
| Gemma, Gemma 2, Gemma 3 | `gemma`, `gemma2`, `gemma3`, `gemma3_text` |
| Gemma 4 | `gemma4`, `gemma4_text`, `gemma4_unified` |
| gpt-oss | `gpt_oss` |
| Nemotron-H (Mamba-2 hybrid, dense and MoE) | `nemotron_h` |
| GLM 5.2 | `glm_moe_dsa` |
| Muse Glimmer | `muse_glimmer`, `muse_glimmer_text` |
| Nomic BERT (embeddings) | `nomic_bert` |
| Whisper (speech-to-text) | `whisper` |

Speculative-decoding drafters convert too: DFlash, DFlash 2 and DSpark block
drafters, EAGLE-3 heads, and Gemma 4's "assistant" MTP heads.

GGUF files are recognized by `general.architecture`: `llama`, `qwen2`,
`qwen3`, `qwen35`, `qwen36`, their `moe` forms, `gemma`, `gemma2`, `gemma3`,
`gemma4`, `nemotron_h`, `nemotron_h_moe`, `nomic-bert`, `glm-dsa` and
`muse-glimmer`.

A vision or audio tower in a Hugging Face or MLX checkpoint is converted into
the same `.base` file. For GGUF, see [Vision towers from GGUF](#vision-towers-from-gguf).

## Basic conversion

```sh
basert convert ./path/to/checkpoint \
    --target base-q4 \
    --output models/my-model.base
```

`--target` selects a quant scheme: `base-q2`, `base-q3`, `base-q4` (the
default), `base-q5`, `base-q6`, `base-q8`, `bf16`, `mxfp4` or `nvfp4`. The
schemes follow the [canonical quant spec](../reference/quantization.md).
Without `--output`, the file is written beside the input as `<input>.base`.

## Profile-driven conversion

For per-tensor control, pass a **profile** instead of a single target. The
generic profiles ship in `base-convert/profiles/`: `default-q4`, `default-q6`,
`default-q8`, and variants of them that change the scale dtype or the
embedding precision.

```sh
basert convert ./checkpoint \
    --profile base-convert/profiles/default-q4.json \
    --output models/my-model.base
```

With `--profile`, per-tensor quant decisions come from the profile's rules;
`--target` becomes the fallback for tensors the profile's catch-all rule doesn't
cover. The profile name is recorded in the file's header (`quant_profile`) for
audit. Write your own; see [Quant profiles](../reference/profiles.md).

### Whisper

Whisper converts to f16 unless a Whisper profile is given; a bare `--target` is
ignored. To quantize its linear projections:

```sh
basert convert ./whisper-large-v3-turbo \
    --profile base-convert/profiles/whisper-q8.json \
    --output models/whisper-large-v3-turbo-q8.base
```

`whisper-q4.json` and `whisper-f16.json` are the other two.

## AWQ calibration

Activation-aware weight quantization (AWQ) improves low-bit quality. The
converter runs AWQ's scale search itself and folds the scales into the model's
own weights, so the result is an ordinary `.base` file that needs no special
kernel.

AWQ needs an activation-stats sidecar for the checkpoint and a profile whose
rules ask for it with `"algo": "awq"` (see [Quant profiles](../reference/profiles.md)):

```sh
basert convert ./checkpoint \
    --profile my-awq-profile.json \
    --awq-profile awq_profile.json \
    --output models/my-model.base
```

Tensors whose rule has `"algo": "awq"` and a `base_qN` dtype run the AWQ search
before they are packed. Every other tensor is quantized as its rule says,
without AWQ; the generic `default-*` profiles ask for none.

The sidecar is written by the engine's calibration mode in the C API
(`baseRT_calibrate_begin`, `baseRT_calibrate_prefill`, `baseRT_calibrate_save`;
see [`baseRT.h`](https://github.com/basecompute/baseRT/blob/main/include/baseRT/baseRT.h)).
It records a fingerprint of the source weights, and the converter refuses a
sidecar whose fingerprint does not match the checkpoint. Three commands help
with this:

| Command | What it prints |
| --- | --- |
| `basert fingerprint <checkpoint>` | The fingerprint a sidecar for this checkpoint must carry. |
| `basert inventory <checkpoint>` | The canonical tensor names and the AWQ fold groups the converter would form (JSON). |
| `basert awq-search <checkpoint> --awq-profile <sidecar> --tensors <names> --cell <bits:group>` | The alpha AWQ picks for the named fold groups, with its objective curve. |

`--imatrix` uses the same sidecar differently: instead of rescaling weights, it
weights each group's quantization fit by the channels' activation size, so
the error lands away from salient channels.

## Keeping source quantization

Some sources are already quantized in a form BaseRT runs directly. These
options copy those weights instead of dequantizing and re-packing them, which
adds no error.

- **GGUF k-quants and i-quants.** `--kquant-passthrough` copies Q2_K, Q3_K,
  Q4_K, Q5_K, Q6_K, IQ4_XS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S blocks into the
  `.base` file as they are, for the engine's k-quant kernels to read. On
  mixture-of-experts models, Q4_K, Q5_K and IQ4_XS expert weights are kept
  too. Other tensors in the file (f16/f32 norms, Q4_0/Q8_0 weights) take the
  normal path. A file with weights this build cannot run is refused when it is
  loaded.
- **MLX 4-bit.** `--target base-q4` from an MLX 4-bit checkpoint with groups of
  64 copies the codes verbatim, since the two schemes are identical.
  `--no-mlx-passthrough` requantizes instead. `--mlx-passthrough` goes further
  and copies every quantized payload as it is (no `--profile`); add
  `--validate` to re-open the result and byte-compare it against the source.
- **gpt-oss.** The MXFP4 expert weights are carried across verbatim.

```sh
basert convert ./model-Q4_K_M.gguf --kquant-passthrough --output models/my-model.base
```

> [!WARNING]
> **Quantizing from already-quantized inputs**
>
> The spec expects fp16/bf16/fp32 sources. With `--profile`, a GGUF whose
> weights are quantized (Q4_K_M, Q8_0, …) or an MLX 4-bit/8-bit checkpoint is
> refused, because re-quantizing compounds error. Passthrough tensors are
> exempt. `--allow-quant-from-quant` accepts the extra error, for when you
> don't have the fp16 checkpoint.

## Vision towers from GGUF

llama.cpp ships a model's vision tower as a separate `mmproj-*.gguf` file, so a
GGUF conversion is text-only by default. `--mmproj` folds the tower into the
same `.base` file:

```sh
basert convert ./model.gguf --mmproj ./mmproj-model.gguf --output models/my-vlm.base
```

The mmproj file does not carry every multimodal setting (image token ids,
pooling, soft-token count). Built-in defaults per projector type fill them in,
and every assumed value is printed. `--mmproj-config <config.json>` takes them
from the original Hugging Face config instead.

## Speculative-decoding heads

A model file can carry its own speculator head, which `basert serve` drafts
with. For Gemma 4, Google publishes the head as a separate "assistant"
checkpoint (`google/gemma-4-<size>-it-assistant`). Convert it on its own, then
embed it:

```sh
basert convert ./gemma-4-E4B-it-assistant --output head.base
basert add-speculator models/gemma-4-E4B-it.base --head head.base -o models/gemma-4-E4B-it-mtp.base
```

`basert convert ... --mtp-head head.base` does the same in one step for a new
conversion. `--quantize-drafter --target base-q8` quantizes a drafter's or
head's own layers instead of keeping it f16. `--head-version` (or
`--mtp-head-version`) records a version for the head.

## Mixed-precision tools

These commands build a model file tensor by tensor:

| Command | What it does |
| --- | --- |
| `basert overlay <checkpoint> --spec <spec.json> -o <overlay.base>` | Quantizes exactly the tensors a spec names into a small `.base` file. |
| `basert merge <base.base> <overlay.base>... -o <out.base>` | Splices overlay files into a model file; later overlays win. `--lm-only` leaves out the vision/audio towers and speculator head. |
| `basert fakequant <checkpoint> --tensor <name> --dtype <dtype> --out <file>` | Writes one tensor's dequantized f32 values at a given format (or, with `--bundle`, as an existing `.base` file stores it). |

Run `basert <command> --help` for the spec format and every option.

## Common flags

| Flag | Meaning |
| --- | --- |
| `-o, --output <path>` | Output `.base` file (defaults to `<input>.base`). |
| `--target <scheme>` | Quant scheme (or fallback when `--profile` is set). Default `base-q4`. |
| `--profile <path>` | Per-tensor canonical-quant profile JSON. |
| `--awq-profile <path>` | AWQ activation-stats sidecar (needs `--profile`). |
| `--awq-search-rows <N>` | Score N evenly spaced rows per tensor in the AWQ search (`0` = all rows). |
| `--imatrix` | Importance-weighted rounding from the `--awq-profile` sidecar. |
| `--kquant-passthrough` | GGUF only: keep k-quant and i-quant blocks as they are. |
| `--mmproj <path>` | GGUF only: fold an `mmproj-*.gguf` vision tower into the file. |
| `--mlx-passthrough` | MLX 4-bit only: copy every quantized payload verbatim. |
| `--allow-quant-from-quant` | Allow `--profile` on already-quantized inputs. |
| `--mtp-head <head.base>` | Embed a converted Gemma 4 MTP head. |
| `--quantize-drafter` | Quantize a drafter's or head's own layers with `--target`. |
| `--direct-write` | Write straight into the output file, so peak disk use is the file size rather than twice it. For very large models. |
| `--synthetic` | Generate a dummy file (CI/testing; no input is read). |

`basert convert --help` lists the rest.

## Inspecting the result

```sh
basert inspect models/my-model.base                     # header + tensor inventory + slots
basert inspect models/my-model.base --verify-checksums  # also verify per-tensor xxhash64 (slow)
```
