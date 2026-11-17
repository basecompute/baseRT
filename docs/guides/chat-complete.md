# Chat & completion

Three runtime tools cover interactive generation, one-shot generation and
speech-to-text. Each takes a hub id (`basecompute/Qwen3-8B`, optionally with
`:variant` or `--variant`) or a path to a `.base` file. A hub id that is not
installed yet is pulled first; see [Managing models](models.md).

## Interactive chat

```sh
basert chat <model> [options]
```

Drops you into a REPL. `/clear` resets the conversation; `/quit`, `quit` or
`exit` leaves. The banner shows the model's architecture and what it takes
(text, vision, audio), and each response prints prefill/decode throughput.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--temp <F>` | `0.7` | Sampling temperature. |
| `--top-k <N>` | `40` | Top-K candidates. |
| `--top-p <F>` | `0.9` | Nucleus sampling threshold. |
| `--min-p <F>` | `0.0` | Min-P filter. |
| `--repeat-penalty <F>` | `1.1` | Repetition penalty. |
| `--max-tokens <N>` | `2048` | Max tokens per response. |
| `--max-context <N>` | auto | Context window size. See [Context window](#context-window). |
| `--think` / `--no-think` | `--no-think` | Show the model's reasoning, or disable thinking mode. |
| `--kv-bits 4\|8\|16\|84` | auto | KV cache element width. See [KV-cache width](#kv-cache-width). |
| `--paged-kv` | off | Paged KV cache + block-table dispatch. |
| `--max-batch-size <N>` | `1` | Max concurrent sequences for batched decode. |
| `--no-ane` | off | Prefill on the GPU alone (no Neural Engine split). |
| `-v, --verbose` | off | Show engine diagnostics (load, GPU, dispatch table). |

```sh
basert chat basecompute/Qwen3-8B --temp 0.8 --top-p 0.95 --think
```

`--think` works with any model whose chat template has a thinking switch, such
as Qwen 3 and Gemma 4.

## One-shot completion

```sh
basert complete <model> --prompt <text> [options]
```

Runs a single prompt and prints the result, for scripts and pipelines.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--prompt <text>` | — | Prompt text; `-` reads stdin. Required unless `--prompt-file`. |
| `--prompt-file <path>` | — | Read the prompt from a file. |
| `--system <text>` | — | System prompt (only with `--chat`). |
| `--chat` | off | Wrap the prompt with the model's chat template. Recommended for instruct models. |
| `--image <path>` | — | Attach an image (vision models). |
| `--audio <path>` | — | Attach a WAV file (audio models). |
| `-n, --max-tokens <N>` | `2048` | Max tokens to generate. |
| `--temp <F>` | `0.0` | Temperature (0 = greedy). |
| `--top-k <N>` | `40` | Top-K. |
| `--top-p <F>` | `0.9` | Top-P. |
| `--repeat-penalty <F>` | `1.0` | Repetition penalty. |
| `--spec` | off | Speculative decoding with the head the model file carries. |
| `--max-context <N>` | auto | KV-cache context budget. |
| `--kv-bits 4\|8\|16\|84` | auto | KV cache element width. |
| `--paged-kv` | off | Paged KV cache. |
| `--max-batch-size <N>` | `1` | Max concurrent sequences for batched decode. |
| `--no-ane` | off | Prefill on the GPU alone. |
| `-v, --verbose` | off | Show engine diagnostics. |

```sh
# plain completion
basert complete basecompute/Qwen3-8B --prompt "List three Metal GPU tips." -n 128

# chat-templated, with a system prompt
basert complete basecompute/Qwen3-8B --chat \
  --system "You are concise." --prompt "Why is RoPE useful?"

# prompt from stdin
echo "The capital of France is" | basert complete basecompute/Qwen3-8B --prompt -
```

### Images and audio

`--image` and `--audio` need the model's marker in the prompt, where the image
or audio goes. On Gemma 4 the markers are `<|image|>` and `<|audio|>`:

```sh
basert complete basecompute/gemma-4-E2B-it --chat \
  --image ./photo.png --prompt "<|image|>Describe this image."
```

Audio is read from WAV (16-bit PCM or 32-bit float, any channel count) and
resampled to 16 kHz.

## Speech-to-text

`basert transcribe` runs a Whisper model over a WAV file and prints the text.
The catalog has every Whisper size, from `basecompute/whisper-tiny` to
`basecompute/whisper-large-v3-turbo`, in f16, Q8 and Q4.

```sh
basert transcribe basecompute/whisper-large-v3-turbo ./meeting.wav
basert transcribe basecompute/whisper-small ./interview.wav --lang auto
basert transcribe basecompute/whisper-medium ./talk-fr.wav --task translate
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--lang <code>` | `en` | Language code, or `auto` to detect it. `auto` when `--task translate` is given without `--lang`. |
| `--task <task>` | `transcribe` | `transcribe`, or `translate` (any language to English). |
| `--prompt <text>` | — | Initial prompt to bias the decode (names, style, spellings). |
| `--no-condition-on-prev` | off | Don't condition each window on the previous text. |
| `--no-timestamps` | off | Disable timestamps (faster, no seeking). |

The transcript goes to stdout; the detected language and timings go to stderr.
The `.en` models (`basecompute/whisper-small.en`, …) are English-only.

The server exposes the same models at `/v1/audio/transcriptions`; see
[Serving an API](serving.md).

## Context window

When `--max-context` is not given, the window is sized for this machine: the
GPU memory budget, less the weights, over what a token of KV cache costs,
capped at the model's trained window. Pin `--max-context` on a Mac that shares
its GPU with another program, since the measurement can't see another process's
memory.

## KV-cache width

`--kv-bits` trades memory for precision: `16` (f16) is highest quality, `8` and
`4` shrink the cache so you can fit longer contexts or more concurrent
sequences, and `84` stores keys at 8 bits and values at 4. The default is
chosen per model. Pair with `--paged-kv` for block-table-based allocation.

## Neural Engine prefill

On M4-, M5- and M6-generation Macs, prompt processing for large dense models
and the Qwen 3.5, 3.6 and 3.8 hybrids is split between the GPU and the Neural
Engine; smaller dense models prefill on the GPU alone. The first load of such a model compiles
its Neural Engine programs in the background, and prefill runs on the GPU alone
until they are ready. `--no-ane` keeps prefill on the GPU.
