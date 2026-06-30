# Chat & completion

Two runtime tools cover interactive and one-shot generation.

## Interactive chat

```sh
basert chat <model> [options]
```

Drops you into a REPL. `/clear` resets the conversation; `quit` exits. Each
response prints prefill/decode throughput.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--temp <F>` | `0.7` | Sampling temperature. |
| `--top-k <N>` | `40` | Top-K candidates. |
| `--top-p <F>` | `0.9` | Nucleus sampling threshold. |
| `--min-p <F>` | `0.0` | Min-P filter. |
| `--repeat-penalty <F>` | `1.1` | Repetition penalty. |
| `--max-tokens <N>` | `256` | Max tokens per response. |
| `--max-context <N>` | `4096` | Context window size. |
| `--think` / `--no-think` | on | Show / hide `<think>` blocks (Qwen3 thinking mode). |
| `--kv-bits 4\|8\|16` | auto | KV cache element width. |
| `--paged-kv` | off | Paged KV cache + block-table dispatch. |
| `--max-batch-size <N>` | `1` | Max concurrent sequences for batched decode. |

```sh
basert chat Qwen/Qwen3-4B --temp 0.8 --top-p 0.95 --max-context 8192 --no-think
```

## One-shot completion

```sh
basert complete <model> --prompt <text> [options]
```

Runs a single prompt and prints the result — ideal for scripts and pipelines.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--prompt <text>` | — | Prompt text (required). |
| `--system <text>` | — | System prompt (only with `--chat`). |
| `--chat` | off | Wrap `--prompt` with the model's chat template. |
| `--image <path>` | — | Attach an image (multimodal models). |
| `--audio <path>` | — | Attach audio (multimodal models). |
| `--max-tokens <N>` | `256` | Max tokens to generate. |
| `--temp <F>` | `0.0` | Temperature (0 = greedy). |
| `--top-k <N>` | `40` | Top-K. |
| `--top-p <F>` | `0.9` | Top-P. |
| `--repeat-penalty <F>` | `1.0` | Repetition penalty. |
| `--kv-bits 4\|8\|16` | auto | KV cache element width. |
| `--paged-kv` | off | Paged KV cache. |

```sh
# plain completion
basert complete Qwen/Qwen3-4B --prompt "List three Metal GPU tips." --max-tokens 128

# chat-templated, with a system prompt
basert complete Qwen/Qwen3-4B --chat \
  --system "You are concise." --prompt "Why is RoPE useful?"

# multimodal
basert complete <vlm-model> --image ./photo.png --prompt "Describe this image."
```

## KV-cache width

`--kv-bits` trades memory for precision: `16` (f16) is highest quality, `8` and
`4` shrink the cache so you can fit longer contexts or more concurrent
sequences. The default is chosen per model. Pair with `--paged-kv` for
block-table-based allocation.
