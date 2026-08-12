"""
BaseRT -- Python bindings for the BaseRT LLM inference engine (Apple Silicon / Metal).

Usage:
    import baseRT

    with baseRT.Model("model.base") as m:
        for token in m.generate("Hello, world!", max_tokens=128):
            print(token, end="", flush=True)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

__version__ = "0.2.2"

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _lib_names() -> list[str]:
    """Platform-appropriate shared-library filenames, most likely first.

    Metal/macOS is the only shipping backend today, but naming the candidates
    by platform means the loader needs no change when a Linux (CUDA/ROCm) or
    Windows build appears — the C ABI is identical, only the file differs.
    """
    if sys.platform == "darwin":
        return ["libbaseRT.dylib"]
    if sys.platform == "win32":
        return ["baseRT.dll", "libbaseRT.dll"]
    return ["libbaseRT.so"]  # Linux / other Unix


def _find_library() -> str:
    """Locate the BaseRT shared library."""
    env = os.environ.get("BASERT_LIB_PATH")
    if env:
        return env

    # Walk upward from this file to find the project build directory, trying
    # each platform-appropriate filename in each candidate directory.
    here = Path(__file__).resolve().parent
    search_dirs = [
        here / ".." / ".." / ".." / "build",  # bindings/python/baseRT -> build/
        here / ".." / ".." / "build",
        Path("build"),
    ]
    names = _lib_names()
    for d in search_dirs:
        for name in names:
            resolved = (d / name).resolve()
            if resolved.is_file():
                return str(resolved)

    raise OSError(
        f"Cannot find {' / '.join(names)}. Set BASERT_LIB_PATH or build with "
        "`make shared`."
    )


def _load_lib() -> ctypes.CDLL:
    path = _find_library()
    return ctypes.CDLL(path)


_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_lib()
        _setup_signatures(_lib)
    return _lib


# ---------------------------------------------------------------------------
# ctypes struct mirrors
# ---------------------------------------------------------------------------


class BaseRTModelConfig(ctypes.Structure):
    """Mirror of the C BaseRTModelConfig struct."""

    _fields_ = [
        ("dim", ctypes.c_uint32),
        ("n_layers", ctypes.c_uint32),
        ("n_heads", ctypes.c_uint32),
        ("n_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("q_dim", ctypes.c_uint32),
        ("kv_dim", ctypes.c_uint32),
        ("ffn_dim", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("norm_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("sliding_window_pattern", ctypes.c_uint32),
        ("sliding_window", ctypes.c_uint32),
        ("rope_local_theta", ctypes.c_float),
        ("architecture", ctypes.c_char * 32),
        # Encoder (Whisper)
        ("enc_n_layers", ctypes.c_uint32),
        ("enc_n_heads", ctypes.c_uint32),
        ("enc_dim", ctypes.c_uint32),
        ("enc_ffn_dim", ctypes.c_uint32),
        ("n_mels", ctypes.c_uint32),
        ("enc_max_seq_len", ctypes.c_uint32),
        # Gemma 4 specific
        ("n_embd_per_layer", ctypes.c_uint32),
        ("n_layer_kv_from_start", ctypes.c_uint32),
        ("logit_softcap", ctypes.c_float),
        ("attention_scale", ctypes.c_float),
        ("head_dim_swa", ctypes.c_uint32),
        ("head_dim_global", ctypes.c_uint32),
        ("global_rope_partial_factor", ctypes.c_float),
        ("swa_layers", ctypes.c_uint8 * 64),
        ("ffn_dims", ctypes.c_uint32 * 128),
        ("n_kv_heads_per_layer", ctypes.c_uint32 * 128),
        # Qwen3.5 / 3.6 hybrid linear-attention (Gated DeltaNet)
        ("attn_output_gate", ctypes.c_uint8),
        ("_qwen35_pad", ctypes.c_uint8 * 3),
        ("partial_rotary_factor", ctypes.c_float),
        ("full_attention_interval", ctypes.c_uint32),
        ("linear_attn_layers", ctypes.c_uint8 * 64),
        ("gdn_num_k_heads", ctypes.c_uint32),
        ("gdn_num_v_heads", ctypes.c_uint32),
        ("gdn_key_head_dim", ctypes.c_uint32),
        ("gdn_value_head_dim", ctypes.c_uint32),
        ("gdn_conv_kernel", ctypes.c_uint32),
        # Mixture-of-Experts (0 = dense)
        ("n_experts", ctypes.c_uint32),
        ("n_experts_used", ctypes.c_uint32),
        ("n_experts_shared", ctypes.c_uint32),
        ("expert_ffn_dim", ctypes.c_uint32),
        ("expert_gating", ctypes.c_uint8),
        ("norm_topk_prob", ctypes.c_uint8),
        ("_moe_pad", ctypes.c_uint8 * 2),
        # Vision tower
        ("vision_n_layers", ctypes.c_uint32),
        ("vision_dim", ctypes.c_uint32),
        ("vision_n_heads", ctypes.c_uint32),
        ("vision_head_dim", ctypes.c_uint32),
        ("vision_ffn_dim", ctypes.c_uint32),
        ("vision_patch_size", ctypes.c_uint32),
        ("vision_image_size", ctypes.c_uint32),
        ("vision_pooling_kernel", ctypes.c_uint32),
        ("vision_soft_tokens", ctypes.c_uint32),
        ("vision_norm_eps", ctypes.c_float),
        ("vision_rope_theta", ctypes.c_float),
        ("vision_pos_embed_size", ctypes.c_uint32),
        ("image_token_id", ctypes.c_uint32),
        ("boi_token_id", ctypes.c_uint32),
        ("eoi_token_id", ctypes.c_uint32),
        # Vision-tower family selector + Qwen3-VL-style extras
        ("vision_arch", ctypes.c_uint32),
        ("vision_spatial_merge", ctypes.c_uint32),
        ("vision_temporal_patch", ctypes.c_uint32),
        ("vision_out_dim", ctypes.c_uint32),
        # Audio tower
        ("audio_n_layers", ctypes.c_uint32),
        ("audio_dim", ctypes.c_uint32),
        ("audio_n_heads", ctypes.c_uint32),
        ("audio_head_dim", ctypes.c_uint32),
        ("audio_ffn_dim", ctypes.c_uint32),
        ("audio_output_proj_dim", ctypes.c_uint32),
        ("audio_chunk_size", ctypes.c_uint32),
        ("audio_left_context", ctypes.c_uint32),
        ("audio_conv_kernel", ctypes.c_uint32),
        ("audio_soft_tokens", ctypes.c_uint32),
        ("audio_logit_softcap", ctypes.c_float),
        ("audio_norm_eps", ctypes.c_float),
        ("audio_gradient_clip", ctypes.c_float),
        ("audio_residual_weight", ctypes.c_float),
        ("audio_ms_per_token", ctypes.c_float),
        ("audio_sscp_channels", ctypes.c_uint32 * 2),
        ("audio_token_id", ctypes.c_uint32),
        ("boa_token_id", ctypes.c_uint32),
        ("eoa_token_id", ctypes.c_uint32),
        ("mrope_section", ctypes.c_uint32 * 3),
        ("mrope_interleaved", ctypes.c_uint8),
        ("_mrope_pad", ctypes.c_uint8 * 3),
        ("rope_scaling_factor", ctypes.c_float),
        ("rope_low_freq_factor", ctypes.c_float),
        ("rope_high_freq_factor", ctypes.c_float),
        ("rope_orig_max_pos", ctypes.c_uint32),
        ("rope_scaling_type", ctypes.c_uint32),
        # Muse Glimmer (0 / empty = not applicable)
        ("qk_scale_factor", ctypes.c_float),
        ("output_multiplier", ctypes.c_float),
        ("post_norm_eps", ctypes.c_float),
        ("nope_layers", ctypes.c_uint8 * 64),
        ("embed_norm_eps", ctypes.c_float),
        ("vision_window_layers", ctypes.c_uint8 * 64),
        ("vision_window_size", ctypes.c_uint32),
        ("vision_pos_embed_h", ctypes.c_uint32),
        ("vision_pos_embed_w", ctypes.c_uint32),
        ("vision_adapter_dim", ctypes.c_uint32),
        ("video_token_id", ctypes.c_uint32),
    ]


class BaseRTSamplingConfig(ctypes.Structure):
    """Mirror of the C BaseRTSamplingConfig struct.

    Extended in baseRT 0.2 with OpenAI-compat penalties (presence,
    frequency), a deterministic-sample seed, and a per-token logit_bias
    map (parallel arrays). Older callers that only set the first five
    fields keep working — new fields default to "disabled" via
    ctypes' zero-init.
    """

    _fields_ = [
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint32),
        ("n_logit_bias", ctypes.c_int32),
        ("logit_bias_tokens", ctypes.POINTER(ctypes.c_int32)),
        ("logit_bias_values", ctypes.POINTER(ctypes.c_float)),
    ]


class BaseRTGenerationStats(ctypes.Structure):
    """Mirror of the C BaseRTGenerationStats struct."""

    _fields_ = [
        ("prompt_tokens", ctypes.c_int),
        ("generated_tokens", ctypes.c_int),
        ("prefill_time_ms", ctypes.c_float),
        ("decode_time_ms", ctypes.c_float),
        ("prefill_tokens_per_sec", ctypes.c_float),
        ("decode_tokens_per_sec", ctypes.c_float),
    ]


class BaseRTTranscribeStats(ctypes.Structure):
    """Mirror of the C BaseRTTranscribeStats struct."""

    _fields_ = [
        ("n_tokens", ctypes.c_int),
        ("audio_ms", ctypes.c_float),
        ("encode_ms", ctypes.c_float),
        ("decode_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
    ]


class BaseRTTranscribeSegment(ctypes.Structure):
    """Mirror of the C BaseRTTranscribeSegment struct."""

    _fields_ = [
        ("start_ms", ctypes.c_int),
        ("end_ms", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("avg_logprob", ctypes.c_float),
        ("no_speech_prob", ctypes.c_float),
        ("compression_ratio", ctypes.c_float),
        ("temperature", ctypes.c_float),
    ]


# Callback type: bool (*)(uint32_t token_id, const char *text, void *user_data)
BASERT_TOKEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_void_p
)

# Segment callback type: bool (*)(int start_ms, int end_ms, const char *text, void *user_data)
BASERT_SEGMENT_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)

# ---------------------------------------------------------------------------
# Pythonic dataclasses returned to the user
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Decoded model configuration."""

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    q_dim: int
    kv_dim: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    norm_eps: float
    rope_theta: float
    sliding_window_pattern: int
    rope_local_theta: float
    architecture: str
    enc_n_layers: int
    enc_n_heads: int
    enc_dim: int
    enc_ffn_dim: int
    n_mels: int
    enc_max_seq_len: int

    @staticmethod
    def _from_c(c: BaseRTModelConfig) -> "ModelConfig":
        return ModelConfig(
            dim=c.dim,
            n_layers=c.n_layers,
            n_heads=c.n_heads,
            n_kv_heads=c.n_kv_heads,
            head_dim=c.head_dim,
            q_dim=c.q_dim,
            kv_dim=c.kv_dim,
            ffn_dim=c.ffn_dim,
            vocab_size=c.vocab_size,
            max_seq_len=c.max_seq_len,
            norm_eps=c.norm_eps,
            rope_theta=c.rope_theta,
            sliding_window_pattern=c.sliding_window_pattern,
            rope_local_theta=c.rope_local_theta,
            architecture=c.architecture.decode("utf-8").rstrip("\x00"),
            enc_n_layers=c.enc_n_layers,
            enc_n_heads=c.enc_n_heads,
            enc_dim=c.enc_dim,
            enc_ffn_dim=c.enc_ffn_dim,
            n_mels=c.n_mels,
            enc_max_seq_len=c.enc_max_seq_len,
        )


@dataclass
class GenerationStats:
    """Statistics from a generation call."""

    prompt_tokens: int
    generated_tokens: int
    prefill_time_ms: float
    decode_time_ms: float
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float

    @staticmethod
    def _from_c(c: BaseRTGenerationStats) -> "GenerationStats":
        return GenerationStats(
            prompt_tokens=c.prompt_tokens,
            generated_tokens=c.generated_tokens,
            prefill_time_ms=c.prefill_time_ms,
            decode_time_ms=c.decode_time_ms,
            prefill_tokens_per_sec=c.prefill_tokens_per_sec,
            decode_tokens_per_sec=c.decode_tokens_per_sec,
        )


@dataclass
class TranscribeStats:
    """Statistics from a transcription call."""

    n_tokens: int
    audio_ms: float
    encode_ms: float
    decode_ms: float
    total_ms: float

    @staticmethod
    def _from_c(c: BaseRTTranscribeStats) -> "TranscribeStats":
        return TranscribeStats(
            n_tokens=c.n_tokens,
            audio_ms=c.audio_ms,
            encode_ms=c.encode_ms,
            decode_ms=c.decode_ms,
            total_ms=c.total_ms,
        )


@dataclass
class TranscribeSegment:
    """One segment of the last transcription (verbose_json surface).

    avg_logprob / no_speech_prob / compression_ratio / temperature are
    window-level values applied to every segment decoded in that 30 s
    window (the engine's documented approximation).
    """

    start_ms: int
    end_ms: int
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float

    @staticmethod
    def _from_c(c: BaseRTTranscribeSegment) -> "TranscribeSegment":
        return TranscribeSegment(
            start_ms=c.start_ms,
            end_ms=c.end_ms,
            text=c.text.decode("utf-8") if c.text else "",
            avg_logprob=c.avg_logprob,
            no_speech_prob=c.no_speech_prob,
            compression_ratio=c.compression_ratio,
            temperature=c.temperature,
        )


@dataclass
class TensorInfo:
    """Metadata about a single model tensor."""

    index: int
    name: str
    dtype: int
    raw_dtype: str


# ---------------------------------------------------------------------------
# C function signatures
# ---------------------------------------------------------------------------


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes / restype for every C function we call."""

    # Model lifecycle
    lib.baseRT_load_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.baseRT_load_model.restype = ctypes.c_void_p

    lib.baseRT_free_model.argtypes = [ctypes.c_void_p]
    lib.baseRT_free_model.restype = None

    # Model info
    lib.baseRT_get_config.argtypes = [ctypes.c_void_p]
    lib.baseRT_get_config.restype = BaseRTModelConfig

    # BaseRTModelConfig is mirrored by hand above; a size mismatch means the
    # mirror drifted from include/baseRT/types.h and every field after the
    # divergence point decodes as garbage. Fail at import, not at use.
    # (Older libraries predate the symbol; skip the check there.)
    if hasattr(lib, "baseRT_model_config_sizeof"):
        lib.baseRT_model_config_sizeof.argtypes = []
        lib.baseRT_model_config_sizeof.restype = ctypes.c_size_t
        c_size = lib.baseRT_model_config_sizeof()
        py_size = ctypes.sizeof(BaseRTModelConfig)
        if c_size != py_size:
            raise RuntimeError(
                f"BaseRTModelConfig layout drift: library says {c_size} bytes, "
                f"Python mirror is {py_size} bytes. Update the _fields_ list in "
                "bindings/python/baseRT/__init__.py to match include/baseRT/types.h."
            )

    lib.baseRT_model_memory.argtypes = [ctypes.c_void_p]
    lib.baseRT_model_memory.restype = ctypes.c_size_t

    lib.baseRT_get_error.argtypes = []
    lib.baseRT_get_error.restype = ctypes.c_char_p

    # Tokenization
    lib.baseRT_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.baseRT_encode.restype = ctypes.c_int

    lib.baseRT_decode_token.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.baseRT_decode_token.restype = ctypes.c_char_p

    # Generation
    lib.baseRT_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        BaseRTSamplingConfig,
        BASERT_TOKEN_CALLBACK,
        ctypes.c_void_p,
    ]
    lib.baseRT_generate.restype = BaseRTGenerationStats

    lib.baseRT_generate_continue.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        BaseRTSamplingConfig,
        BASERT_TOKEN_CALLBACK,
        ctypes.c_void_p,
    ]
    lib.baseRT_generate_continue.restype = BaseRTGenerationStats

    # Low-level API
    lib.baseRT_prefill.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.baseRT_prefill.restype = ctypes.c_uint32

    # Multimodal
    lib.baseRT_prefill_image.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    lib.baseRT_prefill_image.restype = ctypes.c_uint32

    lib.baseRT_image_num_tokens.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.baseRT_image_num_tokens.restype = ctypes.c_int

    # Audio multimodal
    lib.baseRT_prefill_audio.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.baseRT_prefill_audio.restype = ctypes.c_uint32

    lib.baseRT_audio_num_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.baseRT_audio_num_tokens.restype = ctypes.c_int

    lib.baseRT_decode_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
    ]
    lib.baseRT_decode_step.restype = ctypes.c_uint32

    lib.baseRT_chain_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.baseRT_chain_decode.restype = ctypes.c_int

    lib.baseRT_get_position.argtypes = [ctypes.c_void_p]
    lib.baseRT_get_position.restype = ctypes.c_int

    lib.baseRT_set_speculation.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.baseRT_set_speculation.restype = None

    lib.baseRT_reset.argtypes = [ctypes.c_void_p]
    lib.baseRT_reset.restype = None

    # Whisper transcription
    lib.baseRT_transcribe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(BaseRTTranscribeStats),
    ]
    lib.baseRT_transcribe.restype = ctypes.c_char_p

    lib.baseRT_transcribe_pcm.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(BaseRTTranscribeStats),
    ]
    lib.baseRT_transcribe_pcm.restype = ctypes.c_char_p

    lib.baseRT_set_timestamps.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.baseRT_set_timestamps.restype = None

    lib.baseRT_set_task.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.baseRT_set_task.restype = ctypes.c_bool

    lib.baseRT_set_initial_prompt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.baseRT_set_initial_prompt.restype = None

    lib.baseRT_set_condition_on_previous_text.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    lib.baseRT_set_condition_on_previous_text.restype = None

    lib.baseRT_transcribe_language.argtypes = [ctypes.c_void_p]
    lib.baseRT_transcribe_language.restype = ctypes.c_char_p

    lib.baseRT_transcribe_audio_duration_ms.argtypes = [ctypes.c_void_p]
    lib.baseRT_transcribe_audio_duration_ms.restype = ctypes.c_int

    lib.baseRT_transcribe_segment_count.argtypes = [ctypes.c_void_p]
    lib.baseRT_transcribe_segment_count.restype = ctypes.c_int

    lib.baseRT_transcribe_segment.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(BaseRTTranscribeSegment),
    ]
    lib.baseRT_transcribe_segment.restype = ctypes.c_bool

    lib.baseRT_is_whisper.argtypes = [ctypes.c_void_p]
    lib.baseRT_is_whisper.restype = ctypes.c_bool

    # Streaming transcription
    lib.baseRT_transcribe_pcm_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(BaseRTTranscribeStats),
        BASERT_SEGMENT_CALLBACK,
        ctypes.c_void_p,
    ]
    lib.baseRT_transcribe_pcm_stream.restype = ctypes.c_char_p

    lib.baseRT_transcribe_stream.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(BaseRTTranscribeStats),
        BASERT_SEGMENT_CALLBACK,
        ctypes.c_void_p,
    ]
    lib.baseRT_transcribe_stream.restype = ctypes.c_char_p

    # Embeddings
    lib.baseRT_embed.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.baseRT_embed.restype = ctypes.c_int

    lib.baseRT_embed_text.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.baseRT_embed_text.restype = ctypes.c_int

    lib.baseRT_embedding_dim.argtypes = [ctypes.c_void_p]
    lib.baseRT_embedding_dim.restype = ctypes.c_int

    # Chat templates
    lib.baseRT_format_chat.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.baseRT_format_chat.restype = ctypes.c_char_p

    lib.baseRT_chat_template.argtypes = [ctypes.c_void_p]
    lib.baseRT_chat_template.restype = ctypes.c_char_p

    # Token counting
    lib.baseRT_token_count.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.baseRT_token_count.restype = ctypes.c_int

    # Model inspection
    lib.baseRT_tensor_count.argtypes = [ctypes.c_void_p]
    lib.baseRT_tensor_count.restype = ctypes.c_int

    lib.baseRT_tensor_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.baseRT_tensor_name.restype = ctypes.c_char_p

    lib.baseRT_tensor_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.baseRT_tensor_dtype.restype = ctypes.c_uint32

    lib.baseRT_tensor_raw_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.baseRT_tensor_raw_dtype.restype = ctypes.c_char_p


# ---------------------------------------------------------------------------
# Helper: get last error
# ---------------------------------------------------------------------------


def get_error() -> Optional[str]:
    """Return the last BaseRT error message, or None."""
    lib = _get_lib()
    msg = lib.baseRT_get_error()
    if msg:
        return msg.decode("utf-8")
    return None


class BaseRTError(RuntimeError):
    """Raised when the BaseRT C library reports an error."""

    pass


def _check_model(handle: ctypes.c_void_p) -> None:
    if not handle:
        err = get_error() or "unknown error"
        raise BaseRTError(f"Failed to load model: {err}")


class _ReentrancyGuardedLock:
    """Per-handle serialization lock plus a thread-local re-entrancy guard.

    Wraps the non-reentrant per-handle ``threading.Lock`` that serializes every
    native call touching the single non-thread-safe C handle. On top of that it
    adds a per-instance, thread-local depth counter so a *direct* (non-stream)
    generate/transcribe native call -- which invokes the user's Python callback
    SYNCHRONOUSLY while this lock is held -- cannot self-deadlock:

    * The direct-path callback trampoline calls :meth:`enter_callback` right
      before running the user callback and :meth:`exit_callback` in a ``finally``
      right after. Between those, this thread's marker is > 0.
    * :meth:`acquire` (used by every ``with self._handle_lock`` guarded method)
      checks the marker BEFORE touching the underlying lock. If the marker is
      > 0 -- i.e. the caller is a user callback re-entering a handle method on
      the SAME thread while the native call still holds the lock -- it raises
      :class:`BaseRTError` immediately instead of blocking forever on the
      non-reentrant lock. This turns a self-deadlock into a clear exception and
      never issues a nested native call, so generation is not corrupted.
    * Unrelated threads (marker 0) acquire and block-and-wait exactly as before,
      so the stream-vs-reuse serialization is unchanged. The stream worker's
      token callback only enqueues text (never :meth:`enter_callback`, never a
      handle method), so its thread's marker stays 0 and that path is unaffected.

    The marker is per instance (its own :class:`threading.local`), so being
    inside model A's callback does not reject a concurrent call to model B.

    .. warning::

        The one case the thread-local guard cannot rescue is a callback that
        synchronously hands work to a *different* thread which then calls this
        model -- e.g. ``on_token = lambda t: executor.submit(model.position).result()``
        inside a direct :meth:`Model.generate`. That other thread's marker is 0,
        so it does not raise; it blocks on the busy handle lock while the worker
        thread (parked waiting on the other thread's result) still holds it, and
        the two DEADLOCK. This is NOT eliminated and is NOT a defect: the model
        is single-owner / non-thread-safe and the handle is exclusively busy for
        the callback's duration, so it is a single-owner-contract violation -- do
        NOT call model APIs from a *different* thread that a callback is
        synchronously waiting on. Distinguishing this from an unrelated concurrent
        call is impossible without a model-wide flag, which would wrongly reject
        legitimate concurrent callers (the reason the guard is thread-local, not a
        coarse flag). The supported reentry case -- a same-thread call from within
        the callback -- raises :class:`BaseRTError` cleanly instead.
    """

    _REENTRY_MSG = (
        "Cannot call a model API from within a generation callback "
        "(the native handle is busy for the callback's duration)"
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = threading.local()

    def _get_depth(self) -> int:
        return getattr(self._depth, "value", 0)

    def is_reentrant(self) -> bool:
        """True if THIS thread is currently inside a direct-path user callback.

        Lets a lifecycle method (e.g. :meth:`Model.close`) detect a callback-
        initiated, same-thread re-entry BEFORE it touches the underlying lock --
        so it can reject the call up front, before mutating any state, instead of
        only discovering the re-entry when :meth:`acquire` would deadlock.
        """
        return self._get_depth() > 0

    # Backwards/alternate spelling for callers that prefer the intent name.
    in_callback = is_reentrant

    def enter_callback(self) -> None:
        """Mark this thread as running inside a direct-path user callback."""
        self._depth.value = self._get_depth() + 1

    def exit_callback(self) -> None:
        """Balance :meth:`enter_callback`; always call from a ``finally``."""
        self._depth.value = self._get_depth() - 1

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        # Guard BEFORE acquiring: if this thread is inside a direct-path user
        # callback, the native call already holds the lock on this same thread,
        # so acquiring the non-reentrant lock would deadlock. Fail loudly.
        if self._get_depth() > 0:
            raise BaseRTError(self._REENTRY_MSG)
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_ReentrancyGuardedLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Model:
    """High-level Python wrapper around the BaseRT C API.

    Supports use as a context manager::

        with baseRT.Model("model.base") as m:
            print(m.generate_text("Once upon a time"))
    """

    def __init__(
        self,
        model_path: str,
        kernel_library_path: Optional[str] = None,
        max_context: int = 0,
    ) -> None:
        self._lib = _get_lib()
        # In-flight stream() workers as (cancel_event, thread, done_event)
        # triples, guarded by _streams_lock so close() can cancel them and prove
        # each has actually left its native baseRT_generate call before freeing
        # the C handle. done_event is the authoritative "native call returned,
        # safe to free" signal: the worker sets it ONLY in _run's finally, after
        # baseRT_generate has truly returned. close() WAITS on it rather than
        # trusting thread.join(), which a KeyboardInterrupt can make return
        # spuriously (Thread marked stopped while the native thread still runs).
        self._streams: List[
            Tuple[threading.Event, threading.Thread, threading.Event]
        ] = []
        self._streams_lock = threading.Lock()
        # Per-handle lock serializing every native call that touches the single
        # non-thread-safe C handle (baseRT_generate, prefill, decode, encode,
        # transcribe, embed, info accessors, ...). Mirrors the Swift binding's
        # handleLock. It is only ever held around the actual C call (never a
        # whole method body), so no guarded method nests a second acquire, and
        # the stream worker's token callback merely enqueues tokens -- it never
        # calls a handle method -- so generation cannot re-enter it. Crucially,
        # an abandoned-but-still-running stream worker holds this lock for its
        # whole baseRT_generate, so any reuse of the handle BLOCKS here until
        # that native call returns, preventing overlapping calls on the single-
        # owner handle. close() does NOT take this lock while waiting on a worker
        # (it waits on done_event instead), so cancel+wait cannot deadlock
        # against a worker that needs the lock.
        #
        # It is a _ReentrancyGuardedLock, not a plain Lock: a DIRECT (non-stream)
        # generate/transcribe holds it while running the user's callback
        # synchronously. If that callback calls a handle method on the SAME
        # thread, the non-reentrant lock would self-deadlock; the guard raises
        # BaseRTError instead. Mirrors the Swift binding's re-entrancy handling.
        self._handle_lock = _ReentrancyGuardedLock()
        # Set under _streams_lock in close(); stream() refuses to register new
        # workers once this is set so close() can never miss (or race) one.
        self._closing = False
        # Handle awaiting the native free. close() moves the live handle here
        # (clearing self._handle) and only clears it once baseRT_free_model has
        # actually run. Because the ONLY reference to a not-yet-freed handle then
        # lives on the instance -- never a stack local -- an interrupted close()
        # (e.g. KeyboardInterrupt mid worker-join) leaves the handle reachable
        # for a later close()/__del__ to resume and free exactly once.
        self._pending_free_handle: Optional[ctypes.c_void_p] = None
        # Default before load so a failed load still leaves a well-formed handle
        # attribute for close()/__del__ to inspect.
        self._handle: Optional[ctypes.c_void_p] = None
        mp = model_path.encode("utf-8")
        ml = kernel_library_path.encode("utf-8") if kernel_library_path else None
        self._handle = self._lib.baseRT_load_model(mp, ml, max_context)
        _check_model(self._handle)

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release all GPU resources.

        Cancels any in-flight :meth:`stream` workers and waits for each worker's
        ``done_event`` -- set only after its ``baseRT_generate`` call has truly
        returned -- so the C handle is never freed underneath a live native call.
        The event, not ``thread.join()``, is the authoritative proof a worker has
        stopped: on CPython a ``KeyboardInterrupt`` during ``join()`` can leave
        the ``Thread`` marked stopped while its native thread is still inside
        ``baseRT_generate``, so a resumed ``join()`` would return spuriously and
        we would free under a live call. ``Event.wait()`` cannot: the event is
        set only by the worker, after the native call returns.

        Shutdown is resumable and frees the native handle exactly once, even if
        an earlier ``close()`` was interrupted (e.g. ``KeyboardInterrupt`` while
        waiting on a worker's ``done_event``). The first caller flips ``_closing``
        and moves the live handle onto ``self._pending_free_handle`` (clearing
        ``self._handle``); that instance attribute -- never a stack local --
        stays the sole reference to the not-yet-freed handle. So if this call
        unwinds mid-wait, a later ``close()`` / ``__del__`` re-enters, re-waits on
        the (idempotent, already-set-or-still-running) ``done_event``s, and only
        then completes the free. Because the wait blocks on an unset event, a
        resumed ``close()`` (a) cannot free while any worker's ``done_event`` is
        unset, and (b) eventually proceeds once every event is set. The final
        claim+free runs atomically under the lock: whichever caller finds the
        handle still pending clears it and frees once, and every other concurrent
        or resumed caller then observes ``None`` and returns -- so there is never
        a double free, and the handle is never freed while a worker is inside a
        C call.
        """
        lock = getattr(self, "_streams_lock", None)
        if lock is None:
            return  # __init__ failed before the lock existed; nothing to free
        # Reject a callback-initiated shutdown BEFORE mutating ANY lifecycle
        # state. A direct (non-stream) generate/transcribe runs the user callback
        # synchronously while _handle_lock is held on THIS thread; if that
        # callback calls close(), the old code only tripped the re-entrancy guard
        # once close() reached `with self._handle_lock` far below -- by which time
        # it had already flipped _closing and moved/cleared self._handle, leaving
        # the model half-closed even though the call is rejected. Check the
        # thread-local depth at the TOP, before _closing is set and before the
        # handle is snapshotted/cleared, so a rejected reentrant close() leaves
        # the model fully intact and still usable. Unrelated threads (depth 0) and
        # the normal non-reentrant close path are unaffected.
        handle_lock = getattr(self, "_handle_lock", None)
        if handle_lock is not None and handle_lock.is_reentrant():
            raise BaseRTError(
                "Cannot close a model from within a generation callback "
                "(the native handle is busy for the callback's duration)"
            )
        with lock:
            if not self._closing:
                self._closing = True
                # Move the live handle onto the instance so the only reference
                # to it survives an interrupted join and a later close() can
                # resume freeing it.
                self._pending_free_handle = self._handle
                self._handle = None
            # On first entry this is the handle just moved; on re-entry (resumed
            # after an interrupt) or for a concurrent caller it is whatever is
            # still pending. None means already freed or never allocated.
            handle = self._pending_free_handle
            if handle is None:
                return
            streams = list(self._streams)
        # Cancel + wait outside the lock so each worker's generator can
        # re-acquire it to deregister itself; new stream() calls are already
        # refused by _closing. We set the cancel flag so the native token loop
        # stops promptly (returning False from the callback) and then WAIT ON
        # done_event -- NOT thread.join(). done_event is set only in the worker's
        # finally, after baseRT_generate has actually returned, so waiting on it
        # proves the native call is no longer running. thread.join() is not
        # trusted here: a KeyboardInterrupt during a prior join() can leave the
        # Thread object marked stopped while its native thread is still inside
        # baseRT_generate, so a resumed join() would return at once and we would
        # free the handle underneath a live call. Waiting on done_event is
        # idempotent (an already-set event returns instantly) and safe to resume:
        # if this wait is interrupted, _pending_free_handle is still set, so a
        # later close()/__del__ retries and only frees once every done_event is
        # set. Cancelling first guarantees the wait cannot block forever.
        for cancel, thread, done in streams:
            cancel.set()
            done.wait()
        # Claim and free atomically under BOTH locks. Clearing the pending
        # handle and calling baseRT_free_model in the same critical section
        # leaves no interruptible Python-level gap between them, and guarantees a
        # single native free across concurrent/resumed callers.
        #
        # We take _handle_lock (not just _streams_lock) around the free so we
        # never free underneath an in-flight GUARDED non-worker call. A method
        # like reset() that acquired _handle_lock BEFORE close() cleared
        # self._handle read the (still valid) handle and is running native on
        # it; acquiring _handle_lock here BLOCKS until that call releases, then
        # frees. A method that only acquires _handle_lock AFTER this point sees
        # self._handle is None via _ensure_open and raises instead of using the
        # freed handle. Both orderings are therefore safe.
        #
        # No deadlock: the lock order is always _streams_lock -> _handle_lock
        # (stream() holds _streams_lock and then acquires _handle_lock via
        # encode(); here close() does the same), and nothing ever acquires them
        # in the reverse order -- guarded methods take only _handle_lock, and the
        # stream worker releases _handle_lock before it touches _streams_lock.
        # Workers already released _handle_lock before setting the done_events we
        # waited on above, so none of them holds it now. Blocking on _handle_lock
        # while _pending_free_handle is still set keeps the free resumable: an
        # interrupt here leaves the handle reachable for a later close()/__del__.
        with lock:
            handle = self._pending_free_handle
            if handle is None:
                return  # another caller already claimed and freed it
            with self._handle_lock:
                self._pending_free_handle = None
                self._lib.baseRT_free_model(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # never raise from a finalizer

    def _ensure_open(self) -> None:
        """Raise if the handle has been closed/freed.

        MUST be called while holding ``_handle_lock``, as the very first thing
        inside every guarded method's ``with self._handle_lock`` block, BEFORE
        the native call. ``close()`` clears ``self._handle`` (and sets
        ``_closing``) under ``_streams_lock``, then frees the handle under
        ``_handle_lock``. So a method that was queued on ``_handle_lock`` behind
        an in-flight worker/call while ``close()`` ran will, once it finally
        acquires the lock, observe ``self._handle is None`` here and fail
        cleanly instead of issuing a native call on a freed handle. Reading
        ``self._handle`` under the GIL is atomic, so this check is race-free.
        """
        if self._closing or self._handle is None:
            raise BaseRTError("model is closed")

    # -- model info ----------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        """Return the model configuration."""
        with self._handle_lock:
            self._ensure_open()
            c = self._lib.baseRT_get_config(self._handle)
        return ModelConfig._from_c(c)

    @property
    def memory_bytes(self) -> int:
        """Total GPU memory used by the model in bytes."""
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_model_memory(self._handle)

    @property
    def is_whisper(self) -> bool:
        """True if this is a Whisper (audio) model."""
        with self._handle_lock:
            self._ensure_open()
            return bool(self._lib.baseRT_is_whisper(self._handle))

    @property
    def position(self) -> int:
        """Current KV cache position (number of tokens processed)."""
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_get_position(self._handle)

    # -- tokenization --------------------------------------------------------

    def encode(self, text: str, max_tokens: int = 8192) -> List[int]:
        """Encode text into token IDs.

        Public, lifecycle-checked entry point: acquires ``_handle_lock``,
        validates the handle via :meth:`_ensure_open`, and only then issues the
        native call. There is deliberately NO public ``handle`` parameter -- an
        external caller must never be able to inject a foreign or already-freed
        handle (which would bypass the close/free lifecycle and cause a
        use-after-free). The captured-handle fast path lives in the private
        :meth:`_encode_with_handle` and is only reachable from internal code that
        has already validated the handle under ``_handle_lock``.
        """
        with self._handle_lock:
            self._ensure_open()
            handle = self._handle
            return self._encode_with_handle(text, max_tokens, handle)

    def _encode_with_handle(
        self,
        text: str,
        max_tokens: int,
        handle: ctypes.c_void_p,
    ) -> List[int]:
        """Run ``baseRT_encode`` on an already-validated, lock-held handle.

        PRIVATE: the caller MUST already hold ``_handle_lock`` and must have
        validated ``handle`` (either via :meth:`_ensure_open` or by capturing a
        still-live handle under a lock that blocks ``close()``). This method does
        NOT call :meth:`_ensure_open`, so it must never be exposed on the public
        surface: doing so would let an external caller pass a stale/foreign
        handle and issue a native call on freed memory.
        """
        buf = (ctypes.c_uint32 * max_tokens)()
        n = self._lib.baseRT_encode(handle, text.encode("utf-8"), buf, max_tokens)
        if n < 0:
            raise BaseRTError(get_error() or "encode failed")
        return list(buf[:n])

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its text representation."""
        with self._handle_lock:
            self._ensure_open()
            s = self._lib.baseRT_decode_token(self._handle, token_id)
        return s.decode("utf-8") if s else ""

    # -- sampling config helper ----------------------------------------------

    @staticmethod
    def _make_sampling(
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Tuple[BaseRTSamplingConfig, Any]:
        """Build a sampling config struct.

        Returns the struct plus an opaque "anchor" object that holds the
        backing arrays for ``logit_bias_*`` pointers — callers must keep the
        anchor alive for the duration of the FFI call, otherwise the engine
        dereferences freed memory.
        """
        cfg = BaseRTSamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )
        anchor: Any = None
        if logit_bias:
            n = len(logit_bias)
            toks = (ctypes.c_int32 * n)(*logit_bias.keys())
            vals = (ctypes.c_float * n)(*logit_bias.values())
            cfg.n_logit_bias = n
            cfg.logit_bias_tokens = ctypes.cast(toks, ctypes.POINTER(ctypes.c_int32))
            cfg.logit_bias_values = ctypes.cast(vals, ctypes.POINTER(ctypes.c_float))
            anchor = (toks, vals)
        return cfg, anchor

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        callback: Optional[Callable[[int, str], bool]] = None,
    ) -> GenerationStats:
        """Generate tokens from a prompt.

        Args:
            prompt: Either a string (will be tokenised) or a list of token IDs.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            min_p: Minimum probability threshold.
            repeat_penalty: Repetition penalty.
            callback: Optional function ``(token_id, text) -> bool``.
                      Return False to stop generation.

        Returns:
            GenerationStats with timing information.

        Warning:
            ``callback`` runs synchronously while the native handle is busy and
            ``_handle_lock`` is held on this thread. Do NOT call other methods of
            this model from within it: a same-thread call raises
            :class:`BaseRTError` (the thread-local re-entrancy guard). Delegating
            synchronously to a DIFFERENT thread that then calls this model can
            DEADLOCK -- that thread blocks on the busy handle while this one waits
            on it. That is a single-owner-contract violation, not a defect, and is
            unsupported.
        """
        tokens = self._to_token_array(prompt)
        sampling, _anchor = self._make_sampling(temperature, top_k, top_p, min_p, repeat_penalty)

        # Stash for any exception the user callback (or the re-entrancy guard)
        # raises. ctypes CANNOT propagate a Python exception across the C
        # boundary: it prints the traceback and returns 0/None (False) to C,
        # silently swallowing it while generation may continue. So the trampoline
        # catches EVERYTHING, stashes it, returns the STOP value (False) so
        # baseRT_generate halts promptly, and we re-raise below -- OUTSIDE the
        # ctypes boundary. Mirrors the Rust binding's catch_unwind trampoline.
        cb_exc: List[BaseException] = []

        if callback is not None:
            @BASERT_TOKEN_CALLBACK
            def _cb(tid: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                # This runs synchronously inside baseRT_generate while
                # _handle_lock is held on this thread. Mark the thread as
                # in-callback so any handle method the user callback calls
                # raises instead of self-deadlocking on the non-reentrant lock;
                # always balanced by the finally so the marker never leaks.
                self._handle_lock.enter_callback()
                try:
                    return callback(tid, text.decode("utf-8") if text else "")
                except BaseException as exc:
                    cb_exc.append(exc)
                    return False  # STOP: never let the exception cross ctypes
                finally:
                    self._handle_lock.exit_callback()

            cb = _cb
        else:
            cb = BASERT_TOKEN_CALLBACK(0)  # NULL

        with self._handle_lock:
            self._ensure_open()
            stats_c = self._lib.baseRT_generate(
                self._handle, tokens, len(tokens), max_tokens, sampling, cb, None
            )
        if cb_exc:
            raise cb_exc[0]  # re-raise outside the ctypes boundary
        return GenerationStats._from_c(stats_c)

    def generate_continue(
        self,
        new_tokens: Union[str, List[int]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
        callback: Optional[Callable[[int, str], bool]] = None,
    ) -> GenerationStats:
        """Continue generation from the current KV cache state (multi-turn).

        Warning:
            ``callback`` runs synchronously while the native handle is busy. Do
            NOT call other methods of this model from within it: a same-thread
            call raises :class:`BaseRTError`, and delegating synchronously to a
            DIFFERENT thread that then calls this model can DEADLOCK on the busy
            handle. That is a single-owner-contract violation, not a defect. See
            :meth:`generate`.
        """
        tokens = self._to_token_array(new_tokens)
        sampling, _anchor = self._make_sampling(temperature, top_k, top_p, min_p, repeat_penalty)

        # See generate(): stash any callback/guard exception, STOP native
        # generation (return False), and re-raise outside the ctypes boundary.
        cb_exc: List[BaseException] = []

        if callback is not None:
            @BASERT_TOKEN_CALLBACK
            def _cb(tid: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                # Runs synchronously inside baseRT_generate_continue while
                # _handle_lock is held on this thread -- mark in-callback so a
                # re-entrant handle method raises instead of self-deadlocking.
                self._handle_lock.enter_callback()
                try:
                    return callback(tid, text.decode("utf-8") if text else "")
                except BaseException as exc:
                    cb_exc.append(exc)
                    return False  # STOP: never let the exception cross ctypes
                finally:
                    self._handle_lock.exit_callback()

            cb = _cb
        else:
            cb = BASERT_TOKEN_CALLBACK(0)

        with self._handle_lock:
            self._ensure_open()
            stats_c = self._lib.baseRT_generate_continue(
                self._handle, tokens, len(tokens), max_tokens, sampling, cb, None
            )
        if cb_exc:
            raise cb_exc[0]  # re-raise outside the ctypes boundary
        return GenerationStats._from_c(stats_c)

    def stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
    ) -> Iterator[str]:
        """Stream generated text token-by-token as a Python generator.

        Yields text fragments as they are produced. After the generator is
        exhausted, ``stats`` is available via the generator's ``.stats``
        attribute (only after full consumption).

        Example::

            gen = model.stream("Once upon a time")
            for text in gen:
                print(text, end="", flush=True)
        """
        # Reject a callback-initiated stream start BEFORE registering a worker or
        # touching ANY lifecycle state (_streams / _closing / the handle). A
        # direct (non-stream) generate/transcribe runs the user callback
        # synchronously while _handle_lock is held on THIS thread; if that
        # callback starts a new stream, the worker's _handle_lock.acquire() (or
        # this method's registration) could deadlock or mutate lifecycle state
        # against a concurrent close() before the reentrancy guard would fire.
        # Trip the thread-local re-entrancy marker up front -- consistent with
        # how close() was hardened -- so a reentrant start raises cleanly with no
        # state change. Unrelated threads (depth 0) are unaffected.
        if self._handle_lock.is_reentrant():
            raise BaseRTError(
                "Cannot start a stream from within a generation callback "
                "(the native handle is busy for the callback's duration)"
            )
        q: queue.Queue[Union[str, BaseException, None]] = queue.Queue()
        stats_holder: List[Optional[GenerationStats]] = [None]
        cancel = threading.Event()
        # Set ONLY after baseRT_generate has truly returned (in _run's finally).
        # This is the authoritative "native call has returned, safe to free"
        # signal close() waits on -- robust to a thread.join() that a
        # KeyboardInterrupt made return spuriously.
        done = threading.Event()

        def _run(
            handle: ctypes.c_void_p,
            tokens: ctypes.Array[ctypes.c_uint32],
        ) -> None:
            @BASERT_TOKEN_CALLBACK
            def _cb(tid: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                q.put(text.decode("utf-8") if text else "")
                # Returning False stops generation (consumer gone or close()).
                return not cancel.is_set()

            try:
                sampling, _anchor = self._make_sampling(
                    temperature, top_k, top_p, min_p, repeat_penalty
                )
                # Use the handle captured under _streams_lock at registration
                # time, not self._handle: close() clears self._handle the instant
                # it takes ownership, but waits on done_event before freeing, so
                # the captured handle stays valid for the lifetime of this call.
                #
                # Hold _handle_lock for the ENTIRE native generate. If this
                # generator is abandoned (caller `break`s out of the loop and
                # keeps a reference, so Python does not auto-close it), the
                # worker keeps running until it hits max_tokens; while it does,
                # this lock is held, so any reuse of the model on another thread
                # blocks in its own `with self._handle_lock` until this native
                # call returns -- no overlapping baseRT_generate on the single
                # non-thread-safe handle, no KV/GPU corruption. Released in the
                # `finally` here BEFORE done.set() below, so close()'s wait on
                # done_event never observes a set event while the lock (and thus
                # the native call) is still live.
                self._handle_lock.acquire()
                try:
                    # Re-check cancellation/open state INSIDE the locked window,
                    # right before the native call. This worker may have been
                    # QUEUED on _handle_lock behind another handle call; while it
                    # waited, close() or the consumer could have cancelled it
                    # (cancel.set()) and/or close() could have cleared the handle.
                    # If so, skip baseRT_generate entirely rather than burn GPU
                    # time (prompt prefill + generation to max_tokens) on an
                    # already-cancelled stream and delay later calls. Mirrors the
                    # Swift skip-on-cancel inside the locked window. The outer
                    # finally still fires the sentinel and done event on this
                    # early-out path, so close()/consumers never hang.
                    if cancel.is_set() or self._closing or self._handle is None:
                        return
                    stats_c = self._lib.baseRT_generate(
                        handle, tokens, len(tokens), max_tokens, sampling, _cb, None
                    )
                finally:
                    self._handle_lock.release()
                stats_holder[0] = GenerationStats._from_c(stats_c)
            except BaseException as exc:  # re-raised on the consumer side
                q.put(exc)
            finally:
                q.put(None)  # sentinel: always unblock the consumer
                # baseRT_generate has now returned (normally, via cancel, or by
                # raising) and _handle_lock has been released: the native handle
                # is no longer in use by this worker. Signal close() it is safe
                # to free. Set LAST so the event never fires while the native
                # call -- or the lock guarding it -- could still be live.
                done.set()

        with self._streams_lock:
            # Register and start under the same lock close() uses, so close()
            # never snapshots a half-registered worker: it either sees a
            # started, joinable thread here, or refuses us via _closing below.
            if self._closing:
                raise BaseRTError("Model is closing; cannot start new stream")
            # Capture the live handle under the lock: while _closing is unset,
            # close() has not yet cleared self._handle, so this is guaranteed
            # non-None and stays valid until close() joins this worker.
            handle = self._handle
            # Tokenize a string prompt HERE, on the calling thread, under the
            # lock and against the captured handle -- never inside the worker
            # against self._handle. Otherwise close() on another thread could
            # null self._handle before the worker tokenizes, feeding a null
            # handle into baseRT_encode and crashing. Doing it now means the
            # worker only ever runs baseRT_generate on the captured handle.
            #
            # _to_token_array takes _handle_lock ONLY when it actually tokenizes
            # a string (the sole native call); a pre-tokenized list prompt makes
            # no native call and never touches _handle_lock, so a worker with a
            # list prompt can still register while another handle call holds the
            # lock. Lock order stays _streams_lock -> _handle_lock.
            tokens = self._to_token_array(prompt, handle)
            t = threading.Thread(target=_run, args=(handle, tokens), daemon=True)
            entry = (cancel, t, done)
            t.start()
            self._streams.append(entry)

        error: Optional[BaseException] = None
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    error = item
                    continue  # sentinel follows immediately
                yield item
        finally:
            # Consumer stopped (exhausted, closed early, or errored): signal
            # the worker to halt generation and wait for it to leave the C
            # call before returning. Wait on done_event -- set only after
            # baseRT_generate has truly returned -- so we never deregister the
            # entry while the native call could still be running.
            cancel.set()
            done.wait()
            with self._streams_lock:
                if entry in self._streams:
                    self._streams.remove(entry)
        if error is not None:
            raise error

    def generate_text(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repeat_penalty: float = 1.0,
    ) -> str:
        """Generate and return the full response text as a single string."""
        pieces: List[str] = []

        def _cb(tid: int, text: str) -> bool:
            pieces.append(text)
            return True

        self.generate(
            prompt, max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, _cb
        )
        return "".join(pieces)

    # -- low-level API -------------------------------------------------------

    def prefill(self, tokens: Union[str, List[int]]) -> int:
        """Run prefill, returning the first generated token ID."""
        arr = self._to_token_array(tokens)
        # baseRT_prefill returns the first generated token (argmax of prefill
        # logits) with no 0-sentinel; token id 0 is a valid token, so do not
        # treat it as an error. Errors surface via baseRT_get_error elsewhere.
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_prefill(self._handle, arr, len(arr))

    def prefill_image(self, tokens: Union[str, List[int]], image_path: str) -> int:
        """Multimodal prefill: run vision tower on image, splice features into
        the prompt at image_token_id positions, then run LLM prefill.
        Returns the first generated token ID."""
        arr = self._to_token_array(tokens)
        with self._handle_lock:
            self._ensure_open()
            tid = self._lib.baseRT_prefill_image(
                self._handle, arr, len(arr), image_path.encode("utf-8")
            )
        if tid == 0:
            raise BaseRTError(get_error() or "prefill_image failed")
        return tid

    def image_num_tokens(self, image_path: str) -> int:
        """Return the number of image placeholder tokens the vision tower
        produces for the given image (depends on image dimensions)."""
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_image_num_tokens(
                self._handle, image_path.encode("utf-8")
            )

    def prefill_audio(
        self, tokens: Union[str, List[int]], pcm_samples: List[float]
    ) -> int:
        """Audio prefill: run Conformer encoder on PCM (16kHz mono float32),
        splice features into prompt at audio_token_id positions.
        Returns first generated token ID."""
        arr = self._to_token_array(tokens)
        pcm_arr = (ctypes.c_float * len(pcm_samples))(*pcm_samples)
        with self._handle_lock:
            self._ensure_open()
            tid = self._lib.baseRT_prefill_audio(
                self._handle, arr, len(arr), pcm_arr, len(pcm_samples)
            )
        if tid == 0:
            raise BaseRTError(get_error() or "prefill_audio failed")
        return tid

    def audio_num_tokens(self, n_samples: int) -> int:
        """Return the number of audio placeholder tokens for n_samples of 16kHz PCM."""
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_audio_num_tokens(self._handle, n_samples)

    def decode_step(self, token_id: int, position: int) -> int:
        """Run a single decode step, returning the next token ID."""
        # baseRT_decode_step returns the sampled token ID with no 0-sentinel;
        # token id 0 is a valid token and must not be treated as an error.
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_decode_step(self._handle, token_id, position)

    def chain_decode(
        self, first_token: int, start_position: int, count: int
    ) -> List[int]:
        """Chain decode multiple tokens in one GPU submission."""
        buf = (ctypes.c_uint32 * count)()
        with self._handle_lock:
            self._ensure_open()
            n = self._lib.baseRT_chain_decode(
                self._handle, first_token, start_position, count, buf
            )
        if n < 0:
            raise BaseRTError(get_error() or "chain_decode failed")
        return list(buf[:n])

    def reset(self) -> None:
        """Reset KV cache and internal state."""
        with self._handle_lock:
            self._ensure_open()
            self._lib.baseRT_reset(self._handle)

    def set_speculation(self, enabled: bool) -> None:
        """Enable or disable speculative decoding (n-gram prediction)."""
        with self._handle_lock:
            self._ensure_open()
            self._lib.baseRT_set_speculation(self._handle, enabled)

    # -- whisper transcription -----------------------------------------------

    def transcribe(
        self,
        wav_path: str,
        language: Optional[str] = None,
    ) -> tuple[str, TranscribeStats]:
        """Transcribe a WAV file.

        Returns:
            Tuple of (transcribed_text, stats).
        """
        stats_c = BaseRTTranscribeStats()
        lang = language.encode("utf-8") if language else None
        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_transcribe(
                self._handle, wav_path.encode("utf-8"), lang, ctypes.byref(stats_c)
            )
        if not result:
            raise BaseRTError(get_error() or "transcription failed")
        return result.decode("utf-8"), TranscribeStats._from_c(stats_c)

    def transcribe_pcm(
        self,
        samples: List[float],
        language: Optional[str] = None,
    ) -> tuple[str, TranscribeStats]:
        """Transcribe raw float32 PCM audio (16 kHz, mono).

        Returns:
            Tuple of (transcribed_text, stats).
        """
        n = len(samples)
        arr = (ctypes.c_float * n)(*samples)
        stats_c = BaseRTTranscribeStats()
        lang = language.encode("utf-8") if language else None
        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_transcribe_pcm(
                self._handle, arr, n, lang, ctypes.byref(stats_c)
            )
        if not result:
            raise BaseRTError(get_error() or "transcription failed")
        return result.decode("utf-8"), TranscribeStats._from_c(stats_c)

    def transcribe_stream(
        self,
        wav_path: str,
        language: Optional[str] = None,
        callback: Optional[Callable[[int, int, str], bool]] = None,
    ) -> tuple[str, TranscribeStats]:
        """Transcribe a WAV file with per-segment streaming callback.

        Args:
            wav_path: Path to the WAV file.
            language: Language code (e.g. "en", "auto"). Default: "en".
            callback: Optional function ``(start_ms, end_ms, text) -> bool``.
                      Return False to stop transcription.

        Returns:
            Tuple of (transcribed_text, stats).

        Warning:
            ``callback`` runs synchronously while the native handle is busy. Do
            NOT call other methods of this model from within it: a same-thread
            call raises :class:`BaseRTError`, and delegating synchronously to a
            DIFFERENT thread that then calls this model can DEADLOCK on the busy
            handle. That is a single-owner-contract violation, not a defect. See
            :meth:`generate`.
        """
        stats_c = BaseRTTranscribeStats()
        lang = language.encode("utf-8") if language else None

        # See generate(): the segment callback runs inside the native transcribe
        # call via ctypes, which cannot propagate a Python exception across the C
        # boundary. Stash it, STOP transcription (return False), re-raise after.
        cb_exc: List[BaseException] = []

        if callback is not None:
            @BASERT_SEGMENT_CALLBACK
            def _cb(start_ms: int, end_ms: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                # Runs synchronously inside the native transcribe call while
                # _handle_lock is held on this thread -- mark in-callback so a
                # re-entrant handle method raises instead of self-deadlocking;
                # balanced by the finally so the marker never leaks.
                self._handle_lock.enter_callback()
                try:
                    return callback(start_ms, end_ms, text.decode("utf-8") if text else "")
                except BaseException as exc:
                    cb_exc.append(exc)
                    return False  # STOP: never let the exception cross ctypes
                finally:
                    self._handle_lock.exit_callback()

            cb = _cb
        else:
            cb = BASERT_SEGMENT_CALLBACK(0)  # NULL

        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_transcribe_stream(
                self._handle, wav_path.encode("utf-8"), lang, ctypes.byref(stats_c), cb, None
            )
        if cb_exc:
            raise cb_exc[0]  # re-raise outside the ctypes boundary
        if not result:
            raise BaseRTError(get_error() or "transcription failed")
        return result.decode("utf-8"), TranscribeStats._from_c(stats_c)

    def transcribe_pcm_stream(
        self,
        samples: List[float],
        language: Optional[str] = None,
        callback: Optional[Callable[[int, int, str], bool]] = None,
    ) -> tuple[str, TranscribeStats]:
        """Transcribe raw float32 PCM audio with per-segment streaming callback.

        Args:
            samples: List of float32 PCM samples (16 kHz, mono).
            language: Language code (e.g. "en", "auto"). Default: "en".
            callback: Optional function ``(start_ms, end_ms, text) -> bool``.
                      Return False to stop transcription.

        Returns:
            Tuple of (transcribed_text, stats).

        Warning:
            ``callback`` runs synchronously while the native handle is busy. Do
            NOT call other methods of this model from within it: a same-thread
            call raises :class:`BaseRTError`, and delegating synchronously to a
            DIFFERENT thread that then calls this model can DEADLOCK on the busy
            handle. That is a single-owner-contract violation, not a defect. See
            :meth:`generate`.
        """
        n = len(samples)
        arr = (ctypes.c_float * n)(*samples)
        stats_c = BaseRTTranscribeStats()
        lang = language.encode("utf-8") if language else None

        # See generate(): the segment callback runs inside the native transcribe
        # call via ctypes, which cannot propagate a Python exception across the C
        # boundary. Stash it, STOP transcription (return False), re-raise after.
        cb_exc: List[BaseException] = []

        if callback is not None:
            @BASERT_SEGMENT_CALLBACK
            def _cb(start_ms: int, end_ms: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                # Runs synchronously inside the native transcribe call while
                # _handle_lock is held on this thread -- mark in-callback so a
                # re-entrant handle method raises instead of self-deadlocking;
                # balanced by the finally so the marker never leaks.
                self._handle_lock.enter_callback()
                try:
                    return callback(start_ms, end_ms, text.decode("utf-8") if text else "")
                except BaseException as exc:
                    cb_exc.append(exc)
                    return False  # STOP: never let the exception cross ctypes
                finally:
                    self._handle_lock.exit_callback()

            cb = _cb
        else:
            cb = BASERT_SEGMENT_CALLBACK(0)  # NULL

        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_transcribe_pcm_stream(
                self._handle, arr, n, lang, ctypes.byref(stats_c), cb, None
            )
        if cb_exc:
            raise cb_exc[0]  # re-raise outside the ctypes boundary
        if not result:
            raise BaseRTError(get_error() or "transcription failed")
        return result.decode("utf-8"), TranscribeStats._from_c(stats_c)

    def set_timestamps(self, enabled: bool) -> None:
        """Enable or disable timestamp generation for Whisper transcription."""
        with self._handle_lock:
            self._ensure_open()
            self._lib.baseRT_set_timestamps(self._handle, enabled)

    def set_task(self, task: str) -> None:
        """Set the Whisper task: "transcribe" (default) or "translate".

        Raises BaseRTError on an unknown task or on "translate" with an
        English-only model.
        """
        with self._handle_lock:
            self._ensure_open()
            if not self._lib.baseRT_set_task(self._handle, task.encode("utf-8")):
                raise BaseRTError(get_error() or "baseRT_set_task failed")

    def set_initial_prompt(self, text: Optional[str]) -> None:
        """Set an initial prompt to bias Whisper decoding (None clears it)."""
        with self._handle_lock:
            self._ensure_open()
            self._lib.baseRT_set_initial_prompt(
                self._handle, text.encode("utf-8") if text else None
            )

    def set_condition_on_previous_text(self, enabled: bool) -> None:
        """Condition each 30s window on previous decoded text (default True)."""
        with self._handle_lock:
            self._ensure_open()
            self._lib.baseRT_set_condition_on_previous_text(self._handle, enabled)

    def transcribe_language(self) -> str:
        """Language code of the last transcription (detected or requested)."""
        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_transcribe_language(self._handle)
            return result.decode("utf-8") if result else ""

    def transcribe_audio_duration_ms(self) -> int:
        """Duration of the last transcription's source audio in milliseconds.

        The OpenAI verbose_json ``duration`` field is this value in
        fractional seconds. 0 if no transcription has run.
        """
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_transcribe_audio_duration_ms(self._handle)

    def transcribe_segments(self) -> List[TranscribeSegment]:
        """Per-segment metadata for the last transcription.

        Returns a list of :class:`TranscribeSegment` (start/end timestamps,
        text, and the verbose_json quality metrics). Empty if no
        transcription has run. Segment text is copied out, so the returned
        list stays valid after later transcriptions.
        """
        with self._handle_lock:
            self._ensure_open()
            n = self._lib.baseRT_transcribe_segment_count(self._handle)
            segments: List[TranscribeSegment] = []
            for i in range(n):
                seg_c = BaseRTTranscribeSegment()
                if not self._lib.baseRT_transcribe_segment(self._handle, i, ctypes.byref(seg_c)):
                    break
                segments.append(TranscribeSegment._from_c(seg_c))
            return segments

    # -- embeddings ----------------------------------------------------------

    def embed(self, tokens: List[int], max_dims: int = 0) -> List[float]:
        """Compute embeddings from token IDs.

        Args:
            tokens: List of token IDs.
            max_dims: Maximum embedding dimensions (0 = use model default).

        Returns:
            List of float embedding values.
        """
        # baseRT_embedding_dim + baseRT_embed both touch the handle; hold the
        # lock across both so the sizing query and the embed cannot interleave
        # with another handle user. Direct _lib calls (not the guarded
        # embedding_dim property) so there is no nested re-acquire.
        with self._handle_lock:
            self._ensure_open()
            # Capture the handle once under the lock. ctypes releases the GIL
            # during each native call, so a concurrent close() could clear the
            # self._handle attribute between the sizing query and the embed; a
            # captured local stays valid for both. close() can only FREE under
            # _handle_lock (held here), so the free waits until we release.
            handle = self._handle
            if max_dims <= 0:
                max_dims = self._lib.baseRT_embedding_dim(handle)
            arr = (ctypes.c_uint32 * len(tokens))(*tokens)
            out = (ctypes.c_float * max_dims)()
            dim = self._lib.baseRT_embed(handle, arr, len(tokens), out, max_dims)
        if dim <= 0:
            raise BaseRTError(get_error() or "embed failed")
        return list(out[:dim])

    def embed_text(self, text: str, max_dims: int = 0) -> List[float]:
        """Compute embeddings from text directly.

        Args:
            text: Input text to embed.
            max_dims: Maximum embedding dimensions (0 = use model default).

        Returns:
            List of float embedding values.
        """
        with self._handle_lock:
            self._ensure_open()
            # Capture once: the sizing query and the embed are two native calls
            # under one lock, and each releases the GIL. A captured local keeps
            # the handle stable across both even if a concurrent close() clears
            # the self._handle attribute (its free still waits on _handle_lock).
            handle = self._handle
            if max_dims <= 0:
                max_dims = self._lib.baseRT_embedding_dim(handle)
            out = (ctypes.c_float * max_dims)()
            dim = self._lib.baseRT_embed_text(
                handle, text.encode("utf-8"), out, max_dims
            )
        if dim <= 0:
            raise BaseRTError(get_error() or "embed_text failed")
        return list(out[:dim])

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for this model."""
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_embedding_dim(self._handle)

    # -- chat templates ------------------------------------------------------

    def format_chat(self, system: str, user: str) -> str:
        """Format a chat prompt using the model's native template.

        Args:
            system: System prompt.
            user: User message.

        Returns:
            Formatted chat string.
        """
        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_format_chat(
                self._handle, system.encode("utf-8"), user.encode("utf-8")
            )
        if not result:
            raise BaseRTError(get_error() or "format_chat failed")
        return result.decode("utf-8")

    @property
    def chat_template(self) -> str:
        """Get the chat template name for the loaded model."""
        with self._handle_lock:
            self._ensure_open()
            result = self._lib.baseRT_chat_template(self._handle)
        return result.decode("utf-8") if result else ""

    # -- token counting ------------------------------------------------------

    def token_count(self, text: str) -> int:
        """Count tokens in text without allocating an output buffer.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        with self._handle_lock:
            self._ensure_open()
            return self._lib.baseRT_token_count(self._handle, text.encode("utf-8"))

    # -- model inspection ----------------------------------------------------

    def tensors(self) -> List[TensorInfo]:
        """Return metadata for every tensor in the model."""
        # Hold the lock across the whole enumeration: count and the per-index
        # name/dtype lookups must observe a consistent handle state, and none of
        # these are guarded methods, so there is no nested re-acquire.
        result: List[TensorInfo] = []
        with self._handle_lock:
            self._ensure_open()
            # Capture once: this enumerates many native calls (count + per-index
            # name/raw_dtype/dtype), each releasing the GIL. A concurrent close()
            # could clear the self._handle attribute mid-loop; the captured local
            # keeps every call on the same valid handle. The actual free waits on
            # _handle_lock, which we hold for the whole enumeration.
            handle = self._handle
            count = self._lib.baseRT_tensor_count(handle)
            for i in range(count):
                name_b = self._lib.baseRT_tensor_name(handle, i)
                raw_b = self._lib.baseRT_tensor_raw_dtype(handle, i)
                result.append(
                    TensorInfo(
                        index=i,
                        name=name_b.decode("utf-8") if name_b else "",
                        dtype=self._lib.baseRT_tensor_dtype(handle, i),
                        raw_dtype=raw_b.decode("utf-8") if raw_b else "",
                    )
                )
        return result

    # -- helpers -------------------------------------------------------------

    def _to_token_array(
        self,
        prompt: Union[str, List[int]],
        handle: Optional[ctypes.c_void_p] = None,
    ) -> ctypes.Array[ctypes.c_uint32]:
        """Coerce a prompt into a native token array.

        When ``handle`` is ``None`` (the common path) tokenization of a string
        prompt goes through the public, lifecycle-checked :meth:`encode`. When an
        explicit ``handle`` is supplied the CALLER MUST have validated that
        handle under a lock that blocks ``close()`` from freeing it -- the stream
        worker captures a still-live handle under ``_streams_lock`` while
        ``_closing`` is unset. ``_handle_lock`` is taken HERE, but only around the
        native tokenize of a string prompt (the private
        :meth:`_encode_with_handle`); a pre-tokenized list prompt issues no
        native call and never touches ``_handle_lock``. No public code path can
        supply a ``handle``, so no stale/foreign handle can enter here.
        """
        if isinstance(prompt, str):
            if handle is None:
                ids = self.encode(prompt)
            else:
                # Lock order _streams_lock -> _handle_lock (caller holds the
                # former); mirrors the original encode(), which acquired
                # _handle_lock internally around the same native call.
                with self._handle_lock:
                    ids = self._encode_with_handle(prompt, 8192, handle)
        else:
            ids = prompt
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        return arr

    def __repr__(self) -> str:
        if self._handle:
            cfg = self.config
            mem_mb = self.memory_bytes / (1024 * 1024)
            return (
                f"<baseRT.Model arch={cfg.architecture!r} "
                f"params={cfg.dim}d/{cfg.n_layers}L/{cfg.n_heads}H "
                f"vocab={cfg.vocab_size} mem={mem_mb:.0f}MB>"
            )
        return "<baseRT.Model (closed)>"
