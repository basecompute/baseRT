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

__version__ = "0.1.0"

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
        """Release all GPU resources."""
        if getattr(self, "_handle", None):
            self._lib.baseRT_free_model(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    # -- model info ----------------------------------------------------------

    @property
    def config(self) -> ModelConfig:
        """Return the model configuration."""
        c = self._lib.baseRT_get_config(self._handle)
        return ModelConfig._from_c(c)

    @property
    def memory_bytes(self) -> int:
        """Total GPU memory used by the model in bytes."""
        return self._lib.baseRT_model_memory(self._handle)

    @property
    def is_whisper(self) -> bool:
        """True if this is a Whisper (audio) model."""
        return bool(self._lib.baseRT_is_whisper(self._handle))

    @property
    def position(self) -> int:
        """Current KV cache position (number of tokens processed)."""
        return self._lib.baseRT_get_position(self._handle)

    # -- tokenization --------------------------------------------------------

    def encode(self, text: str, max_tokens: int = 8192) -> List[int]:
        """Encode text into token IDs."""
        buf = (ctypes.c_uint32 * max_tokens)()
        n = self._lib.baseRT_encode(
            self._handle, text.encode("utf-8"), buf, max_tokens
        )
        if n < 0:
            raise BaseRTError(get_error() or "encode failed")
        return list(buf[:n])

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its text representation."""
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
        """
        tokens = self._to_token_array(prompt)
        sampling, _anchor = self._make_sampling(temperature, top_k, top_p, min_p, repeat_penalty)

        if callback is not None:
            @BASERT_TOKEN_CALLBACK
            def _cb(tid: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                return callback(tid, text.decode("utf-8") if text else "")

            cb = _cb
        else:
            cb = BASERT_TOKEN_CALLBACK(0)  # NULL

        stats_c = self._lib.baseRT_generate(
            self._handle, tokens, len(tokens), max_tokens, sampling, cb, None
        )
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
        """Continue generation from the current KV cache state (multi-turn)."""
        tokens = self._to_token_array(new_tokens)
        sampling, _anchor = self._make_sampling(temperature, top_k, top_p, min_p, repeat_penalty)

        if callback is not None:
            @BASERT_TOKEN_CALLBACK
            def _cb(tid: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                return callback(tid, text.decode("utf-8") if text else "")

            cb = _cb
        else:
            cb = BASERT_TOKEN_CALLBACK(0)

        stats_c = self._lib.baseRT_generate_continue(
            self._handle, tokens, len(tokens), max_tokens, sampling, cb, None
        )
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
        q: queue.Queue[Optional[str]] = queue.Queue()
        stats_holder: List[Optional[GenerationStats]] = [None]

        def _run() -> None:
            @BASERT_TOKEN_CALLBACK
            def _cb(tid: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                q.put(text.decode("utf-8") if text else "")
                return True

            tokens = self._to_token_array(prompt)
            sampling, _anchor = self._make_sampling(
                temperature, top_k, top_p, min_p, repeat_penalty
            )
            stats_c = self._lib.baseRT_generate(
                self._handle, tokens, len(tokens), max_tokens, sampling, _cb, None
            )
            stats_holder[0] = GenerationStats._from_c(stats_c)
            q.put(None)  # sentinel

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

        t.join()

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
        return self._lib.baseRT_prefill(self._handle, arr, len(arr))

    def prefill_image(self, tokens: Union[str, List[int]], image_path: str) -> int:
        """Multimodal prefill: run vision tower on image, splice features into
        the prompt at image_token_id positions, then run LLM prefill.
        Returns the first generated token ID, or 0 on error."""
        arr = self._to_token_array(tokens)
        return self._lib.baseRT_prefill_image(
            self._handle, arr, len(arr), image_path.encode("utf-8")
        )

    def image_num_tokens(self, image_path: str) -> int:
        """Return the number of image placeholder tokens the vision tower
        produces for the given image (depends on image dimensions)."""
        return self._lib.baseRT_image_num_tokens(
            self._handle, image_path.encode("utf-8")
        )

    def prefill_audio(
        self, tokens: Union[str, List[int]], pcm_samples: List[float]
    ) -> int:
        """Audio prefill: run Conformer encoder on PCM (16kHz mono float32),
        splice features into prompt at audio_token_id positions.
        Returns first generated token ID, or 0 on error."""
        arr = self._to_token_array(tokens)
        pcm_arr = (ctypes.c_float * len(pcm_samples))(*pcm_samples)
        return self._lib.baseRT_prefill_audio(
            self._handle, arr, len(arr), pcm_arr, len(pcm_samples)
        )

    def audio_num_tokens(self, n_samples: int) -> int:
        """Return the number of audio placeholder tokens for n_samples of 16kHz PCM."""
        return self._lib.baseRT_audio_num_tokens(self._handle, n_samples)

    def decode_step(self, token_id: int, position: int) -> int:
        """Run a single decode step, returning the next token ID."""
        return self._lib.baseRT_decode_step(self._handle, token_id, position)

    def chain_decode(
        self, first_token: int, start_position: int, count: int
    ) -> List[int]:
        """Chain decode multiple tokens in one GPU submission."""
        buf = (ctypes.c_uint32 * count)()
        n = self._lib.baseRT_chain_decode(
            self._handle, first_token, start_position, count, buf
        )
        if n < 0:
            raise BaseRTError(get_error() or "chain_decode failed")
        return list(buf[:n])

    def reset(self) -> None:
        """Reset KV cache and internal state."""
        self._lib.baseRT_reset(self._handle)

    def set_speculation(self, enabled: bool) -> None:
        """Enable or disable speculative decoding (n-gram prediction)."""
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
        """
        stats_c = BaseRTTranscribeStats()
        lang = language.encode("utf-8") if language else None

        if callback is not None:
            @BASERT_SEGMENT_CALLBACK
            def _cb(start_ms: int, end_ms: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                return callback(start_ms, end_ms, text.decode("utf-8") if text else "")

            cb = _cb
        else:
            cb = BASERT_SEGMENT_CALLBACK(0)  # NULL

        result = self._lib.baseRT_transcribe_stream(
            self._handle, wav_path.encode("utf-8"), lang, ctypes.byref(stats_c), cb, None
        )
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
        """
        n = len(samples)
        arr = (ctypes.c_float * n)(*samples)
        stats_c = BaseRTTranscribeStats()
        lang = language.encode("utf-8") if language else None

        if callback is not None:
            @BASERT_SEGMENT_CALLBACK
            def _cb(start_ms: int, end_ms: int, text: bytes, _ud: ctypes.c_void_p) -> bool:
                return callback(start_ms, end_ms, text.decode("utf-8") if text else "")

            cb = _cb
        else:
            cb = BASERT_SEGMENT_CALLBACK(0)  # NULL

        result = self._lib.baseRT_transcribe_pcm_stream(
            self._handle, arr, n, lang, ctypes.byref(stats_c), cb, None
        )
        if not result:
            raise BaseRTError(get_error() or "transcription failed")
        return result.decode("utf-8"), TranscribeStats._from_c(stats_c)

    def set_timestamps(self, enabled: bool) -> None:
        """Enable or disable timestamp generation for Whisper transcription."""
        self._lib.baseRT_set_timestamps(self._handle, enabled)

    # -- embeddings ----------------------------------------------------------

    def embed(self, tokens: List[int], max_dims: int = 0) -> List[float]:
        """Compute embeddings from token IDs.

        Args:
            tokens: List of token IDs.
            max_dims: Maximum embedding dimensions (0 = use model default).

        Returns:
            List of float embedding values.
        """
        if max_dims <= 0:
            max_dims = self._lib.baseRT_embedding_dim(self._handle)
        arr = (ctypes.c_uint32 * len(tokens))(*tokens)
        out = (ctypes.c_float * max_dims)()
        dim = self._lib.baseRT_embed(self._handle, arr, len(tokens), out, max_dims)
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
        if max_dims <= 0:
            max_dims = self._lib.baseRT_embedding_dim(self._handle)
        out = (ctypes.c_float * max_dims)()
        dim = self._lib.baseRT_embed_text(
            self._handle, text.encode("utf-8"), out, max_dims
        )
        if dim <= 0:
            raise BaseRTError(get_error() or "embed_text failed")
        return list(out[:dim])

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for this model."""
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
        result = self._lib.baseRT_format_chat(
            self._handle, system.encode("utf-8"), user.encode("utf-8")
        )
        if not result:
            raise BaseRTError(get_error() or "format_chat failed")
        return result.decode("utf-8")

    @property
    def chat_template(self) -> str:
        """Get the chat template name for the loaded model."""
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
        return self._lib.baseRT_token_count(self._handle, text.encode("utf-8"))

    # -- model inspection ----------------------------------------------------

    def tensors(self) -> List[TensorInfo]:
        """Return metadata for every tensor in the model."""
        count = self._lib.baseRT_tensor_count(self._handle)
        result: List[TensorInfo] = []
        for i in range(count):
            name_b = self._lib.baseRT_tensor_name(self._handle, i)
            raw_b = self._lib.baseRT_tensor_raw_dtype(self._handle, i)
            result.append(
                TensorInfo(
                    index=i,
                    name=name_b.decode("utf-8") if name_b else "",
                    dtype=self._lib.baseRT_tensor_dtype(self._handle, i),
                    raw_dtype=raw_b.decode("utf-8") if raw_b else "",
                )
            )
        return result

    # -- helpers -------------------------------------------------------------

    def _to_token_array(
        self, prompt: Union[str, List[int]]
    ) -> ctypes.Array[ctypes.c_uint32]:
        if isinstance(prompt, str):
            ids = self.encode(prompt)
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
