"""
Tests for the baseRT Python bindings that do NOT require a GPU or model file.

Covers: ctypes struct layout, dataclass construction, SamplingConfig defaults,
library path resolution, error handling, and the token callback type.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the bindings package is importable regardless of working directory.
_BINDINGS_DIR = Path(__file__).resolve().parent.parent
if str(_BINDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(_BINDINGS_DIR))

import baseRT
from baseRT import (
    BASERT_TOKEN_CALLBACK,
    BaseRTGenerationStats,
    BaseRTModelConfig,
    BaseRTSamplingConfig,
    BaseRTTranscribeStats,
    GenerationStats,
    ModelConfig,
    TranscribeStats,
    TensorInfo,
    BaseRTError,
    _LIB_NAME,
    _find_library,
)


# -------------------------------------------------------------------------
# ctypes struct sizes and field expectations
# -------------------------------------------------------------------------


class TestBaseRTModelConfig:
    """Verify BaseRTModelConfig struct layout."""

    def test_field_names(self):
        names = [f[0] for f in BaseRTModelConfig._fields_]
        expected = [
            "dim", "n_layers", "n_heads", "n_kv_heads", "head_dim",
            "q_dim", "kv_dim", "ffn_dim", "vocab_size", "max_seq_len",
            "norm_eps", "rope_theta", "sliding_window_pattern", "rope_local_theta",
            "architecture",
            "enc_n_layers", "enc_n_heads", "enc_dim", "enc_ffn_dim",
            "n_mels", "enc_max_seq_len",
        ]
        assert names == expected

    def test_field_count(self):
        assert len(BaseRTModelConfig._fields_) == 21

    def test_architecture_field_is_char_array(self):
        # architecture should be a fixed 32-byte char array
        for name, ctype, *rest in BaseRTModelConfig._fields_:
            if name == "architecture":
                inst = BaseRTModelConfig()
                field_val = getattr(inst, "architecture")
                assert len(field_val) <= 32
                break
        else:
            pytest.fail("architecture field not found")

    def test_default_zeroed(self):
        cfg = BaseRTModelConfig()
        assert cfg.dim == 0
        assert cfg.n_layers == 0
        assert cfg.vocab_size == 0
        assert cfg.norm_eps == 0.0
        assert cfg.rope_theta == 0.0
        assert cfg.architecture == b""

    def test_roundtrip_values(self):
        cfg = BaseRTModelConfig()
        cfg.dim = 4096
        cfg.n_layers = 32
        cfg.n_heads = 32
        cfg.n_kv_heads = 8
        cfg.head_dim = 128
        cfg.vocab_size = 151936
        cfg.norm_eps = 1e-5
        cfg.rope_theta = 500000.0
        cfg.architecture = b"qwen3"
        cfg.enc_n_layers = 6
        cfg.n_mels = 80

        assert cfg.dim == 4096
        assert cfg.n_layers == 32
        assert cfg.n_kv_heads == 8
        assert cfg.vocab_size == 151936
        assert abs(cfg.norm_eps - 1e-5) < 1e-10
        assert cfg.architecture == b"qwen3"
        assert cfg.enc_n_layers == 6
        assert cfg.n_mels == 80


class TestBaseRTSamplingConfig:
    """Verify BaseRTSamplingConfig struct layout and defaults."""

    def test_field_names(self):
        names = [f[0] for f in BaseRTSamplingConfig._fields_]
        assert names == [
            "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
            "presence_penalty", "frequency_penalty", "seed", "n_logit_bias",
            "logit_bias_tokens", "logit_bias_values",
        ]

    def test_field_count(self):
        assert len(BaseRTSamplingConfig._fields_) == 11

    def test_construction_with_values(self):
        sc = BaseRTSamplingConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            min_p=0.05,
            repeat_penalty=1.1,
            presence_penalty=0.5,
            frequency_penalty=-0.25,
            seed=42,
        )
        assert abs(sc.temperature - 0.7) < 1e-6
        assert sc.top_k == 50
        assert abs(sc.top_p - 0.95) < 1e-6
        assert abs(sc.min_p - 0.05) < 1e-6
        assert abs(sc.repeat_penalty - 1.1) < 1e-6
        assert abs(sc.presence_penalty - 0.5) < 1e-6
        assert abs(sc.frequency_penalty - (-0.25)) < 1e-6
        assert sc.seed == 42

    def test_default_zeroed(self):
        sc = BaseRTSamplingConfig()
        assert sc.temperature == 0.0
        assert sc.top_k == 0
        assert sc.top_p == 0.0
        assert sc.presence_penalty == 0.0
        assert sc.frequency_penalty == 0.0
        assert sc.seed == 0
        assert sc.n_logit_bias == 0

    def test_size_is_reasonable(self):
        # 7 floats + 1 uint32 + 1 int32 (+ pad) + 2 pointers (8B each on 64-bit)
        # = 28 + 4 + 4 (+ 4 pad) + 16 = 56 bytes on 64-bit Apple Silicon
        assert ctypes.sizeof(BaseRTSamplingConfig) >= 56


class TestBaseRTGenerationStats:
    """Verify BaseRTGenerationStats struct layout."""

    def test_field_names(self):
        names = [f[0] for f in BaseRTGenerationStats._fields_]
        assert names == [
            "prompt_tokens", "generated_tokens",
            "prefill_time_ms", "decode_time_ms",
            "prefill_tokens_per_sec", "decode_tokens_per_sec",
        ]

    def test_field_count(self):
        assert len(BaseRTGenerationStats._fields_) == 6

    def test_roundtrip(self):
        gs = BaseRTGenerationStats()
        gs.prompt_tokens = 42
        gs.generated_tokens = 100
        gs.prefill_time_ms = 12.5
        gs.decode_time_ms = 800.0
        gs.prefill_tokens_per_sec = 3360.0
        gs.decode_tokens_per_sec = 125.0
        assert gs.prompt_tokens == 42
        assert gs.generated_tokens == 100
        assert abs(gs.prefill_time_ms - 12.5) < 1e-3


class TestBaseRTTranscribeStats:
    """Verify BaseRTTranscribeStats struct layout."""

    def test_field_names(self):
        names = [f[0] for f in BaseRTTranscribeStats._fields_]
        assert names == ["n_tokens", "audio_ms", "encode_ms", "decode_ms", "total_ms"]

    def test_field_count(self):
        assert len(BaseRTTranscribeStats._fields_) == 5

    def test_roundtrip(self):
        ts = BaseRTTranscribeStats()
        ts.n_tokens = 200
        ts.audio_ms = 30000.0
        ts.encode_ms = 50.0
        ts.decode_ms = 400.0
        ts.total_ms = 450.0
        assert ts.n_tokens == 200
        assert abs(ts.total_ms - 450.0) < 1e-3


# -------------------------------------------------------------------------
# Token callback type
# -------------------------------------------------------------------------


class TestTokenCallback:
    """Verify the BASERT_TOKEN_CALLBACK cfunctype."""

    def test_null_callback(self):
        cb = BASERT_TOKEN_CALLBACK(0)
        # A null callback should be falsy when cast
        assert not cb

    def test_callable_callback(self):
        called_with = []

        @BASERT_TOKEN_CALLBACK
        def my_cb(tid, text, ud):
            called_with.append((tid, text))
            return True

        # The wrapped callback should be truthy
        assert my_cb


# -------------------------------------------------------------------------
# Pythonic dataclass construction
# -------------------------------------------------------------------------


class TestModelConfigDataclass:
    """Test ModelConfig dataclass and _from_c conversion."""

    def test_construction(self):
        mc = ModelConfig(
            dim=2048, n_layers=24, n_heads=16, n_kv_heads=4,
            head_dim=128, q_dim=2048, kv_dim=512, ffn_dim=5504,
            vocab_size=32000, max_seq_len=4096, norm_eps=1e-5,
            rope_theta=10000.0, sliding_window_pattern=0,
            rope_local_theta=0.0, architecture="llama",
            enc_n_layers=0, enc_n_heads=0, enc_dim=0, enc_ffn_dim=0,
            n_mels=0, enc_max_seq_len=0,
        )
        assert mc.dim == 2048
        assert mc.architecture == "llama"
        assert mc.vocab_size == 32000

    def test_from_c_struct(self):
        c = BaseRTModelConfig()
        c.dim = 768
        c.n_layers = 12
        c.n_heads = 12
        c.n_kv_heads = 12
        c.head_dim = 64
        c.q_dim = 768
        c.kv_dim = 768
        c.ffn_dim = 3072
        c.vocab_size = 50257
        c.max_seq_len = 2048
        c.norm_eps = 1e-5
        c.rope_theta = 10000.0
        c.sliding_window_pattern = 0
        c.rope_local_theta = 0.0
        c.architecture = b"gpt2"
        c.enc_n_layers = 0
        c.enc_n_heads = 0
        c.enc_dim = 0
        c.enc_ffn_dim = 0
        c.n_mels = 0
        c.enc_max_seq_len = 0

        mc = ModelConfig._from_c(c)
        assert mc.dim == 768
        assert mc.n_layers == 12
        assert mc.vocab_size == 50257
        assert mc.architecture == "gpt2"
        assert mc.norm_eps == pytest.approx(1e-5)

    def test_from_c_strips_null_bytes_in_architecture(self):
        c = BaseRTModelConfig()
        c.architecture = b"qwen3\x00\x00\x00"
        mc = ModelConfig._from_c(c)
        assert mc.architecture == "qwen3"
        assert "\x00" not in mc.architecture


class TestGenerationStatsDataclass:
    """Test GenerationStats dataclass and _from_c conversion."""

    def test_construction(self):
        gs = GenerationStats(
            prompt_tokens=10, generated_tokens=50,
            prefill_time_ms=5.0, decode_time_ms=200.0,
            prefill_tokens_per_sec=2000.0, decode_tokens_per_sec=250.0,
        )
        assert gs.prompt_tokens == 10
        assert gs.generated_tokens == 50

    def test_from_c_struct(self):
        c = BaseRTGenerationStats()
        c.prompt_tokens = 15
        c.generated_tokens = 75
        c.prefill_time_ms = 3.0
        c.decode_time_ms = 300.0
        c.prefill_tokens_per_sec = 5000.0
        c.decode_tokens_per_sec = 250.0

        gs = GenerationStats._from_c(c)
        assert gs.prompt_tokens == 15
        assert gs.generated_tokens == 75
        assert gs.prefill_time_ms == pytest.approx(3.0)
        assert gs.decode_tokens_per_sec == pytest.approx(250.0)


class TestTranscribeStatsDataclass:
    """Test TranscribeStats dataclass and _from_c conversion."""

    def test_construction(self):
        ts = TranscribeStats(
            n_tokens=100, audio_ms=30000.0,
            encode_ms=50.0, decode_ms=400.0, total_ms=450.0,
        )
        assert ts.n_tokens == 100
        assert ts.total_ms == 450.0

    def test_from_c_struct(self):
        c = BaseRTTranscribeStats()
        c.n_tokens = 80
        c.audio_ms = 15000.0
        c.encode_ms = 25.0
        c.decode_ms = 200.0
        c.total_ms = 225.0

        ts = TranscribeStats._from_c(c)
        assert ts.n_tokens == 80
        assert ts.audio_ms == pytest.approx(15000.0)
        assert ts.total_ms == pytest.approx(225.0)


class TestTensorInfoDataclass:
    """Test TensorInfo dataclass."""

    def test_construction(self):
        ti = TensorInfo(index=0, name="model.embed_tokens.weight", dtype=7, raw_dtype="Q4_0")
        assert ti.index == 0
        assert ti.name == "model.embed_tokens.weight"
        assert ti.dtype == 7
        assert ti.raw_dtype == "Q4_0"


# -------------------------------------------------------------------------
# SamplingConfig helper via Model._make_sampling
# -------------------------------------------------------------------------


class TestMakeSampling:
    """Test Model._make_sampling static method."""

    def test_defaults(self):
        sc, anchor = baseRT.Model._make_sampling()
        assert sc.temperature == pytest.approx(0.0)
        assert sc.top_k == 40
        assert sc.top_p == pytest.approx(0.9)
        assert sc.min_p == pytest.approx(0.0)
        assert sc.repeat_penalty == pytest.approx(1.0)
        assert sc.presence_penalty == pytest.approx(0.0)
        assert sc.frequency_penalty == pytest.approx(0.0)
        assert sc.seed == 0
        assert sc.n_logit_bias == 0
        assert anchor is None

    def test_custom_values(self):
        sc, _anchor = baseRT.Model._make_sampling(
            temperature=1.2, top_k=100, top_p=0.95,
            min_p=0.05, repeat_penalty=1.3,
            presence_penalty=0.3, frequency_penalty=-0.2, seed=123,
        )
        assert sc.temperature == pytest.approx(1.2)
        assert sc.top_k == 100
        assert sc.top_p == pytest.approx(0.95)
        assert sc.min_p == pytest.approx(0.05)
        assert sc.repeat_penalty == pytest.approx(1.3)
        assert sc.presence_penalty == pytest.approx(0.3)
        assert sc.frequency_penalty == pytest.approx(-0.2)
        assert sc.seed == 123

    def test_greedy_defaults(self):
        # temperature=0 means greedy
        sc, _anchor = baseRT.Model._make_sampling(temperature=0.0)
        assert sc.temperature == pytest.approx(0.0)

    def test_logit_bias_population(self):
        sc, anchor = baseRT.Model._make_sampling(logit_bias={42: 5.0, 100: -3.5})
        assert sc.n_logit_bias == 2
        # Anchor must keep the backing arrays alive — without it the ctypes
        # pointers would dangle once _make_sampling returns.
        assert anchor is not None
        assert sc.logit_bias_tokens[0] in (42, 100)
        assert sc.logit_bias_tokens[1] in (42, 100)


# -------------------------------------------------------------------------
# Library path resolution (_find_library)
# -------------------------------------------------------------------------


class TestFindLibrary:
    """Test the _find_library helper."""

    def test_env_override(self, tmp_path):
        fake_lib = tmp_path / "fake_libbaseRT.dylib"
        fake_lib.touch()
        with mock.patch.dict(os.environ, {"BASERT_LIB_PATH": str(fake_lib)}):
            result = _find_library()
            assert result == str(fake_lib)

    def test_env_override_takes_precedence_over_filesystem(self, tmp_path):
        # Even if the env path doesn't exist, _find_library returns it
        # (the validation happens later in ctypes.CDLL)
        fake_path = str(tmp_path / "nonexistent.dylib")
        with mock.patch.dict(os.environ, {"BASERT_LIB_PATH": fake_path}):
            result = _find_library()
            assert result == fake_path

    def test_raises_when_not_found(self):
        # Clear env var and point __file__ somewhere with no build dir nearby
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("baseRT.__file__", "/nonexistent/bindings/python/baseRT/__init__.py"):
                with pytest.raises(OSError, match="Cannot find"):
                    _find_library()

    def test_finds_library_in_build_dir(self, tmp_path):
        # Simulate the project layout:
        # tmp_path/bindings/python/baseRT/__init__.py
        # tmp_path/build/libbaseRT.dylib
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        fake_lib = build_dir / _LIB_NAME
        fake_lib.touch()

        bindings_dir = tmp_path / "bindings" / "python" / "baseRT"
        bindings_dir.mkdir(parents=True)

        # Patch the module's __file__ so Path(__file__) resolves relative to tmp_path
        fake_init = bindings_dir / "__init__.py"
        fake_init.touch()

        # We need to patch the `here` variable inside _find_library
        # The function does: here = Path(__file__).resolve().parent
        # We mock __file__ at the module level
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("baseRT.__file__", str(fake_init)):
                result = _find_library()
                assert result == str(fake_lib.resolve())

    def test_lib_name_constant(self):
        assert _LIB_NAME == "libbaseRT.dylib"


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------


class TestBaseRTError:
    """Test the BaseRTError exception class."""

    def test_is_runtime_error(self):
        assert issubclass(BaseRTError, RuntimeError)

    def test_message(self):
        err = BaseRTError("test error message")
        assert str(err) == "test error message"


class TestCheckModel:
    """Test _check_model helper."""

    def test_raises_on_null_handle(self):
        # _check_model raises BaseRTError when handle is falsy
        with mock.patch("baseRT.get_error", return_value="model not found"):
            with pytest.raises(BaseRTError, match="Failed to load model: model not found"):
                baseRT._check_model(None)

    def test_raises_with_unknown_error_on_null_handle(self):
        with mock.patch("baseRT.get_error", return_value=None):
            with pytest.raises(BaseRTError, match="unknown error"):
                baseRT._check_model(None)

    def test_no_raise_on_valid_handle(self):
        # A non-null handle should not raise
        baseRT._check_model(ctypes.c_void_p(0x1234))


# -------------------------------------------------------------------------
# Model loading with nonexistent model
# -------------------------------------------------------------------------


class TestModelLoadError:
    """Test that loading a nonexistent model raises an appropriate error."""

    def test_nonexistent_model_raises(self):
        """Loading a nonexistent model should raise either OSError (lib not found)
        or BaseRTError (lib found but model load fails)."""
        with pytest.raises((OSError, BaseRTError)):
            baseRT.Model("/nonexistent/path/to/model.base")


# -------------------------------------------------------------------------
# Version constant
# -------------------------------------------------------------------------


class TestVersion:
    def test_version_string(self):
        assert isinstance(baseRT.__version__, str)
        parts = baseRT.__version__.split(".")
        assert len(parts) == 3
        # All parts should be numeric
        for p in parts:
            assert p.isdigit()


# -------------------------------------------------------------------------
# Struct size consistency checks
# -------------------------------------------------------------------------


class TestStructSizes:
    """Sanity check that struct sizes are within expected ranges.

    These are not exact because padding may vary, but they catch
    egregious mismatches (e.g., accidentally duplicating fields).
    """

    def test_model_config_size(self):
        size = ctypes.sizeof(BaseRTModelConfig)
        # 14 uint32 + 2 float + 32 char + 6 uint32 = 20*4 + 2*4 + 32 = 120 min
        assert size >= 112
        # Should not be absurdly large
        assert size <= 256

    def test_sampling_config_size(self):
        size = ctypes.sizeof(BaseRTSamplingConfig)
        # 5 floats + 2 floats (presence/freq) + 1 uint32 + 1 int32 + padding +
        # 2 pointers (8 bytes each on 64-bit) ≈ 56 bytes
        assert size >= 56
        assert size <= 80

    def test_generation_stats_size(self):
        size = ctypes.sizeof(BaseRTGenerationStats)
        # 2 ints + 4 floats = 24 bytes
        assert size >= 24
        assert size <= 48

    def test_transcribe_stats_size(self):
        size = ctypes.sizeof(BaseRTTranscribeStats)
        # 1 int + 4 floats = 20 bytes
        assert size >= 20
        assert size <= 40


# -------------------------------------------------------------------------
# _get_lib caching
# -------------------------------------------------------------------------


class TestGetLib:
    """Test _get_lib lazy loading and caching."""

    def test_get_lib_caches(self):
        """Calling _get_lib twice should return the same object (if lib loads)."""
        # We can only test the caching logic if the lib was already loaded or
        # if we mock it. Let's test with a mock.
        sentinel = object()
        with mock.patch("baseRT._lib", sentinel):
            result = baseRT._get_lib()
            assert result is sentinel

    def test_get_lib_calls_load_when_none(self):
        fake_lib = mock.MagicMock()
        with mock.patch("baseRT._lib", None):
            with mock.patch("baseRT._load_lib", return_value=fake_lib) as load_mock:
                with mock.patch("baseRT._setup_signatures") as setup_mock:
                    result = baseRT._get_lib()
                    load_mock.assert_called_once()
                    setup_mock.assert_called_once_with(fake_lib)
