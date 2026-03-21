# NeMo ASR Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `nemo` STT backend using NeMo's cache-aware streaming with FastConformer models, Silero VAD for endpoint detection.

**Architecture:** Single file `backends/nemo.py` implementing `ASRBackend` protocol. Global singleton NeMo model + per-connection Silero VAD clones + per-connection cache state. Inference in `push_audio()` (incremental, like qwen3). `CacheAwareStreamingAudioBuffer` handles feature extraction.

**Tech Stack:** `nemo_toolkit[asr]`, `torch` (Silero VAD + NeMo), `numpy`

**Spec:** `docs/superpowers/specs/2026-03-21-nemo-backend-design.md`

**Branch:** `feature/nemo-backend` (already created from `master`)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `backends/nemo.py` | NemoBackend, _StreamingState, singletons, Silero VAD |
| Create | `tests/mock/test_nemo_backend.py` | All mock tests for nemo backend |
| Modify | `backends/registry.py` | Add `"nemo"` entry |
| Modify | `config.py` | Add `STT_NEMO_*` settings |
| Modify | `app.py` | Add nemo preload on startup |
| Modify | `pyproject.toml` | Add `nemo` optional extra |
| Modify | `CLAUDE.md` | Add nemo docs |

---

## Task 1: Config, Registry, and Dependencies

**Files:**
- Modify: `config.py:6-49`
- Modify: `pyproject.toml:16-17`
- Modify: `backends/registry.py:7-10`
- Modify: `app.py:16-21`

- [ ] **Step 1: Add nemo config fields to `config.py`**

Add after the existing qwen-asr streaming config (after line 46):

```python
    # NeMo backend config
    nemo_model: str = "stt_en_fastconformer_transducer_large_streaming"
    nemo_chunk_size: float = 0.34
    nemo_shift_size: float = 0.34
    nemo_left_chunks: int = 2
    nemo_vad_threshold: float = 0.5
```

- [ ] **Step 2: Add nemo optional dependency to `pyproject.toml`**

Add after the `qwen3` entry in `[project.optional-dependencies]`:

```toml
nemo = ["nemo_toolkit[asr]", "torch"]
```

- [ ] **Step 3: Add nemo to backend registry in `backends/registry.py`**

Add to the `_BACKENDS` dict:

```python
    "nemo": "backends.nemo.NemoBackend",
```

- [ ] **Step 4: Add nemo preload to `app.py`**

In the `lifespan` function, after the qwen3 preload block (after line 20), add:

```python
    elif settings.backend == "nemo":
        from backends.nemo import preload_model

        preload_model()
```

- [ ] **Step 5: Commit**

```bash
git add config.py pyproject.toml backends/registry.py app.py
git commit -m "feat: add nemo backend config, registry entry, and dependency"
```

---

## Task 2: Create `backends/nemo.py` with `_StreamingState` and Singletons

**Files:**
- Create: `backends/nemo.py`
- Create: `tests/mock/test_nemo_backend.py`

- [ ] **Step 1: Write failing tests**

Create `tests/mock/test_nemo_backend.py`:

```python
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from backends.base import ASRResult
from config import settings


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pcm_silence(n_samples: int) -> bytes:
    """PCM16LE silence (all zeros)."""
    return b"\x00\x00" * n_samples


def _make_pcm_tone(n_samples: int, amplitude: int = 10000) -> bytes:
    """PCM16LE constant-amplitude samples (square wave)."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _make_mock_nemo_model():
    """Create a mock NeMo ASR model.

    Simulates conformer_stream_step() returning transcription + cache tensors.
    """
    mock_model = MagicMock()

    # cfg.preprocessor attributes
    mock_model.cfg.preprocessor.sample_rate = 16000
    mock_model.cfg.preprocessor.window_stride = 0.01  # 10ms
    mock_model.cfg.preprocessor.features = 80

    # encoder attributes
    mock_model.encoder.pre_encode_cache_size = 3
    mock_model.encoder.setup_streaming_params = MagicMock()

    # get_initial_cache_state returns 3 zero tensors
    mock_model.encoder.get_initial_cache_state.return_value = (
        MagicMock(),  # cache_last_channel
        MagicMock(),  # cache_last_time
        MagicMock(),  # cache_last_channel_len
    )

    # conformer_stream_step returns (pred_out, texts, caches..., hypotheses)
    def _mock_stream_step(**kwargs):
        return (
            MagicMock(),        # pred_out
            ["hello world"],    # transcribed_texts
            MagicMock(),        # cache_last_channel
            MagicMock(),        # cache_last_time
            MagicMock(),        # cache_last_channel_len
            None,               # previous_hypotheses
        )

    mock_model.conformer_stream_step.side_effect = _mock_stream_step

    # change_decoding_strategy as no-op
    mock_model.change_decoding_strategy = MagicMock()

    return mock_model


def _make_mock_vad_model(speech_prob: float = 0.0):
    """Create a mock Silero VAD model."""
    mock_vad = MagicMock()
    mock_vad.return_value = MagicMock(item=MagicMock(return_value=speech_prob))
    mock_vad.reset_states = MagicMock()
    return mock_vad


def _make_mock_streaming_buffer():
    """Create a mock CacheAwareStreamingAudioBuffer.

    Returns a fresh iterator each time __iter__ is called so multi-chunk tests work.
    """
    mock_buf = MagicMock()
    mock_signal = MagicMock()
    mock_length = MagicMock()
    mock_buf.__iter__ = MagicMock(
        side_effect=lambda: iter([(mock_signal, mock_length)])
    )
    mock_buf.append_audio_chunk = MagicMock()
    return mock_buf


def _make_backend(mock_nemo=None, mock_vad=None, language="auto"):
    """Create and configure a NemoBackend with models mocked out."""
    if mock_nemo is None:
        mock_nemo = _make_mock_nemo_model()
    if mock_vad is None:
        mock_vad = _make_mock_vad_model()

    mock_streaming_buf = _make_mock_streaming_buffer()

    with (
        patch("backends.nemo._get_nemo_model", return_value=mock_nemo),
        patch("backends.nemo._get_vad_base_model", return_value=mock_vad),
        patch("backends.nemo.CacheAwareStreamingAudioBuffer", return_value=mock_streaming_buf),
    ):
        from backends.nemo import NemoBackend

        b = NemoBackend()
        b.configure(sample_rate=16000, language=language)
    return b


# ── Configure & Close tests ─────────────────────────────────────────────────


class TestConfigureAndClose:
    def test_configure_initializes_state(self):
        backend = _make_backend()
        assert backend._sample_rate == 16000
        assert backend._language == "auto"
        assert backend._streaming is not None
        assert backend._streaming.current_text == ""
        assert backend._in_speech is False
        assert backend._endpoint_fired is False

    def test_configure_initializes_caches(self):
        mock_nemo = _make_mock_nemo_model()
        backend = _make_backend(mock_nemo)
        assert backend._streaming.cache_last_channel is not None
        assert backend._streaming.cache_last_time is not None
        assert backend._streaming.cache_last_channel_len is not None
        assert backend._streaming.cache_pre_encode is not None
        assert backend._streaming.previous_hypotheses is None
        assert backend._streaming.previous_pred_out is None

    def test_close_is_noop(self):
        backend = _make_backend()
        backend.close()  # should not raise

    def test_close_idempotent(self):
        backend = _make_backend()
        backend.close()
        backend.close()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestConfigureAndClose -v
```

Expected: FAIL (backends.nemo doesn't exist yet)

- [ ] **Step 3: Create `backends/nemo.py`**

```python
from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np
import structlog

from backends.base import ASRResult
from config import settings

logger = structlog.get_logger()

# ── Lazy import placeholder for CacheAwareStreamingAudioBuffer ───────────────
# Actual import happens inside functions to avoid requiring nemo at module load.
# For patching in tests, we re-export the name at module level after lazy import.
CacheAwareStreamingAudioBuffer = None  # set by _ensure_nemo_imports()


def _ensure_nemo_imports() -> None:
    """Lazy-import NeMo streaming utilities."""
    global CacheAwareStreamingAudioBuffer
    if CacheAwareStreamingAudioBuffer is None:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer as _Buf,
        )
        CacheAwareStreamingAudioBuffer = _Buf


# ── StreamingState ───────────────────────────────────────────────────────────


class _StreamingState:
    """Per-connection streaming context with NeMo cache state."""

    def __init__(
        self,
        streaming_buffer: object,
        cache_last_channel: object,
        cache_last_time: object,
        cache_last_channel_len: object,
        cache_pre_encode: object,
        chunk_size_samples: int,
    ) -> None:
        self.streaming_buffer = streaming_buffer
        self.cache_last_channel = cache_last_channel
        self.cache_last_time = cache_last_time
        self.cache_last_channel_len = cache_last_channel_len
        self.cache_pre_encode = cache_pre_encode
        self.previous_hypotheses = None
        self.previous_pred_out = None
        self.current_text: str = ""
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.chunk_size_samples: int = chunk_size_samples


# ── Module-level NeMo model singleton ────────────────────────────────────────


_nemo_model: object | None = None
_vad_base_model: object | None = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_nemo_model() -> object:
    global _nemo_model
    if _nemo_model is None:
        with _model_lock:
            if _nemo_model is None:
                import torch
                import nemo.collections.asr as nemo_asr
                from omegaconf import open_dict

                model_name = settings.nemo_model
                # Load model: local .nemo file or NGC pretrained name
                if model_name.endswith(".nemo") or Path(model_name).is_file():
                    model = nemo_asr.models.ASRModel.restore_from(model_name)
                else:
                    model = nemo_asr.models.ASRModel.from_pretrained(model_name)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.freeze()
                model.eval()

                # Convert chunk_size from seconds to encoder frames
                window_stride = model.cfg.preprocessor.window_stride
                # FastConformer stride is typically 4
                encoder_stride = getattr(model.encoder, "subsampling_factor", 4)
                chunk_frames = int(
                    settings.nemo_chunk_size / (window_stride * encoder_stride)
                )
                shift_frames = int(
                    settings.nemo_shift_size / (window_stride * encoder_stride)
                )

                model.encoder.setup_streaming_params(
                    chunk_size=chunk_frames,
                    shift_size=shift_frames,
                    left_chunks=settings.nemo_left_chunks,
                )

                # Configure decoding for streaming
                decoding_cfg = model.cfg.decoding
                with open_dict(decoding_cfg):
                    decoding_cfg.strategy = "greedy_batch"
                    decoding_cfg.preserve_alignments = True
                    decoding_cfg.fused_batch_size = -1
                model.change_decoding_strategy(decoding_cfg)

                logger.info(
                    "nemo.model.loaded",
                    model=model_name,
                    chunk_frames=chunk_frames,
                    shift_frames=shift_frames,
                )
                _nemo_model = model
    return _nemo_model


def _get_vad_base_model() -> object:
    global _vad_base_model
    if _vad_base_model is None:
        with _model_lock:
            if _vad_base_model is None:
                import torch

                _vad_base_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                )
                logger.info("nemo.vad_base.loaded")
    return _vad_base_model


def preload_model() -> None:
    """Preload both NeMo and VAD models at startup."""
    logger.info("nemo.preload.start")
    _get_nemo_model()
    _get_vad_base_model()
    logger.info("nemo.preload.done")


# ── Helper: create fresh streaming state ─────────────────────────────────────


def _create_streaming_state(model: object) -> _StreamingState:
    """Create a fresh _StreamingState with initialized caches."""
    import torch

    _ensure_nemo_imports()

    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=1)
    )
    cache_pre_encode = torch.zeros(
        1,
        model.encoder.pre_encode_cache_size,
        model.cfg.preprocessor.features,
    )

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=True,
        pad_and_drop_preencoded=True,
    )

    chunk_size_samples = int(settings.nemo_chunk_size * model.cfg.preprocessor.sample_rate)

    return _StreamingState(
        streaming_buffer=streaming_buffer,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
        cache_pre_encode=cache_pre_encode,
        chunk_size_samples=chunk_size_samples,
    )


# ── NemoBackend ──────────────────────────────────────────────────────────────


class NemoBackend:
    """NeMo ASR backend using cache-aware streaming with FastConformer."""

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._language = language

        model = _get_nemo_model()
        self._streaming = _create_streaming_state(model)

        # Per-connection Silero VAD (deepcopy for independent LSTM state)
        vad_base = _get_vad_base_model()
        self._vad_model = copy.deepcopy(vad_base)
        self._vad_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._in_speech: bool = False
        self._silence_ms_accum: float = 0.0
        self._endpoint_fired: bool = False

    def push_audio(self, pcm_data: bytes) -> None:
        pass  # implemented in Task 3

    def get_partial(self) -> ASRResult | None:
        return None  # implemented in Task 3

    def finalize(self) -> ASRResult:
        return ASRResult(text="", is_partial=False, is_endpoint=True)  # implemented in Task 4

    def detect_endpoint(self) -> bool:
        return False  # implemented in Task 5

    def reset_segment(self) -> None:
        model = _get_nemo_model()
        self._streaming = _create_streaming_state(model)

        self._vad_model = copy.deepcopy(_get_vad_base_model())
        self._vad_buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._endpoint_fired = False

    def close(self) -> None:
        pass  # model is shared singleton, not per-connection
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestConfigureAndClose -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backends/nemo.py tests/mock/test_nemo_backend.py
git commit -m "feat: add NemoBackend skeleton with _StreamingState and singletons"
```

---

## Task 3: Implement `push_audio` and `get_partial`

**Files:**
- Modify: `backends/nemo.py`
- Modify: `tests/mock/test_nemo_backend.py`

Inference happens in `push_audio()` when enough audio accumulates (>= chunk_size_samples). `get_partial()` is a cheap read.

- [ ] **Step 1: Write failing tests**

Append to `tests/mock/test_nemo_backend.py`:

```python
class TestPushAudioAndPartial:
    def setup_method(self):
        self.mock_nemo = _make_mock_nemo_model()
        self.mock_vad = _make_mock_vad_model()
        self.mock_streaming_buf = _make_mock_streaming_buffer()
        self.nemo_patcher = patch(
            "backends.nemo._get_nemo_model", return_value=self.mock_nemo
        )
        self.vad_patcher = patch(
            "backends.nemo._get_vad_base_model", return_value=self.mock_vad
        )
        self.buf_patcher = patch(
            "backends.nemo.CacheAwareStreamingAudioBuffer",
            return_value=self.mock_streaming_buf,
        )
        self.nemo_patcher.start()
        self.vad_patcher.start()
        self.buf_patcher.start()

    def teardown_method(self):
        self.nemo_patcher.stop()
        self.vad_patcher.stop()
        self.buf_patcher.stop()

    def test_push_audio_accumulates_float32(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        pcm = _make_pcm_tone(1600, amplitude=5000)
        backend.push_audio(pcm)

        assert backend._streaming.audio_buffer.dtype == np.float32
        assert len(backend._streaming.audio_buffer) == 1600

        expected = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(
            backend._streaming.audio_buffer, expected
        )

    def test_push_audio_empty_is_noop(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend.push_audio(b"")
        assert len(backend._streaming.audio_buffer) == 0

    def test_push_audio_also_fills_vad_buffer(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend.push_audio(_make_pcm_tone(1600))
        assert len(backend._vad_buffer) == 1600

    def test_push_audio_triggers_inference_at_chunk_size(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        # chunk_size_samples = 0.34 * 16000 = 5440
        chunk_samples = backend._streaming.chunk_size_samples

        # Push less than chunk — no inference
        backend.push_audio(_make_pcm_tone(chunk_samples - 100))
        self.mock_nemo.conformer_stream_step.assert_not_called()

        # Push remaining — should trigger inference
        backend.push_audio(_make_pcm_tone(200))
        self.mock_nemo.conformer_stream_step.assert_called_once()

    def test_push_audio_updates_current_text(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        chunk_samples = backend._streaming.chunk_size_samples
        backend.push_audio(_make_pcm_tone(chunk_samples + 100))
        assert backend._streaming.current_text == "hello world"

    def test_get_partial_empty_returns_none(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        result = backend.get_partial()
        assert result is None

    def test_get_partial_returns_current_text(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend._streaming.current_text = "hello world"
        result = backend.get_partial()
        assert result is not None
        assert result.text == "hello world"
        assert result.is_partial is True
        assert result.is_endpoint is False

    def test_multiple_chunks_call_inference_multiple_times(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        chunk_samples = backend._streaming.chunk_size_samples
        # Push 2 full chunks
        backend.push_audio(_make_pcm_tone(chunk_samples * 2 + 100))
        assert self.mock_nemo.conformer_stream_step.call_count == 2

    def test_cache_state_updated_after_step(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        old_channel = backend._streaming.cache_last_channel
        chunk_samples = backend._streaming.chunk_size_samples
        backend.push_audio(_make_pcm_tone(chunk_samples + 100))
        # Cache should have been updated (different object after mock returns new one)
        assert backend._streaming.cache_last_channel is not old_channel
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestPushAudioAndPartial -v
```

Expected: FAIL (push_audio is a no-op stub, get_partial returns None)

- [ ] **Step 3: Implement `push_audio` and `get_partial`**

Replace the stubs in `NemoBackend`:

```python
    def push_audio(self, pcm_data: bytes) -> None:
        import torch

        n_samples = len(pcm_data) // 2
        if n_samples == 0:
            return

        samples_i16 = np.frombuffer(pcm_data[: n_samples * 2], dtype=np.int16)
        audio_f32 = samples_i16.astype(np.float32) / 32768.0

        self._streaming.audio_buffer = np.append(
            self._streaming.audio_buffer, audio_f32
        )
        self._vad_buffer = np.append(self._vad_buffer, audio_f32)

        # Run inference when enough audio has accumulated
        s = self._streaming
        while len(s.audio_buffer) >= s.chunk_size_samples:
            chunk = s.audio_buffer[: s.chunk_size_samples]
            s.audio_buffer = s.audio_buffer[s.chunk_size_samples:]

            model = _get_nemo_model()
            with _infer_lock:
                # Feed raw audio to streaming buffer for feature extraction
                s.streaming_buffer.append_audio_chunk(chunk)

                # Iterate to get mel features
                for processed_signal, processed_signal_length in s.streaming_buffer:
                    # Prepend pre-encode cache
                    if s.cache_pre_encode is not None:
                        processed_signal = torch.cat(
                            [s.cache_pre_encode.to(processed_signal.device), processed_signal],
                            dim=1,
                        )

                    with torch.no_grad():
                        (
                            pred_out,
                            transcribed_texts,
                            s.cache_last_channel,
                            s.cache_last_time,
                            s.cache_last_channel_len,
                            s.previous_hypotheses,
                        ) = model.conformer_stream_step(
                            processed_signal=processed_signal,
                            processed_signal_length=processed_signal_length,
                            cache_last_channel=s.cache_last_channel,
                            cache_last_time=s.cache_last_time,
                            cache_last_channel_len=s.cache_last_channel_len,
                            keep_all_outputs=False,
                            previous_hypotheses=s.previous_hypotheses,
                            previous_pred_out=s.previous_pred_out,
                            drop_extra_pre_encoded=True,
                            return_transcription=True,
                        )

                    s.previous_pred_out = pred_out

                    if transcribed_texts and transcribed_texts[0]:
                        s.current_text = transcribed_texts[0]

    def get_partial(self) -> ASRResult | None:
        if not self._streaming.current_text:
            return None
        return ASRResult(
            text=self._streaming.current_text,
            is_partial=True,
            is_endpoint=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestPushAudioAndPartial -v
```

Expected: All 9 tests PASS

- [ ] **Step 5: Run all tests**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py -v
```

Expected: All 13 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backends/nemo.py tests/mock/test_nemo_backend.py
git commit -m "feat: implement push_audio with cache-aware streaming and get_partial"
```

---

## Task 4: Implement `finalize`

**Files:**
- Modify: `backends/nemo.py`
- Modify: `tests/mock/test_nemo_backend.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/mock/test_nemo_backend.py`:

```python
class TestFinalize:
    def setup_method(self):
        self.mock_nemo = _make_mock_nemo_model()
        self.mock_vad = _make_mock_vad_model()
        self.mock_streaming_buf = _make_mock_streaming_buffer()
        self.nemo_patcher = patch(
            "backends.nemo._get_nemo_model", return_value=self.mock_nemo
        )
        self.vad_patcher = patch(
            "backends.nemo._get_vad_base_model", return_value=self.mock_vad
        )
        self.buf_patcher = patch(
            "backends.nemo.CacheAwareStreamingAudioBuffer",
            return_value=self.mock_streaming_buf,
        )
        self.nemo_patcher.start()
        self.vad_patcher.start()
        self.buf_patcher.start()

    def teardown_method(self):
        self.nemo_patcher.stop()
        self.vad_patcher.stop()
        self.buf_patcher.stop()

    def test_finalize_empty_state(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        result = backend.finalize()
        assert result.text == ""
        assert result.is_partial is False
        assert result.is_endpoint is True

    def test_finalize_returns_current_text(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend._streaming.current_text = "existing text"
        # Push some remaining audio (less than chunk_size)
        backend._streaming.audio_buffer = np.array([0.1] * 1000, dtype=np.float32)
        result = backend.finalize()
        assert result.is_partial is False
        assert result.is_endpoint is True
        # After finalize, should have called conformer_stream_step
        self.mock_nemo.conformer_stream_step.assert_called()

    def test_finalize_uses_keep_all_outputs(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend._streaming.audio_buffer = np.array([0.1] * 1000, dtype=np.float32)
        backend.finalize()
        call_kwargs = self.mock_nemo.conformer_stream_step.call_args
        assert call_kwargs.kwargs.get("keep_all_outputs") is True

    def test_finalize_no_remaining_audio(self):
        """Finalize with no remaining audio returns current_text."""
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend._streaming.current_text = "already transcribed"
        result = backend.finalize()
        assert result.text == "already transcribed"
        self.mock_nemo.conformer_stream_step.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestFinalize -v
```

- [ ] **Step 3: Implement `finalize`**

Replace the stub in `NemoBackend`:

```python
    def finalize(self) -> ASRResult:
        import torch

        s = self._streaming

        # Flush remaining audio if any
        if len(s.audio_buffer) > 0:
            model = _get_nemo_model()
            # Pad to chunk_size if needed
            if len(s.audio_buffer) < s.chunk_size_samples:
                pad_len = s.chunk_size_samples - len(s.audio_buffer)
                s.audio_buffer = np.append(
                    s.audio_buffer, np.zeros(pad_len, dtype=np.float32)
                )

            chunk = s.audio_buffer[: s.chunk_size_samples]
            s.audio_buffer = s.audio_buffer[s.chunk_size_samples:]

            with _infer_lock:
                s.streaming_buffer.append_audio_chunk(chunk)

                for processed_signal, processed_signal_length in s.streaming_buffer:
                    if s.cache_pre_encode is not None:
                        processed_signal = torch.cat(
                            [s.cache_pre_encode.to(processed_signal.device), processed_signal],
                            dim=1,
                        )

                    with torch.no_grad():
                        (
                            pred_out,
                            transcribed_texts,
                            s.cache_last_channel,
                            s.cache_last_time,
                            s.cache_last_channel_len,
                            s.previous_hypotheses,
                        ) = model.conformer_stream_step(
                            processed_signal=processed_signal,
                            processed_signal_length=processed_signal_length,
                            cache_last_channel=s.cache_last_channel,
                            cache_last_time=s.cache_last_time,
                            cache_last_channel_len=s.cache_last_channel_len,
                            keep_all_outputs=True,
                            previous_hypotheses=s.previous_hypotheses,
                            previous_pred_out=s.previous_pred_out,
                            drop_extra_pre_encoded=True,
                            return_transcription=True,
                        )

                    if transcribed_texts and transcribed_texts[0]:
                        s.current_text = transcribed_texts[0]

        return ASRResult(
            text=s.current_text,
            is_partial=False,
            is_endpoint=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestFinalize -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Run all tests**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backends/nemo.py tests/mock/test_nemo_backend.py
git commit -m "feat: implement finalize with keep_all_outputs flush"
```

---

## Task 5: Implement `detect_endpoint` (Silero VAD)

**Files:**
- Modify: `backends/nemo.py`
- Modify: `tests/mock/test_nemo_backend.py`

Same Silero VAD pattern as whisper backend.

- [ ] **Step 1: Write failing tests**

Append to `tests/mock/test_nemo_backend.py`:

```python
class TestVAD:
    """Tests for Silero VAD-based endpoint detection.

    Uses persistent patchers so _get_nemo_model/_get_vad_base_model remain
    mocked when push_audio/detect_endpoint call them at runtime.
    """

    def _setup_patchers(self, mock_vad):
        self._mock_nemo = _make_mock_nemo_model()
        self._nemo_patcher = patch(
            "backends.nemo._get_nemo_model", return_value=self._mock_nemo
        )
        self._vad_patcher = patch(
            "backends.nemo._get_vad_base_model", return_value=mock_vad
        )
        self._buf_patcher = patch(
            "backends.nemo.CacheAwareStreamingAudioBuffer",
            return_value=_make_mock_streaming_buffer(),
        )
        self._nemo_patcher.start()
        self._vad_patcher.start()
        self._buf_patcher.start()

    def _teardown_patchers(self):
        self._nemo_patcher.stop()
        self._vad_patcher.stop()
        self._buf_patcher.stop()

    def test_silence_does_not_trigger_speech(self):
        mock_vad = _make_mock_vad_model(speech_prob=0.1)
        self._setup_patchers(mock_vad)
        try:
            backend = _make_backend(self._mock_nemo, mock_vad)
            backend.push_audio(_make_pcm_silence(512))
            backend.detect_endpoint()
            assert backend._in_speech is False
        finally:
            self._teardown_patchers()

    def test_speech_triggers_in_speech(self):
        mock_vad = _make_mock_vad_model(speech_prob=0.9)
        self._setup_patchers(mock_vad)
        try:
            backend = _make_backend(self._mock_nemo, mock_vad)
            backend.push_audio(_make_pcm_tone(512))
            backend.detect_endpoint()
            assert backend._in_speech is True
        finally:
            self._teardown_patchers()

    def test_endpoint_after_sustained_silence(self):
        call_count = [0]
        speech_chunks = 2

        def _vad_side_effect(chunk, sr):
            call_count[0] += 1
            prob = 0.9 if call_count[0] <= speech_chunks else 0.1
            result = MagicMock()
            result.item.return_value = prob
            return result

        mock_vad = MagicMock()
        mock_vad.side_effect = _vad_side_effect
        mock_vad.reset_states = MagicMock()

        self._setup_patchers(mock_vad)
        try:
            backend = _make_backend(self._mock_nemo, mock_vad)
            backend.push_audio(_make_pcm_tone(1024))
            backend.detect_endpoint()
            assert backend._in_speech is True

            backend.push_audio(_make_pcm_silence(8192))
            result = backend.detect_endpoint()
            assert result is True
        finally:
            self._teardown_patchers()

    def test_endpoint_fires_once(self):
        call_count = [0]
        speech_chunks = 2

        def _vad_side_effect(chunk, sr):
            call_count[0] += 1
            prob = 0.9 if call_count[0] <= speech_chunks else 0.1
            result = MagicMock()
            result.item.return_value = prob
            return result

        mock_vad = MagicMock()
        mock_vad.side_effect = _vad_side_effect
        mock_vad.reset_states = MagicMock()

        self._setup_patchers(mock_vad)
        try:
            backend = _make_backend(self._mock_nemo, mock_vad)
            backend.push_audio(_make_pcm_tone(1024))
            backend.detect_endpoint()
            backend.push_audio(_make_pcm_silence(8192))
            assert backend.detect_endpoint() is True
            assert backend.detect_endpoint() is False
        finally:
            self._teardown_patchers()

    def test_speech_resets_silence_accumulator(self):
        call_count = [0]
        pattern = [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]

        def _vad_side_effect(chunk, sr):
            idx = min(call_count[0], len(pattern) - 1)
            call_count[0] += 1
            result = MagicMock()
            result.item.return_value = pattern[idx]
            return result

        mock_vad = MagicMock()
        mock_vad.side_effect = _vad_side_effect
        mock_vad.reset_states = MagicMock()

        self._setup_patchers(mock_vad)
        try:
            backend = _make_backend(self._mock_nemo, mock_vad)
            backend.push_audio(_make_pcm_tone(6144))
            result = backend.detect_endpoint()
            assert result is False
        finally:
            self._teardown_patchers()

    def test_no_endpoint_without_speech(self):
        mock_vad = _make_mock_vad_model(speech_prob=0.1)
        self._setup_patchers(mock_vad)
        try:
            backend = _make_backend(self._mock_nemo, mock_vad)
            backend.push_audio(_make_pcm_silence(16000))
            assert backend.detect_endpoint() is False
        finally:
            self._teardown_patchers()


class TestResetSegment:
    def setup_method(self):
        self.mock_nemo = _make_mock_nemo_model()
        self.mock_vad = _make_mock_vad_model(speech_prob=0.9)
        self.nemo_patcher = patch(
            "backends.nemo._get_nemo_model", return_value=self.mock_nemo
        )
        self.vad_patcher = patch(
            "backends.nemo._get_vad_base_model", return_value=self.mock_vad
        )
        self.buf_patcher = patch(
            "backends.nemo.CacheAwareStreamingAudioBuffer",
            return_value=_make_mock_streaming_buffer(),
        )
        self.nemo_patcher.start()
        self.vad_patcher.start()
        self.buf_patcher.start()

    def teardown_method(self):
        self.nemo_patcher.stop()
        self.vad_patcher.stop()
        self.buf_patcher.stop()

    def test_reset_clears_all_state(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        backend.push_audio(_make_pcm_tone(1600))
        backend.detect_endpoint()

        old_streaming = backend._streaming
        backend.reset_segment()

        assert backend._streaming is not old_streaming
        assert backend._streaming.current_text == ""
        assert backend._streaming.previous_hypotheses is None
        assert backend._streaming.previous_pred_out is None
        assert backend._in_speech is False
        assert backend._silence_ms_accum == 0.0
        assert backend._endpoint_fired is False
        assert len(backend._vad_buffer) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestVAD tests/mock/test_nemo_backend.py::TestResetSegment -v
```

- [ ] **Step 3: Implement `detect_endpoint`**

Replace the stub in `NemoBackend`:

```python
    def detect_endpoint(self) -> bool:
        # Process vad_buffer in 512-sample chunks
        # Note: torch may not be available in mock tests, but the mock VAD
        # model accepts any input type, so we pass numpy directly when
        # torch is unavailable.
        while len(self._vad_buffer) >= 512:
            chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]

            try:
                import torch
                audio_input = torch.tensor(chunk, dtype=torch.float32)
            except ImportError:
                audio_input = chunk
            speech_prob = self._vad_model(audio_input, self._sample_rate).item()

            chunk_ms = 512 / self._sample_rate * 1000  # 32ms at 16kHz

            if speech_prob >= settings.nemo_vad_threshold:
                self._in_speech = True
                self._silence_ms_accum = 0.0
            elif self._in_speech:
                self._silence_ms_accum += chunk_ms

        if (
            self._in_speech
            and not self._endpoint_fired
            and self._silence_ms_accum >= settings.vad_silence_ms
        ):
            self._endpoint_fired = True
            return True

        return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py::TestVAD tests/mock/test_nemo_backend.py::TestResetSegment -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Run all tests**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backends/nemo.py tests/mock/test_nemo_backend.py
git commit -m "feat: implement detect_endpoint with Silero VAD and reset_segment"
```

---

## Task 6: Run All Tests and Update Docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run full mock test suite**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v
```

Expected: ALL tests pass (existing + new nemo tests)

- [ ] **Step 2: Update CLAUDE.md**

In the Commands section, add:

```markdown
# Install with NeMo backend (requires CUDA)
uv sync --group dev --extra nemo

# Start the server (nemo backend, requires CUDA)
STT_BACKEND=nemo STT_NEMO_MODEL=stt_en_fastconformer_transducer_large_streaming uvicorn app:app --host 0.0.0.0 --port 8000
```

In the Architecture section, add to the bullet list:

```markdown
- **`backends/nemo.py`** — NeMo ASR backend using cache-aware streaming with FastConformer. Per-connection encoder caches + Silero VAD. Global singleton model with `_infer_lock`. Inference in `push_audio()` (incremental, like qwen3). `CacheAwareStreamingAudioBuffer` handles mel-spectrogram feature extraction.
```

In the Configuration section, update:

```markdown
All settings via `STT_` prefixed env vars (see `config.py`). Key ones: `STT_BACKEND` (mock/qwen3/nemo), ...
```

Add a new NeMo Backend section:

```markdown
## NeMo Backend

Uses NeMo's cache-aware streaming with FastConformer for low-latency incremental transcription.

**Key characteristics:**
- Inference in `push_audio()` (incremental, like qwen3) — each chunk processed with cached encoder context
- Silero VAD (neural) for endpoint detection — per-connection model instances
- `CacheAwareStreamingAudioBuffer` handles mel-spectrogram feature extraction and normalization
- Per-connection cache state: `cache_last_channel`, `cache_last_time`, `cache_pre_encode`, etc.
- Language is model-specific (change `STT_NEMO_MODEL` to switch language)

**Config:** `STT_NEMO_MODEL` (default `stt_en_fastconformer_transducer_large_streaming`), `STT_NEMO_CHUNK_SIZE` (default `0.34`s), `STT_NEMO_SHIFT_SIZE` (default `0.34`s), `STT_NEMO_LEFT_CHUNKS` (default `2`), `STT_NEMO_VAD_THRESHOLD` (default `0.5`). Reuses `STT_VAD_SILENCE_MS`.
```

In the Testing Notes section, update the test structure:

```
tests/
├── mock/     # Tests that run with STT_BACKEND=mock (no GPU needed)
├── qwen3/   # Tests that require CUDA + qwen3 backend
└── nemo/    # Tests that require CUDA + nemo backend (future)
```

Update the Running Tests table:

```markdown
| nemo | `STT_BACKEND=nemo .venv/bin/python -m pytest tests/nemo/ -v` | Yes |
```

Add to Mock Tests section:

```markdown
- NeMo backend tests mock `_get_nemo_model`, `_get_vad_base_model`, and `CacheAwareStreamingAudioBuffer`
```

- [ ] **Step 3: Run full test suite to confirm nothing broken**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py tests/mock/test_qwen3_backend.py tests/mock/test_mock_backend.py tests/mock/test_schema.py tests/mock/test_session.py -v
```

Expected: ALL tests pass

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add nemo backend documentation to CLAUDE.md"
```

---

## Verification Checklist

After all tasks are complete, verify:

- [ ] `STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v` — all tests pass
- [ ] `backends/nemo.py` implements full `ASRBackend` protocol (configure, push_audio, get_partial, finalize, detect_endpoint, reset_segment, close)
- [ ] Lazy imports: `nemo.collections.asr`, `torch`, `omegaconf` only imported inside functions
- [ ] Registry includes `"nemo"` entry
- [ ] `app.py` preloads nemo model on startup
- [ ] `pyproject.toml` has `nemo` optional extra
- [ ] `CLAUDE.md` documents nemo backend
- [ ] All commits on `feature/nemo-backend` branch
