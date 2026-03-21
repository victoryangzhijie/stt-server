# Whisper Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `whisper` STT backend using `faster-whisper` with streaming transcription (Local Agreement Policy from whisper_streaming) and Silero VAD for endpoint detection.

**Architecture:** Single file `backends/whisper.py` implementing `ASRBackend` protocol. Global singleton whisper model + per-connection Silero VAD clones. Streaming engine uses `_HypothesisBuffer` for text stability (ported from reference). Inference in `get_partial()`/`finalize()`, not `push_audio()`.

**Tech Stack:** `faster-whisper` (CTranslate2), `torch` (Silero VAD), `numpy`, `copy.deepcopy`

**Spec:** `docs/superpowers/specs/2026-03-21-whisper-backend-design.md`

**Reference code:** `/Users/rason/workspaces/whisper_streaming/whisper_online.py` (HypothesisBuffer at line 359, OnlineASRProcessor at line 426)

**Branch:** `feature/whisper-backend` (create from `master`)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `backends/whisper.py` | WhisperBackend, _HypothesisBuffer, _StreamingState, singletons, Silero VAD |
| Create | `tests/mock/test_whisper_backend.py` | All mock tests for whisper backend |
| Modify | `backends/registry.py` | Add `"whisper"` entry |
| Modify | `config.py` | Add `STT_WHISPER_*` settings |
| Modify | `app.py` | Add whisper preload on startup |
| Modify | `pyproject.toml` | Add `whisper` optional extra |
| Modify | `CLAUDE.md` | Add whisper docs |

---

## Task 1: Create Feature Branch and Config

**Files:**
- Modify: `config.py:6-49`
- Modify: `pyproject.toml:16-17`
- Modify: `backends/registry.py:7-10`

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b feature/whisper-backend master
```

- [ ] **Step 2: Add whisper config fields to `config.py`**

Add these fields to the `Settings` class after the existing qwen-asr streaming config (after line 46):

```python
    # Whisper backend config
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 5
    whisper_vad_threshold: float = 0.5
```

- [ ] **Step 3: Add whisper optional dependency to `pyproject.toml`**

Add after the `qwen3` entry in `[project.optional-dependencies]`:

```toml
whisper = ["faster-whisper", "torch"]
```

- [ ] **Step 4: Add whisper to backend registry in `backends/registry.py`**

Add to the `_BACKENDS` dict:

```python
_BACKENDS: dict[str, str] = {
    "qwen3": "backends.qwen3.Qwen3Backend",
    "mock": "backends.mock.MockBackend",
    "whisper": "backends.whisper.WhisperBackend",
}
```

- [ ] **Step 5: Add whisper preload to `app.py`**

In the `lifespan` function, after the qwen3 preload block (after line 20), add:

```python
    elif settings.backend == "whisper":
        from backends.whisper import preload_model

        preload_model()
```

- [ ] **Step 6: Commit**

```bash
git add config.py pyproject.toml backends/registry.py app.py
git commit -m "feat: add whisper backend config, registry entry, and dependency"
```

---

## Task 2: Implement `_HypothesisBuffer`

**Files:**
- Create: `backends/whisper.py`
- Create: `tests/mock/test_whisper_backend.py`

The `_HypothesisBuffer` is ported from `/Users/rason/workspaces/whisper_streaming/whisper_online.py` lines 359-424. It tracks text stability using word-level local agreement between consecutive transcriptions.

- [ ] **Step 1: Write failing tests for `_HypothesisBuffer`**

Create `tests/mock/test_whisper_backend.py`:

```python
from __future__ import annotations

import pytest


class TestHypothesisBuffer:
    """Tests for _HypothesisBuffer — word-level local agreement."""

    def _make_buffer(self):
        from backends.whisper import _HypothesisBuffer
        return _HypothesisBuffer()

    def test_flush_returns_common_prefix(self):
        """Two identical inserts → words confirmed on flush."""
        buf = self._make_buffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

        # First insert: goes into buffer, flush moves buffer→new, returns nothing
        buf.insert(words, offset=0.0)
        committed = buf.flush()
        assert committed == []

        # Second identical insert: flush finds common prefix → committed
        buf.insert(words, offset=0.0)
        committed = buf.flush()
        assert len(committed) == 2
        assert committed[0][2] == "hello"
        assert committed[1][2] == "world"

    def test_unstable_text_not_committed(self):
        """Differing inserts → nothing committed."""
        buf = self._make_buffer()

        buf.insert([(0.0, 0.5, "hello")], offset=0.0)
        buf.flush()

        buf.insert([(0.0, 0.5, "goodbye")], offset=0.0)
        committed = buf.flush()
        assert committed == []

    def test_partial_agreement(self):
        """Only the matching prefix is committed."""
        buf = self._make_buffer()

        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "world")], offset=0.0)
        buf.flush()

        buf.insert([(0.0, 0.5, "hello"), (0.5, 1.0, "there")], offset=0.0)
        committed = buf.flush()
        assert len(committed) == 1
        assert committed[0][2] == "hello"

    def test_complete_returns_buffered(self):
        """complete() returns unconfirmed words in buffer."""
        buf = self._make_buffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]
        buf.insert(words, offset=0.0)
        buf.flush()

        buffered = buf.complete()
        assert len(buffered) == 2
        assert buffered[0][2] == "hello"

    def test_pop_committed_removes_old(self):
        """pop_commited trims committed words before a timestamp."""
        buf = self._make_buffer()
        words = [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

        buf.insert(words, offset=0.0)
        buf.flush()
        buf.insert(words, offset=0.0)
        buf.flush()

        # Now committed has both words. Pop entries ending before 0.6
        buf.pop_commited(0.6)
        assert len(buf.commited_in_buffer) == 1
        assert buf.commited_in_buffer[0][2] == "world"

    def test_insert_filters_old_words(self):
        """insert() drops words before last_commited_time."""
        buf = self._make_buffer()

        # Commit "hello" first
        words1 = [(0.0, 0.5, "hello")]
        buf.insert(words1, offset=0.0)
        buf.flush()
        buf.insert(words1, offset=0.0)
        buf.flush()
        # last_commited_time is now 0.5

        # New insert with words starting before last_commited_time are filtered
        words2 = [(0.0, 0.3, "old"), (0.5, 1.0, "world")]
        buf.insert(words2, offset=0.0)
        # "old" should be filtered (start 0.0 < 0.5 - 0.1 = 0.4)
        assert all(w[0] >= buf.last_commited_time - 0.1 for w in buf.new)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestHypothesisBuffer -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` (backends.whisper doesn't exist yet)

- [ ] **Step 3: Implement `_HypothesisBuffer` in `backends/whisper.py`**

Create `backends/whisper.py` with the _HypothesisBuffer class. Port from reference (`/Users/rason/workspaces/whisper_streaming/whisper_online.py` lines 359-424):

```python
from __future__ import annotations

import copy
import threading

import numpy as np
import structlog

from backends.base import ASRResult
from config import settings

logger = structlog.get_logger()


# ── HypothesisBuffer (ported from whisper_streaming) ─────────────────────────


class _HypothesisBuffer:
    """Tracks text stability using word-level local agreement.

    Each word is a tuple (start_time, end_time, text).
    Ported from whisper_streaming/whisper_online.py HypothesisBuffer.
    """

    def __init__(self) -> None:
        self.commited_in_buffer: list[tuple[float, float, str]] = []
        self.buffer: list[tuple[float, float, str]] = []
        self.new: list[tuple[float, float, str]] = []
        self.last_commited_time: float = 0.0
        self.last_commited_word: str | None = None

    def insert(self, new: list[tuple[float, float, str]], offset: float) -> None:
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, _b, _t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for _j in range(i):
                                self.new.pop(0)
                            break

    def flush(self) -> list[tuple[float, float, str]]:
        commit: list[tuple[float, float, str]] = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float) -> None:
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[tuple[float, float, str]]:
        return self.buffer
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestHypothesisBuffer -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backends/whisper.py tests/mock/test_whisper_backend.py
git commit -m "feat: implement _HypothesisBuffer with word-level local agreement"
```

---

## Task 3: Implement `_StreamingState` and Global Singletons

**Files:**
- Modify: `backends/whisper.py`
- Modify: `tests/mock/test_whisper_backend.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/mock/test_whisper_backend.py`:

```python
import struct
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pcm_silence(n_samples: int) -> bytes:
    """PCM16LE silence (all zeros)."""
    return b"\x00\x00" * n_samples


def _make_pcm_tone(n_samples: int, amplitude: int = 10000) -> bytes:
    """PCM16LE constant-amplitude samples (square wave)."""
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _make_mock_whisper_model():
    """Create a mock faster-whisper WhisperModel.

    Returns segments with timestamped words when transcribe() is called.
    """
    mock_model = MagicMock()

    def _mock_transcribe(audio, **kwargs):
        """Return (segments_iter, info). Each segment has .words list."""
        seg = MagicMock()
        word1 = MagicMock()
        word1.start = 0.0
        word1.end = 0.5
        word1.word = "hello"
        word2 = MagicMock()
        word2.start = 0.5
        word2.end = 1.0
        word2.word = " world"
        seg.words = [word1, word2]
        seg.no_speech_prob = 0.0
        seg.end = 1.0
        info = MagicMock()
        return iter([seg]), info

    mock_model.transcribe.side_effect = _mock_transcribe
    return mock_model


def _make_mock_vad_model(speech_prob: float = 0.0):
    """Create a mock Silero VAD model.

    Returns a configurable speech probability on each call.
    The model must support reset_states() for deepcopy compatibility.
    """
    mock_vad = MagicMock()
    mock_vad.return_value = MagicMock(item=MagicMock(return_value=speech_prob))
    mock_vad.reset_states = MagicMock()
    return mock_vad


def _make_backend(mock_whisper=None, mock_vad=None, language="auto"):
    """Create and configure a WhisperBackend with models mocked out."""
    if mock_whisper is None:
        mock_whisper = _make_mock_whisper_model()
    if mock_vad is None:
        mock_vad = _make_mock_vad_model()

    with (
        patch("backends.whisper._get_whisper_model", return_value=mock_whisper),
        patch("backends.whisper._get_vad_base_model", return_value=mock_vad),
    ):
        from backends.whisper import WhisperBackend

        b = WhisperBackend()
        b.configure(sample_rate=16000, language=language)
    return b


class TestConfigureAndClose:
    def test_configure_initializes_state(self):
        backend = _make_backend()
        assert backend._sample_rate == 16000
        assert backend._language == "auto"
        assert backend._streaming is not None
        assert backend._in_speech is False
        assert backend._endpoint_fired is False

    def test_configure_language(self):
        backend = _make_backend(language="en")
        assert backend._streaming.language == "en"

    def test_configure_auto_language(self):
        backend = _make_backend(language="auto")
        assert backend._streaming.language is None

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
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestConfigureAndClose -v
```

Expected: FAIL (WhisperBackend doesn't exist yet)

- [ ] **Step 3: Implement `_StreamingState`, global singletons, and `WhisperBackend` skeleton**

Append to `backends/whisper.py` after the `_HypothesisBuffer` class:

```python
# ── StreamingState ───────────────────────────────────────────────────────────


class _StreamingState:
    """Per-connection streaming context."""

    def __init__(self, language: str | None, buffer_trimming_sec: float = 15.0) -> None:
        self.audio_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.hypothesis = _HypothesisBuffer()
        self.committed: list[tuple[float, float, str]] = []
        self.committed_text: str = ""
        self.buffer_trimming_sec: float = buffer_trimming_sec
        self.buffer_time_offset: float = 0.0
        self.language: str | None = language
        self.sample_rate: int = 16000


# ── Module-level whisper model singleton ─────────────────────────────────────


_whisper_model: object | None = None
_vad_base_model: object | None = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_whisper_model() -> object:
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel

                _whisper_model = WhisperModel(
                    settings.whisper_model,
                    compute_type=settings.whisper_compute_type,
                )
                logger.info(
                    "whisper.model.loaded",
                    model=settings.whisper_model,
                    compute_type=settings.whisper_compute_type,
                )
    return _whisper_model


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
                logger.info("whisper.vad_base.loaded")
    return _vad_base_model


def preload_model() -> None:
    """Preload both whisper and VAD models at startup."""
    logger.info("whisper.preload.start")
    _get_whisper_model()
    _get_vad_base_model()
    logger.info("whisper.preload.done")


# ── WhisperBackend ───────────────────────────────────────────────────────────


class WhisperBackend:
    """Whisper ASR backend using faster-whisper with local agreement streaming."""

    def configure(
        self,
        sample_rate: int,
        language: str = "auto",
        hotwords: list[str] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._language = language

        lang = None if language == "auto" else language
        self._streaming = _StreamingState(language=lang)

        # Per-connection Silero VAD (deepcopy for independent LSTM state)
        vad_base = _get_vad_base_model()
        self._vad_model = copy.deepcopy(vad_base)
        self._vad_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._in_speech: bool = False
        self._silence_ms_accum: float = 0.0
        self._endpoint_fired: bool = False

    def push_audio(self, pcm_data: bytes) -> None:
        pass  # implemented in Task 4

    def get_partial(self) -> ASRResult | None:
        return None  # implemented in Task 5

    def finalize(self) -> ASRResult:
        return ASRResult(text="", is_partial=False, is_endpoint=True)  # implemented in Task 5

    def detect_endpoint(self) -> bool:
        return False  # implemented in Task 6

    def reset_segment(self) -> None:
        lang = None if self._language == "auto" else self._language
        self._streaming = _StreamingState(language=lang)

        self._vad_model = copy.deepcopy(_get_vad_base_model())
        self._vad_buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._endpoint_fired = False

    def close(self) -> None:
        pass  # model is shared singleton, not per-connection
```

Note: `numpy`, `copy`, and `structlog` are imported at module level (always available). Only `faster_whisper` and `torch` use lazy imports inside functions (they're in the optional `whisper` extra), so `STT_BACKEND=mock` works without them installed.

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestConfigureAndClose -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backends/whisper.py tests/mock/test_whisper_backend.py
git commit -m "feat: add WhisperBackend skeleton with _StreamingState and singletons"
```

---

## Task 4: Implement `push_audio`

**Files:**
- Modify: `backends/whisper.py`
- Modify: `tests/mock/test_whisper_backend.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/mock/test_whisper_backend.py`:

```python
class TestPushAudio:
    def test_push_audio_accumulates_float32(self):
        import numpy as np

        backend = _make_backend()
        pcm = _make_pcm_tone(1600, amplitude=5000)
        backend.push_audio(pcm)

        # Audio buffer should have float32 samples
        assert backend._streaming.audio_buffer.dtype == np.float32
        assert len(backend._streaming.audio_buffer) == 1600

        # Values should be normalized to [-1, 1]
        expected = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(
            backend._streaming.audio_buffer, expected
        )

    def test_push_audio_appends(self):
        backend = _make_backend()
        backend.push_audio(_make_pcm_tone(800))
        backend.push_audio(_make_pcm_tone(800))
        assert len(backend._streaming.audio_buffer) == 1600

    def test_push_audio_also_fills_vad_buffer(self):
        backend = _make_backend()
        backend.push_audio(_make_pcm_tone(1600))
        assert len(backend._vad_buffer) == 1600

    def test_push_audio_empty_is_noop(self):
        backend = _make_backend()
        backend.push_audio(b"")
        assert len(backend._streaming.audio_buffer) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestPushAudio -v
```

Expected: FAIL (push_audio is a no-op stub)

- [ ] **Step 3: Implement `push_audio`**

Replace the stub in `WhisperBackend.push_audio`:

```python
    def push_audio(self, pcm_data: bytes) -> None:
        n_samples = len(pcm_data) // 2
        if n_samples == 0:
            return

        samples_i16 = np.frombuffer(pcm_data[: n_samples * 2], dtype=np.int16)
        audio_f32 = samples_i16.astype(np.float32) / 32768.0

        self._streaming.audio_buffer = np.append(
            self._streaming.audio_buffer, audio_f32
        )
        self._vad_buffer = np.append(self._vad_buffer, audio_f32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestPushAudio -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backends/whisper.py tests/mock/test_whisper_backend.py
git commit -m "feat: implement WhisperBackend.push_audio with PCM16LE to float32 conversion"
```

---

## Task 5: Implement `get_partial` and `finalize` (Streaming Engine)

**Files:**
- Modify: `backends/whisper.py`
- Modify: `tests/mock/test_whisper_backend.py`

This is the core streaming logic. `get_partial()` re-transcribes the full audio buffer and uses `_HypothesisBuffer` for local agreement. `finalize()` commits all remaining text.

Reference: `/Users/rason/workspaces/whisper_streaming/whisper_online.py` lines 477-527 (`process_iter`) and lines 603-611 (`finish`).

- [ ] **Step 1: Write failing tests**

Add to `tests/mock/test_whisper_backend.py`:

```python
class TestStreamingEngine:
    def setup_method(self):
        self.mock_whisper = _make_mock_whisper_model()
        self.mock_vad = _make_mock_vad_model()
        self.whisper_patcher = patch(
            "backends.whisper._get_whisper_model", return_value=self.mock_whisper
        )
        self.vad_patcher = patch(
            "backends.whisper._get_vad_base_model", return_value=self.mock_vad
        )
        self.whisper_patcher.start()
        self.vad_patcher.start()

    def teardown_method(self):
        self.whisper_patcher.stop()
        self.vad_patcher.stop()

    def test_get_partial_empty_returns_none(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        result = backend.get_partial()
        assert result is None

    def test_get_partial_with_audio(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        # Push enough audio so transcription runs
        backend.push_audio(_make_pcm_tone(16000))  # 1 second

        result = backend.get_partial()
        assert result is not None
        assert result.is_partial is True
        assert result.is_endpoint is False
        # First call: words go into buffer but nothing committed yet
        # The text should contain the buffered (incomplete) words
        assert "hello" in result.text or "world" in result.text

    def test_committed_text_grows(self):
        """Two get_partial calls with same words → text gets committed."""
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend.push_audio(_make_pcm_tone(16000))

        # First partial: buffer filled, nothing committed
        r1 = backend.get_partial()
        # Second partial: same words → local agreement → committed
        r2 = backend.get_partial()

        assert r2 is not None
        assert "hello" in r2.text

    def test_get_partial_calls_transcribe(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend.push_audio(_make_pcm_tone(16000))
        backend.get_partial()

        self.mock_whisper.transcribe.assert_called_once()
        call_kwargs = self.mock_whisper.transcribe.call_args
        assert call_kwargs[1].get("word_timestamps") is True

    def test_get_partial_passes_language(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad, language="en")
        backend.push_audio(_make_pcm_tone(16000))
        backend.get_partial()

        call_kwargs = self.mock_whisper.transcribe.call_args[1]
        assert call_kwargs.get("language") == "en"

    def test_get_partial_auto_language_passes_none(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad, language="auto")
        backend.push_audio(_make_pcm_tone(16000))
        backend.get_partial()

        call_kwargs = self.mock_whisper.transcribe.call_args[1]
        assert call_kwargs.get("language") is None

    def test_buffer_trimming(self):
        """Audio buffer trimmed when exceeding threshold."""
        import numpy as np

        backend = _make_backend(self.mock_whisper, self.mock_vad)
        # Set a very short trimming threshold for testing
        backend._streaming.buffer_trimming_sec = 1.0

        # Push 2 seconds of audio (exceeds 1s threshold)
        backend.push_audio(_make_pcm_tone(32000))
        assert len(backend._streaming.audio_buffer) == 32000

        # First partial: fills buffer
        backend.get_partial()
        # Second partial: commits words and should trigger trimming
        backend.get_partial()

        # Buffer should have been trimmed (exact size depends on committed timestamps)
        # At minimum, it should be less than the original 32000
        assert len(backend._streaming.audio_buffer) < 32000


class TestFinalize:
    def setup_method(self):
        self.mock_whisper = _make_mock_whisper_model()
        self.mock_vad = _make_mock_vad_model()
        self.whisper_patcher = patch(
            "backends.whisper._get_whisper_model", return_value=self.mock_whisper
        )
        self.vad_patcher = patch(
            "backends.whisper._get_vad_base_model", return_value=self.mock_vad
        )
        self.whisper_patcher.start()
        self.vad_patcher.start()

    def teardown_method(self):
        self.whisper_patcher.stop()
        self.vad_patcher.stop()

    def test_finalize_returns_full_text(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend.push_audio(_make_pcm_tone(16000))

        result = backend.finalize()
        assert result.is_partial is False
        assert result.is_endpoint is True
        # finalize should include all text (committed + buffered)
        assert isinstance(result.text, str)

    def test_finalize_empty_state(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        result = backend.finalize()
        assert result.text == ""
        assert result.is_partial is False
        assert result.is_endpoint is True

    def test_finalize_calls_transcribe(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend.push_audio(_make_pcm_tone(16000))
        backend.finalize()
        self.mock_whisper.transcribe.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestStreamingEngine tests/mock/test_whisper_backend.py::TestFinalize -v
```

Expected: FAIL (get_partial returns None always, finalize returns empty)

- [ ] **Step 3: Implement `get_partial` and `finalize`**

Replace the stubs in `WhisperBackend`:

```python
    def _build_prompt(self, s: _StreamingState) -> str:
        """Build initial_prompt from committed words BEFORE buffer_time_offset.

        Only words from the trimmed-away audio portion are used as prompt context.
        Words still inside the audio buffer are re-transcribed, so including them
        in the prompt would cause hallucination/repetition.

        Ported from whisper_streaming/whisper_online.py OnlineASRProcessor.prompt()
        """
        # Find committed words whose end time is before buffer_time_offset
        k = max(0, len(s.committed) - 1)
        while k > 0 and s.committed[k - 1][1] > s.buffer_time_offset:
            k -= 1

        prompt_words = s.committed[:k]
        prompt_texts: list[str] = []
        length = 0
        for _, _, t in reversed(prompt_words):
            length += len(t) + 1
            if length > 200:
                break
            prompt_texts.append(t)
        return "".join(reversed(prompt_texts))

    def get_partial(self) -> ASRResult | None:
        s = self._streaming
        if len(s.audio_buffer) == 0:
            return None

        model = _get_whisper_model()
        with _infer_lock:
            segments, _info = model.transcribe(
                s.audio_buffer,
                language=s.language,
                initial_prompt=self._build_prompt(s),
                beam_size=settings.whisper_beam_size,
                word_timestamps=True,
                condition_on_previous_text=True,
            )
            # Extract timestamped words from segments
            tsw: list[tuple[float, float, str]] = []
            for segment in segments:
                if segment.no_speech_prob > 0.9:
                    continue
                for word in segment.words:
                    tsw.append((word.start, word.end, word.word))

        # Feed into hypothesis buffer for local agreement
        s.hypothesis.insert(tsw, s.buffer_time_offset)
        committed = s.hypothesis.flush()
        s.committed.extend(committed)

        # Build committed text from newly committed words
        if committed:
            # faster-whisper includes spaces in word text, use "" separator
            s.committed_text += "".join(t for _, _, t in committed)

        # Buffer trimming: use segment-based trimming like reference
        if len(s.audio_buffer) / s.sample_rate > s.buffer_trimming_sec:
            self._trim_buffer(s)

        # Build partial text: committed + incomplete
        incomplete = s.hypothesis.complete()
        partial_text = s.committed_text + "".join(t for _, _, t in incomplete)

        if not partial_text.strip():
            return None

        return ASRResult(text=partial_text.strip(), is_partial=True, is_endpoint=False)

    def _trim_buffer(self, s: _StreamingState) -> None:
        """Trim audio buffer at a safe boundary, preserving context for dedup.

        Ported from whisper_streaming/whisper_online.py chunk_at().
        Trims at the second-to-last committed word (not the very last), leaving
        enough committed audio context for HypothesisBuffer's dedup logic.
        """
        if len(s.committed) < 2:
            return

        # Trim at the second-to-last committed word's end time
        # This preserves the last committed word's audio for dedup context
        trim_time = s.committed[-2][1]
        cut_seconds = trim_time - s.buffer_time_offset
        cut_samples = int(cut_seconds * s.sample_rate)

        if cut_samples > 0 and cut_samples < len(s.audio_buffer):
            s.audio_buffer = s.audio_buffer[cut_samples:]
            s.buffer_time_offset = trim_time
            s.hypothesis.pop_commited(trim_time)

    def finalize(self) -> ASRResult:
        s = self._streaming
        if len(s.audio_buffer) == 0:
            return ASRResult(
                text=s.committed_text.strip(),
                is_partial=False,
                is_endpoint=True,
            )

        model = _get_whisper_model()
        with _infer_lock:
            segments, _info = model.transcribe(
                s.audio_buffer,
                language=s.language,
                initial_prompt=self._build_prompt(s),
                beam_size=settings.whisper_beam_size,
                word_timestamps=True,
                condition_on_previous_text=True,
            )
            tsw: list[tuple[float, float, str]] = []
            for segment in segments:
                if segment.no_speech_prob > 0.9:
                    continue
                for word in segment.words:
                    tsw.append((word.start, word.end, word.word))

        # On finalize, commit everything
        s.hypothesis.insert(tsw, s.buffer_time_offset)
        flushed = s.hypothesis.flush()

        # Append flushed words to committed_text (same as get_partial does)
        if flushed:
            s.committed_text += "".join(t for _, _, t in flushed)

        # Also include any remaining buffered (incomplete) text
        remaining = s.hypothesis.complete()
        remaining_text = "".join(t for _, _, t in remaining)
        full_text = s.committed_text + remaining_text

        return ASRResult(
            text=full_text.strip(),
            is_partial=False,
            is_endpoint=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestStreamingEngine tests/mock/test_whisper_backend.py::TestFinalize -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backends/whisper.py tests/mock/test_whisper_backend.py
git commit -m "feat: implement get_partial and finalize with local agreement streaming"
```

---

## Task 6: Implement `detect_endpoint` (Silero VAD)

**Files:**
- Modify: `backends/whisper.py`
- Modify: `tests/mock/test_whisper_backend.py`

Reference: `/Users/rason/workspaces/whisper_streaming/silero_vad_iterator.py` for Silero VAD usage patterns.

- [ ] **Step 1: Write failing tests**

Add to `tests/mock/test_whisper_backend.py`:

```python
class TestVAD:
    """Tests for Silero VAD-based endpoint detection."""

    def _make_backend_with_vad(self, speech_prob=0.0):
        """Create backend with configurable VAD mock."""
        mock_whisper = _make_mock_whisper_model()
        mock_vad = _make_mock_vad_model(speech_prob=speech_prob)
        return _make_backend(mock_whisper, mock_vad), mock_vad

    def test_silence_does_not_trigger_speech(self):
        backend, _ = self._make_backend_with_vad(speech_prob=0.1)
        # Push 512 samples (one VAD chunk) of "silence" (low prob)
        backend.push_audio(_make_pcm_silence(512))
        backend.detect_endpoint()
        assert backend._in_speech is False

    def test_speech_triggers_in_speech(self):
        backend, mock_vad = self._make_backend_with_vad(speech_prob=0.9)
        backend.push_audio(_make_pcm_tone(512))
        backend.detect_endpoint()
        assert backend._in_speech is True

    def test_endpoint_after_sustained_silence(self):
        """Speech followed by enough silence triggers endpoint."""
        mock_whisper = _make_mock_whisper_model()

        # VAD returns high prob first, then low
        call_count = [0]
        speech_chunks = 2  # first 2 chunks are speech
        def _vad_side_effect(chunk, sr):
            call_count[0] += 1
            prob = 0.9 if call_count[0] <= speech_chunks else 0.1
            result = MagicMock()
            result.item.return_value = prob
            return result

        mock_vad = MagicMock()
        mock_vad.side_effect = _vad_side_effect
        mock_vad.reset_states = MagicMock()

        backend = _make_backend(mock_whisper, mock_vad)

        # Speech: 2 * 512 = 1024 samples
        backend.push_audio(_make_pcm_tone(1024))
        backend.detect_endpoint()
        assert backend._in_speech is True

        # Silence: enough to exceed vad_silence_ms (500ms = 8000 samples at 16kHz)
        # 8000 / 512 = ~16 chunks, each chunk = 32ms
        backend.push_audio(_make_pcm_silence(8192))
        result = backend.detect_endpoint()
        assert result is True

    def test_endpoint_fires_once(self):
        mock_whisper = _make_mock_whisper_model()
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

        backend = _make_backend(mock_whisper, mock_vad)
        backend.push_audio(_make_pcm_tone(1024))
        backend.detect_endpoint()
        backend.push_audio(_make_pcm_silence(8192))
        assert backend.detect_endpoint() is True
        # Second call: still silence, but endpoint already fired
        assert backend.detect_endpoint() is False

    def test_speech_resets_silence_accumulator(self):
        """Intermittent speech should reset silence accumulation."""
        mock_whisper = _make_mock_whisper_model()
        call_count = [0]
        # Pattern: 2 speech, 4 silence, 2 speech, 4 silence
        # 4 silence chunks = 4*32ms = 128ms < 500ms threshold
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

        backend = _make_backend(mock_whisper, mock_vad)
        # Push all audio at once (12 * 512 = 6144 samples)
        backend.push_audio(_make_pcm_tone(6144))
        result = backend.detect_endpoint()
        assert result is False

    def test_no_endpoint_without_speech(self):
        """Silence alone should never fire an endpoint."""
        backend, _ = self._make_backend_with_vad(speech_prob=0.1)
        backend.push_audio(_make_pcm_silence(16000))
        assert backend.detect_endpoint() is False


class TestResetSegment:
    def setup_method(self):
        self.mock_whisper = _make_mock_whisper_model()
        self.mock_vad = _make_mock_vad_model(speech_prob=0.9)
        self.whisper_patcher = patch(
            "backends.whisper._get_whisper_model", return_value=self.mock_whisper
        )
        self.vad_patcher = patch(
            "backends.whisper._get_vad_base_model", return_value=self.mock_vad
        )
        self.whisper_patcher.start()
        self.vad_patcher.start()

    def teardown_method(self):
        self.whisper_patcher.stop()
        self.vad_patcher.stop()

    def test_reset_clears_all_state(self):
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend.push_audio(_make_pcm_tone(1600))
        backend.detect_endpoint()

        old_streaming = backend._streaming
        backend.reset_segment()

        assert backend._streaming is not old_streaming
        assert backend._in_speech is False
        assert backend._silence_ms_accum == 0.0
        assert backend._endpoint_fired is False
        assert len(backend._vad_buffer) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestVAD tests/mock/test_whisper_backend.py::TestResetSegment -v
```

Expected: FAIL (detect_endpoint returns False always)

- [ ] **Step 3: Implement `detect_endpoint`**

Replace the stub in `WhisperBackend.detect_endpoint`:

```python
    def detect_endpoint(self) -> bool:
        import torch
        import numpy as np

        # Process vad_buffer in 512-sample chunks
        while len(self._vad_buffer) >= 512:
            chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]

            tensor = torch.tensor(chunk, dtype=torch.float32)
            speech_prob = self._vad_model(tensor, self._sample_rate).item()

            chunk_ms = 512 / self._sample_rate * 1000  # 32ms at 16kHz

            if speech_prob >= settings.whisper_vad_threshold:
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
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py::TestVAD tests/mock/test_whisper_backend.py::TestResetSegment -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backends/whisper.py tests/mock/test_whisper_backend.py
git commit -m "feat: implement detect_endpoint with Silero VAD"
```

---

## Task 7: Run All Tests and Update Docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run full mock test suite**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v
```

Expected: ALL tests pass (both existing tests and new whisper tests)

- [ ] **Step 2: Update CLAUDE.md**

Add whisper backend sections. In the Commands section, add:

```markdown
# Install with Whisper backend (requires CUDA for GPU acceleration)
uv sync --group dev --extra whisper

# Start the server (whisper backend, requires CUDA for GPU)
STT_BACKEND=whisper STT_WHISPER_MODEL=large-v3-turbo uvicorn app:app --host 0.0.0.0 --port 8000
```

In the Architecture section, add to the bullet list:

```markdown
- **`backends/whisper.py`** — Whisper backend using `faster-whisper` with local agreement streaming (ported from whisper_streaming). Per-connection Silero VAD for endpoint detection. Global singleton whisper model with `_infer_lock`. Inference in `get_partial()`/`finalize()` (re-transcribes full buffer each time, unlike qwen3's incremental approach).
```

In the Configuration section, update:

```markdown
All settings via `STT_` prefixed env vars (see `config.py`). Key ones: `STT_BACKEND` (mock/qwen3/whisper), ...
```

Add a new Whisper Backend section:

```markdown
## Whisper Backend

Uses `faster-whisper` with streaming via Local Agreement Policy (ported from `whisper_streaming`). Text is only emitted when confirmed stable across consecutive transcriptions.

**Key differences from qwen3:**
- Inference in `get_partial()`/`finalize()` (re-transcribes full audio buffer), not `push_audio()`
- Silero VAD (neural) instead of energy-based RMS
- Per-connection VAD model instances (Silero LSTM state not shareable)

**Config:** `STT_WHISPER_MODEL` (default `large-v3-turbo`), `STT_WHISPER_COMPUTE_TYPE` (default `float16`), `STT_WHISPER_BEAM_SIZE` (default `5`), `STT_WHISPER_VAD_THRESHOLD` (default `0.5`). Reuses `STT_VAD_SILENCE_MS`.
```

In the Testing Notes section, update the test structure:

```
tests/
├── mock/     # Tests that run with STT_BACKEND=mock (no GPU needed)
├── qwen3/   # Tests that require CUDA + qwen3 backend
└── whisper/  # Tests that require CUDA + whisper backend (future)
```

Update the Running Tests table:

```markdown
| whisper | `STT_BACKEND=whisper .venv/bin/python -m pytest tests/whisper/ -v` | Yes |
```

Add to Mock Tests section:

```markdown
- Whisper backend tests mock `_get_whisper_model` and `_get_vad_base_model` for unit tests
```

- [ ] **Step 3: Run full test suite again to confirm nothing broken**

```bash
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v
```

Expected: ALL tests pass

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add whisper backend documentation to CLAUDE.md"
```

---

## Verification Checklist

After all tasks are complete, verify:

- [ ] `STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v` — all tests pass
- [ ] `backends/whisper.py` implements full `ASRBackend` protocol (configure, push_audio, get_partial, finalize, detect_endpoint, reset_segment, close)
- [ ] Lazy imports: `faster_whisper` and `torch` are only imported inside functions
- [ ] Registry includes `"whisper"` entry
- [ ] `app.py` preloads whisper model on startup
- [ ] `pyproject.toml` has `whisper` optional extra
- [ ] `CLAUDE.md` documents whisper backend
- [ ] All commits on `feature/whisper-backend` branch
