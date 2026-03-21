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
        buf.insert(words, offset=0.0)
        committed = buf.flush()
        assert committed == []
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
        buf.pop_commited(0.6)
        assert len(buf.commited_in_buffer) == 1
        assert buf.commited_in_buffer[0][2] == "world"

    def test_insert_filters_old_words(self):
        """insert() drops words before last_commited_time."""
        buf = self._make_buffer()
        words1 = [(0.0, 0.5, "hello")]
        buf.insert(words1, offset=0.0)
        buf.flush()
        buf.insert(words1, offset=0.0)
        buf.flush()
        words2 = [(0.0, 0.3, "old"), (0.5, 1.0, "world")]
        buf.insert(words2, offset=0.0)
        assert all(w[0] >= buf.last_commited_time - 0.1 for w in buf.new)


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
    """Create a mock faster-whisper WhisperModel."""
    mock_model = MagicMock()

    def _mock_transcribe(audio, **kwargs):
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
    """Create a mock Silero VAD model."""
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
        backend.close()

    def test_close_idempotent(self):
        backend = _make_backend()
        backend.close()
        backend.close()


class TestPushAudio:
    def test_push_audio_accumulates_float32(self):
        import numpy as np

        backend = _make_backend()
        pcm = _make_pcm_tone(1600, amplitude=5000)
        backend.push_audio(pcm)

        assert backend._streaming.audio_buffer.dtype == np.float32
        assert len(backend._streaming.audio_buffer) == 1600

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
