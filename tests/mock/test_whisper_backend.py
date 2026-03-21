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
        backend.push_audio(_make_pcm_tone(16000))
        result = backend.get_partial()
        assert result is not None
        assert result.is_partial is True
        assert result.is_endpoint is False
        assert "hello" in result.text or "world" in result.text

    def test_committed_text_grows(self):
        """Two get_partial calls with same words → text gets committed."""
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend.push_audio(_make_pcm_tone(16000))
        r1 = backend.get_partial()
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
        backend = _make_backend(self.mock_whisper, self.mock_vad)
        backend._streaming.buffer_trimming_sec = 1.0
        backend.push_audio(_make_pcm_tone(32000))
        assert len(backend._streaming.audio_buffer) == 32000
        backend.get_partial()
        backend.get_partial()
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


class TestVAD:
    """Tests for Silero VAD-based endpoint detection."""

    def _make_backend_with_vad(self, speech_prob=0.0):
        """Create backend with configurable VAD mock."""
        mock_whisper = _make_mock_whisper_model()
        mock_vad = _make_mock_vad_model(speech_prob=speech_prob)
        return _make_backend(mock_whisper, mock_vad), mock_vad

    def test_silence_does_not_trigger_speech(self):
        backend, _ = self._make_backend_with_vad(speech_prob=0.1)
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
        assert backend._in_speech is True

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
        assert backend.detect_endpoint() is False

    def test_speech_resets_silence_accumulator(self):
        """Intermittent speech should reset silence accumulation."""
        mock_whisper = _make_mock_whisper_model()
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

        backend = _make_backend(mock_whisper, mock_vad)
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
