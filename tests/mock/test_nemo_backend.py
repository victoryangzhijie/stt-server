from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from backends.base import ASRResult
from config import settings


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pcm_silence(n_samples: int) -> bytes:
    return b"\x00\x00" * n_samples


def _make_pcm_tone(n_samples: int, amplitude: int = 10000) -> bytes:
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _make_mock_nemo_model():
    mock_model = MagicMock()
    mock_model.cfg.preprocessor.sample_rate = 16000
    mock_model.cfg.preprocessor.window_stride = 0.01
    mock_model.cfg.preprocessor.features = 80
    mock_model.encoder.pre_encode_cache_size = 3
    mock_model.encoder.setup_streaming_params = MagicMock()
    mock_model.encoder.get_initial_cache_state.return_value = (
        MagicMock(), MagicMock(), MagicMock(),
    )

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
    mock_model.change_decoding_strategy = MagicMock()
    return mock_model


def _make_mock_vad_model(speech_prob: float = 0.0):
    mock_vad = MagicMock()
    mock_vad.return_value = MagicMock(item=MagicMock(return_value=speech_prob))
    mock_vad.reset_states = MagicMock()
    return mock_vad


def _make_mock_streaming_buffer():
    mock_buf = MagicMock()
    mock_signal = MagicMock()
    mock_length = MagicMock()
    mock_buf.__iter__ = MagicMock(
        side_effect=lambda: iter([(mock_signal, mock_length)])
    )
    mock_buf.append_audio_chunk = MagicMock()
    return mock_buf


def _make_backend(mock_nemo=None, mock_vad=None, language="auto"):
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
        backend.close()

    def test_close_idempotent(self):
        backend = _make_backend()
        backend.close()
        backend.close()


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
        backend.push_audio(_make_pcm_tone(chunk_samples * 2 + 100))
        assert self.mock_nemo.conformer_stream_step.call_count == 2

    def test_cache_state_updated_after_step(self):
        backend = _make_backend(self.mock_nemo, self.mock_vad)
        old_channel = backend._streaming.cache_last_channel
        chunk_samples = backend._streaming.chunk_size_samples
        backend.push_audio(_make_pcm_tone(chunk_samples + 100))
        assert backend._streaming.cache_last_channel is not old_channel
