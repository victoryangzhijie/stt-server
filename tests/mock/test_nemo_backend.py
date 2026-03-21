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
