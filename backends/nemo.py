from __future__ import annotations

import copy
import threading
from pathlib import Path

import numpy as np
import structlog

from backends.base import ASRResult
from config import settings

logger = structlog.get_logger()

# Lazy import placeholder — set by _ensure_nemo_imports()
CacheAwareStreamingAudioBuffer = None


def _ensure_nemo_imports() -> None:
    global CacheAwareStreamingAudioBuffer
    if CacheAwareStreamingAudioBuffer is None:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer as _Buf,
        )
        CacheAwareStreamingAudioBuffer = _Buf


class _StreamingState:
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
                if model_name.endswith(".nemo") or Path(model_name).is_file():
                    model = nemo_asr.models.ASRModel.restore_from(model_name)
                else:
                    model = nemo_asr.models.ASRModel.from_pretrained(model_name)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)
                model.freeze()
                model.eval()

                window_stride = model.cfg.preprocessor.window_stride
                encoder_stride = getattr(model.encoder, "subsampling_factor", 4)
                chunk_frames = int(settings.nemo_chunk_size / (window_stride * encoder_stride))
                shift_frames = int(settings.nemo_shift_size / (window_stride * encoder_stride))

                model.encoder.setup_streaming_params(
                    chunk_size=chunk_frames,
                    shift_size=shift_frames,
                    left_chunks=settings.nemo_left_chunks,
                )

                decoding_cfg = model.cfg.decoding
                with open_dict(decoding_cfg):
                    decoding_cfg.strategy = "greedy_batch"
                    decoding_cfg.preserve_alignments = True
                    decoding_cfg.fused_batch_size = -1
                model.change_decoding_strategy(decoding_cfg)

                logger.info("nemo.model.loaded", model=model_name, chunk_frames=chunk_frames)
                _nemo_model = model
    return _nemo_model


def _get_vad_base_model() -> object:
    global _vad_base_model
    if _vad_base_model is None:
        with _model_lock:
            if _vad_base_model is None:
                import torch
                _vad_base_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad", model="silero_vad",
                )
                logger.info("nemo.vad_base.loaded")
    return _vad_base_model


def preload_model() -> None:
    logger.info("nemo.preload.start")
    _get_nemo_model()
    _get_vad_base_model()
    logger.info("nemo.preload.done")


def _zeros_tensor(
    dim0: int, dim1: int, dim2: int
) -> object:
    """Create a zero tensor; uses torch when available."""
    try:
        import torch
        return torch.zeros(dim0, dim1, dim2)
    except ImportError:  # pragma: no cover — only hit in mock tests without torch
        return np.zeros((dim0, dim1, dim2), dtype=np.float32)


def _create_streaming_state(model: object) -> _StreamingState:
    _ensure_nemo_imports()

    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=1)
    )
    cache_pre_encode = _zeros_tensor(
        1, model.encoder.pre_encode_cache_size, model.cfg.preprocessor.features,
    )

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=True, pad_and_drop_preencoded=True,
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


class NemoBackend:
    def configure(
        self, sample_rate: int, language: str = "auto", hotwords: list[str] | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._language = language

        model = _get_nemo_model()
        self._streaming = _create_streaming_state(model)

        vad_base = _get_vad_base_model()
        self._vad_model = copy.deepcopy(vad_base)
        self._vad_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._in_speech: bool = False
        self._silence_ms_accum: float = 0.0
        self._endpoint_fired: bool = False

    def push_audio(self, pcm_data: bytes) -> None:
        pass  # Task 3

    def get_partial(self) -> ASRResult | None:
        return None  # Task 3

    def finalize(self) -> ASRResult:
        return ASRResult(text="", is_partial=False, is_endpoint=True)  # Task 4

    def detect_endpoint(self) -> bool:
        return False  # Task 5

    def reset_segment(self) -> None:
        model = _get_nemo_model()
        self._streaming = _create_streaming_state(model)
        self._vad_model = copy.deepcopy(_get_vad_base_model())
        self._vad_buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._silence_ms_accum = 0.0
        self._endpoint_fired = False

    def close(self) -> None:
        pass
