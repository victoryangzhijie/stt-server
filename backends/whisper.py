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
