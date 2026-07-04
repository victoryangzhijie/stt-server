"""Voice activity detection interface and the dependency-free energy detector."""

from __future__ import annotations

import abc
import array
import math

from stt_server.config.settings import VadConfig


class VadDetector(abc.ABC):
    @abc.abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """Classify one fixed-size PCM16 frame."""

    def reset(self) -> None:  # noqa: B027
        """Reset detector state (default no-op)."""
        pass


class EnergyVad(VadDetector):
    def __init__(self, threshold_dbfs: float = -40.0) -> None:
        self.threshold_dbfs = threshold_dbfs

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) % 2:
            frame = frame[:-1]  # drop the trailing byte of an odd-length frame
        samples = array.array("h")
        samples.frombytes(frame)
        if not samples:
            return False
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        dbfs = 20 * math.log10(max(rms, 1e-9) / 32768.0)
        return dbfs > self.threshold_dbfs


def make_vad(cfg: VadConfig) -> VadDetector:
    if cfg.kind == "energy":
        return EnergyVad(threshold_dbfs=cfg.threshold_dbfs)
    if cfg.kind == "silero":
        from stt_server.core.vad_silero import SileroVad

        return SileroVad(model_path=cfg.model_path, threshold=cfg.threshold)
    raise NotImplementedError(f"VAD kind {cfg.kind!r} not available yet")
