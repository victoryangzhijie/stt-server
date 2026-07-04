"""VAD-frame-driven endpointing state machine. Pure and clock-free:
time advances only through frames, so behavior is fully deterministic."""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Literal

from stt_server.config.settings import EndpointingConfig


class EndpointerState(enum.Enum):
    IDLE = "idle"
    SPEECH = "speech"
    ENDPOINTING = "endpointing"


@dataclass(frozen=True)
class StartUtterance:
    frames: tuple[bytes, ...]  # pre-roll buffer, oldest first, incl. triggering frames


@dataclass(frozen=True)
class SpeechAudio:
    frame: bytes


@dataclass(frozen=True)
class EndUtterance:
    reason: Literal["silence", "max_duration", "input_ended"]


EndpointAction = StartUtterance | SpeechAudio | EndUtterance


class Endpointer:
    def __init__(self, cfg: EndpointingConfig) -> None:
        self.cfg = cfg
        self.state = EndpointerState.IDLE
        preroll_frames = max(1, cfg.pre_roll_ms // cfg.frame_ms)
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._speech_streak = 0
        self._silence_ms = 0
        self._utterance_ms = 0

    def process(self, frame: bytes, is_speech: bool) -> list[EndpointAction]:
        if self.state is EndpointerState.IDLE:
            return self._process_idle(frame, is_speech)
        return self._process_active(frame, is_speech)

    def _process_idle(self, frame: bytes, is_speech: bool) -> list[EndpointAction]:
        self._preroll.append(frame)
        self._speech_streak = self._speech_streak + 1 if is_speech else 0
        if self._speech_streak < self.cfg.speech_start_frames:
            return []
        frames = tuple(self._preroll)
        self.state = EndpointerState.SPEECH
        self._utterance_ms = len(frames) * self.cfg.frame_ms
        self._silence_ms = 0
        return [StartUtterance(frames=frames)]

    def _process_active(self, frame: bytes, is_speech: bool) -> list[EndpointAction]:
        actions: list[EndpointAction] = [SpeechAudio(frame=frame)]
        self._utterance_ms += self.cfg.frame_ms
        if is_speech:
            self.state = EndpointerState.SPEECH
            self._silence_ms = 0
        else:
            self.state = EndpointerState.ENDPOINTING
            self._silence_ms += self.cfg.frame_ms
        if self._silence_ms >= self.cfg.min_silence_ms:
            actions.append(EndUtterance(reason="silence"))
            self._reset()
        elif self._utterance_ms >= self.cfg.max_utterance_ms:
            actions.append(EndUtterance(reason="max_duration"))
            self._reset()
        return actions

    def flush(self) -> list[EndpointAction]:
        if self.state is EndpointerState.IDLE:
            return []
        self._reset()
        return [EndUtterance(reason="input_ended")]

    def _reset(self) -> None:
        self.state = EndpointerState.IDLE
        self._preroll.clear()
        self._speech_streak = 0
        self._silence_ms = 0
        self._utterance_ms = 0
