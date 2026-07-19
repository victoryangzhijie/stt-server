"""Internal data model: audio chunks and the unified transcript event stream."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # pcm16


@dataclass(frozen=True)
class AudioChunk:
    """PCM16 mono 16 kHz little-endian audio with its ingest timestamp."""

    data: bytes
    ingest_ts: float  # time.monotonic() at ingest

    @property
    def duration_ms(self) -> float:
        return len(self.data) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000.0


class EventType(enum.Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    PARTIAL = "partial"
    STABILIZED = "stabilized"
    FINAL = "final"
    ERROR = "error"


@dataclass(frozen=True)
class TranscriptEvent:
    """Unified event emitted by the session core; all protocol encoders consume only this."""

    type: EventType
    session_id: str
    utterance_id: int
    seq: int
    audio_time_ms: float  # position in the audio stream when the event was produced
    emitted_ts: float  # time.monotonic() when emitted
    stable_text: str = ""  # PARTIAL: committed prefix
    volatile_text: str = ""  # PARTIAL: uncommitted tail
    text: str = ""  # FINAL: final text; STABILIZED: newly committed delta
    # FINAL: detected language (canonical name), when the backend reports one
    language: str | None = None
    error_code: str | None = None
    recoverable: bool = True
    message: str = ""  # human-readable detail, primarily for ERROR events
    latency: dict[str, float] = field(default_factory=dict)
