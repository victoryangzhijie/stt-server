"""The backend plugin contract every STT backend implements."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from stt_server.core.events import AudioChunk


@dataclass(frozen=True)
class BackendCapabilities:
    streaming: bool
    languages: tuple[str, ...]
    native_endpointing: bool = False
    batch_decode: bool = False


@dataclass(frozen=True)
class BackendEvent:
    kind: Literal["partial", "final"]
    text: str
    audio_time_ms: float


@dataclass(frozen=True)
class StreamConfig:
    language: str | None = None


class BackendUnavailableError(Exception):
    code = "backend_unavailable"


class SttStream(abc.ABC):
    """One stream per utterance: created on speech-start, finalized on endpoint,
    closed after the FINAL event is emitted."""

    @abc.abstractmethod
    async def push_audio(self, chunk: AudioChunk) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[BackendEvent]:
        """Yields partials then exactly one final; the iterator ends after finalize()."""

    @abc.abstractmethod
    async def finalize(self) -> None:
        """Endpoint reached: flush and emit the final event, then end the iterator."""

    @abc.abstractmethod
    async def close(self) -> None: ...


class SttBackend(abc.ABC):
    name: str
    capabilities: BackendCapabilities

    @abc.abstractmethod
    async def start(self) -> None:
        """Load models / warm up. Called once at server startup."""

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def create_stream(self, cfg: StreamConfig) -> SttStream: ...
