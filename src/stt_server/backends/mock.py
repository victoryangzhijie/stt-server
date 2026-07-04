"""Deterministic scripted backend: the reference implementation for all tests.

Partial timing is driven by accumulated *audio time*, never the wall clock,
so tests and benchmarks are fully deterministic."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from stt_server.backends.base import (
    BackendCapabilities,
    BackendEvent,
    StreamConfig,
    SttBackend,
    SttStream,
)
from stt_server.backends.registry import register_backend
from stt_server.core.events import AudioChunk


@dataclass(frozen=True)
class MockUtteranceScript:
    partials: tuple[str, ...]
    final: str


DEFAULT_SCRIPTS = [
    MockUtteranceScript(
        partials=("the", "the quick", "the quick brown", "the quick brown fox"),
        final="The quick brown fox.",
    ),
    MockUtteranceScript(
        partials=("hello", "hello world"),
        final="Hello world.",
    ),
]

_SENTINEL = None


class MockStream(SttStream):
    def __init__(self, script: MockUtteranceScript, partial_interval_ms: float) -> None:
        self._script = script
        self._interval = partial_interval_ms
        self._audio_ms = 0.0
        self._emitted = 0
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._done = False

    async def push_audio(self, chunk: AudioChunk) -> None:
        if self._done:
            return
        self._audio_ms += chunk.duration_ms
        while (
            self._emitted < len(self._script.partials)
            and self._audio_ms >= (self._emitted + 1) * self._interval
        ):
            await self._queue.put(
                BackendEvent(
                    kind="partial",
                    text=self._script.partials[self._emitted],
                    audio_time_ms=(self._emitted + 1) * self._interval,
                )
            )
            self._emitted += 1

    async def events(self) -> AsyncIterator[BackendEvent]:
        while True:
            ev = await self._queue.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def finalize(self) -> None:
        if self._done:
            return
        self._done = True
        await self._queue.put(
            BackendEvent(kind="final", text=self._script.final, audio_time_ms=self._audio_ms)
        )
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        if not self._done:
            self._done = True
            await self._queue.put(_SENTINEL)


@register_backend("mock")
class MockBackend(SttBackend):
    name = "mock"
    capabilities = BackendCapabilities(streaming=True, languages=("en",))

    def __init__(
        self,
        partial_interval_ms: float = 240.0,
        scripts: list[MockUtteranceScript] | None = None,
    ) -> None:
        self._interval = partial_interval_ms
        self._scripts = scripts or DEFAULT_SCRIPTS
        self._counter = 0

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def create_stream(self, cfg: StreamConfig) -> MockStream:
        script = self._scripts[self._counter % len(self._scripts)]
        self._counter += 1
        return MockStream(script, self._interval)
