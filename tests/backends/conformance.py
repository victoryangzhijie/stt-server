"""Behavioral contract for SttBackend implementations.

Subclass and override the `backend` fixture. Run against mock in CI;
runnable against real backends locally (Plan 3)."""

import asyncio

import pytest

from stt_server.backends.base import StreamConfig, SttBackend
from stt_server.core.events import AudioChunk

ONE_SECOND_PCM = b"\x00\x01" * 16000  # 1 s of quiet non-zero pcm16


class BackendConformanceSuite:
    @pytest.fixture
    def backend(self) -> SttBackend:
        raise NotImplementedError("subclass must provide a backend fixture")

    async def _run_utterance(self, backend: SttBackend, audio: bytes) -> list:
        stream = await backend.create_stream(StreamConfig())
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(AudioChunk(data=audio, ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        await stream.close()
        return events

    async def test_start_stop(self, backend):
        await backend.start()
        await backend.stop()

    async def test_finalize_yields_exactly_one_final_last(self, backend):
        await backend.start()
        try:
            events = await self._run_utterance(backend, ONE_SECOND_PCM)
            finals = [e for e in events if e.kind == "final"]
            assert len(finals) == 1
            assert events[-1].kind == "final"
            assert isinstance(finals[0].text, str) and finals[0].text
        finally:
            await backend.stop()

    async def test_partials_are_ordered_by_audio_time(self, backend):
        await backend.start()
        try:
            events = await self._run_utterance(backend, ONE_SECOND_PCM)
            times = [e.audio_time_ms for e in events]
            assert times == sorted(times)
        finally:
            await backend.stop()

    async def test_close_without_finalize_ends_iterator(self, backend):
        await backend.start()
        try:
            stream = await backend.create_stream(StreamConfig())
            await stream.push_audio(AudioChunk(data=ONE_SECOND_PCM, ingest_ts=0.0))
            await stream.close()

            async def drain():
                return [ev async for ev in stream.events()]

            await asyncio.wait_for(drain(), timeout=5.0)  # must not hang
        finally:
            await backend.stop()

    async def test_push_after_finalize_is_noop(self, backend):
        await backend.start()
        try:
            stream = await backend.create_stream(StreamConfig())
            events = []

            async def reader():
                async for ev in stream.events():
                    events.append(ev)

            task = asyncio.create_task(reader())
            await stream.push_audio(AudioChunk(data=ONE_SECOND_PCM, ingest_ts=0.0))
            await stream.finalize()
            await asyncio.wait_for(task, timeout=5.0)
            events_after_finalize = list(events)

            # push_audio after finalize must not raise and must not produce
            # any further events — the iterator has already ended, so the
            # only observable final was the one already delivered above.
            await stream.push_audio(AudioChunk(data=ONE_SECOND_PCM, ingest_ts=0.0))
            await stream.close()

            assert events == events_after_finalize
            assert [e for e in events if e.kind == "final"] == [events_after_finalize[-1]]
            assert events[-1].kind == "final"
        finally:
            await backend.stop()
