import asyncio

import pytest

from stt_server.backends.base import StreamConfig
from stt_server.backends.mock import MockBackend, MockUtteranceScript
from stt_server.core.events import AudioChunk

from .conformance import BackendConformanceSuite


class TestMockConformance(BackendConformanceSuite):
    @pytest.fixture
    def backend(self):
        return MockBackend()


SCRIPT = MockUtteranceScript(partials=("i", "i want", "i want a coffee"), final="I want a coffee.")


async def test_partials_follow_audio_time():
    backend = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT])
    await backend.start()
    stream = await backend.create_stream(StreamConfig())
    events = []
    task = asyncio.create_task(_drain(stream, events))

    # 250 ms of audio -> partials at 100 ms and 200 ms boundaries = 2 partials
    await stream.push_audio(AudioChunk(data=b"\x00" * 8000, ingest_ts=0.0))
    await _until(lambda: len(events) == 2)
    assert [e.text for e in events] == ["i", "i want"]

    await stream.finalize()
    await asyncio.wait_for(task, timeout=5.0)
    assert [e.kind for e in events] == ["partial", "partial", "final"]
    assert events[-1].text == "I want a coffee."
    await stream.close()
    await backend.stop()


async def test_streams_cycle_scripts():
    s2 = MockUtteranceScript(partials=("ok",), final="OK.")
    backend = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT, s2])
    await backend.start()
    finals = []
    for _ in range(3):
        stream = await backend.create_stream(StreamConfig())
        events = []
        task = asyncio.create_task(_drain(stream, events))
        await stream.push_audio(AudioChunk(data=b"\x00" * 8000, ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        await stream.close()
        finals.append(events[-1].text)
    assert finals == ["I want a coffee.", "OK.", "I want a coffee."]
    await backend.stop()


async def test_create_stream_increments_per_instance_counter():
    # The script selector is a plain per-instance counter (not a shared
    # itertools.cycle iterator), so it is directly inspectable/reasoned
    # about — this is what a concurrent load generator (Plan 4) depends on
    # for deterministic per-backend script assignment regardless of stream
    # interleaving.
    s2 = MockUtteranceScript(partials=("ok",), final="OK.")
    backend = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT, s2])
    await backend.start()
    assert backend._counter == 0
    await backend.create_stream(StreamConfig())
    assert backend._counter == 1
    await backend.create_stream(StreamConfig())
    assert backend._counter == 2
    await backend.stop()


async def test_script_selection_is_per_instance_counter_not_shared_cycle():
    # Two independently-constructed backend instances must each start their
    # own script selection at index 0 — a shared itertools.cycle keyed off
    # the class (or module) would leak state across instances (and across
    # concurrently interleaved streams from different backends).
    s2 = MockUtteranceScript(partials=("ok",), final="OK.")
    backend_a = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT, s2])
    backend_b = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT, s2])
    await backend_a.start()
    await backend_b.start()

    async def one_final(backend):
        stream = await backend.create_stream(StreamConfig())
        events = []
        task = asyncio.create_task(_drain(stream, events))
        await stream.push_audio(AudioChunk(data=b"\x00" * 8000, ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        await stream.close()
        return events[-1].text

    assert await one_final(backend_a) == "I want a coffee."
    assert await one_final(backend_b) == "I want a coffee."
    await backend_a.stop()
    await backend_b.stop()


async def _drain(stream, sink):
    async for ev in stream.events():
        sink.append(ev)


async def _until(predicate) -> None:
    async with asyncio.timeout(2.0):
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.001)
