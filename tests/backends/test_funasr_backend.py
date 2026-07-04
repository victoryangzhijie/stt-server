import asyncio
import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from stt_server.backends.base import BackendUnavailableError, StreamConfig
from stt_server.backends.registry import create_backend
from stt_server.config.settings import BackendDef
from stt_server.core.events import AudioChunk

FUNASR_AVAILABLE = importlib.util.find_spec("funasr") is not None


@pytest.mark.skipif(FUNASR_AVAILABLE, reason="only valid when funasr is NOT installed")
def test_missing_funasr_raises_unavailable_with_extras_hint():
    with pytest.raises(BackendUnavailableError) as exc_info:
        create_backend(BackendDef(type="funasr", options={}))
    message = str(exc_info.value)
    assert "funasr" in message
    assert "pip install" in message
    assert "stt-server[funasr]" in message


def _bare_backend():
    """A FunasrBackend without going through __init__ (which requires the
    funasr package to be importable); attributes set to post-__init__
    defaults."""
    from stt_server.backends.funasr.backend import FunasrBackend

    backend = FunasrBackend.__new__(FunasrBackend)
    backend._model_name = "paraformer-zh-streaming"
    backend._pool_workers = 1
    backend._chunk_size = (0, 10, 5)
    backend._model_instance = None
    backend._executor = None
    return backend


def test_zero_stride_chunk_size_raises_value_error_in_stream():
    # A chunk_size like (0, 0, 5) (config typo) would make push_audio's
    # whole-chunk while-loop spin forever; it must fail at construction.
    with pytest.raises(ValueError, match="chunk_size"):
        _make_stream(_ScriptedFakeModel(), chunk_size=(0, 0, 5))


def test_zero_stride_chunk_size_raises_value_error_in_backend():
    from stt_server.backends.funasr.backend import FunasrBackend

    # Validated before the funasr-availability check, so this fails the same
    # way whether or not the extra is installed.
    with pytest.raises(ValueError, match="chunk_size"):
        FunasrBackend(chunk_size=(0, 0, 5))


async def test_create_stream_before_start_raises_runtime_error():
    backend = _bare_backend()
    with pytest.raises(RuntimeError, match="not started"):
        await backend.create_stream(StreamConfig())


async def test_create_stream_language_mismatch_is_ignored_without_error():
    # FunASR's loaded model is pinned to whatever language it was trained
    # for; a client-supplied language mismatch must be a silent (debug
    # logged) no-op, never an error — OpenAI treats `language` as a hint.
    backend = _bare_backend()
    backend._executor = ThreadPoolExecutor(max_workers=1)
    backend._model_instance = object()
    try:
        stream = await backend.create_stream(StreamConfig(language="fr"))
        assert stream is not None
    finally:
        backend._executor.shutdown(wait=True)


async def test_stop_with_inflight_decode_does_not_block_event_loop():
    backend = _bare_backend()
    backend._executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    backend._executor.submit(release.wait)  # slow "generate" in flight
    threading.Timer(0.5, release.set).start()  # unblocks even if loop stalls

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.wait_for(backend.stop(), timeout=5.0)
        assert ticks >= 5, "event loop was starved while stop() waited for the executor"
    finally:
        hb.cancel()
        release.set()


class _ScriptedFakeModel:
    """Stub for `funasr.AutoModel`: records every `generate()` call and pops
    scripted text increments off a queue. Silence/no-new-text calls can be
    scripted as `""` (returned as `[{"text": ""}]`) or `None` (returned as
    `[]`) to exercise both of FunASR's documented empty-result shapes."""

    def __init__(self, texts: list[str | None] | None = None, delay: float = 0.0) -> None:
        self._texts = list(texts or [])
        self._delay = delay
        self.calls: list[dict] = []

    def generate(
        self,
        input,
        cache,
        is_final,
        chunk_size,
        encoder_chunk_look_back,
        decoder_chunk_look_back,
    ):
        if self._delay:
            time.sleep(self._delay)
        self.calls.append(
            {
                "input": input,
                "cache": cache,
                "is_final": is_final,
                "chunk_size": chunk_size,
                "encoder_chunk_look_back": encoder_chunk_look_back,
                "decoder_chunk_look_back": decoder_chunk_look_back,
            }
        )
        text = self._texts.pop(0) if self._texts else None
        if text is None:
            return []
        return [{"text": text}]


def _make_stream(model, chunk_size=(0, 1, 0)):
    from stt_server.backends.funasr.backend import FunasrStream

    executor = ThreadPoolExecutor(max_workers=1)
    stream = FunasrStream(model, executor, chunk_size)
    return stream, executor


# chunk_size=(0, 1, 0) -> chunk_stride_samples = 1 * 960 = 960 samples
# -> chunk_stride_bytes = 960 * 2 = 1920 bytes
STRIDE_BYTES = 1920


async def test_sub_chunk_push_buffers_without_calling_generate():
    model = _ScriptedFakeModel()
    stream, executor = _make_stream(model)
    try:
        sub_chunk = b"\x00\x01" * (STRIDE_BYTES // 2 // 2)
        await stream.push_audio(AudioChunk(data=sub_chunk, ingest_ts=0.0))
        assert model.calls == []
        assert len(stream._buffer) == STRIDE_BYTES // 2
    finally:
        executor.shutdown(wait=True)


async def test_exact_chunk_triggers_one_generate_call():
    model = _ScriptedFakeModel(texts=["hello"])
    stream, executor = _make_stream(model)
    try:
        await stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        assert len(model.calls) == 1
        assert model.calls[0]["is_final"] is False
        assert model.calls[0]["chunk_size"] == [0, 1, 0]
        assert len(stream._buffer) == 0
    finally:
        executor.shutdown(wait=True)


async def test_multi_chunk_push_splits_into_multiple_generate_calls_with_remainder():
    model = _ScriptedFakeModel(texts=["a", "b"])
    stream, executor = _make_stream(model)
    try:
        remainder = 500
        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * ((STRIDE_BYTES * 2 + remainder) // 2), ingest_ts=0.0)
        )
        assert len(model.calls) == 2
        assert all(c["is_final"] is False for c in model.calls)
        assert len(stream._buffer) == remainder
    finally:
        executor.shutdown(wait=True)


async def test_finalize_flushes_remainder_with_is_final_true():
    model = _ScriptedFakeModel(texts=["partial", "tail"])
    stream, executor = _make_stream(model)
    try:
        # one full chunk (partial) + a sub-chunk remainder
        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * ((STRIDE_BYTES + 200) // 2), ingest_ts=0.0)
        )
        assert len(model.calls) == 1
        await stream.finalize()
        assert len(model.calls) == 2
        assert model.calls[-1]["is_final"] is True
        assert len(model.calls[-1]["input"]) == 100  # 200 bytes / 2 bytes-per-sample
    finally:
        executor.shutdown(wait=True)


async def test_cumulative_partial_emission_and_exactly_one_final():
    # Two full chunks pushed (2 generate() calls, each contributing a partial
    # text increment) followed by finalize() with an exactly-empty buffer
    # (no third chunk's worth of remainder, so finalize's own generate call
    # contributes no further text) — isolates that partials accumulate
    # cumulatively while finalize() still emits exactly one final.
    model = _ScriptedFakeModel(texts=["hello", " world", None])
    stream, executor = _make_stream(model)
    try:
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        await stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        partials = [e for e in events if e.kind == "partial"]
        finals = [e for e in events if e.kind == "final"]
        assert [e.text for e in partials] == ["hello", "hello world"]
        assert len(finals) == 1
        assert finals[0].text == "hello world"
        assert events[-1].kind == "final"
        assert len(model.calls) == 3
        assert model.calls[-1]["is_final"] is True
    finally:
        executor.shutdown(wait=True)


async def test_empty_result_shapes_do_not_crash_or_emit_partial():
    # texts=[None, ""] exercises both FunASR "no new text" shapes: `[]` and
    # `[{"text": ""}]`.
    model = _ScriptedFakeModel(texts=[None, ""])
    stream, executor = _make_stream(model)
    try:
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        assert [e for e in events if e.kind == "partial"] == []
        assert len(events) == 1
        assert events[0].kind == "final"
        assert events[0].text == ""
    finally:
        executor.shutdown(wait=True)


async def test_push_and_finalize_after_done_are_noops():
    model = _ScriptedFakeModel(texts=["hi"])
    stream, executor = _make_stream(model)
    try:
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        events_after_finalize = list(events)
        calls_after_finalize = len(model.calls)

        await stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        await stream.finalize()
        await stream.close()

        assert events == events_after_finalize
        assert len(model.calls) == calls_after_finalize
    finally:
        executor.shutdown(wait=True)


async def test_close_waits_for_inflight_generate_and_sentinel_is_last():
    model = _ScriptedFakeModel(texts=["hello"], delay=0.2)
    stream, executor = _make_stream(model)
    try:
        push = asyncio.create_task(
            stream.push_audio(AudioChunk(data=b"\x00\x01" * (STRIDE_BYTES // 2), ingest_ts=0.0))
        )
        await asyncio.sleep(0.05)  # generate() now in flight
        await stream.close()
        await push

        async def drain():
            return [ev async for ev in stream.events()]

        events = await asyncio.wait_for(drain(), timeout=5.0)
        assert [(e.kind, e.text) for e in events] == [("partial", "hello")]
        assert stream._queue.empty(), "an event was enqueued after the sentinel"
    finally:
        executor.shutdown(wait=True)
