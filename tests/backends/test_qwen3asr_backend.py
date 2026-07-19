import asyncio
import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

from stt_server.backends.base import BackendUnavailableError, StreamConfig
from stt_server.backends.registry import create_backend
from stt_server.config.settings import BackendDef, load_settings
from stt_server.core.events import AudioChunk

QWEN_ASR_AVAILABLE = importlib.util.find_spec("qwen_asr") is not None

SAMPLE_RATE = 16000


@pytest.mark.skipif(QWEN_ASR_AVAILABLE, reason="only valid when qwen_asr is NOT installed")
def test_missing_qwen_asr_raises_unavailable_with_extras_hint():
    with pytest.raises(BackendUnavailableError) as exc_info:
        create_backend(BackendDef(type="qwen3asr", options={}))
    message = str(exc_info.value)
    assert "qwen_asr" in message
    assert "pip install" in message
    assert "stt-server[qwen3asr]" in message


def test_zero_or_negative_redecode_interval_raises_value_error_in_backend():
    from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend

    # Validated before the qwen_asr-availability check (mirrors
    # FunasrBackend's chunk_size validation ordering), so this fails the
    # same way whether or not the extra is installed.
    with pytest.raises(ValueError, match="redecode_interval_ms"):
        Qwen3AsrBackend(redecode_interval_ms=0)


def _bare_backend(redecode_interval_ms: float = 480):
    """A Qwen3AsrBackend without going through __init__ (which requires the
    qwen_asr package to be importable); attributes set to post-__init__
    defaults."""
    from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend

    backend = Qwen3AsrBackend.__new__(Qwen3AsrBackend)
    backend._model_name = "Qwen/Qwen3-ASR-0.6B"
    backend._gpu_memory_utilization = 0.8
    backend._max_concurrent = 1
    backend._language = None
    backend._redecode_interval_ms = redecode_interval_ms
    backend._model_instance = None
    backend._executor = None
    backend._decode_lock = threading.Lock()
    return backend


async def test_create_stream_before_start_raises_runtime_error():
    backend = _bare_backend()
    with pytest.raises(RuntimeError, match="not started"):
        await backend.create_stream(StreamConfig())


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


@dataclass
class _FakeStreamingState:
    """Mirrors the real `qwen_asr.inference.qwen3_asr.ASRStreamingState`
    fields this adapter reads/writes: `chunk_id` (bump-per-decode marker),
    `buffer` (framework-internal pending-samples accumulator), and
    `text` (cumulative REPLACE-semantics hypothesis)."""

    chunk_size_samples: int
    chunk_id: int = 0
    buffer: list = field(default_factory=list)
    text: str = ""
    language: str = ""


class _FakeQwen3AsrEngine:
    """Stub for `qwen_asr.Qwen3ASRModel` (vLLM backend): pins the real
    `init_streaming_state`/`streaming_transcribe`/`finish_streaming_transcribe`
    call signatures (no **kwargs catch-all) and reimplements the real
    source's documented internal chunk-buffering behavior (buffer samples;
    once a full `chunk_size_samples` chunk has accumulated, "decode" —
    here, just pop a scripted text and bump `chunk_id`) closely enough to
    exercise this adapter's redecode-cadence detection."""

    def __init__(self, texts: list[str | None] | None = None, delay: float = 0.0) -> None:
        self._texts = list(texts or [])
        self._delay = delay
        self.streaming_calls: list[dict] = []
        self.finish_calls: list[dict] = []

    def init_streaming_state(
        self,
        context: str = "",
        language: str | None = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> _FakeStreamingState:
        if chunk_size_sec <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")
        chunk_size_samples = max(1, int(round(chunk_size_sec * SAMPLE_RATE)))
        return _FakeStreamingState(chunk_size_samples=chunk_size_samples)

    def streaming_transcribe(self, pcm16k: Any, state: _FakeStreamingState) -> _FakeStreamingState:
        if self._delay:
            time.sleep(self._delay)
        state.buffer.extend(list(pcm16k))
        while len(state.buffer) >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            self.streaming_calls.append({"is_final": False, "chunk_len": len(chunk)})
            text = self._texts.pop(0) if self._texts else None
            if text is not None:
                state.text = text
            state.chunk_id += 1
        return state

    def finish_streaming_transcribe(self, state: _FakeStreamingState) -> _FakeStreamingState:
        if not state.buffer:
            return state  # documented no-op: nothing left to flush
        if self._delay:
            time.sleep(self._delay)
        self.finish_calls.append({"is_final": True, "tail_len": len(state.buffer)})
        text = self._texts.pop(0) if self._texts else None
        if text is not None:
            state.text = text
        state.buffer = []
        return state


def _make_stream(engine, chunk_size_sec=0.1):
    from stt_server.backends.qwen3asr.backend import Qwen3AsrStream

    executor = ThreadPoolExecutor(max_workers=4)
    state = engine.init_streaming_state(chunk_size_sec=chunk_size_sec)
    stream = Qwen3AsrStream(engine, executor, state, threading.Lock())
    return stream, executor, state


# chunk_size_sec=0.1 -> chunk_size_samples = 1600 samples at 16 kHz
# -> a push of 1600 samples' worth of pcm16 bytes (3200 bytes) crosses exactly
# one redecode interval.
SAMPLES_PER_INTERVAL = 1600
BYTES_PER_INTERVAL = SAMPLES_PER_INTERVAL * 2


async def test_push_below_interval_does_not_decode():
    engine = _FakeQwen3AsrEngine()
    stream, executor, state = _make_stream(engine)
    try:
        sub_interval = b"\x00\x01" * (BYTES_PER_INTERVAL // 2 // 2)
        await stream.push_audio(AudioChunk(data=sub_interval, ingest_ts=0.0))
        assert engine.streaming_calls == []
        assert state.chunk_id == 0
    finally:
        executor.shutdown(wait=True)


async def test_push_crossing_interval_decodes_with_full_accumulated_audio():
    engine = _FakeQwen3AsrEngine(texts=["hello"])
    stream, executor, state = _make_stream(engine)
    try:
        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
        )
        assert len(engine.streaming_calls) == 1
        assert engine.streaming_calls[0]["chunk_len"] == SAMPLES_PER_INTERVAL
        assert state.chunk_id == 1
    finally:
        executor.shutdown(wait=True)


async def test_multiple_intervals_emit_multiple_cumulative_partials():
    engine = _FakeQwen3AsrEngine(texts=["hello", "hello world"])
    stream, executor, state = _make_stream(engine)
    try:
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
        )
        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
        )
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        partials = [e for e in events if e.kind == "partial"]
        finals = [e for e in events if e.kind == "final"]
        assert [e.text for e in partials] == ["hello", "hello world"]
        assert len(finals) == 1
        assert finals[0].text == "hello world"  # no remainder buffer -> finish is a no-op
        assert events[-1].kind == "final"
        assert engine.finish_calls == []
    finally:
        executor.shutdown(wait=True)


async def test_finalize_flushes_remainder_and_emits_exactly_one_final():
    engine = _FakeQwen3AsrEngine(texts=["partial", "tail"])
    stream, executor, state = _make_stream(engine)
    try:
        # one full interval (partial) + a sub-interval remainder
        remainder_samples = 200
        await stream.push_audio(
            AudioChunk(
                data=b"\x00\x01" * (SAMPLES_PER_INTERVAL + remainder_samples), ingest_ts=0.0
            )
        )
        assert len(engine.streaming_calls) == 1
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        assert len(engine.finish_calls) == 1
        assert engine.finish_calls[0]["tail_len"] == remainder_samples
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        assert finals[0].text == "tail"
        assert events[-1].kind == "final"
    finally:
        executor.shutdown(wait=True)


async def test_redecode_tick_with_unchanged_text_emits_no_duplicate_partial():
    # A redecode tick over silence legitimately leaves state.text unchanged
    # (fake scripts None -> text stays as-is). Matches SherpaStream's
    # changed-text dedup: no duplicate partial event may be emitted, so the
    # stabilizer's partial count isn't inflated by silence ticks.
    engine = _FakeQwen3AsrEngine(texts=["hello", None, "hello world"])
    stream, executor, state = _make_stream(engine)
    try:
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        for _ in range(3):  # three full intervals -> three redecode ticks
            await stream.push_audio(
                AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
            )
        assert state.chunk_id == 3  # all three ticks really decoded
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        partials = [e for e in events if e.kind == "partial"]
        assert [e.text for e in partials] == ["hello", "hello world"]  # no duplicate
        assert [e.kind for e in events][-1] == "final"
    finally:
        executor.shutdown(wait=True)


async def test_create_stream_passes_interval_and_language_to_init_streaming_state():
    # create_stream()'s own logic: redecode_interval_ms -> chunk_size_sec
    # conversion (480 -> 0.48) and constructor-language pass-through.
    backend = _bare_backend(redecode_interval_ms=480)
    backend._language = "English"
    backend._executor = ThreadPoolExecutor(max_workers=1)

    init_calls: list[dict] = []

    class _RecordingEngine(_FakeQwen3AsrEngine):
        def init_streaming_state(
            self,
            context="",
            language=None,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        ):
            init_calls.append({"language": language, "chunk_size_sec": chunk_size_sec})
            return super().init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )

    backend._model_instance = _RecordingEngine()
    try:
        stream = await backend.create_stream(StreamConfig())
        assert init_calls == [{"language": "English", "chunk_size_sec": 0.48}]
        assert stream._state.chunk_size_samples == int(round(0.48 * SAMPLE_RATE))
    finally:
        backend._executor.shutdown(wait=True)


async def test_create_stream_per_request_language_overrides_constructor_language():
    # StreamConfig.language, when set, takes precedence over the backend's
    # constructor-time language (cfg.language or self._language).
    backend = _bare_backend(redecode_interval_ms=480)
    backend._language = "English"
    backend._executor = ThreadPoolExecutor(max_workers=1)

    init_calls: list[dict] = []

    class _RecordingEngine(_FakeQwen3AsrEngine):
        def init_streaming_state(
            self,
            context="",
            language=None,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=2.0,
        ):
            init_calls.append({"language": language})
            return super().init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )

    backend._model_instance = _RecordingEngine()
    try:
        await backend.create_stream(StreamConfig(language="Chinese"))
        assert init_calls == [{"language": "Chinese"}]
    finally:
        backend._executor.shutdown(wait=True)


async def test_push_and_finalize_after_done_are_noops():
    engine = _FakeQwen3AsrEngine(texts=["hi"])
    stream, executor, state = _make_stream(engine)
    try:
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
        )
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        events_after_finalize = list(events)
        calls_after_finalize = len(engine.streaming_calls) + len(engine.finish_calls)

        await stream.push_audio(
            AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
        )
        await stream.finalize()
        await stream.close()

        assert events == events_after_finalize
        assert len(engine.streaming_calls) + len(engine.finish_calls) == calls_after_finalize
    finally:
        executor.shutdown(wait=True)


async def test_close_waits_for_inflight_decode_and_sentinel_is_last():
    engine = _FakeQwen3AsrEngine(texts=["hello"], delay=0.2)
    stream, executor, state = _make_stream(engine)
    try:
        push = asyncio.create_task(
            stream.push_audio(
                AudioChunk(data=b"\x00\x01" * (BYTES_PER_INTERVAL // 2), ingest_ts=0.0)
            )
        )
        await asyncio.sleep(0.05)  # streaming_transcribe now in flight
        await stream.close()
        await push

        async def drain():
            return [ev async for ev in stream.events()]

        events = await asyncio.wait_for(drain(), timeout=5.0)
        assert [(e.kind, e.text) for e in events] == [("partial", "hello")]
        assert stream._queue.empty(), "an event was enqueued after the sentinel"
    finally:
        executor.shutdown(wait=True)


async def test_decode_lock_serializes_concurrent_generate_calls():
    # vLLM's synchronous LLM.generate() is unsafe under concurrent thread
    # invocation (its V1 SyncMPClient correlates requests/outputs in shared
    # state), so the backend holds a single threading.Lock across every
    # generate() call. Verify that across several concurrent streams sharing
    # one backend lock, at most ONE generate() (streaming_transcribe here) is
    # ever in flight at a time -- the lock, not the executor's max_workers, is
    # the real concurrency bound (and prevents the cross-request prompt-scaffold
    # contamination seen under concurrent file uploads).
    executor = ThreadPoolExecutor(max_workers=4)
    engine = _FakeQwen3AsrEngine()
    in_flight = 0
    max_seen = 0
    lock = threading.Lock()
    release = threading.Event()

    def blocking_streaming_transcribe(pcm16k, state):
        nonlocal in_flight, max_seen
        with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        release.wait(timeout=5.0)
        with lock:
            in_flight -= 1
        return state

    engine.streaming_transcribe = blocking_streaming_transcribe  # type: ignore[method-assign]

    from stt_server.backends.qwen3asr.backend import Qwen3AsrStream

    decode_lock = threading.Lock()  # the backend-wide lock, shared by all streams
    streams = [
        Qwen3AsrStream(
            engine, executor, engine.init_streaming_state(chunk_size_sec=0.1), decode_lock
        )
        for _ in range(4)
    ]
    try:
        tasks = [
            asyncio.create_task(
                s.push_audio(AudioChunk(data=b"\x00\x01" * BYTES_PER_INTERVAL, ingest_ts=0.0))
            )
            for s in streams
        ]
        await asyncio.sleep(0.3)  # let all 4 attempt their decode
        assert max_seen == 1, "decode lock must serialize generate() calls"
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_qwen3asr_config_roundtrips_through_settings():
    settings = load_settings("configs/qwen3asr.yaml")
    backend_def = settings.backends["qwen3asr"]
    assert backend_def.type == "qwen3asr"
    assert backend_def.options == {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "gpu_memory_utilization": 0.8,
        "max_concurrent": 8,
        "language": None,
        "redecode_interval_ms": 480,
    }
    # The options dict's keys must be exactly the Qwen3AsrBackend kwargs, so
    # a bare `Qwen3AsrBackend(**backend_def.options)` (skipping only the
    # availability check) succeeds against the real constructor signature.
    import inspect

    from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend

    sig = inspect.signature(Qwen3AsrBackend.__init__)
    accepted = set(sig.parameters) - {"self"}
    assert set(backend_def.options) <= accepted


def test_capabilities_flags():
    from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend

    caps = Qwen3AsrBackend.capabilities
    assert caps.streaming is True
    assert caps.native_endpointing is False
    assert caps.batch_decode is True
    assert "en" in caps.languages
    assert "zh" in caps.languages
