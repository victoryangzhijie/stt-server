"""Spec §13 backpressure: bounded per-session audio queue + shed policy.

These tests use a synthetic single-frame chunk size (30ms tone, aligned to
frame_ms=30) so each `push_audio()` call maps to exactly one VAD frame and
(once mid-utterance) exactly one backend `push_audio()` call — that keeps the
"dropped + processed == pushed" accounting exact and unambiguous.
"""

from __future__ import annotations

import asyncio

from stt_server.backends.base import (
    BackendCapabilities,
    BackendEvent,
    SttBackend,
    SttStream,
)
from stt_server.config.settings import EndpointingConfig, StabilizerConfig
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import EnergyVad
from stt_server.metrics.registry import AUDIO_DROPPED
from tests.helpers.audio import make_tone

# pre_roll_ms=0 + speech_start_frames=1 => StartUtterance fires on the very
# first speech frame with exactly that one frame in its pre-roll burst, so
# every processed chunk (Start or mid-utterance) maps to exactly one
# backend push_audio() call. min_silence_ms is huge so nothing endpoints
# except the explicit end_input() flush.
EP_CFG = EndpointingConfig(
    frame_ms=30, pre_roll_ms=0, min_silence_ms=1_000_000, max_utterance_ms=10**9,
    speech_start_frames=1,
)
STAB_CFG = StabilizerConfig(min_partials=1, min_stable_ms=0.0)
ONE_FRAME = make_tone(30)  # exactly one 30ms VAD frame of "speech"


class _RecordingStream(SttStream):
    """A stream whose push_audio() records every call and optionally sleeps,
    simulating a backend decode that can't keep up with the input rate."""

    def __init__(self, calls: list[AudioChunk], delay: float = 0.0) -> None:
        self._calls = calls
        self._delay = delay
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._audio_ms = 0.0
        self._done = False

    async def push_audio(self, chunk: AudioChunk) -> None:
        self._calls.append(chunk)
        if self._delay:
            await asyncio.sleep(self._delay)
        self._audio_ms += chunk.duration_ms

    async def events(self):
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def finalize(self) -> None:
        if self._done:
            return
        self._done = True
        await self._queue.put(BackendEvent(kind="final", text="ok", audio_time_ms=self._audio_ms))
        await self._queue.put(None)

    async def close(self) -> None:
        if not self._done:
            self._done = True
            await self._queue.put(None)


class _RecordingBackend(SttBackend):
    capabilities = BackendCapabilities(streaming=True, languages=("en",))

    def __init__(self, name: str, delay: float = 0.0) -> None:
        self.name = name
        self.delay = delay
        self.calls: list[AudioChunk] = []

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def create_stream(self, cfg):
        return _RecordingStream(self.calls, delay=self.delay)


def make_bp_session(backend, *, audio_queue_chunks: int, audio_overflow_policy: str) -> Session:
    return Session(
        session_id=f"s-bp-{audio_overflow_policy}",
        backend=backend,
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
        metrics_labels={"backend": backend.name, "api": "native"},
        audio_queue_chunks=audio_queue_chunks,
        audio_overflow_policy=audio_overflow_policy,
    )


async def _collect(session: Session, out: list) -> None:
    async for ev in session.events():
        out.append(ev)


async def test_drop_oldest_sheds_excess_and_survives():
    backend = _RecordingBackend("slow-drop", delay=0.02)
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="drop_oldest")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    before = AUDIO_DROPPED.labels(backend="slow-drop")._value.get()

    # Push 20 single-frame chunks back-to-back with no yield in between, so
    # the bounded queue (size 4) genuinely overflows — nothing drains it
    # concurrently until we hit a real await point below.
    for _ in range(20):
        await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))

    await session.end_input()
    await asyncio.wait_for(task, timeout=5.0)

    dropped = AUDIO_DROPPED.labels(backend="slow-drop")._value.get() - before
    processed = len(backend.calls)

    assert dropped > 0, "queue of size 4 fed 20 chunks fast must shed some"
    assert dropped + processed == 20
    finals = [e for e in events if e.type == EventType.FINAL]
    assert len(finals) == 1
    assert not any(e.type == EventType.ERROR for e in events)


async def test_drop_oldest_rate_limits_the_warning_log(monkeypatch):
    backend = _RecordingBackend("slow-drop-log", delay=0.02)
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="drop_oldest")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    calls = []
    monkeypatch.setattr(
        "stt_server.core.session.logger.warning", lambda *a, **k: calls.append((a, k))
    )

    for _ in range(20):
        await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))

    await session.end_input()
    await asyncio.wait_for(task, timeout=5.0)

    # 16 drops happen back-to-back (no real time elapses between them), so
    # the 1/s rate limit must collapse them to a single log line.
    assert len(calls) == 1


async def test_error_policy_emits_backpressure_error_and_ends_cleanly():
    backend = _RecordingBackend("slow-error", delay=0.02)
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="error")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    for _ in range(20):
        await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))

    # Must not hang: events() must end on its own once the shed policy aborts.
    await asyncio.wait_for(task, timeout=5.0)

    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].error_code == "backpressure"
    assert errors[0].recoverable is False
    assert not any(e.type == EventType.FINAL for e in events)

    # Further calls after the shed-triggered end are no-ops (I-1 style guard).
    await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
    await session.end_input()
    await session.abort()


async def test_lossless_drain_when_consumer_keeps_up():
    """Pins the deterministic-file-mode invariant: a fast backend + a small
    queue must never drop anything, because the feeder drains concurrently
    between each push (simulated here by yielding after every push, matching
    the interleaving a real network receive loop naturally provides)."""
    backend = _RecordingBackend("fast", delay=0.0)
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="drop_oldest")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    before = AUDIO_DROPPED.labels(backend="fast")._value.get()

    for _ in range(20):
        await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
        await asyncio.sleep(0)  # let the feeder drain concurrently

    await session.end_input()
    await asyncio.wait_for(task, timeout=5.0)

    dropped = AUDIO_DROPPED.labels(backend="fast")._value.get() - before
    assert dropped == 0
    assert len(backend.calls) == 20
    finals = [e for e in events if e.type == EventType.FINAL]
    assert len(finals) == 1


class _RaisingVad:
    """A VAD whose is_speech() raises: an unexpected pipeline failure NOT
    wrapped by _push_audio_safe, exercising the feeder's exception guard."""

    def is_speech(self, frame: bytes) -> bool:
        raise RuntimeError("vad boom")


async def test_unexpected_pipeline_exception_emits_internal_error_no_hang():
    """Review Critical: an exception escaping the pipeline outside
    _push_audio_safe (e.g. a raising VAD) must not kill the feeder task
    silently — the session must emit a terminal internal_error ERROR and
    events() must end (no hang)."""
    backend = _RecordingBackend("vad-boom")
    session = Session(
        session_id="s-vad-boom",
        backend=backend,
        vad=_RaisingVad(),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
        audio_queue_chunks=4,
        audio_overflow_policy="drop_oldest",
    )
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)  # must NOT hang

    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].error_code == "internal_error"
    assert errors[0].recoverable is False
    assert "vad boom" in errors[0].message
    # The feeder task must have finished (not leaked, not still pending).
    feeder = session._feeder
    assert feeder is not None
    await asyncio.wait_for(asyncio.shield(feeder), timeout=5.0)
    assert feeder.done()


class _SlowFinalizeStream(_RecordingStream):
    async def finalize(self) -> None:
        await asyncio.sleep(30)  # long enough that only abort() can end it


class _SlowFinalizeBackend(_RecordingBackend):
    async def create_stream(self, cfg):
        return _SlowFinalizeStream(self.calls, delay=self.delay)


async def test_abort_during_finalize_cancels_feeder_for_real():
    """Review Important 2: abort() while the feeder is blocked in the
    EndUtterance finalize path must actually cancel the feeder task — the
    old `except CancelledError: return` swallowed the feeder's own
    cancellation (feeder.cancelled() was False after abort)."""
    backend = _SlowFinalizeBackend("slow-finalize")
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="drop_oldest")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
    await session.end_input()
    # Let the feeder drain the chunk + sentinel and block inside finalize().
    for _ in range(20):
        await asyncio.sleep(0)
    feeder = session._feeder
    assert feeder is not None and not feeder.done()

    await asyncio.wait_for(session.abort(), timeout=5.0)  # must NOT hang
    await asyncio.wait_for(task, timeout=5.0)  # events() must end

    assert feeder.cancelled(), "feeder swallowed its own cancellation"
    assert not any(e.type == EventType.FINAL for e in events)


async def test_push_audio_after_end_input_is_noop():
    """Review Important 3: once end_input() has been called, late pushes are
    no-ops — the chunk never reaches the backend (and can never race the
    _END_INPUT sentinel out of a full queue under drop_oldest)."""
    backend = _RecordingBackend("fast-noop")
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="drop_oldest")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    for _ in range(3):
        await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
        await asyncio.sleep(0)
    await session.end_input()
    # Late pushes: must be swallowed without touching the queue or backend.
    for _ in range(5):
        await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)

    assert len(backend.calls) == 3
    finals = [e for e in events if e.type == EventType.FINAL]
    assert len(finals) == 1


async def test_feeder_task_done_after_error_path_session_end():
    """A mid-utterance backend failure ends the session via _push_audio_safe
    (no end_input()/abort() involved): the feeder task must still exit — no
    task parked forever on an empty queue."""

    class _PushFailStream(_RecordingStream):
        async def push_audio(self, chunk: AudioChunk) -> None:
            raise RuntimeError("decode boom")

    class _PushFailBackend(_RecordingBackend):
        async def create_stream(self, cfg):
            return _PushFailStream(self.calls)

    backend = _PushFailBackend("push-fail")
    session = make_bp_session(backend, audio_queue_chunks=4, audio_overflow_policy="drop_oldest")
    events: list = []
    task = asyncio.create_task(_collect(session, events))

    await session.push_audio(AudioChunk(data=ONE_FRAME, ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)

    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].error_code == "backend_error"

    feeder = session._feeder
    assert feeder is not None
    await asyncio.wait_for(asyncio.shield(feeder), timeout=5.0)
    assert feeder.done() and not feeder.cancelled()
