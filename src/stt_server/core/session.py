"""Protocol-agnostic session: AudioChunks in, TranscriptEvents out.

Pipeline: FrameSlicer -> VAD -> Endpointer -> backend SttStream -> Stabilizer.
One backend stream per utterance (created on StartUtterance, closed after FINAL)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import structlog

from stt_server.backends.base import StreamConfig, SttBackend, SttStream
from stt_server.core.audio import FrameSlicer
from stt_server.core.endpointing import (
    EndpointAction,
    Endpointer,
    EndUtterance,
    SpeechAudio,
    StartUtterance,
)
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import VadDetector
from stt_server.metrics.registry import (
    AUDIO_DROPPED,
    AUDIO_SECONDS,
    ERRORS,
    FINAL_MS,
    FIRST_PARTIAL_MS,
    UTTERANCES,
)

logger = structlog.get_logger(__name__)

# Sentinel enqueued onto the audio queue by end_input() so the feeder task
# processes it strictly after every audio chunk queued ahead of it (FIFO),
# guaranteeing lossless, deterministic draining before the endpointer flush
# / finalize path runs.
_END_INPUT = object()

# Minimum interval between "audio.dropped" warnings for a single session, so
# a sustained overflow logs at a bounded rate instead of once per chunk.
_DROP_LOG_INTERVAL_S = 1.0


@dataclass
class SessionStats:
    """Per-session counters, accumulated unconditionally (independent of
    whether Prometheus observation is enabled via `metrics_labels`) so API
    adapters always have data for the end-of-session `session.summary` log
    line."""

    audio_seconds: float = 0.0
    utterances: int = 0
    final_latencies_ms: list[float] = field(default_factory=list)


class Session:
    def __init__(
        self,
        session_id: str,
        backend: SttBackend,
        vad: VadDetector,
        endpointer: Endpointer,
        stabilizer_factory: Callable[[], Stabilizer],
        stream_config: StreamConfig = StreamConfig(),  # noqa: B008 (frozen, immutable default)
        metrics_labels: dict[str, str] | None = None,
        audio_queue_chunks: int = 64,
        audio_overflow_policy: str = "drop_oldest",
    ) -> None:
        self.session_id = session_id
        self._backend = backend
        self._vad = vad
        self._endpointer = endpointer
        self._stabilizer_factory = stabilizer_factory
        self._stream_config = stream_config
        self._metrics_labels = metrics_labels
        self._audio_overflow_policy = audio_overflow_policy
        self.stats = SessionStats()

        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        # Bounded input-audio queue (spec §13 backpressure): push_audio()
        # enqueues onto this instead of running the pipeline inline, so a
        # slow backend never stalls the caller (e.g. the WS receive loop).
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=audio_queue_chunks)
        self._feeder: asyncio.Task[None] | None = None
        self._last_drop_log_ts = 0.0
        self._slicer = FrameSlicer(endpointer.cfg.frame_ms)
        self._seq = 0
        self._utterance_id = 0
        self._audio_ms = 0.0
        self._stream: SttStream | None = None
        self._reader: asyncio.Task[str] | None = None
        self._stabilizer: Stabilizer | None = None
        self._utterance_started_ts = 0.0
        self._first_partial_ms: float | None = None
        self._ended = False
        self._input_ended = False

    def _ensure_feeder(self) -> None:
        if self._feeder is None and not self._ended:
            self._feeder = asyncio.create_task(self._feed())

    async def push_audio(self, chunk: AudioChunk) -> None:
        # _input_ended guard: once end_input() has queued the _END_INPUT
        # sentinel, late pushes are no-ops. Without it, a late push against a
        # full queue under drop_oldest could evict the sentinel itself.
        if self._ended or self._input_ended:
            return
        self._ensure_feeder()
        try:
            self._audio_queue.put_nowait(chunk)
            return
        except asyncio.QueueFull:
            pass

        if self._audio_overflow_policy == "error":
            await self._shed_error()
            return

        # drop_oldest: evict the oldest queued chunk to make room for the new
        # one. No `await` runs between the eviction and the re-insertion, so
        # this can never race with the feeder task (single-threaded event
        # loop) — the re-insertion is guaranteed to succeed.
        with contextlib.suppress(asyncio.QueueEmpty):
            self._audio_queue.get_nowait()
        if self._metrics_labels is not None:
            AUDIO_DROPPED.labels(backend=self._metrics_labels["backend"]).inc()
        self._log_drop_ratelimited()
        with contextlib.suppress(asyncio.QueueFull):
            self._audio_queue.put_nowait(chunk)

    def _log_drop_ratelimited(self) -> None:
        now = time.monotonic()
        if now - self._last_drop_log_ts >= _DROP_LOG_INTERVAL_S:
            self._last_drop_log_ts = now
            logger.warning(
                "audio.dropped", session_id=self.session_id, policy="drop_oldest"
            )

    async def _shed_error(self) -> None:
        """audio_overflow_policy="error": emit ERROR exactly once and end the
        session cleanly (mirrors _push_audio_safe's cleanup shape)."""
        if self._ended:
            return
        self._emit(
            EventType.ERROR,
            error_code="backpressure",
            recoverable=False,
            message="audio queue overflow: shedding under policy=error",
        )
        await self.abort()

    async def _feed(self) -> None:
        """Single consumer of the bounded audio queue: dequeues chunks (or
        the end-of-input sentinel) and runs the existing inline pipeline.

        Returns as soon as the session ends for any reason, rather than
        looping back to a `get()` that would otherwise block forever once
        nothing is left to push it (e.g. a mid-utterance backend failure
        ends the session without anyone calling end_input()).

        Any unexpected exception NOT already converted to an ERROR event by
        the pipeline (e.g. a VAD whose is_speech() raises) must not escape
        this task silently — that would leave the session hung (`_ended`
        never set, events() blocked forever). Convert it to a terminal
        ERROR + cleanup, mirroring _push_audio_safe's shape. CancelledError
        (abort()) is BaseException and deliberately not caught here."""
        try:
            while True:
                item = await self._audio_queue.get()
                if self._ended:
                    return
                if item is _END_INPUT:
                    for action in self._endpointer.flush():
                        await self._apply(action, time.monotonic())
                    self._end()
                    return
                await self._process_chunk(item)
                if self._ended:
                    return
        except Exception as exc:
            self._emit(
                EventType.ERROR,
                error_code="internal_error",
                recoverable=False,
                message=str(exc),
            )
            if self._stream is not None:
                with contextlib.suppress(Exception):
                    await self._stream.close()
                self._stream = None
            self._reader = None
            self._end()

    async def _process_chunk(self, chunk: AudioChunk) -> None:
        audio_seconds = chunk.duration_ms / 1000.0
        self.stats.audio_seconds += audio_seconds
        if self._metrics_labels is not None:
            AUDIO_SECONDS.labels(api=self._metrics_labels["api"]).inc(audio_seconds)
        for frame in self._slicer.push(chunk.data):
            if self._ended:
                break
            self._audio_ms += self._endpointer.cfg.frame_ms
            is_speech = self._vad.is_speech(frame)
            for action in self._endpointer.process(frame, is_speech):
                await self._apply(action, chunk.ingest_ts)

    async def end_input(self) -> None:
        if self._ended or self._input_ended:
            return
        # Set synchronously, before any await: closes the window where a
        # concurrent push_audio() against a full queue could (under
        # drop_oldest) evict the _END_INPUT sentinel we are about to queue.
        self._input_ended = True
        self._ensure_feeder()
        # Enqueue the sentinel after any audio already queued ahead of it
        # (FIFO), so the feeder drains all pending audio before finalizing.
        # Poll rather than a blocking put(): a blocking put() could hang
        # forever if the feeder has already exited (e.g. a backend error
        # ended the session while this queue was full).
        while True:
            if self._ended:
                return
            try:
                self._audio_queue.put_nowait(_END_INPUT)
                return
            except asyncio.QueueFull:
                await asyncio.sleep(0)

    async def abort(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._feeder is not None:
            self._feeder.cancel()
            with contextlib.suppress(BaseException):
                await self._feeder
            self._feeder = None
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(BaseException):
                await self._reader
            self._reader = None
        if self._stream is not None:
            await self._stream.close()
            self._stream = None
        self._queue.put_nowait(None)

    def _end(self) -> None:
        self._ended = True
        self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def _apply(self, action: EndpointAction, ingest_ts: float) -> None:
        if isinstance(action, StartUtterance):
            try:
                self._stream = await self._backend.create_stream(self._stream_config)
            except Exception as exc:
                self._emit(
                    EventType.ERROR,
                    error_code="backend_error",
                    recoverable=False,
                    message=str(exc),
                )
                self._end()
                return
            self._stabilizer = self._stabilizer_factory()
            self._utterance_started_ts = time.monotonic()
            self._first_partial_ms = None
            self._emit(EventType.SPEECH_START)
            self._reader = asyncio.create_task(self._read_backend(self._stream))
            for frame in action.frames:
                if not await self._push_audio_safe(
                    self._stream, AudioChunk(data=frame, ingest_ts=ingest_ts)
                ):
                    return
        elif isinstance(action, SpeechAudio):
            if self._stream is None:
                return
            await self._push_audio_safe(
                self._stream, AudioChunk(data=action.frame, ingest_ts=ingest_ts)
            )
        elif isinstance(action, EndUtterance):
            stream, reader = self._stream, self._reader
            if stream is None or reader is None:
                return
            self._emit(EventType.SPEECH_END)
            endpoint_ts = time.monotonic()
            try:
                await stream.finalize()
                final_text = await reader
            except asyncio.CancelledError:
                # A concurrent abort() owns cleanup of this stream/reader —
                # but if THIS task (the feeder, since _apply only runs on the
                # feeder task) is itself being cancelled by that abort(), the
                # cancellation must propagate rather than be swallowed, or
                # the feeder would report a clean finish while abort() thinks
                # it cancelled it.
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                return
            except Exception as exc:
                self._emit(
                    EventType.ERROR,
                    error_code="backend_error",
                    recoverable=False,
                    message=str(exc),
                )
                with contextlib.suppress(Exception):
                    await stream.close()
                self._stream = None
                self._reader = None
                self._end()
                return
            latency = {"final_ms": (time.monotonic() - endpoint_ts) * 1000.0}
            if self._first_partial_ms is not None:
                latency["first_partial_ms"] = self._first_partial_ms
            self._emit(EventType.FINAL, text=final_text, latency=latency)
            self.stats.utterances += 1
            self.stats.final_latencies_ms.append(latency["final_ms"])
            if self._metrics_labels is not None:
                backend_label = self._metrics_labels["backend"]
                UTTERANCES.labels(backend=backend_label, end_reason=action.reason).inc()
                FINAL_MS.labels(backend=backend_label).observe(latency["final_ms"])
                if "first_partial_ms" in latency:
                    FIRST_PARTIAL_MS.labels(backend=backend_label).observe(
                        latency["first_partial_ms"]
                    )
            await stream.close()
            self._stream = None
            self._reader = None
            self._utterance_id += 1

    async def _push_audio_safe(self, stream: SttStream, chunk: AudioChunk) -> bool:
        """push_audio() to the backend stream, converting a native decode
        failure into an ERROR event + session end (mirrors the EndUtterance
        finalize failure handler's shape). Returns False if the push failed
        (caller must stop feeding further frames for this action)."""
        try:
            await stream.push_audio(chunk)
        except Exception as exc:
            self._emit(
                EventType.ERROR,
                error_code="backend_error",
                recoverable=False,
                message=str(exc),
            )
            with contextlib.suppress(Exception):
                await stream.close()
            self._stream = None
            self._reader = None
            self._end()
            return False
        return True

    async def _read_backend(self, stream: SttStream) -> str:
        """Consume backend events for one utterance; returns the final text."""
        if self._stabilizer is None:
            return ""
        final_text = ""
        async for bev in stream.events():
            if bev.kind == "partial":
                if self._first_partial_ms is None:
                    self._first_partial_ms = (
                        time.monotonic() - self._utterance_started_ts
                    ) * 1000.0
                upd = self._stabilizer.update(bev.text, bev.audio_time_ms)
                self._emit(
                    EventType.PARTIAL,
                    stable_text=upd.stable_text,
                    volatile_text=upd.volatile_text,
                )
                if upd.newly_committed:
                    self._emit(EventType.STABILIZED, text=upd.newly_committed)
            else:
                final_text = bev.text
        return final_text

    def _emit(self, type_: EventType, **kwargs) -> None:
        if type_ is EventType.ERROR and self._metrics_labels is not None:
            ERRORS.labels(code=kwargs.get("error_code") or "unknown").inc()
        ev = TranscriptEvent(
            type=type_,
            session_id=self.session_id,
            utterance_id=self._utterance_id,
            seq=self._seq,
            audio_time_ms=self._audio_ms,
            emitted_ts=time.monotonic(),
            **kwargs,
        )
        self._seq += 1
        self._queue.put_nowait(ev)
