import asyncio

from stt_server.backends.base import BackendCapabilities, SttBackend, SttStream
from stt_server.backends.mock import MockBackend, MockUtteranceScript
from stt_server.config.settings import EndpointingConfig, StabilizerConfig
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import EnergyVad
from stt_server.metrics.registry import (
    AUDIO_SECONDS,
    ERRORS,
    FINAL_MS,
    FIRST_PARTIAL_MS,
    UTTERANCES,
)
from tests.helpers.audio import make_silence, make_tone

EP_CFG = EndpointingConfig(
    frame_ms=30, pre_roll_ms=90, min_silence_ms=90, max_utterance_ms=60000, speech_start_frames=2
)
# Commit instantly so PARTIAL stable/volatile behavior is easy to assert.
STAB_CFG = StabilizerConfig(min_partials=1, min_stable_ms=0.0)
SCRIPTS = [
    MockUtteranceScript(partials=("hello", "hello world"), final="Hello world."),
    MockUtteranceScript(partials=("again",), final="Again."),
]


def make_session(scripts=SCRIPTS) -> Session:
    return Session(
        session_id="s-test",
        backend=MockBackend(partial_interval_ms=100.0, scripts=list(scripts)),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
    )


async def run_session(session: Session, audio: bytes) -> list:
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=audio, ingest_ts=0.0))
    await session.end_input()
    await asyncio.wait_for(task, timeout=5.0)
    return events


async def test_single_utterance_event_flow():
    events = await run_session(make_session(), make_tone(600) + make_silence(300))
    types = [e.type for e in events]

    assert types[0] == EventType.SPEECH_START
    assert types.count(EventType.FINAL) == 1
    assert types[-1] == EventType.FINAL

    final = events[-1]
    assert final.text == "Hello world."          # backend final, verbatim
    assert final.utterance_id == 0
    assert "final_ms" in final.latency
    assert "first_partial_ms" in final.latency

    partials = [e for e in events if e.type == EventType.PARTIAL]
    assert len(partials) == 2
    assert partials[0].volatile_text or partials[0].stable_text  # carries text
    assert all(events.index(p) < events.index(final) for p in partials)

    stabilized = [e.text for e in events if e.type == EventType.STABILIZED]
    assert " ".join(stabilized).split() == ["hello", "world"]

    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_two_utterances_increment_utterance_id():
    audio = (
        make_tone(600) + make_silence(300) + make_tone(600) + make_silence(300)
    )
    events = await run_session(make_session(), audio)
    finals = [e for e in events if e.type == EventType.FINAL]
    assert [f.utterance_id for f in finals] == [0, 1]
    assert [f.text for f in finals] == ["Hello world.", "Again."]


async def test_end_input_mid_utterance_flushes_final():
    events = await run_session(make_session(), make_tone(600))  # no trailing silence
    finals = [e for e in events if e.type == EventType.FINAL]
    assert len(finals) == 1
    assert finals[0].text == "Hello world."


async def test_silence_only_produces_no_events():
    events = await run_session(make_session(), make_silence(600))
    assert events == []


async def test_abort_ends_event_stream():
    session = make_session()
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await session.abort()
    await asyncio.wait_for(task, timeout=5.0)  # iterator must end
    assert not any(e.type == EventType.FINAL for e in events)


class FailingBackend(SttBackend):
    name = "failing"
    capabilities = BackendCapabilities(streaming=True, languages=("en",))

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def create_stream(self, cfg):
        raise RuntimeError("boom")


async def test_backend_failure_emits_error_and_ends_stream():
    session = Session(
        session_id="s-err",
        backend=FailingBackend(),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
    )
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    # make_tone(300) at frame_ms=30 slices into 10 frames; StartUtterance (and
    # the create_stream failure) fires partway through, well before the last
    # frame of this single push_audio() call is processed.
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)  # must NOT hang
    errors = [e for e in events if e.type == EventType.ERROR]
    # Exactly one ERROR despite many frames past the failure point in the same
    # chunk: covers the push_audio() mid-chunk `_ended` guard (Fix 3).
    assert len(errors) == 1
    assert errors[0].error_code == "backend_error"
    assert errors[0].recoverable is False
    assert "boom" in errors[0].message


class PushFailStream(SttStream):
    """A stream whose push_audio() raises after `fail_after` successful
    calls, simulating a native backend decode failure mid-utterance."""

    def __init__(self, fail_after: int = 0) -> None:
        self._fail_after = fail_after
        self._calls = 0

    async def push_audio(self, chunk: AudioChunk) -> None:
        self._calls += 1
        if self._calls > self._fail_after:
            raise RuntimeError("decode boom")

    async def events(self):
        # Never yields: the reader task blocks here until close()/abort()
        # cancels it, matching a real backend stream mid-decode.
        await asyncio.Future()
        yield  # pragma: no cover - unreachable, satisfies async generator typing

    async def finalize(self) -> None: ...

    async def close(self) -> None: ...


class PushFailBackend(SttBackend):
    name = "push-failing"
    capabilities = BackendCapabilities(streaming=True, languages=("en",))

    def __init__(self, fail_after: int = 0) -> None:
        self._fail_after = fail_after

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def create_stream(self, cfg):
        return PushFailStream(fail_after=self._fail_after)


async def test_push_audio_failure_mid_utterance_emits_error_and_ends_stream():
    """SpeechAudio path: push_audio() succeeds during the pre-roll frames but
    raises on a later, mid-utterance frame (I-1)."""
    session = Session(
        session_id="s-push-err",
        backend=PushFailBackend(fail_after=2),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
    )
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=make_tone(600), ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)  # must not hang

    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].error_code == "backend_error"
    assert errors[0].recoverable is False
    assert "decode boom" in errors[0].message
    assert not any(e.type == EventType.FINAL for e in events)


async def test_push_audio_failure_during_preroll_emits_error_and_ends_stream():
    """StartUtterance pre-roll path: push_audio() raises on the very first
    pre-roll frame (I-1)."""
    session = Session(
        session_id="s-push-err-preroll",
        backend=PushFailBackend(fail_after=0),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
    )
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)  # must not hang

    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 1
    assert errors[0].error_code == "backend_error"
    assert "decode boom" in errors[0].message


async def test_calls_after_abort_are_noops():
    session = make_session()
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await session.abort()
    # none of these may raise, emit, or enqueue a second sentinel
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await session.end_input()
    await session.abort()
    await asyncio.wait_for(task, timeout=5.0)
    assert not any(e.type == EventType.FINAL for e in events)


async def test_stabilizer_uses_audio_time_not_wall_clock():
    """Faster-than-real-time input must still commit stable prefixes.

    With min_stable_ms=150 and partials at 100ms audio intervals, the third
    partial (audio time 300ms) must commit the token first seen at 100ms
    (300ms - 100ms = 200ms >= 150ms) even though wall-clock elapsed is ~0ms.
    """
    session = Session(
        session_id="s-clock",
        backend=MockBackend(
            partial_interval_ms=100.0,
            scripts=[
                MockUtteranceScript(
                    partials=("hello", "hello world", "hello world again"),
                    final="Hello world again.",
                )
            ],
        ),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(
            StabilizerConfig(min_partials=2, min_stable_ms=150.0)
        ),
    )
    events = await run_session(session, make_tone(600) + make_silence(300))
    stabilized = [e for e in events if e.type == EventType.STABILIZED]
    # Wall-clock elapsed is near zero; only audio time can satisfy 150ms.
    assert stabilized, "no STABILIZED events — stabilizer is on the wall clock"
    assert stabilized[0].text == "hello"


# -- Session.stats: accumulates regardless of metrics_labels --


async def test_stats_accumulate_without_metrics_labels():
    session = make_session()
    events = await run_session(session, make_tone(600) + make_silence(300))
    finals = [e for e in events if e.type == EventType.FINAL]

    assert session.stats.utterances == len(finals) == 1
    assert session.stats.audio_seconds > 0
    assert session.stats.final_latencies_ms == [finals[0].latency["final_ms"]]


async def test_stats_accumulate_across_two_utterances():
    session = make_session()
    audio = make_tone(600) + make_silence(300) + make_tone(600) + make_silence(300)
    events = await run_session(session, audio)
    finals = [e for e in events if e.type == EventType.FINAL]

    assert session.stats.utterances == len(finals) == 2
    assert session.stats.final_latencies_ms == [f.latency["final_ms"] for f in finals]


async def test_stats_present_even_when_session_has_no_speech():
    session = make_session()
    await run_session(session, make_silence(600))
    assert session.stats.utterances == 0
    assert session.stats.final_latencies_ms == []
    assert session.stats.audio_seconds > 0  # silence is still ingested audio


# -- Session(metrics_labels=...): observes the Prometheus registry --


async def test_metrics_labels_none_by_default_does_not_touch_prometheus():
    session = make_session()
    await run_session(session, make_tone(600) + make_silence(300))
    assert UTTERANCES.labels(backend="mock", end_reason="silence")._value.get() == 0.0


async def test_metrics_labels_observes_utterances_and_latency_histograms():
    session = Session(
        session_id="s-metrics",
        backend=MockBackend(partial_interval_ms=100.0, scripts=list(SCRIPTS)),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
        metrics_labels={"backend": "mock", "api": "native"},
    )
    await run_session(session, make_tone(600) + make_silence(300))

    assert UTTERANCES.labels(backend="mock", end_reason="silence")._value.get() == 1.0
    assert FINAL_MS.labels(backend="mock")._sum.get() > 0.0
    assert FIRST_PARTIAL_MS.labels(backend="mock")._sum.get() > 0.0


async def test_metrics_labels_observes_audio_seconds_by_api():
    session = Session(
        session_id="s-metrics-audio",
        backend=MockBackend(partial_interval_ms=100.0, scripts=list(SCRIPTS)),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
        metrics_labels={"backend": "mock", "api": "realtime"},
    )
    await run_session(session, make_tone(600) + make_silence(300))

    assert AUDIO_SECONDS.labels(api="realtime")._value.get() == session.stats.audio_seconds
    assert AUDIO_SECONDS.labels(api="native")._value.get() == 0.0


async def test_metrics_labels_observes_end_reason_input_ended():
    session = Session(
        session_id="s-metrics-flush",
        backend=MockBackend(partial_interval_ms=100.0, scripts=list(SCRIPTS)),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
        metrics_labels={"backend": "mock", "api": "native"},
    )
    await run_session(session, make_tone(600))  # no trailing silence: flush on end_input

    assert UTTERANCES.labels(backend="mock", end_reason="input_ended")._value.get() == 1.0


async def test_metrics_labels_observes_errors_on_backend_failure():
    session = Session(
        session_id="s-metrics-err",
        backend=FailingBackend(),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
        metrics_labels={"backend": "failing", "api": "native"},
    )
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await asyncio.wait_for(task, timeout=5.0)

    assert ERRORS.labels(code="backend_error")._value.get() == 1.0
