"""Unit tests for the nemotron backend that run WITHOUT nemo or torch.

The real streaming step is deliberately factored so that every torch call
lives behind `_TorchOps` (see the backend module docstring): these tests
substitute a list-based `_ListOps` and a `_FakeNemoModel` that pins the real
NeMo 3.0.0 call signatures, so the adapter logic under test — byte-stride
buffering, the 320-sample raw look-back, the 9-frame left feature context,
cache/hypothesis threading, prompt-under-lock, tag stripping and the final
flush — is the same code the GPU runs. No `import torch` anywhere here.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from stt_server.backends.base import BackendUnavailableError, StreamConfig
from stt_server.backends.registry import create_backend
from stt_server.config.settings import BackendDef, load_settings
from stt_server.core.events import AudioChunk

NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None

# Synthetic geometry: att_context_size [56, 3] -> chunk_size[1] = 8 + 8*3 = 32
# feature frames = 5120 samples = 10240 bytes of PCM16. Small enough to push
# whole chunks in a test, and a real trained lookahead.
CHUNK_FRAMES = 32
PRE_ENCODE_FRAMES = 9
DROP_EXTRA_PRE_ENCODED = 2
CHUNK_SAMPLES = CHUNK_FRAMES * 160
CHUNK_BYTES = CHUNK_SAMPLES * 2


def _pcm(n_samples: int) -> bytes:
    """`n_samples` of quiet, non-zero PCM16."""
    return b"\x00\x01" * n_samples


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _ListOps:
    """Stand-in for `_TorchOps`.

    Audio is a flat list of floats; a "feature tensor" is a flat list with
    one element per frame (the 128 mel bins are irrelevant to the adapter
    logic, only widths are). Every method mirrors the real one's contract.
    """

    def audio_tensor(self, samples: Any) -> list[float]:
        return list(samples)

    def zeros_audio(self, n: int) -> list[float]:
        return [0.0] * n

    def cat_audio(self, left: list[float], right: list[float]) -> list[float]:
        return list(left) + list(right)

    def tail_audio(self, audio: list[float], n: int) -> list[float]:
        return list(audio[-n:])

    def audio_len(self, audio: list[float]) -> int:
        return len(audio)

    def batch_audio(self, audio: list[float]) -> list[float]:
        return list(audio)  # the batch dimension is fiction here

    def length_tensor(self, n: int) -> int:
        return n

    def cat_feats(self, left: list[float], right: list[float]) -> list[float]:
        return list(left) + list(right)

    def head_feats(self, feats: list[float], n: int) -> list[float]:
        return list(feats[:n])

    def tail_feats(self, feats: list[float], n: int) -> list[float]:
        return list(feats[-n:])

    def zeros_feats(self, like: Any, n_mels: int, n: int) -> list[float]:
        return [0.0] * n

    def feats_width(self, feats: list[float]) -> int:
        return len(feats)

    def inference_mode(self):
        return contextlib.nullcontext()

    def autocast(self):
        return contextlib.nullcontext()


class _FakeHypothesis:
    """`nemo...Hypothesis`, reduced to the one field this adapter reads."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakePreprocessor:
    """`AudioToMelSpectrogramPreprocessor.__call__`, real kwargs pinned.

    Reproduces the quirk the adapter compensates for: NeMo reports
    `length // 160` frames but the `center=True` STFT yields one extra
    trailing column that NeMo zeroes rather than drops
    (`features.py:405-408`, `:481-485`).
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, input_signal: Any, length: Any) -> tuple[list[float], list[int]]:
        call_index = len(self.calls)
        self.calls.append(length)
        n_frames = length // 160
        feats = [call_index * 1000.0 + i for i in range(n_frames)]
        return feats + [0.0], [n_frames]


class _FakeEncoder:
    def __init__(self) -> None:
        self.cache_state_calls: list[int] = []

    def get_initial_cache_state(
        self, batch_size: int = 1, dtype: Any = None, device: Any = None, max_dim: int = 0
    ) -> tuple[list, list, list]:
        self.cache_state_calls.append(batch_size)
        # Fresh objects on every call: the per-stream isolation test asserts
        # two streams never share a cache object.
        return (["cache_last_channel"], ["cache_last_time"], [0])


class _FakeNemoModel:
    """Pins the real NeMo 3.0.0 signatures this adapter depends on.

    No `**kwargs` catch-alls anywhere: a keyword this adapter gets wrong is
    a TypeError here, which is the whole point of the stub.
    """

    def __init__(self, texts: list[str | None] | None = None, delay: float = 0.0) -> None:
        self.encoder = _FakeEncoder()
        self.preprocessor = _FakePreprocessor()
        self.prompt_calls: list[str] = []
        self.step_calls: list[dict] = []
        self._texts = list(texts or [])
        self._delay = delay

    # PromptStreamingMixin.set_inference_prompt (mixins.py:949)
    def set_inference_prompt(self, target_lang: str) -> None:
        self.prompt_calls.append(target_lang)

    # ASRModuleMixin.conformer_stream_step (mixins.py:602-615)
    def conformer_stream_step(
        self,
        processed_signal: Any,
        processed_signal_length: Any = None,
        cache_last_channel: Any = None,
        cache_last_time: Any = None,
        cache_last_channel_len: Any = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: Any = None,
        previous_pred_out: Any = None,
        drop_extra_pre_encoded: int | None = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
        bypass_pre_encode: bool = False,
    ):
        if self._delay:
            time.sleep(self._delay)
        # hyp.text is cumulative because partial hypotheses are merged in
        # place (rnnt_greedy_decoding.py:828-834).
        base = previous_hypotheses[0].text if previous_hypotheses else ""
        scripted = self._texts.pop(0) if self._texts else None
        text = base if scripted is None else scripted
        self.step_calls.append(
            {
                "signal_width": len(processed_signal),
                "processed_signal_length": processed_signal_length,
                "keep_all_outputs": keep_all_outputs,
                "drop_extra_pre_encoded": drop_extra_pre_encoded,
                "previous_pred_out": previous_pred_out,
                "return_transcription": return_transcription,
                "cache_last_channel": cache_last_channel,
                # The prompt in force at the moment of this step: the global
                # prompt must have been set immediately before, under the lock.
                "prompt": self.prompt_calls[-1] if self.prompt_calls else None,
            }
        )
        best_hyp = [_FakeHypothesis(text)]
        return (
            [[1, 2]],  # greedy_predictions
            best_hyp,  # all_hyp_or_transcribed_texts (== best_hyp for RNNT)
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            best_hyp,
        )


def _geometry():
    from stt_server.backends.nemotron.backend import _Geometry

    return _Geometry(
        chunk_frames=CHUNK_FRAMES,
        pre_encode_frames=PRE_ENCODE_FRAMES,
        drop_extra_pre_encoded=DROP_EXTRA_PRE_ENCODED,
        n_mels=128,
    )


def _make_engine(model: _FakeNemoModel, decode_lock: threading.Lock | None = None):
    from stt_server.backends.nemotron.backend import _NemoEngine

    return _NemoEngine(
        model=model,
        preprocessor=model.preprocessor,
        ops=_ListOps(),
        geometry=_geometry(),
        decode_lock=decode_lock or threading.Lock(),
    )


def _make_stream(
    model: _FakeNemoModel,
    locale: str = "auto",
    strip_lang_tags: bool = True,
    requested_language: str | None = None,
    decode_lock: threading.Lock | None = None,
    executor: ThreadPoolExecutor | None = None,
):
    from stt_server.backends.nemotron.backend import NemotronStream

    executor = executor or ThreadPoolExecutor(max_workers=4)
    stream = NemotronStream(
        _make_engine(model, decode_lock),
        executor,
        locale=locale,
        strip_lang_tags=strip_lang_tags,
        requested_language=requested_language,
    )
    return stream, executor


async def _collect(stream) -> tuple[asyncio.Task, list]:
    events: list = []

    async def reader():
        async for ev in stream.events():
            events.append(ev)

    return asyncio.create_task(reader()), events


def _bare_backend(**overrides):
    """A NemotronBackend without going through __init__ (which requires the
    `nemo` package to be importable); attributes set to post-__init__
    defaults."""
    from stt_server.backends.nemotron.backend import NemotronBackend, normalize_language

    backend = NemotronBackend.__new__(NemotronBackend)
    backend._model_name = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    backend._att_context_size = (56, 3)
    backend._language = overrides.get("language")
    backend._locale = normalize_language(backend._language)
    backend._strip_lang_tags = overrides.get("strip_lang_tags", True)
    backend._device = "cuda"
    backend._amp = False
    backend._max_concurrent = 1
    backend._warmup_enabled = False
    backend._model_instance = None
    backend._engine = None
    backend._executor = None
    backend._decode_lock = threading.Lock()
    return backend


# --------------------------------------------------------------------------
# Construction / availability
# --------------------------------------------------------------------------


@pytest.mark.skipif(NEMO_AVAILABLE, reason="only valid when nemo is NOT installed")
def test_missing_nemo_raises_unavailable_with_extras_hint():
    with pytest.raises(BackendUnavailableError) as exc_info:
        create_backend(BackendDef(type="nemotron", options={}))
    message = str(exc_info.value)
    assert "nemo_toolkit" in message
    assert "pip install" in message
    assert "stt-server[nemotron]" in message


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"att_context_size": (56,)}, "att_context_size"),
        ({"att_context_size": (56, 6, 1)}, "att_context_size"),
        ({"att_context_size": ("56", 6)}, "att_context_size"),
        ({"att_context_size": (0, 6)}, "left context"),
        ({"att_context_size": (-1, 6)}, "left context"),
        ({"att_context_size": (56, 1)}, "right context"),  # legal but untrained
        ({"att_context_size": (56, 4)}, "right context"),
        ({"device": "tpu"}, "device"),
        ({"max_concurrent": 0}, "max_concurrent"),
        ({"max_concurrent": -2}, "max_concurrent"),
    ],
)
def test_option_validation_runs_before_the_availability_gate(kwargs, match):
    from stt_server.backends.nemotron.backend import NemotronBackend

    # Validated *before* the nemo find_spec gate, so these fail identically
    # whether or not the extra is installed (mirrors FunasrBackend).
    with pytest.raises(ValueError, match=match):
        NemotronBackend(**kwargs)


async def test_create_stream_before_start_raises_runtime_error():
    backend = _bare_backend()
    with pytest.raises(RuntimeError, match="not started"):
        await backend.create_stream(StreamConfig())


async def test_stop_with_inflight_decode_does_not_block_event_loop():
    backend = _bare_backend()
    backend._executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    backend._executor.submit(release.wait)  # slow "step" in flight
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


# --------------------------------------------------------------------------
# Buffering / stepping
# --------------------------------------------------------------------------


async def test_push_below_one_chunk_does_not_step():
    model = _FakeNemoModel()
    stream, executor = _make_stream(model)
    try:
        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES // 2), ingest_ts=0.0))
        assert model.step_calls == []
        # State (and therefore the encoder cache) is allocated lazily on the
        # first real step, never in create_stream / push below a chunk.
        assert model.encoder.cache_state_calls == []
    finally:
        executor.shutdown(wait=True)


async def test_full_chunk_steps_once_with_the_expected_tensor_geometry():
    model = _FakeNemoModel(texts=["hello"])
    stream, executor = _make_stream(model, locale="en-US")
    try:
        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        assert len(model.step_calls) == 1
        call = model.step_calls[0]
        # 9 frames of left feature context + 32 new frames (§A step 3).
        assert call["signal_width"] == PRE_ENCODE_FRAMES + CHUNK_FRAMES
        assert call["processed_signal_length"] == PRE_ENCODE_FRAMES + CHUNK_FRAMES
        assert call["keep_all_outputs"] is False  # mid-stream
        assert call["drop_extra_pre_encoded"] == DROP_EXTRA_PRE_ENCODED
        assert call["previous_pred_out"] is None  # CTC-only parameter
        assert call["return_transcription"] is True
        assert call["prompt"] == "en-US"
        # The preprocessor saw the 320-sample raw look-back plus the chunk.
        assert model.preprocessor.calls == [320 + CHUNK_SAMPLES]
        assert model.encoder.cache_state_calls == [1]
    finally:
        executor.shutdown(wait=True)


async def test_multiple_chunks_emit_cumulative_partials():
    model = _FakeNemoModel(texts=["hello", "hello world"])
    stream, executor = _make_stream(model)
    try:
        task, events = await _collect(stream)
        for _ in range(2):
            await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        partials = [e for e in events if e.kind == "partial"]
        assert [e.text for e in partials] == ["hello", "hello world"]
        assert [e.audio_time_ms for e in events] == sorted(e.audio_time_ms for e in events)
        # Second step carried the first step's look-back, not a fresh 320 zeros.
        assert model.preprocessor.calls == [320 + CHUNK_SAMPLES] * 2
    finally:
        executor.shutdown(wait=True)


async def test_unchanged_text_emits_no_duplicate_partial():
    # A step over silence legitimately leaves the cumulative hypothesis
    # unchanged (the fake scripts None -> text stays as-is). Matches the
    # sherpa/qwen3asr dedup: a duplicate would inflate the stabilizer's
    # partial-confirmation count.
    model = _FakeNemoModel(texts=["hello", None, "hello world"])
    stream, executor = _make_stream(model)
    try:
        task, events = await _collect(stream)
        for _ in range(3):
            await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        assert len(model.step_calls) == 3  # all three chunks really decoded
        partials = [e for e in events if e.kind == "partial"]
        assert [e.text for e in partials] == ["hello", "hello world"]
    finally:
        executor.shutdown(wait=True)


async def test_finalize_flushes_remainder_with_keep_all_outputs_and_one_final():
    model = _FakeNemoModel(texts=["hello", "hello world"])
    stream, executor = _make_stream(model)
    try:
        remainder_samples = 1600  # 100 ms, well below one chunk
        await stream.push_audio(
            AudioChunk(data=_pcm(CHUNK_SAMPLES + remainder_samples), ingest_ts=0.0)
        )
        assert len(model.step_calls) == 1

        task, events = await _collect(stream)
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        assert len(model.step_calls) == 2
        flush = model.step_calls[1]
        assert flush["keep_all_outputs"] is True
        # The remainder is zero-padded to a full chunk for the preprocessor,
        # then the features are truncated to the frames really covered by
        # audio (1600 // 160 = 10), so processed_signal_length stays honest.
        assert model.preprocessor.calls[-1] == 320 + CHUNK_SAMPLES
        assert flush["signal_width"] == PRE_ENCODE_FRAMES + 10
        assert flush["processed_signal_length"] == PRE_ENCODE_FRAMES + 10

        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        assert finals[0].text == "hello world"
        assert events[-1].kind == "final"
        assert stream._queue.empty(), "an event was enqueued after the sentinel"
    finally:
        executor.shutdown(wait=True)


async def test_finalize_with_empty_remainder_emits_one_final_without_stepping():
    model = _FakeNemoModel(texts=["hello"])
    stream, executor = _make_stream(model)
    try:
        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        task, events = await _collect(stream)
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        assert len(model.step_calls) == 1  # nothing left to decode
        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        assert finals[0].text == "hello"
        assert events[-1].kind == "final"
    finally:
        executor.shutdown(wait=True)


async def test_finalize_without_any_audio_emits_one_empty_final_and_never_touches_the_model():
    model = _FakeNemoModel()
    stream, executor = _make_stream(model)
    try:
        task, events = await _collect(stream)
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        assert model.step_calls == []
        assert model.encoder.cache_state_calls == []
        assert [(e.kind, e.text) for e in events] == [("final", "")]
    finally:
        executor.shutdown(wait=True)


async def test_push_and_finalize_after_done_are_noops():
    model = _FakeNemoModel(texts=["hi"])
    stream, executor = _make_stream(model)
    try:
        task, events = await _collect(stream)
        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        events_after_finalize = list(events)
        steps_after_finalize = len(model.step_calls)

        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await stream.finalize()
        await stream.close()

        assert events == events_after_finalize
        assert len(model.step_calls) == steps_after_finalize
    finally:
        executor.shutdown(wait=True)


async def test_close_waits_for_inflight_decode_and_sentinel_is_last():
    model = _FakeNemoModel(texts=["hello"], delay=0.2)
    stream, executor = _make_stream(model)
    try:
        push = asyncio.create_task(
            stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        )
        await asyncio.sleep(0.05)  # the step is now in flight
        await stream.close()
        await push

        async def drain():
            return [ev async for ev in stream.events()]

        events = await asyncio.wait_for(drain(), timeout=5.0)
        assert [(e.kind, e.text) for e in events] == [("partial", "hello")]
        assert stream._queue.empty(), "an event was enqueued after the sentinel"
    finally:
        executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# Cross-stream isolation: the global prompt + decode lock
# --------------------------------------------------------------------------


async def test_decode_lock_serializes_steps_across_streams():
    # The language prompt is model-global (mixins.py:949-969) and
    # cache_aware_stream_step mutates the shared
    # encoder.streaming_cfg.drop_extra_pre_encoded for the duration of the
    # call (mixins/streaming.py:53-74), so at most ONE step may be in flight
    # across all streams -- the lock, not the executor's max_workers, is the
    # real concurrency bound.
    from stt_server.backends.nemotron.backend import NemotronStream

    model = _FakeNemoModel()
    executor = ThreadPoolExecutor(max_workers=4)
    decode_lock = threading.Lock()
    engine = _make_engine(model, decode_lock)

    in_flight = 0
    max_seen = 0
    counter_lock = threading.Lock()
    release = threading.Event()

    def blocking_step(
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        previous_hypotheses=None,
        previous_pred_out=None,
        drop_extra_pre_encoded=None,
        return_transcription=True,
        return_log_probs=False,
        bypass_pre_encode=False,
    ):
        nonlocal in_flight, max_seen
        with counter_lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        release.wait(timeout=5.0)
        with counter_lock:
            in_flight -= 1
        best_hyp = [_FakeHypothesis("")]
        return (
            [[]],
            best_hyp,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            best_hyp,
        )

    model.conformer_stream_step = blocking_step  # type: ignore[method-assign]

    streams = [
        NemotronStream(engine, executor, locale="en-US", strip_lang_tags=True)
        for _ in range(4)
    ]
    try:
        tasks = [
            asyncio.create_task(
                s.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
            )
            for s in streams
        ]
        await asyncio.sleep(0.3)  # let all four attempt their step
        assert max_seen == 1, "the decode lock must serialize model steps"
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
    finally:
        release.set()
        executor.shutdown(wait=True)


async def test_interleaved_streams_set_their_own_prompt_before_every_step():
    # Session isolation for the model-global prompt: each step must be
    # preceded, under the same lock, by set_inference_prompt(<that stream's
    # locale>). Interleaving two languages proves the prompt switches per
    # step rather than being set once per stream.
    from stt_server.backends.nemotron.backend import NemotronStream

    model = _FakeNemoModel()
    executor = ThreadPoolExecutor(max_workers=2)
    decode_lock = threading.Lock()
    engine = _make_engine(model, decode_lock)
    zh = NemotronStream(engine, executor, locale="zh-CN")
    en = NemotronStream(engine, executor, locale="en-US")
    try:
        for _ in range(2):
            await zh.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
            await en.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        assert [c["prompt"] for c in model.step_calls] == [
            "zh-CN",
            "en-US",
            "zh-CN",
            "en-US",
        ]
        assert model.prompt_calls == ["zh-CN", "en-US", "zh-CN", "en-US"]
    finally:
        executor.shutdown(wait=True)


async def test_each_stream_owns_its_own_encoder_cache_objects():
    from stt_server.backends.nemotron.backend import NemotronStream

    model = _FakeNemoModel()
    executor = ThreadPoolExecutor(max_workers=2)
    engine = _make_engine(model)
    first = NemotronStream(engine, executor, locale="auto")
    second = NemotronStream(engine, executor, locale="auto")
    try:
        await first.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await second.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        assert model.encoder.cache_state_calls == [1, 1]
        assert first._state is not None and second._state is not None
        assert first._state.cache_last_channel is not second._state.cache_last_channel
        assert first._state.cache_last_time is not second._state.cache_last_time
        assert first._state.cache_last_channel_len is not second._state.cache_last_channel_len
        assert first._state.prev_hyps is not second._state.prev_hyps
    finally:
        executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# Language handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("  ", "auto"),
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("zh", "zh-CN"),
        ("zh-CN", "zh-CN"),
        ("zh-cn", "zh-CN"),
        ("zh_CN", "zh-CN"),
        ("ZH_cn", "zh-CN"),
        ("chinese", "zh-CN"),
        ("Chinese", "zh-CN"),
        ("mandarin", "zh-CN"),
        ("en", "en-US"),
        ("english", "en-US"),
        ("English", "en-US"),
        ("en-GB", "en-GB"),
        ("en_gb", "en-GB"),
        ("no", "nb-NO"),
        ("xx", "auto"),
        ("klingon", "auto"),
        ("zz-ZZ", "auto"),
    ],
)
def test_normalize_language(value, expected):
    from stt_server.backends.nemotron.backend import normalize_language

    assert normalize_language(value) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("你好。 <zh-CN>", ("你好。", "zh-CN")),
        ("hello there <en-US>", ("hello there", "en-US")),
        ("<en-US>hello <en-US>", ("hello", "en-US")),
        ("a <en-US> b <zh-CN>", ("a b", "zh-CN")),  # last tag wins
        ("no tag here", ("no tag here", None)),
        ("", ("", None)),
    ],
)
def test_split_lang_tag(raw, expected):
    from stt_server.backends.nemotron.backend import split_lang_tag

    assert split_lang_tag(raw) == expected


def test_split_lang_tag_without_stripping_keeps_the_text():
    from stt_server.backends.nemotron.backend import split_lang_tag

    assert split_lang_tag("你好。 <zh-CN>", strip=False) == ("你好。 <zh-CN>", "zh-CN")


@pytest.mark.parametrize(
    ("locale", "expected"),
    [
        ("zh-CN", "Chinese"),
        ("en-US", "English"),
        ("en-GB", "English"),
        ("fr-CA", "French"),
        ("nn-NO", "Norwegian Nynorsk"),
        ("auto", None),
        (None, None),
        ("qq-QQ", "qq-QQ"),  # unmapped: fall back to the locale itself
    ],
)
def test_locale_to_language_name(locale, expected):
    from stt_server.backends.nemotron.backend import locale_to_language_name

    assert locale_to_language_name(locale) == expected


@pytest.mark.parametrize(
    ("att_context_size", "expected"),
    [((56, 6), 8960), ([56, 6], 8960), ((56, 3), 5120), ((56, 0), 1280), ((56, 13), 17920)],
)
def test_chunk_samples_for(att_context_size, expected):
    from stt_server.backends.nemotron.backend import chunk_samples_for

    # (8 + 8R) feature frames x 160 samples: [56,6] -> 560 ms, [56,3] -> 320 ms.
    assert chunk_samples_for(att_context_size) == expected


async def test_final_reports_the_detected_language_in_auto_mode():
    model = _FakeNemoModel(texts=["你好 <zh-CN>", "你好世界 <zh-CN>"])
    stream, executor = _make_stream(model, locale="auto", requested_language=None)
    try:
        task, events = await _collect(stream)
        await stream.push_audio(
            AudioChunk(data=_pcm(CHUNK_SAMPLES + 1600), ingest_ts=0.0)
        )
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        partials = [e for e in events if e.kind == "partial"]
        finals = [e for e in events if e.kind == "final"]
        assert [e.text for e in partials] == ["你好"]  # tag stripped
        assert all(e.language is None for e in partials)  # partials never carry it
        assert finals[0].text == "你好世界"
        assert finals[0].language == "Chinese"
    finally:
        executor.shutdown(wait=True)


async def test_final_reports_the_requested_language_when_explicit():
    # An explicitly pinned language is echoed back even if the model's tag
    # said otherwise: the caller asked for it, the prompt enforced it.
    model = _FakeNemoModel(texts=["hello <en-US>"])
    stream, executor = _make_stream(
        model, locale="en-US", requested_language="English"
    )
    try:
        task, events = await _collect(stream)
        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        finals = [e for e in events if e.kind == "final"]
        assert finals[0].text == "hello"
        assert finals[0].language == "English"
        assert model.prompt_calls == ["en-US"]
    finally:
        executor.shutdown(wait=True)


async def test_strip_lang_tags_false_keeps_the_raw_tag_but_still_detects():
    model = _FakeNemoModel(texts=["hello <en-US>"])
    stream, executor = _make_stream(model, locale="auto", strip_lang_tags=False)
    try:
        task, events = await _collect(stream)
        await stream.push_audio(AudioChunk(data=_pcm(CHUNK_SAMPLES), ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)

        finals = [e for e in events if e.kind == "final"]
        assert finals[0].text == "hello <en-US>"
        assert finals[0].language == "English"
    finally:
        executor.shutdown(wait=True)


async def test_create_stream_uses_the_constructor_language():
    backend = _bare_backend(language="chinese")
    model = _FakeNemoModel()
    backend._engine = _make_engine(model, backend._decode_lock)
    backend._executor = ThreadPoolExecutor(max_workers=1)
    try:
        stream = await backend.create_stream(StreamConfig())
        assert stream._locale == "zh-CN"
        assert stream._requested_language == "Chinese"
    finally:
        backend._executor.shutdown(wait=True)


async def test_per_request_language_overrides_the_constructor_language():
    backend = _bare_backend(language="zh-CN")
    model = _FakeNemoModel()
    backend._engine = _make_engine(model, backend._decode_lock)
    backend._executor = ThreadPoolExecutor(max_workers=1)
    try:
        stream = await backend.create_stream(StreamConfig(language="en"))
        assert stream._locale == "en-US"
        assert stream._requested_language == "English"

        # An unrecognized hint falls back to auto, never an error.
        fallback = await backend.create_stream(StreamConfig(language="klingon"))
        assert fallback._locale == "auto"
        assert fallback._requested_language is None
    finally:
        backend._executor.shutdown(wait=True)


# --------------------------------------------------------------------------
# Config / capabilities
# --------------------------------------------------------------------------


def test_nemotron_config_roundtrips_through_settings():
    from stt_server.backends.nemotron.backend import NemotronBackend

    settings = load_settings("configs/nemotron.yaml")
    backend_def = settings.backends["nemotron"]
    assert backend_def.type == "nemotron"
    assert backend_def.options == {
        "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "att_context_size": [56, 6],
        "language": None,
        "strip_lang_tags": True,
        "device": "cuda",
        "amp": False,
        "max_concurrent": 8,
        "warmup": True,
    }
    # The options' keys must be exactly NemotronBackend kwargs, so a bare
    # `NemotronBackend(**backend_def.options)` (skipping only the
    # availability check) matches the real constructor signature.
    sig = inspect.signature(NemotronBackend.__init__)
    accepted = set(sig.parameters) - {"self"}
    assert set(backend_def.options) <= accepted


def test_capabilities_flags():
    from stt_server.backends.nemotron.backend import SUPPORTED_LOCALES, NemotronBackend

    caps = NemotronBackend.capabilities
    assert caps.streaming is True
    assert caps.native_endpointing is False
    assert caps.batch_decode is False
    # zh + en from one streaming model is this backend's reason to exist.
    assert caps.languages[:2] == ("en", "zh")
    assert len(set(caps.languages)) == len(caps.languages)  # deduped
    assert len(SUPPORTED_LOCALES) == 40
    assert set(caps.languages) == {loc.split("-")[0] for loc in SUPPORTED_LOCALES}
