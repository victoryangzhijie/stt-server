"""Model+GPU-marked tests: run the real Nemotron backend against real audio.

Requires the `nemotron` extra (`nemo_toolkit[asr]>=3.0.0`) AND a CUDA GPU.
Not runnable on this development machine (no CUDA); written for the A10
bring-up described in `docs/nemotron_a10_runbook.md`, which is also where
the UNVERIFIED checklist these tests are meant to close lives.

The zh/en language tests need real speech in each language. They are gated
on `benchmarks/audio/{zh,en}_sample.wav` (16 kHz mono, produced by the
runbook's §6 fixture step) and skip cleanly when those files are absent —
they are gitignored, since this repo ships exactly one committed audio
fixture with documented provenance.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import wave
from pathlib import Path

import pytest

from stt_server.backends.base import StreamConfig
from stt_server.core.events import AudioChunk

from .conformance import BackendConformanceSuite

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPEECH_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"
ZH_SAMPLE_PATH = REPO_ROOT / "benchmarks" / "audio" / "zh_sample.wav"
EN_SAMPLE_PATH = REPO_ROOT / "benchmarks" / "audio" / "en_sample.wav"

NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None

# CUDA-availability gate: NeMo will happily import on a CPU-only box, and
# `device: cuda` then fails at model load rather than skipping. Detect via
# `shutil.which("nvidia-smi")` rather than `torch.cuda.is_available()`: it is
# a cheap stdlib call that works at collection time even when the (heavy,
# torch-pulling) `nemotron` extra is not installed at all.
CUDA_AVAILABLE = shutil.which("nvidia-smi") is not None

pytestmark = [
    pytest.mark.model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not NEMO_AVAILABLE,
        reason="nemo_toolkit not installed; pip install 'stt-server[nemotron]'",
    ),
    pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason=(
            "no CUDA GPU detected (nvidia-smi not on PATH); the cache-aware "
            "NeMo model is loaded onto `device: cuda` and fails (rather than "
            "skipping) on a CPU-only box -- see docs/nemotron_a10_runbook.md"
        ),
    ),
]

CHUNK_BYTES_100MS = 3200  # 100 ms at 16 kHz / 16-bit mono


def _speech_fixture() -> bytes:
    return SPEECH_FIXTURE_PATH.read_bytes()


def _read_wav_pcm16(path: Path) -> bytes:
    """Read a 16 kHz mono PCM16 WAV into raw little-endian frames."""
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16000, f"{path.name}: expected 16 kHz"
        assert wav.getnchannels() == 1, f"{path.name}: expected mono"
        assert wav.getsampwidth() == 2, f"{path.name}: expected 16-bit PCM"
        return wav.readframes(wav.getnframes())


def _require_sample(path: Path) -> bytes:
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} not present; generate it with "
            "docs/nemotron_a10_runbook.md §6"
        )
    return _read_wav_pcm16(path)


async def _stream_utterance(
    backend, audio: bytes, cfg: StreamConfig | None = None, chunk_bytes: int = CHUNK_BYTES_100MS
) -> list:
    """Push `audio` in real-ish chunks through one stream; return its events."""
    stream = await backend.create_stream(cfg or StreamConfig())
    events: list = []

    async def reader():
        async for ev in stream.events():
            events.append(ev)

    task = asyncio.create_task(reader())
    for i in range(0, len(audio), chunk_bytes):
        await stream.push_audio(AudioChunk(data=audio[i : i + chunk_bytes], ingest_ts=0.0))
    await stream.finalize()
    await asyncio.wait_for(task, timeout=120.0)
    await stream.close()
    return events


def _assert_clean_final(events: list) -> str:
    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 1
    assert events[-1].kind == "final"
    text = finals[0].text.strip()
    assert text, "expected a non-empty transcript from real speech audio"
    # The `<xx-XX>` locale tag must have been stripped by the adapter
    # (NeMo's own strip_lang_tags is deliberately left off).
    assert "<" not in text and ">" not in text, f"locale tag leaked into text: {text!r}"
    times = [e.audio_time_ms for e in events]
    assert times == sorted(times)
    return text


class TestNemotronConformance(BackendConformanceSuite):
    @pytest.fixture(scope="module")
    def backend(self):
        from stt_server.backends.nemotron.backend import NemotronBackend

        return NemotronBackend()

    async def _run_utterance(self, backend, audio: bytes) -> list:
        # The conformance suite's synthetic constant-amplitude PCM produces
        # no transcript; drive the real speech fixture instead.
        return await super()._run_utterance(backend, _speech_fixture())


async def test_real_model_streams_incremental_partials_and_one_final():
    from stt_server.backends.nemotron.backend import NemotronBackend

    backend = NemotronBackend()
    await backend.start()
    try:
        events = await _stream_utterance(backend, _speech_fixture())
        text = _assert_clean_final(events)
        finals = [e for e in events if e.kind == "final"]
        # The 1 s fixture is too short for the model to commit a language
        # decision: on the A10 bring-up (2026-09) the `<xx-XX>` tag never
        # appeared on it, in any of auto/zh-CN/en-US, while the 6 s
        # zh/en_sample.wav clips emit it reliably (pinned by
        # test_auto_detects_zh). Language detection therefore needs a few
        # seconds of speech; asserting it here would fail on a clip too
        # short to exercise it.
        print(f"\n[nemotron real transcript, en fixture] {text!r} ({finals[0].language})\n")
    finally:
        await backend.stop()


async def test_explicit_zh_and_en_language():
    from stt_server.backends.nemotron.backend import NemotronBackend

    zh_audio = _require_sample(ZH_SAMPLE_PATH)
    en_audio = _require_sample(EN_SAMPLE_PATH)

    backend = NemotronBackend()
    await backend.start()
    try:
        zh_events = await _stream_utterance(
            backend, zh_audio, StreamConfig(language="zh-CN")
        )
        zh_text = _assert_clean_final(zh_events)
        assert [e for e in zh_events if e.kind == "final"][0].language == "Chinese"

        en_events = await _stream_utterance(backend, en_audio, StreamConfig(language="en"))
        en_text = _assert_clean_final(en_events)
        assert [e for e in en_events if e.kind == "final"][0].language == "English"

        print(f"\n[nemotron zh explicit] {zh_text!r}\n[nemotron en explicit] {en_text!r}\n")
    finally:
        await backend.stop()


async def test_auto_detects_zh():
    from stt_server.backends.nemotron.backend import NemotronBackend

    zh_audio = _require_sample(ZH_SAMPLE_PATH)
    en_audio = _require_sample(EN_SAMPLE_PATH)

    backend = NemotronBackend(language=None)  # auto language ID
    await backend.start()
    try:
        zh_events = await _stream_utterance(backend, zh_audio)
        zh_text = _assert_clean_final(zh_events)
        assert [e for e in zh_events if e.kind == "final"][0].language == "Chinese"

        en_events = await _stream_utterance(backend, en_audio)
        en_text = _assert_clean_final(en_events)
        assert [e for e in en_events if e.kind == "final"][0].language == "English"

        print(f"\n[nemotron auto zh] {zh_text!r}\n[nemotron auto en] {en_text!r}\n")
    finally:
        await backend.stop()


async def test_final_flush_without_trailing_silence():
    """The tail of an utterance that ends mid-chunk must survive finalize().

    `finalize()` zero-pads the sub-chunk remainder and decodes it with
    `keep_all_outputs=True` so the encoder does not truncate the trailing
    frames. Cut the fixture so the last 400 ms are strictly less than one
    560 ms chunk, push everything, and finalize immediately — no trailing
    silence to hide a lost tail.
    """
    from stt_server.backends.nemotron.backend import NemotronBackend

    audio = _require_sample(EN_SAMPLE_PATH)
    tail_bytes = 400 * 16 * 2  # 400 ms of PCM16 at 16 kHz, below one chunk
    assert len(audio) > tail_bytes

    backend = NemotronBackend()
    await backend.start()
    try:
        stream = await backend.create_stream(StreamConfig(language="en"))
        events: list = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        head = audio[: len(audio) - tail_bytes]
        for i in range(0, len(head), CHUNK_BYTES_100MS):
            await stream.push_audio(
                AudioChunk(data=head[i : i + CHUNK_BYTES_100MS], ingest_ts=0.0)
            )
        await stream.push_audio(AudioChunk(data=audio[len(audio) - tail_bytes :], ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=120.0)
        await stream.close()

        final_text = _assert_clean_final(events)
        partials = [e for e in events if e.kind == "partial"]
        last_partial = partials[-1].text.strip() if partials else ""
        assert len(final_text) >= len(last_partial), (
            "the final flush lost the utterance tail: "
            f"final={final_text!r} last_partial={last_partial!r}"
        )
        print(f"\n[nemotron final-flush] partial={last_partial!r} final={final_text!r}\n")
    finally:
        await backend.stop()


async def test_concurrent_streams_are_isolated():
    """Four interleaved streams, two zh + two en, each explicitly pinned.

    The language prompt is model-global and is re-applied under the decode
    lock immediately before every step, so interleaving must not leak one
    stream's language (or hypothesis) into another's.
    """
    from stt_server.backends.nemotron.backend import NemotronBackend

    zh_audio = _require_sample(ZH_SAMPLE_PATH)
    en_audio = _require_sample(EN_SAMPLE_PATH)

    backend = NemotronBackend()
    await backend.start()
    try:
        specs = [
            ("zh-CN", "Chinese", zh_audio),
            ("en-US", "English", en_audio),
            ("zh-CN", "Chinese", zh_audio),
            ("en-US", "English", en_audio),
        ]
        streams = [await backend.create_stream(StreamConfig(language=lang)) for lang, _, _ in specs]
        collected: list[list] = [[] for _ in streams]

        async def reader(index: int):
            async for ev in streams[index].events():
                collected[index].append(ev)

        tasks = [asyncio.create_task(reader(i)) for i in range(len(streams))]

        longest = max(len(audio) for _, _, audio in specs)
        for offset in range(0, longest, CHUNK_BYTES_100MS):
            for stream, (_lang, _name, audio) in zip(streams, specs, strict=True):
                piece = audio[offset : offset + CHUNK_BYTES_100MS]
                if piece:
                    await stream.push_audio(AudioChunk(data=piece, ingest_ts=0.0))
        for stream in streams:
            await stream.finalize()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=300.0)
        for stream in streams:
            await stream.close()

        for events, (_lang, expected_name, _audio) in zip(collected, specs, strict=True):
            text = _assert_clean_final(events)
            final = [e for e in events if e.kind == "final"][0]
            assert final.language == expected_name
            print(f"\n[nemotron concurrent {expected_name}] {text!r}")
    finally:
        await backend.stop()
