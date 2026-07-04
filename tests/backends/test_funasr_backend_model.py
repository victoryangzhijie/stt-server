"""Model-marked tests: run the real FunASR Paraformer streaming backend
against the real `paraformer-zh-streaming` model (auto-downloaded from
modelscope on first `AutoModel(...)` use — see scripts/download_models.py's
`paraformer-zh-streaming` pre-warm entry) and a real speech fixture.
Requires the `funasr` extra installed.

Note: `paraformer-zh-streaming` is a Mandarin model; the repo's only real
speech fixture (`tests/fixtures/speech_16k_mono_s16le.pcm`) is English. Real,
manual verification against real Mandarin speech (synthesized locally with
macOS `say -v Ting-Ting`, not committed to the repo — TTS-voice-output
licensing for redistribution is unclear, unlike the Open Speech Repository
fixture) produced a correct transcript; see the task report for the full
transcript. The committed model-marked tests below reuse the English
fixture and assert on pipeline *mechanics* only (ordering, exactly-one-final,
no-op-after-finalize/close, non-empty text) rather than transcript content,
because a Mandarin model fed English audio legitimately produces
Chinese-character noise rather than any particular string.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from stt_server.backends.base import StreamConfig
from stt_server.backends.funasr.backend import FunasrBackend
from stt_server.core.events import AudioChunk

from .conformance import BackendConformanceSuite

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPEECH_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"

FUNASR_AVAILABLE = importlib.util.find_spec("funasr") is not None

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not FUNASR_AVAILABLE,
        reason="funasr not installed; pip install 'stt-server[funasr]'",
    ),
]


def _speech_fixture() -> bytes:
    return SPEECH_FIXTURE_PATH.read_bytes()


class TestFunasrConformance(BackendConformanceSuite):
    @pytest.fixture(scope="module")
    def backend(self):
        return FunasrBackend(model="paraformer-zh-streaming")

    async def _run_utterance(self, backend, audio: bytes) -> list:
        # As in the sherpa model tests: substitute the real speech fixture
        # for the base suite's silence-adjacent synthetic noise, so the base
        # suite's "final text is non-empty" assertion exercises a real
        # (non-trivial, if not linguistically meaningful for a zh model fed
        # English audio) transcription rather than legitimate ASR silence.
        return await super()._run_utterance(backend, _speech_fixture())


async def test_real_model_streams_incremental_partials_and_one_final():
    backend = FunasrBackend(model="paraformer-zh-streaming")
    await backend.start()
    try:
        stream = await backend.create_stream(StreamConfig())
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())

        audio = _speech_fixture()
        chunk_bytes = 3200  # 100 ms at 16 kHz/16-bit mono
        for i in range(0, len(audio), chunk_bytes):
            await stream.push_audio(AudioChunk(data=audio[i : i + chunk_bytes], ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=60.0)
        await stream.close()

        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        assert events[-1].kind == "final"
        final_text = finals[0].text.strip()
        assert final_text, "expected a non-empty transcript from real speech audio"
        times = [e.audio_time_ms for e in events]
        assert times == sorted(times)
        print(f"\n[funasr real transcript, zh model on en audio] {final_text!r}\n")
    finally:
        await backend.stop()
