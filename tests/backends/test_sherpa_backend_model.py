"""Model-marked tests: run the real sherpa-onnx streaming Zipformer backend
against real model weights (see scripts/download_models.py sherpa-zipformer-en)
and a real speech fixture. Requires the `sherpa` extra installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from stt_server.backends.base import StreamConfig
from stt_server.backends.sherpa.backend import SherpaBackend
from stt_server.core.events import AudioChunk

from .conformance import BackendConformanceSuite

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
SPEECH_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason=(
            "sherpa model not downloaded; run "
            "`python scripts/download_models.py sherpa-zipformer-en`"
        ),
    ),
]


def _speech_fixture() -> bytes:
    return SPEECH_FIXTURE_PATH.read_bytes()


class TestSherpaConformance(BackendConformanceSuite):
    @pytest.fixture(scope="module")
    def backend(self):
        return SherpaBackend(model_dir=str(MODEL_DIR))

    async def _run_utterance(self, backend, audio: bytes) -> list:
        # The base suite's ONE_SECOND_PCM is silence-adjacent synthetic
        # noise, not real speech; a real ASR model legitimately returns an
        # empty transcript for it, which would fail the base suite's
        # "final text is non-empty" assertion. Substitute a real speech
        # fixture so the conformance checks (ordering, exactly-one-final,
        # no-op-after-finalize, ...) are exercised against a real,
        # non-trivial transcription instead.
        return await super()._run_utterance(backend, _speech_fixture())


async def test_real_transcript_is_nonempty_and_reasonable():
    backend = SherpaBackend(model_dir=str(MODEL_DIR))
    await backend.start()
    try:
        stream = await backend.create_stream(StreamConfig())
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())

        audio = _speech_fixture()
        # push in small chunks to exercise true incremental streaming
        chunk_bytes = 1600  # 50 ms at 16kHz/16-bit mono
        for i in range(0, len(audio), chunk_bytes):
            await stream.push_audio(AudioChunk(data=audio[i : i + chunk_bytes], ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=30.0)
        await stream.close()

        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1
        final_text = finals[0].text.strip()
        assert final_text, "expected a non-empty transcript from real speech audio"
        print(f"\n[sherpa real transcript] {final_text!r}\n")
    finally:
        await backend.stop()
