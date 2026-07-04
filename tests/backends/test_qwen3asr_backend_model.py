"""Model+GPU-marked tests: run the real Qwen3-ASR backend (vLLM backend,
`Qwen/Qwen3-ASR-0.6B` by default) against a real speech fixture.

Requires the `qwen3asr` extra installed AND a CUDA GPU (the framework's
streaming interface — `init_streaming_state`/`streaming_transcribe`/
`finish_streaming_transcribe` — is documented as "vLLM backend only", and
the vLLM backend requires CUDA). Not runnable on this development machine
(no CUDA); written for and deferred to the user's CUDA box, per Plan 4.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
from pathlib import Path

import pytest

from stt_server.backends.base import StreamConfig
from stt_server.core.events import AudioChunk

from .conformance import BackendConformanceSuite

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPEECH_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"

QWEN_ASR_AVAILABLE = importlib.util.find_spec("qwen_asr") is not None

# CUDA-availability gate (M-6): the vLLM backend requires a CUDA GPU, and
# initializing it on a CUDA-less box FAILS (vLLM init error) rather than
# skipping cleanly -- even when the `qwen3asr` extra IS installed (e.g. a
# CPU-only CI runner that has the package but no GPU). Detect via
# `shutil.which("nvidia-smi")` rather than `torch.cuda.is_available()`:
# checking for the CLI binary on PATH is a cheap, allocation-free stdlib
# call that works at collection time even when the (heavy, torch-pulling)
# `qwen3asr` extra isn't installed at all -- importing torch just to decide
# whether to skip would defeat the point of gating cheaply.
CUDA_AVAILABLE = shutil.which("nvidia-smi") is not None

pytestmark = [
    pytest.mark.model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not QWEN_ASR_AVAILABLE,
        reason="qwen_asr not installed; pip install 'stt-server[qwen3asr]'",
    ),
    pytest.mark.skipif(
        not CUDA_AVAILABLE,
        reason=(
            "no CUDA GPU detected (nvidia-smi not on PATH); the vLLM backend "
            "requires CUDA and fails to initialize (rather than skipping) on "
            "a CPU-only box even with the qwen3asr extra installed -- see "
            "benchmarks/cuda_runbook.md"
        ),
    ),
]


def _speech_fixture() -> bytes:
    return SPEECH_FIXTURE_PATH.read_bytes()


class TestQwen3AsrConformance(BackendConformanceSuite):
    @pytest.fixture(scope="module")
    def backend(self):
        from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend

        return Qwen3AsrBackend(model="Qwen/Qwen3-ASR-0.6B")

    async def _run_utterance(self, backend, audio: bytes) -> list:
        return await super()._run_utterance(backend, _speech_fixture())


async def test_real_model_streams_incremental_partials_and_one_final():
    from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend

    backend = Qwen3AsrBackend(model="Qwen/Qwen3-ASR-0.6B")
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
        print(f"\n[qwen3asr real transcript, en fixture] {final_text!r}\n")
    finally:
        await backend.stop()
