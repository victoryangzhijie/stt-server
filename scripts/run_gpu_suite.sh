#!/usr/bin/env bash
# CUDA-box GPU suite: real qwen3asr model tests, GPU + full-CPU Docker image
# builds, and real-backend accuracy/load benchmark runs. See
# benchmarks/cuda_runbook.md for prerequisites, expected artifacts/durations,
# and how to hand results back. Not runnable on a CPU-only dev machine (the
# whole point of this script).
#
# Each phase is independently skippable via an env flag so a partial re-run
# (e.g. after fixing a docker build) doesn't have to repeat everything:
#   SKIP_SYNC=1        skip phase 1 (uv sync)
#   SKIP_TESTS=1       skip phase 2 (qwen3asr model+gpu pytest)
#   SKIP_GPU_IMAGE=1   skip phase 3 (docker build, GPU image)
#   SKIP_CPU_IMAGE=1   skip phase 4 (docker build, full CPU image)
#   SKIP_ACCURACY=1    skip phase 5 (accuracy run vs qwen3asr)
#   SKIP_LOAD=1        skip phase 6 (load run vs qwen3asr)
#   SKIP_FUNASR_CHECK=1 skip phase 7 (funasr zero-length is_final=True check)
#
# Run from the repo root: `bash scripts/run_gpu_suite.sh`

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# On the CUDA box there is no aliyun-mirror problem (that's a constraint of
# THIS dev machine's environment, not the project) — UV_DEFAULT_INDEX does
# not need to be set here. It is kept as an OPTIONAL prefix pattern for
# parity with the session policy this repo was developed under: if your box
# also sits behind a mirror that doesn't carry every package/wheel this repo
# needs (e.g. GPU-specific wheels), run phase 1 manually with
# `UV_DEFAULT_INDEX="https://pypi.org/simple" uv sync ...` instead of relying
# on this script's plain `uv sync` call.

phase() {
    echo
    echo "==== [phase $1] $2 ===="
}

if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    # funasr included so phase 7's carried-item check actually runs (it
    # skips silently when the extra is absent).
    phase 1 "uv sync --extra qwen3asr --extra bench --extra funasr"
    uv sync --extra qwen3asr --extra bench --extra funasr
else
    phase 1 "SKIPPED (SKIP_SYNC=1)"
fi

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    phase 2 "pytest -m 'model and gpu' (qwen3asr model tests)"
    uv run pytest -m "model and gpu" tests/backends/test_qwen3asr_backend_model.py
else
    phase 2 "SKIPPED (SKIP_TESTS=1)"
fi

if [[ "${SKIP_GPU_IMAGE:-0}" != "1" ]]; then
    phase 3 "docker build GPU image (deploy/Dockerfile.gpu)"
    docker build -f deploy/Dockerfile.gpu -t stt-server:gpu .
else
    phase 3 "SKIPPED (SKIP_GPU_IMAGE=1)"
fi

if [[ "${SKIP_CPU_IMAGE:-0}" != "1" ]]; then
    phase 4 "docker build full CPU image (deploy/Dockerfile)"
    docker build -f deploy/Dockerfile -t stt-server:cpu .
else
    phase 4 "SKIPPED (SKIP_CPU_IMAGE=1)"
fi

if [[ "${SKIP_ACCURACY:-0}" != "1" ]]; then
    phase 5 "accuracy run vs qwen3asr (test-clean, n=100, seed 42, pace 1.0)"
    # --pace defaults to 1.0 (real-time) and must NOT be overridden to 0 here:
    # per the Task 7 methodology finding (benchmarks/README.md "Arm A validity
    # guard"), a real backend fed audio faster than real-time can trip the
    # server's default drop_oldest backpressure and silently shed audio,
    # invalidating the WER. run_accuracy's WS pass asserts
    # stt_audio_dropped_total stays at 0 by default (--assert-no-drops); do
    # not pass --no-assert-no-drops on this run.
    uv run python -m benchmarks.run_accuracy \
        --config configs/qwen3asr.yaml --model qwen3-asr-0.6b \
        --split test-clean --n 100 --seed 42
else
    phase 5 "SKIPPED (SKIP_ACCURACY=1)"
fi

if [[ "${SKIP_LOAD:-0}" != "1" ]]; then
    phase 6 "load run vs qwen3asr"
    uv run python -m benchmarks.run_load \
        --config configs/qwen3asr.yaml --model qwen3-asr-0.6b \
        --utterance-seconds 5 --start 2 --step 2 --max 16 \
        --window-seconds 30 --slo-final-ms 2000 --slo-pct 95 --seed 42 \
        --synthetic
else
    phase 6 "SKIPPED (SKIP_LOAD=1)"
fi

if [[ "${SKIP_FUNASR_CHECK:-0}" != "1" ]]; then
    phase 7 "funasr zero-length is_final=True real-model check (Plan 3 carried item)"
    # Carried item from Plan 3 (see .superpowers/sdd/progress.md): the unit
    # tests only ever feed FunasrStream a buffer SMALLER than one decode
    # stride, so finalize()'s "remaining = whatever didn't fill a whole
    # stride" path is exercised, but never the case where the buffer drains
    # to EXACTLY zero (an utterance whose length is an exact multiple of the
    # 600ms/19200-byte default stride) -- i.e. whether the real FunASR model
    # tolerates `generate(input=<empty array>, is_final=True, ...)`. This
    # phase verifies that against the real model. Independent of CUDA/GPU
    # (funasr runs on CPU) but grouped here since it needs a real model
    # download and this is the box where "run everything once" happens;
    # skips cleanly if the funasr extra isn't installed.
    uv run python - <<'PYEOF'
import asyncio
import importlib.util
import sys

if importlib.util.find_spec("funasr") is None:
    print("[skip] funasr extra not installed; skipping zero-length is_final=True check")
    sys.exit(0)

from stt_server.backends.base import StreamConfig
from stt_server.backends.funasr.backend import FunasrBackend
from stt_server.core.events import AudioChunk


async def main() -> None:
    backend = FunasrBackend(model="paraformer-zh-streaming")
    await backend.start()
    try:
        stream = await backend.create_stream(StreamConfig())
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())

        # Exactly one decode stride's worth of audio (chunk_size[1]=10 *
        # 960 samples/60ms-unit * 2 bytes/sample = 19200 bytes for the
        # default chunk_size=(0, 10, 5)), so push_audio's while-loop drains
        # the buffer to exactly zero and finalize() calls
        # generate(input=<empty array>, is_final=True, ...) on the real
        # model -- the exact case the Plan 3 item flagged as unverified.
        chunk_stride_bytes = 19200
        silence = b"\x00" * chunk_stride_bytes
        await stream.push_audio(AudioChunk(data=silence, ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=60.0)
        await stream.close()

        finals = [e for e in events if e.kind == "final"]
        assert len(finals) == 1, f"expected exactly one final event, got {events!r}"
        assert events[-1].kind == "final"
        print(f"[ok] funasr zero-length is_final=True check passed: {finals[0]!r}")
    finally:
        await backend.stop()


asyncio.run(main())
PYEOF
else
    phase 7 "SKIPPED (SKIP_FUNASR_CHECK=1)"
fi

echo
echo "==== GPU suite complete ===="
echo "Results (if phases 5/6 ran) are in benchmarks/results/ (gitignored)."
echo "See benchmarks/cuda_runbook.md for how to hand them back."
