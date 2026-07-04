"""Tests for benchmarks/client_ws.py, client_file.py, server.py, results.py.

`test_stream_utterance_against_mock_backend` boots a real `stt-server`
subprocess (mock config, ephemeral port) via `ServerUnderTest` — this is the
one test in the suite that needs a live socket (websockets can't be driven
through Starlette's `TestClient`). It stays unmarked because it completes in
well under 10s on the mock backend."""

from __future__ import annotations

import os
import socket
import wave
from io import BytesIO
from pathlib import Path

import pytest
from benchmarks.client_file import wrap_wav
from benchmarks.client_ws import UtteranceResult, pacing_delay, stream_utterance
from benchmarks.results import markdown_table, percentiles, write_result
from benchmarks.server import ServerUnderTest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MOCK_CONFIG = str(REPO_ROOT / "configs" / "mock.yaml")
SPEECH_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_wrap_wav_roundtrips_via_stdlib_wave() -> None:
    pcm16 = b"\x01\x00\x02\x00\x03\x00\x04\x00"

    wav_bytes = wrap_wav(pcm16, sample_rate=16000)

    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == pcm16


def test_percentiles_math() -> None:
    result = percentiles([float(x) for x in range(1, 101)])  # 1..100

    assert result["n"] == 100
    assert result["p50"] == 50
    assert result["p95"] == 95
    assert result["p99"] == 99
    assert result["mean"] == pytest.approx(50.5)


def test_percentiles_empty() -> None:
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}


def test_percentiles_single_value() -> None:
    result = percentiles([42.0])
    assert result["p50"] == result["p95"] == result["p99"] == 42.0
    assert result["n"] == 1


def test_markdown_table_basic() -> None:
    rows = [{"name": "a", "wer": 0.1}, {"name": "b", "wer": 0.2}]

    table = markdown_table(rows, columns=["name", "wer"])

    lines = table.splitlines()
    assert lines[0] == "| name | wer |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| a | 0.1 |"
    assert lines[3] == "| b | 0.2 |"


def test_write_result_includes_meta_block(tmp_path: Path) -> None:
    out_path = write_result("smoke", {"n": 3}, results_dir=tmp_path)

    assert out_path.exists()
    assert out_path.parent == tmp_path
    assert out_path.name.startswith("smoke-")
    assert out_path.suffix == ".json"

    import json

    payload = json.loads(out_path.read_text())
    assert payload["n"] == 3
    assert "meta" in payload
    assert "git_sha" in payload["meta"]
    assert "platform" in payload["meta"]
    assert "python" in payload["meta"]


def test_write_result_preserves_caller_supplied_meta(tmp_path: Path) -> None:
    out_path = write_result("smoke", {"meta": {"seed": 42, "n": 5}}, results_dir=tmp_path)

    import json

    payload = json.loads(out_path.read_text())
    assert payload["meta"]["seed"] == 42
    assert payload["meta"]["n"] == 5
    assert "git_sha" in payload["meta"]


def _write_never_ready_python(tmp_path: Path, extra_lines: str = "") -> Path:
    """A fake `python` for ServerUnderTest: ignores its `-m stt_server ...`
    args, optionally runs `extra_lines`, then sleeps forever (never serves
    /readyz). `exec` keeps it a single process so SIGTERM lands directly."""
    script = tmp_path / "fake-python"
    script.write_text(f"#!/bin/sh\n{extra_lines}\nexec sleep 60\n")
    script.chmod(0o755)
    return script


def _assert_process_dead(pid: int) -> None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return  # dead and reaped
    pytest.fail(f"child process {pid} is still alive after teardown")


def test_server_under_test_kills_child_on_readiness_timeout(tmp_path: Path) -> None:
    """Regression: __enter__ raising out of _wait_ready must not leak the
    spawned subprocess (a raised __enter__ means __exit__ never runs)."""
    fake_python = _write_never_ready_python(tmp_path)
    sut = ServerUnderTest(
        MOCK_CONFIG, port=_free_port(), python=str(fake_python), ready_timeout_s=1.5
    )

    with pytest.raises(TimeoutError):
        sut.__enter__()

    assert sut.pid is not None
    _assert_process_dead(sut.pid)


def test_server_under_test_raises_and_reaps_on_early_exit(tmp_path: Path) -> None:
    """A child that exits before becoming ready raises RuntimeError (not a
    120s hang) and leaves no live process behind."""
    script = tmp_path / "fake-python"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(0o755)
    sut = ServerUnderTest(
        MOCK_CONFIG, port=_free_port(), python=str(script), ready_timeout_s=5.0
    )

    with pytest.raises(RuntimeError, match="exited early"):
        sut.__enter__()

    assert sut.pid is not None
    _assert_process_dead(sut.pid)


def test_server_under_test_merges_env_and_captures_log(tmp_path: Path) -> None:
    """`env=` must MERGE over os.environ (not replace it — a replaced env
    drops PATH etc.), and child stdout must land in `.log_path`."""
    fake_python = _write_never_ready_python(
        tmp_path, extra_lines='echo "MARKER=$MARKER"\necho "SAW_PATH=$PATH"'
    )
    sut = ServerUnderTest(
        MOCK_CONFIG,
        port=_free_port(),
        python=str(fake_python),
        env={"MARKER": "bench-marker-1"},
        ready_timeout_s=1.5,
    )

    with pytest.raises(TimeoutError):
        sut.__enter__()

    assert sut.log_path is not None and sut.log_path.exists()
    log = sut.log_path.read_text()
    assert "MARKER=bench-marker-1" in log  # explicit env var reached the child
    assert "SAW_PATH=" in log and "SAW_PATH=\n" not in log  # inherited PATH survived
    assert sut.pid is not None
    _assert_process_dead(sut.pid)


def test_pacing_delay_uses_cumulative_deadlines() -> None:
    """Deadline arithmetic: chunk i's send time is t0 + i*interval/pace, so
    a late chunk never pushes later chunks' deadlines back (no drift)."""
    t0 = 100.0
    # On schedule: chunk 3 at pace 1.0 with 100ms chunks is due at t0+0.3.
    assert pacing_delay(t0, 3, 100, 1.0, now=100.25) == pytest.approx(0.05)
    # Running late: no negative sleep, send immediately...
    assert pacing_delay(t0, 3, 100, 1.0, now=100.5) == 0.0
    # ...and the NEXT chunk's deadline is still anchored to t0 (catch-up,
    # not fixed-interval drift): chunk 5 due at t0+0.5 even after lateness.
    assert pacing_delay(t0, 5, 100, 1.0, now=100.5) == 0.0
    assert pacing_delay(t0, 6, 100, 1.0, now=100.5) == pytest.approx(0.1)
    # pace=2.0 halves the interval; pace=0 disables pacing entirely.
    assert pacing_delay(t0, 4, 100, 2.0, now=100.0) == pytest.approx(0.2)
    assert pacing_delay(t0, 1000, 100, 0.0, now=100.0) == 0.0


def test_stream_utterance_against_mock_backend() -> None:
    pcm16 = SPEECH_FIXTURE.read_bytes()
    port = _free_port()

    with ServerUnderTest(MOCK_CONFIG, port=port) as server:
        result = _run_stream(server.base_ws_url, pcm16)

    assert isinstance(result, UtteranceResult)
    assert result.hypothesis  # mock backend always emits scripted text
    assert result.client_final_ms >= 0
    assert len(result.partials) > 0
    for audio_time_ms, stable_text, volatile_text in result.partials:
        assert isinstance(audio_time_ms, float)
        assert isinstance(stable_text, str)
        assert isinstance(volatile_text, str)


def _run_stream(base_ws_url: str, pcm16: bytes) -> UtteranceResult:
    import asyncio

    return asyncio.run(
        stream_utterance(base_ws_url, "mock", pcm16, chunk_ms=100, pace=0)
    )
