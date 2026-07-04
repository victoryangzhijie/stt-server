"""Unit tests for `benchmarks/run_load.py` decision logic and
`benchmarks/sampling.py` -- no server subprocess, no network (except the
`ResourceSampler` test, which samples the current live test process)."""

from __future__ import annotations

import time

import pytest
from benchmarks._drops import parse_prometheus_metrics
from benchmarks.run_load import (
    check_capacity,
    rung_passes,
)

from stt_server.config.settings import LimitsConfig, Settings

# --------------------------------------------------------------------------
# rung_passes truth table
# --------------------------------------------------------------------------


def test_rung_passes_when_under_slo_and_no_errors() -> None:
    latencies = [float(x) for x in range(1, 101)]  # p95 = 95

    assert rung_passes(latencies, slo_ms=95.0, pct=95, errors=0) is True


def test_rung_fails_when_over_slo() -> None:
    latencies = [float(x) for x in range(1, 101)]  # p95 = 95

    assert rung_passes(latencies, slo_ms=94.0, pct=95, errors=0) is False


def test_rung_fails_on_any_errors_even_if_latency_ok() -> None:
    latencies = [10.0, 20.0, 30.0]

    assert rung_passes(latencies, slo_ms=1000.0, pct=95, errors=1) is False


def test_rung_fails_on_empty_latencies_with_zero_errors() -> None:
    # Defined as FAIL, not a vacuous pass -- see rung_passes' docstring: no
    # measurements means the rung proved nothing about the SLO.
    assert rung_passes([], slo_ms=1000.0, pct=95, errors=0) is False


def test_rung_passes_exactly_at_slo_boundary() -> None:
    # <=, not <: a latency exactly equal to the SLO threshold passes.
    assert rung_passes([100.0], slo_ms=100.0, pct=95, errors=0) is True
    assert rung_passes([100.0001], slo_ms=100.0, pct=95, errors=0) is False


def test_rung_passes_uses_requested_percentile_not_p95_always() -> None:
    latencies = [float(x) for x in range(1, 101)]  # p50=50, p99=99

    assert rung_passes(latencies, slo_ms=50.0, pct=50, errors=0) is True
    assert rung_passes(latencies, slo_ms=98.0, pct=99, errors=0) is False
    assert rung_passes(latencies, slo_ms=99.0, pct=99, errors=0) is True


def test_rung_single_sample() -> None:
    assert rung_passes([500.0], slo_ms=500.0, pct=95, errors=0) is True
    assert rung_passes([500.0], slo_ms=499.0, pct=95, errors=0) is False


# --------------------------------------------------------------------------
# Prometheus text-format parser
# --------------------------------------------------------------------------


def test_parse_prometheus_metrics_sums_across_label_combinations() -> None:
    text = """
# HELP stt_errors_total Total session errors, labeled by error code.
# TYPE stt_errors_total counter
stt_errors_total{code="internal"} 3.0
stt_errors_total{code="timeout"} 2.0
# HELP stt_sessions_active Number of concurrently active STT sessions.
# TYPE stt_sessions_active gauge
stt_sessions_active 4.0
"""
    result = parse_prometheus_metrics(text)

    assert result["stt_errors_total"] == pytest.approx(5.0)
    assert result["stt_sessions_active"] == pytest.approx(4.0)


def test_parse_prometheus_metrics_handles_no_labels() -> None:
    text = "stt_rejections_total 0.0\n"

    result = parse_prometheus_metrics(text)

    assert result["stt_rejections_total"] == pytest.approx(0.0)


def test_parse_prometheus_metrics_ignores_comments_and_blank_lines() -> None:
    text = "\n# just a comment\n\nstt_audio_dropped_total{backend=\"mock\"} 7\n\n"

    result = parse_prometheus_metrics(text)

    assert result == {"stt_audio_dropped_total": 7.0}


def test_parse_prometheus_metrics_empty_text() -> None:
    assert parse_prometheus_metrics("") == {}


def test_parse_prometheus_metrics_skips_unparseable_value() -> None:
    text = "some_weird_line_without_a_number\nstt_errors_total 1.0\n"

    result = parse_prometheus_metrics(text)

    assert result == {"stt_errors_total": 1.0}


def test_parse_prometheus_metrics_handles_trailing_timestamp() -> None:
    # The exposition format allows `name{labels} value timestamp_ms`: the
    # VALUE is the first token after the name/labels, never the last.
    text = (
        'stt_errors_total{code="internal"} 3.0 1712345678000\n'
        "stt_sessions_active 4.0 1712345678000\n"
    )

    result = parse_prometheus_metrics(text)

    assert result["stt_errors_total"] == pytest.approx(3.0)
    assert result["stt_sessions_active"] == pytest.approx(4.0)


def test_parse_prometheus_metrics_labels_with_spaces_in_values() -> None:
    # Quoted label values may contain spaces; the value token must still be
    # found after the closing brace, not by naive whitespace splitting.
    text = 'stt_errors_total{code="two words here"} 2.0\n'

    result = parse_prometheus_metrics(text)

    assert result == {"stt_errors_total": 2.0}


# --------------------------------------------------------------------------
# worker loop: per-utterance watchdog timeout + empty-hypothesis handling
# --------------------------------------------------------------------------


async def test_worker_loop_times_out_wedged_stream_and_counts_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream_utterance that never resolves (server never sends
    session.closed) must be cut off by the watchdog and counted as an
    error -- not hang the worker (and the whole gathered rung) forever."""
    import asyncio

    from benchmarks import run_load

    async def never_resolves(*args, **kwargs):
        await asyncio.Event().wait()  # blocks forever

    monkeypatch.setattr(run_load, "stream_utterance", never_resolves)
    monkeypatch.setattr(run_load, "STREAM_TIMEOUT_MARGIN_S", 0.05)

    latencies: list[float] = []
    errors = [0]
    deadline = time.monotonic() + 0.2
    # 4 bytes of PCM16 -> audio_s ~ 0, so the timeout is ~ the margin.
    await asyncio.wait_for(
        run_load._worker_loop(
            "ws://unused", "mock", [b"\x00\x00\x01\x00"], 0, deadline, latencies, errors
        ),
        timeout=5.0,  # the test's own guard: the loop must terminate
    )

    assert errors[0] >= 1
    assert latencies == []


async def test_worker_loop_counts_whitespace_only_hypothesis_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple empty finals join to a truthy ' ' -- whitespace-only must
    still count as no-transcript (error), contributing no latency sample."""
    from benchmarks.client_ws import UtteranceResult

    from benchmarks import run_load

    async def whitespace_result(*args, **kwargs):
        return UtteranceResult(
            utt_id="",
            hypothesis=" ",  # two empty finals joined with " "
            server_final_ms=None,
            server_first_partial_ms=None,
            client_final_ms=1.0,
        )

    monkeypatch.setattr(run_load, "stream_utterance", whitespace_result)

    latencies: list[float] = []
    errors = [0]
    deadline = time.monotonic() + 0.05
    await run_load._worker_loop(
        "ws://unused", "mock", [b"\x00\x00"], 0, deadline, latencies, errors
    )

    assert errors[0] >= 1
    assert latencies == []


# --------------------------------------------------------------------------
# capacity sanity check
# --------------------------------------------------------------------------


def test_check_capacity_passes_when_cap_covers_max() -> None:
    settings = Settings(limits=LimitsConfig(max_sessions=64))

    check_capacity(settings, max_concurrency=64)  # no raise


def test_check_capacity_raises_when_max_exceeds_cap() -> None:
    settings = Settings(limits=LimitsConfig(max_sessions=10))

    with pytest.raises(SystemExit, match="max_sessions=10"):
        check_capacity(settings, max_concurrency=64)


# --------------------------------------------------------------------------
# ResourceSampler against the current test process
# --------------------------------------------------------------------------


def test_resource_sampler_start_stop_against_current_process() -> None:
    pytest.importorskip("psutil")
    import os

    from benchmarks.sampling import ResourceSampler

    sampler = ResourceSampler(os.getpid(), interval_s=0.1)
    sampler.start()
    # Burn a little CPU so at least one sample tick has something to see.
    deadline = time.monotonic() + 0.3
    total = 0
    while time.monotonic() < deadline:
        total += 1
    result = sampler.stop()

    assert "samples" in result
    assert result["rss_mb_peak"] >= 0.0
    assert result["cpu_pct_peak"] >= 0.0
    assert isinstance(result["samples"], list)


def test_resource_sampler_handles_dead_process_gracefully() -> None:
    pytest.importorskip("psutil")
    import subprocess

    from benchmarks.sampling import ResourceSampler

    proc = subprocess.Popen(["true"])
    proc.wait()

    sampler = ResourceSampler(proc.pid, interval_s=0.05)
    # Constructing/starting against an already-dead pid must not raise.
    sampler.start()
    time.sleep(0.2)
    result = sampler.stop()

    assert result["samples"] == []
    assert result["cpu_pct_peak"] == 0.0
    assert result["rss_mb_peak"] == 0.0
