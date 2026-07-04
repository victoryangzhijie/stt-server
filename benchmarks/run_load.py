"""Concurrency-ramp load generator with SLO gating (spec §10.3).

Ramps concurrent WS streaming workers from `--start` in steps of `--step`
up to `--max`. At each "rung" (a fixed concurrency level held for
`--window-seconds`), every worker loops utterances back-to-back over a real
`stt-server` subprocess (`benchmarks.server.ServerUnderTest`), and the run
collects:

- client-observed final-transcript latency (`client_final_ms` from
  `benchmarks.client_ws.stream_utterance`) across all utterances completed
  in the window;
- `/metrics` deltas (before vs. after the window) for
  `stt_audio_dropped_total` (per-session backpressure shedding, spec §13),
  `stt_rejections_total` (capacity/pre-parse rejections), and
  `stt_errors_total` (session errors);
- CPU%/RSS(/GPU) resource samples via `benchmarks.sampling.ResourceSampler`.

A rung PASSES iff `p{slo-pct}(client_final_ms) <= slo-final-ms` AND there
were zero backpressure/error events (utterance-level errors + the three
metric deltas above, all summed into one count -- see `rung_passes`). The
ramp stops at the first failing rung (or at `--max`), and
`max_passing_concurrency` is the highest N that passed.

Usage::

    uv run python -m benchmarks.run_load \\
        --config configs/mock.yaml --model mock \\
        --utterance-seconds 10 --start 2 --step 2 --max 64 \\
        --slo-final-ms 1200 --slo-pct 95 --seed 42 \\
        [--python /path/venv/bin/python] [--port 8100] \\
        [--synthetic] [--window-seconds 30]

Audio source: with `--synthetic`, the repo's 1s real-speech fixture
(`tests/fixtures/speech_16k_mono_s16le.pcm`) is tiled to `--utterance-seconds`
and every worker replays that single buffer in a loop -- this is the mode
used for the mock backend and CI smoke runs, where LibriSpeech audio would
add nothing (the mock backend ignores input audio content) but a network
download would add minutes. Without `--synthetic`, a small LibriSpeech
`test-clean` manifest (downloaded on demand via `benchmarks.corpus`) is used
instead so real backends are exercised against real speech; each worker
cycles through the manifest's utterances (offset by worker index so workers
don't all send byte-identical audio in lockstep).

Capacity-limit note (see `check_capacity`): the server's own
`limits.max_sessions` (spec §13, `LimitsConfig` in
`src/stt_server/config/settings.py`, default 100) will reject connections
past that cap with a capacity rejection *before* the ramp's own SLO logic
ever sees the utterance -- indistinguishable, from the ramp's perspective,
from the backend actually falling over. Rather than silently absorb those
as ordinary rung failures (which would conflate "the config's session cap is
too low for this experiment" with "the backend can't keep up"), this runner
fails fast at startup if `--config`'s `limits.max_sessions < --max`, with a
message pointing at the `STT__LIMITS__MAX_SESSIONS` env override -- a load
run should deliberately size the cap to the ramp it's about to run, not
discover the mismatch 40 rungs in.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import math
import sys
import time
from pathlib import Path

from benchmarks._drops import fetch_metrics as _fetch_metrics
from benchmarks.client_ws import stream_utterance
from benchmarks.corpus import build_manifest, download_subset, load_pcm16
from benchmarks.results import percentiles, write_result
from benchmarks.sampling import ResourceSampler
from benchmarks.server import ServerUnderTest
from stt_server.config.settings import Settings, load_settings

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEECH_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"

# Metric families this runner cares about (spec §13 shed policy / §7
# metrics) -- deltas of these three, summed, count toward a rung's
# "backpressure/ERROR events" total alongside client-observed errors.
_TRACKED_METRICS = ("stt_audio_dropped_total", "stt_rejections_total", "stt_errors_total")

# Per-utterance watchdog margin: an utterance streamed at pace=1.0 takes its
# own audio duration on the wire, then the server needs some grace to
# endpoint + finalize. A server that never sends `session.closed` (wedged
# session -- exactly the failure mode a load test exists to surface) would
# otherwise hang its worker's `await stream_utterance(...)` forever, wedging
# the gather and the whole run. Anything slower than audio-duration + this
# margin is treated as an error, not waited out.
STREAM_TIMEOUT_MARGIN_S = 30.0


# parse_prometheus_metrics/_fetch_metrics moved to benchmarks._drops when
# the zero-drops guard was factored out for all runners; run_load imports
# fetch_metrics from there (aliased to its old private name above).


def _percentile(xs: list[float], pct: int) -> float:
    """Nearest-rank percentile for an arbitrary `pct` (0-100), matching the
    method `benchmarks.results.percentiles` uses for its fixed p50/95/99."""
    ordered = sorted(xs)
    n = len(ordered)
    rank = math.ceil(pct / 100 * n)
    idx = max(0, min(n - 1, rank - 1))
    return ordered[idx]


def rung_passes(latencies: list[float], slo_ms: float, pct: int, errors: int) -> bool:
    """Pure pass/fail decision for one concurrency rung.

    PASSES iff there were zero backpressure/ERROR events (`errors == 0`,
    a count the caller pre-sums from client-side errors + `/metrics`
    deltas) AND `p{pct}(latencies) <= slo_ms`.

    `latencies == [] and errors == 0` is defined as FAIL, not a vacuous
    pass: no completed measurements in the window means the rung proved
    NOTHING about whether the SLO holds at that concurrency (e.g. every
    worker's utterance was still in flight when the window closed), and a
    load gate that passes on missing data is worse than useless -- it would
    report a concurrency level as safe when it was never actually
    exercised.
    """
    if errors > 0:
        return False
    if not latencies:
        return False
    return _percentile(latencies, pct) <= slo_ms


def check_capacity(settings: Settings, max_concurrency: int) -> None:
    """Fail fast if the config's session cap can't hold `max_concurrency`
    concurrent sessions -- see the capacity-limit note in this module's
    docstring. Raises `SystemExit` (a config problem, not a bug) rather than
    letting the ramp run into a wall of indistinguishable-from-overload 429
    capacity rejections."""
    if settings.limits.max_sessions < max_concurrency:
        raise SystemExit(
            f"--max={max_concurrency} exceeds this config's "
            f"limits.max_sessions={settings.limits.max_sessions}; rungs beyond "
            "the cap would be rejected before the backend ever sees them, "
            "polluting the ramp. Raise the cap (e.g. "
            f"`STT__LIMITS__MAX_SESSIONS={max_concurrency}` or higher, or "
            "edit the config's `limits.max_sessions`) and re-run."
        )


def _build_audio_pool(args: argparse.Namespace) -> list[bytes]:
    """The pool of PCM16 audio buffers workers cycle through. `--synthetic`
    tiles the repo's speech fixture to `--utterance-seconds` and returns a
    single-buffer pool (every worker replays the same audio); otherwise a
    small seeded LibriSpeech `test-clean` manifest is downloaded (if not
    already cached) and each buffer decoded to 16 kHz mono PCM16."""
    if args.synthetic:
        base = SPEECH_FIXTURE.read_bytes()
        target_bytes = int(args.utterance_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE)
        target_bytes = max(target_bytes, len(base))
        reps = target_bytes // len(base) + 1
        return [(base * reps)[:target_bytes]]

    # Real-backend path: enough distinct utterances that workers aren't all
    # sending byte-identical audio, but small enough not to demand the full
    # ~350MB test-clean download for a short load run.
    n = min(50, max(args.max, 5))
    split_dir = download_subset("test-clean")
    manifest = build_manifest(split_dir, n=n, seed=args.seed)
    return [load_pcm16(utt) for utt in manifest]


async def _worker_loop(
    base_ws_url: str,
    model: str,
    audio_pool: list[bytes],
    start_index: int,
    deadline: float,
    latencies: list[float],
    error_counter: list[int],
) -> None:
    """Stream utterances from `audio_pool` back-to-back (cycling, offset by
    `start_index` so concurrent workers desync rather than all starting on
    the same buffer) until `deadline` (a `time.monotonic()` timestamp).
    `latencies`/`error_counter` are shared mutable accumulators -- safe
    without locking because everything here runs on one event loop thread,
    interleaved only at `await` points."""
    pool_cycle = itertools.cycle(audio_pool)
    for _ in range(start_index % len(audio_pool)):
        next(pool_cycle)

    while time.monotonic() < deadline:
        pcm16 = next(pool_cycle)
        audio_s = len(pcm16) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        try:
            # Watchdog timeout (see STREAM_TIMEOUT_MARGIN_S): a wedged
            # session that never closes must count as an error, not hang
            # the worker (and hence the whole gathered rung) forever.
            # asyncio.TimeoutError flows into the same except-as-error path
            # as connection/protocol failures.
            result = await asyncio.wait_for(
                stream_utterance(base_ws_url, model, pcm16, pace=1.0),
                timeout=audio_s + STREAM_TIMEOUT_MARGIN_S,
            )
        except Exception:
            error_counter[0] += 1
            continue
        # .strip(): a multi-segment utterance whose finals are ALL empty
        # joins to a truthy " " -- whitespace-only is still "no transcript".
        if not result.hypothesis.strip():
            # No final transcript arrived -- consistent with run_accuracy's
            # treatment, this is an error and contributes NOTHING to the
            # latency population (not a fake 0.0 sample).
            error_counter[0] += 1
            continue
        latencies.append(result.client_final_ms)


async def _run_rung(
    base_ws_url: str, model: str, audio_pool: list[bytes], n_workers: int, window_s: float
) -> tuple[list[float], int]:
    deadline = time.monotonic() + window_s
    latencies: list[float] = []
    error_counter = [0]
    await asyncio.gather(
        *(
            _worker_loop(base_ws_url, model, audio_pool, i, deadline, latencies, error_counter)
            for i in range(n_workers)
        )
    )
    return latencies, error_counter[0]


def run(args: argparse.Namespace) -> dict:
    settings = load_settings(args.config)
    check_capacity(settings, args.max)

    audio_pool = _build_audio_pool(args)

    payload: dict = {
        "model": args.model,
        "config": args.config,
        "utterance_seconds": args.utterance_seconds,
        "start": args.start,
        "step": args.step,
        "max": args.max,
        "slo_final_ms": args.slo_final_ms,
        "slo_pct": args.slo_pct,
        "window_seconds": args.window_seconds,
        "seed": args.seed,
        "synthetic": args.synthetic,
        "rungs": [],
        "max_passing_concurrency": 0,
    }

    with ServerUnderTest(args.config, port=args.port, python=args.python) as server:
        n = args.start
        while n <= args.max:
            sampler = ResourceSampler(server.pid)
            sampler.start()
            metrics_before = _fetch_metrics(server.base_url)

            latencies, client_errors = asyncio.run(
                _run_rung(server.base_ws_url, args.model, audio_pool, n, args.window_seconds)
            )

            metrics_after = _fetch_metrics(server.base_url)
            resources = sampler.stop()

            deltas = {
                name: metrics_after.get(name, 0.0) - metrics_before.get(name, 0.0)
                for name in _TRACKED_METRICS
            }
            # Declared reliance: RECOVERABLE server ERROR events are
            # invisible to the client (client_ws's receiver only terminates
            # on `recoverable: false`; a recoverable error frame is silently
            # skipped), so detecting them leans entirely on the
            # stt_errors_total metrics delta -- client_errors alone would
            # under-count. The flip side is a potential double-count: a
            # fatal server error typically yields BOTH an empty-hypothesis
            # client error AND an stt_errors_total increment. That's purely
            # cosmetic for the gate -- rung_passes is zero-tolerance
            # (errors > 0 fails), so counting one event twice can never
            # flip a passing rung to failing or vice versa.
            total_errors = (
                client_errors
                + int(deltas["stt_audio_dropped_total"])
                + int(deltas["stt_rejections_total"])
                + int(deltas["stt_errors_total"])
            )
            passed = rung_passes(latencies, args.slo_final_ms, args.slo_pct, total_errors)

            row: dict = {
                "concurrency": n,
                "latency_ms": percentiles(latencies),
                "client_errors": client_errors,
                "dropped_chunks": deltas["stt_audio_dropped_total"],
                "rejections": deltas["stt_rejections_total"],
                "server_errors": deltas["stt_errors_total"],
                "passed": passed,
                "cpu_pct_peak": resources["cpu_pct_peak"],
                "rss_mb_peak": resources["rss_mb_peak"],
            }
            if "gpu_util_pct_peak" in resources:
                row["gpu_util_pct_peak"] = resources["gpu_util_pct_peak"]
                row["gpu_mem_mb_peak"] = resources["gpu_mem_mb_peak"]

            payload["rungs"].append(row)

            if not passed:
                break
            payload["max_passing_concurrency"] = n
            n += args.step

    out_path = write_result(f"load-{args.model}", payload)
    payload["_out_path"] = str(out_path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_load",
        description=(
            "Concurrency-ramp load generator: rung-by-rung SLO gate with "
            "resource sampling (spec §10.3)."
        ),
    )
    parser.add_argument("--config", required=True, help="stt-server YAML config path")
    parser.add_argument(
        "--model", required=True, help="model name queried over the wire (see run_accuracy.py)"
    )
    parser.add_argument(
        "--utterance-seconds",
        type=float,
        required=True,
        help="synthetic utterance duration (seconds); ignored for real-manifest audio",
    )
    parser.add_argument("--start", type=int, required=True, help="starting concurrency")
    parser.add_argument("--step", type=int, required=True, help="concurrency increment per rung")
    parser.add_argument("--max", type=int, required=True, help="maximum concurrency to ramp to")
    parser.add_argument(
        "--slo-final-ms", type=float, required=True, help="SLO threshold for final latency (ms)"
    )
    parser.add_argument(
        "--slo-pct", type=int, required=True, help="percentile gated against --slo-final-ms"
    )
    parser.add_argument("--seed", type=int, required=True, help="audio-pool sampling seed")
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter to run the server with (default: sys.executable)",
    )
    parser.add_argument("--port", type=int, default=8100, help="port for the spawned server")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="tile the repo speech fixture instead of downloading a LibriSpeech manifest",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="how long to hold each concurrency rung (default: 30s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args)
    print(
        f"wrote {payload['_out_path']} "
        f"(max_passing_concurrency={payload['max_passing_concurrency']})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
