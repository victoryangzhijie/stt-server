"""HTTP load matrix client for `POST /v1/audio/transcriptions` (spec §10.3).

A standalone asyncio+httpx client (no stt_server/benchmarks imports) that
drives ONE concurrency level of the HTTP transcription endpoint: it round-robins
the curated ``bench-audio-v1-en`` clips in ``benchmarks/data/http/``, warms the
server with a fixed number of untimed requests, then fires N timed requests at a
fixed concurrency and records per-request latency. Result JSON shape::

    {
      "meta": { "dataset": "bench-audio-v1-en", "model", "base_url",
                "concurrency", "requests", "n_warmup_excluded", "seed",
                "latency_def", "started_at", "ended_at", "token_used": bool },
      "per_request": [ { "index", "file", "http_status", "latency_ms",
                         "text_chars", "error" }, ... ],   # timed only
      "summary": { "concurrency", "n", "n_errors", "error_rate",
                   "latency_ms": {"p50","p95","p99","mean","n"},
                   "throughput_req_per_s", "wall_seconds",
                   "gpu_mem_peak_mib", "gpu_util_peak_pct" }
    }

``latency_def`` documents exactly what each number is: end-to-end wall-clock per
HTTP request (ms), client-side on loopback -- the latency a real caller sees,
covering server queueing + decode + transcription but not transit (negligible on
127.0.0.1). Percentiles use the same nearest-rank method as
``benchmarks.results.percentiles`` so this matrix is directly comparable to the
WS ladder output.

Usage (one level; the run is orchestrated per-level so each level gets its own
GPU-monitor CSV, a VRAM-settle gap, and its own commit)::

    uv run python benchmarks/bench_http_stt.py \\
        --concurrency 8 --requests 50 --warmup 5 \\
        --base-url http://127.0.0.1:8100 --model qwen3-asr-0.6b \\
        --out reports/2026-08-01/load_http/c8.json \\
        --gpu-csv reports/2026-08-01/gpu_monitor/http_c8.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import httpx

DATASET = "bench-audio-v1-en"
# Match the server's _decode_wav contract (src/stt_server/api/transcriptions_http.py):
# 16 kHz mono 16-bit PCM WAV -- which is exactly what validate.py guarantees for
# every clip in benchmarks/data/http/, so the WAV bytes are POSTed verbatim.
REQUEST_TIMEOUT_S = 180.0


def _percentiles(xs: list[float]) -> dict:
    """Nearest-rank p50/p95/p99/mean/n, identical to benchmarks.results.percentiles."""
    n = len(xs)
    if n == 0:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
    ordered = sorted(xs)

    def _pct(p: float) -> float:
        rank = math.ceil(p / 100 * n)
        idx = max(0, min(n - 1, rank - 1))
        return ordered[idx]

    return {"p50": _pct(50), "p95": _pct(95), "p99": _pct(99), "mean": sum(ordered) / n, "n": n}


def _gpu_peaks(csv_path: Path) -> tuple[float | None, float | None]:
    """Max memory.used (MiB) and utilization.gpu (%) from a `nvidia-smi --query-gpu=
    timestamp,memory.used,utilization.gpu --format=csv` log. Returns (None, None) if
    the file is missing/empty. Cell values carry units ("1234 MiB", "0 %"), so the
    first integer in each cell is extracted."""
    if not csv_path.exists():
        return None, None
    mem_max: float | None = None
    util_max: float | None = None
    for line in csv_path.read_text().splitlines():
        if not line.strip() or line.startswith("timestamp"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        nums = [re.search(r"\d+", p) for p in parts[1:3]]
        if nums[0]:
            mem = float(nums[0].group())
            mem_max = mem if mem_max is None else max(mem_max, mem)
        if nums[1]:
            util = float(nums[1].group())
            util_max = util if util_max is None else max(util_max, util)
    return mem_max, util_max


async def _one(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    wav: bytes,
    fname: str,
    headers: dict | None,
) -> dict:
    """Single POST; returns a per-request record (always, even on error)."""
    files = {"file": (fname, wav, "audio/wav")}
    data = {"model": model, "response_format": "json"}
    rec: dict = {
        "file": fname,
        "http_status": None,
        "latency_ms": None,
        "text_chars": None,
        "error": None,
    }
    t0 = time.monotonic()
    try:
        resp = await client.post(url, data=data, files=files, headers=headers)
        rec["http_status"] = resp.status_code
        rec["latency_ms"] = (time.monotonic() - t0) * 1000.0
        if resp.status_code != 200:
            rec["error"] = f"http {resp.status_code}: {resp.text[:200]}"
            return rec
        body = resp.json()
        rec["text_chars"] = len(body.get("text", "") or "")
        if rec["text_chars"] == 0:
            rec["error"] = "empty transcript"
        return rec
    except Exception as exc:  # noqa: BLE001 -- record every failure mode
        rec["latency_ms"] = (time.monotonic() - t0) * 1000.0
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec


async def _run_level(
    url: str,
    model: str,
    clips: list[tuple[str, bytes]],
    concurrency: int,
    n_requests: int,
    n_warmup: int,
    headers: dict | None,
) -> tuple[list[dict], float, int, int]:
    """Fire `n_warmup` untimed requests then `n_requests` timed requests at
    `concurrency`. Returns (timed_records, timed_wall_seconds, warmup_ok,
    warmup_err). Files are picked round-robin (offset by request index) from
    `clips`, which is already deterministically shuffled by the caller."""
    limits = httpx.Limits(
        max_connections=max(concurrency, 1) * 2, max_keepalive_connections=concurrency
    )
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Warmup: sequential, untimed -- primes model weights / KV cache so the
        # first timed request isn't paying a one-off cold cost.
        warmup_ok = warmup_err = 0
        for i in range(n_warmup):
            fname, wav = clips[i % len(clips)]
            rec = await _one(client, url, model, wav, fname, headers)
            if rec["error"] is None:
                warmup_ok += 1
            else:
                warmup_err += 1

        # Timed batch: at most `concurrency` in flight, exactly n_requests total.
        records: list[dict] = [None] * n_requests  # type: ignore[list-item]

        async def task(i: int) -> None:
            async with sem:
                fname, wav = clips[i % len(clips)]
                rec = await _one(client, url, model, wav, fname, headers)
                rec["index"] = i
                records[i] = rec

        wall_t0 = time.monotonic()
        await asyncio.gather(*(task(i) for i in range(n_requests)))
        wall = time.monotonic() - wall_t0
        return records, wall, warmup_ok, warmup_err


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python benchmarks/bench_http_stt.py")
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="qwen3-asr-0.6b")
    ap.add_argument("--data-dir", default="benchmarks/data/http")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--gpu-csv", default=None, help="nvidia-smi monitor CSV for peak readout")
    ap.add_argument("--seed", type=int, default=42, help="deterministic clip-order shuffle seed")
    ap.add_argument("--token", default=None, help="bearer token (server default: none)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    wavs = sorted(data_dir.glob("*.wav"))
    if not wavs:
        print(f"no wav clips under {data_dir}", file=sys.stderr)
        return 2
    # Deterministic shuffle then round-robin so a given seed yields the same
    # clip sequence (and thus the same file/concurrency interleaving) every run.
    rng = random.Random(args.seed)
    rng.shuffle(wavs)
    clips: list[tuple[str, bytes]] = [(w.name, w.read_bytes()) for w in wavs]

    url = f"{args.base_url}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else None

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records, wall, warmup_ok, warmup_err = asyncio.run(
        _run_level(
            url,
            args.model,
            clips,
            args.concurrency,
            args.requests,
            args.warmup,
            headers,
        )
    )
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
    errors = [r for r in records if r["error"] is not None]
    n = len(records)
    n_err = len(errors)
    pct = _percentiles(latencies)
    throughput = (n / wall) if wall > 0 else None
    mem_peak, util_peak = _gpu_peaks(Path(args.gpu_csv)) if args.gpu_csv else (None, None)

    payload = {
        "meta": {
            "dataset": DATASET,
            "model": args.model,
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "n_warmup_excluded": args.warmup,
            "warmup_ok": warmup_ok,
            "warmup_err": warmup_err,
            "seed": args.seed,
            "token_used": args.token is not None,
            "latency_def": (
                "end-to-end wall-clock per HTTP request (ms), measured client-side on "
                "loopback 127.0.0.1: from just before POST to full response body received. "
                "Covers server queueing + VAD + decode + transcription (the latency a real "
                "caller observes); excludes only network transit, negligible on loopback. "
                "Errors still record an elapsed latency (up to the failure point). "
                "Percentiles are nearest-rank, matching benchmarks.results.percentiles."
            ),
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "per_request": records,
        "summary": {
            "concurrency": args.concurrency,
            "n": n,
            "n_errors": n_err,
            "error_rate": (n_err / n) if n else None,
            "latency_ms": pct,
            "throughput_req_per_s": throughput,
            "wall_seconds": round(wall, 3),
            "gpu_mem_peak_mib": mem_peak,
            "gpu_util_peak_pct": util_peak,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"c={args.concurrency} n={n} err={n_err} "
        f"p50={pct['p50']:.0f} p95={pct['p95']:.0f} p99={pct['p99']:.0f}ms "
        f"thr={throughput:.2f}r/s wall={wall:.1f}s "
        f"gpu_mem_peak={mem_peak}MiB util_peak={util_peak}% -> {out}"
        if throughput and pct["p50"]
        else f"c={args.concurrency} n={n} err={n_err} -> {out}"
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
