#!/usr/bin/env python3
"""Validate the bench-audio-v1-en dataset.

Stdlib-only (ffprobe via subprocess) so it runs under plain ``python3`` and
``uv run`` with no extra dependencies::

    uv run --no-project python benchmarks/data/validate.py
    # or, equivalently:
    python3 benchmarks/data/validate.py

For every clip in ``manifest.csv`` it re-measures with ffprobe and checks:
  * decodable
  * 16 kHz sample rate, mono, 16-bit PCM (sample_fmt ``s16``) WAV
  * duration inside its bucket  ->  http/ 10-20s, ws/ 30-60s
Plus manifest integrity: row count == on-disk WAV count, no missing paths,
no empty transcripts. Prints a per-file PASS/FAIL list, bucket min/median/max,
and exits non-zero on any failure.
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.csv"
HTTP = HERE / "http"
WS = HERE / "ws"
SR = 16000
BUCKETS = {"http/": (10.0, 20.0), "ws/": (30.0, 60.0)}
DUR_EPS = 0.05  # tolerate ~PCM rounding at bucket edges


def ffprobe(path: Path) -> dict:
    """Return audio stream + duration of ``path``, or {'error': ...}."""
    p = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries",
            "stream=sample_rate,channels,sample_fmt:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        return {"error": (p.stderr.strip() or "ffprobe failed")[:80]}
    data = json.loads(p.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    try:
        dur = float(fmt.get("duration") or stream.get("duration") or "")
    except (TypeError, ValueError):
        dur = None
    return {
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
        "sample_fmt": stream.get("sample_fmt"),
        "duration": dur,
    }


def bucket_of(rel: str) -> str:
    for prefix in BUCKETS:
        if rel.startswith(prefix):
            return prefix
    return ""


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: manifest not found at {MANIFEST}", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(MANIFEST.open()))
    failures: list[str] = []
    http_durs: list[float] = []
    ws_durs: list[float] = []

    # --- manifest <-> disk integrity ---
    on_disk: set[str] = set()
    for bucket in (HTTP, WS):
        if bucket.exists():
            on_disk.update(
                str(p.relative_to(HERE)).replace("\\", "/")
                for p in bucket.glob("*.wav")
            )
    manifest_paths = {r["relative_path"] for r in rows}
    extra = sorted(on_disk - manifest_paths)
    missing = sorted(manifest_paths - on_disk)
    if extra:
        failures.append(f"{len(extra)} file(s) on disk absent from manifest: {extra[:5]}")
    if missing:
        failures.append(f"{len(missing)} manifest path(s) missing on disk: {missing[:5]}")
    empty_tx = [r["relative_path"] for r in rows if not (r.get("transcript") or "").strip()]
    if empty_tx:
        failures.append(f"{len(empty_tx)} empty transcript(s): {empty_tx[:5]}")

    # --- per-file checks ---
    print(f"Validating {len(rows)} clips from manifest.csv (path relative to {HERE.name}/)\n")
    for r in rows:
        rel = r["relative_path"]
        info = ffprobe(HERE / rel)
        bucket = bucket_of(rel)
        lo, hi = BUCKETS.get(bucket, (0.0, 0.0))
        problems: list[str] = []
        dur: float | None = None
        if "error" in info:
            problems.append(f"undecodable: {info['error']}")
        else:
            if str(info["sample_rate"]) != str(SR):
                problems.append(f"sample_rate={info['sample_rate']}")
            if str(info["channels"]) != "1":
                problems.append(f"channels={info['channels']}")
            if info["sample_fmt"] != "s16":
                problems.append(f"sample_fmt={info['sample_fmt']}")
            dur = info["duration"]
            if dur is None:
                problems.append("duration=unknown")
            else:
                if not (lo - DUR_EPS <= dur <= hi + DUR_EPS):
                    problems.append(f"duration={dur:.2f}s outside [{lo},{hi}]")
                elif bucket == "http/":
                    http_durs.append(dur)
                elif bucket == "ws/":
                    ws_durs.append(dur)
        tag = "PASS" if not problems else "FAIL"
        if problems:
            failures.append(f"{rel}: " + "; ".join(problems))
        durstr = f"{dur:6.2f}s" if dur is not None else "    ?s"
        print(f"  [{tag}] {rel:<34} {durstr}")

    # --- summary ---
    def summarize(name: str, durs: list[float], lo: float, hi: float) -> None:
        if not durs:
            failures.append(f"no {name} clips passed duration measurement")
            print(f"\n{name}: 0 clips")
            return
        in_bucket = all(lo - DUR_EPS <= d <= hi + DUR_EPS for d in durs)
        if not in_bucket:
            failures.append(f"{name} has out-of-bucket durations")
        print(
            f"\n{name} {len(durs)} clips  |  duration min/median/max = "
            f"{min(durs):.2f} / {statistics.median(durs):.2f} / {max(durs):.2f} s"
            f"  (bucket {lo:.0f}-{hi:.0f}s)"
        )

    print("\n=== summary ===")
    summarize("http/", http_durs, *BUCKETS["http/"])
    summarize("ws/", ws_durs, *BUCKETS["ws/"])
    n_http = len(list(HTTP.glob("*.wav"))) if HTTP.exists() else 0
    n_ws = len(list(WS.glob("*.wav"))) if WS.exists() else 0
    print(
        f"\nmanifest rows={len(rows)}  http files={n_http}  ws files={n_ws}  "
        f"(rows must equal http+ws)"
    )

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
