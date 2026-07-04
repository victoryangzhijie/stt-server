"""Result JSON writing + shared stats/formatting helpers for benchmark runners."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _meta(**extra) -> dict:
    return {
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": sys.version,
        **extra,
    }


def write_result(
    name: str, payload: dict, results_dir: Path = DEFAULT_RESULTS_DIR
) -> Path:
    """Write `payload` (augmented with a `meta` block: git SHA, platform,
    python version, plus anything already under `payload["meta"]`) to
    `benchmarks/results/<name>-<YYYYMMDD-HHMMSS>.json`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{name}-{timestamp}.json"

    extra_meta = payload.get("meta", {})
    full_payload = {**payload, "meta": _meta(**extra_meta)}

    out_path.write_text(json.dumps(full_payload, indent=2, default=str))
    return out_path


def percentiles(xs: list[float]) -> dict:
    """`{"p50", "p95", "p99", "mean", "n"}` over `xs` (nearest-rank; empty
    input yields all-`None` values with `n=0`)."""
    n = len(xs)
    if n == 0:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}

    ordered = sorted(xs)

    def _pct(p: float) -> float:
        # Nearest-rank method: rank = ceil(p/100 * n), index = rank - 1,
        # clamped to [0, n-1].
        rank = math.ceil(p / 100 * n)
        idx = max(0, min(n - 1, rank - 1))
        return ordered[idx]

    return {
        "p50": _pct(50),
        "p95": _pct(95),
        "p99": _pct(99),
        "mean": sum(ordered) / n,
        "n": n,
    }


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    """Render `rows` as a GitHub-flavored Markdown table with `columns` (in
    order); missing keys render as an empty cell."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
