"""Zero-drops guard shared by every benchmark runner, plus the minimal
Prometheus text parser it (and `run_load`) needs. Stdlib-only: importing
this module must never drag in optional extras (`psutil`, `jiwer`, ...), so
it lives on its own rather than inside `run_load`.

Why this exists (Task 7 review, found by direct reproduction): a fast
client (`--pace 0`) can outrun a real backend's decode; the server's
default backpressure (`limits.audio_queue_chunks=64`,
`audio_overflow_policy=drop_oldest` -- the defaults whenever a config has
no `limits:` section, including `configs/sherpa.yaml`) then silently sheds
audio, and any WER measured through the server is "accuracy under active
audio drop", not the pipeline's accuracy. Every runner that streams audio
through a `Session` therefore brackets its streaming pass with
`fetch_metrics` scrapes and calls `guard_drops` on the
`stt_audio_dropped_total` delta.
"""

from __future__ import annotations

import urllib.request

DROPPED_METRIC = "stt_audio_dropped_total"


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Minimal Prometheus text-exposition-format parser: sums the value of
    every sample line for a given metric name, ACROSS all label
    combinations (e.g. `stt_errors_total{code="a"} 1` and
    `stt_errors_total{code="b"} 2` both contribute to a single
    `"stt_errors_total": 3.0` entry) -- callers here only need family-level
    totals, not per-label breakdowns. Ignores `#HELP`/`#TYPE` comment lines
    and blank lines; handles the optional trailing timestamp the exposition
    format allows (`name{labels} value timestamp_ms` -- the VALUE is the
    first token after the name/labels, never the last token); a line whose
    value token isn't a valid float is skipped rather than raising
    (forward-compatible with metric lines this parser doesn't understand).

    Not handled (irrelevant for `prometheus_client.generate_latest` output,
    which this parser exists to consume): label VALUES containing a literal
    `}` character would confuse the labels/value split below."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split off the metric name (+ optional {labels} block, which may
        # itself contain spaces inside quoted label values -- so a naive
        # whitespace split of the whole line is wrong when labels exist).
        if "{" in line:
            name = line.split("{", 1)[0]
            _, brace, after_labels = line.partition("}")
            if not brace:
                continue  # opening { without closing }: malformed, skip
            value_tokens = after_labels.split()
        else:
            tokens = line.split()
            name = tokens[0]
            value_tokens = tokens[1:]
        if not value_tokens:
            continue
        try:
            value = float(value_tokens[0])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def fetch_metrics(base_url: str) -> dict[str, float]:
    """Scrape `<base_url>/metrics` and return family-level totals (see
    `parse_prometheus_metrics`)."""
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5.0) as resp:  # noqa: S310
        text = resp.read().decode("utf-8")
    return parse_prometheus_metrics(text)


def audio_dropped_delta(
    metrics_before: dict[str, float], metrics_after: dict[str, float]
) -> float:
    """The `stt_audio_dropped_total` increase between two `fetch_metrics`
    scrapes (a missing family -- e.g. the counter never incremented, so
    prometheus_client emitted no sample line for some label set -- counts
    as 0.0 on that side)."""
    return metrics_after.get(DROPPED_METRIC, 0.0) - metrics_before.get(DROPPED_METRIC, 0.0)


def no_drops_failure_message(dropped_delta: float, assert_no_drops: bool) -> str | None:
    """Decide whether a through-the-server streaming pass is INVALID
    because the server shed audio (`stt_audio_dropped_total` increased
    while it ran). Returns the hard-fail error message, or `None` when the
    run is acceptable.

    A nonzero delta with `assert_no_drops=True` (every runner's default)
    must abort loudly, never produce a quietly-invalid WER. With
    `assert_no_drops=False` the delta is still recorded in the result JSON
    but does not fail the run (an explicit opt-in to measuring shed-audio
    behavior)."""
    if not assert_no_drops or dropped_delta == 0:
        return None
    return (
        f"server shed audio: {DROPPED_METRIC} increased by {dropped_delta:g} "
        "during the run -- its WER/latency would measure behavior under active "
        "audio drop, not the pipeline. Remedies: stream at real time "
        "(--pace 1.0), raise the server's limits.audio_queue_chunks, or set "
        "limits.audio_overflow_policy=error in the config; pass "
        "--no-assert-no-drops only if you explicitly want to measure "
        "under-shedding behavior."
    )


def guard_drops(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
    assert_no_drops: bool,
) -> float:
    """One-liner for runners: compute the dropped-audio delta between two
    scrapes, raise `RuntimeError` (see `no_drops_failure_message`) when the
    assertion is armed and the delta is nonzero, and return the delta so
    the caller records it in its result payload either way."""
    delta = audio_dropped_delta(metrics_before, metrics_after)
    failure = no_drops_failure_message(delta, assert_no_drops)
    if failure is not None:
        raise RuntimeError(failure)
    return delta
