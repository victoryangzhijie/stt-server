"""Stabilizer flicker/commit-latency grid study (spec §10.4), plus the
realtime-API delta-duplication measurement (carried finding M-8).

For each point of a `StabilizerConfig` parameter grid (e.g. `min_partials`,
`min_stable_ms` -- the REAL field names in
`src/stt_server/config/settings.py`), boots a real `stt-server` subprocess
with a temp config derived from `--config` with just that grid point's
stabilizer values substituted, streams a LibriSpeech manifest subset over
it, and scores:

- ``--api native`` (default): the native `/ws/transcribe` wire protocol
  (`benchmarks.client_ws.stream_utterance`). Reports mean flicker_rate,
  mean commit_latency_ms (the three pure metric functions below -- THE
  deliverable this study exists to compute), and corpus WER.
- ``--api realtime``: the OpenAI-compatible `/v1/realtime?intent=
  transcription` wire protocol (`benchmarks.client_realtime.stream_realtime`).
  Reports `delta_duplication_ratio` (M-8: how much the realtime adapter's
  `conversation.item.input_audio_transcription.delta` stream repeats itself
  relative to an append-only wire -- see that function's docstring) and
  corpus WER. Flicker/commit-latency are NOT computed in this mode: the
  realtime protocol has no `PARTIAL`/volatile-text wire representation to
  measure them from (see `encode_realtime` in
  `src/stt_server/api/realtime_ws.py` -- `PARTIAL` has no OpenAI encoding).

Usage::

    uv run python -m benchmarks.run_stabilizer_study \\
        --config configs/sherpa.yaml --model sherpa --n 25 --seed 42 \\
        --grid "min_partials=1,2,3;min_stable_ms=0,240,480" \\
        [--api native|realtime] [--split test-clean] [--port 8100] \\
        [--python /path/to/venv/bin/python] [--pace 1.0] \\
        [--assert-no-drops | --no-assert-no-drops]

Per-grid-point validity guard: each point's streaming pass (either API --
both stream through a Session with the same bounded audio queue) is
bracketed with /metrics scrapes and hard-fails (unless
``--no-assert-no-drops``) when ``stt_audio_dropped_total`` increased --
audio shedding under drop_oldest (see ``benchmarks._drops``) invalidates
that point's flicker/WER/duplication figures. The delta is recorded as
``points[i].audio_dropped_delta`` either way.

Writes `stabilizer-study-<model>-<timestamp>.json` via
`benchmarks.results.write_result`: one row (`points[i]`) per grid point,
each `{"params": {...}, "wer": ..., "errors": ..., "n": ...,
"flicker_rate": {...percentiles...}, "commit_latency_ms": {...}}` (native)
or `{"params": {...}, "wer": ..., "errors": ..., "n": ...,
"delta_duplication_ratio": {...}}` (realtime).

Failure semantics: the result JSON is written ONCE, after the WHOLE grid
completes -- a failure mid-grid (a grid point's server failing to boot, a
tripped `ConsecutiveErrorBreaker`, Ctrl-C) aborts the entire run with NO
partial results file. Re-run with a smaller `--grid` to salvage the points
that worked.

CLI deltas from the brief's example invocation (both noted per the task
brief's instruction to flag CLI additions):

- `--pace` (pass-through to the streaming clients, default 1.0): the
  required smoke command passes `--pace 0`, so this had to exist; mirrors
  `run_accuracy.py`'s flag of the same name/meaning.
- `--split` (default `"test-clean"`, NOT required): the brief's example
  CLI omits `--split` entirely. `benchmarks.corpus.build_manifest` needs
  *some* split, so rather than silently hardcoding one, it's exposed as an
  optional flag defaulting to the value implied by the example.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
import tempfile
from pathlib import Path

import yaml

from benchmarks._drops import fetch_metrics, guard_drops
from benchmarks.client_realtime import RealtimeResult, stream_realtime
from benchmarks.client_ws import UtteranceResult
from benchmarks.corpus import Utterance, build_manifest, download_subset, load_pcm16
from benchmarks.results import percentiles, write_result
from benchmarks.run_accuracy import ConsecutiveErrorBreaker, _run_ws_mode, corpus_wer
from benchmarks.server import ServerUnderTest

# ---------------------------------------------------------------------------
# Pure flicker/commit-latency metrics (spec §10.4). These three functions are
# THE deliverable Task 9's report cites -- exact semantics are documented
# here and exercised with fully worked examples in
# tests/benchmarks/test_flicker_metrics.py.
# ---------------------------------------------------------------------------


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common PREFIX of `a` and `b` (character-wise,
    not word-wise). `common_prefix_len("THE CAT", "THE CAR") == 6`
    (`"THE CA"`)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def retracted_chars(partials: list[str]) -> int:
    """Sum, over every CONSECUTIVE pair of volatile-hypothesis strings, of
    the characters the user SAW then LOST: `len(prev) -
    common_prefix_len(prev, cur)` whenever `cur` does not simply extend
    `prev` (i.e. `not cur.startswith(prev)`). A pair where `cur` extends
    `prev` contributes 0 -- pure growth is not flicker.

    Worked example (the brief's own): `partials = ["THE", "THE CAT", "THE
    CAR"]`.
      - pair ("THE", "THE CAT"): "THE CAT".startswith("THE") -> extends,
        contributes 0.
      - pair ("THE CAT", "THE CAR"): "THE CAR" does NOT start with "THE
        CAT" -> retraction. common_prefix_len("THE CAT", "THE CAR") == 6
        ("THE CA"). len("THE CAT") == 7. Contributes 7 - 6 == 1 (the "T" of
        "CAT" was shown, then retracted when "CAR" replaced it).
      - total: 1.

    `partials` is assumed to be in temporal (arrival) order; only adjacent
    pairs are compared -- a volatile hypothesis that later reappears after
    an intervening detour is NOT credited back (this measures churn, not
    net displayed-then-restored characters)."""
    total = 0
    for prev, cur in zip(partials, partials[1:], strict=False):
        if cur.startswith(prev):
            continue
        total += len(prev) - common_prefix_len(prev, cur)
    return total


def flicker_rate(partials: list[str], final: str) -> float:
    """`retracted_chars(partials) / max(1, len(final))` -- retracted
    characters normalized by the final transcript's length, so utterances
    of different lengths are comparable. `max(1, ...)` guards an empty
    `final` from a ZeroDivisionError (an empty final can only arise from an
    errored/silent utterance, which callers exclude from aggregation
    anyway -- this is a defensive floor, not a claim that a 0-length final
    has a meaningful flicker rate)."""
    return retracted_chars(partials) / max(1, len(final))


def _hyp_text_at_tick(stable_text: str, volatile_text: str) -> str:
    """The COMBINED user-visible hypothesis at one tick: `stable_text`
    followed by `volatile_text`, single-space-joined (with sensible joining
    when either side is empty -- no leading/trailing space artifacts).
    This is the string a viewer's screen shows, and it is what the flicker
    metrics must be fed: volatile_text ALONE shrinks on every normal commit
    (words migrate from volatile to stable), which is growth from the
    viewer's perspective, not a retraction."""
    if not volatile_text:
        return stable_text
    if not stable_text:
        return volatile_text
    return f"{stable_text} {volatile_text}"


def _hyp_words_at_tick(stable_text: str, volatile_text: str) -> list[str]:
    """The words visible to a viewer at one tick: `stable_text` followed by
    `volatile_text` (mirroring `client_ws`'s own `(audio_time_ms,
    stable_text, volatile_text)` triples, NOT the wire's separately-spaced
    STABILIZED deltas)."""
    return _hyp_text_at_tick(stable_text, volatile_text).split()


def commit_latency_word_latencies(
    partials_with_time: list[tuple[float, str, str]], final: str
) -> list[float]:
    """Per-word commit latencies: for each of `final`'s words that is
    INCLUDED (see the exclusion rule below), (time the word became part of
    STABLE text) - (time the word FIRST appeared in any hypothesis, stable
    or volatile). `commit_latency_ms` is the mean of this list; the list
    itself is exposed so callers can also report the INCLUDED-word count
    (`len(...)`) -- the mean's denominator shrinks as more words are
    excluded, and grid comparisons need to see that shrinkage.

    Word identity is POSITIONAL: word `i` means `final.split()[i]`, matched
    against word `i` of a tick's stable-text split (for "became stable") or
    combined stable+volatile split (for "first appeared") -- never matched
    by searching for the word elsewhere in the string. Comparison is
    casefold-only (no punctuation stripping), consistent with the realtime
    encoder's own reconciliation rule (see `delta_duplication_ratio` and
    `src/stt_server/api/realtime_ws.py`'s FINAL-vs-wire_sent casefold
    comparison) -- a word whose ONLY difference from its partial-stream
    appearance is trailing punctuation (`"fox"` vs `"fox."`) will not
    positionally match and is therefore EXCLUDED (see below), not
    penalized with a large or a zero latency.

    Every tick is scanned (not just the first matching one) and the
    MINIMUM matching time is kept, so out-of-order or repeated matches at
    the same position can't inflate latency.

    Words that never appear at their position in ANY tick's stable text
    within `partials_with_time` are EXCLUDED from the mean entirely -- they
    are not given a synthetic 0-latency, and they do not lower the
    denominator's weight of other words. Rationale (explicit design
    decision per the brief): `commit_latency_ms` takes no timestamp for
    `final` itself, so a word that only ever appears in `final` (never
    committed to stable text during any observed partial) has NO
    observable "became stable" tick to measure FROM -- it was never
    provisional in the data this function can see, so contributing
    anything (0 or otherwise) would be fabricating a data point instead of
    describing one.

    A degenerate defensive fallback: if a word's stable_time IS found but
    (against the invariant that being stable at tick t implies being in
    the combined hypothesis at tick t) no first_appear_time was recorded,
    first_appear_time is set equal to stable_time (contributing a latency
    of exactly 0 for that word) rather than raising or excluding --
    reachable only if a caller's `partials_with_time` skips volatile text
    inconsistently with its stable text.

    Worked example: `final = "hello world"`, `partials_with_time =
    [(100.0, "hello world", "")]` (single partial identical to final).
    Both words are in `stable_text` at position 0/1 at t=100, and also in
    the combined hypothesis at t=100 (same tick) -> first_appear_time ==
    stable_time == 100.0 for both -> latencies [0.0, 0.0] -> mean 0.0.

    Returns [] when there are no partials at all, when `final` is empty,
    or when every word of `final` is excluded."""
    final_words = final.split()
    if not final_words:
        return []

    first_appear_time: list[float | None] = [None] * len(final_words)
    stable_time: list[float | None] = [None] * len(final_words)

    for audio_time_ms, stable_text, volatile_text in partials_with_time:
        hyp_words = _hyp_words_at_tick(stable_text, volatile_text)
        stable_words = stable_text.split()
        for i, target in enumerate(final_words):
            target_cf = target.casefold()
            if i < len(hyp_words) and hyp_words[i].casefold() == target_cf:
                if first_appear_time[i] is None or audio_time_ms < first_appear_time[i]:
                    first_appear_time[i] = audio_time_ms
            if i < len(stable_words) and stable_words[i].casefold() == target_cf:
                if stable_time[i] is None or audio_time_ms < stable_time[i]:
                    stable_time[i] = audio_time_ms

    latencies: list[float] = []
    for i in range(len(final_words)):
        if stable_time[i] is None:
            continue  # never observed becoming stable -- EXCLUDED, see docstring
        appear = first_appear_time[i] if first_appear_time[i] is not None else stable_time[i]
        latencies.append(max(0.0, stable_time[i] - appear))
    return latencies


def commit_latency_ms(
    partials_with_time: list[tuple[float, str, str]], final: str
) -> float:
    """Mean of `commit_latency_word_latencies(partials_with_time, final)`
    (see that function for the full per-word semantics, positional matching
    rule, and the exclusion decision). Returns 0.0 (not NaN/None) when
    there are no partials at all, or when every word of `final` is
    excluded -- an explicit "nothing observed" sentinel, matching this
    module's other metrics' zero-safe defaults."""
    latencies = commit_latency_word_latencies(partials_with_time, final)
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# M-8: realtime delta-duplication measurement.
# ---------------------------------------------------------------------------


def extra_chars(joined_deltas: str, completed_transcript: str) -> int:
    """`max(0, len(joined_deltas) - len(completed_transcript))` -- how many
    MORE characters an `"".join(deltas)` reconstruction contains than the
    `completed` transcript it should equal under wire-perfect (append-only,
    non-duplicating) delta semantics. Floored at 0: a joined-deltas string
    SHORTER than completed (e.g. the connection dropped before the last
    delta arrived) is a different failure mode this ratio doesn't attempt
    to quantify, not "negative duplication"."""
    return max(0, len(joined_deltas) - len(completed_transcript))


def delta_duplication_ratio(joined_deltas: str, completed_transcript: str) -> float:
    """`extra_chars(joined_deltas, completed_transcript) /
    len(completed_transcript)`. 0.0 = wire-perfect append-only reconstruction
    (`"".join(deltas) == completed`, modulo the casefold-insensitive casing
    drift `src/stt_server/api/realtime_ws.py`'s own FINAL-reconciliation
    logic explicitly tolerates -- see that module's `sender()` closure).
    A positive value quantifies the "shrinking final" fallback: when a
    FINAL transcript does NOT casefold-startswith the exact text already
    sent on the wire (the stabilizer's own committed text was revised
    downstream of what was already streamed), the adapter re-sends the
    ENTIRE final text as one more delta rather than a true incremental
    remainder -- duplicating everything already sent for that item.

    `completed_transcript == ""` returns 0.0 (nothing to divide by, and an
    empty completed transcript cannot itself have been duplicated)."""
    if not completed_transcript:
        return 0.0
    return extra_chars(joined_deltas, completed_transcript) / len(completed_transcript)


# ---------------------------------------------------------------------------
# Grid parsing + temp-config derivation.
# ---------------------------------------------------------------------------

# Verified against `StabilizerConfig` in src/stt_server/config/settings.py
# (as of this writing: `min_partials: int = 2`, `min_stable_ms: float =
# 400.0`) -- these are the REAL field names, not the brief's illustrative
# example (which happened to already match).
STABILIZER_FIELD_TYPES: dict[str, type] = {
    "min_partials": int,
    "min_stable_ms": float,
}


def parse_grid(grid_str: str) -> list[dict]:
    """Parse `"min_partials=1,2,3;min_stable_ms=0,240,480"` (semicolon-
    separated axes, comma-separated values per axis) into the CROSS PRODUCT
    of grid points, e.g. `[{"min_partials": 1, "min_stable_ms": 0.0},
    {"min_partials": 1, "min_stable_ms": 240.0}, ...]` (9 points for that
    example: 3 x 3). Values are cast per `STABILIZER_FIELD_TYPES`; an axis
    name not in that mapping raises `ValueError` immediately -- a typo'd
    field name should fail the run at argument-parsing time, not silently
    produce a grid point whose override is ignored by
    `StabilizerConfig`'s own validation."""
    axes: dict[str, list] = {}
    for clause in grid_str.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        name, sep, values_str = clause.partition("=")
        name = name.strip()
        if not sep:
            raise ValueError(f"malformed --grid clause {clause!r}; expected name=v1,v2,...")
        if name not in STABILIZER_FIELD_TYPES:
            raise ValueError(
                f"unknown stabilizer field {name!r} in --grid; known fields: "
                f"{sorted(STABILIZER_FIELD_TYPES)} (see StabilizerConfig in "
                "src/stt_server/config/settings.py)"
            )
        caster = STABILIZER_FIELD_TYPES[name]
        values = [caster(v.strip()) for v in values_str.split(",") if v.strip()]
        if not values:
            raise ValueError(f"--grid axis {name!r} has no values")
        axes[name] = values

    if not axes:
        raise ValueError(f"--grid {grid_str!r} produced no axes")

    names = list(axes)
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.product(*(axes[n] for n in names))
    ]


def make_temp_config(base_config_path: str, overrides: dict, tmpdir: Path) -> Path:
    """Load `base_config_path`'s YAML, patch its `stabilizer:` mapping with
    `overrides` (leaving any stabilizer field NOT in `overrides`, and every
    other top-level section, untouched), and write the merged config to a
    new file under `tmpdir`. Returns the temp file's path."""
    with open(base_config_path) as f:
        data = yaml.safe_load(f) or {}
    stabilizer = dict(data.get("stabilizer") or {})
    stabilizer.update(overrides)
    data["stabilizer"] = stabilizer

    suffix = "-".join(f"{k}_{v}" for k, v in overrides.items()) or "default"
    out_path = tmpdir / f"stabilizer-{suffix}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f)
    return out_path


# ---------------------------------------------------------------------------
# Per-grid-point run + scoring.
# ---------------------------------------------------------------------------


async def _run_realtime_mode(
    server: ServerUnderTest, model: str, manifest: list[Utterance], pace: float
) -> tuple[list[RealtimeResult], int]:
    """Realtime-protocol analog of `run_accuracy._run_ws_mode`: stream each
    utterance sequentially (concurrency 1) over `/v1/realtime?intent=
    transcription`. An utterance whose stream raises, or completes with no
    transcript, is recorded as `hypothesis=""` and counted as an error --
    the run continues (unless `MAX_CONSECUTIVE_ERRORS` error in a row; see
    `ConsecutiveErrorBreaker`, reused as-is from `run_accuracy`)."""
    results: list[RealtimeResult] = []
    errors = 0
    breaker = ConsecutiveErrorBreaker("realtime")
    for utt in manifest:
        pcm16 = load_pcm16(utt)
        try:
            result = await stream_realtime(server.base_ws_url, model, pcm16, pace=pace)
        except Exception:
            result = RealtimeResult(utt_id=utt.id, hypothesis="", items=[])
            errors += 1
            breaker.record(errored=True)
            results.append(result)
            continue
        result.utt_id = utt.id
        errored = not result.hypothesis
        if errored:
            errors += 1
        breaker.record(errored)
        results.append(result)
    return results, errors


def _score_native(
    manifest: list[Utterance], results: list[UtteranceResult], errors: int
) -> dict:
    references = [utt.ref_text for utt in manifest]
    hyps_by_id = {r.utt_id: r.hypothesis for r in results}
    hypotheses = [hyps_by_id.get(utt.id, "") for utt in manifest]

    flicker_values: list[float] = []
    commit_values: list[float] = []
    included_words = 0
    for r in results:
        if not r.hypothesis:
            continue  # errored/silent utterance: excluded, not a fake 0-sample

        # Segment the partial trail by SERVER-SIDE utterance id: one audio
        # buffer can endpoint into several utterances, and the hypothesis
        # RESETS at each boundary -- diffing across a boundary would score
        # the reset as a giant fake retraction, and commit-latency's
        # positional word matching would mis-align every segment after the
        # first against the joined multi-segment hypothesis.
        segments: dict[int, list[tuple[float, str, str]]] = {}
        for tick, uid in zip(r.partials, r.partial_utterance_ids, strict=True):
            segments.setdefault(uid, []).append(tick)

        # Feed the metrics the COMBINED user-visible hypothesis per tick
        # (stable + volatile), NOT volatile_text alone: volatile_text
        # shrinks on every normal commit (words migrate to stable_text),
        # which from the viewer's perspective is growth, not retraction --
        # a volatile-only sequence makes every commit look like a huge
        # retraction and wildly inflates flicker_rate.
        retracted_total = 0
        word_latencies: list[float] = []
        for uid, final_text in r.finals:
            seg = segments.get(uid, [])
            hyp_seq = [_hyp_text_at_tick(s, v) for _, s, v in seg]
            retracted_total += retracted_chars(hyp_seq)
            word_latencies.extend(commit_latency_word_latencies(seg, final_text))

        # One flicker/commit sample per MANIFEST utterance (same population
        # as WER): per-segment retractions summed, normalized by the whole
        # joined hypothesis's length; per-segment word latencies pooled.
        flicker_values.append(retracted_total / max(1, len(r.hypothesis)))
        included_words += len(word_latencies)
        if word_latencies:
            commit_values.append(sum(word_latencies) / len(word_latencies))
        else:
            # Same 0.0 "nothing observed" sentinel commit_latency_ms uses.
            commit_values.append(0.0)

    return {
        "wer": corpus_wer(references, hypotheses),
        "errors": errors,
        "n": len(manifest),
        "flicker_rate": percentiles(flicker_values),
        "commit_latency_ms": percentiles(commit_values),
        # Total words (across all scored utterances) that actually entered
        # the commit-latency mean. commit_latency_ms's exclusion rule (words
        # never observed becoming stable are dropped) shrinks the
        # denominator, and MORE aggressive stabilizer settings exclude MORE
        # words -- Task 9 needs this count to spot artificially-low means
        # driven by shrinkage rather than genuinely faster commits.
        "commit_latency_included_words": included_words,
    }


def _score_realtime(
    manifest: list[Utterance], results: list[RealtimeResult], errors: int
) -> dict:
    references = [utt.ref_text for utt in manifest]
    hyps_by_id = {r.utt_id: r.hypothesis for r in results}
    hypotheses = [hyps_by_id.get(utt.id, "") for utt in manifest]

    ratios: list[float] = []
    for r in results:
        for item in r.items:
            if item.completed_transcript is None:
                # Transcript NEVER ARRIVED (connection ended before this
                # item's `.completed` event): the item is unmeasurable and
                # is excluded. Distinct from a transcript that arrived
                # EMPTY (`""`), which IS measured -- it contributes a 0.0
                # ratio via delta_duplication_ratio's empty-completed rule
                # (an empty completed transcript cannot have been
                # duplicated).
                continue
            joined = "".join(item.deltas)
            ratios.append(delta_duplication_ratio(joined, item.completed_transcript))

    return {
        "wer": corpus_wer(references, hypotheses),
        "errors": errors,
        "n": len(manifest),
        "delta_duplication_ratio": percentiles(ratios),
    }


def run(args: argparse.Namespace) -> dict:
    split_dir = download_subset(args.split)
    manifest = build_manifest(split_dir, args.n, args.seed)
    grid = parse_grid(args.grid)

    payload: dict = {
        "model": args.model,
        "config": args.config,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "pace": args.pace,
        "api": args.api,
        "grid": args.grid,
        "points": [],
    }

    with tempfile.TemporaryDirectory(prefix="stabilizer-study-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for point in grid:
            config_path = make_temp_config(args.config, point, tmpdir)
            with ServerUnderTest(str(config_path), port=args.port, python=args.python) as server:
                # Zero-drops guard (benchmarks._drops), per grid point: both
                # APIs stream through a Session with the same bounded audio
                # queue, so either can shed under drop_oldest when the
                # client outruns a real backend (--pace 0) -- a shed pass's
                # flicker/WER/duplication figures would be invalid.
                metrics_before = fetch_metrics(server.base_url)
                if args.api == "realtime":
                    results, errors = asyncio.run(
                        _run_realtime_mode(server, args.model, manifest, args.pace)
                    )
                    scored = _score_realtime(manifest, results, errors)
                else:
                    results, errors = asyncio.run(
                        _run_ws_mode(server, args.model, manifest, args.pace)
                    )
                    scored = _score_native(manifest, results, errors)
                metrics_after = fetch_metrics(server.base_url)
            scored["audio_dropped_delta"] = guard_drops(
                metrics_before, metrics_after, args.assert_no_drops
            )
            payload["points"].append({"params": point, **scored})

    out_path = write_result(f"stabilizer-study-{args.model}", payload)
    payload["_out_path"] = str(out_path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_stabilizer_study",
        description=(
            "Stabilizer flicker-rate/commit-latency grid study (spec §10.4), "
            "plus the realtime-API delta-duplication measurement (M-8)."
        ),
    )
    parser.add_argument("--config", required=True, help="stt-server YAML config path")
    parser.add_argument(
        "--model",
        required=True,
        help="model name queried over the wire (see run_accuracy.py for the lookup rule)",
    )
    parser.add_argument(
        "--split",
        default="test-clean",
        choices=["test-clean", "test-other"],
        help="LibriSpeech split (default: test-clean; not required -- see module docstring)",
    )
    parser.add_argument("--n", type=int, required=True, help="number of utterances to sample")
    parser.add_argument("--seed", type=int, required=True, help="manifest sampling seed")
    parser.add_argument(
        "--grid",
        required=True,
        help='stabilizer parameter grid, e.g. "min_partials=1,2,3;min_stable_ms=0,240,480"',
    )
    parser.add_argument(
        "--api",
        default="native",
        choices=["native", "realtime"],
        help="wire protocol to stream over (default: native /ws/transcribe)",
    )
    parser.add_argument("--port", type=int, default=8100, help="port for the spawned server")
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter to run the server with (default: sys.executable)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="streaming pacing multiplier (1.0=real-time, 0=as-fast-as-possible)",
    )
    parser.add_argument(
        "--assert-no-drops",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "hard-fail the run if stt_audio_dropped_total increased during any "
            "grid point's streaming pass (audio was shed; that point's metrics "
            "would be invalid). --no-assert-no-drops records the delta without "
            "failing."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args)
    print(f"wrote {payload['_out_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
