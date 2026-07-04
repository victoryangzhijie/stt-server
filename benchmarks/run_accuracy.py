"""Accuracy (WER) + concurrency-1 latency benchmark runner (spec §10.1/§10.2).

The real-time-paced WS streaming pass IS the concurrency-1 latency
measurement -- one pass over the manifest yields both the streamed-mode WER
and server-/client-observed latency percentiles; there is no separate
"latency-only" pass.

Usage::

    uv run python -m benchmarks.run_accuracy \\
        --config configs/sherpa.yaml --model sherpa --split test-clean \\
        --n 100 --seed 42 [--modes ws,file] [--port 8100] \\
        [--python /path/venv/bin/python] [--pace 1.0] \\
        [--assert-no-drops | --no-assert-no-drops]

WS-mode validity guard: the WS pass is bracketed with /metrics scrapes and
hard-fails (unless ``--no-assert-no-drops``) when ``stt_audio_dropped_total``
increased -- a fast client (``--pace 0``) can outrun a real backend and trip
the server's default drop_oldest audio shedding, silently invalidating the
WER (see ``benchmarks._drops``). The delta is recorded as
``results.ws.audio_dropped_delta`` either way. File mode is exempt: the
whole upload is pushed as ONE AudioChunk, which cannot overflow the
chunk-counted queue (see the comment in ``run()``).

``--model`` is the *query-string* model name the server's registry resolves
via ``resolve_backend`` (``src/stt_server/api/app.py``): it's looked up in
the running config's top-level ``models:`` mapping (e.g. ``models: {sherpa:
sherpa}`` in ``configs/sherpa.yaml``) to find the *backend key*, which must
in turn have a `backends:` entry in that same config. Pass whatever key your
``--config`` file's ``models:`` mapping uses -- for ``configs/mock.yaml``
that's ``mock``.

Writes a timestamped JSON result via ``benchmarks.results.write_result``
named ``accuracy-<model>-<split>-<timestamp>.json`` under
``benchmarks/results/`` (gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import jiwer

from benchmarks._drops import fetch_metrics, guard_drops
from benchmarks.client_file import transcribe_file
from benchmarks.client_ws import UtteranceResult, stream_utterance
from benchmarks.corpus import Utterance, build_manifest, download_subset, load_pcm16
from benchmarks.results import percentiles, write_result
from benchmarks.server import ServerUnderTest

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

# jiwer 4.0.0's `wer()` computes a single corpus-level figure when given
# lists of reference/hypothesis strings: internally it sums (substitutions +
# deletions + insertions) and reference word counts across ALL sentences,
# then divides once -- i.e. it is length-weighted, NOT the mean of each
# utterance's own WER. That distinction matters: a mean-of-per-utterance-WER
# would count a 3-word utterance's 100% miss exactly as heavily as a
# 30-word utterance's 10% miss, wildly over-weighting short utterances
# relative to their actual share of the corpus's words. We always call
# `corpus_wer` once over the full manifest's lists, never averaging
# per-utterance `jiwer.wer` calls.
_WER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level (length-weighted) WER over the full lists, with the same
    normalization (lowercase, strip punctuation, collapse whitespace)
    applied to both `references` and `hypotheses`."""
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )


def aggregate_latency(xs: list[float | None]) -> dict:
    """`percentiles` over the non-`None` entries of `xs` (no-final/error
    utterances record `None` and are excluded, not treated as 0)."""
    return percentiles([x for x in xs if x is not None])


# Early-abort circuit breaker: if this many utterances error CONSECUTIVELY,
# the server is almost certainly dead (crashed mid-run / unreachable) and
# every remaining utterance would just burn its full network timeout — abort
# with a clear error instead of degrading into n recorded errors. Scattered
# (non-consecutive) failures reset the counter, preserving the
# continue-on-error semantics for flaky individual utterances.
MAX_CONSECUTIVE_ERRORS = 5


class ConsecutiveErrorBreaker:
    """Counts consecutive errored utterances; `record(errored)` raises
    `RuntimeError` once `limit` errors occur in a row (a success resets the
    count to zero)."""

    def __init__(self, mode: str, limit: int = MAX_CONSECUTIVE_ERRORS) -> None:
        self.mode = mode
        self.limit = limit
        self.count = 0

    def record(self, errored: bool) -> None:
        if not errored:
            self.count = 0
            return
        self.count += 1
        if self.count >= self.limit:
            raise RuntimeError(
                f"{self.mode} mode: {self.count} consecutive utterances errored; "
                "server looks dead (crashed or unreachable) — aborting run "
                "instead of timing out on every remaining utterance"
            )


async def _run_ws_mode(
    server: ServerUnderTest, model: str, manifest: list[Utterance], pace: float
) -> tuple[list[UtteranceResult], int]:
    """Stream each utterance in the manifest sequentially (concurrency 1)
    over the WS wire protocol. An utterance whose stream raises, or that
    completes with no final transcript, is recorded as `hypothesis=""` and
    counted as an error -- the run continues rather than crashing (unless
    `MAX_CONSECUTIVE_ERRORS` utterances error in a row: see
    `ConsecutiveErrorBreaker`)."""
    results: list[UtteranceResult] = []
    errors = 0
    breaker = ConsecutiveErrorBreaker("ws")
    for utt in manifest:
        pcm16 = load_pcm16(utt)
        try:
            result = await stream_utterance(server.base_ws_url, model, pcm16, pace=pace)
        except Exception:
            result = UtteranceResult(
                utt_id=utt.id,
                hypothesis="",
                server_final_ms=None,
                server_first_partial_ms=None,
                # None despite the field's `float` hint (see the errored-
                # utterance note below): an errored utterance must contribute
                # NOTHING to the latency population.
                client_final_ms=None,  # type: ignore[arg-type]
            )
            errors += 1
            breaker.record(errored=True)
            results.append(result)
            continue
        result.utt_id = utt.id
        errored = not result.hypothesis
        if errored:
            errors += 1
            # No final transcript arrived, so `stream_utterance`'s 0.0
            # `client_final_ms` is a sentinel, not a measurement. Overwrite
            # it with None (intentionally violating the field's `float`
            # hint) so `aggregate_latency` excludes it — a literal 0.0 would
            # be a fake near-zero sample dragging p50/mean and inflating n.
            result.client_final_ms = None  # type: ignore[assignment]
        breaker.record(errored)
        results.append(result)
    return results, errors


async def _run_file_mode(
    server: ServerUnderTest, model: str, manifest: list[Utterance]
) -> tuple[list[dict], int]:
    """POST each utterance to `/v1/audio/transcriptions` sequentially. A
    request that raises, or returns an empty transcript, is recorded as
    `hypothesis=""` and counted as an error -- the run continues (unless
    `MAX_CONSECUTIVE_ERRORS` utterances error in a row: see
    `ConsecutiveErrorBreaker`)."""
    results: list[dict] = []
    errors = 0
    breaker = ConsecutiveErrorBreaker("file")
    for utt in manifest:
        pcm16 = load_pcm16(utt)
        audio_seconds = len(pcm16) / BYTES_PER_SAMPLE / SAMPLE_RATE
        try:
            text, wall_seconds = await transcribe_file(server.base_url, pcm16, model)
            errored = not text
        except Exception:
            text = ""
            errored = True
            errors += 1
        else:
            if errored:
                errors += 1
        if errored:
            # Errored utterances contribute NOTHING to the wall_seconds (and
            # hence RTF) populations: None is filtered out by
            # `aggregate_latency`, whereas a 0.0 sentinel — or even a real
            # wall time for an empty-transcript response — would inject a
            # non-representative sample into the percentiles.
            wall_seconds = None
        results.append(
            {
                "utt_id": utt.id,
                "hypothesis": text,
                "wall_seconds": wall_seconds,
                "audio_seconds": audio_seconds,
            }
        )
        breaker.record(errored)
    return results, errors


def _score_ws(manifest: list[Utterance], results: list[UtteranceResult], errors: int) -> dict:
    references = [utt.ref_text for utt in manifest]
    hyps_by_id = {r.utt_id: r.hypothesis for r in results}
    hypotheses = [hyps_by_id.get(utt.id, "") for utt in manifest]

    return {
        "wer": corpus_wer(references, hypotheses),
        "errors": errors,
        "n": len(manifest),
        "latency_ms": {
            "server_first_partial": aggregate_latency(
                [r.server_first_partial_ms for r in results]
            ),
            "server_final": aggregate_latency([r.server_final_ms for r in results]),
            "client_final": aggregate_latency([r.client_final_ms for r in results]),
        },
    }


def _score_file(manifest: list[Utterance], results: list[dict], errors: int) -> dict:
    references = [utt.ref_text for utt in manifest]
    hyps_by_id = {r["utt_id"]: r["hypothesis"] for r in results}
    hypotheses = [hyps_by_id.get(utt.id, "") for utt in manifest]

    # Errored records carry wall_seconds=None (see _run_file_mode) and so
    # contribute to neither the wall_seconds nor the RTF population.
    rtf_values: list[float | None] = []
    for r in results:
        wall = r["wall_seconds"]
        if r["audio_seconds"] > 0 and r["hypothesis"] and wall is not None and wall > 0:
            rtf_values.append(r["audio_seconds"] / wall)
        else:
            rtf_values.append(None)

    return {
        "wer": corpus_wer(references, hypotheses),
        "errors": errors,
        "n": len(manifest),
        "wall_seconds": aggregate_latency([r["wall_seconds"] for r in results]),
        "rtf": aggregate_latency(rtf_values),
    }


def run(args: argparse.Namespace) -> dict:
    split_dir = download_subset(args.split)
    manifest = build_manifest(split_dir, args.n, args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    payload: dict = {
        "model": args.model,
        "config": args.config,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "pace": args.pace,
        "modes": modes,
        "results": {},
        "errors": 0,
    }

    with ServerUnderTest(args.config, port=args.port, python=args.python) as server:
        if "ws" in modes:
            # Zero-drops guard (benchmarks._drops): at --pace 0 a fast
            # client can outrun a real backend and trip the server's
            # default drop_oldest audio shedding, silently invalidating the
            # WER. Bracket the WS pass with /metrics scrapes; a nonzero
            # stt_audio_dropped_total delta hard-fails the run unless
            # --no-assert-no-drops.
            metrics_before = fetch_metrics(server.base_url)
            ws_results, ws_errors = asyncio.run(
                _run_ws_mode(server, args.model, manifest, args.pace)
            )
            metrics_after = fetch_metrics(server.base_url)
            dropped_delta = guard_drops(metrics_before, metrics_after, args.assert_no_drops)
            payload["results"]["ws"] = _score_ws(manifest, ws_results, ws_errors)
            payload["results"]["ws"]["audio_dropped_delta"] = dropped_delta
            payload["errors"] += ws_errors

        # File mode needs no drops guard: /v1/audio/transcriptions decodes
        # the whole upload and pushes it through its Session as ONE
        # AudioChunk (`run_file_session` in
        # src/stt_server/api/transcriptions_http.py), so the bounded audio
        # queue (maxsize=audio_queue_chunks, counted in CHUNKS not bytes)
        # holds at most that single chunk and can never overflow -- the
        # drop_oldest path is unreachable for this endpoint.
        if "file" in modes:
            file_results, file_errors = asyncio.run(
                _run_file_mode(server, args.model, manifest)
            )
            payload["results"]["file"] = _score_file(manifest, file_results, file_errors)
            payload["errors"] += file_errors

    out_path = write_result(f"accuracy-{args.model}-{args.split}", payload)
    payload["_out_path"] = str(out_path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_accuracy",
        description=(
            "Accuracy (WER) and concurrency-1 latency benchmark against a "
            "LibriSpeech manifest, streamed (WS) and/or file mode."
        ),
    )
    parser.add_argument("--config", required=True, help="stt-server YAML config path")
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name as queried over the wire (e.g. `?model=<this>` for "
            "WS, `model=<this>` form field for file mode). This is looked up "
            "in --config's top-level `models:` mapping to find the backend "
            "key (see `resolve_backend` in src/stt_server/api/app.py) -- pass "
            "whichever key your config's `models:` section uses."
        ),
    )
    parser.add_argument("--split", required=True, choices=["test-clean", "test-other"])
    parser.add_argument("--n", type=int, required=True, help="number of utterances to sample")
    parser.add_argument("--seed", type=int, required=True, help="manifest sampling seed")
    parser.add_argument(
        "--modes", default="ws,file", help="comma-separated subset of {ws,file} (default: both)"
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
        help="WS pacing multiplier (1.0=real-time, 0=as-fast-as-possible)",
    )
    parser.add_argument(
        "--assert-no-drops",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "hard-fail the run if stt_audio_dropped_total increased during the "
            "WS pass (audio was shed; WER would be invalid). "
            "--no-assert-no-drops records the delta without failing."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args)
    print(f"wrote {payload['_out_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
