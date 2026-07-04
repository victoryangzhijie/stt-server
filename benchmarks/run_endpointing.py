"""Endpointing experiment (spec §10.5, "where supported"): server VAD
(Arm A, through the serving system) vs sherpa-onnx NATIVE endpoint
detection (Arm B, direct recognizer, bypassing the server entirely).

This is an OFFLINE experiment script, NOT a new serving mode: the server
pipeline is not modified anywhere by this task -- `SherpaBackend` still
builds its recognizer with `enable_endpoint_detection=False`
(`src/stt_server/backends/sherpa/backend.py`) and Arm A streams through that
same unmodified server. Wiring native endpointing into the actual serving
path (replacing/augmenting `Endpointer` + VAD for backends that support it)
is explicitly future work -- this script only measures how the two
detection mechanisms COMPARE, in isolation, on the same padded audio.

sherpa-onnx is the only local backend with native endpoint detection
(`enable_endpoint_detection=True` on `OnlineRecognizer.from_transducer`,
plus rule-based endpoint config) -- this experiment is sherpa-only.

Both arms are fed the SAME padded audio: each manifest utterance gets 1s of
true (digital-zero) leading silence and 1s of trailing silence
(`pad_with_silence`), which gives an exact, arm-independent ground truth for
"where does true speech end" -- both arms' endpoint-detection latency is
measured against that same timestamp, never against anything either arm
itself reports.

Design note -- Arm B execution (in-process, not a subprocess worker):
Arm B imports `sherpa_onnx` directly IN THIS SAME PROCESS (no `--arm
b-worker` subprocess split). That is the simplest design that keeps the CLI
a single command, at the cost that the WHOLE script -- not just the
server-under-test `ServerUnderTest` spawns for Arm A -- must run under an
interpreter that has `sherpa_onnx` (plus the `bench` extra's `jiwer`/
`soundfile`) installed. `sherpa_onnx` is still imported LAZILY (inside
`_build_sherpa_recognizer`), so importing this module, and running its pure
functions (segmentation counting, latency arithmetic, `pad_with_silence`),
never requires the package -- only actually running Arm B does. Concretely:
run this whole script with the sherpa venv's python (see
`benchmarks/README.md` for the exact recipe), and pass that SAME
interpreter to `--python` so Arm A's spawned server process (which also
needs `sherpa_onnx`) uses it too.

Usage::

    <sherpa-venv>/bin/python -m benchmarks.run_endpointing \\
        --config configs/sherpa.yaml --model sherpa-zipformer-en \\
        --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \\
        --split test-clean --n 25 --seed 42 \\
        --python <sherpa-venv>/bin/python \\
        [--port 8100] [--pace 1.0] [--lead-s 1.0] [--trail-s 1.0] \\
        [--assert-no-drops | --no-assert-no-drops]

Arm A validity guard: `--pace` defaults to 1.0 (REAL-TIME streaming) and
the run scrapes `/metrics` around Arm A, hard-failing (unless
`--no-assert-no-drops`) if `stt_audio_dropped_total` increased -- at
`--pace 0` a fast client can outrun a real backend's decode and trip the
server's default `drop_oldest` audio shedding (`configs/sherpa.yaml` has no
`limits:` section), silently turning Arm A's WER into a measurement of
accuracy-under-audio-drop (found by direct reproduction in Task 7 review).

Writes `endpointing-sherpa-<timestamp>.json` via
`benchmarks.results.write_result`: `arm_a` / `arm_b` blocks (each `{"wer":
..., "n": ..., "segmentation": {"under"/"correct"/"over": n},
<latency-key>: {...percentiles...}}`; `arm_a` also records
`audio_dropped_delta`, `arm_b` also records `premature_fires` and a
`per_utterance` list of raw `fire_times_ms`/`true_speech_end_ms`) plus a
`comparison` block.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass

from benchmarks._drops import fetch_metrics, guard_drops
from benchmarks.client_ws import UtteranceResult, stream_utterance
from benchmarks.corpus import Utterance, build_manifest, download_subset, load_pcm16
from benchmarks.results import percentiles, write_result
from benchmarks.run_accuracy import ConsecutiveErrorBreaker, aggregate_latency, corpus_wer
from benchmarks.server import ServerUnderTest

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_MS = 100

# ---------------------------------------------------------------------------
# Pure functions: padding construction, segmentation counting, endpoint-
# latency arithmetic. Unit-tested on synthetic timelines in
# tests/benchmarks/test_endpointing_metrics.py -- no venv/model/server
# dependency.
# ---------------------------------------------------------------------------


def pad_with_silence(
    pcm16: bytes,
    lead_s: float = 1.0,
    trail_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[bytes, float, float]:
    """Pad `pcm16` (16-bit little-endian mono PCM) with `lead_s` seconds of
    leading TRUE silence (digital zero, not just quiet noise) and `trail_s`
    seconds of trailing true silence. Returns `(padded_pcm16,
    true_speech_start_ms, true_speech_end_ms)`: the boundaries of the real
    speech inside the padded buffer, computed purely from the padding
    construction (`lead_s` and the original buffer's own duration) --
    independent of anything either arm detects. Both arms' endpoint-
    detection latency is measured against `true_speech_end_ms`.

    `true_speech_end_ms == true_speech_start_ms` when `pcm16` is empty (a
    degenerate zero-length "speech" span, not an error)."""
    lead_bytes = int(sample_rate * lead_s) * BYTES_PER_SAMPLE
    trail_bytes = int(sample_rate * trail_s) * BYTES_PER_SAMPLE
    padded = (b"\x00" * lead_bytes) + pcm16 + (b"\x00" * trail_bytes)

    true_start_ms = lead_s * 1000.0
    speech_duration_ms = (len(pcm16) / BYTES_PER_SAMPLE / sample_rate) * 1000.0
    true_end_ms = true_start_ms + speech_duration_ms
    return padded, true_start_ms, true_end_ms


def segmentation_label(n_segments: int) -> str:
    """Classify a per-utterance segment count against the expected
    exactly-one-segment-per-padded-utterance ground truth: `0` -> `"under"`
    (the arm never finalized/endpointed at all -- the utterance was
    swallowed), `1` -> `"correct"`, anything greater -> `"over"` (the
    utterance was split into more pieces than the padding construction
    intends -- exactly one isolated speech span per utterance)."""
    if n_segments == 0:
        return "under"
    if n_segments == 1:
        return "correct"
    return "over"


def segmentation_counts(n_segments_list: list[int]) -> dict:
    """Tally `segmentation_label` over a whole manifest run:
    `{"under": n, "correct": n, "over": n}` -- always all three keys
    present, 0 for any category with no members."""
    counts = {"under": 0, "correct": 0, "over": 0}
    for n in n_segments_list:
        counts[segmentation_label(n)] += 1
    return counts


def endpoint_fire_latency_ms(fire_time_ms: float, true_speech_end_ms: float) -> float:
    """`fire_time_ms - true_speech_end_ms`: the audio-time gap between an
    endpoint FIRING (a server FINAL event's `audio_time_ms`, or sherpa's
    `is_endpoint(stream)` going true) and the KNOWN true end of speech (from
    `pad_with_silence`'s construction). POSITIVE means the detector waited
    past the true speech end before firing -- expected, since every real
    endpointer needs some trailing-silence evidence before committing; a
    large NEGATIVE value would mean it fired mid-speech."""
    return fire_time_ms - true_speech_end_ms


def bucket_fire_latencies(
    fire_times_ms: list[float], true_speech_end_ms: float
) -> tuple[list[float], list[float]]:
    """Split an utterance's fires into two latency populations:
    `(premature, post_speech_end)`. A PREMATURE fire (`fire_time <
    true_speech_end_ms`) happened while true speech was still playing -- it
    is a MID-UTTERANCE fire (e.g. sherpa's rule2 firing on an internal
    pause long enough to satisfy its trailing-silence threshold), NOT a
    faster detection of the utterance's end; reporting its (negative)
    latency in the same percentile population as genuine end-detections
    would corrupt the stat. A POST-speech-end fire (`fire_time >=
    true_speech_end_ms`, boundary inclusive: firing exactly AT the true end
    is a 0-latency detection) is a genuine end-detection sample.

    Both lists hold `endpoint_fire_latency_ms` values (fire - true end), in
    fire order. Empty `fire_times_ms` (under-segmentation) yields
    `([], [])` -- no fabricated samples in either bucket."""
    premature: list[float] = []
    post: list[float] = []
    for fire_ms in fire_times_ms:
        latency = endpoint_fire_latency_ms(fire_ms, true_speech_end_ms)
        if fire_ms < true_speech_end_ms:
            premature.append(latency)
        else:
            post.append(latency)
    return premature, post


# The no-drops guard (fetch, delta, decision, error message) lives in
# benchmarks._drops -- it originated here (Task 7 review) and was factored
# out so run_accuracy / run_stabilizer_study share the identical semantics.


# ---------------------------------------------------------------------------
# Arm A: through the server (native WS, unmodified `configs/sherpa.yaml`
# pipeline -- server-side VAD + Endpointer decide segmentation).
# ---------------------------------------------------------------------------


async def _run_arm_a(
    server: ServerUnderTest,
    model: str,
    manifest: list[Utterance],
    padded_by_id: dict[str, tuple[bytes, float, float]],
    pace: float,
) -> list[UtteranceResult]:
    """Stream each manifest utterance's PADDED audio over `/ws/transcribe`
    (concurrency 1, same shape as `run_accuracy._run_ws_mode`). An utterance
    whose stream raises is recorded as `hypothesis=""` with no finals (so it
    scores as `segmentation_label(0) == "under"`, and contributes nothing to
    the final-latency population) -- the run continues rather than crashing,
    unless `MAX_CONSECUTIVE_ERRORS` error in a row."""
    results: list[UtteranceResult] = []
    breaker = ConsecutiveErrorBreaker("endpointing-arm-a")
    for utt in manifest:
        padded_pcm16, _true_start_ms, _true_end_ms = padded_by_id[utt.id]
        try:
            result = await stream_utterance(server.base_ws_url, model, padded_pcm16, pace=pace)
        except Exception:
            result = UtteranceResult(
                utt_id=utt.id,
                hypothesis="",
                server_final_ms=None,
                server_first_partial_ms=None,
                client_final_ms=None,  # type: ignore[arg-type]
            )
            breaker.record(errored=True)
            results.append(result)
            continue
        result.utt_id = utt.id
        breaker.record(errored=not result.hypothesis)
        results.append(result)
    return results


def _score_arm_a(manifest: list[Utterance], results: list[UtteranceResult]) -> dict:
    references = [utt.ref_text for utt in manifest]
    hyps_by_id = {r.utt_id: r.hypothesis for r in results}
    hypotheses = [hyps_by_id.get(utt.id, "") for utt in manifest]

    n_segments_by_id = {r.utt_id: len(r.finals) for r in results}
    n_segments_list = [n_segments_by_id.get(utt.id, 0) for utt in manifest]
    latency_by_id = {r.utt_id: r.server_final_ms for r in results}
    final_latencies = [latency_by_id.get(utt.id) for utt in manifest]

    return {
        "wer": corpus_wer(references, hypotheses),
        "n": len(manifest),
        "segmentation": segmentation_counts(n_segments_list),
        "final_latency_ms": aggregate_latency(final_latencies),
    }


# ---------------------------------------------------------------------------
# Arm B: direct recognizer, native endpointing (bypasses the server, and the
# Session/VAD/Endpointer/Stabilizer pipeline, entirely).
# ---------------------------------------------------------------------------


def _build_sherpa_recognizer(model_dir: str, num_threads: int = 4):
    """Build an `OnlineRecognizer` with NATIVE endpoint detection enabled and
    the package's DEFAULT endpoint rules. `sherpa_onnx` is imported lazily
    here (not at module scope) so this module remains importable -- and its
    pure functions testable -- without the package installed.

    Verified via `inspect.signature(sherpa_onnx.OnlineRecognizer.from_transducer)`
    on the installed `sherpa-onnx==1.10.46` package: the real kwarg names are
    `rule1_min_trailing_silence` (default 2.4s), `rule2_min_trailing_silence`
    (default 1.2s), `rule3_min_utterance_length` (default 20.0s) -- passed
    here only implicitly (left at their defaults), which is what "default
    endpoint rules" means for this experiment. These are the SAME kwargs
    `SherpaBackend._build_recognizer` (`src/stt_server/backends/sherpa/backend.py`)
    leaves off (that backend hard-codes `enable_endpoint_detection=False`);
    this experiment is the only place that flips it on.

    Reuses `SherpaBackend`'s model-file globbing (`_find_one`, which prefers
    non-`.int8.` weights) rather than duplicating it -- importing
    `stt_server.backends.sherpa.backend` does NOT itself require
    `sherpa_onnx` (it's lazy there too), so this stays safe to import from a
    sherpa_onnx-free interpreter.
    """
    import sherpa_onnx

    from stt_server.backends.sherpa.backend import _find_one

    tokens = os.path.join(model_dir, "tokens.txt")
    encoder = _find_one(model_dir, "encoder-*.onnx")
    decoder = _find_one(model_dir, "decoder-*.onnx")
    joiner = _find_one(model_dir, "joiner-*.onnx")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
    )


@dataclass
class ArmBResult:
    utt_id: str
    hypothesis: str
    fire_times_ms: list[float]
    true_speech_end_ms: float


def run_arm_b_utterance(
    recognizer, padded_pcm16: bytes, true_speech_end_ms: float, chunk_ms: int = CHUNK_MS
) -> tuple[list[float], str]:
    """Feed `padded_pcm16` to a fresh `recognizer.create_stream()` in
    `chunk_ms`-sized steps (mirrors real streaming: sherpa's endpoint rules
    are trailing-silence-duration-based, so feeding the whole buffer in one
    shot would never let a rule "see" incremental silence accrue).

    After each chunk is accepted and decoded to exhaustion
    (`is_ready`/`decode_stream`), checks `recognizer.is_endpoint(stream)`.
    On EVERY fire: records the audio time fed so far (including the firing
    chunk) and the text at that instant (`get_result`), then
    `recognizer.reset(stream)` -- sherpa's `is_endpoint` stays true forever
    on a stream that is never reset, so without this every subsequent chunk
    would also register as a fire.

    After all audio is fed, `stream.input_finished()` + one more decode
    drain flushes any trailing partial decode state; that flush is NOT
    itself scored as a fire (only `is_endpoint` firings count as segments --
    matching what a real endpointing consumer reacts to), but its text is
    used as a fallback hypothesis when the stream never fired at all
    (under-segmentation: still score whatever transcript accrued for WER,
    since a garbled/missing transcript is itself the failure this arm's WER
    should reflect).

    Returns `(fire_times_ms, hypothesis)`: `hypothesis` is the text at the
    FIRST fire when one occurred (the padded construction means everything
    after `true_speech_end_ms` is silence, so the first fire's text already
    holds the full utterance), else the post-`input_finished()` flush text.
    """
    from stt_server.backends._audio import pcm16_bytes_to_float32

    stream = recognizer.create_stream()
    fire_times_ms: list[float] = []
    texts_at_fire: list[str] = []
    audio_ms = 0.0
    chunk_bytes = max(1, SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_ms // 1000)

    for i in range(0, len(padded_pcm16), chunk_bytes):
        chunk = padded_pcm16[i : i + chunk_bytes]
        samples = pcm16_bytes_to_float32(chunk)
        stream.accept_waveform(SAMPLE_RATE, samples)
        audio_ms += (len(chunk) / BYTES_PER_SAMPLE / SAMPLE_RATE) * 1000.0
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        if recognizer.is_endpoint(stream):
            fire_times_ms.append(audio_ms)
            texts_at_fire.append(recognizer.get_result(stream))
            recognizer.reset(stream)

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    flush_text = recognizer.get_result(stream)

    hypothesis = texts_at_fire[0] if texts_at_fire else flush_text
    return fire_times_ms, hypothesis


def _run_arm_b(
    recognizer,
    manifest: list[Utterance],
    padded_by_id: dict[str, tuple[bytes, float, float]],
) -> list[ArmBResult]:
    results: list[ArmBResult] = []
    for utt in manifest:
        padded_pcm16, _true_start_ms, true_end_ms = padded_by_id[utt.id]
        fire_times_ms, hypothesis = run_arm_b_utterance(recognizer, padded_pcm16, true_end_ms)
        results.append(
            ArmBResult(
                utt_id=utt.id,
                hypothesis=hypothesis,
                fire_times_ms=fire_times_ms,
                true_speech_end_ms=true_end_ms,
            )
        )
    return results


def _score_arm_b(manifest: list[Utterance], results: list[ArmBResult]) -> dict:
    references = [utt.ref_text for utt in manifest]
    hyps_by_id = {r.utt_id: r.hypothesis for r in results}
    hypotheses = [hyps_by_id.get(utt.id, "") for utt in manifest]

    fires_by_id = {r.utt_id: r for r in results}
    n_segments_list = [len(fires_by_id[utt.id].fire_times_ms) for utt in manifest]

    # Bucket every fire (across the whole manifest) into premature vs
    # post-speech-end populations (see bucket_fire_latencies): only
    # post-speech-end fires are genuine end-detections, so ONLY that bucket
    # feeds the latency percentiles; premature (mid-utterance) fires are
    # reported as a separate COUNT -- a negative latency is a mid-utterance
    # fire, not a faster detection, and must never lower the percentiles.
    premature_all: list[float] = []
    post_all: list[float] = []
    per_utterance: list[dict] = []
    for utt in manifest:
        r = fires_by_id[utt.id]
        premature, post = bucket_fire_latencies(r.fire_times_ms, r.true_speech_end_ms)
        premature_all.extend(premature)
        post_all.extend(post)
        # Raw per-utterance fire data, so anomalies (e.g. a premature fire
        # on a mid-utterance pause) are diagnosable from the JSON alone.
        per_utterance.append(
            {
                "utt_id": utt.id,
                "fire_times_ms": r.fire_times_ms,
                "true_speech_end_ms": r.true_speech_end_ms,
            }
        )

    return {
        "wer": corpus_wer(references, hypotheses),
        "n": len(manifest),
        "segmentation": segmentation_counts(n_segments_list),
        "endpoint_latency_ms": percentiles(post_all),
        "premature_fires": len(premature_all),
        "per_utterance": per_utterance,
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    split_dir = download_subset(args.split)
    manifest = build_manifest(split_dir, args.n, args.seed)

    padded_by_id: dict[str, tuple[bytes, float, float]] = {}
    for utt in manifest:
        pcm16 = load_pcm16(utt)
        padded_by_id[utt.id] = pad_with_silence(pcm16, lead_s=args.lead_s, trail_s=args.trail_s)

    with ServerUnderTest(args.config, port=args.port, python=args.python) as server:
        # Bracket Arm A with /metrics scrapes: a nonzero
        # stt_audio_dropped_total delta means the server SHED audio during
        # the run (drop_oldest backpressure -- the default policy when the
        # config has no limits: section) and Arm A's WER would measure
        # accuracy-under-drop, not endpointing quality. See
        # benchmarks._drops for the review finding behind this.
        metrics_before = fetch_metrics(server.base_url)
        arm_a_results = asyncio.run(
            _run_arm_a(server, args.model, manifest, padded_by_id, args.pace)
        )
        metrics_after = fetch_metrics(server.base_url)
    dropped_delta = guard_drops(metrics_before, metrics_after, args.assert_no_drops)
    arm_a_score = _score_arm_a(manifest, arm_a_results)
    arm_a_score["audio_dropped_delta"] = dropped_delta

    recognizer = _build_sherpa_recognizer(args.model_dir, args.num_threads)
    arm_b_results = _run_arm_b(recognizer, manifest, padded_by_id)
    arm_b_score = _score_arm_b(manifest, arm_b_results)

    payload: dict = {
        "model": args.model,
        "config": args.config,
        "model_dir": args.model_dir,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "lead_s": args.lead_s,
        "trail_s": args.trail_s,
        "arm_a": arm_a_score,
        "arm_b": arm_b_score,
        "comparison": {
            "wer_delta_arm_b_minus_arm_a": arm_b_score["wer"] - arm_a_score["wer"],
            "segmentation_arm_a": arm_a_score["segmentation"],
            "segmentation_arm_b": arm_b_score["segmentation"],
        },
    }

    out_path = write_result("endpointing-sherpa", payload)
    payload["_out_path"] = str(out_path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run_endpointing",
        description=(
            "Endpointing experiment (spec §10.5): server VAD (Arm A, through "
            "the server) vs sherpa-onnx native endpoint detection (Arm B, "
            "direct recognizer). Run this WHOLE script under the sherpa "
            "venv's python -- Arm B imports sherpa_onnx in-process; see "
            "benchmarks/README.md for the exact venv recipe."
        ),
    )
    parser.add_argument("--config", required=True, help="stt-server YAML config path (Arm A)")
    parser.add_argument(
        "--model",
        required=True,
        help="model name queried over the wire for Arm A (see run_accuracy.py for the lookup rule)",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="sherpa streaming Zipformer model directory (Arm B, direct recognizer)",
    )
    parser.add_argument("--split", required=True, choices=["test-clean", "test-other"])
    parser.add_argument("--n", type=int, required=True, help="number of utterances to sample")
    parser.add_argument("--seed", type=int, required=True, help="manifest sampling seed")
    parser.add_argument(
        "--port", type=int, default=8100, help="port for the spawned server (Arm A)"
    )
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter to run the Arm A server with (default: sys.executable)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help=(
            "Arm A WS pacing multiplier (default 1.0=real-time; 0=as-fast-as-"
            "possible -- WARNING: pace 0 against a real backend can outrun the "
            "decoder and trip the server's drop_oldest audio shedding, which "
            "--assert-no-drops (default on) will catch and abort)"
        ),
    )
    parser.add_argument(
        "--assert-no-drops",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "hard-fail the run if stt_audio_dropped_total increased during Arm A "
            "(audio was shed; WER would be invalid). --no-assert-no-drops records "
            "the delta without failing."
        ),
    )
    parser.add_argument(
        "--lead-s", type=float, default=1.0, help="leading silence padding, seconds (both arms)"
    )
    parser.add_argument(
        "--trail-s", type=float, default=1.0, help="trailing silence padding, seconds (both arms)"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Arm B recognizer num_threads"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args)
    print(f"wrote {payload['_out_path']}")


if __name__ == "__main__":
    main(sys.argv[1:])
