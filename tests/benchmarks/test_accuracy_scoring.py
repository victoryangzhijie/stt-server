"""Unit tests for `benchmarks/run_accuracy.py` scoring/aggregation. No
network, no server, no model — these exercise only `corpus_wer` and the
latency/RTF aggregation helpers on fabricated data.

Requires the `bench` extra (jiwer); guarded with `pytest.importorskip` so an
ML-free CI env without `--extra bench` stays green."""

from __future__ import annotations

import pytest

jiwer = pytest.importorskip("jiwer")

from benchmarks.client_ws import UtteranceResult  # noqa: E402
from benchmarks.corpus import Utterance  # noqa: E402
from benchmarks.run_accuracy import (  # noqa: E402
    ConsecutiveErrorBreaker,
    _score_file,
    _score_ws,
    aggregate_latency,
    corpus_wer,
)


def test_corpus_wer_single_pair_hand_computed() -> None:
    # ref "the cat sat" (3 words) vs hyp "the cat sat on" (4 words): the
    # alignment is 3 correct words + 1 insertion ("on"), 0 substitutions, 0
    # deletions. WER = (S + D + I) / N_ref = (0 + 0 + 1) / 3 = 1/3.
    wer = corpus_wer(["the cat sat"], ["the cat sat on"])

    assert wer == pytest.approx(1 / 3)


def test_corpus_wer_is_length_weighted_not_mean_of_means() -> None:
    # Utterance A: ref "a b c" (3 words), hyp "" (all 3 deleted) -> WER 1.0.
    # Utterance B: ref "a b c d e f g h i j" (10 words), hyp identical ->
    # WER 0.0.
    # Mean-of-per-utterance-WER would give (1.0 + 0.0) / 2 = 0.5.
    # Corpus-level (length-weighted) WER is total errors / total ref words:
    # (3 + 0) / (3 + 10) = 3/13 ~= 0.2308 -- the two utterances are NOT
    # weighted equally; the 10-word utterance's perfect score dominates.
    references = ["a b c", "a b c d e f g h i j"]
    hypotheses = ["", "a b c d e f g h i j"]

    wer = corpus_wer(references, hypotheses)

    assert wer == pytest.approx(3 / 13)
    mean_of_means = (1.0 + 0.0) / 2
    assert wer != pytest.approx(mean_of_means)


def test_corpus_wer_applies_same_normalization_to_both_sides() -> None:
    # Case, punctuation, and spacing differences that vanish under the
    # shared ToLowerCase/RemovePunctuation/RemoveMultipleSpaces/Strip
    # transform should not count as errors.
    wer = corpus_wer(["The Cat, Sat!"], ["the   cat sat"])

    assert wer == pytest.approx(0.0)


def test_corpus_wer_all_substitutions_is_one() -> None:
    # Same word count on both sides, every word wrong: 3 substitutions,
    # 0 deletions/insertions -> WER = 3/3 = 1.0.
    wer = corpus_wer(["the cat sat"], ["completely different words"])

    assert wer == pytest.approx(1.0)


def test_aggregate_latency_shape() -> None:
    result = aggregate_latency([100.0, 200.0, None, 300.0])

    # None entries (no-final/error utterances) are excluded from the
    # percentile computation but shouldn't crash it.
    assert result["n"] == 3
    assert result["p50"] == 200.0


def test_aggregate_latency_all_none() -> None:
    result = aggregate_latency([None, None])

    assert result == {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}


def test_aggregate_latency_empty() -> None:
    result = aggregate_latency([])

    assert result["n"] == 0


def _utt(i: int, ref: str) -> Utterance:
    from pathlib import Path

    return Utterance(
        id=f"1-2-{i:04d}", flac_path=Path(f"/nonexistent/{i}.flac"), ref_text=ref, duration_s=1.0
    )


def test_score_ws_excludes_errored_utterance_from_latency_populations() -> None:
    # Two successes + one errored record (the shape _run_ws_mode produces on
    # a per-utterance exception: empty hypothesis, all latencies None —
    # including client_final_ms despite its `float` hint).
    manifest = [_utt(0, "a b"), _utt(1, "c d"), _utt(2, "e f")]
    results = [
        UtteranceResult(
            utt_id="1-2-0000", hypothesis="a b", server_final_ms=100.0,
            server_first_partial_ms=50.0, client_final_ms=200.0,
        ),
        UtteranceResult(
            utt_id="1-2-0001", hypothesis="c d", server_final_ms=300.0,
            server_first_partial_ms=150.0, client_final_ms=400.0,
        ),
        UtteranceResult(  # errored: contributes to NO latency population
            utt_id="1-2-0002", hypothesis="", server_final_ms=None,
            server_first_partial_ms=None, client_final_ms=None,  # type: ignore[arg-type]
        ),
    ]

    score = _score_ws(manifest, results, errors=1)

    assert score["errors"] == 1
    assert score["n"] == 3  # utterance count, not sample count
    lat = score["latency_ms"]
    # Every latency population has exactly the 2 successful samples — no
    # fake near-zero entry from the errored utterance.
    assert lat["server_first_partial"]["n"] == 2
    assert lat["server_final"]["n"] == 2
    assert lat["client_final"]["n"] == 2
    assert lat["client_final"]["p50"] == 200.0
    assert lat["client_final"]["mean"] == pytest.approx(300.0)
    # WER still scores the errored utterance as an empty hypothesis: 2 of 6
    # reference words deleted -> 2/6.
    assert score["wer"] == pytest.approx(2 / 6)


def test_score_file_excludes_errored_utterance_from_wall_and_rtf() -> None:
    manifest = [_utt(0, "a b"), _utt(1, "c d"), _utt(2, "e f")]
    results = [
        {"utt_id": "1-2-0000", "hypothesis": "a b", "wall_seconds": 0.5, "audio_seconds": 1.0},
        {"utt_id": "1-2-0001", "hypothesis": "c d", "wall_seconds": 0.25, "audio_seconds": 1.0},
        # Errored record as _run_file_mode produces it: wall_seconds=None.
        {"utt_id": "1-2-0002", "hypothesis": "", "wall_seconds": None, "audio_seconds": 1.0},
    ]

    score = _score_file(manifest, results, errors=1)

    assert score["errors"] == 1
    assert score["n"] == 3
    # wall_seconds and RTF populations contain only the 2 real samples.
    assert score["wall_seconds"]["n"] == 2
    assert score["wall_seconds"]["mean"] == pytest.approx(0.375)
    assert score["rtf"]["n"] == 2
    # RTF = audio/wall: 1.0/0.5 = 2.0 and 1.0/0.25 = 4.0.
    assert score["rtf"]["p50"] == pytest.approx(2.0)
    assert score["rtf"]["mean"] == pytest.approx(3.0)
    assert score["wer"] == pytest.approx(2 / 6)


def test_consecutive_error_breaker_trips_at_limit() -> None:
    breaker = ConsecutiveErrorBreaker("ws", limit=3)

    breaker.record(errored=True)
    breaker.record(errored=True)
    with pytest.raises(RuntimeError, match="3 consecutive"):
        breaker.record(errored=True)


def test_consecutive_error_breaker_resets_on_success() -> None:
    # Scattered failures never trip the breaker: a success resets the count,
    # preserving continue-on-error semantics for flaky individual utterances.
    breaker = ConsecutiveErrorBreaker("file", limit=3)

    for _ in range(10):
        breaker.record(errored=True)
        breaker.record(errored=True)
        breaker.record(errored=False)  # reset before hitting the limit

    breaker.record(errored=True)  # count restarts at 1, no raise
