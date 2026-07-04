"""Unit tests for the pure segmentation-counting + endpoint-latency
arithmetic behind `run_endpointing.py` (spec §10.5). Everything here is
hand-built synthetic event timelines -- no venv, no model, no server.

Guarded with `pytest.importorskip("jiwer")` (like `test_accuracy_scoring.py`)
because `run_endpointing` imports `benchmarks.run_accuracy` at module scope
(for `ConsecutiveErrorBreaker`/`corpus_wer`), which itself imports `jiwer`."""

from __future__ import annotations

import pytest

jiwer = pytest.importorskip("jiwer")

from benchmarks._drops import no_drops_failure_message  # noqa: E402
from benchmarks.run_endpointing import (  # noqa: E402
    bucket_fire_latencies,
    endpoint_fire_latency_ms,
    pad_with_silence,
    segmentation_counts,
    segmentation_label,
)

# -- pad_with_silence --


def test_pad_with_silence_default_one_second_each_side():
    # 1 second of 16 kHz mono PCM16 speech (arbitrary non-zero bytes).
    speech = b"\x01\x02" * 16000
    padded, true_start_ms, true_end_ms = pad_with_silence(speech)

    # 1s lead + 1s speech + 1s trail = 3s of PCM16 @16kHz mono = 3 * 32000 bytes.
    assert len(padded) == 3 * 16000 * 2
    assert padded[: 16000 * 2] == b"\x00" * (16000 * 2)  # lead is true silence
    assert padded[16000 * 2 : 16000 * 2 + len(speech)] == speech
    assert padded[16000 * 2 + len(speech) :] == b"\x00" * (16000 * 2)
    assert true_start_ms == 1000.0
    assert true_end_ms == 2000.0  # 1s lead + 1s speech


def test_pad_with_silence_custom_lead_and_trail():
    speech = b"\x01\x02" * 8000  # 0.5s @16kHz mono PCM16
    padded, true_start_ms, true_end_ms = pad_with_silence(speech, lead_s=0.25, trail_s=0.75)

    lead_bytes = int(16000 * 0.25) * 2
    trail_bytes = int(16000 * 0.75) * 2
    assert len(padded) == lead_bytes + len(speech) + trail_bytes
    assert true_start_ms == 250.0
    assert true_end_ms == 750.0  # 250ms lead + 500ms speech


def test_pad_with_silence_empty_speech():
    padded, true_start_ms, true_end_ms = pad_with_silence(b"", lead_s=1.0, trail_s=1.0)

    assert len(padded) == 2 * 16000 * 2  # just the two silence halves
    assert true_start_ms == 1000.0
    assert true_end_ms == 1000.0  # zero-length speech: start == end


# -- segmentation_label / segmentation_counts --


def test_segmentation_label_zero_is_under():
    assert segmentation_label(0) == "under"


def test_segmentation_label_one_is_correct():
    assert segmentation_label(1) == "correct"


def test_segmentation_label_more_than_one_is_over():
    assert segmentation_label(2) == "over"
    assert segmentation_label(5) == "over"


def test_segmentation_counts_tallies_all_three_categories():
    counts = segmentation_counts([0, 1, 1, 1, 2, 3, 0])

    assert counts == {"under": 2, "correct": 3, "over": 2}


def test_segmentation_counts_empty_list_is_all_zero():
    assert segmentation_counts([]) == {"under": 0, "correct": 0, "over": 0}


def test_segmentation_counts_all_correct():
    assert segmentation_counts([1, 1, 1]) == {"under": 0, "correct": 3, "over": 0}


# -- endpoint_fire_latency_ms / bucket_fire_latencies --


def test_endpoint_fire_latency_positive_when_fire_after_true_end():
    # Fired 400ms of audio-time after the true speech end -- the detector
    # needed 400ms of trailing silence evidence before committing.
    assert endpoint_fire_latency_ms(2400.0, 2000.0) == pytest.approx(400.0)


def test_endpoint_fire_latency_zero_when_fire_exactly_at_true_end():
    assert endpoint_fire_latency_ms(2000.0, 2000.0) == 0.0


def test_endpoint_fire_latency_negative_when_fire_before_true_end():
    # A detector that (incorrectly) fired mid-speech.
    assert endpoint_fire_latency_ms(1800.0, 2000.0) == pytest.approx(-200.0)


def test_bucket_fire_latencies_splits_premature_from_post():
    # One fire mid-speech (1800 < 2000: premature, latency -200) and one
    # after the true end (2400: post-speech-end, latency +400).
    premature, post = bucket_fire_latencies([1800.0, 2400.0], 2000.0)
    assert premature == [pytest.approx(-200.0)]
    assert post == [pytest.approx(400.0)]


def test_bucket_fire_latencies_fire_exactly_at_true_end_is_post():
    # Boundary: a fire AT the true speech end is a (0-latency) detection of
    # the end, not a mid-speech fire.
    premature, post = bucket_fire_latencies([2000.0], 2000.0)
    assert premature == []
    assert post == [0.0]


def test_bucket_fire_latencies_no_fires():
    assert bucket_fire_latencies([], 2000.0) == ([], [])


def test_bucket_fire_latencies_all_premature():
    premature, post = bucket_fire_latencies([500.0, 1200.0], 2000.0)
    assert premature == [pytest.approx(-1500.0), pytest.approx(-800.0)]
    assert post == []


def test_bucket_fire_latencies_multiple_post_fires_all_kept():
    # Over-segmentation after the true end: both post fires are samples of
    # the post-speech-end population (counted as "over" by
    # segmentation_counts separately).
    premature, post = bucket_fire_latencies([2100.0, 2600.0], 2000.0)
    assert premature == []
    assert post == [pytest.approx(100.0), pytest.approx(600.0)]


# -- no_drops_failure_message (audio-shedding guard for Arm A) --


def test_no_drops_zero_delta_passes():
    assert no_drops_failure_message(0.0, assert_no_drops=True) is None


def test_no_drops_nonzero_delta_fails_with_actionable_message():
    msg = no_drops_failure_message(165.0, assert_no_drops=True)
    assert msg is not None
    assert "165" in msg
    # The message must name the remedies: pacing, queue size, overflow policy.
    assert "--pace" in msg
    assert "audio_queue_chunks" in msg
    assert "audio_overflow_policy" in msg


def test_no_drops_nonzero_delta_with_assertion_disabled_passes():
    # --no-assert-no-drops: the delta is still RECORDED in the payload, but
    # it no longer hard-fails the run.
    assert no_drops_failure_message(165.0, assert_no_drops=False) is None


def test_no_drops_zero_delta_with_assertion_disabled_passes():
    assert no_drops_failure_message(0.0, assert_no_drops=False) is None
