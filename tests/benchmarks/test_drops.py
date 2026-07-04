"""Unit tests for the shared zero-drops guard (`benchmarks._drops`) wiring
helpers: the delta arithmetic and the raise-or-return `guard_drops`
one-liner every runner calls. The pure decision function
(`no_drops_failure_message`) is covered in `test_endpointing_metrics.py`
(where the guard originated), and the Prometheus parser in
`test_load_logic.py`. `benchmarks._drops` is stdlib-only, so this file
needs no `importorskip` guard."""

from __future__ import annotations

import pytest
from benchmarks._drops import audio_dropped_delta, guard_drops


def test_audio_dropped_delta_simple_increase():
    before = {"stt_audio_dropped_total": 3.0}
    after = {"stt_audio_dropped_total": 168.0}
    assert audio_dropped_delta(before, after) == pytest.approx(165.0)


def test_audio_dropped_delta_missing_family_counts_as_zero():
    # prometheus_client emits no sample line until a counter first
    # increments -- a scrape without the family means 0, on either side.
    assert audio_dropped_delta({}, {"stt_audio_dropped_total": 5.0}) == 5.0
    assert audio_dropped_delta({}, {}) == 0.0


def test_guard_drops_returns_zero_delta_without_raising():
    m = {"stt_audio_dropped_total": 7.0}
    assert guard_drops(m, m, assert_no_drops=True) == 0.0


def test_guard_drops_raises_on_nonzero_delta_when_armed():
    before = {"stt_audio_dropped_total": 0.0}
    after = {"stt_audio_dropped_total": 165.0}
    with pytest.raises(RuntimeError, match="165"):
        guard_drops(before, after, assert_no_drops=True)


def test_guard_drops_disarmed_returns_nonzero_delta_for_recording():
    # --no-assert-no-drops: no raise, but the delta is still RETURNED so the
    # runner records it in the result payload.
    before = {"stt_audio_dropped_total": 0.0}
    after = {"stt_audio_dropped_total": 42.0}
    assert guard_drops(before, after, assert_no_drops=False) == 42.0
