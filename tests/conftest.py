"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from stt_server.metrics.registry import (
    AUDIO_DROPPED,
    AUDIO_SECONDS,
    ERRORS,
    FINAL_MS,
    FIRST_PARTIAL_MS,
    REJECTIONS,
    SESSIONS_ACTIVE,
    UTTERANCES,
)

# Labeled metrics support .clear() (drops every label child); SESSIONS_ACTIVE
# is an unlabeled Gauge, whose bare value has no child to clear, so it is
# reset with .set(0) instead.
_LABELED_METRICS = (
    AUDIO_DROPPED,
    AUDIO_SECONDS,
    UTTERANCES,
    FIRST_PARTIAL_MS,
    FINAL_MS,
    REJECTIONS,
    ERRORS,
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    """All stt_server metrics live in one process-wide CollectorRegistry (by
    design — see metrics/registry.py), so without this fixture one test's
    counts would leak into the next test's assertions. Reset before *and*
    after so a test that reads live process metrics never depends on run
    order either."""
    SESSIONS_ACTIVE.set(0)
    for metric in _LABELED_METRICS:
        metric.clear()
    yield
    SESSIONS_ACTIVE.set(0)
    for metric in _LABELED_METRICS:
        metric.clear()
