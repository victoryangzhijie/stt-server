"""Metric families for stt-server (spec §7).

Own `CollectorRegistry` rather than prometheus_client's process-global
default: keeps `/metrics` output limited to exactly these families (no
surprise collectors registered by an unrelated import elsewhere in the
process) and keeps tests hermetic (a fixture can `.clear()` every metric
between tests without touching global state other code might depend on).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

# Shared by both latency histograms per the task brief.
_LATENCY_BUCKETS_MS = (50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000)

SESSIONS_ACTIVE = Gauge(
    "stt_sessions_active",
    "Number of concurrently active STT sessions.",
    registry=REGISTRY,
)

AUDIO_SECONDS = Counter(
    "stt_audio_seconds_ingested_total",
    "Total seconds of audio ingested, labeled by API surface.",
    ["api"],
    registry=REGISTRY,
)

UTTERANCES = Counter(
    "stt_utterances_total",
    "Total utterances finalized, labeled by backend and endpoint reason.",
    ["backend", "end_reason"],
    registry=REGISTRY,
)

FIRST_PARTIAL_MS = Histogram(
    "stt_first_partial_latency_ms",
    "Latency from utterance start to the first partial transcript, in milliseconds.",
    ["backend"],
    buckets=_LATENCY_BUCKETS_MS,
    registry=REGISTRY,
)

FINAL_MS = Histogram(
    "stt_final_latency_ms",
    "Latency from endpoint detection to the final transcript, in milliseconds.",
    ["backend"],
    buckets=_LATENCY_BUCKETS_MS,
    registry=REGISTRY,
)

REJECTIONS = Counter(
    "stt_rejections_total",
    "Total requests rejected before a session was created, labeled by reason.",
    ["reason"],
    registry=REGISTRY,
)

ERRORS = Counter(
    "stt_errors_total",
    "Total session errors, labeled by error code.",
    ["code"],
    registry=REGISTRY,
)

AUDIO_DROPPED = Counter(
    "stt_audio_dropped_total",
    "Total PCM chunks shed by the per-session backpressure policy, labeled by backend.",
    ["backend"],
    registry=REGISTRY,
)
