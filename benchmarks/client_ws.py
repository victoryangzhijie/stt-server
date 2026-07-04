"""Instrumented native-WS streaming client (the benchmark superset of
`examples/ws_client.py`): paced audio streaming + latency/partial capture
against the wire format in `stt_server.api.native_ws.encode_native`."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def pacing_delay(t0: float, chunk_index: int, chunk_ms: int, pace: float, now: float) -> float:
    """Seconds to sleep before sending chunk `chunk_index` under
    cumulative-deadline pacing: chunk i's send deadline is
    `t0 + i * (chunk_ms/1000) / pace`. `pace <= 0` disables pacing
    (as-fast-as-possible); a missed deadline yields 0.0 (send immediately,
    never a negative sleep)."""
    if pace <= 0:
        return 0.0
    deadline = t0 + chunk_index * (chunk_ms / 1000.0) / pace
    return max(0.0, deadline - now)


@dataclass
class UtteranceResult:
    utt_id: str
    hypothesis: str
    server_final_ms: float | None
    server_first_partial_ms: float | None
    client_final_ms: float
    partials: list[tuple[float, str, str]] = field(default_factory=list)
    # Server-side utterance id of each entry in `partials` (parallel list,
    # same length/order). One audio buffer can endpoint into SEVERAL
    # server-side utterances; per-utterance metrics (e.g. the stabilizer
    # study's flicker/commit-latency) need this to segment the partial
    # trail — the combined hypothesis RESETS at each utterance boundary,
    # which a whole-session pairwise diff would misread as a retraction.
    partial_utterance_ids: list[int] = field(default_factory=list)
    # (utterance_id, final_text) per FINAL event, in arrival order — the
    # per-segment counterpart of `hypothesis` (which joins these texts).
    finals: list[tuple[int, str]] = field(default_factory=list)


async def stream_utterance(
    base_ws_url: str,
    model: str,
    pcm16: bytes,
    *,
    chunk_ms: int = 100,
    pace: float = 1.0,
    token: str | None = None,
) -> UtteranceResult:
    """Stream `pcm16` (16 kHz mono PCM16 bytes) to `/ws/transcribe?model=...`
    in `chunk_ms`-sized chunks, pacing sends at `chunk_ms / pace` intervals
    (`pace=1.0` is real time; `pace=0` disables sleeping — as-fast-as-possible,
    for file-mode parity checks). Sends `{"type": "input_done"}` once all
    audio has been sent, then drains events until `session.closed`.

    A single audio buffer may endpoint into more than one utterance (e.g. it
    contains a pause): ALL `final` events are collected and their texts
    joined with a space into `hypothesis`. `utt_id` is left blank here (the
    manifest-level id isn't known to the wire protocol) — callers that track
    a `benchmarks.corpus.Utterance` should set `result.utt_id` themselves.
    """
    url = f"{base_ws_url}/ws/transcribe?model={model}"
    connect_kwargs: dict = {}
    if token is not None:
        connect_kwargs["additional_headers"] = {"Authorization": f"Bearer {token}"}

    partials: list[tuple[float, str, str]] = []
    partial_utterance_ids: list[int] = []
    finals: list[tuple[int, str]] = []
    final_latencies: list[dict] = []
    final_recv_ts: list[float] = []

    async with websockets.connect(url, **connect_kwargs) as ws:

        async def receiver() -> None:
            async for raw in ws:
                ev = json.loads(raw)
                ev_type = ev.get("type")
                if ev_type == "partial":
                    partials.append(
                        (ev["audio_time_ms"], ev.get("stable_text", ""),
                         ev.get("volatile_text", ""))
                    )
                    partial_utterance_ids.append(ev.get("utterance_id", 0))
                elif ev_type == "final":
                    finals.append((ev.get("utterance_id", 0), ev.get("text", "")))
                    final_latencies.append(ev.get("latency", {}))
                    final_recv_ts.append(time.monotonic())
                elif ev_type == "session.closed":
                    return
                elif ev_type == "error" and not ev.get("recoverable", True):
                    return

        recv_task = asyncio.create_task(receiver())

        chunk_bytes = max(1, SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_ms // 1000)
        # Cumulative-deadline pacing: chunk i is sent at t0 + i*interval, so
        # per-iteration jitter (event-loop scheduling, ws.send time) never
        # accumulates into drift over long utterances — a naive fixed
        # sleep-per-chunk would.
        t0 = time.monotonic()
        for idx, i in enumerate(range(0, len(pcm16), chunk_bytes)):
            delay = pacing_delay(t0, idx, chunk_ms, pace, time.monotonic())
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(pcm16[i : i + chunk_bytes])
        last_send_ts = time.monotonic()

        await ws.send(json.dumps({"type": "input_done"}))
        await recv_task

    hypothesis = " ".join(text for _, text in finals)
    server_final_ms = final_latencies[-1].get("final_ms") if final_latencies else None
    server_first_partial_ms = (
        final_latencies[0].get("first_partial_ms") if final_latencies else None
    )
    client_final_ms = (
        (final_recv_ts[-1] - last_send_ts) * 1000.0 if final_recv_ts else 0.0
    )

    return UtteranceResult(
        utt_id="",
        hypothesis=hypothesis,
        server_final_ms=server_final_ms,
        server_first_partial_ms=server_first_partial_ms,
        client_final_ms=client_final_ms,
        partials=partials,
        partial_utterance_ids=partial_utterance_ids,
        finals=finals,
    )
