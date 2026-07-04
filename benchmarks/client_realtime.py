"""Minimal OpenAI-Realtime (`/v1/realtime?intent=transcription`) streaming
client for the M-8 delta-duplication benchmark (`benchmarks.run_stabilizer_
study`). Mirrors `client_ws.stream_utterance`'s pacing/chunking, but speaks
the realtime event vocabulary instead of the native wire format -- see
`tests/api/test_realtime_ws.py` and `src/stt_server/api/realtime_ws.py` for
the protocol this is a client for."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field

import websockets

from benchmarks.client_ws import pacing_delay

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


@dataclass
class RealtimeItem:
    """One `conversation.item.input_audio_transcription.*` stream: every
    `.delta` event's text, in arrival order, plus the `.completed` event's
    `transcript` (`None` if the connection closed before this item
    completed -- see `stream_realtime`)."""

    item_id: str
    deltas: list[str] = field(default_factory=list)
    completed_transcript: str | None = None


@dataclass
class RealtimeResult:
    utt_id: str
    hypothesis: str
    items: list[RealtimeItem] = field(default_factory=list)


async def stream_realtime(
    base_ws_url: str,
    model: str,
    pcm16: bytes,
    *,
    chunk_ms: int = 100,
    pace: float = 1.0,
    token: str | None = None,
) -> RealtimeResult:
    """Stream `pcm16` (16 kHz mono PCM16 bytes) to `/v1/realtime?intent=
    transcription&model=...` as `chunk_ms`-sized base64 `input_audio_
    buffer.append` frames, paced identically to `client_ws.stream_utterance`
    (see `pacing_delay`, reused as-is). Sends `input_audio_buffer.commit`
    once all audio has been sent, then drains events until the server
    closes the connection -- the realtime adapter closes the socket itself
    once the commit's pending utterance(s) finish sending (see
    `src/stt_server/api/realtime_ws.py`'s `input_audio_buffer.commit`
    handler; there is no `session.closed` event in this protocol to wait
    for, unlike the native wire format).

    One `pcm16` buffer may endpoint into more than one utterance (e.g. it
    contains a pause): each surfaces as its own `item_id`. ALL items'
    `completed_transcript`s are joined with a space into `.hypothesis`
    (`None`/incomplete items contribute nothing), mirroring
    `stream_utterance`'s multi-segment join. `utt_id` is left blank here
    (unknown to the wire protocol) -- callers that track a
    `benchmarks.corpus.Utterance` should set `result.utt_id` themselves.
    """
    url = f"{base_ws_url}/v1/realtime?intent=transcription&model={model}"
    connect_kwargs: dict = {}
    if token is not None:
        connect_kwargs["additional_headers"] = {"Authorization": f"Bearer {token}"}

    items_by_id: dict[str, RealtimeItem] = {}
    item_order: list[str] = []

    def _item(iid: str) -> RealtimeItem:
        item = items_by_id.get(iid)
        if item is None:
            item = RealtimeItem(item_id=iid)
            items_by_id[iid] = item
            item_order.append(iid)
        return item

    async with websockets.connect(url, **connect_kwargs) as ws:

        async def receiver() -> None:
            async for raw in ws:
                ev = json.loads(raw)
                ev_type = ev.get("type")
                if ev_type == "conversation.item.input_audio_transcription.delta":
                    _item(ev["item_id"]).deltas.append(ev.get("delta", ""))
                elif ev_type == "conversation.item.input_audio_transcription.completed":
                    _item(ev["item_id"]).completed_transcript = ev.get("transcript", "")
                elif ev_type == "error" and not ev.get("recoverable", True):
                    return
                # transcription_session.created/.updated and
                # input_audio_buffer.speech_started/.speech_stopped are
                # received but not tracked -- nothing here needs them.

        recv_task = asyncio.create_task(receiver())

        chunk_bytes = max(1, SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_ms // 1000)
        t0 = time.monotonic()
        for idx, i in enumerate(range(0, len(pcm16), chunk_bytes)):
            delay = pacing_delay(t0, idx, chunk_ms, pace, time.monotonic())
            if delay > 0:
                await asyncio.sleep(delay)
            chunk = pcm16[i : i + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await recv_task

    hypothesis = " ".join(
        items_by_id[iid].completed_transcript
        for iid in item_order
        if items_by_id[iid].completed_transcript
    ).strip()

    return RealtimeResult(
        utt_id="",
        hypothesis=hypothesis,
        items=[items_by_id[iid] for iid in item_order],
    )
