"""OpenAI Realtime transcription-intent adapter (beta-style event vocabulary).

Pure protocol translation over the session core. Deltas are sourced from
STABILIZED events (committed-prefix growth) — append-only by construction.
See docs/openai-compat.md for the tested compatibility matrix."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import time
import uuid

import structlog
from fastapi import APIRouter, WebSocket

from stt_server.api.guards import check_token, session_deadline
from stt_server.backends.base import BackendUnavailableError
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import make_vad
from stt_server.metrics.registry import REJECTIONS

logger = structlog.get_logger(__name__)
router = APIRouter()


def item_id_for(session_id: str, utterance_id: int) -> str:
    return f"item_{session_id[:8]}_{utterance_id}"


def _error(code: str, message: str) -> dict:
    return {
        "type": "error",
        "error": {"type": "invalid_request_error", "code": code, "message": message},
    }


def encode_realtime(ev: TranscriptEvent, item_id: str) -> dict | None:
    if ev.type is EventType.SPEECH_START:
        return {
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": ev.audio_time_ms,
            "item_id": item_id,
        }
    if ev.type is EventType.SPEECH_END:
        return {
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": ev.audio_time_ms,
            "item_id": item_id,
        }
    if ev.type is EventType.STABILIZED:
        return {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": item_id,
            "content_index": 0,
            "delta": ev.text,
        }
    if ev.type is EventType.FINAL:
        return {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "content_index": 0,
            "transcript": ev.text,
        }
    if ev.type is EventType.ERROR:
        return _error(ev.error_code or "internal_error", ev.message)
    return None  # PARTIAL has no OpenAI representation


@router.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket, intent: str = "", model: str = "mock") -> None:
    from stt_server.api.app import resolve_backend  # local import: avoid cycle

    await ws.accept()
    if intent != "transcription":
        await ws.send_json(_error("unsupported_intent",
                                  "only intent=transcription is supported"))
        await ws.close(code=4400)
        return
    if not check_token(ws.app, ws.headers.get("authorization")):
        await ws.send_json(_error("unauthorized", "missing or invalid bearer token"))
        await ws.close(code=4401)
        return
    # Guard order: intent -> token -> capacity -> backend resolution.
    # Capacity is checked before backend resolution (matching native_ws) so
    # that a full server rejects with "capacity" rather than spending a
    # backend lookup first.
    if not ws.app.state.slots.acquire():
        await ws.send_json(_error("capacity", "max concurrent sessions reached"))
        await ws.close(code=4429)
        return

    # Everything below acquires cleanup responsibility for the slot: wrap the
    # entire post-acquire lifecycle (backend resolution, session construction,
    # the initial `created` send, and the receive loop) in one try/finally so
    # the slot is released exactly once no matter which post-acquire path
    # exits — normal completion, an early `return`, or an exception raised
    # before the receive loop even starts (e.g. `ws.send_json` failing
    # because the client already dropped the connection).
    try:
        try:
            backend = resolve_backend(ws.app, model)
        except BackendUnavailableError as exc:
            await ws.send_json(_error(exc.code, str(exc)))
            await ws.close(code=4404)
            return
        settings = ws.app.state.settings
        session = Session(
            session_id=uuid.uuid4().hex,
            backend=backend,
            vad=make_vad(settings.vad),
            endpointer=Endpointer(settings.endpointing),
            stabilizer_factory=lambda: Stabilizer(settings.stabilizer),
            metrics_labels={"backend": backend.name, "api": "realtime"},
            audio_queue_chunks=settings.limits.audio_queue_chunks,
            audio_overflow_policy=settings.limits.audio_overflow_policy,
        )
        log = logger.bind(session_id=session.session_id, api="realtime", model=model)

        async def sender() -> None:
            # Track the EXACT text already sent on the wire per utterance, so
            # that an OpenAI client reconstructing via "".join(deltas) gets
            # correctly spaced text back. STABILIZED.text is a bare committed
            # token ("the", "quick", ...) with no inter-word space of its
            # own, so the space has to be injected here, on the wire, not
            # assumed by the reader.
            wire_sent: dict[int, str] = {}

            async def _send(wire: dict | None) -> None:
                if wire is not None:
                    await ws.send_json(wire)

            async for ev in session.events():
                iid = item_id_for(ev.session_id, ev.utterance_id)
                if ev.type is EventType.STABILIZED:
                    uid = ev.utterance_id
                    prior = wire_sent.get(uid, "")
                    to_send = ev.text if not prior else " " + ev.text
                    wire_sent[uid] = prior + to_send
                    wire = encode_realtime(ev, iid)
                    if wire is not None:
                        wire["delta"] = to_send
                    await _send(wire)
                    continue
                if ev.type is EventType.FINAL:
                    uid = ev.utterance_id
                    already = wire_sent.get(uid, "")
                    # Case/punctuation-insensitive: FINAL commonly applies casing
                    # and punctuation normalization that raw stabilizer commits
                    # never had (e.g. mock backend commits "the"/"fox", FINAL
                    # says "The"/"fox."). That cosmetic drift must not trigger a
                    # full-text duplicate catch-up. Note: casing of text already
                    # sent on the wire can never be corrected retroactively —
                    # only a remainder can be appended, so casing drift in the
                    # already-sent portion is an inherent, unfixable limitation.
                    if already.casefold() != ev.text.casefold():
                        remainder = (
                            ev.text[len(already):]
                            if ev.text.casefold().startswith(already.casefold())
                            else ev.text
                        )
                        if remainder:
                            catch_up = encode_realtime(
                                TranscriptEvent(
                                    type=EventType.STABILIZED, session_id=ev.session_id,
                                    utterance_id=uid, seq=ev.seq,
                                    audio_time_ms=ev.audio_time_ms, emitted_ts=ev.emitted_ts,
                                    text=remainder,
                                ),
                                iid,
                            )
                            await _send(catch_up)
                    await _send(encode_realtime(ev, iid))
                    wire_sent.pop(uid, None)
                    continue
                await _send(encode_realtime(ev, iid))

        await ws.send_json({
            "type": "transcription_session.created",
            "session": {"id": session.session_id, "intent": "transcription"},
        })
        log.info("session.opened")
        deadline = session_deadline(settings)
        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                # No background timer: the deadline is only checked here, on
                # message arrival, so a quiet session times out on its NEXT
                # message rather than at the exact deadline instant.
                if time.monotonic() > deadline:
                    REJECTIONS.labels(reason="session_timeout").inc()
                    with contextlib.suppress(Exception):
                        await ws.send_json(_error("session_timeout",
                                                   "session exceeded max duration"))
                        await ws.close(code=4408)
                    break
                if msg.get("bytes") is not None:  # extension: binary audio frames
                    await session.push_audio(
                        AudioChunk(data=msg["bytes"], ingest_ts=time.monotonic())
                    )
                    continue
                if not msg.get("text"):
                    continue
                try:
                    event = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json(_error("bad_request", "invalid JSON event"))
                    continue
                if not isinstance(event, dict):
                    await ws.send_json(_error("bad_request", "event must be a JSON object"))
                    continue
                etype = event.get("type")
                if etype == "transcription_session.update":
                    await ws.send_json({
                        "type": "transcription_session.updated",
                        "session": event.get("session", {}),
                    })
                elif etype == "input_audio_buffer.append":
                    raw_audio = event.get("audio")
                    if not raw_audio:  # missing key, explicit None, or ""
                        await ws.send_json(
                            _error("bad_request", "audio must be non-empty base64")
                        )
                        continue
                    try:
                        audio = base64.b64decode(raw_audio, validate=True)
                    except (binascii.Error, ValueError, TypeError):
                        await ws.send_json(_error("bad_request", "audio must be base64"))
                        continue
                    if not audio:
                        await ws.send_json(
                            _error("bad_request", "audio must be non-empty base64")
                        )
                        continue
                    await session.push_audio(
                        AudioChunk(data=audio, ingest_ts=time.monotonic())
                    )
                elif etype == "input_audio_buffer.commit":
                    await session.end_input()
                    try:
                        await send_task
                        await ws.close()
                    except Exception:
                        await session.abort()
                    return
                else:
                    await ws.send_json(
                        _error("unknown_event", f"unsupported event type: {etype!r}")
                    )
        finally:
            with contextlib.suppress(BaseException):
                await session.abort()
            if not send_task.done():
                send_task.cancel()
            with contextlib.suppress(BaseException):
                await send_task
            log.info(
                "session.summary",
                audio_seconds=session.stats.audio_seconds,
                utterance_count=session.stats.utterances,
                final_latencies_ms=session.stats.final_latencies_ms,
            )
            log.info("session.closed")
    finally:
        ws.app.state.slots.release()
