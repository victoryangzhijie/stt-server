"""Native WebSocket protocol: the internal TranscriptEvent stream as JSON.

Richer than the OpenAI protocol (exposes the stable/volatile split); used by
benchmark clients so instrumentation never depends on OpenAI framing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid

import structlog
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

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


async def _handle_control_frame(
    ws: WebSocket, session: Session, text: str, send_task: asyncio.Task
) -> bool:
    """Handle one text (control) WebSocket frame for ws_transcribe().

    Returns True to keep the receive loop running, False to stop it — the
    caller's existing finally/abort cleanup is idempotent and handles the
    rest either way (including the "input_done" happy path, which already
    fully drains send_task and closes the socket itself before returning
    False here)."""
    error_message: str | None = None
    try:
        control = json.loads(text)
    except json.JSONDecodeError:
        error_message = "invalid JSON control frame"
    else:
        if not isinstance(control, dict):
            error_message = "control frame must be a JSON object"

    if error_message is not None:
        try:
            await ws.send_json(
                {"type": "error", "code": "bad_request", "recoverable": True,
                 "message": error_message}
            )
        except Exception:
            await session.abort()
            return False
        return True

    if control.get("type") == "input_done":
        await session.end_input()
        try:
            await send_task
            await ws.close()
        except Exception:
            await session.abort()
        return False

    try:
        await ws.send_json(
            {
                "type": "error",
                "code": "bad_request",
                "recoverable": True,
                "message": f"unknown control type: {control.get('type')!r}",
            }
        )
    except Exception:
        await session.abort()
        return False
    return True


def encode_native(ev: TranscriptEvent) -> dict:
    out: dict = {
        "type": ev.type.value,
        "session_id": ev.session_id,
        "utterance_id": ev.utterance_id,
        "seq": ev.seq,
        "audio_time_ms": ev.audio_time_ms,
    }
    if ev.type is EventType.PARTIAL:
        out["stable_text"] = ev.stable_text
        out["volatile_text"] = ev.volatile_text
    elif ev.type in (EventType.STABILIZED, EventType.FINAL):
        out["text"] = ev.text
    if ev.type is EventType.FINAL:
        out["latency"] = ev.latency
    if ev.type is EventType.ERROR:
        out["code"] = ev.error_code
        out["recoverable"] = ev.recoverable
        out["message"] = ev.message
    return out


@router.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket, model: str = "mock") -> None:
    from stt_server.api.app import resolve_backend  # local import: avoid cycle

    await ws.accept()
    if not check_token(ws.app, ws.headers.get("authorization")):
        await ws.send_json(
            {"type": "error", "code": "unauthorized", "recoverable": False,
             "message": "missing or invalid bearer token"}
        )
        await ws.close(code=4401)
        return
    if not ws.app.state.slots.acquire():
        await ws.send_json(
            {"type": "error", "code": "capacity", "recoverable": False,
             "message": "max concurrent sessions reached"}
        )
        await ws.close(code=4429)
        return

    try:
        try:
            backend = resolve_backend(ws.app, model)
        except BackendUnavailableError as exc:
            await ws.send_json(
                {"type": "error", "code": exc.code, "recoverable": False, "message": str(exc)}
            )
            await ws.close(code=4404)
            return

        settings = ws.app.state.settings
        session = Session(
            session_id=uuid.uuid4().hex,
            backend=backend,
            vad=make_vad(settings.vad),
            endpointer=Endpointer(settings.endpointing),
            stabilizer_factory=lambda: Stabilizer(settings.stabilizer),
            metrics_labels={"backend": backend.name, "api": "native"},
            audio_queue_chunks=settings.limits.audio_queue_chunks,
            audio_overflow_policy=settings.limits.audio_overflow_policy,
        )
        log = logger.bind(session_id=session.session_id, api="native", model=model)
        log.info("session.opened")
        deadline = session_deadline(settings)

        async def sender() -> None:
            async for ev in session.events():
                await ws.send_json(encode_native(ev))
            await ws.send_json({"type": "session.closed"})

        send_task = asyncio.create_task(sender())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    await session.abort()
                    break
                # No background timer: the deadline is only checked here, on
                # message arrival, so a quiet session times out on its NEXT
                # message rather than at the exact deadline instant.
                if time.monotonic() > deadline:
                    REJECTIONS.labels(reason="session_timeout").inc()
                    with contextlib.suppress(Exception):
                        await ws.send_json(
                            {"type": "error", "code": "session_timeout", "recoverable": False,
                             "message": "session exceeded max duration"}
                        )
                        await ws.close(code=4408)
                    await session.abort()
                    break
                if msg.get("bytes") is not None:
                    await session.push_audio(
                        AudioChunk(data=msg["bytes"], ingest_ts=time.monotonic())
                    )
                elif msg.get("text"):
                    if not await _handle_control_frame(ws, session, msg["text"], send_task):
                        break
        except WebSocketDisconnect:
            await session.abort()
        finally:
            # Idempotent: covers any unexpected exception path that didn't already
            # abort explicitly, so the reader task and backend stream always get
            # cleaned up.
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
