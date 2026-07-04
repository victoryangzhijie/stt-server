"""OpenAI Audio Transcriptions-style file endpoint.

File mode reuses the streaming pipeline (spec §3.3): decoded PCM is pushed
through a normal Session faster than real time, and the response is built
from the resulting TranscriptEvents."""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
import uuid
import wave
from typing import Annotated

import structlog
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from stt_server.api.guards import check_token
from stt_server.backends.base import BackendUnavailableError, StreamConfig
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import make_vad
from stt_server.metrics.registry import REJECTIONS

logger = structlog.get_logger(__name__)
router = APIRouter()

SAMPLE_RATE = 16000


class UnsupportedFormatError(Exception):
    pass


def _decode_wav(data: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise UnsupportedFormatError(
                    f"need 16 kHz mono pcm16 WAV, got {w.getframerate()} Hz "
                    f"{w.getnchannels()}ch {w.getsampwidth() * 8}-bit"
                )
            return w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise UnsupportedFormatError(f"not a WAV file: {exc}") from exc


def _err(
    status: int, code: str, message: str, *, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message}},
                        headers=headers)


async def run_file_session(app, model: str, pcm: bytes,
                           language: str | None) -> list[TranscriptEvent]:
    from stt_server.api.app import resolve_backend  # local import: avoid cycle

    backend = resolve_backend(app, model)
    settings = app.state.settings
    session = Session(
        session_id=uuid.uuid4().hex,
        backend=backend,
        vad=make_vad(settings.vad),
        endpointer=Endpointer(settings.endpointing),
        stabilizer_factory=lambda: Stabilizer(settings.stabilizer),
        stream_config=StreamConfig(language=language),
        metrics_labels={"backend": backend.name, "api": "file"},
        audio_queue_chunks=settings.limits.audio_queue_chunks,
        audio_overflow_policy=settings.limits.audio_overflow_policy,
    )
    log = logger.bind(session_id=session.session_id, api="file", model=model)
    events: list[TranscriptEvent] = []

    async def collect() -> None:
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    try:
        await session.push_audio(AudioChunk(data=pcm, ingest_ts=time.monotonic()))
        await session.end_input()
        await task
    except BaseException:
        await session.abort()
        raise
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        log.info(
            "session.summary",
            audio_seconds=session.stats.audio_seconds,
            utterance_count=session.stats.utterances,
            final_latencies_ms=session.stats.final_latencies_ms,
        )
    return events


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile,
    model: Annotated[str, Form()],
    response_format: Annotated[str, Form()] = "json",
    language: Annotated[str | None, Form()] = None,
    authorization: Annotated[str | None, Header()] = None,
):
    app = request.app
    if not check_token(app, authorization):
        return _err(401, "unauthorized", "missing or invalid bearer token",
                    headers={"WWW-Authenticate": "Bearer"})
    if response_format not in ("json", "text", "verbose_json"):
        return _err(400, "bad_request", f"unsupported response_format {response_format!r}")

    max_upload_bytes = app.state.settings.limits.max_upload_bytes
    # A declared-oversized `Content-Length` is already rejected upstream by
    # `UploadSizeGuardMiddleware` (pure ASGI, registered in `create_app`)
    # before Starlette's multipart parser ever buffers this request's body —
    # that's the real DoS fix, since this handler only runs at all once the
    # body has already been parsed into `file`. What remains here is the
    # post-read byte-count check below, which is the necessary backstop for
    # the one case the middleware can't cover: a chunked-transfer request
    # (or any request without a trustworthy `Content-Length` header), where
    # the true size isn't known until the body has actually been read.
    if not app.state.slots.acquire():
        return _err(429, "capacity", "max concurrent sessions reached")
    try:
        data = await file.read()
        # Post-read guard: catches chunked transfers or a missing/wrong
        # Content-Length header that the pre-check above couldn't rely on.
        if len(data) > max_upload_bytes:
            REJECTIONS.labels(reason="upload_too_large").inc()
            return _err(413, "upload_too_large",
                        f"upload exceeds {max_upload_bytes} bytes")
        try:
            pcm = _decode_wav(data)
        except UnsupportedFormatError as exc:
            return _err(400, "unsupported_format", str(exc))
        try:
            events = await run_file_session(app, model, pcm, language)
        except BackendUnavailableError as exc:
            return _err(404, "model_not_found", str(exc))
    finally:
        app.state.slots.release()

    error_events = [e for e in events if e.type is EventType.ERROR]
    if error_events:
        ev = error_events[0]
        return _err(500, ev.error_code or "backend_error", ev.message or "backend failure")

    finals = [e for e in events if e.type is EventType.FINAL]
    text = " ".join(f.text for f in finals)
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "json":
        return {"text": text}

    starts = {e.utterance_id: e.audio_time_ms
              for e in events if e.type is EventType.SPEECH_START}
    segments = [
        {
            "id": i,
            "start": starts.get(f.utterance_id, 0.0) / 1000.0,
            "end": f.audio_time_ms / 1000.0,
            "text": f.text,
        }
        for i, f in enumerate(finals)
    ]
    duration = len(pcm) / 2 / SAMPLE_RATE
    return {
        "task": "transcribe",
        "language": language or "en",
        "duration": duration,
        "text": text,
        "segments": segments,
    }
