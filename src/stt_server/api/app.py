"""FastAPI application factory."""

from __future__ import annotations

import contextlib
import json

import structlog
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from stt_server.api import native_ws, realtime_ws, transcriptions_http
from stt_server.api.guards import SessionSlots
from stt_server.backends.base import BackendUnavailableError, SttBackend
from stt_server.backends.registry import create_backend
from stt_server.config.settings import Settings
from stt_server.metrics.registry import REGISTRY, REJECTIONS

logger = structlog.get_logger(__name__)

_UPLOAD_GUARD_PATH = "/v1/audio/transcriptions"


class UploadSizeGuardMiddleware:
    """Pure-ASGI middleware (spec §5.4 / M-1): rejects a declared-oversized
    upload on `POST /v1/audio/transcriptions` using only the `Content-Length`
    header, BEFORE Starlette's multipart parser ever buffers the body.

    Deliberately not a `BaseHTTPMiddleware` subclass or an `@app.middleware`
    function — both of those sit *after* body buffering has already started
    (Starlette wraps `receive()` and the route/endpoint's own
    `await request.form()` call does the actual multipart parse; a
    `BaseHTTPMiddleware` also duplicates the ASGI receive channel through an
    internal queue). This class is `app.add_middleware(...)`-registered as
    the outermost ASGI layer, so for the guarded path+method it can inspect
    `scope["headers"]` and, if rejecting, respond via `send()` directly
    without ever calling `receive()` — the handler, FastAPI's dependency
    injection, and Starlette's multipart parser never run for a rejected
    request.

    Requests without a `Content-Length` header (chunked transfer-encoding)
    fall through unmodified to the app; `transcriptions_http.py` keeps its
    own post-read byte-count check as the backstop for that case.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self._app = app
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != _UPLOAD_GUARD_PATH
        ):
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            max_upload_bytes = self._settings.limits.max_upload_bytes
            if declared_size is not None and declared_size > max_upload_bytes:
                REJECTIONS.labels(reason="upload_too_large").inc()
                # separators match JSONResponse's compact output so this body is
                # byte-identical to the endpoint's post-read 413 envelope
                body = json.dumps(
                    {
                        "error": {
                            "code": "upload_too_large",
                            "message": f"upload exceeds {max_upload_bytes} bytes",
                        }
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return

        await self._app(scope, receive, send)


def create_app(settings: Settings) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        backends = {key: create_backend(defn) for key, defn in settings.backends.items()}
        started: list[SttBackend] = []
        for key, backend in backends.items():
            try:
                await backend.start()
            except Exception as exc:
                logger.error("backend.start_failed", backend=key, error=str(exc))
                for started_backend in reversed(started):
                    await started_backend.stop()
                raise
            started.append(backend)
        app.state.backends = backends
        app.state.ready = True
        yield
        app.state.ready = False
        for backend in backends.values():
            await backend.stop()

    app = FastAPI(title="stt-server", lifespan=lifespan)
    app.add_middleware(UploadSizeGuardMiddleware, settings=settings)
    app.state.settings = settings
    app.state.slots = SessionSlots(limit=settings.limits.max_sessions)
    app.state.ready = False
    app.include_router(native_ws.router)
    app.include_router(realtime_ws.router)
    app.include_router(transcriptions_http.router)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response):
        if not app.state.ready:
            response.status_code = 503
            return {"status": "not_ready"}
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


def resolve_backend(app: FastAPI, model: str) -> SttBackend:
    backend_key = app.state.settings.models.get(model)
    if backend_key is None or backend_key not in app.state.backends:
        raise BackendUnavailableError(f"unknown model {model!r}")
    return app.state.backends[backend_key]
