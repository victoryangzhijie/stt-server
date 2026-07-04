import json

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from stt_server.api.app import UploadSizeGuardMiddleware, create_app
from stt_server.config.settings import AuthConfig, LimitsConfig
from stt_server.metrics.registry import AUDIO_SECONDS, REJECTIONS, UTTERANCES
from tests.api.test_native_ws import make_test_settings
from tests.helpers.audio import make_silence, make_tone, make_wav

WAV = lambda: make_wav(make_tone(600) + make_silence(300))  # noqa: E731


def post(client, **kw):
    files = {"file": ("a.wav", kw.pop("body", WAV()), "audio/wav")}
    data = {"model": kw.pop("model", "mock")}
    data.update(kw.pop("data", {}))
    return client.post("/v1/audio/transcriptions", files=files, data=data,
                       headers=kw.pop("headers", {}))


def test_json_response():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        r = post(client)
    assert r.status_code == 200
    assert r.json() == {"text": "The quick brown fox."}


def test_text_response():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        r = post(client, data={"response_format": "text"})
    assert r.status_code == 200
    assert r.text.strip() == "The quick brown fox."


def test_verbose_json_has_segments():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        body = make_wav(
            make_tone(600) + make_silence(300) + make_tone(600) + make_silence(300)
        )
        r = post(client, body=body, data={"response_format": "verbose_json"})
    assert r.status_code == 200
    j = r.json()
    assert j["task"] == "transcribe"
    assert len(j["segments"]) == 2
    assert j["segments"][0]["text"] == "The quick brown fox."
    assert j["segments"][1]["id"] == 1
    assert j["segments"][0]["end"] <= j["segments"][1]["start"]
    assert j["duration"] > 0
    assert j["text"] == "The quick brown fox. Hello world."


def test_unknown_model_404():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        r = post(client, model="nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_wrong_rate_400():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        r = post(client, body=make_wav(make_tone(200), rate=44100))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_format"


def test_auth_enforced():
    s = make_test_settings()
    s.auth = AuthConfig(tokens=["sekrit"])
    app = create_app(s)
    with TestClient(app) as client:
        assert post(client).status_code == 401
        r = post(client, headers={"Authorization": "Bearer sekrit"})
        assert r.status_code == 200


def test_empty_file_400():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        r = post(client, body=b"")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_format"


def test_backend_failure_returns_500():
    from stt_server.backends.base import BackendCapabilities, StreamConfig, SttBackend
    from stt_server.backends.registry import register_backend

    @register_backend("exploding")
    class ExplodingBackend(SttBackend):
        name = "exploding"
        capabilities = BackendCapabilities(streaming=True, languages=("en",))

        async def start(self) -> None: ...

        async def stop(self) -> None: ...

        async def create_stream(self, cfg: StreamConfig):
            raise RuntimeError("gpu on fire")

    s = make_test_settings()
    from stt_server.config.settings import BackendDef
    s.backends["exploding"] = BackendDef(type="exploding")
    s.models["exploding"] = "exploding"
    app = create_app(s)
    with TestClient(app) as client:
        r = post(client, model="exploding")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "backend_error"
    assert "gpu on fire" in r.json()["error"]["message"]


def test_http_capacity_429():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        # exhaust the slots synchronously before the request
        while app.state.slots.acquire():
            pass
        r = post(client)
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "capacity"


def test_401_carries_www_authenticate_header():
    s = make_test_settings()
    s.auth = AuthConfig(tokens=["sekrit"])
    app = create_app(s)
    with TestClient(app) as client:
        r = post(client)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_case_insensitive_bearer_scheme_accepted():
    s = make_test_settings()
    s.auth = AuthConfig(tokens=["sekrit"])
    app = create_app(s)
    with TestClient(app) as client:
        r = post(client, headers={"Authorization": "bearer sekrit"})
    assert r.status_code == 200


def test_upload_too_large_413_content_length_precheck():
    # A WAV body well over the configured cap must be rejected via the
    # Content-Length pre-check before the handler ever reads `file`.
    s = make_test_settings()
    s.limits = LimitsConfig(max_upload_bytes=100)
    app = create_app(s)
    with TestClient(app) as client:
        r = post(client)  # default WAV() body is far bigger than 100 bytes
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "upload_too_large"


def test_upload_too_large_413_post_read_check_when_content_length_understated():
    # Content-Length can be absent (chunked transfer) or simply wrong. Spoof
    # a small declared size so the pre-check passes, and confirm the actual
    # post-read byte count is still enforced.
    s = make_test_settings()
    s.limits = LimitsConfig(max_upload_bytes=100)
    app = create_app(s)
    with TestClient(app) as client:
        r = post(client, headers={"content-length": "1"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "upload_too_large"


async def test_upload_size_guard_middleware_rejects_before_receive_or_app():
    # Proves the pure-ASGI middleware responds using only scope["headers"] —
    # it never calls receive() (no body read) and never calls the wrapped
    # app (the handler/Starlette multipart parser never runs).
    async def app_that_must_not_run(scope, receive, send):
        raise AssertionError("wrapped app must not run for a rejected request")

    async def receive_that_must_not_run():
        raise AssertionError("middleware must not call receive() before rejecting")

    settings = make_test_settings()
    settings.limits = LimitsConfig(max_upload_bytes=100)
    middleware = UploadSizeGuardMiddleware(app_that_must_not_run, settings)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [(b"content-length", b"999999")],
    }
    messages = []

    async def send(message):
        messages.append(message)

    await middleware(scope, receive_that_must_not_run, send)

    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 413
    body = json.loads(messages[1]["body"])
    assert body == {
        "error": {"code": "upload_too_large", "message": "upload exceeds 100 bytes"}
    }


async def test_upload_size_guard_middleware_passes_through_small_content_length():
    calls = []

    async def app_that_records(scope, receive, send):
        calls.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    settings = make_test_settings()
    settings.limits = LimitsConfig(max_upload_bytes=100)
    middleware = UploadSizeGuardMiddleware(app_that_records, settings)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "headers": [(b"content-length", b"10")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    messages = []

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    assert len(calls) == 1
    assert messages[0]["status"] == 200


async def test_upload_size_guard_middleware_ignores_other_paths():
    calls = []

    async def app_that_records(scope, receive, send):
        calls.append(scope["path"])

    settings = make_test_settings()
    settings.limits = LimitsConfig(max_upload_bytes=100)
    middleware = UploadSizeGuardMiddleware(app_that_records, settings)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/healthz",
        "headers": [(b"content-length", b"999999")],
    }

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        pass

    await middleware(scope, receive, send)
    assert calls == ["/healthz"]


# -- observability: session.summary log line + Prometheus metrics --


def test_transcriptions_logs_session_summary():
    app = create_app(make_test_settings())
    with capture_logs() as cap:
        with TestClient(app) as client:
            post(client)
    summary = next(c for c in cap if c["event"] == "session.summary")
    assert summary["utterance_count"] == 1
    assert summary["audio_seconds"] > 0
    assert len(summary["final_latencies_ms"]) == 1


def test_transcriptions_observes_prometheus_metrics():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        post(client)
    assert UTTERANCES.labels(backend="mock", end_reason="silence")._value.get() == 1.0
    assert AUDIO_SECONDS.labels(api="file")._value.get() > 0.0


def test_upload_too_large_increments_rejections_counter():
    s = make_test_settings()
    s.limits = LimitsConfig(max_upload_bytes=100)
    app = create_app(s)
    with TestClient(app) as client:
        post(client)  # pre-read Content-Length check
        post(client, headers={"content-length": "1"})  # post-read check
    assert REJECTIONS.labels(reason="upload_too_large")._value.get() == 2.0
