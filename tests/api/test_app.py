import pytest
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
from structlog.testing import capture_logs

from stt_server.api.app import create_app
from stt_server.backends.base import BackendCapabilities, StreamConfig, SttBackend
from stt_server.backends.registry import register_backend
from stt_server.config.settings import BackendDef, Settings


def test_health_and_ready():
    app = create_app(Settings())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200


def test_metrics_endpoint_returns_prometheus_content_type():
    app = create_app(Settings())
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"] == CONTENT_TYPE_LATEST
        assert b"# HELP stt_sessions_active" in r.content


def test_readyz_503_before_startup():
    # Without entering the TestClient as a context manager, the lifespan's
    # startup phase never runs — readyz must report not-ready rather than
    # crash or default to ready.
    app = create_app(Settings())
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json() == {"status": "not_ready"}


def test_lifespan_stops_started_backends_in_reverse_order_on_start_failure():
    stop_calls: list[str] = []

    @register_backend("probe_ok")
    class ProbeBackend(SttBackend):
        name = "probe_ok"
        capabilities = BackendCapabilities(streaming=True, languages=("en",))

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            stop_calls.append("probe")

        async def create_stream(self, cfg: StreamConfig):
            raise NotImplementedError

    @register_backend("probe_exploding")
    class ExplodingOnStartBackend(SttBackend):
        name = "probe_exploding"
        capabilities = BackendCapabilities(streaming=True, languages=("en",))

        async def start(self) -> None:
            raise RuntimeError("gpu init failed")

        async def stop(self) -> None:
            stop_calls.append("exploding")

        async def create_stream(self, cfg: StreamConfig):
            raise NotImplementedError

    settings = Settings(
        backends={
            "probe": BackendDef(type="probe_ok"),
            "exploding": BackendDef(type="probe_exploding"),
        },
        models={},
    )
    app = create_app(settings)

    with capture_logs() as cap:
        with pytest.raises(RuntimeError, match="gpu init failed"):
            with TestClient(app):
                pass

    # Only the already-started "probe" backend must have been stopped;
    # "exploding" never finished starting so it must not be stopped too.
    assert stop_calls == ["probe"]
    events = [c["event"] for c in cap]
    assert "backend.start_failed" in events
    failure_log = next(c for c in cap if c["event"] == "backend.start_failed")
    assert failure_log.get("backend") == "exploding"
