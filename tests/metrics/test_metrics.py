"""End-to-end coverage of the `/metrics` endpoint: drives real sessions
through the native WS adapter and asserts on the rendered Prometheus text
body, rather than on the metric objects directly (that's covered per-module
in tests/core/test_session.py, tests/api/test_guards.py, and the other
tests/api/test_*_ws.py / test_transcriptions_http.py files)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST

from stt_server.api.app import create_app
from stt_server.config.settings import AuthConfig
from tests.api.test_native_ws import drain, make_test_settings
from tests.helpers.audio import make_silence, make_tone


def test_metrics_content_type():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"] == CONTENT_TYPE_LATEST


def test_metrics_body_reflects_a_completed_session():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            drain(ws)
        body = client.get("/metrics").text

    assert 'stt_utterances_total{backend="mock",end_reason="silence"} 1.0' in body
    assert 'stt_audio_seconds_ingested_total{api="native"}' in body
    assert "stt_final_latency_ms_bucket" in body
    assert "stt_first_partial_latency_ms_bucket" in body


def test_metrics_sessions_active_returns_to_zero_after_close():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            mid_body = client.get("/metrics").text
            assert "stt_sessions_active 1.0" in mid_body
            ws.send_text(json.dumps({"type": "input_done"}))
            drain(ws)
        final_body = client.get("/metrics").text
    assert "stt_sessions_active 0.0" in final_body


def test_metrics_rejections_increments_on_401():
    app = create_app(make_test_settings())
    app.state.settings.auth = AuthConfig(tokens=["sekrit"])
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            msg = ws.receive_json()
        assert msg["code"] == "unauthorized"
        body = client.get("/metrics").text
    assert 'stt_rejections_total{reason="unauthorized"} 1.0' in body
