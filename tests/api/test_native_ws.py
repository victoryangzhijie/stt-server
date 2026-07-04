import json
import time

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from stt_server.api.app import create_app
from stt_server.config.settings import (
    BackendDef,
    EndpointingConfig,
    LimitsConfig,
    Settings,
    StabilizerConfig,
)
from stt_server.metrics.registry import AUDIO_SECONDS, REJECTIONS, UTTERANCES
from tests.helpers.audio import make_silence, make_tone


def make_test_settings() -> Settings:
    return Settings(
        endpointing=EndpointingConfig(
            frame_ms=30, pre_roll_ms=90, min_silence_ms=90, speech_start_frames=2
        ),
        stabilizer=StabilizerConfig(min_partials=1, min_stable_ms=0.0),
        backends={"mock": BackendDef(type="mock", options={"partial_interval_ms": 100.0})},
        models={"mock": "mock"},
    )


def drain(ws) -> list[dict]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["type"] == "session.closed":
            return msgs


def test_ws_transcribe_end_to_end():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            msgs = drain(ws)

    types = [m["type"] for m in msgs]
    assert types[0] == "speech_start"
    assert "partial" in types
    final = next(m for m in msgs if m["type"] == "final")
    assert final["text"] == "The quick brown fox."  # first DEFAULT_SCRIPTS entry
    assert final["utterance_id"] == 0
    assert "final_ms" in final["latency"]
    partial = next(m for m in msgs if m["type"] == "partial")
    assert {"stable_text", "volatile_text", "seq", "session_id"} <= partial.keys()
    assert types[-1] == "session.closed"


def test_ws_unknown_model_sends_error():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=nope") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["code"] == "backend_unavailable"


def test_malformed_control_frame_gets_error_and_session_survives():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_text("not json {")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_request"
            # session still works end to end afterwards
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            msgs = drain(ws)
    assert any(m["type"] == "final" for m in msgs)


def test_non_dict_control_frame_gets_error_and_session_survives():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_text("123")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_request"
            # session still works end to end afterwards
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            msgs = drain(ws)
    assert any(m["type"] == "final" for m in msgs)


def test_client_disconnect_without_input_done_is_clean():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_bytes(make_tone(300))
        # context exit closes the socket mid-utterance; server must handle
        # the disconnect path (abort) without raising


def test_session_timeout_closes_with_4408():
    # No background timer: the deadline is only checked when the next
    # message arrives, so a quiet session times out on its NEXT message.
    s = make_test_settings()
    s.limits = LimitsConfig(max_session_seconds=0.05)
    app = create_app(s)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            time.sleep(0.1)
            ws.send_bytes(make_silence(30))
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "session_timeout"
            closed = ws.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 4408


def test_unknown_control_type_gets_error_and_session_survives():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_text(json.dumps({"type": "bogus"}))
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["code"] == "bad_request"
            assert "bogus" in err["message"]
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            msgs = drain(ws)
    assert any(m["type"] == "final" for m in msgs)


# -- observability: session.summary log line + Prometheus metrics --


def test_ws_transcribe_logs_session_summary():
    app = create_app(make_test_settings())
    with capture_logs() as cap:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/transcribe?model=mock") as ws:
                ws.send_bytes(make_tone(600) + make_silence(300))
                ws.send_text(json.dumps({"type": "input_done"}))
                drain(ws)
    summary = next(c for c in cap if c["event"] == "session.summary")
    assert summary["utterance_count"] == 1
    assert summary["audio_seconds"] > 0
    assert len(summary["final_latencies_ms"]) == 1


def test_ws_transcribe_observes_prometheus_metrics():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            drain(ws)
    assert UTTERANCES.labels(backend="mock", end_reason="silence")._value.get() == 1.0
    assert AUDIO_SECONDS.labels(api="native")._value.get() > 0.0


def test_ws_session_timeout_increments_rejections_counter():
    s = make_test_settings()
    s.limits = LimitsConfig(max_session_seconds=0.05)
    app = create_app(s)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            time.sleep(0.1)
            ws.send_bytes(make_silence(30))
            ws.receive_json()
    assert REJECTIONS.labels(reason="session_timeout")._value.get() == 1.0
