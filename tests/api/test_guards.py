import json
import time

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from stt_server.api.app import create_app
from stt_server.api.guards import SessionSlots, check_token, session_deadline
from stt_server.config.settings import AuthConfig, LimitsConfig, Settings
from stt_server.metrics.registry import REJECTIONS, SESSIONS_ACTIVE
from tests.helpers.audio import make_silence, make_tone


def test_session_slots_counting():
    slots = SessionSlots(limit=2)
    assert slots.acquire() and slots.acquire()
    assert not slots.acquire()          # full
    slots.release()
    assert slots.acquire()
    assert slots.active == 2


def test_ws_requires_token_when_configured():
    app = create_app(Settings(auth=AuthConfig(tokens=["sekrit"])))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "error" and msg["code"] == "unauthorized"


def test_ws_accepts_valid_token_and_counts_slots():
    app = create_app(
        Settings(auth=AuthConfig(tokens=["sekrit"]), limits=LimitsConfig(max_sessions=1))
    )
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer sekrit"}
        with client.websocket_connect("/ws/transcribe?model=mock", headers=headers) as ws1:
            # second concurrent session must be rejected for capacity
            with client.websocket_connect("/ws/transcribe?model=mock", headers=headers) as ws2:
                msg = ws2.receive_json()
                assert msg["type"] == "error" and msg["code"] == "capacity"
            ws1.send_text(json.dumps({"type": "input_done"}))
            while ws1.receive_json()["type"] != "session.closed":
                pass
        # slot released after close
        assert app.state.slots.active == 0


def test_no_tokens_configured_means_open_access():
    app = create_app(Settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            types = []
            while True:
                m = ws.receive_json()
                types.append(m["type"])
                if m["type"] == "session.closed":
                    break
    assert "final" in types


# -- check_token: scheme parsing, constant-time compare, rejection logging --


def test_check_token_accepts_case_insensitive_bearer_scheme():
    app = create_app(Settings(auth=AuthConfig(tokens=["sekrit"])))
    assert check_token(app, "Bearer sekrit")
    assert check_token(app, "bearer sekrit")
    assert check_token(app, "BEARER sekrit")


def test_check_token_rejects_non_bearer_scheme():
    app = create_app(Settings(auth=AuthConfig(tokens=["sekrit"])))
    assert not check_token(app, "Basic sekrit")


def test_check_token_rejects_missing_or_empty_credentials():
    app = create_app(Settings(auth=AuthConfig(tokens=["sekrit"])))
    assert not check_token(app, "Bearer")
    assert not check_token(app, "Bearer ")
    assert not check_token(app, None)


def test_check_token_checks_every_configured_token_no_early_exit():
    app = create_app(Settings(auth=AuthConfig(tokens=["a", "b", "sekrit"])))
    assert check_token(app, "Bearer sekrit")
    assert not check_token(app, "Bearer nope")


def test_check_token_logs_auth_rejected_without_token_material():
    app = create_app(Settings(auth=AuthConfig(tokens=["sekrit"])))
    with capture_logs() as cap:
        assert not check_token(app, "Bearer wrong-token-value")
    events = [c["event"] for c in cap]
    assert "auth.rejected" in events
    dump = repr(cap)
    assert "wrong-token-value" not in dump
    assert "sekrit" not in dump


# -- SessionSlots.release(): anomaly logging on underflow, still clamps --


def test_slots_release_underflow_logs_anomaly_and_still_clamps():
    slots = SessionSlots(limit=2)
    with capture_logs() as cap:
        slots.release()
    assert slots.active == 0
    events = [c["event"] for c in cap]
    assert "slots.release_underflow" in events


# -- session_deadline(): a future monotonic timestamp --


def test_session_deadline_is_max_session_seconds_ahead():
    settings = Settings(limits=LimitsConfig(max_session_seconds=10.0))
    before = time.monotonic()
    deadline = session_deadline(settings)
    after = time.monotonic()
    assert before + 10.0 <= deadline <= after + 10.0


# -- SessionSlots drives the SESSIONS_ACTIVE gauge --


def test_slots_acquire_and_release_drive_sessions_active_gauge():
    slots = SessionSlots(limit=2)
    assert SESSIONS_ACTIVE._value.get() == 0.0
    slots.acquire()
    assert SESSIONS_ACTIVE._value.get() == 1.0
    slots.acquire()
    assert SESSIONS_ACTIVE._value.get() == 2.0
    slots.release()
    assert SESSIONS_ACTIVE._value.get() == 1.0
    slots.release()
    assert SESSIONS_ACTIVE._value.get() == 0.0


# -- rejection counters --


def test_check_token_rejection_increments_rejections_counter():
    app = create_app(Settings(auth=AuthConfig(tokens=["sekrit"])))
    assert REJECTIONS.labels(reason="unauthorized")._value.get() == 0.0
    check_token(app, "Bearer wrong")
    check_token(app, None)
    check_token(app, "Basic sekrit")
    assert REJECTIONS.labels(reason="unauthorized")._value.get() == 3.0


def test_slots_acquire_over_limit_increments_rejections_counter():
    slots = SessionSlots(limit=1)
    slots.acquire()
    assert REJECTIONS.labels(reason="capacity")._value.get() == 0.0
    with capture_logs() as cap:
        assert not slots.acquire()
    assert REJECTIONS.labels(reason="capacity")._value.get() == 1.0
    rejected = [c for c in cap if c["event"] == "capacity.rejected"]
    assert len(rejected) == 1
    assert rejected[0]["active"] == 1
    assert rejected[0]["limit"] == 1
