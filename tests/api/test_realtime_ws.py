import base64
import json
import time

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from stt_server.api.app import create_app
from stt_server.api.realtime_ws import encode_realtime, item_id_for
from stt_server.config.settings import LimitsConfig
from stt_server.core.events import EventType, TranscriptEvent
from stt_server.metrics.registry import AUDIO_SECONDS, REJECTIONS, UTTERANCES
from tests.api.test_native_ws import make_test_settings
from tests.helpers.audio import make_silence, make_tone


def ev(type_, **kw):
    base = dict(
        type=type_, session_id="sess01234", utterance_id=0, seq=0,
        audio_time_ms=120.0, emitted_ts=0.0,
    )
    base.update(kw)
    return TranscriptEvent(**base)


def test_encoder_table():
    iid = item_id_for("sess01234", 0)
    assert encode_realtime(ev(EventType.SPEECH_START), iid) == {
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": 120.0,
        "item_id": iid,
    }
    assert encode_realtime(ev(EventType.SPEECH_END), iid) == {
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": 120.0,
        "item_id": iid,
    }
    assert encode_realtime(ev(EventType.STABILIZED, text="hello"), iid) == {
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": iid,
        "content_index": 0,
        "delta": "hello",
    }
    assert encode_realtime(ev(EventType.FINAL, text="Hello world."), iid) == {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": iid,
        "content_index": 0,
        "transcript": "Hello world.",
    }
    assert encode_realtime(ev(EventType.PARTIAL, stable_text="a", volatile_text="b"), iid) is None
    err = encode_realtime(
        ev(EventType.ERROR, error_code="backend_error", message="boom", recoverable=False), iid
    )
    assert err["type"] == "error"
    assert err["error"]["code"] == "backend_error"
    assert err["error"]["message"] == "boom"


def collect_until_completed(ws) -> list[dict]:
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] == "conversation.item.input_audio_transcription.completed":
            return msgs


def test_realtime_transcription_end_to_end():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription&model=mock") as ws:
            created = ws.receive_json()
            assert created["type"] == "transcription_session.created"

            ws.send_text(json.dumps({
                "type": "transcription_session.update",
                "session": {"input_audio_transcription": {"language": "en"}},
            }))
            updated = ws.receive_json()
            assert updated["type"] == "transcription_session.updated"

            audio = make_tone(600) + make_silence(300)
            ws.send_text(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode(),
            }))
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            msgs = collect_until_completed(ws)

    types = [m["type"] for m in msgs]
    assert "input_audio_buffer.speech_started" in types
    completed = msgs[-1]
    assert completed["transcript"] == "The quick brown fox."
    # Deltas must concatenate via plain "".join (as an OpenAI client does)
    # into the final transcript, spaces and all. Only casing is allowed to
    # drift: the mock backend's raw partials ("the"/"fox") never carry
    # FINAL's casing normalization ("The"/"fox."), and casing of text
    # already sent on the wire can't be corrected retroactively.
    deltas = [m["delta"] for m in msgs
              if m["type"] == "conversation.item.input_audio_transcription.delta"]
    assert "".join(deltas).casefold() == completed["transcript"].casefold()
    item_ids = {m["item_id"] for m in msgs if "item_id" in m}
    assert len(item_ids) == 1


def test_realtime_unknown_event_type_gets_error():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription&model=mock") as ws:
            ws.receive_json()  # created
            ws.send_text(json.dumps({"type": "response.create"}))
            err = ws.receive_json()
    assert err["type"] == "error"
    assert err["error"]["type"] == "invalid_request_error"
    assert "response.create" in err["error"]["message"]


def test_realtime_requires_token_when_configured():
    from stt_server.config.settings import AuthConfig

    s = make_test_settings()
    s.auth = AuthConfig(tokens=["sekrit"])
    app = create_app(s)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription&model=mock") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["error"]["code"] == "unauthorized"


def test_realtime_capacity_checked_before_backend_resolution():
    # Guard order is intent -> token -> capacity -> backend resolution.
    # With slots exhausted AND an unknown model, the capacity rejection
    # must win (backend resolution should never even run).
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        while app.state.slots.acquire():
            pass
        with client.websocket_connect(
            "/v1/realtime?intent=transcription&model=bogus-model"
        ) as ws:
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["error"]["code"] == "capacity"


def test_realtime_input_audio_buffer_append_rejects_missing_none_empty_audio():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription&model=mock") as ws:
            ws.receive_json()  # created

            ws.send_text(json.dumps({"type": "input_audio_buffer.append"}))
            err1 = ws.receive_json()

            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": None}))
            err2 = ws.receive_json()

            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": ""}))
            err3 = ws.receive_json()

            # session still works afterwards: no silent empty audio was ever pushed
            audio = make_tone(600) + make_silence(300)
            ws.send_text(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode(),
            }))
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            msgs = collect_until_completed(ws)

    for err in (err1, err2, err3):
        assert err["type"] == "error"
        assert err["error"]["code"] == "bad_request"
    assert msgs[-1]["transcript"] == "The quick brown fox."


def test_realtime_session_timeout_closes_with_4408():
    s = make_test_settings()
    s.limits = LimitsConfig(max_session_seconds=0.05)
    app = create_app(s)
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription&model=mock") as ws:
            ws.receive_json()  # created
            time.sleep(0.1)
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["error"]["code"] == "session_timeout"
            closed = ws.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 4408


def test_realtime_multi_utterance_distinct_item_ids_and_deltas_reconcile():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?intent=transcription&model=mock") as ws:
            ws.receive_json()  # created
            audio = (
                make_tone(600) + make_silence(300) + make_tone(600) + make_silence(300)
            )
            ws.send_text(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode(),
            }))
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            msgs = []
            completed = []
            while len(completed) < 2:
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] == "conversation.item.input_audio_transcription.completed":
                    completed.append(m)

    item_ids = {m["item_id"] for m in completed}
    assert len(item_ids) == 2
    assert completed[0]["transcript"] == "The quick brown fox."
    assert completed[1]["transcript"] == "Hello world."
    for c in completed:
        iid = c["item_id"]
        deltas = [
            m["delta"] for m in msgs
            if m["type"] == "conversation.item.input_audio_transcription.delta"
            and m["item_id"] == iid
        ]
        assert "".join(deltas).casefold() == c["transcript"].casefold()


# -- observability: session.summary log line + Prometheus metrics --


def test_realtime_logs_session_summary():
    app = create_app(make_test_settings())
    with capture_logs() as cap:
        with TestClient(app) as client:
            with client.websocket_connect(
                "/v1/realtime?intent=transcription&model=mock"
            ) as ws:
                ws.receive_json()  # created
                audio = make_tone(600) + make_silence(300)
                ws.send_text(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode(),
                }))
                ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
                collect_until_completed(ws)
    summary = next(c for c in cap if c["event"] == "session.summary")
    assert summary["utterance_count"] == 1
    assert summary["audio_seconds"] > 0
    assert len(summary["final_latencies_ms"]) == 1


def test_realtime_observes_prometheus_metrics():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/realtime?intent=transcription&model=mock"
        ) as ws:
            ws.receive_json()  # created
            audio = make_tone(600) + make_silence(300)
            ws.send_text(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode(),
            }))
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            collect_until_completed(ws)
    assert UTTERANCES.labels(backend="mock", end_reason="silence")._value.get() == 1.0
    assert AUDIO_SECONDS.labels(api="realtime")._value.get() > 0.0


def test_realtime_session_timeout_increments_rejections_counter():
    s = make_test_settings()
    s.limits = LimitsConfig(max_session_seconds=0.05)
    app = create_app(s)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/realtime?intent=transcription&model=mock"
        ) as ws:
            ws.receive_json()  # created
            time.sleep(0.1)
            ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
            ws.receive_json()
    assert REJECTIONS.labels(reason="session_timeout")._value.get() == 1.0
