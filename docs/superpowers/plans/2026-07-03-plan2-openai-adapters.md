# Plan 2: OpenAI Protocol Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** OpenAI-compatible Realtime transcription WebSocket and file-transcription HTTP endpoints as pure encoders over the existing session core, plus auth/limits enforcement, the stabilizer audio-time clock fix, and a tested compat matrix.

**Architecture:** Protocol adapters live in `src/stt_server/api/` and convert wire formats to/from `TranscriptEvent` — no ASR logic. File mode reuses the streaming pipeline by pushing decoded audio faster than real time (which is why the stabilizer must switch to audio-time first). See spec §3.3, §5.

**Tech Stack:** Same as Plan 1 (Python 3.12+, uv, FastAPI, pytest). No new runtime dependencies; file decode uses stdlib `wave` for 16 kHz mono PCM16 WAV (other formats are Plan 3, with soundfile/ffmpeg).

**This is Plan 2 of 4.** Backlog items absorbed from `.superpowers/sdd/progress.md`: stabilizer audio-time decision (DECIDED), TranscriptEvent `message` field, stabilizer empty-string test, unknown-control-type error, real-socket WS smoke test.

## Global Constraints

- Python **3.12+**; all tooling through **uv**. On this machine prefix every uv command: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv ...`; check `git status --short uv.lock` before each commit (restore if dirty; never commit it).
- Internal audio format: **PCM16 mono 16 kHz little-endian**.
- `src/stt_server/core/` and `src/stt_server/backends/` must never import from `src/stt_server/api/`.
- Encoders are pure functions `TranscriptEvent → wire dict`; **no compatibility claim without a test** (spec §5.1).
- OpenAI event vocabulary per spec §5.1 (beta-style `transcription_session.*` names); GA renames (`session.update` with `session.type: "transcription"`) documented as unsupported in the compat matrix, not implemented.
- CI stays ML-free; all tests run against the mock backend.
- All new config is externalized (already-existing `AuthConfig.tokens`, `LimitsConfig.max_sessions`).

---

### Task 1: Stabilizer audio-time clock fix

The final review DECIDED this: `Session` currently feeds `time.monotonic()*1000` to `Stabilizer.update()`, so faster-than-real-time streams (file mode, tests) never satisfy `min_stable_ms`. Switch to backend audio time.

**Files:**
- Modify: `src/stt_server/core/session.py` (the `_read_backend` partial branch)
- Test: `tests/core/test_session.py` (new test), `tests/core/test_stabilizer.py` (empty-string backlog test)

**Interfaces:**
- Consumes: `BackendEvent.audio_time_ms` (Task 4 of Plan 1).
- Produces: unchanged public API; stabilizer commits now depend only on audio-time progression.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_session.py`:

```python
async def test_stabilizer_uses_audio_time_not_wall_clock():
    """Faster-than-real-time input must still commit stable prefixes.

    With min_stable_ms=150 and partials at 100ms audio intervals, the second
    partial (audio time 200ms) must commit the token first seen at 100ms even
    though wall-clock elapsed is ~0ms.
    """
    session = Session(
        session_id="s-clock",
        backend=MockBackend(
            partial_interval_ms=100.0,
            scripts=[
                MockUtteranceScript(
                    partials=("hello", "hello world", "hello world again"),
                    final="Hello world again.",
                )
            ],
        ),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(
            StabilizerConfig(min_partials=2, min_stable_ms=150.0)
        ),
    )
    events = await run_session(session, make_tone(600) + make_silence(300))
    stabilized = [e for e in events if e.type == EventType.STABILIZED]
    # Wall-clock elapsed is near zero; only audio time can satisfy 150ms.
    assert stabilized, "no STABILIZED events — stabilizer is on the wall clock"
    assert stabilized[0].text == "hello"
```

Append to `tests/core/test_stabilizer.py` (backlog item):

```python
def test_empty_update_is_harmless():
    st = Stabilizer(CFG)
    st.update("hello world", now_ms=0)
    u = st.update("", now_ms=500)
    assert u.stable_text == ""      # nothing committed yet, nothing invented
    assert u.volatile_text == ""
    assert u.newly_committed == ""
```

- [ ] **Step 2: Run tests to verify the session test fails**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest tests/core/test_session.py::test_stabilizer_uses_audio_time_not_wall_clock tests/core/test_stabilizer.py::test_empty_update_is_harmless -v`
Expected: session test FAILS on `assert stabilized` (wall clock ~0ms < 150ms); stabilizer test may already pass (behavior correct by inspection — keep it as a pin).

- [ ] **Step 3: Implement**

In `src/stt_server/core/session.py`, `_read_backend`, replace the stabilizer call:

```python
                upd = self._stabilizer.update(bev.text, time.monotonic() * 1000.0)
```

with:

```python
                upd = self._stabilizer.update(bev.text, bev.audio_time_ms)
```

- [ ] **Step 4: Run the full suite**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest`
Expected: 48 passed (46 + 2 new). Note: existing tests use `min_stable_ms=0.0`, so they are insensitive to the clock source; if any fail, the change is wrong — stop and re-read.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core/session.py tests/core/test_session.py tests/core/test_stabilizer.py
git commit -m "fix: stabilizer consumes backend audio time, not wall clock"
```

---

### Task 2: ERROR message field + unknown-control-type error

Backlog items: `TranscriptEvent` has no `message`, so ERROR events carry no human-readable text; native WS silently ignores unknown control types.

**Files:**
- Modify: `src/stt_server/core/events.py` (add field), `src/stt_server/core/session.py` (populate it), `src/stt_server/api/native_ws.py` (encode it; unknown-control error)
- Test: `tests/core/test_session.py`, `tests/api/test_native_ws.py`

**Interfaces:**
- Produces: `TranscriptEvent.message: str = ""` — human-readable error text; `encode_native` includes `"message"` for ERROR events; native WS answers unknown control types with `{"type": "error", "code": "bad_request", "recoverable": true, "message": "unknown control type: <t>"}` and keeps the session alive.

- [ ] **Step 1: Write the failing tests**

In `tests/core/test_session.py`, extend `test_backend_failure_emits_error_and_ends_stream` with:

```python
    assert "boom" in errors[0].message
```

Append to `tests/api/test_native_ws.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest tests/core/test_session.py tests/api/test_native_ws.py -v`
Expected: the two touched tests FAIL (`message` attribute missing / no error frame for bogus type).

- [ ] **Step 3: Implement**

`src/stt_server/core/events.py` — add to `TranscriptEvent` after `error_code`:

```python
    message: str = ""  # human-readable detail, primarily for ERROR events
```

`src/stt_server/core/session.py` — both ERROR emission sites gain `message=str(exc)`:

```python
            self._emit(
                EventType.ERROR,
                error_code="backend_error",
                recoverable=False,
                message=str(exc),
            )
```

`src/stt_server/api/native_ws.py` — in `encode_native`'s ERROR branch add:

```python
        out["message"] = ev.message
```

and in the control-frame handling, after the `input_done` check, add an else branch that sends the `bad_request` error (same try/except-send pattern as the malformed-JSON path) with message `f"unknown control type: {control.get('type')!r}"` and continues the loop.

- [ ] **Step 4: Run the full suite**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest && UV_DEFAULT_INDEX="https://pypi.org/simple" uv run ruff check .`
Expected: 49 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core/events.py src/stt_server/core/session.py src/stt_server/api/native_ws.py tests/core/test_session.py tests/api/test_native_ws.py
git commit -m "feat: ERROR events carry a message; native WS rejects unknown control types"
```

---

### Task 3: Auth + session limits

Spec §5.4: static bearer-token list; config-enforced max concurrent sessions. Applies to all three APIs (native WS, Realtime WS, file HTTP). Token list empty → auth disabled (dev default).

**Files:**
- Create: `src/stt_server/api/guards.py`
- Modify: `src/stt_server/api/app.py` (session counter on app.state), `src/stt_server/api/native_ws.py` (use guards)
- Test: `tests/api/test_guards.py`

**Interfaces:**
- Produces (in `stt_server.api.guards`):
  - `check_token(app, authorization: str | None) -> bool` — True if `settings.auth.tokens` is empty or `authorization == f"Bearer {t}"` for some configured `t`.
  - `class SessionSlots` — `__init__(limit: int)`, `acquire() -> bool` (False when full), `release()`, `.active: int`. Stored as `app.state.slots` by `create_app`.
  - WS behavior (native + realtime): unauthorized → send `{"type":"error","code":"unauthorized","recoverable":false,...}` then close(4401); no capacity → code `"capacity"`, close(4429).
  - HTTP behavior: unauthorized → 401 JSON `{"error": {"code": "unauthorized", "message": ...}}`; no capacity → 429 same shape with code `"capacity"`.

- [ ] **Step 1: Write the failing tests**

`tests/api/test_guards.py`:

```python
import json

from fastapi.testclient import TestClient

from stt_server.api.app import create_app
from stt_server.api.guards import SessionSlots
from stt_server.config.settings import AuthConfig, LimitsConfig, Settings
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest tests/api/test_guards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_server.api.guards'`.

- [ ] **Step 3: Implement**

`src/stt_server/api/guards.py`:

```python
"""Auth and capacity guards shared by all protocol adapters."""

from __future__ import annotations

from fastapi import FastAPI


def check_token(app: FastAPI, authorization: str | None) -> bool:
    tokens = app.state.settings.auth.tokens
    if not tokens:
        return True
    if authorization is None or not authorization.startswith("Bearer "):
        return False
    return authorization.removeprefix("Bearer ") in tokens


class SessionSlots:
    """Bounded counter for concurrent sessions. Single event loop: no locking."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.active >= self._limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)
```

`src/stt_server/api/app.py` — in `create_app`, after `app.state.settings = settings`:

```python
    app.state.slots = SessionSlots(limit=settings.limits.max_sessions)
```

(with `from stt_server.api.guards import SessionSlots`.)

`src/stt_server/api/native_ws.py` — at the top of `ws_transcribe`, after `await ws.accept()` and before backend resolution:

```python
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
```

and wrap everything after the acquire in `try: ... finally: ws.app.state.slots.release()` (the release joins the existing `finally` block; take care that the early error/close paths before acquire do NOT release).

- [ ] **Step 4: Run the full suite**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest && UV_DEFAULT_INDEX="https://pypi.org/simple" uv run ruff check .`
Expected: 53 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/api/guards.py src/stt_server/api/app.py src/stt_server/api/native_ws.py tests/api/test_guards.py
git commit -m "feat: bearer-token auth and concurrent-session limits on the API layer"
```

---

### Task 4: OpenAI Realtime transcription WebSocket adapter

Spec §5.1. Endpoint `GET /v1/realtime?intent=transcription`. Beta-style event vocabulary; deltas are sourced from committed-prefix growth (`STABILIZED` events) — an exact semantic match for OpenAI's append-only deltas.

**Files:**
- Create: `src/stt_server/api/realtime_ws.py`
- Modify: `src/stt_server/api/app.py` (include router)
- Test: `tests/api/test_realtime_ws.py`

**Interfaces:**
- Consumes: `Session`, `guards.check_token`/`slots`, `resolve_backend`, `TranscriptEvent`/`EventType`, `AudioChunk`.
- Produces (in `stt_server.api.realtime_ws`):
  - `encode_realtime(ev: TranscriptEvent, item_id: str) -> dict | None` — pure encoder; returns None for event types with no OpenAI representation (`PARTIAL` — OpenAI has no stable/volatile split).
  - `item_id_for(session_id: str, utterance_id: int) -> str` = `f"item_{session_id[:8]}_{utterance_id}"`.
  - Client → server events handled: `transcription_session.update` (acknowledged with `transcription_session.updated`; `model` in `input_audio_transcription` selects backend at session start via query param `model` instead — document), `input_audio_buffer.append` (`{"audio": "<base64 pcm16>"}`), `input_audio_buffer.commit` (maps to `end_input` — flush).
  - Server → client: `transcription_session.created` (on connect), `transcription_session.updated`, `input_audio_buffer.speech_started` (`audio_start_ms`, `item_id`) ← SPEECH_START, `input_audio_buffer.speech_stopped` (`audio_end_ms`, `item_id`) ← SPEECH_END, `conversation.item.input_audio_transcription.delta` (`item_id`, `content_index: 0`, `delta`) ← STABILIZED, `conversation.item.input_audio_transcription.completed` (`item_id`, `content_index: 0`, `transcript`) ← FINAL, `error` (`{"type":"error","error":{"type":"invalid_request_error","code":...,"message":...}}`) ← ERROR/protocol errors.
  - **Delta/completed consistency rule:** on FINAL, if the concatenated deltas sent for the utterance do not equal the final transcript, send one catch-up delta with the remainder before `completed`. (`FINAL.text` is authoritative; stabilizer commits may lag.)
  - Binary WS frames are also accepted as audio (equivalent to `append`), matching our native pattern; documented as an extension in the compat matrix.

- [ ] **Step 1: Write the failing tests**

`tests/api/test_realtime_ws.py`:

```python
import base64
import json

from fastapi.testclient import TestClient

from stt_server.api.app import create_app
from stt_server.api.realtime_ws import encode_realtime, item_id_for
from stt_server.core.events import EventType, TranscriptEvent
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
    # deltas concatenate to the final transcript (catch-up rule)
    deltas = [m["delta"] for m in msgs
              if m["type"] == "conversation.item.input_audio_transcription.delta"]
    joined = "".join(deltas)
    assert joined.split() == completed["transcript"].split() or joined == completed["transcript"]
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest tests/api/test_realtime_ws.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_server.api.realtime_ws'`.

- [ ] **Step 3: Implement**

`src/stt_server/api/realtime_ws.py`:

```python
"""OpenAI Realtime transcription-intent adapter (beta-style event vocabulary).

Pure protocol translation over the session core. Deltas are sourced from
STABILIZED events (committed-prefix growth) — append-only by construction.
See docs/openai-compat.md for the tested compatibility matrix."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
import uuid

import structlog
from fastapi import APIRouter, WebSocket

from stt_server.api.guards import check_token
from stt_server.backends.base import BackendUnavailableError
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import make_vad

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
    try:
        backend = resolve_backend(ws.app, model)
    except BackendUnavailableError as exc:
        await ws.send_json(_error(exc.code, str(exc)))
        await ws.close(code=4404)
        return
    if not ws.app.state.slots.acquire():
        await ws.send_json(_error("capacity", "max concurrent sessions reached"))
        await ws.close(code=4429)
        return

    settings = ws.app.state.settings
    session = Session(
        session_id=uuid.uuid4().hex,
        backend=backend,
        vad=make_vad(settings.vad),
        endpointer=Endpointer(settings.endpointing),
        stabilizer_factory=lambda: Stabilizer(settings.stabilizer),
    )
    log = logger.bind(session_id=session.session_id, api="realtime", model=model)

    async def sender() -> None:
        # Track per-utterance delta text to enforce the catch-up rule.
        sent: dict[int, list[str]] = {}
        async for ev in session.events():
            iid = item_id_for(ev.session_id, ev.utterance_id)
            if ev.type is EventType.STABILIZED:
                sent.setdefault(ev.utterance_id, []).append(ev.text)
            if ev.type is EventType.FINAL:
                already = " ".join(sent.get(ev.utterance_id, []))
                if already.split() != ev.text.split():
                    remainder = ev.text[len(already):].strip() if ev.text.startswith(already) else ev.text
                    catch_up = encode_realtime(
                        TranscriptEvent(
                            type=EventType.STABILIZED, session_id=ev.session_id,
                            utterance_id=ev.utterance_id, seq=ev.seq,
                            audio_time_ms=ev.audio_time_ms, emitted_ts=ev.emitted_ts,
                            text=remainder,
                        ),
                        iid,
                    )
                    if remainder:
                        await ws.send_json(catch_up)
            wire = encode_realtime(ev, iid)
            if wire is not None:
                await ws.send_json(wire)

    await ws.send_json({
        "type": "transcription_session.created",
        "session": {"id": session.session_id, "intent": "transcription"},
    })
    log.info("session.opened")
    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
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
                try:
                    audio = base64.b64decode(event.get("audio", ""), validate=True)
                except (binascii.Error, ValueError):
                    await ws.send_json(_error("bad_request", "audio must be base64"))
                    continue
                await session.push_audio(AudioChunk(data=audio, ingest_ts=time.monotonic()))
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
        await session.abort()
        if not send_task.done():
            send_task.cancel()
        import contextlib
        with contextlib.suppress(BaseException):
            await send_task
        ws.app.state.slots.release()
        log.info("session.closed")
```

In `src/stt_server/api/app.py` add `from stt_server.api import native_ws, realtime_ws` and `app.include_router(realtime_ws.router)`.

Note for the implementer: move `import contextlib` to the top of the file (shown inline above only for locality); ruff will flag it otherwise.

- [ ] **Step 4: Run the full suite**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest && UV_DEFAULT_INDEX="https://pypi.org/simple" uv run ruff check .`
Expected: 56 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/api/realtime_ws.py src/stt_server/api/app.py tests/api/test_realtime_ws.py
git commit -m "feat: OpenAI Realtime transcription-intent WebSocket adapter"
```

---

### Task 5: File transcription HTTP endpoint

Spec §5.2. `POST /v1/audio/transcriptions`, multipart (`file`, `model`, optional `response_format` ∈ {json, text, verbose_json}, optional `language`). Decodes 16 kHz mono PCM16 WAV via stdlib `wave`; other formats/rates → 400 `unsupported_format` (Plan 3 adds soundfile/ffmpeg). Streams decoded audio through the standard session pipeline faster than real time; response concatenates utterance finals.

**Files:**
- Create: `src/stt_server/api/transcriptions_http.py`
- Modify: `src/stt_server/api/app.py` (include router)
- Test: `tests/api/test_transcriptions_http.py`, helper in `tests/helpers/audio.py`

**Interfaces:**
- Consumes: `Session`, guards, `resolve_backend`.
- Produces:
  - `run_file_session(app, model, pcm: bytes, language: str | None) -> list[TranscriptEvent]` — builds a Session, pushes the whole PCM buffer, `end_input()`, collects events. Raises `BackendUnavailableError` for unknown models.
  - Response shapes: `json` → `{"text": "<finals joined by space>"}`; `text` → plain text body; `verbose_json` → `{"task": "transcribe", "language": <lang or "en">, "duration": <seconds float>, "text": ..., "segments": [{"id": i, "start": <s>, "end": <s>, "text": ...} per FINAL]}` with `start`/`end` from each utterance's SPEECH_START/FINAL `audio_time_ms`.
  - Errors: 401/429 per Task 3 shapes; 404 `{"error": {"code": "model_not_found", ...}}`; 400 `{"error": {"code": "unsupported_format", ...}}` for non-WAV/wrong rate/stereo; 400 `bad_request` for missing file.
- Test helper: `make_wav(pcm: bytes, rate: int = 16000, channels: int = 1) -> bytes` added to `tests/helpers/audio.py` (uses `wave` + `io.BytesIO`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/helpers/audio.py`:

```python
def make_wav(pcm: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
```

`tests/api/test_transcriptions_http.py`:

```python
from fastapi.testclient import TestClient

from stt_server.api.app import create_app
from stt_server.config.settings import AuthConfig, Settings
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest tests/api/test_transcriptions_http.py -v`
Expected: FAIL with 404s (route does not exist).

- [ ] **Step 3: Implement**

`src/stt_server/api/transcriptions_http.py`:

```python
"""OpenAI Audio Transcriptions-style file endpoint.

File mode reuses the streaming pipeline (spec §3.3): decoded PCM is pushed
through a normal Session faster than real time, and the response is built
from the resulting TranscriptEvents."""

from __future__ import annotations

import io
import time
import uuid
import wave
from typing import Annotated

import structlog
from fastapi import APIRouter, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from stt_server.api.guards import check_token
from stt_server.backends.base import BackendUnavailableError
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import make_vad

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
    except wave.Error as exc:
        raise UnsupportedFormatError(f"not a WAV file: {exc}") from exc


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message}})


async def run_file_session(app, model: str, pcm: bytes,
                           language: str | None) -> list[TranscriptEvent]:
    from stt_server.api.app import resolve_backend  # local import: avoid cycle
    from stt_server.backends.base import StreamConfig

    backend = resolve_backend(app, model)
    settings = app.state.settings
    session = Session(
        session_id=uuid.uuid4().hex,
        backend=backend,
        vad=make_vad(settings.vad),
        endpointer=Endpointer(settings.endpointing),
        stabilizer_factory=lambda: Stabilizer(settings.stabilizer),
        stream_config=StreamConfig(language=language),
    )
    events: list[TranscriptEvent] = []

    async def collect() -> None:
        async for ev in session.events():
            events.append(ev)

    import asyncio

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=pcm, ingest_ts=time.monotonic()))
    await session.end_input()
    await task
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
        return _err(401, "unauthorized", "missing or invalid bearer token")
    if response_format not in ("json", "text", "verbose_json"):
        return _err(400, "bad_request", f"unsupported response_format {response_format!r}")
    if not app.state.slots.acquire():
        return _err(429, "capacity", "max concurrent sessions reached")
    try:
        data = await file.read()
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
```

In `src/stt_server/api/app.py`: import and `app.include_router(transcriptions_http.router)`. Move the local `import asyncio` in `run_file_session` to module top (shown inline for locality; ruff will flag it).

Note on `starts`: SPEECH_START's `audio_time_ms` is the stream position at utterance start (Plan 1 semantics), giving segment start; FINAL's is the end position.

- [ ] **Step 4: Run the full suite**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest && UV_DEFAULT_INDEX="https://pypi.org/simple" uv run ruff check .`
Expected: 62 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/api/transcriptions_http.py src/stt_server/api/app.py tests/api/test_transcriptions_http.py tests/helpers/audio.py
git commit -m "feat: OpenAI-style file transcription endpoint reusing the streaming pipeline"
```

---

### Task 6: Compat matrix doc + real-socket smoke test

Spec §5.1: `docs/openai-compat.md` with a full event matrix — supported / partially supported / not supported, each row citing its test. Backlog: one true-network-socket test (uvicorn in a thread) so coverage isn't TestClient-only.

**Files:**
- Create: `docs/openai-compat.md`, `tests/api/test_real_socket.py`

**Interfaces:**
- Consumes: everything shipped in Tasks 4-5.
- Produces: documentation + smoke test; no new runtime code.

- [ ] **Step 1: Write the real-socket test**

`tests/api/test_real_socket.py`:

```python
"""One true-network smoke test: uvicorn on a real port, websockets client.

Everything else uses TestClient (in-process ASGI); this guards the
socket-level path (HTTP upgrade, binary frames over TCP)."""

import asyncio
import base64
import json
import socket
import threading
import time

import pytest
import uvicorn
from websockets.asyncio.client import connect

from stt_server.api.app import create_app
from tests.api.test_native_ws import make_test_settings
from tests.helpers.audio import make_silence, make_tone


@pytest.fixture(scope="module")
def server_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(create_app(make_test_settings()), host="127.0.0.1",
                            port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn failed to start")
        time.sleep(0.05)
    yield port
    server.should_exit = True
    thread.join(timeout=5)


async def test_realtime_over_real_socket(server_port):
    uri = f"ws://127.0.0.1:{server_port}/v1/realtime?intent=transcription&model=mock"
    async with connect(uri) as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "transcription_session.created"
        audio = make_tone(600) + make_silence(300)
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode(),
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        transcript = None
        async with asyncio.timeout(10):
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "conversation.item.input_audio_transcription.completed":
                    transcript = m["transcript"]
                    break
        assert transcript == "The quick brown fox."
```

- [ ] **Step 2: Run it**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest tests/api/test_real_socket.py -v`
Expected: 1 passed (a few seconds — real server boot).

- [ ] **Step 3: Write docs/openai-compat.md**

Content requirements (write actual tables, citing test functions by name):

- **Realtime transcription WS** (`/v1/realtime?intent=transcription`), beta-style vocabulary. Rows: `transcription_session.update` (partially supported — acknowledged verbatim, session config like `turn_detection`/`model` not applied; model selected via `?model=` query param instead → `test_realtime_transcription_end_to_end`); `input_audio_buffer.append` (supported, base64 PCM16 mono 16 kHz only — OpenAI uses 24 kHz default; document rate difference → same test); `input_audio_buffer.commit` (supported = flush+close; OpenAI keeps sessions open — documented difference); binary frames (extension, not in OpenAI protocol); server events `transcription_session.created/updated`, `speech_started/speech_stopped` (supported), `input_audio_buffer.committed`, `conversation.item.added` (NOT supported — not emitted); `delta`/`completed` (supported, deltas from committed-prefix growth, catch-up rule → encoder table test); `error` (supported, OpenAI error envelope).
- Note that the GA API renamed the vocabulary (`session.update` with `session.type: "transcription"`, `gpt-realtime-whisper` models); this server implements the beta-style names only.
- **Transcriptions HTTP** (`POST /v1/audio/transcriptions`). Rows: `file` (WAV 16 kHz mono PCM16 only — OpenAI accepts many formats), `model` (maps to configured backends), `response_format` json/text/verbose_json (supported → tests), srt/vtt (NOT supported), `language` (passed to backend), `prompt`/`temperature`/`timestamp_granularities` (NOT supported, ignored is-vs-error: they are **ignored silently**, document this), auth via `Authorization: Bearer` (supported).
- State the blanket rule from the spec: no compatibility claim without a test; every "supported" row names its test.

- [ ] **Step 4: Run the full suite**

Run: `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest && UV_DEFAULT_INDEX="https://pypi.org/simple" uv run ruff check .`
Expected: 63 passed, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add docs/openai-compat.md tests/api/test_real_socket.py
git commit -m "docs: tested OpenAI compatibility matrix; test: real-socket smoke test"
```

---

## Plan 2 exit criteria

- Full suite green (63 tests), ruff clean, uv.lock untouched by any commit.
- `POST /v1/audio/transcriptions` returns correct json/text/verbose_json against the mock backend.
- A Realtime client can stream base64 audio and receive `speech_started` → deltas → `completed` over a real socket.
- Auth + capacity enforced on all three APIs (native WS, realtime WS, file HTTP).
- `docs/openai-compat.md` exists; every "supported" claim cites a test.
- Layering grep still clean: `grep -r "stt_server.api" src/stt_server/core src/stt_server/backends` → no matches.


