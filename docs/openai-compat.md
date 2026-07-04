# OpenAI compatibility matrix

This document is the tested compatibility surface for the two OpenAI-style
adapters this server ships: the Realtime transcription WebSocket
(`/v1/realtime?intent=transcription`) and the Transcriptions HTTP endpoint
(`POST /v1/audio/transcriptions`). It describes actual, shipped behavior —
not the aspirational behavior in the design spec.

**Blanket rule: no compatibility claim without a test.** Every row marked
*Supported* below cites the test function(s) in `tests/api/` that exercise
it. Rows marked *Partially supported* or *Not supported* describe a gap or
divergence from the OpenAI API and are grounded in the adapter source
(`src/stt_server/api/realtime_ws.py`, `src/stt_server/api/transcriptions_http.py`)
where no test is cited.

## Realtime transcription WebSocket — `/v1/realtime?intent=transcription`

This server implements the **beta-style** transcription-intent event
vocabulary (`transcription_session.*`, `input_audio_buffer.*`,
`conversation.item.input_audio_transcription.*`). It does **not** implement
the GA Realtime API's renamed vocabulary — `session.update` with
`session.type: "transcription"`, `gpt-realtime-whisper` model IDs, etc. A
client written against the GA naming will not work against this server
without translation.

### Client → server events

| Event | Support | Notes | Test |
|---|---|---|---|
| `transcription_session.update` | Partially supported | Acknowledged verbatim: the server echoes the submitted `session` object back as `transcription_session.updated` without validating or applying any of it. Config such as `turn_detection`, `input_audio_transcription.model`, or language hints has **no effect** — the backend model is selected once, at connect time, via the `?model=` query parameter, not through this event. | `test_realtime_transcription_end_to_end` |
| `input_audio_buffer.append` | Supported | `audio` must be base64-encoded PCM16 mono **16 kHz**. OpenAI's Realtime API defaults to 24 kHz PCM16 — clients targeting this server must send/resample to 16 kHz, not the OpenAI default. | `test_realtime_transcription_end_to_end` |
| `input_audio_buffer.commit` | Supported, different semantics | Treated as flush-and-close: the session ends its input, emits any final transcript, and the server then closes the WebSocket. The real OpenAI API keeps the session open after a commit so further turns can follow. Here, one connection = one commit = one utterance stream. | `test_realtime_transcription_end_to_end` |
| Binary WebSocket frames | Extension (not part of the OpenAI protocol) | Raw PCM16 bytes sent as a binary WS frame are accepted as an alternative to base64 `input_audio_buffer.append` (see the `msg.get("bytes")` branch in `realtime_ws.py`). **Not exercised by any test on this route** — `tests/api/test_real_socket.py` only sends base64 JSON over the real socket; binary-frame coverage exists only for the unrelated native `/ws/transcribe` route (`test_native_ws.py`, `test_guards.py`). Treat binary frames on `/v1/realtime` as untested. |

### Server → client events

| Event | Support | Notes | Test |
|---|---|---|---|
| `transcription_session.created` | Supported | Sent immediately on accept, before any client message. | `test_realtime_transcription_end_to_end` |
| `transcription_session.updated` | Supported (echo only — see the partial-support note above) | | `test_realtime_transcription_end_to_end` |
| `input_audio_buffer.speech_started` / `speech_stopped` | Supported | | `test_encoder_table`, `test_realtime_transcription_end_to_end` |
| `input_audio_buffer.committed` | Not supported | Never emitted — `encode_realtime()` has no case that produces it, and the commit handler doesn't send one either. | |
| `conversation.item.added` | Not supported | Never emitted — no code path constructs this event type. | |
| `conversation.item.input_audio_transcription.delta` | Supported | Sourced from `STABILIZED` events (committed-prefix growth), so deltas are append-only by construction. `STABILIZED.text` is a bare token ("the", "quick", ...) with no inter-word space of its own; the adapter tracks exactly what's already been sent per utterance and injects the missing space on the wire, so a client that reconstructs the transcript via plain `"".join(deltas)` gets correctly spaced text. `FINAL` commonly applies casing/punctuation normalization the raw deltas never had (e.g. mock backend deltas "the"/"fox" vs. FINAL "The"/"fox."); a casefold-insensitive "catch-up" delta covers any un-sent remainder so the join still matches modulo case. Casing of text already on the wire can never be corrected retroactively — an inherent, documented limitation. | `test_encoder_table`, `test_realtime_transcription_end_to_end` |
| `conversation.item.input_audio_transcription.completed` | Supported | | `test_encoder_table`, `test_realtime_transcription_end_to_end` |
| `error` | Supported | OpenAI-style envelope: `{"type": "error", "error": {"type": "invalid_request_error", "code": ..., "message": ...}}`. | `test_encoder_table`, `test_realtime_unknown_event_type_gets_error` |

Socket-level behavior (HTTP upgrade + real TCP, not just in-process
`TestClient`) is additionally covered end to end by
`test_realtime_over_real_socket` in `tests/api/test_real_socket.py`, which
boots a real `uvicorn` server on a loopback port and drives it with a real
`websockets` client.

## Transcriptions HTTP endpoint — `POST /v1/audio/transcriptions`

| Field | Support | Notes | Test |
|---|---|---|---|
| `file` | Supported, constrained | Must be a WAV container, 16 kHz mono PCM16. Any other sample rate, channel count, or bit depth is rejected with `400 unsupported_format`; a non-WAV/empty body is rejected the same way. OpenAI accepts a much broader set of containers/codecs/rates. | `test_json_response` (valid WAV succeeds), `test_wrong_rate_400` (44.1 kHz rejected), `test_empty_file_400` |
| `model` | Supported | Selects the configured backend by name. Unknown model → `404 model_not_found`. | `test_json_response`, `test_unknown_model_404` |
| `response_format=json` | Supported | | `test_json_response` |
| `response_format=text` | Supported | | `test_text_response` |
| `response_format=verbose_json` | Supported | Includes `segments` with per-segment `start`/`end` (seconds) and `text`. | `test_verbose_json_has_segments` |
| `response_format=srt` / `vtt` | Not supported | Rejected with `400 bad_request` — the handler only accepts `("json", "text", "verbose_json")`. No dedicated test; grounded in the `response_format not in (...)` check in `transcriptions_http.py`. | |
| `language` | Supported (backend-dependent honoring) | Declared as an optional multipart `Form` field, forwarded to the backend via `StreamConfig.language`, and echoed back in the `verbose_json` response's `language` field (default `"en"` when omitted). Whether the value actually changes decoding depends on the backend: `qwen3asr` honors a genuine per-request override (`cfg.language or` its constructor default); `sherpa_onnx`/`funasr` wrap a single language-fixed model and silently ignore a mismatching `language` (debug-logged once, never an error — see `docs/backends.md` §5). No current test asserts on the response's echoed `language` field for a non-default value, so that part of the row is not claimed as tested. | `test_create_stream_per_request_language_overrides_constructor_language` (qwen3asr), `test_create_stream_language_mismatch_is_ignored_without_error` (sherpa, funasr) |
| `prompt` / `temperature` / `timestamp_granularities` | Not supported, silently ignored | Not declared as `Form` fields in the endpoint signature. If a client sends them as multipart fields, FastAPI simply drops the extra fields — no error, and no effect on the response. | |
| `Authorization: Bearer <token>` | Supported | | `test_auth_enforced` |
| Missing `file` part | Framework-level `422` | `file` is a required `UploadFile` parameter; if it's omitted, FastAPI's own request validation rejects the request with `422` before the endpoint body ever runs. This is Starlette/FastAPI behavior, not application code, and has no project test. | |
| Error envelope | Partially supported, different shape | Every error response from this endpoint is `{"error": {"code": ..., "message": ...}}` (see `_err()` in `transcriptions_http.py`). OpenAI's documented envelope is `{"error": {"type": ..., "code": ..., "message": ..., "param": ...}}` — this server never sends `type` or `param`. A client that reads `error.type` or `error.param` from this endpoint's responses will not find them. | `test_unknown_model_404`, `test_wrong_rate_400`, `test_auth_enforced`, `test_backend_failure_returns_500`, `test_http_capacity_429`, `test_upload_too_large_413_content_length_precheck` |

## Native protocol (out of scope for OpenAI compatibility)

`/ws/transcribe` is this project's own thin JSON protocol (stable/volatile
partials, `session.closed`, etc.) and makes no OpenAI-compatibility claims —
see `tests/api/test_native_ws.py` and `tests/api/test_guards.py` for its
behavior.
