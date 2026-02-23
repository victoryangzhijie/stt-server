# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

In the real CUDA production environment, the STT_BACKEND must prioritize Qwen3.

```bash
# Install dependencies
uv sync --group dev

# Install with Qwen3 backend (requires CUDA)
uv sync --group dev --extra qwen3

# Run mock tests (no GPU needed)
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v

# Run qwen3 tests (requires CUDA)
STT_BACKEND=qwen3 .venv/bin/python -m pytest tests/qwen3/ -v

# Run a single mock test file
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_qwen3_backend.py -v

# Run a single test
STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_qwen3_backend.py::TestVAD::test_endpoint_fires_once -v

# Start the server (mock backend)
STT_BACKEND=mock uvicorn app:app --host 0.0.0.0 --port 8000

# Start the server (qwen3 backend, requires CUDA)
STT_BACKEND=qwen3 STT_VLLM_MODEL=/path/to/Qwen3-ASR-1.7B uvicorn app:app --host 0.0.0.0 --port 8000
```

## Architecture

Real-time STT WebSocket server. Each WebSocket connection spawns three concurrent tasks:

```
Client ──ws──► _recv_pump (async) ──► janus in_queue ──► run_worker (thread) ──► janus out_queue ──► _send_pump (async) ──ws──► Client
```

- **`transport/ws.py`** — WebSocket lifecycle: accepts connection, validates query params, launches the three tasks via `asyncio.gather`, handles cleanup.
- **`inference/worker.py`** — Sync worker thread that drains the input queue, pushes audio to the backend, checks VAD endpoints, emits partial/final results. Uses sentinel objects `COMMIT` and `STOP` for control flow.
- **`backends/base.py`** — `ASRBackend` Protocol that all backends implement. `ASRResult` dataclass for return values.
- **`backends/registry.py`** — Factory using lazy `importlib` import from string paths. Selected via `STT_BACKEND` env var.
- **`session/state.py`** — Per-connection state: segment sequencing, partial throttling (`should_send_partial()`), timing computation for final messages.
- **`protocol/schema.py`** — `ConnParams` validation from query string, message builder functions (`build_ready`, `build_partial`, `build_final`, `build_error`).
- **`observability/`** — `logging.py` has structlog setup + `TroubleshootCollector` (ring buffer of events emitted on abnormal close). `metrics.py` defines all Prometheus counters/histograms.

## Scaling

The global `_infer_lock` in `backends/qwen3.py` serializes all inference calls because vLLM is not thread-safe. This means a single process handles one inference at a time, which becomes a bottleneck under high concurrency.

**Recommended horizontal scaling approach:**
- Run multiple **uvicorn worker processes** (`uvicorn app:app --workers N`), each loading an independent model replica
- **GPU memory planning**: Qwen3-ASR-1.7B in fp16 requires ~3.4 GB per replica. Plan worker count based on available VRAM (e.g., 2-3 workers on a 24 GB GPU)
- Use a **load balancer** (e.g., nginx with `least_conn`) to distribute WebSocket connections across workers
- Each worker has its own `_asr_model` singleton and `_infer_lock`, so inference is parallelized across workers

## Key Patterns

- **Async/sync bridge**: FastAPI async on the outside, sync vLLM inference in a thread, connected by janus queues. The worker reads `in_q_sync` and writes `out_q_sync`.
- **Backend singleton**: `backends/qwen3.py` uses a module-level `_asr_model` (`Qwen3ASRModel`) with double-checked locking via `_asr_lock`. The model loads once on first connection, shared across all connections. All inference calls (`streaming_transcribe`, `finish_streaming_transcribe`) are serialized via `_infer_lock`.
- **Streaming inference**: `qwen-asr` handles chunked inference internally. Audio is fed via `streaming_transcribe()` in `push_audio()` — most calls just buffer (microseconds), inference triggers every `chunk_size_sec` (default 0.5s). `get_partial()` is a pure read of `state.text`. Per-connection `ASRStreamingState` is created in `configure()` and rebuilt in `reset_segment()`.
- **Lazy imports**: `qwen_asr` and `numpy` are imported inside functions in `qwen3.py`, so `STT_BACKEND=mock` works without GPU dependencies installed.
- **Backpressure**: Queue overflow → error event (code 1008) → connection closed. Worker also stops if output queue is full.
- **VAD**: Energy-based RMS on PCM16LE samples using numpy. State machine tracks `_in_speech` / `_silence_ms_accum` / `_endpoint_fired`.

## qwen-asr Streaming API

The Qwen3 backend uses `qwen-asr` package for streaming transcription:

```python
from qwen_asr import Qwen3ASRModel

asr = Qwen3ASRModel.LLM(model="Qwen/Qwen3-ASR-1.7B", **vllm_kwargs)
state = asr.init_streaming_state(chunk_size_sec=0.5, unfixed_chunk_num=2, unfixed_token_num=5)

# Feed audio incrementally (buffers internally, infers every chunk_size_sec)
asr.streaming_transcribe(float32_audio, state)
print(state.text)  # latest transcription

# Flush remaining buffer at end of utterance
asr.finish_streaming_transcribe(state)
```

**Key points:**
- Audio: 16kHz mono float32 numpy array
- `streaming_transcribe()` buffers audio internally, triggers inference when `chunk_size_sec` of audio accumulates
- Previously stable text is locked as prompt prefix; only last `unfixed_token_num` tokens can change
- `state.text` and `state.language` are updated after each inference step

## Configuration

All settings via `STT_` prefixed env vars (see `config.py`). Key ones: `STT_BACKEND` (mock/qwen3), `STT_VLLM_MODEL`, `STT_VAD_THRESHOLD`, `STT_VAD_SILENCE_MS` (default 500ms), `STT_STREAMING_CHUNK_SIZE_SEC` (default 0.5s), `STT_STREAMING_UNFIXED_CHUNK_NUM`, `STT_STREAMING_UNFIXED_TOKEN_NUM`.

## Testing Notes

### Test Structure

```
tests/
├── mock/     # Tests that run with STT_BACKEND=mock (no GPU needed)
└── qwen3/   # Tests that require CUDA + qwen3 backend
```

### Running Tests

| Backend | Command | GPU Required |
|---------|---------|--------------|
| mock | `STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/ -v` | No |
| qwen3 | `STT_BACKEND=qwen3 .venv/bin/python -m pytest tests/qwen3/ -v` | Yes |

### Mock Tests (`tests/mock/`)

- Run with `STT_BACKEND=mock` — no GPU needed
- Includes: unit tests for mock backend, schema, session state, VAD state machine
- Integration tests spin up a real uvicorn server in a background thread (fixture in `conftest.py`)
- Qwen3 backend tests mock `_get_asr_model` for unit tests; VAD tests also require the mock (push_audio calls streaming_transcribe)

### Qwen3 Tests (`tests/qwen3/`)
- In the real CUDA production environment, the STT_BACKEND must prioritize Qwen3.
- Run with `STT_BACKEND=qwen3` and valid model path
- Includes: real audio transcription tests, direct vLLM API tests, WebSocket client tests
- Some tests require a running server; they will skip if server is not available

### Reference Fixtures

- Reference audio in `references/` — WAV file (float32 24kHz) and expected transcription text
- pytest-asyncio with `asyncio_mode = "auto"`
