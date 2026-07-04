# Plan 3: Real Backends, Observability, Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real ASR backends (Silero VAD, sherpa-onnx, FunASR, Qwen3-ASR) behind the existing plugin contract, Prometheus metrics + structured observability, Docker/compose deployment, and the accumulated hardening backlog.

**Architecture:** Backends are optional extras (`pip install .[sherpa]` etc.); CI never installs them and stays green on the mock path. Real-model tests are marked `@pytest.mark.model` and run only on developer hardware. Each backend owns its execution strategy per spec §2.1 (sherpa/FunASR: thread pools; Qwen3-ASR: its official vLLM-based framework on GPU). Metrics are measured in the session core so numbers are comparable across backends by construction (spec §4.4, §7).

**Tech Stack:** prometheus-client; onnxruntime (silero extra); sherpa-onnx (sherpa extra); funasr+torch (funasr extra); qwen3-asr official inference framework (qwen3asr extra, GPU); Docker + compose + Grafana.

**This is Plan 3 of 4.** Plan 4: benchmarks, load generator, reports, final docs.

## Global Constraints

- Python **3.12+**; uv for everything. Machine policy: prefix every uv command `UV_DEFAULT_INDEX="https://pypi.org/simple"`; `git status --short uv.lock` before each commit — the ONLY permitted lock changes are the two dependency commits named in Tasks 1 and 5, locked against PyPI (verify `grep -c aliyun uv.lock` = 0 before committing).
- Internal audio: PCM16 mono 16 kHz LE. `core/` and `backends/` never import from `api/`.
- **CI stays ML-free**: the default `uv sync` (no extras) must keep all non-`model` tests green. Real-model tests: `@pytest.mark.model`, deselected by default via `addopts = "-m 'not model'"`.
- Backend heavy deps are optional extras; importing a backend module without its deps must raise `BackendUnavailableError` at `create_backend` time with an actionable message — the server always boots with mock.
- Every backend must pass the conformance suite (`tests/backends/conformance.py`) — mock in CI, real backends under `-m model` locally.
- Model files live under `models/` (gitignored), fetched by `scripts/download_models.py`.
- All new tunables externalized through `config/settings.py`; no config in code.

---

### Task 1: Hardening — auth, limits, guard unification

Backlog: constant-time token compare, case-insensitive Bearer scheme, WWW-Authenticate on 401, auth/capacity rejection logging, SessionSlots clamp anomaly logging, unified guard order, max upload size (pre-read Content-Length check + post-read enforcement), max session duration, realtime empty-audio error, realtime multi-utterance test.

**Files:**
- Modify: `src/stt_server/api/guards.py`, `src/stt_server/config/settings.py` (LimitsConfig), `src/stt_server/api/native_ws.py`, `src/stt_server/api/realtime_ws.py`, `src/stt_server/api/transcriptions_http.py`, `docs/openai-compat.md` (error-envelope row)
- Test: `tests/api/test_guards.py`, `tests/api/test_realtime_ws.py`, `tests/api/test_transcriptions_http.py`

**Interfaces:**
- `LimitsConfig` gains `max_upload_bytes: int = 26_214_400` (25 MiB, OpenAI's documented cap) and `max_session_seconds: float = 3600.0`.
- `guards.check_token` unchanged signature; internally: scheme parsed case-insensitively, token compared with `secrets.compare_digest` against every configured token (no early exit), logs `auth.rejected` (structlog, no token material) on failure.
- `guards.SessionSlots.release()` logs `slots.release_underflow` anomaly instead of silently clamping (still clamps).
- New `guards.session_deadline(settings) -> float` returning `time.monotonic() + settings.limits.max_session_seconds`; both WS receive loops check it per message and close with code 4408 / error code `"session_timeout"` when exceeded.
- HTTP 401 responses carry `WWW-Authenticate: Bearer` header.
- transcriptions_http: reject `Content-Length > max_upload_bytes` with 413 `"upload_too_large"` BEFORE reading the body; after read, enforce again on actual size (chunked/absent header).
- Guard order unified across both WS adapters: intent (realtime only) → token → **capacity → backend resolution** (capacity before backend, matching realtime's release-safety; native_ws reordered to match).
- realtime `input_audio_buffer.append` with missing/empty/`None` audio → `bad_request` error event (no silent empty push).
- New tests: multi-utterance realtime session (two tones separated by silence → two distinct item_ids, `wire_sent` popped, deltas reconcile per utterance); 413 upload; session-timeout (settings with `max_session_seconds=0.05`, sleep, next frame → 4408 close); case-insensitive `bearer` scheme accepted; 401 carries WWW-Authenticate.

**Steps:** TDD as usual — write the failing tests first (mirror the existing test style in each file), run RED, implement, run `UV_DEFAULT_INDEX="https://pypi.org/simple" uv run pytest` (expect all green, ~74 tests) + ruff, commit `feat: auth/limits hardening — constant-time tokens, upload/session limits, unified guards`.

Implementation notes (exact behaviors, code left to the implementer since it is spread across small diffs — the tests pin every behavior):

```python
# guards.py — token check core
import hmac, secrets
def _token_matches(presented: str, configured: list[str]) -> bool:
    ok = False
    for t in configured:
        if secrets.compare_digest(presented.encode(), t.encode()):
            ok = True  # no early exit; constant-ish across list
    return ok
# scheme parse: scheme, _, credentials = authorization.partition(" ");
# scheme.lower() == "bearer" and credentials non-empty
```

```python
# session deadline check inside both WS receive loops, per iteration:
if time.monotonic() > deadline:
    # native: {"type":"error","code":"session_timeout",...}; realtime: _error("session_timeout", ...)
    # then abort session, close(code=4408), break
```

---

### Task 2: Hardening — lifecycle, registry, mock determinism, conformance

Backlog: lifespan exception-safe startup (stop already-started backends, fail readyz with actionable log), readyz 503 test, registry wraps bad-options TypeError in BackendUnavailableError, per-stream script selection in MockBackend (deterministic under concurrent sessions — each stream takes scripts by per-backend running counter, which is what Plan 4's load generator needs), conformance test for push_audio-after-finalize no-op, VAD odd-length/empty frame handling, sender() encode/send helper refactor in realtime_ws.

**Files:**
- Modify: `src/stt_server/api/app.py`, `src/stt_server/backends/registry.py`, `src/stt_server/backends/mock.py`, `src/stt_server/core/vad.py`, `src/stt_server/api/realtime_ws.py`
- Test: `tests/api/test_app.py`, `tests/backends/test_registry.py`, `tests/backends/test_mock_backend.py`, `tests/backends/conformance.py`, `tests/core/test_vad.py`

**Interfaces:**
- Lifespan: wrap the start loop; on failure, stop started backends in reverse order, log `backend.start_failed` with backend key and exception, re-raise (uvicorn exits non-zero; readyz never true). Test with an exploding-on-start backend registered inline + a probe backend asserting its `stop()` ran.
- `create_backend`: `except TypeError as exc: raise BackendUnavailableError(f"bad options for backend type {defn.type!r}: {exc}") from exc`.
- MockBackend: replace shared `itertools.cycle` with `self._counter` incremented per `create_stream`; script = `scripts[counter % len(scripts)]`. Utterance order remains deterministic per backend instance regardless of stream interleaving.
- Conformance suite adds `test_push_after_finalize_is_noop` (push after finalize → no new events, no exception) — must pass for mock now, real backends later.
- `EnergyVad.is_speech`: empty or odd-length frame → `False` (truncate trailing odd byte), never raises. Tests: `is_speech(b"")`, `is_speech(b"\x01")`, odd-length tone.
- realtime_ws sender(): extract `async def _send(wire: dict | None)` helper; three call sites become one-liners. Pure refactor, behavior pinned by existing tests.

**Steps:** TDD; full suite green (~80 tests) + ruff; commit `feat: lifecycle/registry/mock hardening; VAD frame robustness`.

---

### Task 3: Observability — Prometheus metrics, session summaries, JSON logs everywhere

Spec §7. Metrics measured in the session core; API layer only labels.

**Files:**
- Create: `src/stt_server/metrics/__init__.py`, `src/stt_server/metrics/registry.py`
- Modify: `pyproject.toml` (add `prometheus-client>=0.21` to core deps — lightweight, no reason to make it optional; ALSO add `[tool.pytest.ini_options] addopts = "-m 'not model'"` and `markers = ["model: requires real model weights/deps"]`, and `filterwarnings` for the httpx deprecation), `src/stt_server/core/session.py` (instrument), `src/stt_server/api/app.py` (/metrics endpoint, active-session gauge via slots), `src/stt_server/api/guards.py` (rejection counters), `src/stt_server/__main__.py` (uvicorn log_config JSON), `src/stt_server/logging.py` (uvicorn dictConfig), `.github/workflows/ci.yml` (`uv sync --frozen`, drop double-run: `on: pull_request` + `push: branches: [main]`)
- Test: `tests/metrics/test_metrics.py` (+ `tests/metrics/__init__.py`)

**Interfaces (in `stt_server.metrics.registry`):**
- Module-level `REGISTRY = prometheus_client.CollectorRegistry()` (own registry, not global default — test isolation).
- `SESSIONS_ACTIVE = Gauge("stt_sessions_active", ..., registry=REGISTRY)`
- `AUDIO_SECONDS = Counter("stt_audio_seconds_ingested_total", ..., ["api"], ...)`
- `UTTERANCES = Counter("stt_utterances_total", ..., ["backend", "end_reason"], ...)`
- `FIRST_PARTIAL_MS = Histogram("stt_first_partial_latency_ms", ..., ["backend"], buckets=(50,100,200,300,500,750,1000,1500,2000,3000,5000))`
- `FINAL_MS = Histogram("stt_final_latency_ms", ..., ["backend"], same buckets)`
- `REJECTIONS = Counter("stt_rejections_total", ..., ["reason"], ...)` (reasons: unauthorized, capacity, session_timeout, upload_too_large)
- `ERRORS = Counter("stt_errors_total", ..., ["code"], ...)`
- Session instrumentation: Session gains optional `metrics_labels: dict[str, str] | None = None` ctor arg (`{"backend": ..., "api": ...}`); when set, observes the latency histograms at FINAL emission, counts utterances/errors, adds audio seconds in push_audio. Core imports `stt_server.metrics.registry` — metrics is a leaf module importing nothing from api/ (layering intact; verify grep).
- `create_app`: `GET /metrics` → `generate_latest(REGISTRY)` with correct content type; SESSIONS_ACTIVE set from `slots.active` on acquire/release (increment/decrement in SessionSlots itself, cleanest).
- Per-session summary: at session end (end_input/abort), the api adapters log `session.summary` with audio_seconds, utterance_count, and per-utterance final latencies (session exposes `Session.stats` dataclass property accumulating these).
- `logging.configure_logging` gains `uvicorn_log_config() -> dict` returning a dictConfig routing uvicorn loggers through structlog's JSON renderer; `__main__.py` passes `log_config=uvicorn_log_config()` to `uvicorn.run`.

**Tests:** metrics registry counts (drive a session end-to-end via native WS TestClient with metrics labels and assert `/metrics` body contains `stt_utterances_total{...} 1.0` etc.); /metrics content type; SESSIONS_ACTIVE returns to 0 after close; REJECTIONS increments on 401. Full suite green + ruff; NOTE pytest now runs with `-m 'not model'` by default — assert no test count regression. Commit `feat: Prometheus metrics, session summaries, uniform JSON logging` (pyproject dep change → regenerate uv.lock against PyPI in the same commit; verify aliyun grep 0).

---

### Task 4: Silero VAD (optional `silero` extra)

Spec §4.1: Silero VAD ONNX behind the existing `VadDetector` interface; energy stays the CI default.

**Files:**
- Create: `src/stt_server/core/vad_silero.py`, `scripts/download_models.py` (framework + silero entry)
- Modify: `pyproject.toml` (`[project.optional-dependencies] silero = ["onnxruntime>=1.20"]`), `src/stt_server/core/vad.py` (factory dispatch), `src/stt_server/config/settings.py` (VadConfig gains `model_path: str = "models/silero_vad.onnx"`, `threshold: float = 0.5`)
- Test: `tests/core/test_vad_silero.py`

**Interfaces:**
- `SileroVad(model_path: str, threshold: float = 0.5)` implements `VadDetector`; `is_speech(frame)` accepts the pipeline's 30 ms/480-sample frames by buffering to Silero's required 512-sample windows internally (document: decision latency ≤ one window); `reset()` clears the internal LSTM state and buffer. Import of onnxruntime inside `__init__` → `NotImplementedError` with install hint if missing.
- `make_vad`: `kind="silero"` → try SileroVad; missing dep or model file → actionable error (server startup surfaces it via lifespan logging from Task 2).
- `scripts/download_models.py silero` downloads the official silero_vad.onnx (v5, ~2.3 MB) from the snakers4/silero-vad GitHub release to `models/`, verifying size > 1 MB; script structure takes a registry of named artifacts so later tasks add entries (`sherpa-zipformer-en`, `paraformer-en`, ...). Idempotent (skips existing).
- Tests marked `@pytest.mark.model` (need onnxruntime + model file): tone vs silence classification, reset clears state, 480-sample framing works. One UNMARKED test: `make_vad(VadConfig(kind="silero"))` without onnxruntime installed raises with the install hint (guard with `pytest.importorskip` inverse — skip if onnxruntime IS installed).

**Steps:** TDD (unmarked tests RED/GREEN in CI mode; model tests written and runnable via `uv run --extra silero pytest -m model tests/core/test_vad_silero.py` — run them if the env permits, otherwise document in report). Commit `feat: Silero VAD detector behind optional silero extra + model download script`.

---

### Task 5: Backend extras scaffolding + sherpa-onnx backend

Spec §6.1: sherpa-onnx streaming Zipformer (English), dedicated ThreadPoolExecutor, true incremental partials.

**Files:**
- Create: `src/stt_server/backends/sherpa/__init__.py`, `src/stt_server/backends/sherpa/backend.py`
- Modify: `pyproject.toml` (`sherpa = ["sherpa-onnx>=1.10"]` extra; this task's lock commit), `src/stt_server/backends/__init__.py` (register via lazy import guard), `scripts/download_models.py` (add `sherpa-zipformer-en`: the `sherpa-onnx-streaming-zipformer-en-2023-06-26` model tarball from the k2-fsa GitHub releases, extracted under `models/`), `configs/sherpa.yaml`
- Test: `tests/backends/test_sherpa_backend.py`

**Interfaces:**
- `SherpaBackend(model_dir: str, num_threads: int = 4, pool_workers: int = 8, language: str = "en")` registered as type `"sherpa_onnx"`. `start()`: build `sherpa_onnx.OnlineRecognizer.from_transducer(tokens=..., encoder=..., decoder=..., joiner=..., num_threads=num_threads, sample_rate=16000, feature_dim=80, enable_endpoint_detection=False)` inside `asyncio.to_thread`; missing import → `BackendUnavailableError` with `pip install 'stt-server[sherpa]'` hint at construction time.
- Capabilities: `streaming=True, languages=("en",), native_endpointing=False` (server VAD governs; backend-native endpointing experiment is Plan 4).
- `SherpaStream`: owns one `recognizer.create_stream()`; `push_audio` converts PCM16 bytes → float32 in [-1,1] (`np.frombuffer(data, np.int16).astype(np.float32) / 32768`) and calls `accept_waveform` + drains decodes on the backend's shared `ThreadPoolExecutor` via `loop.run_in_executor`; after each decode, if `get_result()` text changed, queue a `partial` BackendEvent with audio_time_ms = accumulated pushed audio. `finalize()`: `input_finished()`, drain remaining decodes, emit `final` with the last text, end iterator. `close()` idempotent. Registry pattern: module imports sherpa_onnx lazily inside `start`/`__init__` so `stt_server.backends` package import never requires it.
- The decode loop pattern (single-stream sequential decode per utterance — correct for one stream; batching across streams is a Plan 4 measurement):

```python
def _decode_sync(self) -> str:
    while self._recognizer.is_ready(self._stream):
        self._recognizer.decode_stream(self._stream)
    return self._recognizer.get_result(self._stream)
```

- `configs/sherpa.yaml`: silero VAD + sherpa backend + models mapping (`sherpa-zipformer-en: sherpa`), copied from configs/mock.yaml structure.
- Tests: conformance subclass `@pytest.mark.model` (`TestSherpaConformance(BackendConformanceSuite)` with real model fixture, module-scoped backend); plus unmarked structural tests: registry knows type `"sherpa_onnx"` only when importable — creating with type `"sherpa_onnx"` and no sherpa-onnx installed raises `BackendUnavailableError` with the extras hint; float conversion helper unit test (pure numpy-free fallback: use `array` module if numpy absent — implement conversion with `array("h")` to avoid a numpy dependency in core paths).
- IMPORTANT for implementer: verify the exact `sherpa_onnx` API names against the installed package docs (context7 or the package's README) if you install the extra; if the API differs from the plan's sketch, follow the real API and note the deviation in your report.

**Steps:** TDD on the unmarked tests; model tests best-effort (document if env lacks GPU/deps — CPU is fine for sherpa). Commit `feat: sherpa-onnx streaming backend behind optional extra` (+ lock commit for the extras metadata, PyPI-verified).

---

### Task 6: FunASR backend (optional `funasr` extra)

Spec §6.1: Paraformer streaming (EN/ZH), bounded thread pool, chunk-based partials.

**Files:**
- Create: `src/stt_server/backends/funasr/__init__.py`, `src/stt_server/backends/funasr/backend.py`
- Modify: `pyproject.toml` (`funasr = ["funasr>=1.2", "torch>=2.4", "torchaudio>=2.4"]`), `scripts/download_models.py` (FunASR models auto-download via modelscope on first use — entry documents that and pre-warms), `configs/funasr.yaml`
- Test: `tests/backends/test_funasr_backend.py`

**Interfaces:**
- `FunasrBackend(model: str = "paraformer-zh-streaming", pool_workers: int = 4, chunk_size: tuple[int, int, int] = (0, 10, 5))` registered as `"funasr"`. `start()`: `AutoModel(model=self._model)` in a thread. Chunk config: `[0, 10, 5]` = 600 ms streaming chunks (FunASR convention: chunk_size[1] * 60 ms).
- `FunasrStream`: buffers pushed PCM until ≥ one 600 ms chunk (960 ms×16k×2 bytes = 19200 bytes per chunk at [0,10,5]... implementer: compute from chunk_size[1] * 960 bytes), then runs `model.generate(input=chunk_f32, cache=self._cache, is_final=False, chunk_size=list(self._chunk_size), encoder_chunk_look_back=4, decoder_chunk_look_back=1)` on the pool; accumulates returned text increments into a growing hypothesis; emits cumulative text as `partial` events (the session stabilizer handles the rest). `finalize()`: final `generate(..., is_final=True)` with remaining buffer, emit `final` with full accumulated text.
- Capabilities: `streaming=True, languages=("zh", "en"), native_endpointing=False`.
- Same lazy-import + `BackendUnavailableError` extras-hint pattern as Task 5; same unmarked structural tests; conformance subclass under `@pytest.mark.model`.
- IMPORTANT for implementer: FunASR's streaming API details (cache dict, result shape `[{"text": ...}]`) should be verified against the installed funasr package if you install the extra; follow the real API and note deviations.

**Steps:** TDD unmarked; model tests best-effort. Commit `feat: FunASR Paraformer streaming backend behind optional extra`.

---

### Task 7: Qwen3-ASR backend (optional `qwen3asr` extra, GPU)

Spec §6.1: Qwen3-ASR 0.6B/1.7B via the official vLLM-based inference framework (Apache 2.0, released 2026-01-29). GPU flagship; chunk-based streaming via the framework's streaming interface.

**Files:**
- Create: `src/stt_server/backends/qwen3asr/__init__.py`, `src/stt_server/backends/qwen3asr/backend.py`
- Modify: `pyproject.toml` (`qwen3asr = ["qwen3-asr", "vllm>=0.8"]` — implementer: verify the official package name on PyPI (the QwenLM/Qwen3-ASR GitHub README documents it; likely `qwen3-asr` or similar) and pin what exists), `configs/qwen3asr.yaml`
- Test: `tests/backends/test_qwen3asr_backend.py`

**Interfaces:**
- `Qwen3AsrBackend(model: str = "Qwen/Qwen3-ASR-0.6B", gpu_memory_utilization: float = 0.8, max_concurrent: int = 8, language: str | None = None)` registered as `"qwen3asr"`. `start()` initializes the framework's async engine; `create_stream` returns a stream that buffers utterance audio and, because Qwen3-ASR's streaming interface operates on growing audio context, re-decodes the accumulated utterance audio every `redecode_interval_ms` (config, default 480) emitting cumulative partials; `finalize()` runs one last full decode as the final. This "re-decode growing buffer" pattern is the documented adapter strategy for attention-decoder models (spec calls it chunk-based); per-chunk incremental decoding via the framework's native streaming mode is preferred IF the installed framework exposes it — implementer decides based on the actual API and documents the choice in the module docstring and report.
- Capabilities: `streaming=True, languages=(...52 langs — use ("en", "zh") + note...), native_endpointing=False, batch_decode=True` (vLLM batches internally; `batch_decode=True` signals the file path may skip VAD for accuracy parity per spec §3.3 — NOT wired in this plan, Plan 4 measures it).
- Same lazy-import/extras-hint/unmarked-structural-test pattern. Conformance subclass `@pytest.mark.model` + `@pytest.mark.gpu` (add `gpu` marker to pyproject markers; model tests for this backend need CUDA).
- IMPORTANT for implementer: this adapter targets a framework released 2026-01; consult the QwenLM/Qwen3-ASR GitHub README (WebFetch) for the real API before writing the integration body. The plugin contract, config surface, tests, and error handling above are fixed; the framework-call internals follow the real docs. If the docs are unreachable, implement against the documented HuggingFace `transformers` fallback (AutoModelForSpeechSeq2Seq pipeline) and report the substitution.

**Steps:** TDD unmarked; GPU tests documented-not-run locally (user's CUDA box runs them in Plan 4). Commit `feat: Qwen3-ASR backend skeleton with vLLM engine integration behind optional extra`.

---

### Task 8: Docker, compose, Grafana

Spec §9.

**Files:**
- Create: `deploy/Dockerfile` (CPU: mock+sherpa+funasr+silero), `deploy/Dockerfile.gpu` (CUDA base + qwen3asr), `deploy/docker-compose.yaml` (profiles: cpu, gpu, observability), `deploy/prometheus.yml`, `deploy/grafana/dashboard.json`, `deploy/grafana/datasource.yml`, `deploy/grafana/dashboards.yml`, `.dockerignore`
- Modify: `README.md` (quickstart: `docker compose --profile cpu up` + mock curl smoke)
- Test: `tests/deploy/test_dockerfiles.py` (static checks only — CI has no Docker: parse FROM lines, assert uv usage, assert models volume declared)

**Interfaces:**
- `deploy/Dockerfile`: multi-stage; `FROM python:3.12-slim` + uv (copy from `ghcr.io/astral-sh/uv`), `uv sync --frozen --extra sherpa --extra funasr --extra silero`, non-root user, `VOLUME /app/models`, `EXPOSE 8000`, entrypoint `stt-server --config /app/configs/mock.yaml --host 0.0.0.0`. Config overridable via `STT__` env or mounted config.
- `deploy/Dockerfile.gpu`: `FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04`, python3.12 via uv, adds `--extra qwen3asr`.
- compose: `stt-server` (cpu profile, build ., ports 8000, models volume), `stt-server-gpu` (gpu profile, `deploy.resources.reservations.devices` nvidia), `prometheus` + `grafana` (observability profile; Grafana auto-provisioned datasource + dashboard: panels for active sessions, first-partial/final latency p50/p95 (histogram_quantile), utterance rate, rejections, errors).
- Local verification (implementer runs if Docker available, else documents): `docker build -f deploy/Dockerfile .` and mock-profile compose up + `/healthz` curl.

**Steps:** static tests TDD; build best-effort. Commit `feat: Docker CPU/GPU images, compose profiles, Grafana dashboard`.

---

### Task 9: Backend docs + config profiles polish

**Files:**
- Create: `docs/backends.md`
- Modify: `README.md` (backends section: table of the four backends, extras install commands, model download commands, config profiles), `configs/*.yaml` sanity pass (every backend has a profile; comments)

**Interfaces:** `docs/backends.md` documents the plugin contract (SttBackend/SttStream lifecycle: one stream per utterance, events() semantics, finalize/close), how to write a backend (walk through mock as the reference), the conformance suite (how to subclass), execution-strategy guidance per spec §2.1, and per-backend setup (extras, models, config, verified-on matrix: what was actually run where — be honest about GPU-untested Qwen3-ASR).

**Steps:** docs only; cite real test/file names; full suite + ruff; commit `docs: backend development guide and per-backend setup`.

---

## Plan 3 exit criteria

- Default `uv sync` (no extras) + `uv run pytest`: all non-model tests green (~85+); ruff clean; layering grep clean.
- `uv run pytest -m model` collects the real-backend conformance tests (skipped/failing only for missing deps/hardware, documented per backend in reports).
- Sherpa backend verified end-to-end locally on CPU with real model (implementer report shows a real transcription through the native WS API).
- `/metrics` exposes the spec §7 metric families; Grafana dashboard JSON references only exported metric names.
- Docker CPU image builds (if Docker present) or static tests pass.
- The hardening backlog in `.superpowers/sdd/progress.md` under "Plan 3" is fully absorbed (each item either fixed in Tasks 1-3 or explicitly re-deferred with reason).
