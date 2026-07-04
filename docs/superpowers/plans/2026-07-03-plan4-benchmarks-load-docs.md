# Plan 4: Benchmarks, Load Generator, Reports, Final Docs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver spec §10's reproducible benchmark suite (accuracy/latency/load/stabilization/endpointing + bottleneck analysis), absorb the Plan 3 backlog (backpressure §13, pre-parse upload guard, per-request language), run the CPU benchmarks for real on this machine, and finish the documentation set (§12: architecture.md, methodology.md, report.md, README final polish).

**Architecture:** Benchmarks live in a top-level `benchmarks/` Python package behind a `bench` extra, driving the *running server* over the native WS API (spec: accuracy is measured *through the serving system*) and the file API. A shared runner boots the server as a subprocess from a given config, waits for `/readyz`, and tears it down. Server-side, two spec-§13 gaps are closed first (bounded audio queue with shed policy; pre-parse upload guard) because the load generator exercises exactly those paths. GPU numbers come from a one-command CUDA-box runbook the user executes; the report labels CPU numbers (real, this machine) vs GPU (pending) honestly.

**Tech Stack:** Python 3.12, uv, pytest/pytest-asyncio, `websockets` (already present via uvicorn[standard]), jiwer (WER), psutil (resource sampling), pynvml (GPU sampling, optional import), httpx (already a test dep — promote usage in bench file client).

## Global Constraints

- **uv/lockfile session policy (HARD):** prefix EVERY uv command with `UV_DEFAULT_INDEX="https://pypi.org/simple"`. NEVER run `uv add` or bare `uv lock`. pyproject edits are manual. `git status --short uv.lock` must be clean before every commit; restore with `git checkout -- uv.lock`. The ONE permitted lock commit in this plan is Task 3's (bench extra), generated with `UV_DEFAULT_INDEX="https://pypi.org/simple" UV_INDEX_URL="https://pypi.org/simple" uv lock` and verified `grep -c "mirrors.aliyun" uv.lock` → 0 (exit 1 with output 0 = PASS; package names like `aliyun-python-sdk-core` are legitimate and NOT a failure).
- Layering: `src/stt_server/core/`, `backends/`, `metrics/` never import from `src/stt_server/api/` (grep-verified). `benchmarks/` may import `stt_server` freely but nothing in `src/` may import `benchmarks`.
- Default test suite stays ML-free and Docker-free: new tests either run against mock/energy-VAD or are marked (`model`, `gpu`, or new `bench` marker for tests needing the bench extra — register in pyproject and add to the deselect logic ONLY if the bench extra isn't a default dev dep; if jiwer/psutil are cheap pure-Python installs, prefer making them regular dev deps of the test env via the bench extra + CI change, else mark).
- Baseline at plan start: 198 passed / 21 deselected; ruff clean. Every task ends with the full default suite green + ruff clean.
- No model binaries or corpus audio committed. `models/` and any `benchmarks/data/` corpus dir must be gitignored.
- Real-verification honesty: no number appears in `docs/benchmarks/report.md` unless the command that produced it is documented and was actually run; GPU sections say "pending CUDA-box run" explicitly.
- Local environment reality (macOS 12 x86, no CUDA): sherpa works locally via pinned wheel `sherpa-onnx==1.10.46` in a venv (see `.superpowers/sdd/task-5-report.md`); funasr works via the task-6 venv recipe (torch 2.2.2 pin); qwen3asr does NOT run locally. Benchmark RUNS use whatever interpreter the runner is given (`--python` / current env); benchmark CODE must not require heavy extras to import.

---

### Task 1: Backpressure — bounded per-session audio queue with shed policy (spec §13)

Spec §13: "Backpressure: bounded per-session audio queues; if a backend can't keep up, the server sheds by policy (config: `drop_oldest | error`) and logs/counts it — never unbounded memory growth." Currently `Session.push_audio` processes audio inline: a slow backend decode blocks the WS receive loop (socket-level stall, no shedding, no metric). Also absorbs backlog item M-3 (ws_transcribe try-nesting) since this task touches the same file.

**Files:**
- Modify: `src/stt_server/core/session.py`, `src/stt_server/config/settings.py` (new `LimitsConfig` fields), `src/stt_server/metrics/registry.py` (new counter), `src/stt_server/api/native_ws.py` (nesting refactor only — behavior unchanged), `configs/mock.yaml` (comment documenting defaults)
- Test: `tests/core/test_session_backpressure.py`, extend `tests/config/test_settings.py`

**Interfaces:**
- `LimitsConfig` gains `audio_queue_chunks: int = 64` (bounded queue depth in chunks; 64 × 100ms client chunks ≈ 6.4s of audio) and `audio_overflow_policy: Literal["drop_oldest", "error"] = "drop_oldest"`.
- `Session.__init__` gains `audio_queue_chunks: int = 64, audio_overflow_policy: str = "drop_oldest"` kwargs (all three API adapters pass them from settings.limits).
- New metric: `stt_audio_dropped_total{backend}` Counter in `metrics/registry.py` ("PCM chunks shed by backpressure policy").
- `Session.push_audio(chunk)` becomes a non-blocking enqueue onto a bounded `asyncio.Queue(maxsize=audio_queue_chunks)`; a single feeder task (started lazily on first push, or in `__init__` alongside existing tasks — implementer picks what fits the current task lifecycle, and must ensure `abort()`/`end_input()` cancel/drain it exactly like the existing reader task) dequeues and runs the existing inline pipeline (`_apply` path unchanged).
- Overflow behavior: `drop_oldest` → evict the oldest queued chunk, count `stt_audio_dropped_total`, `logger.warning("audio.dropped", ...)` at most once per second per session (rate-limit with a monotonic timestamp — don't log every chunk); `error` → emit ERROR event (`error_code="backpressure"`, recoverable=False), end the session (mirror `_push_audio_safe`'s cleanup shape).
- `end_input()` must process all remaining queued audio before finalizing (drain, then existing behavior) so file-mode transcription stays lossless and deterministic.

**Behavioral tests (write first, RED then GREEN):**
- Slow-backend fake (push_audio sleeps) + queue size 4 + drop_oldest: push 20 chunks fast → session survives, final still emitted, `stt_audio_dropped_total` scrape shows drops, dropped count + processed count == 20.
- Same but policy=error: session emits exactly one ERROR with code "backpressure" and terminates cleanly (events() ends, no hang; reuse the PushFailStream test patterns in `tests/core/test_session.py`).
- Lossless drain: fast backend, queue size 4, push 20 chunks then `end_input()` → all 20 chunks reach the backend (no drops), final emitted. This pins that backpressure NEVER drops when the consumer keeps up.
- Existing 198 tests must stay green — the mock path must be behaviorally identical (mock decodes instantly; queue never fills).
- Settings test: yaml roundtrip of the two new fields; invalid policy string rejected by pydantic.

**M-3 refactor (same commit or a second commit, implementer's choice):** flatten `ws_transcribe`'s 4-level try nesting by extracting the control-frame handling into a module-level helper (e.g. `_handle_control_frame(ws, session, control) -> bool`); no behavior change — the existing native WS tests are the safety net.

**Steps:** TDD; run full suite + ruff; commit `feat: bounded per-session audio queue with drop_oldest/error shed policy` (+ optional `refactor: flatten ws_transcribe control handling`).

---

### Task 2: API hardening — pre-parse upload guard (M-1) + per-request language (carried)

Two carried backlog items, both small, both API-surface.

**Files:**
- Modify: `src/stt_server/api/app.py` (middleware), `src/stt_server/api/transcriptions_http.py`, `src/stt_server/backends/sherpa/backend.py`, `src/stt_server/backends/funasr/backend.py`, `src/stt_server/backends/qwen3asr/backend.py`, `docs/openai-compat.md` (language param row), `docs/backends.md` (delete the known-limitation entry, replace with the new behavior)
- Test: extend `tests/api/test_transcriptions_http.py`, `tests/backends/test_{sherpa,funasr,qwen3asr}_backend.py`

**Interfaces:**
- **M-1:** a pure-ASGI middleware (or Starlette `BaseHTTPMiddleware`-free `@app.middleware`-style function — pick pure ASGI to avoid body buffering) registered in `create_app` that, for `POST /v1/audio/transcriptions`, rejects requests whose `Content-Length` header exceeds `settings.limits.max_upload_bytes` with a 413 + the existing OpenAI error envelope BEFORE the body is read. Requests without Content-Length (chunked) fall through to the existing post-read check (keep it — it's the backstop). Auth still happens in the endpoint (moving auth pre-parse is out of scope; the DoS fix is the body-buffering one). Update the in-code known-limitation comment in `transcriptions_http.py` to describe the remaining chunked-encoding residual only.
- **Language wiring:** each real backend's `create_stream(cfg: StreamConfig)` uses `cfg.language or self._language` where the engine accepts a per-utterance language (qwen3asr: pass to `init_streaming_state(language=...)`; funasr/sherpa: their engines are per-model-language — for those, if `cfg.language` is set and differs from the backend's language capability, IGNORE it but `logger.debug` once; do NOT error, the OpenAI API treats language as a hint). The file endpoint already threads `language` into StreamConfig — verify; native/realtime WS: check whether the protocol carries language (realtime session config may); wire what exists, do not invent protocol fields.

**Tests:** 413-before-read test (send a large Content-Length header with a streaming body that would fail if buffered — e.g. assert the endpoint returns 413 without consuming; simplest practical assertion: Content-Length huge + tiny actual body → 413 with error envelope). Language: qwen3asr fake-engine test asserting `init_streaming_state` receives the per-request language when StreamConfig.language is set, constructor language otherwise; sherpa/funasr tests asserting mismatch is ignored without error.

**Steps:** TDD; full suite + ruff; commit `feat: pre-parse upload guard middleware + per-request language wiring`.

---

### Task 3: Benchmark foundation — package, extra, corpus tooling, clients, server runner

**Files:**
- Create: `benchmarks/__init__.py`, `benchmarks/corpus.py`, `benchmarks/client_ws.py`, `benchmarks/client_file.py`, `benchmarks/server.py`, `benchmarks/results.py`, `benchmarks/README.md`
- Modify: `pyproject.toml` (`bench = ["jiwer>=3.0", "psutil>=5.9", "pynvml>=11.5", "soundfile>=0.12"]` extra — manual edit; register `bench` pytest marker), `.gitignore` (`benchmarks/data/`, `benchmarks/results/`), `.github/workflows/ci.yml` ONLY if needed to keep CI green (bench tests must be skipped without the extra via importorskip — no CI change preferred)
- Test: `tests/benchmarks/test_corpus.py`, `tests/benchmarks/test_clients.py`
- **This task carries the plan's ONE permitted lock commit** (bench extra), per Global Constraints.

**Interfaces (later tasks consume these exact names):**
- `benchmarks/corpus.py`:
  - `LIBRISPEECH_URLS: dict[str, str]` — `{"test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz", "test-other": ".../test-other.tar.gz"}` (implementer verifies exact URLs).
  - `download_subset(split: str, dest: Path) -> Path` — downloads + extracts under `benchmarks/data/` (stdlib urllib+tarfile, path-traversal-guarded like scripts/download_models.py; idempotent).
  - `build_manifest(split_dir: Path, n: int, seed: int) -> list[Utterance]` where `@dataclass Utterance: id: str; flac_path: Path; ref_text: str; duration_s: float` — seeded `random.Random(seed).sample` over the split's utterances (LibriSpeech layout: `<split>/<spk>/<chap>/<spk>-<chap>-<utt>.flac` + `<spk>-<chap>.trans.txt`). FLAC decode: use stdlib-adjacent `soundfile`? NO — soundfile needs libsndfile. Use `ffmpeg`? NO new system deps. **Decision: convert FLAC → 16k mono PCM16 WAV via Python `audioop`-free path is impossible for FLAC; therefore add `soundfile>=0.12` to the bench extra** (it ships bundled libsndfile wheels for macOS/Linux; verify import works locally before committing) and expose `load_pcm16(utt: Utterance) -> bytes` (decode, resample 16k via `audioop.ratecv`-equivalent — Python 3.13 removed audioop; this repo is 3.12, `audioop` works; note it).
  - `normalize_text(s: str) -> str` — lowercase, strip punctuation, collapse whitespace (jiwer's standard transform is applied in Task 4; this is the shared pre-clean).
- `benchmarks/client_ws.py`:
  - `@dataclass UtteranceResult: utt_id: str; hypothesis: str; server_final_ms: float | None; server_first_partial_ms: float | None; client_final_ms: float; partials: list[tuple[float, str, str]]` (partials = (audio_time_ms, stable_text, volatile_text) — Task 7 needs them).
  - `async def stream_utterance(base_ws_url: str, model: str, pcm16: bytes, *, chunk_ms: int = 100, pace: float = 1.0, token: str | None = None) -> UtteranceResult` — connects to `/ws/transcribe?model=...`, streams at `chunk_ms/pace` intervals (pace=1.0 real-time; pace=0 no sleep = as-fast-as-possible for file-parity checks), sends `{"type": "input_done"}`, collects PARTIAL/STABILIZED/FINAL/session.closed per the wire format in `native_ws.py::encode_native` (FINAL carries `latency: {"final_ms":…, "first_partial_ms":…}`), measures `client_final_ms` = monotonic(final received) − monotonic(last audio byte sent). Multi-utterance sessions: a single audio buffer may endpoint into >1 utterance — collect ALL finals and join hypotheses with " ".
- `benchmarks/client_file.py`: `async def transcribe_file(base_url: str, pcm16: bytes, model: str, token: str | None = None) -> tuple[str, float]` — wraps PCM16 in a WAV container (stdlib `wave` + `io.BytesIO`), POSTs multipart to `/v1/audio/transcriptions`, returns (text, wall_seconds). Uses httpx.
- `benchmarks/server.py`: `class ServerUnderTest` context manager — `ServerUnderTest(config_path: str, port: int = 8100, python: str | None = None, env: dict | None = None)`; `__enter__` launches `[python or sys.executable, "-m", "stt_server", "--config", config_path, "--host", "127.0.0.1", "--port", str(port)]` as a subprocess (verify `__main__.py`'s actual CLI flags first and use them exactly), polls `GET /readyz` until 200 (timeout 120s — model loading), exposes `.base_url` / `.base_ws_url` / `.pid`; `__exit__` SIGTERM + wait, SIGKILL fallback. The `python` param is how sherpa/funasr benchmarks run inside their pinned local venvs.
- `benchmarks/results.py`: `def write_result(name: str, payload: dict) -> Path` — JSON to `benchmarks/results/<name>-<YYYYMMDD-HHMMSS>.json` including a `meta` block (git SHA, platform, python, backend, config path, seed, n); `def percentiles(xs: list[float]) -> dict` returning `{"p50":…, "p95":…, "p99":…, "mean":…, "n":…}`; `def markdown_table(rows: list[dict], columns: list[str]) -> str`.
- `benchmarks/README.md`: how to install (`uv sync --extra bench`), corpus download command, one-line description of each runner (Tasks 4–8 fill in their commands).

**Tests (unmarked where possible):** manifest determinism (same seed → same subset; fabricate a fake LibriSpeech dir tree in tmp_path — no download in tests); WAV wrapper roundtrip (client_file's WAV bytes parse back via `wave` with correct params); `percentiles` math; `stream_utterance` against the mock backend via a real in-process server (reuse the app + TestClient? No — websockets needs a live socket; use `ServerUnderTest` with mock config on an ephemeral port, marked `@pytest.mark.bench` if startup cost is high, else keep unmarked if <10s). jiwer/soundfile imports guarded with `pytest.importorskip` so the default ML-free CI stays green if the bench extra isn't installed there — decide based on whether CI installs the extra (prefer NOT changing CI; use importorskip).

**Steps:** TDD; lock commit (`chore: lock bench extra metadata (PyPI-verified)`, mirrors.aliyun grep = 0) separate from the feature commit `feat: benchmark foundation — corpus tooling, WS/file clients, server runner`.

---

### Task 4: Accuracy + latency benchmark (WER streamed & file; latency at concurrency 1)

Spec §10.1 + §10.2 in one runner: the real-time-paced streamed pass IS the concurrency-1 latency measurement — one pass yields both.

**Files:**
- Create: `benchmarks/run_accuracy.py`
- Modify: `benchmarks/README.md` (command + options)
- Test: `tests/benchmarks/test_accuracy_scoring.py`

**Interfaces:**
- CLI: `python -m benchmarks.run_accuracy --config configs/sherpa.yaml --model sherpa --split test-clean --n 100 --seed 42 [--modes ws,file] [--port 8100] [--python /path/venv/bin/python] [--pace 1.0]`. (`--model` is the query-string model name resolved by the server's registry; verify how `resolve_backend` maps names — read `api/app.py`.)
- Flow: build manifest → `ServerUnderTest(config, python=...)` → for each utterance: WS mode `stream_utterance` (pace as given), file mode `transcribe_file` → score with jiwer: `wer(reference, hypothesis)` using `jiwer.Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip(), ReduceToListOfListOfWords()])` applied to both sides (implementer verifies current jiwer API names — they changed across 3.x) → aggregate corpus-level WER (jiwer on concatenated lists, NOT mean of per-utterance WERs — document why in a comment: length weighting) → `write_result("accuracy-<model>-<split>", {...})` including per-mode WER, latency percentiles (server-reported first_partial/final AND client-observed final), RTF (audio seconds / wall seconds, file mode).
- Empty-hypothesis guard: an utterance yielding no final (error) records `hypothesis=""` and increments an `errors` count in the payload rather than crashing the run.
- Unmarked test: scoring unit test with hand-computed WER on 3 fabricated ref/hyp pairs (e.g. ref "the cat sat" hyp "the cat sat on" → insertions=1, WER=1/3); latency aggregation shape test. No network/model in tests.

**Steps:** TDD scoring; smoke the runner end-to-end against MOCK (mock's scripted transcripts make WER=100% vs LibriSpeech refs — that's fine, the smoke asserts mechanics: result JSON exists with all keys, n utterances processed); commit `feat: accuracy+latency benchmark runner (WER streamed/file, concurrency-1 latency)`.

---

### Task 5: Load generator — concurrency ramp until SLO breach, resource sampling

Spec §10.3.

**Files:**
- Create: `benchmarks/run_load.py`, `benchmarks/sampling.py`
- Modify: `benchmarks/README.md`
- Test: `tests/benchmarks/test_load_logic.py`

**Interfaces:**
- `benchmarks/sampling.py`: `class ResourceSampler(pid: int, interval_s: float = 1.0)` — background thread sampling `psutil.Process(pid)` CPU%/RSS (+ children), and GPU util/mem via pynvml if importable AND a device exists (guarded import; absence is silent). `.start()/.stop() -> dict` with time series + peaks.
- `benchmarks/run_load.py` CLI: `python -m benchmarks.run_load --config configs/mock.yaml --model mock --utterance-seconds 10 --start 2 --step 2 --max 64 --slo-final-ms 1200 --slo-pct 95 --seed 42 [--python ...] [--port 8100]`.
- Flow per rung N: spawn N concurrent `stream_utterance` loops (each loops utterances from the manifest — or synthetic speech-shaped audio for mock; use manifest audio when a real backend is configured, and the repo speech fixture `tests/fixtures/speech_16k_mono_s16le.pcm` tiled when `--synthetic` is passed) for a fixed window (`--window-seconds 30`); collect per-utterance client_final_ms; rung PASSES if p{slo-pct}(final_ms) ≤ slo AND zero backpressure ERRORs; ramp until first failing rung or --max. Also scrape the server's `/metrics` before/after each rung and record deltas of `stt_audio_dropped_total`, `stt_rejections_total`, `stt_errors_total` (dropped-audio deltas are the §13 shed policy made visible under load).
- Output: `write_result("load-<model>", ...)` with per-rung rows (N, p50/p95/p99, error count, dropped chunks, CPU%, RSS peak, GPU if present) + `max_passing_concurrency`.
- Unmarked tests: rung pass/fail decision logic (pure function `rung_passes(latencies, slo_ms, pct, errors) -> bool` — extract it precisely so it's testable); sampler start/stop against the current test process (psutil importorskip).

**Steps:** TDD decision logic; smoke a 2-rung ramp against mock locally (seconds, not minutes — small window); commit `feat: load generator with concurrency ramp, SLO gate, resource sampling`.

---

### Task 6: Stabilization study + realtime delta-duplication measurement (M-8)

Spec §10.4 + carried finding M-8.

**Files:**
- Create: `benchmarks/run_stabilizer_study.py`
- Modify: `benchmarks/README.md`
- Test: `tests/benchmarks/test_flicker_metrics.py`

**Interfaces:**
- Flicker metric (pure functions, exact definitions — Task 9's report cites them):
  - `retracted_chars(partials: list[str]) -> int` — sum over consecutive volatile-hypothesis pairs of `len(prev) - common_prefix_len(prev, cur)` when cur does not extend prev (characters the user SAW then lost).
  - `flicker_rate(partials, final) -> float` = retracted_chars / max(1, len(final)).
  - `commit_latency_ms(partials_with_time, final) -> float` — mean over final's words of (time word became part of STABLE text − time word first appeared in any hypothesis). Uses the `(audio_time_ms, stable_text, volatile_text)` triples from `UtteranceResult.partials`.
- CLI: `python -m benchmarks.run_stabilizer_study --config configs/sherpa.yaml --model sherpa --n 25 --seed 42 --grid "min_partials=1,2,3;min_stable_ms=0,240,480"` — for each grid point, boots `ServerUnderTest` with a TEMP config file derived from the base config with the stabilizer params substituted (write to a tempdir; `StabilizerConfig` field names verified against settings.py), streams the manifest subset, computes mean flicker_rate + mean commit_latency + WER per point; result JSON is the grid table.
- **M-8 measurement:** same runner grows a second mode `--api realtime`: stream the same audio over `/v1/realtime?intent=transcription` (reuse the protocol shapes from `tests/api/test_realtime_ws.py` — `input_audio_buffer.append` base64 frames etc.), reconstruct `"".join(deltas)` per item, and report `delta_duplication_ratio` = extra_chars_in_joined_deltas / len(completed_transcript) (0.0 = wire-perfect append-only; >0 quantifies the shrinking-final full-resend fallback). Runs per grid point too — stabilizer settings influence it.
- Unmarked tests: the three pure metrics on hand-built partial sequences (e.g. partials ["THE", "THE CAT", "THE CAR"] → retracted 1 char ("T" of CAT→CAR shares "CA"); include an exact worked example per function in the test).

**Steps:** TDD metrics; smoke 2×1 grid on mock; commit `feat: stabilizer flicker/commit-latency study + realtime delta-duplication metric`.

---

### Task 7: Endpointing experiment — server VAD vs sherpa native endpointing

Spec §10.5, "where supported" — sherpa-onnx is the one local backend with native endpointing (`enable_endpoint_detection=True` in `OnlineRecognizer.from_transducer`, plus rule-based endpoint config). This is an OFFLINE experiment script, NOT a serving mode: the server pipeline is not modified (adding a native-endpointing serving path is future work — say so in the report).

**Files:**
- Create: `benchmarks/run_endpointing.py`
- Modify: `benchmarks/README.md`
- Test: `tests/benchmarks/test_endpointing_metrics.py`

**Interfaces:**
- Arm A (through the server): stream manifest utterances with 1s of leading + trailing synthetic silence over native WS against `configs/sherpa.yaml`; record per utterance: number of utterance segments the server produced, WER, final latency.
- Arm B (direct recognizer, native endpointing): in-process (requires the sherpa venv python — the script imports `sherpa_onnx` lazily and errors actionably without it): build `OnlineRecognizer.from_transducer(..., enable_endpoint_detection=True)` with default endpoint rules (implementer verifies the real kwarg names from the installed package — `rule1_min_trailing_silence` etc.), feed the same padded audio in 100ms steps, record where `recognizer.is_endpoint(stream)` fires, the text at endpoint, and endpoint-detection latency (audio time at fire − audio time at true speech end, known from the padding construction).
- Metrics per arm: WER, mean/95p endpoint latency, over/under-segmentation counts (segments per reference utterance). `write_result("endpointing-sherpa", ...)`.
- Unmarked test: the segmentation-counting + endpoint-latency arithmetic on synthetic event timelines (pure functions, extracted).

**Steps:** TDD metrics; run Arm A+B for real locally with the sherpa venv (this box CAN — task-5 proved it); commit `feat: endpointing experiment — server VAD vs sherpa native endpoint detection`.

---

### Task 8: CUDA-box runbook + GPU test polish (M-6)

**Files:**
- Create: `benchmarks/cuda_runbook.md`, `scripts/run_gpu_suite.sh`
- Modify: `tests/backends/test_qwen3asr_backend_model.py` (M-6 CUDA skipif), `benchmarks/README.md` (link)
- Test: existing suite (the skipif change is itself test code)

**Interfaces:**
- M-6: add a module-level `pytest.mark.skipif` that skips qwen3asr model tests when CUDA is unavailable — detection WITHOUT importing torch at collection time on non-GPU boxes: `shutil.which("nvidia-smi") is None` is the cheap gate (comment why: torch import is heavy and the extra may be absent).
- `scripts/run_gpu_suite.sh` (bash, `set -euo pipefail`, each phase echo'd and independently skippable via env flags): (1) `uv sync --extra qwen3asr --extra bench`; (2) `uv run pytest -m "model and gpu" tests/backends/test_qwen3asr_backend_model.py`; (3) `docker build -f deploy/Dockerfile.gpu .`; (4) full CPU image `docker build -f deploy/Dockerfile .`; (5) accuracy run: `uv run python -m benchmarks.run_accuracy --config configs/qwen3asr.yaml --model qwen3asr --split test-clean --n 100 --seed 42`; (6) load run vs qwen3asr; (7) funasr zero-length `is_final=True` real-model check (one-liner python snippet from the Plan 3 carried item). Results land in `benchmarks/results/` for Task 9's report to ingest later.
- `benchmarks/cuda_runbook.md`: prerequisites (driver, docker+nvidia-container-toolkit, disk), the one command (`bash scripts/run_gpu_suite.sh`), expected artifacts, how to send results back (commit the JSONs or paste).

**Steps:** write script + doc; verify the skipif keeps local suite at full green (qwen model tests now skip for TWO reasons locally — fine); shellcheck the script if available (best-effort); commit `feat: CUDA-box GPU suite runbook + CUDA-availability skipif`.

---

### Task 9: Run the CPU benchmarks for real; write methodology.md + report.md

The payoff task: real numbers from this machine. Long-running (LibriSpeech download ~350MB + real-time-paced streaming ≈ 12min audio per pass) — budget accordingly; run the heavy passes sequentially and keep n=100/seed=42 fixed.

**Files:**
- Create: `docs/benchmarks/methodology.md`, `docs/benchmarks/report.md`
- Modify: `README.md` (link the two docs)
- Test: none new (this task runs, not builds; result JSONs are NOT committed — numbers are transcribed into report.md with their generating commands)

**What to actually run (each command goes in methodology.md verbatim):**
1. Corpus: `test-clean`, n=100, seed=42 (state the subset's total audio minutes in the report).
2. Accuracy+latency: mock (mechanics baseline, WER meaningless — say so), sherpa via its venv (`--python <venv>/bin/python`, document the venv recipe by reference to backends.md), funasr via its venv IF the task-6 venv still works within reasonable effort (paraformer-zh on English test-clean gives garbage WER — document that this measures the PIPELINE, and label the number as cross-lingual/not-comparable; if the venv fight exceeds ~30min, skip funasr locally and mark pending like GPU).
3. Load: mock (--synthetic, find max_passing_concurrency on this box) and sherpa (smaller --max).
4. Stabilizer study: sherpa, 3×3 grid, n=25.
5. Endpointing: sherpa, both arms.
- `methodology.md`: hardware/OS/python/versions table, corpus + seeding, every command, how SLOs were chosen, threats to validity (single machine, thermal, n=100 subset, real-time pacing means wall-clock ≈ audio duration).
- `report.md`: per spec §10.6 the bottleneck-analysis chapter interprets the numbers (where sherpa saturates on this CPU and why — decode thread pool vs GIL vs queue depth, supported by the load run's CPU%/latency curves and dropped-chunk counts); tables generated via `benchmarks.results.markdown_table` from the JSONs; a clearly-marked **"GPU results (pending)"** section listing exactly which runbook artifacts will fill it; honest limitations paragraph.

**Steps:** run everything (keep logs); write both docs; full suite + ruff still green; commit `docs: benchmark methodology and CPU report with real local numbers`.

---

### Task 10: Final documentation — architecture.md + README final polish

Spec §12 completion.

**Files:**
- Create: `docs/architecture.md`
- Modify: `README.md`
- Test: none (docs); citation rule applies

**Interfaces:**
- `docs/architecture.md`: components and data flow (mermaid diagrams: (1) session pipeline AudioChunk→FrameSlicer→VAD→Endpointer→SttStream→Stabilizer→events, (2) process/concurrency model — event loop, per-backend executors, bounded audio queue from Task 1); the execution model (Approach C, per-backend strategy — quote spec §2.1); layering rule and how it's enforced (the grep + where adapters encode); links to backends.md / openai-compat.md / benchmarks docs. Every class/file named must exist (grep-verify, same standard as Task 9 of Plan 3).
- `README.md` final pass: pitch paragraph (portfolio framing: OpenAI-compatible STT serving with measured compatibility + benchmarks), badges-free honest quickstart (existing Docker section), docs index table linking ALL of §12's deliverables, "project status" section (what's real vs pending-GPU).

**Steps:** write; grep-verify citations; full suite + ruff; commit `docs: architecture overview and README final polish`.

---

## Plan 4 exit criteria

- Default `uv run pytest` green (~200+ tests) and ruff clean at HEAD; layering grep clean; uv.lock contains only PyPI URLs (`grep -c "mirrors.aliyun"` = 0).
- Spec §13 backpressure implemented with both policies tested; §5.4 pre-parse upload guard in place.
- `benchmarks/` runners exist for accuracy, load, stabilizer study, endpointing — each with a documented one-command invocation and a smoke test that ran against mock.
- `docs/benchmarks/report.md` contains REAL numbers for mock + sherpa on this machine (WER, latency percentiles, max concurrency, flicker/commit grid, endpointing comparison) and an explicit pending-GPU section; `methodology.md` reproduces every figure.
- `scripts/run_gpu_suite.sh` + `benchmarks/cuda_runbook.md` give the user a one-command CUDA-box procedure.
- `docs/architecture.md` exists, citation-verified; README links the full §12 doc set.
- Plan 3's carried backlog fully absorbed: M-1, M-3, M-6, M-8, StreamConfig.language, funasr zero-length final (in GPU script) — each fixed or explicitly parked with reason in the ledger.
