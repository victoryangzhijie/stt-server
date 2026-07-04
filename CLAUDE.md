# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project is uv-managed (Python 3.12+).

```bash
# Full default test suite (ML-free: mock backend + energy VAD only)
uv run pytest -q

# Single test / file
uv run pytest tests/core/test_session.py::test_name -v

# Lint (only linter in use; no mypy)
uv run ruff check .

# Run the server locally (boots with the mock backend, no models needed)
uv run stt-server --config configs/mock.yaml

# Real-model tests (deselected by default via addopts = "-m 'not model'")
uv run --extra sherpa pytest -m model tests/backends/test_sherpa_backend_model.py
uv run pytest -m "model and gpu" tests/backends/test_qwen3asr_backend_model.py  # needs CUDA

# Download model weights / corpus (both land in gitignored dirs)
uv run python scripts/download_models.py <name|all>
```

### uv / lockfile policy (CRITICAL on this machine)

The development machine sets `UV_DEFAULT_INDEX` globally to an aliyun mirror, which silently rewrites `uv.lock` with mirror URLs — breaking CI and non-CN contributors.

- Prefix **every** uv command with `UV_DEFAULT_INDEX="https://pypi.org/simple"`.
- **Never** run `uv add` or bare `uv lock`. Dependency changes are manual `pyproject.toml` edits; regenerating the lock is a deliberate, separate commit made with `UV_DEFAULT_INDEX="https://pypi.org/simple" UV_INDEX_URL="https://pypi.org/simple" uv lock`.
- Check `git status --short uv.lock` before every commit; restore with `git checkout -- uv.lock` if dirtied. Verify locks with `grep -c "mirrors.aliyun" uv.lock` → must be 0 (package *names* like `aliyun-python-sdk-*` are legitimate transitive deps — only mirror URLs are a failure).
- Pass this policy verbatim to any subagent that runs uv commands.

## Architecture

OpenAI-compatible streaming STT server. The load-bearing design rule: a **protocol-agnostic session core** with API adapters that are pure encoders.

### Layering rule (enforced, non-negotiable)

`src/stt_server/core/`, `backends/`, and `metrics/` must never import from `src/stt_server/api/`. Verified by:

```bash
grep -rn "from stt_server.api\|import stt_server.api" src/stt_server/core src/stt_server/backends src/stt_server/metrics
# empty output (exit 1) = pass
```

Also: nothing under `src/` may import the top-level `benchmarks/` package.

### Session pipeline (src/stt_server/core/)

`AudioChunk` → bounded audio queue (backpressure: `drop_oldest`/`error` policy, fed by a feeder task — `push_audio` is a non-blocking enqueue) → `FrameSlicer` (30 ms frames) → VAD (`EnergyVad` default; `SileroVad` behind the `silero` extra) → `Endpointer` state machine (emits `StartUtterance`/`SpeechAudio`/`EndUtterance`) → per-utterance backend `SttStream` → `Stabilizer` (prefix-commit stable/volatile split) → `TranscriptEvent` queue consumed by the API adapter.

Invariants that tests pin and reviews have repeatedly defended:
- One `SttStream` per utterance; `events()` yields partials then **exactly one** final, then ends; post-done `push_audio`/`finalize` are no-ops.
- The stabilizer and all latency accounting run on **backend audio time** (`audio_time_ms`), never wall clock — this is what makes faster-than-real-time file mode deterministic.
- The session must never hang: every failure path (backend error mid-decode, feeder exception, abort during finalize) emits an ERROR event and terminates `events()`. `end_input()` drains all queued audio before finalizing (lossless file mode).
- `end_input()` sets `_input_ended` synchronously; `abort()` cancels and awaits the feeder/reader tasks.

### Backends (src/stt_server/backends/)

Plugin contract in `base.py`; registration via `@register_backend("<type>")` + import in `backends/__init__.py`. Four backends: `mock` (always available; scripted, deterministic), `sherpa_onnx`, `funasr`, `qwen3asr` — the real three live behind optional extras and follow an identical hardened template (read `sherpa/backend.py` first; it is the canonical example):
- Lazy imports: `importlib.util.find_spec` gate in `__init__` raising `BackendUnavailableError` with a `pip install 'stt-server[<extra>]'` hint; heavy imports only inside methods. `import stt_server.backends` must never pull sherpa/torch/vllm/onnx.
- Per-stream `asyncio.Lock` serializing decode+enqueue with a `_done` recheck inside the lock; lock-aware idempotent `close()`; `finalize()` flips `_done` *before* taking the lock (that flag, not lock scope, guarantees its ordering).
- Sync engines decode on a shared `ThreadPoolExecutor` via `run_in_executor`; `stop()` shuts it down via `await asyncio.to_thread(...)` (never block the loop). Qwen3-ASR uses the `qwen-asr` package's native streaming API over vLLM's **synchronous** `LLM` class (the design spec's async-engine assumption is superseded — see the module docstring).
- Every backend must pass `tests/backends/conformance.py::BackendConformanceSuite`. New backends: subclass it, plus the unmarked structural tests (missing-extra error, config roundtrip in `tests/test_config_profiles.py`).

`docs/backends.md` is the full guide, kept citation-accurate against source.

### API adapters (src/stt_server/api/)

Three surfaces sharing the core: native WS (`/ws/transcribe`, richest — exposes stable/volatile split; benchmark clients use it), OpenAI Realtime WS (`/v1/realtime?intent=transcription`, beta-vocabulary events with wire-accurate append-only deltas + casefold catch-up), OpenAI file HTTP (`POST /v1/audio/transcriptions`). Guards in `guards.py`: constant-time bearer tokens, `SessionSlots` capacity, session deadline; WS close codes 4401/4429/4408. A pure-ASGI middleware in `app.py` rejects oversized uploads via Content-Length *before* the body is buffered. `/metrics` (own `CollectorRegistry`, `metrics/registry.py`) is deliberately unauthenticated — internal-network exposure only.

`docs/openai-compat.md` is the **tested** compatibility matrix: no compatibility claim may be added without a citing test.

### Benchmarks (benchmarks/, `bench` extra)

Runners (`run_accuracy`, `run_load`, `run_stabilizer_study`, `run_endpointing`) drive a real server subprocess (`ServerUnderTest`) over the native WS. Hard-won methodology rule: real-backend WER/latency runs must stream at `--pace 1.0` (real time) — at pace 0 the drop_oldest backpressure silently sheds audio and corrupts results. All streaming runners assert a zero `stt_audio_dropped_total` delta by default (shared `benchmarks/_drops.py`); `run_load` instead *measures* drops. `benchmarks/cuda_runbook.md` + `scripts/run_gpu_suite.sh` is the GPU-box procedure.

## Testing conventions

- CI (and the default env) is ML-free: heavy deps live behind extras (`sherpa`, `funasr`, `qwen3asr`, `silero`, `bench`); tests needing them use `@pytest.mark.model` / `@pytest.mark.gpu` or `pytest.importorskip`. CI runs `uv sync --frozen`.
- macOS 12 local quirks: newest sherpa-onnx wheels fail (`pin sherpa-onnx==1.10.46`), `onnxruntime>=1.20` uninstallable (venv-pin 1.19.2), torch capped at 2.2.x — throwaway venvs in the session scratchpad handle real-model testing; `pyproject.toml` constraints target deployment, not this box.
- `tests/conftest.py` has an autouse fixture resetting all labeled metrics between tests — new labeled metric families must be added to its `_LABELED_METRICS` tuple.
- Don't commit model weights, corpus audio, or `benchmarks/results/` JSONs. The one committed audio fixture (`tests/fixtures/speech_16k_mono_s16le.pcm`) has documented provenance in `tests/fixtures/README.md` — any new third-party fixture needs the same treatment (this is a public Apache-2.0 repo).

## Project history / docs

Built via spec → 4 sequential implementation plans, all under `docs/superpowers/` (design spec + plan docs). `docs/architecture.md` has the component diagrams. Known deferred work: Plan 4 Task 9 (full CPU benchmark report — partial results and resume commands in `.superpowers/sdd/p4-task-9-report.md`), the CUDA-box GPU run, and creating a GitHub remote (CI has never run remotely).
