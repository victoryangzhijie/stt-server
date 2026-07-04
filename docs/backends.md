# Backend development guide

This document is the plugin contract every STT backend implements, a
walkthrough of how to write a new one, the conformance test suite every
backend must pass, and per-backend setup/verification notes for the four
backends shipped in this repo (`mock`, `sherpa_onnx`, `funasr`, `qwen3asr`).

Like `docs/openai-compat.md`, this is the *tested* surface, not the
aspirational one: every file, class, and command named below was
grep-verified against the actual source at the time of writing.

## 1. The plugin contract

The contract lives in `src/stt_server/backends/base.py`:

```python
class SttBackend(abc.ABC):
    name: str
    capabilities: BackendCapabilities

    async def start(self) -> None       # load models / warm up, once at server startup
    async def stop(self) -> None
    async def create_stream(self, cfg: StreamConfig) -> SttStream

class SttStream(abc.ABC):
    async def push_audio(self, chunk: AudioChunk) -> None
    def events(self) -> AsyncIterator[BackendEvent]   # yields partials then exactly one final
    async def finalize(self) -> None    # endpoint reached: flush and emit the final, end the iterator
    async def close(self) -> None
```

`BackendCapabilities` (`streaming`, `languages`, `native_endpointing`,
`batch_decode`), `BackendEvent` (`kind: "partial" | "final"`, `text`,
`audio_time_ms`), `StreamConfig` (`language: str | None`), and
`BackendUnavailableError` (`code = "backend_unavailable"`) round out the
module.

Lifecycle rules, verbatim from the design spec (§6) and enforced by the
conformance suite (§3 below):

- **One `SttStream` per utterance.** A new stream is created on speech
  start (`create_stream`), driven with `push_audio` while the utterance is
  in progress, `finalize()`d when the endpointer detects end-of-speech, and
  `close()`d after that. Backends never need to reset internal state
  mid-utterance — a fresh `SttStream` instance does that for them.
- **`events()` yields partials then EXACTLY ONE final, then ends.** A
  backend may emit zero or more `"partial"` events during `push_audio`, but
  `finalize()` must enqueue exactly one `"final"` event and then a
  sentinel that ends the async iterator. No further events are ever valid
  after the final. This is exercised directly by
  `BackendConformanceSuite.test_finalize_yields_exactly_one_final_last` in
  `tests/backends/conformance.py`.
- **`finalize()` vs. `close()`.** `finalize()` is the normal end-of-utterance
  path: flush any buffered audio, emit the final, end the iterator.
  `close()` is the abnormal/cleanup path — e.g. a session dropped
  mid-utterance without ever reaching an endpoint — and must also end the
  iterator (with no final) if it hasn't already ended.
  `test_close_without_finalize_ends_iterator` covers this.
- **Post-done calls are no-ops.** Once a stream is done (finalized or
  closed), further `push_audio()` and `finalize()` calls must not raise and
  must not produce any further events. `test_push_after_finalize_is_noop`
  covers this exactly.
- **`audio_time_ms` is backend audio time, not wall-clock time.** It's the
  cumulative duration of audio pushed into the stream so far (see
  `MockStream._audio_ms`, `SherpaStream._audio_ms`,
  `FunasrStream`/`Qwen3AsrStream` equivalents), not
  `time.monotonic()`/`asyncio` timestamps. This is what makes partial
  timing deterministic in tests (the mock backend's docstring: "Partial
  timing is driven by accumulated *audio time*, never the wall clock, so
  tests and benchmarks are fully deterministic") and what the endpointing/
  stabilizer layers reason about instead of real time.
  `test_partials_are_ordered_by_audio_time` asserts events are
  monotonically non-decreasing in `audio_time_ms`.

## 2. How to write a backend

### 2.1 Walk through `mock.py`

`src/stt_server/backends/mock.py` is the reference implementation — no
threads, no native library, and short enough to read end to end. Structure
to copy:

- `MockStream(SttStream)` holds an `asyncio.Queue[BackendEvent | None]` and
  a `_done` flag. `push_audio` accumulates `self._audio_ms` and enqueues
  scripted partials whenever accumulated audio time crosses the next
  threshold. `events()` is a `while True: get()` loop that returns when it
  pulls the `_SENTINEL` (`None`). `finalize()` guards on `_done`, flips it,
  enqueues the final event, then the sentinel. `close()` guards on `_done`
  and, if not already done, just enqueues the sentinel (no final).
- `MockBackend(SttBackend)` sets `name` and `capabilities` as class
  attributes, has trivial `start()`/`stop()` (nothing to load), and
  `create_stream()` returns a fresh `MockStream`.
- `@register_backend("mock")` on the class is what makes `type: mock` in a
  config resolve to this class (see §2.2).

### 2.2 Registration

`src/stt_server/backends/registry.py` is a simple name → class dict.
`register_backend(type_name)` is a class decorator that inserts into
`_REGISTRY`; `create_backend(defn: BackendDef)` looks up `defn.type`,
raising `BackendUnavailableError` for an unknown type, and otherwise
constructs `cls(**defn.options)`, wrapping a `TypeError` (bad/missing
kwargs) in the same error type. `src/stt_server/backends/__init__.py` is
what actually triggers registration — it imports `mock`, `sherpa`,
`funasr`, `qwen3asr` purely for their side effect (the `@register_backend`
decorator running at import time):

```python
from stt_server.backends import (
    funasr,  # noqa: F401  (registers the funasr backend)
    mock,  # noqa: F401  (registers the mock backend)
    qwen3asr,  # noqa: F401  (registers the qwen3asr backend)
    sherpa,  # noqa: F401  (registers the sherpa_onnx backend)
)
```

A new backend module must be added to this import list to be reachable at
all, even though it can be imported directly for the extras-not-installed
check to run without ever loading the heavy dependency (see below).

### 2.3 Lazy import + `BackendUnavailableError` extras-hint pattern

Every real backend's module docstring states the same invariant: importing
the backend module (and therefore `stt_server.backends` as a whole, since
`__init__.py` imports every backend) must never require the optional
extra to be installed. The pattern, consistent across
`sherpa/backend.py`, `funasr/backend.py`, and `qwen3asr/backend.py`:

1. At module level, import only stdlib/always-available things
   (`asyncio`, `importlib.util`, `concurrent.futures.ThreadPoolExecutor`,
   the plugin base types).
2. In `__init__`, gate on `importlib.util.find_spec("<pkg>") is None` and
   raise `BackendUnavailableError` with an actionable `pip install
   'stt-server[<extra>]'` message if the package isn't importable. This
   check runs at backend *construction* time (i.e. server startup, via
   `create_backend`), not at module import time.
3. The real `from <pkg> import ...` only happens inside a method that's
   actually called after the availability check has passed — e.g.
   `SherpaBackend._build_recognizer`, `FunasrBackend._build_model`,
   `Qwen3AsrBackend._build_model` — so the heavy dependency (onnxruntime,
   torch/torchaudio, vllm) is never imported unless that specific backend
   is actually configured and started.

This is what lets the server "always boot with the mock backend" (spec
§6) even when none of the optional extras are installed, and it's what
`test_missing_sherpa_onnx_raises_unavailable_with_extras_hint` (in
`tests/backends/test_sherpa_backend.py`) and its funasr/qwen3asr
equivalents assert directly (skipped when the extra actually *is*
installed, so they only run in the environment where they're meaningful).

### 2.4 Hardened concurrency patterns

The three real backends share a native/blocking inference call that must
not run on the asyncio event loop, plus per-stream state (a native
decoder object, a mutable cache dict, a streaming state object) that is
not safe for concurrent use. `SherpaStream` (`sherpa/backend.py`) and
`FunasrStream` (`funasr/backend.py`) are the canonical examples; the same
shape appears in `Qwen3AsrStream` (`qwen3asr/backend.py`). Cite these,
don't reinvent:

- **Per-stream `asyncio.Lock` serializing decode + enqueue.** In
  `push_audio`, `self._lock` wraps the `run_in_executor` call *and* the
  subsequent `self._queue.put(...)` of the resulting partial, so a decode
  result is never enqueued out of order relative to another decode on the
  same stream (see `SherpaStream.push_audio`, `FunasrStream.push_audio`).
  `finalize` holds the lock only around its final decode and enqueues the
  final event *after* releasing it — that is safe because `finalize` flips
  `self._done = True` *before* acquiring the lock, so no concurrent
  `push_audio` or `close()` can enqueue anything afterward; the ordering
  guarantee there rests on the `_done` flag, not on lock-scoped enqueue.
- **`_done` recheck inside the lock.** `push_audio` checks `self._done`
  once before acquiring the lock (fast path for the common already-closed
  case) and — critically — again immediately after acquiring it:
  `if self._done: return  # close()/finalize() won the lock while we
  waited`. Without the second check, a `push_audio` that was blocked
  waiting for the lock could still run its decode and enqueue a partial
  after `close()`/`finalize()` already put the sentinel, corrupting the
  "exactly one final, then done" invariant.
- **Lock-aware `close()`.** `close()` also acquires `self._lock` before
  checking/flipping `_done`, so it waits out any in-flight `push_audio`
  decode rather than racing it — see the comment on `SherpaStream.close`:
  "wait out any in-flight push_audio decode so its partial (enqueued
  inside the lock) always lands *before* the end-of-stream sentinel —
  otherwise the event would be enqueued after the iterator ended, onto a
  dead queue."
- **Non-blocking executor shutdown.** `stop()` calls
  `self._executor.shutdown(wait=True)` — but wrapped in `await
  asyncio.to_thread(...)`, not awaited directly, so waiting for in-flight
  decodes to drain never blocks the event loop (both `SherpaBackend.stop`
  and `FunasrBackend.stop` do this, with the same one-line comment
  explaining why).

### 2.5 Execution-strategy guidance (spec §2.1, "Approach C — per-backend strategy")

The server core is a single asyncio event loop; each backend chooses and
owns its own concurrency model rather than the core imposing one:

| Backend | Strategy | Why |
|---|---|---|
| mock | pure asyncio, no threads | nothing blocking to offload |
| sherpa-onnx | dedicated `ThreadPoolExecutor` | onnxruntime releases the GIL during inference, so a thread pool gets real parallelism; native calls are synchronous |
| FunASR | bounded `ThreadPoolExecutor` | `model.generate(...)` is a synchronous, blocking torch call |
| Qwen3-ASR | bounded `ThreadPoolExecutor` (not an async engine) | verified against the real framework source: `Qwen3ASRModel.LLM(...)` wraps vLLM's **synchronous, blocking** `LLM` class, not `AsyncLLMEngine` — see the module docstring in `qwen3asr/backend.py` for the full research trail. The original plan's assumption that vLLM's async engine could be awaited in-loop with an `asyncio.Semaphore` turned out to be wrong; the adapter uses the executor pattern instead, with `max_concurrent` sized as `pool_workers` in the other two backends |

Rule of thumb for a new backend: if the underlying inference call is a
synchronous/blocking Python call (true of essentially every ONNX/torch/CTranslate2-style
engine), use a per-backend `ThreadPoolExecutor` and `run_in_executor`,
following §2.4's locking pattern. Only skip the executor if the engine's
own API is genuinely `async def` end to end (verify this against the
engine's actual source before assuming it, the way the Qwen3-ASR docstring
does — don't trust a framework's marketing copy about "async" support).

## 3. The conformance suite

`tests/backends/conformance.py` defines `BackendConformanceSuite`, a
pytest base class every backend's test module subclasses:

```python
class BackendConformanceSuite:
    @pytest.fixture
    def backend(self) -> SttBackend:
        raise NotImplementedError("subclass must provide a backend fixture")
```

It asserts, against whatever `backend` fixture the subclass provides:

- `test_start_stop` — `start()`/`stop()` don't raise.
- `test_finalize_yields_exactly_one_final_last` — exactly one final event,
  and it's the last event.
- `test_partials_are_ordered_by_audio_time` — `audio_time_ms` is
  non-decreasing across all emitted events.
- `test_close_without_finalize_ends_iterator` — `close()` without
  `finalize()` still ends `events()` (must not hang).
- `test_push_after_finalize_is_noop` — `push_audio`/`close` after
  `finalize` produce no further events.

Each real backend has two test modules:

- A "static" module runnable without the optional extra/model weights
  present (`tests/backends/test_sherpa_backend.py`,
  `test_funasr_backend.py`, `test_qwen3asr_backend.py`) — these check the
  `BackendUnavailableError` extras-hint and pure-Python helpers (e.g.
  `pcm16_bytes_to_float32`), and are collected/run in the default `uv run
  pytest` suite.
- A `*_model.py` module that subclasses `BackendConformanceSuite` against
  the *real* backend and a real speech fixture
  (`tests/fixtures/speech_16k_mono_s16le.pcm`), marked `pytest.mark.model`
  (and additionally `pytest.mark.gpu` for qwen3asr) and `skipif`'d when the
  extra/model isn't present:
  `TestSherpaConformance`/`test_real_transcript_is_nonempty_and_reasonable`
  in `tests/backends/test_sherpa_backend_model.py`,
  `TestFunasrConformance`/`test_real_model_streams_incremental_partials_and_one_final`
  in `tests/backends/test_funasr_backend_model.py`,
  `TestQwen3AsrConformance`/`test_real_model_streams_incremental_partials_and_one_final`
  in `tests/backends/test_qwen3asr_backend_model.py`.

`pyproject.toml` registers both markers (`model: requires real model
weights/deps`, `gpu: requires CUDA`) and excludes them by default
(`addopts = "-m 'not model'"`), so the default `uv run pytest` never needs
model weights or heavy extras.

To subclass the suite for a new backend, mirror
`tests/backends/test_mock_backend.py`'s pattern for the fast/always-on
case:

```python
class TestMockConformance(BackendConformanceSuite):
    @pytest.fixture
    def backend(self):
        return MockBackend()
```

and, for a real-model variant, mirror `test_sherpa_backend_model.py`'s
`pytestmark = [pytest.mark.model, pytest.mark.skipif(...)]` plus a
`scope="module"` fixture (real backends are expensive to `start()`, so the
model tests build one instance and share it across the suite's test
methods) and an overridden `_run_utterance` that substitutes the real
speech fixture for the base suite's synthetic silence-adjacent PCM (a real
ASR model legitimately returns an empty transcript for near-silent noise,
which would fail the base suite's non-empty-final assertion).

To run a real-backend conformance suite locally:

```
UV_DEFAULT_INDEX="https://pypi.org/simple" uv run --extra sherpa pytest -m model tests/backends/test_sherpa_backend_model.py
UV_DEFAULT_INDEX="https://pypi.org/simple" uv run --extra funasr pytest -m model tests/backends/test_funasr_backend_model.py
UV_DEFAULT_INDEX="https://pypi.org/simple" uv run --extra qwen3asr pytest -m "model and gpu" tests/backends/test_qwen3asr_backend_model.py  # requires CUDA
```

(sherpa and funasr additionally require the model weights downloaded —
see §4's per-backend commands.)

## 4. Per-backend setup

Every profile below lives at `configs/<name>.yaml` and is runnable as
`uv run stt-server --config configs/<name>.yaml` (or the equivalent
`python -m stt_server` invocation).

### mock (`configs/mock.yaml`)

- **Extras:** none — always available, part of the base install.
- **Model:** none; scripted deterministic transcripts
  (`MockUtteranceScript` in `src/stt_server/backends/mock.py`).
- **Config:** `type: mock`, `options.partial_interval_ms: 240`.
- **Capabilities:** `streaming=True, languages=("en",)`.
- **Execution strategy:** pure asyncio (no threads).
- **Verified on:** exercised continuously by the full test suite (mock is
  the default backend for every API/protocol/session test in the repo);
  boots and serves in the CPU Docker image.

### sherpa_onnx (`configs/sherpa.yaml`)

- **Extras:** `pip install 'stt-server[sherpa]'` (`sherpa-onnx>=1.10`).
- **Model download:**
  `uv run python scripts/download_models.py sherpa-zipformer-en` — fetches
  the k2-fsa GitHub release asset
  `sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2` and extracts it
  under `models/`.
- **Config:** `type: sherpa_onnx`, `options.model_dir:
  models/sherpa-onnx-streaming-zipformer-en-2023-06-26`,
  `num_threads: 4`, `pool_workers: 8`, `language: en`. Uses the `silero`
  VAD profile (`configs/sherpa.yaml` ships `vad.kind: silero`, requiring
  `models/silero_vad.onnx` — see the `silero` artifact in
  `scripts/download_models.py`) rather than the energy VAD, for
  higher-quality endpointing in this "flagship CPU streaming" profile.
- **Capabilities:** `streaming=True, languages=("en",),
  native_endpointing=False`.
- **Execution strategy:** dedicated `ThreadPoolExecutor` (onnxruntime
  releases the GIL during inference); one native `OnlineStream` per
  utterance, decode calls serialized per-stream with an `asyncio.Lock`.
- **Verified on:** ran end-to-end on this repo's macOS CPU development
  machine against the real downloaded model and a real speech fixture
  (Task 5). The model-marked test observed a real, sensible partial/final
  transcript **"CANOE SLID"** for the well-known "the canoe slid on the
  smooth planks" calibration sentence (`tests/backends/test_sherpa_backend_model.py`,
  `.superpowers/sdd/task-5-report.md`). Not run in CI (requires the extra
  + downloaded weights); the CPU Docker image bakes the extra in but the
  model weights are bind-mounted at runtime, not baked into the image.

### funasr (`configs/funasr.yaml`)

- **Extras:** `pip install 'stt-server[funasr]'` (`funasr>=1.2`,
  `torch>=2.4`, `torchaudio>=2.4`).
- **Model download:**
  `uv run python scripts/download_models.py paraformer-zh-streaming` —
  this is a "prewarm" entry, not a direct file download: FunASR's
  `AutoModel(model="paraformer-zh-streaming")` pulls weights from
  modelscope into modelscope's own cache directory the first time it's
  constructed; the script just triggers that download ahead of time and
  drops a marker file (`models/.funasr-paraformer-zh-streaming.prewarmed`)
  so re-running is a cheap no-op.
- **Config:** `type: funasr`, `options.model: paraformer-zh-streaming`,
  `pool_workers: 4`, `chunk_size: [0, 10, 5]` (element `[1]`, 10, is the
  decode-chunk length in 60 ms units — 600 ms of audio per `generate()`
  call, i.e. 9600 samples / 19200 bytes of PCM16 at 16 kHz; see the full
  derivation in the `funasr/backend.py` module docstring). Uses the
  `energy` VAD deliberately, not `silero` — see the comment in
  `configs/funasr.yaml`: it keeps this config dependency-light (no
  onnxruntime/silero model download required just to try the funasr
  backend); switch to `silero` (as in `configs/sherpa.yaml`) for better
  endpointing quality in production.
- **Capabilities:** `streaming=True, languages=("zh", "en"),
  native_endpointing=False`.
- **Execution strategy:** bounded `ThreadPoolExecutor`; `model.generate()`
  mutates a per-stream `cache` dict in place and is not safe for
  concurrent calls, so decodes are serialized per-stream with an
  `asyncio.Lock`, mirroring `SherpaStream`.
- **Verified on:** ran end-to-end on this repo's macOS CPU development
  machine against the real downloaded `paraformer-zh-streaming` model
  (Task 6). Because the repo's only committed speech fixture is English
  and this is a Mandarin model, the *committed* model-marked test
  (`tests/backends/test_funasr_backend_model.py`) asserts pipeline
  mechanics only (ordering, exactly-one-final, non-empty text), not
  transcript content. Separately, and not part of the committed test
  suite, the implementer manually verified a real Mandarin transcript by
  synthesizing speech locally with macOS
  `say -v Ting-Ting -o zh_sample.aiff "今天天气很好，我们去公园散步吧。"` and feeding it
  through the real backend, producing a correct transcription of the
  input sentence (punctuation dropped, as expected for a streaming ASR
  hypothesis) — see `.superpowers/sdd/task-6-report.md` for the full
  transcript. Not run in CI; not committed (TTS-voice-output redistribution
  licensing is unclear, unlike the fixture used by the other tests).

### qwen3asr (`configs/qwen3asr.yaml`)

- **Extras:** `pip install 'stt-server[qwen3asr]'`
  (`qwen-asr[vllm]>=0.0.6`, which pulls in `vllm==0.14.0` and its own
  `transformers`/`accelerate` dependencies). Requires a CUDA GPU — the
  framework's streaming interface is documented as "vLLM backend only",
  and the vLLM backend requires CUDA.
- **Model download:** none via `scripts/download_models.py` — the
  `Qwen/Qwen3-ASR-0.6B` model (this config's default) is pulled by the
  `qwen_asr`/`transformers` stack itself (e.g. from the Hugging Face Hub)
  the first time `Qwen3AsrBackend.start()` builds the model; there is no
  separate `download_models.py` entry for it (unlike sherpa/funasr).
- **Config:** `type: qwen3asr`, `options.model: Qwen/Qwen3-ASR-0.6B`,
  `gpu_memory_utilization: 0.8`, `max_concurrent: 8`, `language: null`,
  `redecode_interval_ms: 480` — passed straight through to the framework's
  own `init_streaming_state(..., chunk_size_sec=redecode_interval_ms /
  1000.0)`, so this is the redecode cadence knob, not a separate
  buffer/stride the adapter computes itself (contrast with funasr's
  `chunk_size`, which the adapter has to convert to a byte stride by
  hand). Uses the `silero` VAD profile, same rationale as sherpa.
- **Capabilities:** `streaming=True, languages=("en", "zh")` — a
  representative subset of the 52 languages/dialects the underlying model
  actually supports (see the README's language table cited in the module
  docstring), not an exhaustive capability limit.
- **Execution strategy:** bounded `ThreadPoolExecutor` sized by
  `max_concurrent`, **not** an awaited async engine. Verified against the
  real framework source: `Qwen3ASRModel.LLM(...)` wraps vLLM's
  synchronous, blocking `LLM` class internally, so every
  `generate()`-backed call (`streaming_transcribe`,
  `finish_streaming_transcribe`) is an ordinary blocking call that must run
  off the event loop — see §2.5 and the `qwen3asr/backend.py` module
  docstring for the full research trail that overturned the original
  "AsyncLLMEngine" assumption.
- **Verified on: NOT RUN.** This backend has never executed against a
  real model on real hardware. The implementation, the concurrency
  pattern, and the conformance-suite subclass
  (`tests/backends/test_qwen3asr_backend_model.py`) were all written and
  reviewed against the real upstream framework source and README (Task
  7), but there is no CUDA GPU in this development environment, and the
  vLLM backend requires one. The GPU Docker image
  (`deploy/Dockerfile.gpu`, `--profile gpu`) that would run it is likewise
  unbuilt/unvalidated here. Running this backend for real — on a CUDA box
  — is explicitly deferred to Plan 4. Do not read "written" as "verified"
  for this backend.

## 5. Known limitations

- **`StreamConfig.language` per-request behavior (Plan 4 Task 2).**
  `create_stream(self, cfg: StreamConfig)` now reads `cfg.language` on all
  three real backends:
  - `Qwen3AsrBackend`: the framework's `init_streaming_state()` genuinely
    accepts a per-utterance language, so `cfg.language or self._language`
    is passed straight through — a client-supplied `language` on the file
    endpoint (or a future protocol surface) really does override the
    backend's configured default for that one stream.
  - `SherpaBackend` / `FunasrBackend`: both wrap a single loaded model
    that is fixed to whatever language it was built/trained for — there is
    no per-utterance language knob in the underlying engine. If
    `cfg.language` is set and isn't in the backend's
    `capabilities.languages`, it is ignored (a one-time `logger.debug`,
    never an error — OpenAI's API treats `language` as a hint, not a
    guarantee) and the stream still uses the model's own language.
  - Today the OpenAI file endpoint (`POST /v1/audio/transcriptions`) is
    the only per-request source of `language` — it already threads the
    `language` form field into `StreamConfig`. Neither the native nor the
    OpenAI Realtime WebSocket protocol carries a language field in its
    session config today, so there is nothing to wire for those surfaces
    without inventing a new protocol field, which is out of scope here.
