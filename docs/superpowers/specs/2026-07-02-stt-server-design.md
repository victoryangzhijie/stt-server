# Design: Open-Source OpenAI-Compatible STT Inference Server

**Date:** 2026-07-02
**Status:** Approved
**Working name:** repo `stt-server`, Python package `stt_server` (repo may be
renamed at publish time; the package name is fixed now so implementation can
proceed)

## 1. Purpose and success criteria

A research-oriented, production-minded real-time speech-to-text serving system,
published as a public GitHub repository. The primary purpose is **portfolio /
showcase**: the repo demonstrates real inference-serving engineering — clean
architecture, observability, reproducible benchmarks, and honest documentation.

Success means a new developer can clone the repo, run the server with the mock
backend in under five minutes, understand the architecture from the docs, swap
in a real ASR backend, and reproduce every benchmark figure with documented
commands.

**Decided constraints:**

- Language/stack: Python 3.12+, asyncio, FastAPI/Starlette + uvicorn, managed uniformly with uv.
- Benchmark hardware: user's CUDA GPU box (headline numbers); CPU path fully
  supported for dev and reproducibility.
- Speech language: English primary (LibriSpeech benchmarks); Chinese documented
  as a secondary capability where backends support it (FunASR, Qwen3-ASR).
- Qwen3-ASR: local open-weights integration (0.6B/1.7B, Apache 2.0, released
  2026-01-29) via its official vLLM-based inference framework.
- License: MIT.

**Out of scope (hard):** TTS, LLM features, SIP/telephony, billing, user
management, polished frontend, generic chatbot behavior. No claims of OpenAI
compatibility beyond what is tested and documented.

## 2. Architecture overview

Single asyncio Python process serving all APIs. Scale-out beyond one process is
via replicas behind a load balancer — stated and measured in the benchmark
report, not built.

```
                       ┌─────────────────────────────────────────────┐
 WebSocket (Realtime)  │  API layer (protocol adapters)              │
 ──────────────────►   │   • realtime_ws  (OpenAI Realtime-style)    │
 HTTP (file upload)    │   • transcriptions_http (OpenAI file-style) │
 ──────────────────►   │   • native_ws    (internal JSON protocol)   │
 WebSocket (native)    │  encode/decode ONLY — no ASR logic          │
 ──────────────────►   ├─────────────────────────────────────────────┤
                       │  Session core (protocol-agnostic)           │
                       │   audio ingest → resample/format → VAD →    │
                       │   endpointing → backend stream → transcript │
                       │   stabilizer → TranscriptEvent bus          │
                       ├─────────────────────────────────────────────┤
                       │  Backend plugins (common async interface)   │
                       │   mock │ sherpa-onnx │ FunASR │ Qwen3-ASR   │
                       │   each with its own execution strategy      │
                       └─────────────────────────────────────────────┘
```

**Load-bearing rule:** the session core speaks only internal types. It never
imports from the API layer. Protocol adapters convert wire formats to/from the
internal model at the boundary.

### 2.1 Execution model (Approach C — per-backend strategy)

The server core is one asyncio event loop. The backend plugin interface is
async; each backend declares and owns its concurrency model:

- **mock** — pure asyncio, no threads.
- **sherpa-onnx** — dedicated `ThreadPoolExecutor` (onnxruntime releases the
  GIL during inference).
- **FunASR** — bounded thread pool; optional micro-batching worker if
  measurement shows benefit.
- **Qwen3-ASR** — delegates to the official Qwen3-ASR inference framework's
  vLLM async engine, which manages GPU batching itself.

Rationale: keeps the codebase focused on the novel parts (streaming protocol,
endpointing, stabilization, observability) while giving each backend the
execution shape it actually needs. The benchmark report's bottleneck-analysis
chapter provides measured evidence of where this design saturates per backend.

### 2.2 Repository layout

```
src/stt_server/
  api/            # realtime_ws, transcriptions_http, native_ws, encoders
  core/           # session, audio, vad, endpointing, stabilizer, events
  backends/       # registry, base interface, mock/, sherpa/, funasr/, qwen3asr/
  metrics/        # prometheus registry, latency accounting
  config/         # pydantic-settings models, loader
benchmarks/       # accuracy, latency, load generator, stabilization study
tests/            # protocol, vad, stabilizer, plugin conformance, file API
docs/             # architecture.md, openai-compat.md, backends.md, benchmarks/
deploy/           # Dockerfile, Dockerfile.gpu, docker-compose.yaml, grafana/
examples/         # CLI streaming client, minimal HTML demo page (unpolished)
```

## 3. Internal data model

### 3.1 AudioChunk

Internal audio format is **PCM16 mono 16 kHz**. Adapters convert inbound audio
(resample / channel-mix / decode) at the boundary. Every `AudioChunk` carries
an ingest timestamp (monotonic clock) used for all latency accounting.

### 3.2 TranscriptEvent

Unified event model emitted by the session core; all protocol encoders consume
only this stream:

- `SPEECH_START` / `SPEECH_END` — VAD lifecycle, with audio-time offsets.
- `PARTIAL` — full current hypothesis for the active utterance, split into
  `{stable_text, volatile_text}`.
- `STABILIZED` — fired when the committed prefix grows; carries the delta.
- `FINAL` — utterance complete; final text, audio-time span, latency metadata.
- `ERROR` — structured error with code, recoverability flag.

Every event carries `session_id`, `utterance_id`, sequence number, and timing
metadata (audio-time offsets and wall-clock latency measurements).

### 3.3 Session

The unit of work. One session per WebSocket connection; one ephemeral session
per file-transcription request. **File mode reuses the same pipeline** by
streaming decoded audio through it faster than real time; when a backend
supports whole-file (non-streaming) decoding, the batch path may bypass VAD
for accuracy parity — both paths are benchmarked.

## 4. Pipeline design

### 4.1 VAD

Default: **Silero VAD** (ONNX build, CPU, ~1 ms per 30 ms frame), behind a
small `VadDetector` interface. A trivial energy-based detector serves as a
test/fallback implementation. VAD runs **centrally in the session core**, not
in backends, so endpointing behavior is identical across backends — essential
for fair benchmark comparison. Backends may declare native endpointing
capability; config selects `endpointing: server | backend`, and comparing the
two modes is a documented experiment in the benchmark report.

### 4.2 Endpointing state machine

States: `IDLE → SPEECH → ENDPOINTING → (FINAL, reset)`.

- Speech-start includes a **pre-roll buffer** (config, default ~300 ms) so
  utterance onsets are not clipped.
- Trailing silence of `min_silence_ms` (default ~500 ms) triggers finalize.
- `max_utterance_ms` caps runaway utterances (forced finalize).
- On endpoint: tell backend stream to finalize → emit `FINAL` → increment
  utterance id → reset.

All thresholds are config-exposed.

### 4.3 Transcript stabilizer

Maintains a per-utterance **committed prefix that only grows**. A token is
committed once it survives unchanged (token-level longest-common-prefix
comparison across successive partials) for `N` consecutive partials **and**
`T` ms (defaults: 2 partials / 400 ms; both configurable). `PARTIAL` events
expose `{stable_text, volatile_text}`; `STABILIZED` fires on prefix growth.

Invariants (property-tested):
- Committed prefix never shrinks within an utterance.
- `FINAL` text always equals the backend's final hypothesis (stabilization
  never delays or alters finals; it only shapes partial display).

Documented trade-off: stabilization adds display latency to committed text.
The benchmark suite measures the stability-vs-commit-latency curve across
parameter settings.

### 4.4 Latency accounting

Measured in the core (comparable across backends by construction). Per
utterance: **first-partial latency** (speech-start → first PARTIAL emitted),
**final latency** (silence-end → FINAL emitted), and **real-time factor**.
Aggregated per session and exported as Prometheus histograms.

## 5. API layer

### 5.1 OpenAI Realtime-style WebSocket — `/v1/realtime?intent=transcription`

Implements the transcription-session event vocabulary:

- Client → server: `transcription_session.update`, `input_audio_buffer.append`
  (base64 or binary), `input_audio_buffer.commit` where applicable.
- Server → client: `transcription_session.created/updated`,
  `input_audio_buffer.speech_started` / `speech_stopped`,
  `conversation.item.input_audio_transcription.delta` (append-only deltas
  sourced from committed-prefix growth — an exact semantic match),
  `conversation.item.input_audio_transcription.completed`, `error`.

Encoders are pure functions `TranscriptEvent → wire JSON`, table-tested
against recorded fixtures. **`docs/openai-compat.md`** contains a full event
matrix: supported / partially supported / not supported, each with test
references. No compatibility claim without a test.

### 5.2 OpenAI file-style HTTP — `POST /v1/audio/transcriptions`

Multipart upload; `model` selects backend; `response_format` supports `json`,
`text`, `verbose_json` (segments with timings). Audio decoded via
soundfile/ffmpeg to internal PCM.

### 5.3 Native WebSocket — `/ws/transcribe`

Thin JSON protocol exposing internal events directly, including the
stable/volatile split (inexpressible in the OpenAI protocol). This is the
protocol benchmark clients use, so benchmark instrumentation never depends on
OpenAI framing.

### 5.4 Auth and limits

Static bearer-token list from config (sufficient and honest for this scope).
Config-enforced limits: max concurrent sessions, max session duration, max
upload size. Structured error responses on limit breach.

## 6. Backend plugin interface

```python
class SttBackend(Protocol):
    name: str
    capabilities: BackendCapabilities   # streaming, languages,
                                        # native_endpointing, batch_decode, ...
    async def start(self) -> None       # load model / warm up
    async def stop(self) -> None
    async def create_stream(self, cfg: StreamConfig) -> SttStream

class SttStream(Protocol):
    # Lifecycle: one SttStream per utterance — created on speech-start,
    # finalized on endpoint, closed after FINAL. Keeps backend state reset
    # trivial and the conformance contract unambiguous.
    async def push_audio(self, chunk: AudioChunk) -> None
    def events(self) -> AsyncIterator[BackendEvent]  # partials/finals + timings
    async def finalize(self) -> None    # endpoint reached; flush
    async def close(self) -> None
```

- Registry keyed by backend name; config maps served "model" names →
  configured backend instances.
- Heavy backends are optional extras (`pip install .[sherpa]`, `.[funasr]`,
  `.[qwen3asr]`). Missing deps → structured "backend unavailable" error at
  request time; the server **always boots with the mock backend**.
- A reusable **plugin conformance test suite** (a pytest base class) defines
  the behavioral contract; every backend must pass it.

### 6.1 Backend implementations

| Backend | Model(s) | Streaming | Execution | Role |
|---|---|---|---|---|
| mock | scripted | yes (deterministic timing) | pure asyncio | CI/testing reference; always available |
| sherpa-onnx | streaming Zipformer (English) | true incremental | dedicated thread pool | CPU streaming flagship |
| FunASR | Paraformer streaming (EN/ZH) | chunk-based | bounded thread pool | second real backend; ZH secondary capability |
| Qwen3-ASR | 0.6B / 1.7B open weights | chunk-based via official framework | vLLM async engine (GPU) | GPU flagship; accuracy headline |

The mock backend supports scripted partial sequences with configurable timing
and jitter — the substrate for protocol, endpointing, and stabilizer tests.

## 7. Observability

- **Logs:** structured JSON (structlog); `session_id`/`utterance_id` bound on
  every line; per-session summary on close (audio duration, utterance count,
  latency percentiles).
- **Metrics:** Prometheus `/metrics` — active sessions, audio-seconds
  ingested, per-backend queue depth and inference-duration histograms,
  first-partial/final latency histograms, RTF, endpoint counts, errors.
- **Health:** `/healthz` (liveness), `/readyz` (configured backends loaded).

## 8. Configuration

Pydantic-settings. Single `config.yaml` covering: server (host/port/limits),
VAD + endpointing thresholds, stabilizer parameters, backend definitions and
model paths, auth tokens. Env-var overrides with `STT__` prefix. Example
configs checked in per backend profile. No configuration in code.

## 9. Deployment

- `deploy/Dockerfile` — CPU image: mock + sherpa-onnx + FunASR.
- `deploy/Dockerfile.gpu` — CUDA base; adds Qwen3-ASR / vLLM.
- `deploy/docker-compose.yaml` — profiles: `cpu`, `gpu`, `observability`
  (Prometheus + Grafana with a pre-built dashboard JSON).
- Models mounted as volumes; documented download script (`scripts/`).

## 10. Benchmarking and load testing

All under `benchmarks/`, pinned dependencies, seeded where applicable; every
figure in the report regenerable by one documented command.

1. **Accuracy** — LibriSpeech test-clean/test-other WER per backend, measured
   *through the serving system*: (a) streamed at real-time pace over the
   native WS API, (b) via the file API. jiwer with standard normalization.
   Captures endpointing-induced errors, not just model quality.
2. **Latency** — per-utterance first-partial and final latency distributions
   at concurrency 1, per backend.
3. **Load** — in-repo async load generator replaying real-time-paced audio
   over N concurrent WS sessions; ramp N until latency SLO breaks. Reports
   max sessions per CPU/GPU, p50/p95/p99 vs concurrency, resource usage
   (psutil + pynvml sampling).
4. **Stabilization study** — flicker rate (retracted characters per emitted
   character) vs commit latency across stabilizer parameter grid.
5. **Endpointing experiment** — server VAD vs backend-native endpointing,
   where supported.
6. **Bottleneck analysis** — written chapter interpreting the above: where
   each backend saturates and why (GIL contention, queue depth, GPU batching
   limits), with supporting metrics.

Outputs: `docs/benchmarks/methodology.md` (how to reproduce) and
`docs/benchmarks/report.md` (sample report with real numbers from the CUDA
box, hardware documented).

## 11. Testing

pytest + pytest-asyncio; CI requires **no model backends** — tests run against
the mock backend, with the energy-based VAD so even onnxruntime (Silero VAD's
runtime) is optional in CI. Real-model tests marked `@pytest.mark.model`, run
locally only.

- **Protocol compatibility:** recorded OpenAI-event fixtures exercised against
  the mock backend over a real WebSocket connection; encoder table tests.
- **VAD/endpointing:** synthetic speech/silence patterns through the state
  machine; pre-roll, min-silence, max-utterance edge cases.
- **Stabilizer:** property tests for the invariants in §4.3.
- **Plugin conformance:** reusable suite run against mock in CI; runnable
  against real backends locally.
- **File API:** tiny WAV fixtures end-to-end.

## 12. Documentation deliverables

- `README.md` — pitch, architecture summary, <5-minute quickstart with mock
  backend (one `docker run`), links to all docs.
- `docs/architecture.md` — components, data flow, execution model, diagrams.
- `docs/openai-compat.md` — tested compatibility matrix.
- `docs/backends.md` — plugin contract; how to write a backend.
- `docs/benchmarks/{methodology,report}.md` — as in §10.

## 13. Error handling principles

- Protocol errors (bad JSON, wrong audio format, unknown model) → structured
  error events/responses with stable codes; session survives where the
  protocol allows.
- Backend failures mid-stream → `ERROR` event with recoverability flag;
  session terminates cleanly; metrics increment.
- Backpressure: bounded per-session audio queues; if a backend can't keep up,
  the server sheds by policy (config: `drop_oldest | error`) and logs/counts
  it — never unbounded memory growth.
- Startup: missing model files fail `readyz` with actionable log messages;
  mock backend keeps the server usable.

## 14. Build order (high level, for the implementation plan)

1. Core skeleton: config, events, session, mock backend, native WS API, tests.
2. VAD + endpointing + stabilizer with full test coverage.
3. OpenAI protocol adapters (realtime WS + file HTTP) + compat matrix.
4. Observability (metrics, structured logs, health endpoints).
5. sherpa-onnx backend + plugin conformance hardening.
6. FunASR backend; Qwen3-ASR backend (GPU).
7. Docker/compose + Grafana dashboard.
8. Benchmark suite + load generator; run on GPU box; write report.
9. Documentation pass; publish.
