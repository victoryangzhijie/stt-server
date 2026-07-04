# stt-server

A research-oriented, production-minded real-time speech-to-text serving
system: an OpenAI-compatible WebSocket/HTTP surface in front of a
protocol-agnostic core (VAD → endpointing → per-backend streaming decode →
transcript stabilization), with four pluggable open-source ASR backends
(mock, sherpa-onnx, FunASR, Qwen3-ASR), each running under its own execution
strategy rather than one forced concurrency model. Compatibility claims are
scoped to what's tested — see `docs/openai-compat.md`'s event matrix — and
the benchmark harness under `benchmarks/` is built to make every reported
number reproducible with one documented command, not just presented. See
`docs/superpowers/specs/2026-07-02-stt-server-design.md` for the full design
and the **Project status** section below for what's implemented vs. pending.

## Quickstart (dev)

    uv sync
    uv run pytest

## Backends

| Backend | Type string | Extras | Streaming strategy | Verified on |
|---|---|---|---|---|
| mock | `mock` | none (base install) | pure asyncio | continuous — default backend for the whole test suite |
| sherpa-onnx | `sherpa_onnx` | `sherpa` | dedicated `ThreadPoolExecutor` | real model, macOS CPU, real transcript ("CANOE SLID") |
| FunASR | `funasr` | `funasr` | bounded `ThreadPoolExecutor` | real model, macOS CPU, real Mandarin transcript (manual) |
| Qwen3-ASR | `qwen3asr` | `qwen3asr` | bounded `ThreadPoolExecutor` (vLLM's sync `LLM`, not an async engine) | **not run** — no local CUDA; deferred to Plan 4 |

See [`docs/backends.md`](docs/backends.md) for the plugin contract, how to
write a backend, the conformance test suite, execution-strategy rationale,
and full per-backend setup (extras install, model download commands, config
profiles, honest verified-on notes).

## Documentation index

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Components, data flow (mermaid diagrams for the session pipeline and the concurrency model), execution model, layering rule |
| [`docs/backends.md`](docs/backends.md) | Plugin contract, conformance suite, how to write a backend, per-backend setup/verification |
| [`docs/openai-compat.md`](docs/openai-compat.md) | Tested OpenAI Realtime / file-transcription compatibility matrix |
| [`benchmarks/README.md`](benchmarks/README.md) | Benchmark suite: accuracy, latency, load, stabilizer study, endpointing — one-command invocations |
| [`docs/benchmarks/methodology.md`](docs/benchmarks/methodology.md) | How the CPU benchmark numbers were produced: hardware, environments, every command verbatim, SLO rationale, threats to validity |
| [`docs/benchmarks/report.md`](docs/benchmarks/report.md) | CPU benchmark results: WER, latency, load knee, stabilizer grid, endpointing comparison, bottleneck analysis (GPU pending) |
| [`benchmarks/cuda_runbook.md`](benchmarks/cuda_runbook.md) | One-command GPU benchmark procedure for a CUDA box |

## Quickstart (Docker)

Bring up the CPU image (mock + sherpa-onnx + funasr + silero backends), which
boots on `configs/mock.yaml` by default:

    docker compose -f deploy/docker-compose.yaml --profile cpu up --build

Smoke-test it once it's up:

    curl http://localhost:8000/healthz

A GPU image (qwen3asr backend, requires the NVIDIA Container Toolkit) is
available under the `gpu` profile:

    docker compose -f deploy/docker-compose.yaml --profile gpu up --build

### Configuration

The image ships with `configs/*.yaml` baked in and boots
`configs/mock.yaml` (CPU) / `configs/qwen3asr.yaml` (GPU) by default. To use a
different config (e.g. one with `auth.tokens` set), bind-mount it over the
baked-in path and pass `--config`, or override individual fields with
`STT__`-prefixed environment variables (env wins over the YAML file). The
compose services pass through `STT__AUTH__TOKENS`, `STT__SERVER__PORT`, and
`STT__LIMITS__MAX_SESSIONS` from the shell that runs `docker compose`, so
this works as-is:

    STT__AUTH__TOKENS='["my-secret-token"]' \
        docker compose -f deploy/docker-compose.yaml --profile cpu up

(To pass through additional `STT__` variables, add them to the service's
`environment:` list in `deploy/docker-compose.yaml`.)

Model weights are not baked into the image. Download them to `models/` on
the host — the compose file bind-mounts the repo's `models/` directory into
the container at `/app/models` — before using the
sherpa/funasr/silero/qwen3asr backends:

    uv run python scripts/download_models.py all

### Observability

The `observability` profile adds Prometheus + a pre-provisioned Grafana
dashboard (active sessions, first-partial/final latency percentiles,
utterance/rejection/error rates, audio ingested). Combine it with a server
profile — on its own it would start Prometheus/Grafana with nothing to
scrape:

    docker compose -f deploy/docker-compose.yaml --profile cpu --profile observability up

Grafana is reachable at `http://localhost:3000` (default credentials
`admin` / `admin` — change this before any non-local deployment). Prometheus
itself is intentionally not published to the host.

**Important:** `/metrics` on stt-server is deliberately unauthenticated (see
`src/stt_server/metrics/registry.py` and the design spec). In this compose
setup, Prometheus reaches it only over the internal `internal` Docker
network — it is never published as a separate host port. If you deploy
stt-server behind a reverse proxy or publish port 8000 beyond your trusted
network, make sure `/metrics` is not reachable by untrusted clients (e.g.
block it at the proxy, or keep the whole service on a private network).

## Project status

**Implemented and tested:**

- Four backends behind one plugin contract, each with its own execution
  strategy (§3.1 of `docs/architecture.md`) — mock (continuous, the default
  for the whole test suite), sherpa-onnx and FunASR (real models, verified
  on macOS CPU with real transcripts — see the Backends table above),
  Qwen3-ASR (implemented against the real `qwen-asr` framework API, but
  **not run** — no local CUDA; correctness is exercised only through the
  plugin conformance suite against the mock backend).
- Three protocol adapters (native WS, OpenAI Realtime-style WS, OpenAI
  file-style HTTP) sharing one protocol-agnostic core; auth (constant-time
  bearer tokens), session/upload/duration limits, and a pre-parse upload-size
  guard.
- Backpressure: a bounded per-session audio queue with a configurable
  `drop_oldest | error` shedding policy, both covered by tests.
- Observability: structured JSON logs with per-session summaries, a
  dedicated Prometheus registry, and a pre-provisioned Grafana dashboard.
- Docker: CPU and GPU compose profiles, an `observability` profile, and a
  documented model-download script.
- Benchmark harness: accuracy/WER, latency, load, stabilizer-study, and
  endpointing runners under `benchmarks/`, each with a one-command
  invocation — **run for real on a local CPU box** against mock, sherpa-onnx
  and FunASR; see [`docs/benchmarks/report.md`](docs/benchmarks/report.md)
  for the numbers and
  [`docs/benchmarks/methodology.md`](docs/benchmarks/methodology.md) for
  exactly how they were produced.
- 325 tests green (`uv run pytest`), ruff clean, layering rule holds
  (`grep -rn "from stt_server.api" src/stt_server/core/*.py` returns
  nothing).

**Pending:**

- GPU verification (Qwen3-ASR end-to-end, GPU load/latency numbers) awaits
  access to a CUDA box; `scripts/run_gpu_suite.sh` plus
  `benchmarks/cuda_runbook.md` give a one-command procedure for that box
  once available. `docs/benchmarks/report.md`'s "GPU results (pending)"
  section lists exactly which artifacts will fill it.
