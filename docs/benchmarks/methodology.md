# Benchmark methodology (CPU, local)

This document records exactly how every number in [`report.md`](report.md)
was produced: the machine, the environments, the corpus and its seeding,
each command verbatim, why the SLOs were chosen, and the known threats to
validity. Result JSONs live in the gitignored `benchmarks/results/` and are
not committed — the report transcribes the numbers, and this document makes
each one reproducible. See `benchmarks/README.md` for what each runner
measures and how.

## Hardware / OS / software

| Item | Value |
|---|---|
| CPU | Intel Core i7-4770HQ @ 2.20 GHz (Haswell, 4 physical / 8 logical cores) |
| Memory | 16 GiB |
| GPU | none — CUDA results pending, see the report's "GPU results (pending)" section |
| OS | macOS 12.7.6 (21H1320), x86_64 |
| Python | 3.12.12 (same interpreter version in all three environments) |
| uv | 0.10.7 |
| stt-server | 0.1.0, editable install in every venv |
| git SHA at run time | accuracy runs: `93939ce213f2640c703fe7adabef6f6e59e3a3d3` (tip of plan-4 branch, later merged to main); load/stabilizer runs: `b1096ee07bf85b61d54c6d9ef18cdc6a45250783` (main + docs-only commits — no benchmark or server code differs between the two) |

Three Python environments (three dependency sets; the sherpa/funasr pins
cannot coexist with the project's locked environment on this macOS 12 box):

| Environment | How built | Key packages |
|---|---|---|
| repo venv | `UV_DEFAULT_INDEX="https://pypi.org/simple" uv sync --extra bench` | fastapi 0.139.0, uvicorn 0.49.0, websockets 16.0, pydantic 2.13.4, numpy 2.2.6, jiwer 4.0.0, psutil 7.2.2, soundfile 0.14.0, httpx 0.28.1 |
| sherpa venv | `python3.12 -m venv sherpa-venv && sherpa-venv/bin/pip install -i https://pypi.org/simple -e '.[sherpa,bench]' 'sherpa-onnx==1.10.46' 'onnxruntime==1.19.2' httpx` (recipe in `benchmarks/README.md`; both pins are the newest wheels that load on macOS 12 x86_64) | sherpa-onnx 1.10.46, onnxruntime 1.19.2, numpy 2.5.0, jiwer 4.0.0, psutil 7.2.2, soundfile 0.14.0 |
| funasr venv | Plan 3 Task 6 recipe (`.superpowers/sdd/task-6-report.md`): `torch==2.2.2`/`torchaudio==2.2.2` (newest macOS x86_64 torch wheels), `llvmlite==0.43.0`+`numba==0.60.0` pre-pinned, `numpy<2`, then `funasr`; this task added `pip install -i https://pypi.org/simple 'jiwer>=3.0' 'psutil>=5.9' 'pynvml>=11.5'` | funasr 1.3.14, torch 2.2.2, modelscope 1.38.0, numpy 1.26.4, jiwer 4.0.0 |

Models: `models/sherpa-onnx-streaming-zipformer-en-2023-06-26/` (streaming
Zipformer transducer, English), `models/silero_vad.onnx` (the silero VAD
`configs/sherpa.yaml` uses), and FunASR `paraformer-zh-streaming` resolved
from a pre-warmed ModelScope cache (`MODELSCOPE_CACHE` pointed at a local
cache directory; no model download happened during any run).

In the commands below, `$SHERPA_PY` / `$FUNASR_PY` stand for the venvs'
interpreters and `$MODELSCOPE_CACHE` for the pre-warmed cache path.

## Corpus and seeding

All accuracy/stabilizer/endpointing runs use LibriSpeech **test-clean**
(pre-downloaded under `benchmarks/data/LibriSpeech/`), sampled with
`benchmarks.corpus.build_manifest` — deterministic for a given `(n, seed)`:

| Manifest | Used by | Total audio |
|---|---|---|
| n=100, seed=42 | accuracy + concurrency-1 latency | 800.5 s (13.34 min) |
| n=25, seed=42 | stabilizer study (native + realtime) | 191.8 s (3.20 min) |
| n=10, seed=42 | endpointing experiment | 70.5 s (1.17 min) |

The smaller manifests are prefixes of the same seeded shuffle, so runs
share utterance identity where they overlap. The load runs use either the
tiled synthetic fixture (mock) or `run_load`'s own small seeded manifest
(sherpa, n=8 = `min(50, max(--max, 5))`, seed 42).

## The pace / zero-drops rule (binding)

**Every WER-through-the-server measurement against a real backend streams
at `--pace 1.0` (real time) and asserts a zero `stt_audio_dropped_total`
delta.** Plan 4 Task 7's review established by direct reproduction that a
faster-than-real-time client (`--pace 0`) outruns a real decoder and trips
the server's default backpressure (`audio_queue_chunks=64`,
`audio_overflow_policy=drop_oldest` — the defaults whenever a config has no
`limits:` section, which includes `configs/sherpa.yaml`), silently shedding
audio: the resulting "WER" measures accuracy under active audio drop, not
the pipeline (measured there: WER 0.33 with ~165 dropped chunks at pace 0
vs 0.099 with 0 drops at pace 1.0, n=5). All streaming runners therefore
bracket their pass with `/metrics` scrapes, record the
`stt_audio_dropped_total` delta in the result JSON
(`audio_dropped_delta`), and hard-fail on a nonzero delta
(`--assert-no-drops`, default ON). The report's tables carry the recorded
delta for every streaming run. Mock-backend runs use `--pace 0` (the mock
keeps up by construction; its delta is still asserted and recorded).
`run_accuracy`'s file mode is exempt by construction (the whole upload is
one `AudioChunk`; nothing to overflow), and `run_load` deliberately
*measures* drops instead of asserting (a drop fails the rung's SLO gate).

Consequence: a real-backend streaming pass takes at least as long as the
subset's audio — 13.3 min for the n=100 accuracy pass, ~3.2 min per
stabilizer grid point.

## Commands (verbatim, with the environment that ran each)

All commands run from the repo root, sequentially — never two benchmark
processes at once (CPU contention would contaminate latencies).

### 1. Accuracy + concurrency-1 latency (`run_accuracy`)

Mock — repo venv (pace 0 is valid for the mock; ~12 s wall):

```
UV_DEFAULT_INDEX="https://pypi.org/simple" uv run python -m benchmarks.run_accuracy \
    --config configs/mock.yaml --model mock --split test-clean \
    --n 100 --seed 42 --modes ws,file --port 8100 --pace 0
```

sherpa-onnx — sherpa venv (real-time paced; ~17 min wall):

```
$SHERPA_PY -m benchmarks.run_accuracy \
    --config configs/sherpa.yaml --model sherpa-zipformer-en --split test-clean \
    --n 100 --seed 42 --modes ws,file --port 8101 --pace 1.0 \
    --python $SHERPA_PY
```

FunASR — funasr venv (real-time paced; 17 min 45 s wall):

```
MODELSCOPE_CACHE=$MODELSCOPE_CACHE $FUNASR_PY -m benchmarks.run_accuracy \
    --config configs/funasr.yaml --model paraformer-zh-streaming --split test-clean \
    --n 100 --seed 42 --modes ws,file --port 8102 --pace 1.0 \
    --python $FUNASR_PY
```

`paraformer-zh-streaming` is a **Mandarin** model scored against English
test-clean: its WER is cross-lingual, NOT comparable to sherpa's, and is
reported only as evidence that the pipeline mechanics (streaming, latency
accounting, drop bracketing) work against a second real decoder.

### 2. Load ramp (`run_load`)

Mock — repo venv (~9.5 min wall; all 16 rungs ran):

```
STT__LIMITS__MAX_SESSIONS=256 UV_DEFAULT_INDEX="https://pypi.org/simple" uv run python -m benchmarks.run_load \
    --config configs/mock.yaml --model mock --synthetic \
    --utterance-seconds 10 --start 8 --step 8 --max 128 \
    --window-seconds 30 --slo-final-ms 1200 --slo-pct 95 --seed 42 --port 8103
```

(`STT__LIMITS__MAX_SESSIONS=256` because the server's default
`limits.max_sessions=100` would reject sessions above 100 with a capacity
rejection before the backend saw them, and `run_load.check_capacity` fails
fast when the cap is below `--max`; the env var reaches both the runner's
config load and the spawned server.)

sherpa — sherpa venv (33.5 min wall; see threats-to-validity on wall-clock
gaps; stopped at the first failing rung, concurrency 6):

```
$SHERPA_PY -m benchmarks.run_load \
    --config configs/sherpa.yaml --model sherpa-zipformer-en \
    --utterance-seconds 10 --start 1 --step 1 --max 8 \
    --window-seconds 30 --slo-final-ms 1200 --slo-pct 95 --seed 42 --port 8104 \
    --python $SHERPA_PY
```

(No `--synthetic`: the sherpa ramp streams real LibriSpeech audio — an
8-utterance seeded manifest — so the decoder does real work.
`--utterance-seconds` is required by the CLI but ignored on the
real-manifest path. `--max 8` = the box's logical core count; the
single-stream file-mode RTF of ~3.75 predicted saturation below 8.)

### 3. Stabilizer study (`run_stabilizer_study`)

The 3×3 grid (`min_partials` × `min_stable_ms`, n=25, pace 1.0) was split
into three invocations along the `min_partials` axis purely for
run-management (each grid point is an independent server boot on an
otherwise-identical patched config, so splitting changes nothing about the
measurement; the runner writes its JSON only after its whole sub-grid
completes). All three commands, sherpa venv, run back-to-back:

```
$SHERPA_PY -m benchmarks.run_stabilizer_study \
    --config configs/sherpa.yaml --model sherpa-zipformer-en --n 25 --seed 42 \
    --grid "min_partials=1;min_stable_ms=0,240,480" \
    --api native --split test-clean --port 8105 --pace 1.0 \
    --python $SHERPA_PY
```

```
$SHERPA_PY -m benchmarks.run_stabilizer_study \
    --config configs/sherpa.yaml --model sherpa-zipformer-en --n 25 --seed 42 \
    --grid "min_partials=2;min_stable_ms=0,240,480" \
    --api native --split test-clean --port 8105 --pace 1.0 \
    --python $SHERPA_PY
```

```
$SHERPA_PY -m benchmarks.run_stabilizer_study \
    --config configs/sherpa.yaml --model sherpa-zipformer-en --n 25 --seed 42 \
    --grid "min_partials=3;min_stable_ms=0,240,480" \
    --api native --split test-clean --port 8105 --pace 1.0 \
    --python $SHERPA_PY
```

Realtime-API point (M-8 `delta_duplication_ratio`; one grid point at the
default stabilizer settings `min_partials=2` / `min_stable_ms=400`, i.e.
what `configs/sherpa.yaml` ships):

```
$SHERPA_PY -m benchmarks.run_stabilizer_study \
    --config configs/sherpa.yaml --model sherpa-zipformer-en --n 25 --seed 42 \
    --grid "min_partials=2;min_stable_ms=400" \
    --api realtime --split test-clean --port 8106 --pace 1.0 \
    --python $SHERPA_PY
```

### 4. Endpointing (`run_endpointing`)

Reused from Plan 4 Task 7's corrected post-review run — same machine, pace
1.0, `arm_a.audio_dropped_delta == 0.0` verified in the JSON; git SHA
`93939ce`'s benchmark code is identical to `b1096ee`'s:

```
$SHERPA_PY -m benchmarks.run_endpointing \
    --config configs/sherpa.yaml --model sherpa-zipformer-en \
    --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
    --split test-clean --n 10 --seed 42 --pace 1.0 \
    --lead-s 1.0 --trail-s 1.0 \
    --python $SHERPA_PY
```

## SLO choices

- **`--slo-final-ms 1200` at `--slo-pct 95`** for both load ramps: the
  final transcript should land within ~1.2 s of end-of-speech for the
  server to feel interactive (dictation/agent-turn-taking use cases); 1.2 s
  is the plan's suggested value and, conveniently, sherpa's own default
  endpoint rule2 threshold — a latency budget of the same order as the
  trailing-silence humans already tolerate. Concurrency-1 sherpa
  measurements (client-final p95 ≈ 87 ms) sit ~14× under this SLO, so the
  ramp measures genuine saturation, not an SLO chosen to fail early.
- **Zero-tolerance error gate**: a rung fails on ANY client error, dropped
  chunk, rejection, or server error regardless of latency — for a
  transcription service, shed audio is data loss, not degraded service.
- **30 s windows** (`--window-seconds 30`): long enough for every rung to
  collect multiple utterances per worker and for queue effects to build,
  short enough to keep a 16-rung ramp under 10 minutes. A rung may overrun
  its window by up to one utterance duration (deadline checked between
  utterances).

## Threats to validity

- **Single machine, single run.** Every number comes from one pass on one
  2014-era 4-core laptop CPU. No run-to-run variance was measured; treat
  small deltas (a few ms, a fraction of a WER point) as noise.
- **Thermal throttling / laptop wall-clock gaps.** Runs are long (10-30 min
  sustained decode); later rungs/utterances run hotter than earlier ones,
  and rung N+1 always starts warmer than rung N. Additionally, the sherpa
  load ramp's observed wall time (33.5 min) exceeds what its six 30 s rungs
  imply (~7 min) — consistent with the machine sleeping mid-run between
  rungs. All latency/deadline measurements use `time.monotonic()` (which
  does not advance during macOS sleep), so in-rung measurements are not
  distorted by a sleep gap, but wall times reported here are not decode
  time.
- **`cpu_pct_peak` is 0.0 in every load rung — a known measurement defect,
  not a real reading.** `benchmarks/sampling.py` re-instantiates
  `psutil.Process` objects on every sampling tick, so each
  `cpu_percent(interval=None)` call is a "first call" (which psutil defines
  to return 0.0); the priming call primes instances that are immediately
  discarded. RSS numbers are unaffected (`memory_info()` is stateless).
  The report's bottleneck analysis therefore leans on the latency-vs-
  concurrency curves and single-stream RTF, not on CPU% samples. Fixing the
  sampler is follow-up work (this task runs benchmarks; it does not modify
  code).
- **Subset size.** n=100 (13.3 min) is a small slice of test-clean
  (~5.4 h); published LibriSpeech WERs are not directly comparable. The
  stabilizer grid (n=25) and endpointing (n=10) subsets rank
  configurations; they do not estimate population metrics.
- **Real-time pacing bounds what concurrency-1 latency can show.** At
  `--pace 1.0` the decoder is never saturated during the accuracy WS pass —
  those latency numbers describe an unloaded server (the load ramp is the
  saturation measurement).
- **macOS 12 pinned wheels.** `sherpa-onnx==1.10.46`, `onnxruntime==1.19.2`
  and `torch==2.2.2` are the newest versions that load on this OS; newer
  releases may perform differently.
- **FunASR is cross-lingual here.** `paraformer-zh-streaming` on English
  audio produces near-garbage transcripts by design; its WER row measures
  the pipeline, not the model.
- **Endpointing numbers are reused**, not re-run: Task 7's corrected
  pace-1.0, zero-drop-asserted n=10 run on this same machine.
