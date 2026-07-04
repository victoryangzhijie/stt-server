# Benchmark report — CPU (local)

Real numbers from real runs on a single local machine (Intel i7-4770HQ,
4 cores, macOS 12.7.6). **Read [`methodology.md`](methodology.md) first**:
it holds the exact command for every number here, the environment each ran
in, the corpus seeding, and the threats to validity — including the
binding rule that every real-backend streaming WER was measured at
real-time pace with an asserted-zero `stt_audio_dropped_total` delta
(reported in the tables below as "dropped").

Corpus for accuracy/latency: LibriSpeech test-clean, n=100, seed=42 —
**800.5 s (13.34 min) of audio**. Stabilizer study: n=25 (191.8 s).
Endpointing: n=10 (70.5 s). GPU/qwen3asr numbers are pending (last
section).

## 1. Accuracy (WER, streamed vs file)

| backend | model | pace | WS WER | file WER | errors | dropped (WS delta) | file RTF (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock | mock | 0.0 | 0.9769 | 0.9769 | 0 | 0.0 | 175.57 |
| sherpa | sherpa-zipformer-en | 1.0 | **0.1006** | **0.1006** | 0 | 0.0 | 3.75 |
| funasr | paraformer-zh-streaming | 1.0 | 0.8380 | 0.8380 | 0 | 0.0 | 3.53 |

- **mock**: the WER is meaningless by construction (the mock emits a
  scripted transcript, not a transcription of the audio) — this row is the
  pipeline-mechanics baseline only, and its RTF (~176x real time in file
  mode) bounds the serving overhead with a zero-cost "decoder".
- **sherpa** is the real headline: **10.1% corpus WER** on the n=100
  subset, streamed and file-mode identical. They are identical to full
  precision because file mode routes the upload through the same Session
  pipeline as one chunk — with VAD segmentation producing the same
  segments, the decoder sees equivalent audio both ways. Treat it as one
  accuracy measurement made twice, not two independent ones.
- **funasr** is **cross-lingual and not comparable — it measures pipeline
  mechanics only**: `paraformer-zh-streaming` is a Mandarin model
  transcribing English test-clean, so ~84% WER is expected garbage. The row
  demonstrates the harness + a second real decoder (torch/FunASR) complete
  a full n=100 pass with zero errors and zero dropped chunks.
- File-mode RTF ~3.75 (sherpa) / ~3.53 (funasr): both real decoders run
  ~3.5-3.8x faster than real time single-stream on this 4-core CPU. This
  number drives the load-ramp knee below.

## 2. Latency (concurrency 1, real-time-paced WS pass)

Server-reported first-partial and final processing latencies, plus the
client-observed final latency (includes the wire round-trip), all in ms:

| backend | first partial p50/p95/p99 | server final p50/p95/p99 | client final p50/p95/p99 |
| --- | --- | --- | --- |
| mock | 0.4 / 1.3 / 1.5 | 0.2 / 0.3 / 0.4 | 1.6 / 2.0 / 2.2 |
| sherpa | 570.4 / 872.5 / 967.7 | 0.7 / 12.3 / 75.4 | 4.5 / 86.8 / 93.3 |
| funasr | 1073.0 / 1265.7 / 1268.6 | 174.9 / 248.7 / 313.1 | 180.8 / 346.7 / 582.8 |

Reading guide: *first partial* is time from first speech-bearing audio to
the first partial transcript — for real backends it is dominated by how
much audio the decoder needs before emitting anything plus the stabilizer's
gating (`min_partials=2`, `min_stable_ms=400` in both real configs), not by
compute. *Server final* is the server's processing time to produce the
final after end-of-utterance. sherpa's sub-millisecond p50 final shows the
streaming decoder has already consumed the audio by utterance end — the
final is nearly free; funasr's ~175 ms p50 reflects its chunked
(600 ms-window) decode finishing after end-of-speech. Client-final tracks
server-final closely at concurrency 1 (the wire adds single-digit ms).

## 3. Load (concurrency ramp, 30 s windows, SLO p95 client-final <= 1200 ms, zero-error gate)

`cpu_pct_peak` is omitted from these tables: the sampler currently returns
0.0 for all rungs (known defect — per-tick `psutil.Process`
re-instantiation defeats `cpu_percent`'s two-call protocol; see
methodology.md). RSS is valid.

### mock (synthetic 10 s utterances, `STT__LIMITS__MAX_SESSIONS=256`)

**max_passing_concurrency = 128** (never breached; ramp exhausted at
`--max 128`).

| concurrency | p50 (ms) | p95 (ms) | p99 (ms) | utterances | errors | dropped chunks | rejections | server errors | RSS peak (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 2.9 | 10.7 | 10.8 | 32 | 0 | 0 | 0 | 0 | 52 |
| 16 | 3.2 | 23.3 | 25.3 | 64 | 0 | 0 | 0 | 0 | 56 |
| 24 | 3.9 | 23.3 | 27.3 | 96 | 0 | 0 | 0 | 0 | 63 |
| 32 | 16.1 | 56.5 | 62.4 | 114 | 0 | 0 | 0 | 0 | 75 |
| 40 | 8.0 | 111.2 | 134.1 | 126 | 0 | 0 | 0 | 0 | 89 |
| 48 | 4.9 | 34.7 | 36.8 | 183 | 0 | 0 | 0 | 0 | 89 |
| 56 | 4.7 | 41.3 | 44.8 | 201 | 0 | 0 | 0 | 0 | 110 |
| 64 | 18.5 | 58.0 | 59.4 | 207 | 0 | 0 | 0 | 0 | 110 |
| 72 | 15.3 | 68.0 | 70.4 | 230 | 0 | 0 | 0 | 0 | 116 |
| 80 | 13.9 | 71.7 | 76.4 | 241 | 0 | 0 | 0 | 0 | 116 |
| 88 | 13.7 | 69.9 | 79.1 | 274 | 0 | 0 | 0 | 0 | 107 |
| 96 | 15.3 | 78.6 | 79.8 | 298 | 0 | 0 | 0 | 0 | 124 |
| 104 | 18.6 | 69.3 | 71.7 | 313 | 0 | 0 | 0 | 0 | 124 |
| 112 | 16.9 | 84.9 | 92.2 | 341 | 0 | 0 | 0 | 0 | 114 |
| 120 | 60.1 | 108.3 | 115.4 | 364 | 0 | 0 | 0 | 0 | 116 |
| 128 | 51.2 | 114.7 | 134.9 | 384 | 0 | 0 | 0 | 0 | 118 |

With a zero-cost decoder the serving layer itself (asyncio event loop, WS
framing, VAD, session bookkeeping) sustains 128 concurrent real-time
streams on this laptop at p95 ~115 ms — an order of magnitude under the
SLO — with ~118 MB RSS and zero shed audio. The p95 creep from ~11 ms (c=8)
to ~115 ms (c=128) is event-loop scheduling latency growing roughly
linearly with connection count.

### sherpa (real LibriSpeech audio, 8-utterance seeded pool)

**max_passing_concurrency = 5**; the ramp stopped at the first failing
rung, c=6 (p95 2174.8 ms > 1200 ms SLO — a latency breach, still zero
errors/drops).

| concurrency | p50 (ms) | p95 (ms) | p99 (ms) | utterances | errors | dropped chunks | rejections | server errors | RSS peak (MB) | passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.8 | 76.2 | 76.2 | 3 | 0 | 0 | 0 | 0 | 645 | yes |
| 2 | 8.2 | 108.8 | 108.8 | 8 | 0 | 0 | 0 | 0 | 665 | yes |
| 3 | 17.0 | 291.1 | 291.1 | 13 | 0 | 0 | 0 | 0 | 686 | yes |
| 4 | 88.6 | 620.9 | 620.9 | 17 | 0 | 0 | 0 | 0 | 702 | yes |
| 5 | 286.0 | 1184.6 | 1316.9 | 21 | 0 | 0 | 0 | 0 | 726 | yes |
| 6 | 1024.9 | 2174.8 | 2343.0 | 22 | 0 | 0 | 0 | 0 | 749 | **no** |

Small-n caveat: 30 s windows at low concurrency yield few utterances per
rung (3-22), so per-rung percentiles are coarse; the shape of the curve —
supra-linear latency growth from c=3 onward, roughly doubling per added
stream past c=4 — is the robust signal, not any single cell.

## 4. Bottleneck analysis: where sherpa saturates on this CPU, and why

TODO-BOTTLENECK

## 5. Stabilizer study (sherpa, 3x3 grid, n=25, pace 1.0)

TODO-STABILIZER

## 6. Endpointing (server VAD vs sherpa native endpoint rules, n=10, padded +-1.0 s silence)

Reused from the corrected Task 7 run (pace 1.0, Arm A
`audio_dropped_delta` = 0.0 asserted). Both arms consume identical audio:
each utterance padded with 1.0 s true leading and trailing silence, giving
an arm-independent ground-truth speech end.

| | Arm A — server pipeline (silero VAD + Endpointer) | Arm B — sherpa native endpointing (direct recognizer, default rules) |
| --- | --- | --- |
| WER | 0.0765 | **0.0459** |
| segmentation (under / correct / over) | 0 / 8 / 2 | 9 / 1 / 0 |
| latency | final processing p50 0.79 ms, p95 75.6 ms | post-speech-end endpoint fires: n=0 (no valid sample) |
| premature (mid-utterance) fires | n/a | 1 |
| audio dropped | 0.0 (asserted) | n/a (no server) |

Comparison (from the result JSON's `comparison` block): WER delta
(B minus A) = -0.0306; segmentation tallies as above.

Interpretation:

- **Arm B under-segments 9/10 utterances at 1.0 s of trailing silence** —
  sherpa's default endpoint rules need >=1.2 s (rule2) or >=2.4 s (rule1)
  of trailing silence, so with this padding they simply never fire; its
  endpoint-latency population is empty. The one fire that did occur was
  *premature* (mid-utterance, on an internal pause — 1160 ms before the
  true speech end), which is exactly the failure mode real deployments
  must debounce against. A supplementary Task 7 run with 3.0 s trailing
  silence (documented in `.superpowers/sdd/p4-task-7-report.md`) does
  fire: mean endpoint latency 1374 ms, p50 1535 ms — a bit past the 1.2 s
  rule2 threshold, as expected.
- **Arm A (the server's own VAD+Endpointer, `min_silence_ms=500`)
  segments correctly 8/10 with 2 over-segmentations and never
  under-segments** — its 500 ms silence threshold fires comfortably inside
  1.0 s of padding. Its WER is ~3 points worse than Arm B's on this n=10
  subset, at least partly because over-segmentation splits two utterances
  into multiple decode segments.
- Net: the server pipeline endpoints far more promptly (sub-second
  silence threshold vs 1.2-2.4 s), at some accuracy cost on this tiny
  subset; sherpa's native rules are more accurate when given enough
  silence but are unusable at sub-1.2 s trailing-silence budgets without
  retuning `rule2_min_trailing_silence`.

## 7. GPU results (pending)

No CUDA device exists on this machine; everything qwen3asr/GPU is deferred
to a CUDA box via **`benchmarks/cuda_runbook.md`** and
`scripts/run_gpu_suite.sh`. This section will be filled from exactly these
artifacts (all under gitignored `benchmarks/results/` on the GPU box):

1. `pytest` output of `tests/backends/test_qwen3asr_backend_model.py`
   (currently self-skipping without `nvidia-smi`) — phase 2 of the suite.
2. Docker image builds (CPU + GPU images) — phases 3-4.
3. `accuracy-qwen3asr-test-clean-<timestamp>.json` — accuracy + latency vs
   the same n=100/seed=42 manifest, `--pace 1.0`, zero-drops asserted
   (phase 5). Fills a qwen3asr row in the tables of sections 1-2.
4. `load-qwen3asr-<timestamp>.json` — concurrency ramp with GPU
   utilization/memory sampled per rung via pynvml (phase 6). Adds the GPU
   column section 3's tables omit, and a GPU bottleneck-analysis
   counterpart to section 4.

## 8. Limitations

Everything here is one pass, on one 2014-era 4-core laptop, on a 100-
utterance (13.3 min) slice of test-clean; no variance estimates, no other
hardware, thermal state uncontrolled, and the load sampler's CPU% channel
is currently broken (methodology.md details all of these, plus the
laptop-sleep wall-clock caveat). The sherpa/funasr WER numbers are
harness-comparable but not literature-comparable (subset, streaming
segmentation joined per-utterance, corpus-level jiwer scoring). FunASR's
row is cross-lingual by construction and says nothing about FunASR's
accuracy on its own language. Endpointing conclusions rest on n=10 with
synthetic silence padding. The mock load ceiling (128) is where the ramp
stopped, not where the serving layer fails — the true mechanics knee is
somewhere above. Treat every number as this machine's, not the project's.
