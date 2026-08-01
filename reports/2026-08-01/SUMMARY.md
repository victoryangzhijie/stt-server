# STT benchmark summary — 2026-08-01 (qwen3asr / vLLM on A10)

Backend `qwen3-asr-0.6b` (vLLM 0.14.0, torch 2.9.1+cu128). Dataset `bench-audio-v1-en` (LibriSpeech test-clean). Bare-metal, driver 580 / CUDA 12.8. Full env in `env.txt`.


Latency definition differs by interface — the key to reading the tables:

- **WS** (`run_load`): `client_final_ms` = finalization *tail* (end-of-stream → final), so streaming hides audio length. SLO gate p95(final) ≤ 2000 ms.

- **HTTP** (`bench_http_stt`): end-to-end per-request wall-clock (POST → full response) = the *full* decode, so the serial decode lock shows up directly.


## WS load ladder — `STT__LIMITS__MAX_SESSIONS=16` (real compute boundary)

Ramp c=1..16, 30s windows, p95(final) ≤ 2000ms gate. **max_passing_concurrency = 13.**

| c | pass | p50(ms) | p95(ms) | p99(ms) | n | errors | gpuU% |
|---|------|---------|---------|---------|---|--------|-------|
| 1 | PASS | 42 | 42 | 42 | 3 | 0 | 46.0 |
| 2 | PASS | 29 | 52 | 52 | 8 | 0 | 40.0 |
| 3 | PASS | 30 | 51 | 51 | 13 | 0 | 65.0 |
| 4 | PASS | 30 | 117 | 117 | 18 | 0 | 93.0 |
| 5 | PASS | 34 | 82 | 114 | 23 | 0 | 84.0 |
| 6 | PASS | 44 | 107 | 133 | 30 | 0 | 87.0 |
| 7 | PASS | 37 | 146 | 176 | 37 | 0 | 87.0 |
| 8 | PASS | 40 | 151 | 217 | 43 | 0 | 86.0 |
| 9 | PASS | 52 | 166 | 324 | 48 | 0 | 92.0 |
| 10 | PASS | 70 | 222 | 260 | 52 | 0 | 92.0 |
| 11 | PASS | 122 | 287 | 335 | 55 | 0 | 93.0 |
| 12 | PASS | 534 | 988 | 1564 | 54 | 0 | 93.0 |
| 13 | PASS | 982 | 1980 | 3890 | 51 | 0 | 91.0  ← edge|
| 14 | FAIL | 1314 | 3714 | 6846 | 54 | 0 | 90.0  ← SLO|

## HTTP transcription matrix (cap=8, default config)

| conc | p50(ms) | p95(ms) | p99(ms) | errors/n | gpuU% | throughput |
|------|---------|---------|---------|----------|-------|------------|
| 1 | 1414 | 1991 | 2277 | 0/50 | 88.0 | 0.69 r/s |
| 2 | 2727 | 4016 | 4196 | 0/50 | 85.0 | 0.71 r/s |
| 4 | 5512 | 7869 | 8354 | 0/50 | 90.0 | 0.70 r/s |
| 8 | 10428 | 16233 | 16721 | 0/50 | 93.0 | 0.70 r/s |

## Accuracy (WER, test-clean, n=100, seed=42, pace=1.0)

WS WER = **3.37%**, file WER = **3.37%** (identical — shared pipeline), 0 errors. WS latency: first_partial p50 242ms, server_final p50 41ms.


## Findings

1. **Real compute boundary = c=13** (WS, cap raised to 16). c=1..13 pass; **c=14 fails on pure latency** (p95=3714ms > 2000ms SLO, **zero errors/drops/rejections**). 
Latency is flat (p95 ≤ ~290ms) through c=11, then hockey-sticks: c=12 p95=988ms, c=13 p95=1980ms (at the edge), c=14 p95=3714ms. GPU util saturates ~93% from c=4 — the A10 GPU is the bottleneck. Comfortable headroom ≈ c=10–11; hard SLO limit c=13.

2. The earlier cap=8 run (max_passing_concurrency=7) was **capacity-limited, not compute-limited**: c=8 failed on 8 `max_sessions` rejections with p95 still 142ms. Raising the cap moved the failure from the session limit to true GPU saturation.

3. **HTTP matrix** (file mode, cap=8): 0 errors at c=1/2/4/8. Latency scales ~linearly with concurrency (p50 1414→10428ms) while **throughput is flat at ~0.70 r/s** — serial decode is the bottleneck; concurrency only adds queueing. (WS hides this because its metric is the finalization tail, not full decode.)

4. **WER 3.37%** — Qwen3-ASR-0.6B is accurate on test-clean.

5. **GPU**: ~16.6 GB resident (KV cache, gpu_memory_utilization=0.8); util scales to ~93% and saturates; VRAM returns cleanly to 0 between runs (no EngineCore orphans).


_All result JSONs committed verbatim. Source: `benchmarks/results/{load,accuracy}-*.json` (force-added past gitignore) + `reports/2026-08-01/`._
