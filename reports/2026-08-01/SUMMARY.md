# STT benchmark summary — 2026-08-01 (qwen3asr / vLLM on A10)

Backend: `qwen3-asr-0.6b` (vLLM 0.14.0, torch 2.9.1+cu128). Dataset: `bench-audio-v1-en`. 
Bare-metal, driver 580 / CUDA 12.8. See `env.txt`.


Latency definition differs by interface (this is the key to reading the table):

- **WS** (`run_load`): `client_final_ms` = finalization *tail* latency (end-of-stream → final), 
so streaming hides the audio length; the SLO gate is p95(final) ≤ 2000 ms.

- **HTTP** (`bench_http_stt`): end-to-end per-request wall-clock (POST → full response), 
i.e. the *full* decode, so the serial decode lock shows up directly.


## Summary table

| iface | conc | p50(ms) | p95(ms) | p99(ms) | errors/n | gpuU% | note |
|-------|------|---------|---------|---------|----------|-------|------|
| WS | 1 | 40 | 43 | 43 | 0/3 | 40.0 | PASS |
| WS | 2 | 30 | 51 | 51 | 0/8 | 64.0 | PASS |
| WS | 3 | 30 | 71 | 71 | 0/13 | 65.0 | PASS |
| WS | 4 | 32 | 108 | 108 | 0/18 | 63.0 | PASS |
| WS | 5 | 39 | 74 | 90 | 0/23 | 81.0 | PASS |
| WS | 6 | 41 | 110 | 128 | 0/27 | 94.0 | PASS |
| WS | 7 | 36 | 115 | 142 | 0/30 | 85.0 | PASS |
| WS | 8 | 33 | 142 | 172 | 16/35 | 82.0 | FAIL |
| HTTP | 1 | 1414 | 1991 | 2277 | 0/50 | 88.0 | thr=0.69 r/s |
| HTTP | 2 | 2727 | 4016 | 4196 | 0/50 | 85.0 | thr=0.71 r/s |
| HTTP | 4 | 5512 | 7869 | 8354 | 0/50 | 90.0 | thr=0.70 r/s |
| HTTP | 8 | 10428 | 16233 | 16721 | 0/50 | 93.0 | thr=0.70 r/s |

## Findings

- **WS ladder**: `max_passing_concurrency = 7`. c=1..7 pass with p95(final) 43–142 ms 
(far under the 2000 ms SLO). **c=8 fails on 8 capacity rejections (`limits.max_sessions=8`), 
NOT on latency** — its p95 is still 142 ms. The serial decode lock does NOT break streaming 
SLO until the configured session cap; the backend sustains concurrency far beyond the prior 
“c=2 fails” hypothesis.

- **HTTP matrix**: 0 errors across c=1/2/4/8. Latency scales ~linearly with concurrency 
(p50 1414→2727→5512→10428 ms) while **throughput is flat at ~0.70 r/s** — the qwen3asr serial 
decode is the sole bottleneck; extra concurrency adds pure queueing, no throughput gain. 
(Contrast with WS, where streaming + tail-latency metric masks this.)

- **GPU**: ~16.6 GB resident (vLLM KV cache, gpu_memory_utilization=0.8); util scales 40→94% 
with load. VRAM returns cleanly to 0 between runs (no EngineCore orphans).


_Result JSONs are committed verbatim (not a byte edited). WER row appended below when run completes._
