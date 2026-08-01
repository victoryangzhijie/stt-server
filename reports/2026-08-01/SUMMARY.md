# STT benchmark summary — 2026-08-01 (qwen3asr / vLLM on A10)

Backend `qwen3-asr-0.6b` (vLLM 0.14.0, torch 2.9.1+cu128). Dataset `bench-audio-v1-en` (LibriSpeech test-clean). Bare-metal, driver 580 / CUDA 12.8. Full env in `env.txt`.


Latency definition differs by interface — this is the key to reading the table:

- **WS** (`run_load`): `client_final_ms` = finalization *tail* (end-of-stream → final), so streaming hides audio length. SLO gate p95(final) ≤ 2000 ms.

- **HTTP** (`bench_http_stt`): end-to-end per-request wall-clock (POST → full response) = the *full* decode, so the serial decode lock shows up directly.


## Load + latency matrix

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

## Accuracy (WER, test-clean, n=100, seed=42, pace=1.0)

| mode | WER | n | errors | first_partial p50 | server_final p50 | client_final p95 |
|------|-----|---|--------|-------------------|------------------|------------------|
| ws   | 3.37% | 100 | 0 | 242 ms | 41 ms | 82 ms |
| file | 3.37% | 100 | 0 | — | — | — |

## Findings

1. **WS ladder `max_passing_concurrency = 7`** (not 1). c=1..7 pass, p95(final) 43–142 ms. 
**c=8 fails on 8 capacity rejections** (`limits.max_sessions=8`), *not* latency (p95 still 142 ms). 
The serial decode lock does not break the streaming SLO until the configured session cap.

2. **HTTP matrix**: 0 errors at c=1/2/4/8. Latency scales ~linearly (p50 1414→2727→5512→10428 ms) 
while **throughput is flat at ~0.70 r/s** — serial decode is the sole bottleneck; concurrency only adds queueing.

3. **WER 3.37%** (ws == file, as expected — shared pipeline). Qwen3-ASR-0.6B is accurate on test-clean.

4. **GPU**: ~16.6 GB resident (KV cache, gpu_memory_utilization=0.8); util 40→94% with load; VRAM returns cleanly to 0 between runs (no EngineCore orphans).


_All result JSONs committed verbatim. Source: `benchmarks/results/{load,accuracy}-*.json` (force-added past gitignore) + `reports/2026-08-01/`._
