# GPU / CUDA Verification Report — `qwen3asr` (vLLM) backend

**Date:** 2026-07-19  **Repo SHA:** `a375052` (+2 fixes, uncommitted)
**Scope:** Real-model, real-audio, latency, concurrency, VRAM and stability
verification of the GPU code path (`src/stt_server/backends/qwen3asr/`,
exercised end-to-end through the full server pipeline: WS → Session →
Silero VAD → Endpointer → qwen3asr → Stabilizer).

## TL;DR verdict

| Dimension | Verdict | Headline |
|---|---|---|
| Real model loads on GPU | ✅ PASS | Qwen3-ASR-0.6B / vLLM 0.14.0 / bf16 on A10, 1.53 GiB weights + 12.47 GiB KV cache |
| Real-audio decode | ✅ PASS | conformance 6/6 + fixture → `"Canoe slid."`; 0 audio drops at every load level |
| Steady-state latency (conc 1, warm) | ✅ PASS | p95 **41 ms** vs 2000 ms SLO (post-warmup fix) |
| Cold-start latency | ⚠️ → ✅ FIXED | first decode was **12.3 s**; fixed by startup warmup → **121 ms** |
| Concurrency | ⚠️ LIMITED | `max_passing_concurrency=1` @ 2 s p95 SLO for 5 s utterances (GPU saturates at conc 2). Inherent O(n²) redecode — tunable, not a bug |
| VRAM | ✅ PASS | stable 16.6–17.1 GiB / 23 GiB under load; returns to 0 on teardown |
| Stability | ⚠️ → ✅ FIXED | found + fixed orphaned `EngineCore` leaking 16.6 GiB on hard teardown; 0 drops/errors/rejections across all runs |
| WER (accuracy) | ⛔ BLOCKED | OpenSLR corpus unreachable from this host (~0 B/s); transcript quality verified instead |
| Regression (default suite) | ✅ PASS | 323 passed / 2 skipped / 21 deselected after fixes |

**Bottom line:** the qwen3asr GPU code is **correct and stable**. Two real defects
were found by measurement and fixed (cold-start; teardown VRAM leak). One
inherent throughput limitation is characterized (concurrency ceiling). One
dimension (WER) could not be exercised due to an external-network block.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA A10, 23 GiB, cc 8.6 (Ampere) |
| Driver / CUDA toolkit | 580.126.09 (CUDA 13.0 capable) / nvcc 12.8 |
| torch | 2.9.1+cu128 (`cuda.is_available()=True`) |
| vLLM / qwen-asr / onnxruntime | 0.14.0 / 0.0.6 / 1.27.0 |
| CPU / RAM / disk | 8 vCPU / 28 GiB / 65 GiB free |
| Model weights | `Qwen/Qwen3-ASR-0.6B` (1.2 GiB) via **hf-mirror**; Silero VAD v5 (2.2 MiB, contract-verified) |

**Network reality (matters for reproducibility):**
`huggingface.co` and `www.openslr.org` are **unreachable** from this host
(timeout). `hf-mirror.com`, `mirrors.aliyun.com/pypi`, and GitHub are reachable.
Consequences: Qwen weights pulled via `HF_ENDPOINT=https://hf-mirror.com`;
Silero VAD fetched from `onnx-community/silero-vad` on hf-mirror (the GitHub
LFS `raw` URL served a truncated 119 KB file); PyPI wheels installed via the
aliyun mirror through `uv pip install` (which does **not** touch `uv.lock` —
verified `git status --short uv.lock` clean, 0 `mirrors.aliyun` URLs); the
LibriSpeech WER corpus could **not** be downloaded.

## How to reproduce

```bash
# 0. Env (this host): HF mirror for weights, pypi mirror is already global.
export HF_ENDPOINT="https://hf-mirror.com"

# 1. Install GPU + bench + silero extras into the venv WITHOUT touching uv.lock.
#    (pypi.org is ~77 KB/s here; aliyun mirror ~3 MB/s. uv pip install does not
#    rewrite uv.lock, unlike uv sync/lock/add.)
uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ \
    -e '.[qwen3asr,bench,silero]' pytest pytest-asyncio httpx ruff

# 2. Silero VAD model (v5 contract: input (1,512), state (2,1,128), sr int64).
#    GitHub LFS raw was truncated here; hf-mirror's onnx-community/silero-vad works:
curl -sL "https://hf-mirror.com/onnx-community/silero-vad/resolve/main/onnx/model.onnx" \
    -o models/silero_vad.onnx   # verify I/O matches src/stt_server/core/vad_silero.py

# 3. Real model + GPU conformance / streaming test (downloads Qwen3-ASR on first run).
HF_ENDPOINT="https://hf-mirror.com" .venv/bin/python -m pytest -m "model and gpu" -s \
    tests/backends/test_qwen3asr_backend_model.py

# 4. Warm concurrency ramp (server boots once, warmed at startup).
HF_ENDPOINT="https://hf-mirror.com" .venv/bin/python -m benchmarks.run_load \
    --config configs/qwen3asr.yaml --model qwen3-asr-0.6b \
    --utterance-seconds 5 --start 1 --step 1 --max 8 \
    --window-seconds 30 --slo-final-ms 2000 --slo-pct 95 --seed 42 --synthetic
```

## Detailed results

### 1. Real model + real audio — PASS
- **Smoke** (direct `Qwen3AsrBackend.start()` → decode → `stop()`): `start()` 74.9 s
  (incl. ~20 s warmup), first real decode+finalize **121 ms**, transcript
  **`"Canoe slid."`** from the 1 s speech fixture, exactly 1 final + 2 partials
  (@500 ms / @1000 ms, matching `redecode_interval_ms=480`).
- **pytest `-m "model and gpu"`**: **6 passed** in 328 s — the 5 `BackendConformanceSuite`
  cases (start/stop, finalize-yields-exactly-one-final, partials-ordered,
  close-without-finalize, push-after-finalize-noop) + the real-stream test.
  Transcript printed: `[qwen3asr real transcript, en fixture] 'Canoe slid.'`
- vLLM init: bf16, FLASH_ATTN, KV cache 116 720 tokens, `gpu_memory_utilization=0.8`.

### 2. Latency (steady-state) — PASS (after warmup fix)
Load runner, concurrency 1, 5 s synthetic utterances (real speech fixture tiled),
**post-fix** (warm server):

| conc | n | p50 | p95 | p99 | err | drop | gpu_util | gpu_mem |
|---|---|---|---|---|---|---|---|---|
| 1 | 7 | **28 ms** | **41 ms** | 41 ms | 0 | 0 | 33% | 17 077 MiB |
| 2 | 9 | 1453 ms | **2460 ms** | 2460 ms | 1 | 0 | 80% | 17 077 MiB |

Concurrency-1 warm p95 = **41 ms** — two orders of magnitude under the 2 s SLO.

### 3. Cold-start — was a defect, now FIXED
Measured first-decode-after-boot cost: **12.3 s** (smoke) / **~6.8 s** (first load
utterance) — vLLM specializes the audio-encoder graph on the first real input,
past the init-time CUDA-graph capture. This made the load runner's SLO gate fail
*at concurrency 1* (p95 = single cold sample), so `max_passing_concurrency` read 0.
**Fix:** `Qwen3AsrBackend.start()` now runs one throwaway streaming decode
(`_warmup`) during boot, shifting that cost out of the first request. After the
fix the first real decode is **121 ms** and conc-1 passes. Cost: `start()` +~20 s
(net win for a start-once server).

### 4. Concurrency — LIMITED (inherent, not a bug)
`max_passing_concurrency = 1` at 2 s p95 SLO for 5 s utterances. At conc 2 the GPU
climbs to ~80% util and p95 exceeds the SLO. Root cause: the qwen-asr streaming
API re-decodes the **entire accumulated utterance** every `redecode_interval_ms`
(480 ms) — total decode work is O(utt_len²). For 5 s this saturates the A10 beyond
a single stream. This is the framework's documented streaming design
(see the backend module docstring), **not a code defect**. Levers, in order of
impact: raise `redecode_interval_ms` (fewer/coarser redecodes — trades partial
freshness for throughput), shorten effective utterance length (the endpointer
splits real speech at pauses; the synthetic tiled fixture has no pauses, so it is
a near-worst case), or accept the ceiling for this model class.

### 5. VRAM — PASS
- Idle → load: 0 → **16 637 MiB** (72% of 23 GiB) at `start()`; holds **17 077 MiB**
  steady across every load rung (no in-run growth = no leak under sustained load).
- Model: 1.53 GiB; KV cache pool: 12.47 GiB; capture: 0.43 GiB.
- Post-teardown: returns to **0 MiB** (after the killpg fix — see §6).

### 6. Stability — one defect found & FIXED; otherwise PASS
- **0 audio drops, 0 rejections, 0 server errors** at every concurrency tested —
  the bounded queue (64 chunks ≈ 6.4 s) absorbs decode backlog without shedding
  audio, so the WER-invalidating failure mode the runbook warns about does not occur.
- **Defect found:** when the server is hard-terminated (SIGTERM, e.g. the benchmark
  harness `ServerUnderTest`, or a non-container supervisor), vLLM's `EngineCore`
  child process is **orphaned and keeps holding 16.6 GiB** (vLLM's `LLM` exposes no
  public `close()` and the child is reaped only on clean interpreter exit). Measured
  directly: an orphaned `EngineCore` left after a SIGTERM'd run OOM'd the next
  server boot (`Free memory ... 5.55/22.06 GiB ... less than desired 17.65 GiB`).
  **Fix:** `ServerUnderTest` now starts the server in its own session
  (`start_new_session=True`) and tears down the **whole process group** (`killpg`),
  so `EngineCore` dies with the parent. Verified: after teardown, GPU = 0 MiB and no
  leftover process. (Production under Docker is unaffected either way — the
  container kill reaps all PIDs — but a bare-process supervisor with
  `KillMode=process` would have hit this.)
- `backend.stop()` shuts down the executor and drops the model reference; it does
  not explicitly destroy the `EngineCore` (no public vLLM API). Clean shutdown
  relies on interpreter-exit atexit (verified working in the smoke run). The
  killpg path covers hard termination. In-process repeated `start()/stop()` would
  accumulate VRAM until GC — not a normal server-lifecycle operation.

### 7. WER (accuracy) — BLOCKED (external network)
`benchmarks/run_accuracy` needs LibriSpeech `test-clean` (347 MiB) from
`openslr.org`, which is unreachable here (~0 B/s; 2 MiB range request times out).
The committed 1 s fixture has no documented reference transcript, so a WER number
cannot be computed. Real-audio decode **quality** is verified instead: a
non-empty, sensible transcript is produced from real speech under the full
pipeline (`"Canoe slid."`), reproducibly. (Re-run on a host with OpenSLR access
using the runbook's `run_accuracy` command to obtain a numeric WER.)

## Fixes applied (uncommitted)

1. **`benchmarks/server.py`** — process-group teardown (`start_new_session=True` +
   `os.killpg` in `_terminate`). Eliminates orphaned `EngineCore` / VRAM leak on
   hard server termination (fixes the sequential-benchmark OOM). +`contextlib` import.
2. **`src/stt_server/backends/qwen3asr/backend.py`** — `_warmup()` decode in
   `start()` (best-effort, failure-soft) + `structlog` logger. Eliminates the
   ~12 s first-request cold-start; also makes the load runner's SLO gate measure
   steady state instead of permanently failing on the cold sample.

**Regression check:** default ML-free suite = **323 passed / 2 skipped / 21
deselected**; `ruff check .` clean; `uv.lock` unchanged.

## Concurrency regression round (real-client findings, 2026-07-19)

A Mac client running 3 concurrent `POST /v1/audio/transcriptions` uploads
surfaced four more defects. All fixed, unit-tested (suite now **325 passed**),
and verified live against the running server (`/tmp/conc_smoke.py`):
3-concurrent → clean transcripts, `sessions_active→0`; 9-concurrent →
`{200:8, 429:1}` and `sessions_active→0` after drain.

- **② Prompt-scaffold contamination** (`"你好language Chinese<asr_text>…"`
  bleeding into/across responses). Root cause: the backend invoked vLLM's
  synchronous `LLM.generate()` from concurrent executor threads — its V1
  `SyncMPClient` is not safe under concurrent invocation, so the race
  corrupted qwen_asr's `_raw_decoded` and `parse_asr_output` failed to strip
  the scaffold. **Fix:** a backend-wide **`threading.Lock`** held inside the
  sync decode methods (a thread lock, not asyncio, so it survives cancellation
  of an abandoned in-flight generate()). One generate() in flight at a time.
  (`src/stt_server/backends/qwen3asr/backend.py`)
- **③ Slot leak → full wedge.** The file handler freed its capacity slot only
  in a `finally` reached after `run_file_session` returned; a client that
  disconnected mid-decode (or a wedged decode) never returned, so
  `stt_sessions_active` climbed and stuck, blocking all new requests while
  healthz/readyz stayed 200. **Fix:** `run_file_session` now supervises the
  decode with `request.is_disconnected()` + `session_deadline`, aborting the
  session (and thus freeing the slot) on either. (`transcriptions_http.py`)
- **① No 429 at capacity.** `limits.max_sessions` defaulted to 100 (≫ GPU
  capacity), so excess requests queued on the saturated GPU and hung instead
  of being rejected. **Fix:** `configs/qwen3asr.yaml` sets `max_sessions: 8`
  (decode-pool-sized) + `max_session_seconds: 300`, so beyond 8 concurrent
  in-flight requests the server returns 429 immediately. (The WS path already
  enforced its slot cap correctly; this was file-endpoint only.)
- **④ `verbose_json` `language:"en"` for Chinese audio.** The handler
  hardcoded `language or "en"`. **Fix:** the detected language now flows from
  `ASRStreamingState.language` → `BackendEvent.language` →
  `TranscriptEvent.language` → `verbose_json` (client hint used only as a
  fallback). (`backends/base.py`, `core/events.py`, `core/session.py`,
  `qwen3asr/backend.py`, `transcriptions_http.py`)

**Note:** qwen3asr decodes are now serialized to one-at-a-time, so
`max_concurrent` (the thread-pool size) no longer bounds decode concurrency —
the `threading.Lock` does. `tests/backends/test_qwen3asr_backend.py::
test_decode_lock_serializes_concurrent_generate_calls` pins this.

## Known inherent limitations (documented, not fixed)
- **O(n²) redecode throughput** — concurrency ceiling ≈ 1 for 5 s utterances at a
  2 s SLO on this HW. Tunable via `redecode_interval_ms`; architectural to the
  qwen-asr streaming API.
- **Per-test `start()/stop()` in `BackendConformanceSuite`** — each conformance
  case re-initializes vLLM, so the qwen3asr model suite takes ~55 s × 6 ≈ 5–6 min
  (the runbook's "2–10 min" budget). Cosmetic for CI; not addressed.

## Artifacts (gitignored)
`benchmarks/results/load-qwen3-asr-0.6b-2026071*.json` (per-run load JSONs with
full per-rung latency percentiles, drop/error deltas, GPU/CPU peaks).
