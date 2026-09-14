# GPU / CUDA Verification Report — `nemotron` backend (NeMo cache-aware streaming)

**Date:** 2026-09-06  **Repo SHA:** `2a02ac9` (+1 test fix, see "Code changes")
**Scope:** First-ever execution of `src/stt_server/backends/nemotron/` on real
hardware — NVIDIA A10, following `docs/nemotron_a10_runbook.md`. This closes
the "NOT yet run on real hardware" note in `docs/backends.md#nemotron` and
answers every item of the runbook's §10 UNVERIFIED checklist.

## TL;DR verdict

| Dimension | Verdict | Headline |
|---|---|---|
| Wheel claim (`EncDecRNNTBPEModelWithPrompt` in `nemo_toolkit` 3.0.0) | ✅ PASS | class present at the cited path; both `restore_from(<local .nemo>)` and HF-id `from_pretrained` load it |
| Real-audio decode | ✅ PASS | 10/10 model tests; en test-clean WER **3.74%** (n=100); zh/en TTS clips CER/WER **0.00%** |
| Auto language ID | ✅ PASS | `<xx-XX>` tag emitted and reported as `Chinese`/`English` on ≥6 s speech (not on the 1 s fixture — too short for the model to commit) |
| Latency (conc 1) | ✅ PASS | server-final p95 **42 ms**; per-step decode 31–37 ms across all 4 context sizes (RTF ≤ 0.39) |
| Concurrency ceiling | ✅ PASS (better than assumed) | **16** concurrent 5 s streams at a 2 s p95 SLO, 0 drops; lock-bound, GPU only 37% |
| fp32 vs `amp: true` | ✅ fp32 wins | amp identical quality/VRAM but **27% slower** per step on A10 |
| CPU mode | ✅ WORKS | RTF **0.37** at conc 1 on 8 vCPU — 2.7× faster than real time, not "far below" |
| VRAM | ✅ PASS | **2.9 GiB** resident fp32 (vs qwen3asr's 16.6 GiB); ~3.0–3.5 GiB peaks; no leak |
| Stability | ✅ PASS | 0 drops / 0 errors / 0 rejections in every run; warmup failure-soft path exercised and correct |
| Environment | ⚠️ one box-level fix | system-cuDNN vs torch-cuDNN sublibrary clash → `LD_LIBRARY_PATH` prefix required on this box (documented in the runbook) |

**Bottom line:** the nemotron backend is correct on first execution. One
backend-adjacent test over-assertion was found and fixed (language tag on a
1 s clip). One box-level environment conflict (cuDNN) was found and worked
around without code changes. Accuracy, latency, and concurrency all meet or
exceed the design's assumptions.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA A10, 23 GiB, cc 8.6 (Ampere), driver 580.126.09 (CUDA 13.0 capable) |
| torch / CUDA build | 2.14.0+cu130 (`cuda.is_available()=True`) |
| nemo_toolkit | 3.0.0 (PyPI wheel) |
| Python / CPU / RAM | 3.12.3 (uv venv) / 8 vCPU / 28 GiB |
| Weights | `nvidia/nemotron-3.5-asr-streaming-0.6b` (2.37 GB `.nemo`) via hf-mirror; silero VAD v5 via hf-mirror `onnx-community/silero-vad` |
| Corpus | LibriSpeech test-clean (2620 FLAC / 87 trans / 40 speakers — canonical counts verified), fetched from `k2-fsa/LibriSpeech` on hf-mirror |
| venv | `uv venv` + `uv pip install -e '.[nemotron,silero,bench]'` from aliyun mirror; `uv.lock` untouched (0 mirror URLs) |

## UNVERIFIED checklist — answers

- [x] **`EncDecRNNTBPEModelWithPrompt` in the 3.0.0 wheel** — ✅ CONFIRMED.
  `from nemo.collections.asr.models.rnnt_bpe_models_prompt import
  EncDecRNNTBPEModelWithPrompt` works.
- [x] **`from_pretrained` HF-id resolution + `restore_from` local path** —
  ✅ BOTH WORK. The HF cache was pre-seeded (revision
  `1c8deaecc64b91f034d73e08dd8b64625eb3395d`) so the model tests' default HF
  id resolved without re-downloading; the server/benchmarks used
  `restore_from(models/.../....nemo)`. Log: "Model
  EncDecRNNTBPEModelWithPrompt was successfully restored".
- [x] **fp32 vs AMP** — ✅ fp32 is the right default on A10. `amp: true`
  (bf16 autocast): same transcripts (zh/en TTS identical, no NaNs), same
  resident VRAM (~2.9 GiB), per-step **44.2 ms vs 34.8 ms fp32** (+27% —
  autocast conversion overhead dominates at this model size). Keep
  `amp: false`.
- [x] **Per-step latency at each context size** (6 s of tiled real speech,
  single stream, warm):

  | `att_context_size` | chunk | step med (max) | step RTF |
  |---|---|---|---|
  | `[56, 0]` | 80 ms | 31.4 ms (33.4) | 0.39 |
  | `[56, 3]` | 320 ms | 34.4 ms (35.0) | 0.11 |
  | `[56, 6]` | 560 ms | 34.8 ms (35.3) | 0.06 |
  | `[56, 13]` | 1120 ms | 36.5 ms (37.3) | 0.03 |

  Every setting keeps up with real time for a single stream, with the 560 ms
  default ~16× under budget. The real-time question is answered: the
  constraint on concurrency is the process-wide lock's queueing, not step
  cost.
- [x] **Concurrency ceiling** — ✅ **16** concurrent 5 s synthetic streams
  pass a 2 s p95 final-latency SLO with 0 drops; the 20-rung fails (p95
  3.25 s). GPU util plateaus at **37%** — the shared decode lock, not the
  GPU, is the limiter. `max_concurrent: 8` executor threads are sufficient;
  the shipped `limits.max_sessions: 8` sits at half the measured ceiling
  (p95 290 ms there, 6.9× margin) and was kept conservative on purpose.
- [x] **zh-CN accuracy** — ✅ CER **0.00%** on clean natural TTS (the
  validator's zh clip, 5.95 s, edge-tts `zh-CN-XiaoxiaoNeural`). This is a
  smoke signal, not a corpus benchmark (no Mandarin corpus runner exists in
  this repo); en test-clean WER 3.74% (below) is the only corpus number.
  Nothing here contradicts NVIDIA's FLEURS tiering, and nothing upgrades it.
- [x] **Automatic language ID** — ✅ WORKS per utterance. The `<xx-XX>` tag
  appears in the streaming hypothesis and the adapter reports
  `Chinese`/`English` on the FINAL, over both WS and file surfaces. One
  caveat, measured: on the 1 s speech fixture the tag never appears in any
  mode — a few seconds of speech are needed before the model commits a
  language. The model-suite assertion for the 1 s clip was adjusted to match
  (see "Code changes").
- [x] **Explicit per-request language** — ✅ WORKS. `language=zh` and
  `language=en` over the file endpoint pin the prompt per request (perfect
  transcripts both); cross-language behavior is sane (wrong-language prompt
  on clear foreign speech yields empty/gibberish output, never a crash).
- [x] **Final flush keeps the tail** — ✅ WORKS. `finalize()` with no
  trailing silence returns the complete sentence on both zh and en clips
  (`…ols across the yard.` tail check), and the pinned
  `test_final_flush_without_trailing_silence` passes.
- [x] **MPS / CPU support** — ✅ CPU **works and is far better than
  documented**: `device: cpu` loads in 53.5 s and decodes 6.12 s of speech
  in 2.3 s (**RTF 0.37**, 2.7× faster than real time, concurrency 1, 8
  vCPU). The docs' "expected to be far below real time" is refuted on this
  box; updated in `docs/backends.md`. MPS remains unverifiable on Linux
  (accepted by the constructor, never run).

## Measured numbers

### Load / memory

- `backend.start()` (load + warmup): **53.8 s** direct; server boot→ready
  **54 s**.
- Resident VRAM at rest: **2927 MiB** (fp32). Validator peak 3001 MiB / 39%
  util. Load-ramp peak ~3.5 GiB. Post-stop: ~36 MiB residual cache (cosmetic;
  full release on process exit).

### Accuracy

- LibriSpeech **test-clean, n=100, seed 42, pace 1.0**: **WER 3.74%**, 0
  errors, 0 dropped chunks — identical in ws and file modes.
  (qwen3asr reference on the same corpus/settings: 3.37%.)
- TTS clips (edge-tts): zh CER **0.00%**, en WER **0.00%**, explicit and
  auto modes, both surfaces.

### Latency (concurrency 1, ws)

- server `first_partial` p50 839 ms / p95 940 ms (pre_roll 300 ms + 560 ms
  first chunk + decode dominate — structural, not model-bound)
- server `final` p50 38 ms / p95 42 ms; client `final` p50 39 ms / p95 73 ms

### Concurrency ramp (5 s synthetic utterances, 2 s p95 SLO)

| conc | pass | p50 | p95 | p99 | drop | gpu% |
|---|---|---|---|---|---|---|
| 4 | ✅ | 42 | 106 | 161 | 0 | 28% |
| 8 | ✅ | 61 | 290 | 423 | 0 | 37% |
| 12 | ✅ | 207 | 529 | 730 | 0 | 37% |
| 16 | ✅ | 1336 | 1533 | 1756 | 0 | 37% |
| 20 | ❌ | 2825 | 3253 | 3386 | 0 | 37% |

`max_passing_concurrency = 16`. 0 drops/errors/rejections at every rung.

### Model suite

`pytest -m "model and gpu" tests/backends/test_nemotron_backend_model.py`:
**10/10 passed** (5 conformance + streaming-partials + zh/en explicit +
auto-detect + final-flush + 4-stream isolation). Full ML-free regression:
403 passed / 2 skipped. `ruff check` clean.

## Findings & fixes

1. **Box-level cuDNN sublibrary clash (environment, not code).** The A10 box
   has CUDA toolkits (`/usr/local/cuda-12.8`, `-13.0`) whose cuDNN 9.19.1
   libraries are registered in `ldconfig`. torch 2.14.0+cu130 bundles
   cuDNN 9.24 (`nvidia_cudnn_cu13` wheel); its main lib loaded but the
   lazily-loaded graph/ops sublibraries resolved to the 9.19.1 system copies
   → every conv2d failed with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. The
   backend's failure-soft warmup swallowed the first occurrence exactly as
   designed; the first real decode then raised it. **Fix (no code change):**
   prefix `LD_LIBRARY_PATH` with
   `.venv/lib/python3.12/site-packages/nvidia/cudnn/lib` for every GPU
   command on this box. Documented in the runbook's troubleshooting section.
2. **Fixture quality (procedure, not code).** espeak-ng Mandarin produced
   audio the model decodes to *empty* in every language mode (its pinyin
   approximation is not intelligible to the model); its English was
   recognizable but degraded ("Can you sleep on a smooth blanks…"). Natural
   edge-tts voices (`zh-CN-XiaoxiaoNeural`, `en-US-AriaNeural`) decode
   perfectly. The runbook's Linux-fixture guidance now names edge-tts.
3. **Test over-assertion (fixed).** `test_real_model_streams_...` asserted a
   detected language on the 1 s fixture; the model never emits the tag that
   early. Assertion dropped with the measurement in a comment; language
   detection remains pinned by `test_auto_detects_zh` on 6 s clips.

## Code changes (this report's commit)

- `tests/backends/test_nemotron_backend_model.py` — drop the 1 s-fixture
  language assertion (finding 3).
- `docs/nemotron_a10_runbook.md` — troubleshooting: cuDNN `LD_LIBRARY_PATH`
  prefix; §6 Linux fixtures: use edge-tts, not espeak-ng; checklist items
  marked with answers.
- `docs/backends.md#nemotron` — "Verified on" note with the numbers above;
  CPU-mode claim corrected (RTF 0.37, not "far below real time");
  concurrency guidance (measured ceiling 16; shipped cap 8 and why).
- `configs/nemotron.yaml` — comment updates only (no longer "NOT run
  anywhere"; max_sessions stays 8 with the measured-ceiling rationale).
- `CLAUDE.md` — nemotron no longer "NOT yet run on real hardware".

## Artifacts (gitignored)

- `benchmarks/results/nemotron-a10-validate.json` — validator, 10/10 cases.
- `benchmarks/results/load-nemotron-3.5-asr-streaming-0.6b-20260906-*.json`
  — both ramps (max_passing 8 and 16).
- `benchmarks/results/accuracy-nemotron-3.5-asr-streaming-0.6b-test-clean-20260906-023831.json`
  — WER 3.74% run.
- `benchmarks/audio/{zh,en}_sample.wav` — edge-tts clips (local, gitignored;
  licensing rationale unchanged from the runbook).
- `models/{silero_vad.onnx, nemotron-3.5-asr-streaming-0.6b/*.nemo}`.
