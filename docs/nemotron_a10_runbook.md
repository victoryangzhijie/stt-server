# Nemotron A10 bring-up runbook

Everything needed to take the `nemotron` backend
(`nvidia/nemotron-3.5-asr-streaming-0.6b`) from **never executed** to
**measured**, on an NVIDIA A10 (24 GB, compute capability 8.6).

Read this together with:

- `docs/backends.md#nemotron` — the honest verified-on note and the config
  reference. Every capability claim there is currently unverified.
- `benchmarks/cuda_runbook.md` — the surrounding GPU-box procedure
  (`scripts/run_gpu_suite.sh` phases, how to hand results back).
- `GPU_VERIFICATION_REPORT.md` — the qwen3asr A10 run. Its "How to
  reproduce" section is the source of the mirror/venv commands below; the
  same box constraints apply.

The **UNVERIFIED checklist** at the end is the point of the exercise: work
through it and record an answer for each item.

---

## 1. System prerequisites

The model card asks for these explicitly (NeMo's audio I/O path needs
`libsndfile`, and several NeMo dependencies build from source):

```bash
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
```

Confirm the driver is new enough for whatever CUDA build the installed
`torch` wheel targets:

```bash
nvidia-smi           # driver + CUDA version header, and that the A10 is idle
```

## 2. Install the extras WITHOUT touching `uv.lock`

`uv pip install` resolves into the venv and does **not** rewrite `uv.lock`
(unlike `uv sync` / `uv lock` / `uv add`, which are forbidden by this repo's
policy — see `CLAUDE.md`). This is the same procedure the qwen3asr A10 run
used.

```bash
# `pip install Cython packaging` first: the model card calls for it, and
# several nemo_toolkit dependencies need them present at build time.
uv pip install --python .venv/bin/python Cython packaging

uv pip install --python .venv/bin/python \
    -e '.[nemotron,silero,bench]' pytest pytest-asyncio httpx ruff
```

If pypi.org is slow on the box (it was ~77 KB/s on the qwen3asr A10, versus
~3 MB/s from the aliyun mirror), add the mirror explicitly — again, without
touching the lock:

```bash
uv pip install --python .venv/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    -e '.[nemotron,silero,bench]' pytest pytest-asyncio httpx ruff
```

Then confirm the lock is untouched before doing anything else:

```bash
git status --short uv.lock     # must be empty
grep -c "mirrors.aliyun" uv.lock   # must be 0
```

### Verify CUDA torch

```bash
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "capability:", torch.cuda.get_device_capability(0))
    print("total VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
```

Expected on an A10: `cuda available: True`, `NVIDIA A10`, capability
`(8, 6)`, ~22-23 GiB usable.

Also confirm the NeMo class the backend depends on actually exists in the
installed toolkit (this is UNVERIFIED item #1):

```bash
.venv/bin/python -c "
import nemo, nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
print('nemo', nemo.__version__, EncDecRNNTBPEModelWithPrompt)"
```

## 3. Model weights

Two options; either works, pick one.

**A. Let NeMo pull it at `backend.start()`** — nothing to do; leave
`configs/nemotron.yaml`'s `model: nvidia/nemotron-3.5-asr-streaming-0.6b`
as-is. The first start pays a ~2.4 GB download.

**B. Pre-download the `.nemo` file** into `models/` (recommended, so a
failed first boot doesn't repeat a multi-GB fetch):

```bash
# HF_ENDPOINT is honored by huggingface_hub; use the mirror if the box is
# in CN (this is what the qwen3asr A10 run needed).
HF_ENDPOINT="https://hf-mirror.com" \
    uv run python scripts/download_models.py nemotron-3.5-asr-streaming-0.6b
# -> models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo (~2.37 GB)
```

Then point the config at the local file:

```yaml
      model: models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo
```

### Silero VAD

`configs/nemotron.yaml` uses the silero VAD, so the ONNX model must exist:

```bash
uv run python scripts/download_models.py silero
# If the GitHub LFS raw asset comes back truncated (it did on the qwen3asr
# A10), fetch it from the mirror instead — verify the I/O signature matches
# src/stt_server/core/vad_silero.py (input (1,512), state (2,1,128), sr int64):
curl -sL "https://hf-mirror.com/onnx-community/silero-vad/resolve/main/onnx/model.onnx" \
    -o models/silero_vad.onnx
```

## 4. Start the server

```bash
uv run stt-server --config configs/nemotron.yaml
```

The first start loads the model and runs the warmup decode
(`options.warmup: true`); expect it to take a while. Health check from
another shell:

```bash
curl -s http://127.0.0.1:8000/readyz     # {"status":"ready"}
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/metrics | head -20
```

Record `nvidia-smi`'s `memory.used` right after startup — that is the
resident model footprint, before any request (UNVERIFIED item: fp32 VRAM at
rest).

## 5. Real-model tests

```bash
uv run pytest -m "model and gpu" -s tests/backends/test_nemotron_backend_model.py
```

This is the conformance suite against the real model (start/stop,
finalize-yields-exactly-one-final, partials-ordered, close-without-finalize,
push-after-finalize-noop) plus the streaming and language tests. The zh
cases skip unless a Chinese fixture exists at `benchmarks/audio/zh_sample.wav`
— generate one first (next section).

## 6. Audio fixtures

On a macOS machine (not the A10 box):

```bash
bash scripts/make_tts_fixtures.sh
# -> benchmarks/audio/zh_sample.wav (~6.5s, Tingting)
# -> benchmarks/audio/en_sample.wav (~5.1s, Samantha)
# prints the reference transcripts to pass as --zh-ref / --en-ref
```

Copy them to the box (`benchmarks/audio/` is gitignored — `say` voice output
has unclear redistribution licensing, same reasoning as the FunASR Mandarin
check in `docs/backends.md#funasr`):

```bash
scp benchmarks/audio/*.wav <a10-box>:~/stt-server/benchmarks/audio/
```

On Linux, produce equivalents with **edge-tts** (natural neural voices; the
A10 bring-up measured that the model decodes these perfectly at CER/WER
0.00%):

```bash
uv pip install --python .venv/bin/python edge-tts
python - <<'PY'  # .venv/bin/python
import asyncio, edge_tts
async def gen(text, voice, out):
    await edge_tts.Communicate(text, voice).save(out)
ZH = "今天天气很好，我们打算下午去公园散步，顺便买一杯热咖啡。"
EN = "The canoe slid on the smooth planks, and the boy carried a heavy box of tools across the yard."
asyncio.run(gen(ZH, "zh-CN-XiaoxiaoNeural", "benchmarks/audio/zh_sample.mp3"))
asyncio.run(gen(EN, "en-US-AriaNeural", "benchmarks/audio/en_sample.mp3"))
PY
ffmpeg -y -loglevel error -i benchmarks/audio/zh_sample.mp3 -ar 16000 -ac 1 -c:a pcm_s16le benchmarks/audio/zh_sample.wav
ffmpeg -y -loglevel error -i benchmarks/audio/en_sample.mp3 -ar 16000 -ac 1 -c:a pcm_s16le benchmarks/audio/en_sample.wav
```

**Do not use espeak-ng for the zh clip**: measured on the A10, its Mandarin
synthesis decodes to an *empty* transcript in every language mode (auto,
zh-CN, en-US) — the model cannot understand its pinyin-approximation
output. Its English is recognizable but degraded, which would corrupt the
CER/WER smoke numbers. Convert any output with:
`ffmpeg -i in.any -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.

## 7. Validation run

```bash
uv run python scripts/validate_nemotron_a10.py \
    --zh benchmarks/audio/zh_sample.wav \
    --en benchmarks/audio/en_sample.wav \
    --zh-ref "今天天气很好，我们打算下午去公园散步，顺便买一杯热咖啡。" \
    --en-ref "The canoe slid on the smooth planks, and the boy carried a heavy box of tools across the yard." \
    --base-url http://127.0.0.1:8000 \
    --model nemotron-3.5-asr-streaming-0.6b \
    --pace 1.0 --concurrency 4 \
    --json benchmarks/results/nemotron-a10-validate.json
```

Covers, each as a separately-reported case: zh explicit (`language=zh` over
the file HTTP endpoint — the native WS carries no language field), en
explicit, auto-detection on both clips over both surfaces, the
`input_done`-with-no-trailing-silence final-flush path, N parallel WS
sessions with per-session first-partial/final latency, peak VRAM/utilization
sampled from `nvidia-smi`, and a hard failure if `stt_audio_dropped_total`
increased during the run. Exits nonzero if any case fails.

Do **not** raise `--pace` above 1.0: outrunning a real backend's decode trips
the server's `drop_oldest` backpressure and silently sheds audio, making
every latency and accuracy number meaningless (`benchmarks/cuda_runbook.md`,
"Methodology constraint").

Push concurrency past the default afterwards (`--concurrency 6`, `8`, ...) to
find where `client_final_ms` breaks the SLO or drops appear; that is the
concurrency-ceiling checklist item. `limits.max_sessions` in
`configs/nemotron.yaml` (8) caps it — raise it there first, or sessions are
rejected with 429 before the GPU is the constraint.

## 8. Benchmarks

```bash
# Concurrency ramp against a warm server (synthetic 5 s utterances).
uv run python -m benchmarks.run_load \
    --config configs/nemotron.yaml --model nemotron-3.5-asr-streaming-0.6b \
    --utterance-seconds 5 --start 1 --step 1 --max 8 \
    --window-seconds 30 --slo-final-ms 2000 --slo-pct 95 --seed 42 --synthetic

# English WER through the server, real-time paced (LibriSpeech test-clean).
uv run python -m benchmarks.run_accuracy \
    --config configs/nemotron.yaml --model nemotron-3.5-asr-streaming-0.6b \
    --split test-clean --n 100 --seed 42
```

`run_accuracy` defaults to `--pace 1.0` and asserts a zero
`stt_audio_dropped_total` delta; do not pass `--pace 0` or
`--no-assert-no-drops`. Results land in `benchmarks/results/` (gitignored) —
see `benchmarks/cuda_runbook.md`, "How to send results back".

The corpus is English-only, so `run_accuracy` measures en-US. Mandarin
accuracy has no corpus runner here; the validator's CER printout against the
TTS clip is the only zh number this repo can produce today, and it is a smoke
signal, not a benchmark.

Or run the nemotron phase of the GPU suite together with everything else:

```bash
SKIP_GPU_IMAGE=1 SKIP_CPU_IMAGE=1 bash scripts/run_gpu_suite.sh   # phase 8 = nemotron
SKIP_NEMOTRON=1 bash scripts/run_gpu_suite.sh                     # everything except nemotron
```

## 9. What to record

Write these down (a report file next to `GPU_VERIFICATION_REPORT.md`, or a
commit message):

| Item | Where it comes from |
|---|---|
| GPU model, driver, CUDA version | `nvidia-smi` header |
| `nemo` version and `torch.version.cuda` | §2 verification snippets |
| Model load + warmup time at `start()` | server log timestamps between boot and `ready` |
| Resident VRAM at rest (no requests) | `nvidia-smi` after startup |
| **Peak VRAM** and peak GPU utilization under load | validator's GPU summary; `run_load` result JSON |
| **first-partial latency** (server-side, per utterance) | validator's per-session table; `run_load` |
| **final latency p50 / p95** | `run_load` result JSON per rung |
| Per-step decode time vs the 560 ms chunk | server logs / a direct timing loop; the real-time question |
| **drops = 0** | validator's `no-audio-drops` case; `run_load`'s measured drops |
| **Detected language** on auto cases, zh and en | validator output (the FINAL's reported language) |
| zh CER and en WER on the TTS clips | validator's CER/WER printout |
| en-US WER on test-clean | `run_accuracy` result JSON |
| Concurrency ceiling (last rung meeting the SLO with 0 drops) | `run_load` |

## 10. UNVERIFIED checklist — ANSWERED (A10, 2026-09-06)

Each item below was answered on the A10; full numbers and method in
`NEMOTRON_A10_VERIFICATION_REPORT.md` (same repo root).

- [x] **`nemo_toolkit>=3.0.0` from PyPI contains `EncDecRNNTBPEModelWithPrompt`**
      at `nemo.collections.asr.models.rnnt_bpe_models_prompt`. **CONFIRMED**
      (nemo 3.0.0 wheel; §2 snippet runs clean).
- [x] **`ASRModel.from_pretrained(...)` resolves the HF id** and returns the
      prompt-capable class, and `restore_from(<local .nemo>)` works.
      **CONFIRMED — both paths exercised** (HF cache pre-seeded at revision
      `1c8deae…`; server/benchmarks used the local `.nemo`).
- [x] **fp32 vs AMP.** **fp32 wins on A10.** `amp: true`: identical
      transcripts, no NaNs, same ~2.9 GiB resident VRAM, but per-step
      44.2 ms vs 34.8 ms fp32 (+27%). Keep `amp: false`.
- [x] **Per-step latency.** `[56,0]` 31.4 ms / 80 ms chunk (RTF 0.39);
      `[56,3]` 34.4 / 320 (0.11); `[56,6]` 34.8 / 560 (0.06); `[56,13]`
      36.5 / 1120 (0.03). All real-time-safe per stream; the 560 ms default
      has ~16× headroom.
- [x] **Concurrency ceiling.** **16** concurrent 5 s streams pass a 2 s p95
      SLO with 0 drops (20 fails: p95 3.25 s). GPU util plateaus at 37% —
      the process-wide lock, not the GPU, is the limiter. Shipped
      `limits.max_sessions: 8` kept conservative (p95 290 ms there, ~7×
      margin); see `configs/nemotron.yaml`'s comment.
- [x] **zh-CN accuracy.** CER **0.00%** on clean natural TTS (smoke signal
      only — no Mandarin corpus runner exists). en test-clean WER 3.74%
      (n=100). Neither confirms nor contradicts NVIDIA's tiering.
- [x] **Automatic language ID.** **WORKS**: the `<xx-XX>` tag appears in
      streaming hypotheses and detected `Chinese`/`English` is reported on
      the FINAL. Caveat: needs a few seconds of speech — the tag never
      appears on the 1 s fixture (model test adjusted accordingly).
- [x] **Explicit per-request language.** **WORKS** for `zh`/`en` via the
      file endpoint (perfect transcripts); wrong-language prompts on clear
      foreign speech yield empty/gibberish, never crashes.
- [x] **Final flush keeps the utterance tail.** **CONFIRMED** — zh/en tails
      intact with no trailing silence; the pinned model test passes.
- [x] **MPS / CPU support.** **CPU works better than documented**: RTF
      **0.37** (2.7× faster than real time) at concurrency 1 on 8 vCPU —
      docs updated. MPS remains unverifiable on Linux.

## 11. Troubleshooting (findings from the A10 bring-up)

- **`RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed …
  CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` on every conv2d.** The box has
  CUDA toolkits (`/usr/local/cuda-12.8`, `/usr/local/cuda-13.0`) whose
  cuDNN 9.19.1 libraries are registered in `ldconfig`; torch 2.14.0+cu130
  bundles cuDNN 9.24 (`nvidia_cudnn_cu13` wheel in the venv). The main
  lib loads from the wheel but the lazily-loaded graph/ops sublibraries
  resolve to the system 9.19.1 copies → version mismatch. This makes the
  backend's own warmup fail (logged `nemotron.warmup_failed`, swallowed by
  design) and the first real decode raise. **Fix:** prefix `LD_LIBRARY_PATH`
  with the venv's cuDNN dir for every GPU command:

  ```bash
  export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ```

  Verify with a bare conv2d before touching the server. A stock box with
  only the driver installed (no toolkit) does not hit this.
- **Empty transcripts for a clip that should decode** — before blaming the
  backend, check the audio source: espeak-ng Mandarin decodes to empty in
  every language mode (measured). Use edge-tts per §6.
- **`hf_hub_download` re-downloads the 2.37 GB model in tests even though
  `models/` has it** — the model tests construct `NemotronBackend()` with
  the HF id, which uses the HF cache, not `models/`. Pre-seed the cache
  (blob copy + `refs/main` at the repo revision from the HF API) or just
  let the first `start()` fetch it once with `HF_ENDPOINT` set.
