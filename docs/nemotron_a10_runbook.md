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

On Linux, produce equivalents any way you like and convert:
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

## 10. UNVERIFIED checklist

Nothing about this backend has run on real hardware. Each item below is an
assumption made while writing it; confirm or refute each one and record the
answer.

- [ ] **`nemo_toolkit>=3.0.0` from PyPI contains `EncDecRNNTBPEModelWithPrompt`**
      at `nemo.collections.asr.models.rnnt_bpe_models_prompt`. The whole
      backend depends on this class existing in the published wheel, not
      only in the GitHub main branch. (§2 snippet.)
- [ ] **`ASRModel.from_pretrained(model_name="nvidia/nemotron-3.5-asr-streaming-0.6b")`
      resolves the HF id** and returns that prompt-capable class — and that
      `EncDecRNNTBPEModelWithPrompt.restore_from(<local .nemo>)` works for
      the pre-downloaded file path.
- [ ] **fp32 vs AMP.** Cache-aware NeMo models are documented as
      float32-only, with bf16/fp16 reachable only through `torch.autocast`.
      `options.amp: false` is the default. Measure both: does
      `amp: true` change VRAM, latency, or transcript quality? Does it
      produce NaNs or silently degrade?
- [ ] **Per-step latency at `att_context_size: [56, 6]` (560 ms chunks).**
      A step must complete in well under 560 ms of wall time per stream for
      one session to keep up with real time. Measure it, and measure the
      80 ms (`[56, 0]`) and 320 ms (`[56, 3]`) settings for the
      latency/accuracy trade-off.
- [ ] **Concurrency ceiling.** The backend holds a process-wide lock around
      `set_inference_prompt(...)` + the model step (the prompt is
      model-global and CUDA calls are not thread-safe), so the executor's
      `max_concurrent: 8` does not buy 8x throughput. Find the real ceiling
      and correct `limits.max_sessions` / the docs to match.
- [ ] **zh-CN accuracy.** NVIDIA places zh-CN in the "broad-coverage" tier
      (FLEURS CER ~19-20%) versus "transcription-ready" en-US. Does the
      measured CER on real Mandarin land anywhere near that? If it is much
      worse in streaming mode, say so in `docs/backends.md` and keep
      pointing Mandarin-only users at FunASR.
- [ ] **Automatic language ID actually works per utterance**, and the
      `<xx-XX>` tag really appears in the model's output text so
      `strip_lang_tags` / detected-locale reporting has something to parse.
      Mixed zh+en traffic on one server is the requirement this backend
      exists for.
- [ ] **Explicit per-request language overrides the config default** via the
      file endpoint's `language` form field, for both `zh`/`zh-CN` and
      `en`/`en-US` spellings.
- [ ] **Final flush keeps the utterance tail.** `finalize()` zero-pads the
      buffer remainder to a full chunk and asks for `keep_all_outputs=True`;
      confirm no trailing words are lost when `input_done` arrives with no
      trailing silence (validator case `*-final-flush`).
- [ ] **MPS / CPU support.** `device: cpu` and `device: mps` are accepted by
      the constructor but have never been run. Either verify them (and
      record how far below real time CPU is), or state plainly in the docs
      that they are unsupported.
