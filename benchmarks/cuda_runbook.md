# CUDA-box runbook (Plan 4, M-6)

The qwen3asr backend (vLLM) and the GPU Docker image cannot be exercised on
this repo's CPU-only development machine. This runbook is what to run on a
real CUDA box to close that gap: the real qwen3asr model+GPU tests, both
Docker image builds, and real-backend accuracy/load benchmark runs against
qwen3asr, plus one CPU-only carried-over check (funasr zero-length input)
that piggybacks on the same "run everything for real, once" session.

## Prerequisites

- **NVIDIA driver** new enough for CUDA 12.4 (the base image is
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, per `deploy/Dockerfile.gpu`) —
  check with `nvidia-smi` (also what `tests/backends/test_qwen3asr_backend_model.py`'s
  new skipif gates on: `shutil.which("nvidia-smi") is None`).
- **Docker** with the **NVIDIA Container Toolkit** installed and configured
  (`nvidia-ctk runtime configure` + restart the Docker daemon), so
  `docker run --gpus all` and `docker compose --profile gpu` work. Verify
  with `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04
  nvidia-smi`.
- **uv** (for phases 1/2/5/6) and **git** (repo checkout).
- **Disk space** — rough budget, ~15-20 GB total:
  - `Qwen/Qwen3-ASR-0.6B` weights (pulled by `qwen_asr`/`transformers` on
    first `backend.start()`, per `configs/qwen3asr.yaml`'s comment): a few
    GB.
  - LibriSpeech `test-clean` corpus (`benchmarks/data/`, gitignored): ~350
    MB, downloaded by `benchmarks.corpus.download_subset` the first time
    `run_accuracy`/`run_load` need a manifest.
  - GPU Docker image (`deploy/Dockerfile.gpu`, CUDA runtime base + vLLM +
    its own CUDA/cuDNN wheels): several GB.
  - Full CPU Docker image (`deploy/Dockerfile`, sherpa+funasr+silero,
    torch/torchaudio CPU wheels): several GB (see the Dockerfile's own
    size-warning comment).
  - Docker build cache/layers roughly double the two images' on-disk size
    until pruned (`docker builder prune`).

## The one command

```
git clone <this repo> && cd sst-server
bash scripts/run_gpu_suite.sh
```

Runs all 7 phases in order; see the script's header comment for the
`SKIP_*=1` env flags that make each phase independently skippable (useful
for re-running just the phase that failed, without repeating slow ones like
the Docker builds). The `UV_DEFAULT_INDEX="https://pypi.org/simple"` prefix
used elsewhere in this repo's session policy is **not required on a stock
CUDA box** (that constraint is local to this project's aliyun-mirrored dev
environment) — the script's own header documents it as an optional prefix
to fall back on if your box's uv index doesn't carry every wheel this repo
needs.

## Phases: expected artifacts and rough durations

| # | Phase | Artifact | Rough duration |
|---|---|---|---|
| 1 | `uv sync --extra qwen3asr --extra bench --extra funasr` | resolved `.venv` | 1-15 min (vLLM/torch wheel downloads dominate) — funasr is included so phase 7 actually runs; without it, phase 7 prints `[skip]` and the Plan 3 carried item stays unresolved |
| 2 | `pytest -m "model and gpu"` on the qwen3asr model tests | pytest output; exit 0 | 2-10 min (includes first-run model download + vLLM engine init) |
| 3 | `docker build -f deploy/Dockerfile.gpu .` | `stt-server:gpu` image | 5-15 min |
| 4 | `docker build -f deploy/Dockerfile .` | `stt-server:cpu` image | 5-15 min (torch/torchaudio CPU wheels dominate) |
| 5 | accuracy run vs qwen3asr | `benchmarks/results/accuracy-qwen3-asr-0.6b-test-clean-<timestamp>.json` | minutes-to-~2h — **real-time paced** (see methodology note below): 100 utterances at `--pace 1.0` means the WS pass alone takes roughly the corpus's total audio duration (`test-clean`'s 100-utterance sample is on the order of 10-20 minutes of audio), plus file-mode and per-utterance server-boot/teardown overhead |
| 6 | load run vs qwen3asr | `benchmarks/results/load-qwen3-asr-0.6b-<timestamp>.json` | up to `--max`/`--step` rungs × `--window-seconds` each (script's defaults: up to 8 rungs × 30s = ~4 min, plus per-rung server/model boot) |
| 7 | funasr zero-length `is_final=True` check | stdout `[ok] ...` line (no JSON artifact) | seconds, plus a one-time `paraformer-zh-streaming` model download if not already cached (see `scripts/download_models.py`) |

Phases 5/6 write into `benchmarks/results/`, which is **gitignored** (see
`benchmarks/README.md`) — nothing there is committed automatically.

## Methodology constraint (binding — carried from Plan 4 Task 7)

Real-backend server-WER runs **must** use `--pace >= 1.0` (real-time or
slower), never `--pace 0` (as-fast-as-possible). Task 7's endpointing
experiment found, by direct reproduction, that pacing a fast client above
real-time against a real (non-mock) backend can outrun the backend's decode
and trip the server's default `drop_oldest` backpressure (`configs/qwen3asr.yaml`
has no `limits:` section → default `audio_queue_chunks`, `drop_oldest`),
silently shedding audio — the resulting WER would measure accuracy *under
active audio drop*, not the real pipeline. This generalizes to every
WER-through-the-server benchmark against a real backend (see
`benchmarks/README.md`'s "Arm A validity guard" section for the full
writeup). Every streaming runner (`run_accuracy` WS mode,
`run_stabilizer_study`, `run_endpointing`) therefore asserts zero
`stt_audio_dropped_total` delta by default (`--assert-no-drops`, the
default) and hard-fails otherwise.

Accordingly:

- `scripts/run_gpu_suite.sh`'s phase 5 (`run_accuracy`) invokes qwen3asr
  with **no `--pace` flag at all**, which defaults to `1.0` — do not add
  `--pace 0` to this command. If the run hard-fails on a nonzero
  `audio_dropped_delta`, that is a real finding (the real backend under
  real-time load still can't keep up) — investigate `max_concurrent`/queue
  sizing in `configs/qwen3asr.yaml`, don't silence it with
  `--no-assert-no-drops`.
- `run_load`'s ramp (phase 6) has no such assertion by design — dropped
  chunks there are a *measurement* that fails a rung's SLO gate, not an
  invalid run (see `benchmarks/README.md`).

## How to send results back

`benchmarks/results/*.json` is gitignored by design (per-run artifacts,
not source) — nothing lands in the repo automatically. Choose one:

1. **Commit deliberately.** From the CUDA box (or after copying the files
   to your normal dev checkout — see below), `git add -f
   benchmarks/results/accuracy-qwen3-asr-0.6b-test-clean-*.json
   benchmarks/results/load-qwen3-asr-0.6b-*.json` (the `-f` is required
   since the directory is gitignored) and commit them explicitly, e.g. as
   part of Task 9's report commit. Only commit the specific files you mean
   to keep, not the whole gitignored directory.
2. **Copy the files off the box** without committing from there, e.g.:
   ```
   scp <cuda-box>:~/sst-server/benchmarks/results/*.json ./benchmarks/results/
   ```
   or `rsync`/a shared volume/object storage, then decide on (1) from your
   normal dev machine.
3. **Paste inline** (small JSON, quick sanity-check only) into whatever
   tracks Task 9's findings — not a substitute for (1)/(2) if the numbers
   are meant to be reproducible/citable later.

Either way, note in whatever channel you use: the git SHA the run was
against (`write_result` already embeds this in each JSON's own metadata),
the GPU model/driver/CUDA version (`nvidia-smi` header), and which phases
actually completed vs. were skipped.

## Troubleshooting

- **vLLM / CUDA version mismatch.** vLLM ships prebuilt wheels pinned to
  specific CUDA runtime versions; `deploy/Dockerfile.gpu` pins
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04` to match what `qwen-asr[vllm]`
  (per `pyproject.toml`'s `qwen3asr` extra) expects at the time this was
  written. If phase 2 or 3 fails with a CUDA/driver mismatch (e.g. `CUDA
  driver version is insufficient`, or a vLLM import error referencing a
  missing `libcudart`/`libnvrtc` symbol), first check `nvidia-smi`'s
  reported driver/CUDA version against what the installed `torch`/`vllm`
  wheel expects (`python -c "import torch; print(torch.version.cuda)"`
  inside the venv/image) — a driver too old for the wheel's CUDA build is
  the most common cause, not a code bug.
- **`cuda-runtime` vs `cuda-devel` base image risk (carried from Plan 3).**
  `deploy/Dockerfile.gpu` deliberately uses the *runtime* base image, not
  `-devel`, on the assumption that vLLM ships prebuilt CUDA wheels and
  needs no host-side `nvcc`/CUDA toolkit compiler at build or run time (see
  the Dockerfile's own header comment and `.superpowers/sdd/task-8-report.md`'s
  "Deferred-build risk note" from Plan 3, where this was flagged as
  unproven since the GPU image had never actually been built). If phase 3
  fails with a missing `nvcc`/toolkit-header error during `uv sync --extra
  qwen3asr` (e.g. a source build being triggered for some transitive
  dependency instead of using a prebuilt wheel), that assumption has broken
  for whatever `qwen-asr[vllm]` version is current — the fix is either
  pinning an older `vllm`/`qwen-asr` version known to ship the wheel, or
  switching the base image to `nvidia/cuda:12.4.1-devel-ubuntu22.04` (much
  larger image, but includes the compiler toolchain).
- **`docker run --gpus all` / compose GPU profile not seeing the GPU.**
  Almost always the NVIDIA Container Toolkit isn't installed or the Docker
  daemon wasn't restarted after `nvidia-ctk runtime configure`. Re-run the
  prerequisites' verification command
  (`docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04
  nvidia-smi`) in isolation before touching `docker compose --profile gpu`.
- **Phase 2 tests still skip even with a GPU present.** The skipif added in
  this task checks `shutil.which("nvidia-smi") is None` — if `nvidia-smi`
  isn't on the `PATH` of the user/shell running pytest (e.g. a restricted
  CI service account), the tests skip even though a GPU exists. Make sure
  `nvidia-smi` resolves in the exact shell/user context `uv run pytest`
  runs under.
- **Phase 5 (accuracy) hard-fails on a nonzero `audio_dropped_delta`.** See
  the methodology section above — this means the run itself is invalid,
  not that something is misconfigured to be worked around with
  `--no-assert-no-drops`. Check `configs/qwen3asr.yaml`'s
  `max_concurrent`/any `limits:` section and the GPU's actual decode
  throughput before re-running.
- **Phase 6 (load) rejects sessions immediately at low concurrency.** Check
  `limits.max_sessions` in `configs/qwen3asr.yaml` (default 100 if unset,
  per `src/stt_server/config/settings.py`) against the script's `--max`;
  `run_load.py` fails fast at startup if `max_sessions < --max` — see
  `benchmarks/README.md`'s "Capacity-limit gotcha".
- **Phase 7 fails to find `funasr`.** Expected and handled: the heredoc
  checks `importlib.util.find_spec("funasr")` first and prints `[skip]`
  with exit 0 rather than failing — the funasr extra is unrelated to the
  GPU/qwen3asr work this box exists for, and installing it is optional.
