# Benchmark suite

Implements spec §10: accuracy/WER, latency, load, stabilizer-study, and
endpointing benchmarks against a real `stt-server` process. Lives as a
top-level package (`benchmarks/`, not under `src/`) so nothing in
`src/stt_server` ever depends on it.

## Install

```
uv sync --extra bench
```

The `bench` extra (`jiwer`, `psutil`, `pynvml`, `soundfile`) is only needed
for corpus tooling (FLAC decode), WER scoring, and resource sampling. The
WS/file clients and the `ServerUnderTest` runner (`client_ws.py`,
`client_file.py`, `server.py`) need only the base dev environment (`httpx`
and `websockets` are already present via `uvicorn[standard]`).

Tests that need the extra guard their imports with `pytest.importorskip`, so
`uv run pytest` stays green with or without `--extra bench` installed.

## Corpus

Benchmarks run against [LibriSpeech](https://www.openslr.org/12) `test-clean`
/ `test-other`. Download + extract a split (~350MB for `test-clean`) into the
gitignored `benchmarks/data/`:

```
uv run python -c "from benchmarks.corpus import download_subset; print(download_subset('test-clean'))"
```

Then build a seeded, reproducible manifest of N utterances:

```python
from benchmarks.corpus import build_manifest
manifest = build_manifest(split_dir, n=100, seed=42)
```

`benchmarks/data/` and `benchmarks/results/` are gitignored — no corpus audio
or result JSON is ever committed.

## Runners

Each runner boots a real server subprocess via `benchmarks.server.ServerUnderTest`
(so benchmarks measure the actual wire protocol, not an in-process shortcut),
drives it with `benchmarks.client_ws.stream_utterance` and/or
`benchmarks.client_file.transcribe_file`, and writes a timestamped JSON result
via `benchmarks.results.write_result`.

- `run_accuracy.py` (Task 4) — WER (streamed + file mode) and concurrency-1
  latency (server- and client-observed) against a LibriSpeech manifest. The
  real-time-paced WS pass IS the concurrency-1 latency measurement — one
  pass yields both:

  ```
  uv run python -m benchmarks.run_accuracy \
      --config configs/sherpa.yaml --model sherpa --split test-clean \
      --n 100 --seed 42 [--modes ws,file] [--port 8100] \
      [--python /path/to/venv/bin/python] [--pace 1.0]
  ```

  - `--config` — stt-server YAML config path.
  - `--model` — the model name queried over the wire; looked up in
    `--config`'s top-level `models:` mapping to find the backend key (see
    `resolve_backend` in `src/stt_server/api/app.py`) — pass whichever key
    your config's `models:` section uses.
  - `--split` — `test-clean` or `test-other`.
  - `--n` / `--seed` — manifest size and sampling seed (`build_manifest` is
    deterministic for a given seed + corpus).
  - `--modes` — comma-separated subset of `{ws,file}` (default: both).
  - `--port` — port for the spawned server (default `8100`).
  - `--python` — interpreter to run the server with (default:
    `sys.executable`; needed for GPU/venv-pinned backends).
  - `--pace` — WS pacing multiplier (`1.0` = real-time, `0` = as fast as
    possible).

  Scores corpus-level (length-weighted) WER via `jiwer.wer` over the full
  reference/hypothesis lists — deliberately not the mean of per-utterance
  WERs, which would over-weight short utterances relative to their actual
  share of the corpus's words. Writes
  `accuracy-<model>-<split>-<timestamp>.json` via `write_result`, with
  per-mode WER, latency percentiles (server-reported first-partial/final and
  client-observed final), file-mode RTF (audio seconds / wall seconds;
  higher = faster than real-time), and an `errors` count (utterances
  yielding no transcript are scored as an empty hypothesis rather than
  crashing the run, and contribute nothing to the latency/RTF populations).
  Caveats: a streamed hypothesis is ALL final transcripts joined with `" "`
  (an utterance whose audio endpoints into multiple segments still scores
  as one hypothesis); an empty hypothesis counts as an error even when the
  audio was genuinely silent — a known conflation of "backend failed" and
  "nothing to transcribe". If `MAX_CONSECUTIVE_ERRORS` (5) utterances error
  in a row the run aborts early (dead-server circuit breaker); scattered
  failures still continue-on-error.

- `run_load.py` (Task 5) — concurrency ramp until an SLO breach, with
  CPU/RSS/GPU resource sampling per rung.

  ```
  uv run python -m benchmarks.run_load \
      --config configs/mock.yaml --model mock --synthetic \
      --utterance-seconds 2 --start 2 --step 2 --max 4 \
      --window-seconds 5 --slo-final-ms 5000 --slo-pct 95 --seed 42
  ```

  (the above is a fast local smoke run; a real ramp uses larger
  `--window-seconds`/`--max`, e.g. `--start 2 --step 2 --max 64
  --window-seconds 30 --slo-final-ms 1200 --slo-pct 95`.)

  Ramps concurrency from `--start` in `--step` increments up to `--max`. At
  each rung, `--start`/`--step`/`--max` concurrent WS workers each stream
  utterances back-to-back for `--window-seconds`; `--synthetic` tiles the
  repo's `tests/fixtures/speech_16k_mono_s16le.pcm` fixture to
  `--utterance-seconds` (every worker replays that one buffer) instead of
  downloading a LibriSpeech manifest — use it for the mock backend and any
  quick smoke run. A rung PASSES iff `p{--slo-pct}(client_final_ms) <=
  --slo-final-ms` AND there were zero backpressure/error events (client-side
  errors plus `/metrics` deltas of `stt_audio_dropped_total` /
  `stt_rejections_total` / `stt_errors_total`, summed). The ramp stops at
  the first failing rung (or at `--max`); output is
  `load-<model>-<timestamp>.json` via `write_result`, with per-rung rows
  (concurrency, p50/p95/p99 latency, error/dropped/rejection counts, peak
  CPU%/RSS, and peak GPU util/mem when `pynvml` + an NVIDIA device are both
  present) and `max_passing_concurrency`. Note that a rung can overrun
  `--window-seconds` by up to one utterance duration: the deadline is
  checked between utterances, so an utterance already in flight when the
  window closes runs to completion and its measurement is still counted.

  **Capacity-limit gotcha:** the server's own `limits.max_sessions` (default
  100, `src/stt_server/config/settings.py`) rejects sessions past that cap
  with a capacity rejection *before* the backend ever sees them — the ramp
  can't tell that apart from the backend actually failing. `run_load.py`
  fails fast at startup if `--config`'s `limits.max_sessions < --max`; raise
  the cap first (e.g. `STT__LIMITS__MAX_SESSIONS=128`) if you're ramping
  past 100.
- `run_stabilizer_study.py` (Task 6) — flicker-rate / commit-latency grid
  search over stabilizer settings, plus the realtime-API delta-duplication
  measurement (M-8, spec §10.4).

  ```
  uv run python -m benchmarks.run_stabilizer_study \
      --config configs/sherpa.yaml --model sherpa --n 25 --seed 42 \
      --grid "min_partials=1,2,3;min_stable_ms=0,240,480" \
      [--api native|realtime] [--split test-clean] [--port 8100] \
      [--python /path/to/venv/bin/python] [--pace 1.0]
  ```

  - `--grid` — a `;`-separated set of `StabilizerConfig` field axes (see
    `src/stt_server/config/settings.py`; currently `min_partials` (int) and
    `min_stable_ms` (float) are the only real fields), each a
    `,`-separated list of values, e.g. `"min_partials=1,2,3;
    min_stable_ms=0,240,480"` (9 grid points, the full cross product). An
    unknown field name raises immediately rather than silently no-opping.
  - For each grid point, the base `--config` YAML is loaded, its
    `stabilizer:` block patched with just that point's values (everything
    else, including unspecified stabilizer fields, is left alone), and
    written to a tempfile that boots a fresh `ServerUnderTest` — so every
    grid point runs against an otherwise-identical config.
  - `--api native` (default) streams the manifest over `/ws/transcribe`
    (`benchmarks.client_ws.stream_utterance`) and reports, per grid point:
    corpus WER, mean/percentiles of `flicker_rate` (retracted characters
    over the utterance's own final length — a pure churn measurement, not
    accuracy), and mean/percentiles of `commit_latency_ms` (mean time, per
    word of the final transcript, between the word first appearing in ANY
    hypothesis and it becoming part of the STABLE prefix). Both are pure
    functions (`retracted_chars`, `flicker_rate`, `commit_latency_ms` in
    `run_stabilizer_study.py`) with fully worked examples in their
    docstrings and in `tests/benchmarks/test_flicker_metrics.py` — this is
    the exact math Task 9's report cites.
  - `--api realtime` streams the SAME audio over
    `/v1/realtime?intent=transcription` instead
    (`benchmarks.client_realtime.stream_realtime`, a from-scratch minimal
    client for that protocol) and reports corpus WER plus
    `delta_duplication_ratio` (M-8): for each
    `conversation.item.input_audio_transcription.*` stream,
    `max(0, len("".join(deltas)) - len(completed_transcript)) /
    len(completed_transcript)`. `0.0` means the delta stream was
    wire-perfect (pure append, `"".join(deltas) == completed_transcript`
    modulo casefold-only casing drift); a positive value quantifies the
    "shrinking final" fallback in `src/stt_server/api/realtime_ws.py`,
    where a FINAL transcript that doesn't casefold-extend what was already
    sent triggers a full-text re-send instead of a true incremental
    delta. Flicker-rate/commit-latency are NOT computed in this mode — the
    realtime protocol has no wire representation of `PARTIAL`/volatile
    text to measure them from.
  - `--pace` (default `1.0`) and `--split` (default `"test-clean"`, not
    required) are both additions beyond the task brief's example
    invocation: `--pace` mirrors `run_accuracy.py`'s flag (needed so the
    mock-backend smoke can run with `--pace 0`, as-fast-as-possible);
    `--split` exists because `build_manifest` needs *some* split and the
    brief's example omits the flag entirely.
  - Writes `stabilizer-study-<model>-<timestamp>.json` via `write_result`:
    one row per grid point (`params`, `wer`, `errors`, `n`, plus either
    `flicker_rate`/`commit_latency_ms`/`commit_latency_included_words`
    (the total word count that actually entered the commit-latency means —
    the metric's exclusion rule shrinks the denominator, and grid
    comparisons need to see that shrinkage) or `delta_duplication_ratio`
    depending on `--api`). Note the JSON is written once, after the WHOLE
    grid completes: a mid-grid failure (server boot failure, tripped error
    breaker, Ctrl-C) aborts the run with no partial results file.
- `run_endpointing.py` (Task 7) — server VAD (Arm A, through the server) vs.
  sherpa-onnx NATIVE endpoint detection (Arm B, direct recognizer, spec
  §10.5). This is an OFFLINE experiment script, NOT a new serving mode —
  the server pipeline is not touched anywhere; `SherpaBackend` still builds
  its recognizer with `enable_endpoint_detection=False`. sherpa-onnx is the
  only local backend with native endpointing, so this experiment is
  sherpa-only.

  ```
  <sherpa-venv>/bin/python -m benchmarks.run_endpointing \
      --config configs/sherpa.yaml --model sherpa-zipformer-en \
      --model-dir models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
      --split test-clean --n 25 --seed 42 \
      --python <sherpa-venv>/bin/python \
      [--port 8100] [--pace 1.0] [--lead-s 1.0] [--trail-s 1.0] \
      [--assert-no-drops | --no-assert-no-drops]
  ```

  **Arm A validity guard (audio shedding).** `--pace` defaults to **1.0
  (real-time)** and the runner scrapes `/metrics` before/after Arm A,
  recording the `stt_audio_dropped_total` delta as
  `arm_a.audio_dropped_delta` and hard-failing (unless
  `--no-assert-no-drops`) when it's nonzero. Found by direct reproduction
  in Task 7 review: at `--pace 0` a fast client outruns a real backend's
  decode and trips the server's default `drop_oldest` backpressure
  (`configs/sherpa.yaml` has no `limits:` section →
  `audio_queue_chunks=64`, `drop_oldest`), silently shedding audio — the
  resulting "WER" measures accuracy under active audio drop, not the
  pipeline. **This generalizes to every WER-through-the-server benchmark
  against a real backend, and ALL streaming runners now assert it by
  default:** `run_accuracy` (WS mode), `run_stabilizer_study` (per grid
  point, both native and realtime APIs), and `run_endpointing` (Arm A) each
  bracket their streaming pass with `/metrics` scrapes via the shared
  `benchmarks._drops` helper, record the `stt_audio_dropped_total` delta in
  their result JSON (`results.ws.audio_dropped_delta`,
  `points[i].audio_dropped_delta`, `arm_a.audio_dropped_delta`
  respectively), and hard-fail on a nonzero delta unless
  `--no-assert-no-drops`. `run_accuracy`'s file mode is exempt by
  construction: the whole upload is pushed through its Session as ONE
  AudioChunk, which cannot overflow the chunk-counted queue.
  (`run_load` deliberately has no assertion — dropped chunks there are a
  *measurement* that fails the rung's SLO gate, not an invalid run.)

  Both arms are fed the SAME padded audio: every manifest utterance gets
  `--lead-s` (default 1.0s) of TRUE (digital-zero) leading silence and
  `--trail-s` (default 1.0s) of trailing silence prepended/appended
  (`pad_with_silence`), which gives an exact, arm-independent
  `true_speech_end_ms` ground truth — both arms' endpoint-detection latency
  is measured against that timestamp, never against anything either arm
  itself reports.

  - **Arm A** streams the padded audio over `/ws/transcribe` against
    `--config` (the real, unmodified server pipeline: VAD + `Endpointer` +
    stabilizer decide segmentation). Per utterance: number of segments
    (`len(finals)`), and the existing `server_final_ms` processing latency.
    Reports corpus WER, `segmentation` (`{"under"/"correct"/"over": n}` —
    every padded utterance is constructed to be exactly ONE isolated speech
    span, so `"correct"` means exactly 1 segment), and
    `final_latency_ms` percentiles.
  - **Arm B** builds a real `sherpa_onnx.OnlineRecognizer.from_transducer(...,
    enable_endpoint_detection=True)` directly (bypassing the server, and the
    Session/VAD/Endpointer/Stabilizer pipeline, entirely) with the package's
    DEFAULT endpoint rules, and feeds the same padded audio to a fresh
    `create_stream()` in 100ms steps, checking `is_endpoint(stream)` after
    each step (and `reset(stream)` on every fire, since `is_endpoint` stays
    true forever otherwise). Reports corpus WER, `segmentation`,
    `endpoint_latency_ms` percentiles computed over the POST-speech-end
    fires only (`fire_audio_time_ms - true_speech_end_ms`, for fires at or
    after the true speech end), a separate `premature_fires` count (fires
    BEFORE the true speech end — a premature fire is a MID-UTTERANCE fire,
    e.g. rule2 satisfied by an internal pause; its negative "latency" does
    NOT mean faster detection and is deliberately kept out of the latency
    percentiles), and a `per_utterance` list of raw
    `fire_times_ms`/`true_speech_end_ms` so anomalies are diagnosable from
    the JSON alone.
  - Arm B's hypothesis is the text at the FIRST fire (or the
    post-`input_finished()` flush text when no fire occurred). This relies
    on the padding guarantee that nothing but silence follows
    `true_speech_end_ms` — with real, unpadded audio the first fire's text
    would NOT necessarily hold the full utterance.
  - `write_result("endpointing-sherpa", ...)` writes `arm_a`/`arm_b` blocks
    plus a `comparison` block (WER delta, both segmentation tallies).

  **Design choice — Arm B runs IN-PROCESS, not as a subprocess worker.**
  `sherpa_onnx` is imported directly in the SAME process running this
  script (no `--arm b-worker` split) — the simplest design that keeps the
  CLI a single command. The cost: the WHOLE script (not just the
  `ServerUnderTest` Arm A spawns) must run under an interpreter with
  `sherpa_onnx` installed (plus the `bench` extra's
  `jiwer`/`soundfile`, and `httpx` — pulled in transitively via
  `benchmarks.run_accuracy`, which this module imports for
  `ConsecutiveErrorBreaker`/`corpus_wer`; the `/metrics` scraper comes from
  the stdlib-only `benchmarks._drops`). Run the
  whole command with the sherpa venv's python (recipe below), and pass that
  SAME interpreter to `--python` so Arm A's spawned server process (which
  also needs `sherpa_onnx`, per `configs/sherpa.yaml`) uses it too.

  **Real kwarg names (verified against installed `sherpa-onnx==1.10.46`
  via `inspect.signature(OnlineRecognizer.from_transducer)`):**
  `rule1_min_trailing_silence` (default `2.4` s), `rule2_min_trailing_silence`
  (default `1.2` s), `rule3_min_utterance_length` (default `20.0` s) — Arm B
  leaves all three at these defaults ("default endpoint rules").

  **Real-run finding worth flagging:** with the default `--lead-s`/
  `--trail-s 1.0` (1 second of trailing silence), Arm B's default rules
  (which need ≥1.2s trailing silence to fire under rule2, or ≥2.4s under
  rule1) essentially never see enough trailing silence to endpoint — an
  n=10 real run scored `segmentation_arm_b = {"under": 9, "correct": 1}`.
  This is a genuine property of sherpa's default rule thresholds relative
  to 1s of padding, not a bug in this script — see
  `.superpowers/sdd/p4-task-7-report.md` for the full real-run numbers
  (including a supplementary `--trail-s 3.0` run that does trigger fires)
  and discussion.

  ### sherpa venv recipe (also needed by Task 9)

  Same macOS-12/CoreML-dylib constraint as Task 5's model tests: newest
  `sherpa-onnx` fails to `dlopen` here, so pin `sherpa-onnx==1.10.46`. This
  venv needs the PROJECT installed too (editable), since `--python` boots
  `python -m stt_server` for Arm A, and the `bench` extra (for `jiwer`/
  `soundfile`; the `/metrics` scraper lives in the stdlib-only
  `benchmarks._drops`) plus `onnxruntime` (for `configs/sherpa.yaml`'s
  `vad.kind: silero`) plus `httpx` (transitively imported by
  `run_accuracy`):

  ```
  python3.12 -m venv sherpa-venv
  sherpa-venv/bin/pip install -i https://pypi.org/simple \
      -e '.[sherpa,bench]' 'sherpa-onnx==1.10.46' 'onnxruntime==1.19.2' httpx
  ```

  (`onnxruntime==1.19.2` is likewise the newest version with wheels
  available for this box's Python/platform combination — `>=1.20` has no
  matching wheel here.) Then run everything — both the Arm A server spawn
  (via `--python <venv>/bin/python`) and this script itself — with
  `<venv>/bin/python`, never `uv run`.

Each script's exact CLI is documented in its own module docstring as it
lands; see the plan-4 implementation plan for the full interface list.

## Running the real qwen3asr backend + GPU images (CUDA box)

The qwen3asr backend (vLLM) requires a CUDA GPU and cannot be exercised on
a CPU-only development machine; `tests/backends/test_qwen3asr_backend_model.py`
skips itself accordingly (`shutil.which("nvidia-smi") is None`). To run
those tests, both Docker image builds, and real accuracy/load benchmarks
against qwen3asr, see **`benchmarks/cuda_runbook.md`** — prerequisites,
the single `bash scripts/run_gpu_suite.sh` command, per-phase expected
artifacts/durations, and how to hand results back (the `benchmarks/results/`
JSONs are gitignored, same as everywhere else in this doc).
