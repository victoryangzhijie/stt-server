# bench-audio-v1-en — STT benchmark audio dataset

Curated **English-only** audio used by the STT-server load / latency benchmarks.
Two buckets under this directory, plus a manifest and a validator. The raw
LibriSpeech corpus this is derived from lives (gitignored) at
`benchmarks/data/LibriSpeech/` and is **not** committed.

> **Dataset version: `bench-audio-v1-en`** — see the versioning rule below.

## License & attribution

All audio here is derived from **LibriSpeech** (`test-clean`), released by
OpenSLR under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/).

Please cite:

> V. Panayotov, G. Chen, D. Povey and S. Khudanpur, *"Librispeech: An ASR
> corpus based on public domain audio books,"* IEEE ICASSP 2015,
> pp. 5206–5210.

No TED-LIUM (CC BY-NC-ND, incompatible) and no GigaSpeech (click-through
license) are used. No TTS-synthesized audio; LibriSpeech is the sole corpus.

## Source & how it was obtained

Canonical source: OpenSLR LibriSpeech `resources/12/test-clean.tar.gz`
(346 MB, 2620 utterances, 40 speakers). On this CN host the OpenSLR endpoints
were unusable for the build — `openslr.magicdatatech.com` serves an **expired
TLS certificate**, and `www.openslr.org` throttles to ~30 KB/s here. The
byte-identical tarball was therefore fetched from the `k2-fsa/LibriSpeech`
mirror over `hf-mirror.com` (the HuggingFace CN mirror, since direct
`huggingface.co` is unreachable from this host). Integrity was verified against
the canonical counts: **2620 FLACs / 87 transcripts / 40 speakers**, exactly
matching OpenSLR-12 `test-clean`. The cache lives at
`~/.cache/librispeech/test-clean.tar.gz`; only the curated WAVs below enter git.

## Contents

| Path | Count | Duration bucket | Format |
|---|---|---|---|
| `http/` | 60 WAV | 10–20 s (single bucket) | 16 kHz, mono, 16-bit PCM |
| `ws/` | 8 WAV | 30–60 s | 16 kHz, mono, 16-bit PCM |
| `manifest.csv` | 68 rows | — | per-clip metadata + reference transcript |
| `validate.py` | — | — | ffprobe-based contract checker |

`manifest.csv` columns: `relative_path, duration_s, sample_rate, channels,
source, transcript`. `relative_path` is relative to this directory. `source` is
`"LibriSpeech test-clean <utt-id>"` for single clips, and
`"LibriSpeech test-clean <id1>;<id2>;..."` (semicolon-joined member IDs) for the
spliced `ws/` clips. Transcripts come from LibriSpeech's own reference
transcriptions — the load test does not score WER, but the text is kept for
spot-checking results.

## Why two buckets (design intent)

The load-test matrix varies **only** concurrency (1/2/4/8). Audio duration must
therefore be held constant — an uncontrolled duration is a hidden second
variable.

- **`http/` — single 10–20 s bucket.** One tight duration band so that, at each
  concurrency level, every request does comparable decode work. The HTTP runner
  fires ≥50 requests per level and rotates these files round-robin. Selection
  favours speaker diversity (≤4 clips per speaker; all 40 speakers represented).
- **`ws/` — 30–60 s clips.** Streaming is fed at real-time pace (`--pace 1.0`),
  so a clip that is too short cannot reveal how first-partial / final latency
  degrades under contention. Each clip is a concatenation of **consecutive
  utterances from one speaker, one chapter**, with 0.3 s of silence at every
  splice point, so the audio stays linguistically continuous.

## Format

All WAVs are **16 kHz, mono, 16-bit PCM** (`pcm_s16le`), produced from the
source FLACs with:

```bash
ffmpeg -y -i <in.flac> -ar 16000 -ac 1 -sample_fmt s16 -c:a pcm_s16le <out.wav>
```

`ws/` clips are assembled from chapter-ordered member clips with a 0.3 s silence
slab (`anullsrc`) between neighbours via the concat demuxer (`-c copy`, no
re-encode):

```bash
# 0.3 s silence slab (same format as the clips)
ffmpeg -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000 -t 0.3 \
       -ar 16000 -ac 1 -sample_fmt s16 -c:a pcm_s16le silence.wav
# concat: clip0, silence, clip1, silence, clip2, ...
ffmpeg -f concat -safe 0 -i list.txt -c copy <out.wav>
```

Selection is **seeded** (`random.seed(20260801)` for `http/`, `+1` for `ws/`),
so the same corpus + seed reproduces the same subset.

## Validate

```bash
python3 benchmarks/data/validate.py
# or via uv (touches no project deps / lockfile):
UV_DEFAULT_INDEX="https://pypi.org/simple" uv run --no-project --python 3.12 \
    python benchmarks/data/validate.py
```

It re-probes every clip with `ffprobe`, asserting decodability + 16 kHz / mono /
16-bit PCM + in-bucket duration, checks the manifest (row count == on-disk file
count, no missing paths, no empty transcripts), prints per-file PASS/FAIL plus
bucket min/median/max, and exits non-zero on any failure.

## Usage mandate (versioning)

The **8.1 / 8.8 / 8.15** benchmark runs **must use this directory's data**;
their result-JSON metadata `dataset` field is set to **`bench-audio-v1-en`**
(add it to the `meta` block written by `benchmarks/results.py:write_result`).
If the dataset is ever changed (clips added/removed, durations altered, a new
source), **bump the version number** (e.g. `bench-audio-v2-en`) and note the
change here so historical results stay comparable.

## Note on the WS (streaming) matrix

The streaming load runner `benchmarks/run_load.py` drives a real server
subprocess over the native WebSocket and pulls audio **directly from the
official `test-clean` corpus** via `benchmarks/corpus.py`'s
`download_subset("test-clean")` (run_load.py line ~168). That call is
idempotent — it skips downloading when `benchmarks/data/LibriSpeech/test-clean/`
already exists — so it uses the **pre-extracted** `LibriSpeech/` copy here and
needs **nothing from `http/` or `ws/`**. This curated `ws/` bucket exists for
hand-driven streaming latency probes (e.g. first-partial timing under fixed
concurrency), not for `run_load.py`.
