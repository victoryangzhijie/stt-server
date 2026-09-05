#!/usr/bin/env bash
# Generate the zh + en speech clips that scripts/validate_nemotron_a10.py
# takes as --zh / --en, using macOS's built-in `say` TTS. macOS only (`say`
# and `afconvert` are Apple tools); on Linux use any TTS/recording you like
# and convert with:
#     ffmpeg -i in.any -ar 16000 -ac 1 -c:a pcm_s16le out.wav
#
# Output: benchmarks/audio/{zh_sample,en_sample}.wav, 16 kHz mono s16le —
# exactly what the server pipeline assumes end to end. The directory is
# gitignored: `say` voice output has unclear redistribution licensing (the
# same reason the Plan 3 FunASR Mandarin check was verified manually and its
# audio never committed — see docs/backends.md#funasr), so these clips are
# local artifacts, never repo fixtures.
#
# The reference transcripts are printed at the end; pass them to the
# validator as --zh-ref / --en-ref to get a CER/WER printout.
#
# Usage:  bash scripts/make_tts_fixtures.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/audio"

ZH_TEXT="今天天气很好，我们打算下午去公园散步，顺便买一杯热咖啡。"
EN_TEXT="The canoe slid on the smooth planks, and the boy carried a heavy box of tools across the yard."

# Preferred voices, with fallbacks. `say -v '?'` lists what is installed;
# Tingting is the classic zh_CN voice but a given macOS install may only have
# one of the newer zh_CN voices (Eddy, Flo, Sandy, ...), so fall back to the
# first zh_CN voice present rather than failing.
ZH_VOICE="${ZH_VOICE:-}"
EN_VOICE="${EN_VOICE:-Samantha}"

if ! command -v say >/dev/null 2>&1; then
    echo "!! \`say\` not found — this script is macOS only." >&2
    echo "   Produce 16 kHz mono s16le WAVs some other way (see the header)." >&2
    exit 1
fi
if ! command -v afconvert >/dev/null 2>&1; then
    echo "!! \`afconvert\` not found — this script is macOS only." >&2
    exit 1
fi

if [[ -z "$ZH_VOICE" ]]; then
    if say -v '?' | grep -q '^Tingting'; then
        ZH_VOICE="Tingting"
    else
        # First zh_CN voice in the list; the name is everything before the
        # locale column, trailing spaces trimmed.
        ZH_VOICE="$(say -v '?' | grep 'zh_CN' | head -1 | sed 's/  *zh_CN.*//')"
    fi
fi
if [[ -z "$ZH_VOICE" ]]; then
    echo "!! no zh_CN voice installed. Add one in System Settings > Accessibility >" >&2
    echo "   Spoken Content > System Voice > Manage Voices, then re-run." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

render() {
    local voice="$1" text="$2" out="$3"
    local tmp
    tmp="$(mktemp -t stt-tts).aiff"
    say -v "$voice" -o "$tmp" "$text"
    # LEI16@16000 = little-endian signed 16-bit PCM at 16 kHz; -c 1 = mono.
    afconvert -f WAVE -d LEI16@16000 -c 1 "$tmp" "$out"
    rm -f "$tmp"
    echo "wrote $out  ($(python3 -c "
import sys, wave
with wave.open('$out') as w:
    print(f'{w.getnframes() / w.getframerate():.2f}s, {w.getframerate()} Hz, {w.getnchannels()} ch, {w.getsampwidth() * 8}-bit')
"))"
}

render "$ZH_VOICE" "$ZH_TEXT" "$OUT_DIR/zh_sample.wav"
render "$EN_VOICE" "$EN_TEXT" "$OUT_DIR/en_sample.wav"

cat <<EOF

Voices used: zh=$ZH_VOICE  en=$EN_VOICE

Reference transcripts (pass these to the validator):

  --zh-ref "$ZH_TEXT"
  --en-ref "$EN_TEXT"

Full command:

  uv run python scripts/validate_nemotron_a10.py \\
      --zh benchmarks/audio/zh_sample.wav \\
      --en benchmarks/audio/en_sample.wav \\
      --zh-ref "$ZH_TEXT" \\
      --en-ref "$EN_TEXT" \\
      --model nemotron-3.5-asr-streaming-0.6b \\
      --concurrency 4 --json benchmarks/results/nemotron-a10-validate.json
EOF
