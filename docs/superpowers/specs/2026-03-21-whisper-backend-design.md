# Whisper Backend Design

## Overview

Add a `whisper` STT backend using `faster-whisper` with streaming transcription ported from the `whisper_streaming` reference implementation. Uses Silero VAD for endpoint detection. Follows the same single-file, global-singleton pattern as the existing qwen3 backend.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Whisper library | `faster-whisper` | Fastest, CUDA via CTranslate2, most mature |
| Streaming strategy | Port OnlineASRProcessor (Local Agreement Policy) | Proven stability filtering, prevents partial flickering |
| VAD | Silero VAD (neural) | More accurate than energy-based RMS |
| Model config | Single `STT_WHISPER_MODEL` env var | faster-whisper accepts both names and paths |
| Concurrency | Global singleton + `_infer_lock` | Consistent with qwen3, horizontal scaling via workers |
| Audio buffer | Float32 numpy accumulation | OnlineASRProcessor needs full buffer for re-transcription |
| File structure | Single `backends/whisper.py` | Consistent with qwen3 pattern |
| Branch strategy | Feature branch for development | Isolate work from master |

## Architecture

### File: `backends/whisper.py`

Single file implementing `ASRBackend` protocol, containing:

- `WhisperBackend` class (public, implements protocol)
- `_HypothesisBuffer` class (private, text stability tracking)
- `_StreamingState` class (private, per-connection state)
- Global singleton model management
- Silero VAD integration

### Global Singletons

```
_whisper_model   : WhisperModel (faster-whisper)     loaded once, shared
_whisper_lock    : threading.Lock                     guards model loading
_infer_lock      : threading.Lock                     serializes inference
_vad_model       : Silero VAD model (torch)           loaded once, shared
_vad_lock        : threading.Lock                     guards VAD model loading
```

- `_get_whisper_model()`: lazy loader with double-checked locking
- `_get_vad_model()`: lazy loader, uses `torch.hub.load('snakers4/silero-vad', 'silero_vad')`
- `preload_model()`: eager loading at app startup
- All lazy imports (`faster_whisper`, `torch`, `numpy`) inside functions

### Streaming Engine

Ported from `whisper_streaming/whisper_online.py` OnlineASRProcessor.

#### `_HypothesisBuffer`

Tracks text stability across consecutive transcription iterations:

- `committed`: list of confirmed `(timestamp, word)` tuples
- `buffer`: currently stable but unconfirmed words
- `new`: latest transcription output
- `insert(new_words, offset)`: compare new transcription with previous, keep only new parts
- `flush()`: return longest common prefix between consecutive iterations (local agreement)
- `complete()`: return buffered incomplete text (for partials)

#### `_StreamingState`

Per-connection streaming context:

- `audio_buffer`: growing float32 numpy array
- `hypothesis`: `_HypothesisBuffer` instance
- `committed_text`: confirmed text so far
- `buffer_trimming_sec`: trim threshold (default 15s)
- `sample_rate`: 16000

### Data Flow

```
push_audio(pcm_bytes)
  └─ convert PCM16LE → float32, append to audio_buffer

get_partial()
  ├─ acquire _infer_lock
  ├─ build init_prompt from last 200 chars of committed_text
  ├─ _whisper_model.transcribe(audio_buffer, initial_prompt=...)
  ├─ feed timestamped words into _HypothesisBuffer.insert()
  ├─ flush() → newly confirmed text → append to committed_text
  ├─ trim audio_buffer if > buffer_trimming_sec
  └─ return ASRResult(text=committed_text + buffer.complete(), is_partial=True)

finalize()
  ├─ transcribe remaining audio, commit all text
  └─ return ASRResult(text=full_text, is_partial=False)

detect_endpoint()
  ├─ process _vad_buffer in 512-sample chunks
  ├─ call _vad_model(chunk, 16000) → speech probability
  ├─ if prob >= threshold: _in_speech=True, reset silence
  ├─ if prob < threshold and _in_speech: accumulate silence
  └─ if silence >= STT_VAD_SILENCE_MS and not fired: return True

reset_segment()
  └─ fresh _StreamingState + reset VAD state
```

### VAD (Silero)

Per-connection state:

- `_in_speech`: bool
- `_silence_ms_accum`: float
- `_endpoint_fired`: bool
- `_vad_buffer`: accumulates audio for 512-sample chunk processing

`detect_endpoint()` processes audio in fixed 512-sample chunks (Silero requirement), returns speech probability per chunk. Endpoint fires when speech detected followed by sustained silence >= `STT_VAD_SILENCE_MS`.

## Configuration

New env vars added to `config.py`:

| Var | Type | Default | Description |
|-----|------|---------|-------------|
| `STT_WHISPER_MODEL` | str | `"large-v3-turbo"` | Model name or local path |
| `STT_WHISPER_COMPUTE_TYPE` | str | `"float16"` | CTranslate2 compute type |
| `STT_WHISPER_BEAM_SIZE` | int | `5` | Beam size for decoding |
| `STT_WHISPER_VAD_THRESHOLD` | float | `0.5` | Silero speech probability threshold |

Reuses existing: `STT_VAD_SILENCE_MS` (default 500ms).

## Integration Points

### Registry (`backends/registry.py`)

Add: `"whisper"` → `"backends.whisper.WhisperBackend"`

### App (`app.py`)

Extend startup: if `STT_BACKEND == "whisper"`, call `whisper.preload_model()`.

### Dependencies (`pyproject.toml`)

New optional extra `whisper`: `faster-whisper`, `torch`, `torchaudio`.

Install: `uv sync --group dev --extra whisper`

## Testing

### Mock Tests (`tests/mock/test_whisper_backend.py`)

Run: `STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py -v`

Mock setup patches `_get_whisper_model` and `_get_vad_model`.

**Test classes:**

- **TestVAD**: silence no trigger, speech triggers, endpoint after silence, fires once, speech resets accumulator, no endpoint without speech
- **TestHypothesisBuffer**: flush common prefix, unstable text not committed, complete returns buffered
- **TestStreamingEngine**: empty returns none, partial with audio, committed text grows, buffer trimming
- **TestTranscribe**: push_audio accumulates float32, finalize returns full text, finalize empty state
- **TestConfigureAndClose**: configure initializes, language passthrough, reset clears state, close noop

### Future CUDA Tests (`tests/whisper/`)

Integration tests with real `faster-whisper` model, real audio transcription, WebSocket client tests. To be added when CUDA environment is available.

## CLAUDE.md Updates

Add whisper backend commands, config docs, and install instructions.
