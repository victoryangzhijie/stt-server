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
| Concurrency | Global whisper model singleton + `_infer_lock` | Consistent with qwen3, horizontal scaling via workers |
| Audio buffer | Float32 numpy accumulation | OnlineASRProcessor needs full buffer for re-transcription |
| File structure | Single `backends/whisper.py` | Consistent with qwen3 pattern |
| Branch strategy | Feature branch for development | Isolate work from master |

## Architecture

### File: `backends/whisper.py`

Single file implementing `ASRBackend` protocol, containing:

- `WhisperBackend` class (public, implements protocol)
- `_HypothesisBuffer` class (private, text stability tracking)
- `_StreamingState` class (private, per-connection state)
- Global singleton whisper model management
- Per-connection Silero VAD instances

### Global Singletons

```
_whisper_model   : WhisperModel (faster-whisper)     loaded once, shared across connections
_vad_base_model  : Silero VAD model (torch)           loaded once, deepcopy'd per connection
_model_lock      : threading.Lock                     guards whisper + VAD base model loading
_infer_lock      : threading.Lock                     serializes all whisper inference calls
```

- `_get_whisper_model()`: lazy loader with double-checked locking
- `_get_vad_base_model()`: lazy loader, calls `torch.hub.load('snakers4/silero-vad', 'silero_vad')` once. Per-connection instances created via `copy.deepcopy(_vad_base_model)` in `configure()` to get independent LSTM state.
- `preload_model()`: eager loading at app startup (loads both whisper model and Silero VAD base model, so first connection doesn't pay loading cost)
- All lazy imports (`faster_whisper`, `torch`, `numpy`) inside functions

**Note:** Unlike qwen3 where `push_audio()` does incremental inference, whisper's architecture requires re-transcribing the full audio buffer each time. Therefore inference lives in `get_partial()` and `finalize()`, while `push_audio()` only appends to the buffer. This means `_infer_lock` serializes `get_partial()`/`finalize()` calls across all connections — partials are delivered sequentially. This is acceptable because whisper transcription is the bottleneck regardless, and horizontal scaling (multiple workers) addresses concurrency.

### Streaming Engine

Ported from `whisper_streaming/whisper_online.py` OnlineASRProcessor.

#### `_HypothesisBuffer`

Tracks text stability across consecutive transcription iterations using word-level local agreement:

- `committed`: list of confirmed `(timestamp, word)` tuples — words that appeared in the same position across 2+ consecutive transcriptions
- `buffer`: words from previous iteration, awaiting confirmation by the next iteration
- `new`: latest transcription output (timestamped words)
- `insert(new_words, offset)`: compare `new` against `buffer` word-by-word. Words matching at the same position are confirmed (moved to `committed`). Then `buffer = new_words[len(matched):]`, `new = new_words`.
- `flush()`: find the longest word-level common prefix between `buffer` and `new`. Matching words are confirmed stable — move to `committed` and return them. Remaining words stay in `buffer`.
- `complete()`: return `buffer` words as unconfirmed text (used for partial display)

#### `_StreamingState`

Per-connection streaming context:

- `audio_buffer`: growing float32 numpy array
- `hypothesis`: `_HypothesisBuffer` instance
- `committed_text`: confirmed text so far
- `buffer_trimming_sec`: trim threshold (default 15s)
- `sample_rate`: 16000
- `language`: language code passed to `faster-whisper` transcribe (e.g. `"en"`, `"zh"`). `"auto"` maps to `None` (auto-detection).

### Data Flow

```
push_audio(pcm_bytes)
  └─ convert PCM16LE → float32, append to audio_buffer
     also append float32 chunk to _vad_buffer for VAD processing

get_partial()
  ├─ acquire _infer_lock
  ├─ build init_prompt from last 200 chars of committed_text
  ├─ _whisper_model.transcribe(audio_buffer, initial_prompt=..., word_timestamps=True)
  ├─ extract timestamped words from segments
  ├─ feed into _HypothesisBuffer.insert()
  ├─ flush() → newly confirmed words → append to committed_text
  ├─ trim audio_buffer if > buffer_trimming_sec (see Buffer Trimming below)
  └─ return ASRResult(text=committed_text + hypothesis.complete(), is_partial=True)

finalize()
  ├─ acquire _infer_lock
  ├─ transcribe remaining audio with init_prompt
  ├─ commit ALL text (confirmed + buffered + new)
  └─ return ASRResult(text=full_text, is_partial=False)

detect_endpoint()
  ├─ process _vad_buffer in 512-sample chunks (Silero VAD v5 at 16kHz)
  ├─ call self._vad_model(chunk, 16000) → speech probability
  ├─ if prob >= threshold: _in_speech=True, reset silence
  ├─ if prob < threshold and _in_speech: accumulate silence
  └─ if silence >= STT_VAD_SILENCE_MS and not fired: return True

reset_segment()
  └─ fresh _StreamingState + reset VAD state (_in_speech, _silence_ms_accum, _endpoint_fired, _vad_buffer)
```

### Buffer Trimming

When `audio_buffer` length exceeds `buffer_trimming_sec * sample_rate` samples:

1. Find the timestamp of the last committed word in `_HypothesisBuffer.committed`
2. Convert that timestamp to a sample offset in `audio_buffer`
3. Trim `audio_buffer` to keep only audio from that offset onward
4. Record the trim offset so future transcription timestamps can be adjusted (added back)
5. Clear `committed` entries that fall before the trim point — their text is already in `committed_text`

This prevents unbounded memory growth while preserving enough audio context for accurate re-transcription.

### VAD (Silero)

**Per-connection VAD model instances.** Silero VAD maintains internal LSTM hidden state (`h`, `c`) that is updated on each forward pass. Sharing a single model across connections would corrupt this state. A base model is loaded once globally via `_get_vad_base_model()`, then each `WhisperBackend` instance creates its own clone via `copy.deepcopy()` in `configure()` to get independent LSTM state.

Per-connection state:

- `_vad_model`: per-connection Silero VAD model instance
- `_in_speech`: bool
- `_silence_ms_accum`: float
- `_endpoint_fired`: bool
- `_vad_buffer`: accumulates audio for 512-sample chunk processing (Silero VAD v5 requires exactly 512 samples at 16kHz)

`detect_endpoint()` processes audio in fixed 512-sample chunks, returns speech probability per chunk. Endpoint fires when speech detected followed by sustained silence >= `STT_VAD_SILENCE_MS`.

## Configuration

New env vars added to `config.py`:

| Var | Type | Default | Description |
|-----|------|---------|-------------|
| `STT_WHISPER_MODEL` | str | `"large-v3-turbo"` | Model name or local path |
| `STT_WHISPER_COMPUTE_TYPE` | str | `"float16"` | CTranslate2 compute type |
| `STT_WHISPER_BEAM_SIZE` | int | `5` | Beam size for decoding |
| `STT_WHISPER_VAD_THRESHOLD` | float | `0.5` | Silero speech probability threshold |

Reuses existing: `STT_VAD_SILENCE_MS` (default 500ms) for silence endpoint duration.

**Note:** The existing `STT_VAD_THRESHOLD` (int, default 300) is qwen3-specific (RMS energy threshold) and is NOT used by the whisper backend. The whisper backend uses `STT_WHISPER_VAD_THRESHOLD` (float, 0.0-1.0) for Silero's probability-based detection.

## Integration Points

### Registry (`backends/registry.py`)

Add: `"whisper"` → `"backends.whisper.WhisperBackend"`

### App (`app.py`)

Extend startup: if `STT_BACKEND == "whisper"`, call `whisper.preload_model()`.

### Dependencies (`pyproject.toml`)

New optional extra `whisper`: `faster-whisper`, `torch`.

(`torchaudio` not needed — audio conversion is done via numpy, Silero VAD only requires `torch`.)

Install: `uv sync --group dev --extra whisper`

## Testing

### Mock Tests (`tests/mock/test_whisper_backend.py`)

Run: `STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_whisper_backend.py -v`

Mock setup patches `_get_whisper_model` and creates mock Silero VAD instances.

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
