# NeMo ASR Backend Design

## Overview

Add a `nemo` STT backend using NeMo's cache-aware streaming with FastConformer models. Uses `CacheAwareStreamingAudioBuffer` for chunked feature extraction and Silero VAD for endpoint detection. Follows the same single-file, global-singleton pattern as existing backends.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Streaming mode | Cache-aware streaming | Designed for streaming, low latency, no redundant computation vs buffered |
| Model config | Single `STT_NEMO_MODEL` env var | Supports any NeMo streaming model name or local path |
| VAD | Silero VAD (neural) | Proven from whisper backend, torch already required by NeMo |
| Concurrency | Global singleton model + `_infer_lock` | Consistent with qwen3/whisper, cache state is per-connection |
| Inference location | In `push_audio()` | Cache-aware streaming is incremental (like qwen3), not full re-transcription (like whisper) |
| Audio preprocessing | `CacheAwareStreamingAudioBuffer` | Official NeMo API, handles chunking/features/normalization |
| File structure | Single `backends/nemo.py` | Consistent with existing pattern |

## Architecture

### File: `backends/nemo.py`

Single file implementing `ASRBackend` protocol, containing:

- `NemoBackend` class (public, implements protocol)
- `_StreamingState` class (private, per-connection state with caches)
- Global singleton model management
- Per-connection Silero VAD instances

### Global Singletons

```
_nemo_model      : NeMo ASR model                     loaded once, shared across connections
_vad_base_model  : Silero VAD model (torch)            loaded once, deepcopy'd per connection
_model_lock      : threading.Lock                      guards model loading
_infer_lock      : threading.Lock                      serializes all conformer_stream_step() calls
```

- `_get_nemo_model()`: lazy loader with double-checked locking. Loads model via `nemo_asr.models.ASRModel.from_pretrained()` (for NGC names) or `.restore_from()` (for local `.nemo` paths). Then:
  1. `model.freeze()` — disable gradient tracking
  2. `model.eval()` — set to evaluation mode (disables dropout, etc.)
  3. Convert `STT_NEMO_CHUNK_SIZE` (seconds) to encoder frames: `chunk_frames = int(chunk_size_sec / (model.cfg.preprocessor.window_stride * encoder_stride))` where `encoder_stride` is typically 4 for FastConformer
  4. `model.encoder.setup_streaming_params(chunk_size=chunk_frames, shift_size=shift_frames, left_chunks=left_chunks)`
  5. Configure decoding: set `decoding_cfg.strategy = "greedy_batch"`, `preserve_alignments = True` (for Transducer), call `model.change_decoding_strategy(decoding_cfg)`
- `_get_vad_base_model()`: lazy loader, calls `torch.hub.load('snakers4/silero-vad', 'silero_vad')` once. Per-connection instances created via `copy.deepcopy()` in `configure()`.
- `preload_model()`: eager loading at app startup (loads both NeMo model and Silero VAD base model).
- All lazy imports (`nemo.collections.asr`, `torch`) inside functions.

**Note:** Cache-aware streaming is incremental — each `conformer_stream_step()` processes only the new chunk with cached encoder context. This means inference lives in `push_audio()` (like qwen3), and `get_partial()` is a cheap read of the latest text. The `_infer_lock` serializes `push_audio()` and `finalize()` calls across all connections. All inference runs under `torch.no_grad()`.

### Streaming Engine (Cache-Aware)

#### `_StreamingState`

Per-connection streaming context:

- `streaming_buffer`: `CacheAwareStreamingAudioBuffer` instance — manages audio chunking, mel-spectrogram feature extraction, online normalization
- `cache_last_channel`: tensor — encoder channel cache state between steps (initialized via `model.encoder.get_initial_cache_state(batch_size=1)`)
- `cache_last_time`: tensor — encoder time cache (initialized via `get_initial_cache_state`)
- `cache_last_channel_len`: tensor — cache length tracker (initialized via `get_initial_cache_state`)
- `cache_pre_encode`: tensor — pre-encode cache, prepended to processed signal before each step (initialized as `torch.zeros(...)` based on model's pre-encode cache size)
- `previous_hypotheses`: decoder hypothesis state (for Transducer models, `None` initially)
- `previous_pred_out`: previous prediction output tensor, fed back into next `conformer_stream_step()` call (`None` initially)
- `current_text`: str — latest transcription text from the model
- `audio_buffer`: float32 numpy array — accumulates raw PCM input before feeding to streaming_buffer
- `chunk_size_samples`: int — number of samples per chunk (computed from `STT_NEMO_CHUNK_SIZE * sample_rate`)

**Cache initialization** (in `configure()` and `reset_segment()`):

```python
cache_last_channel, cache_last_time, cache_last_channel_len = \
    model.encoder.get_initial_cache_state(batch_size=1)
cache_pre_encode = torch.zeros(
    1, model.encoder.pre_encode_cache_size, model.cfg.preprocessor.features
)
```

#### `CacheAwareStreamingAudioBuffer` Configuration

```python
streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=asr_model,
    online_normalization=True,        # per-chunk feature normalization for streaming
    pad_and_drop_preencoded=True,     # enables caching from step 1
)
```

### Data Flow

```
push_audio(pcm_bytes)
  ├─ convert PCM16LE → float32, append to audio_buffer
  ├─ also append to _vad_buffer for VAD
  ├─ when audio_buffer >= chunk_size_samples:
  │   ├─ acquire _infer_lock
  │   ├─ feed raw audio chunk to streaming_buffer (add_signal / set raw audio)
  │   ├─ iterate streaming_buffer to get (processed_signal, processed_signal_length)
  │   │   (streaming_buffer handles mel-spectrogram extraction + normalization)
  │   ├─ prepend cache_pre_encode to processed_signal
  │   ├─ with torch.no_grad():
  │   │     pred_out, transcribed_texts,
  │   │     cache_last_channel, cache_last_time, cache_last_channel_len,
  │   │     previous_hypotheses = model.conformer_stream_step(
  │   │         processed_signal=processed_signal,
  │   │         processed_signal_length=processed_signal_length,
  │   │         cache_last_channel=cache_last_channel,
  │   │         cache_last_time=cache_last_time,
  │   │         cache_last_channel_len=cache_last_channel_len,
  │   │         keep_all_outputs=False,
  │   │         previous_hypotheses=previous_hypotheses,
  │   │         previous_pred_out=previous_pred_out,
  │   │         drop_extra_pre_encoded=pad_and_drop_preencoded,
  │   │         return_transcription=True,
  │   │     )
  │   ├─ update all cache state: cache_last_channel, cache_last_time,
  │   │   cache_last_channel_len, previous_hypotheses, previous_pred_out
  │   ├─ update cache_pre_encode from processed_signal tail
  │   ├─ update current_text from transcribed_texts[0]
  │   └─ release _infer_lock
  └─ repeat if multiple chunks accumulated

get_partial()
  └─ cheap read of current_text → return ASRResult(is_partial=True)
     (returns None if current_text is empty)

finalize()
  ├─ acquire _infer_lock
  ├─ flush remaining audio in buffer (pad to chunk_size if needed)
  ├─ feed to streaming_buffer, iterate for (processed_signal, processed_signal_length)
  ├─ prepend cache_pre_encode
  ├─ with torch.no_grad():
  │     run final conformer_stream_step with keep_all_outputs=True
  ├─ update current_text
  └─ return ASRResult(text=current_text, is_partial=False, is_endpoint=True)

detect_endpoint()
  ├─ process _vad_buffer in 512-sample chunks (Silero VAD v5 at 16kHz)
  ├─ call self._vad_model(chunk, 16000) → speech probability
  ├─ if prob >= threshold: _in_speech=True, reset silence
  ├─ if prob < threshold and _in_speech: accumulate silence
  └─ if silence >= STT_VAD_SILENCE_MS and not fired: return True

reset_segment()
  └─ reinitialize _StreamingState:
      - fresh caches via model.encoder.get_initial_cache_state(batch_size=1)
      - fresh cache_pre_encode (zeros)
      - previous_hypotheses = None, previous_pred_out = None
      - new CacheAwareStreamingAudioBuffer instance
      - current_text = ""
      + reset VAD state (_in_speech, _silence_ms_accum, _endpoint_fired, _vad_buffer)
      + deepcopy fresh VAD model
```

### VAD (Silero)

**Per-connection VAD model instances.** Same pattern as whisper backend. Silero VAD maintains internal LSTM hidden state (`h`, `c`) that is updated on each forward pass. A base model is loaded once globally via `_get_vad_base_model()`, then each `NemoBackend` instance creates its own clone via `copy.deepcopy()` in `configure()` to get independent LSTM state.

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
| `STT_NEMO_MODEL` | str | `"stt_en_fastconformer_transducer_large_streaming"` | Model name (NGC) or local path (.nemo) |
| `STT_NEMO_CHUNK_SIZE` | float | `0.34` | Chunk duration in seconds. Converted to encoder frames at model load time via `int(chunk_size_sec / (window_stride * encoder_stride))` |
| `STT_NEMO_SHIFT_SIZE` | float | `0.34` | Shift/hop between chunks in seconds. Converted to frames same as chunk_size |
| `STT_NEMO_LEFT_CHUNKS` | int | `2` | Number of left context chunks for encoder attention |
| `STT_NEMO_VAD_THRESHOLD` | float | `0.5` | Silero speech probability threshold |

Reuses existing: `STT_VAD_SILENCE_MS` (default 500ms) for silence endpoint duration.

**Note:** The existing `STT_VAD_THRESHOLD` (int, default 300) is qwen3-specific (RMS energy threshold) and is NOT used by the nemo backend. The nemo backend uses `STT_NEMO_VAD_THRESHOLD` (float, 0.0-1.0) for Silero's probability-based detection.

**Language:** The `language` parameter from `configure()` is accepted for protocol compliance but not used. NeMo streaming models are language-specific by model choice (e.g., `stt_en_*` for English). To change language, change `STT_NEMO_MODEL`.

## Integration Points

### Registry (`backends/registry.py`)

Add: `"nemo"` → `"backends.nemo.NemoBackend"`

### App (`app.py`)

Extend startup: if `STT_BACKEND == "nemo"`, call `nemo.preload_model()`.

### Dependencies (`pyproject.toml`)

New optional extra `nemo`: `nemo_toolkit[asr]`, `torch`.

(`nemo_toolkit[asr]` pulls in all NeMo ASR dependencies including `omegaconf`, `hydra`, `sentencepiece`, etc.)

Install: `uv sync --group dev --extra nemo`

## Testing

### Mock Tests (`tests/mock/test_nemo_backend.py`)

Run: `STT_BACKEND=mock .venv/bin/python -m pytest tests/mock/test_nemo_backend.py -v`

Mock setup patches `_get_nemo_model`, `_get_vad_base_model`, and `CacheAwareStreamingAudioBuffer` constructor. The mock NeMo model must provide:
- `conformer_stream_step()` returning `(pred_out, transcribed_texts, cache_last_channel, cache_last_time, cache_last_channel_len, previous_hypotheses)`
- `encoder.get_initial_cache_state(batch_size=1)` returning 3 zero tensors
- `encoder.setup_streaming_params()` as a no-op
- `encoder.pre_encode_cache_size` as an integer
- `cfg.preprocessor.sample_rate`, `cfg.preprocessor.window_stride`, `cfg.preprocessor.features`

**Test classes:**

- **TestVAD**: silence no trigger, speech triggers, endpoint after silence, fires once, speech resets accumulator, no endpoint without speech
- **TestStreamingEngine**: empty returns none, push_audio accumulates float32, inference triggers at chunk_size, get_partial returns text, multiple chunks accumulate text, cache state updated after step
- **TestFinalize**: returns full text, empty state, calls conformer_stream_step with keep_all_outputs=True
- **TestConfigureAndClose**: configure initializes caches, reset clears all state + reinitializes caches, close noop

### Future CUDA Tests (`tests/nemo/`)

Integration tests with real NeMo model, real audio transcription, WebSocket client tests. To be added when CUDA environment is available.

## CLAUDE.md Updates

Add nemo backend commands, config docs, and install instructions.
