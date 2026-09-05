"""NVIDIA Nemotron 3.5 ASR Streaming 0.6B backend (NeMo cache-aware FastConformer).

`nemo_toolkit` (and its transitive `torch`) is imported lazily — only inside
`NemotronBackend.__init__` (an `importlib.util.find_spec("nemo")` gate, no
import) and inside `_build_model` / `_TorchOps` — so importing this module,
and `stt_server.backends` as a whole, never requires the optional `nemotron`
extra. The language helpers below are pure stdlib and unit-tested without it.

Model: `nvidia/nemotron-3.5-asr-streaming-0.6b` — a 24-layer cache-aware
streaming FastConformer encoder (8x subsampling, 128 mel bins, d_model 1024)
with an RNNT decoder and a one-hot *language prompt* vector; 40 locales plus
an `auto` language-ID prompt. 16 kHz mono.


Research trail (NeMo 3.0.0 wheel, read from source)
---------------------------------------------------

Everything below was verified by reading `nemo_toolkit==3.0.0`
(`nemo_toolkit-3.0.0-py3-none-any.whl`, the ASR-only split of
NVIDIA-NeMo/NeMo) together with the checkpoint's own `model_config.yaml` and
the repo's `processor_config.json`. Wheel paths are cited so a future
implementer can re-check each claim. Five findings changed the design:

1. **`conformer_stream_step` takes features, never raw audio.**
   `nemo/collections/asr/parts/mixins/mixins.py:602-615` — the first
   parameter is `processed_signal: Tensor` of shape `(B, n_mels=128,
   n_frames)`; there is no `input_signal`/`audio_signal` parameter anywhere
   in the cache-aware path. So this backend runs `model.preprocessor` itself
   (a private clone with `dither=0.0`, `pad_to=0`, following NeMo's own
   recipe at `streaming_utils.py:1716-1726`) and feeds features in. Because
   the mel STFT runs with `center=True` (`features.py:378-384`), chunk-wise
   feature extraction is only exact if at least `n_fft//2 = 256` samples of
   the *previous* audio are prepended and the corresponding leading frames
   are dropped; we prepend `LOOKBACK_SAMPLES = 320` (2 whole frames).
   NVIDIA's own live service uses a single 160-sample frame of look-back
   (`nemo/agents/voice_agent/pipecat/services/nemo/utils.py:100`), which is
   less than 256 and therefore slightly approximate at the chunk seam.

2. **The language prompt is model-global, and it is applied *only* inside
   `model.conformer_stream_step`.** `PromptStreamingMixin.set_inference_prompt`
   (`mixins.py:949-969`) writes a single int attribute
   `model._inference_prompt_index`; `_apply_prompt_to_encoded`
   (`mixins.py:971-1001`) reads it and is invoked at `mixins.py:674`, i.e.
   *after* `encoder.cache_aware_stream_step`. Consequences: (a) NVIDIA's live
   reference (`services/nemo/streaming_asr.py:256`) calls
   `encoder.cache_aware_stream_step` directly and therefore silently skips
   the prompt — do not copy it; (b) per-stream languages can only be served
   by setting the global prompt immediately before each step under a
   process-wide lock, which is what `_NemoEngine.step` does; (c) if the
   prompt is never set, `mixins.py:978-979` silently decodes with *no*
   language conditioning (not "auto") — a quality regression with no error,
   so the prompt is always set, including during warm-up.

3. **Per-call `prompt_vectors` do not work in NeMo 3.0.0's cache-aware
   pipeline.** `CacheAwareRNNTInferenceWrapper.stream_step` accepts
   `prompt_vectors` (`model_wrappers/cache_aware_rnnt_inference_wrapper.py:152`)
   and forwards it to `execute_step` (`:197`), whose body (`:79-140`) never
   reads it. Batching several sessions into one step is therefore impossible
   while their languages differ, and the "nicer" wrapper API would drop the
   prompt entirely. `conformer_stream_step` + a global lock is the only
   working design.

4. **`strip_lang_tags` defaults to `False`** (`rnnt_decoding.py:1900-1902`)
   and the shipped checkpoint's `decoding:` block does not set it, so
   `hyp.text` arrives *with* `<xx-XX>` locale tags (they are real vocabulary
   tokens; the tokenizer lists all 40 in `extra_special_tokens`). We keep
   NeMo's stripping **off** on purpose and strip in this adapter
   (`split_lang_tag`), because the tag is the model's language-ID output and
   is what `BackendEvent.language` reports on the FINAL event.

5. **`att_context_size` right context is restricted to `{0, 3, 6, 13}`.**
   The checkpoint offers exactly `[[56,3],[56,0],[56,6],[56,13]]`
   (`model_config.yaml:602-611`; `processor_config.json`'s
   `supported_num_lookahead_tokens` is `[3, 0, 6, 13]`).
   `set_default_att_context_size` (`conformer_encoder.py:960-975`) only
   *warns* for an unlisted value, so e.g. `[56, 1]` would load and run
   out-of-distribution rather than fail. The constructor rejects it instead.

Supporting facts used by the step arithmetic, all from the same wheel:

* `streaming_cfg` for this checkpoint (`att_context_style: chunked_limited`,
  `subsampling_factor: 8`, `sampling_frames = [1, 8]`,
  `pre_encode_cache_size = [0, 9]`, `conformer_encoder.py:977-1085`):
  `chunk_size = [1+8R, 8+8R]` feature frames, `shift_size` equal to it
  (`cache_drop_size == 0`), `valid_out_len = R+1` encoder frames,
  `drop_extra_pre_encoded = 2`. One feature frame is 160 samples (10 ms;
  `features.py:405-408`), one encoder frame is 80 ms.
* We always prepend the 9 `pre_encode_cache_size[1]` frames of left feature
  context (zeros on the first step) and always pass
  `drop_extra_pre_encoded = 2` — NeMo's `pad_and_drop_preencoded=True`
  scheme (`streaming_utils.py:1603-1604, 1644-1652`). Every step is then
  geometrically identical: `ceil((17+8R)/8) - 2 = R+1` output frames.
* `previous_pred_out` is CTC-only (`mixins.py:702-707`) → always `None`.
  For an RNNT model the returned `all_hyp_or_transcribed_texts` (slot 1) is
  the same `list[Hypothesis]` as `best_hyp` (slot 5), not `list[str]`
  (`mixins.py:718-736`); we read slot 5.
* `hyp.text` is **cumulative** across steps because `partial_hypotheses` are
  merged in place (`rnnt_greedy_decoding.py:828-834`) — REPLACE semantics
  for the stabilizer, exactly like the sherpa/qwen3asr backends.
* `greedy_batch` supports `partial_hypotheses` only on the `loop_labels`
  path (`rnnt_greedy_decoding.py:828-834` vs `:844-845`, which raises);
  `loop_labels` defaults to `True` (`:615, :2473`) so we leave it alone.
  `fused_batch_size = -1` disables the joint's training-time fused loss/WER
  (`rnnt_decoding.py:277, 1029-1033`), matching NVIDIA's live service
  (`streaming_asr.py:175`). `preserve_alignments`/`compute_timestamps` stay
  off: we read `hyp.text`, and alignments grow unboundedly over a long
  utterance.
* `StreamingEncoder.cache_aware_stream_step` **mutates the shared**
  `encoder.streaming_cfg.drop_extra_pre_encoded` for the duration of the
  call and restores it afterwards (`mixins/streaming.py:53-74`) — a second,
  prompt-independent reason the process-wide decode lock is mandatory.
* Loading: `ASRModel.from_pretrained(model_name=...)` resolves the HF id to
  `<name>.nemo` and goes through `huggingface_hub.hf_hub_download`
  (`nemo/core/classes/common.py:1130-1177`), so `HF_ENDPOINT`/`HF_HOME`
  apply; `ASRModel.restore_from(path, map_location=...)` is the local-file
  form. Both return the checkpoint's own target class,
  `EncDecRNNTBPEModelWithPrompt` (`common.py:786-800`). We deliberately do
  **not** call `EncDecRNNTBPEModelWithPrompt.restore_from` directly: that
  override just delegates back to `EncDecRNNTBPEModel.restore_from`
  (`rnnt_bpe_models_prompt.py:109-141`), so the plain `ASRModel` entry point
  is the documented path and yields the same object.


UNVERIFIED (pending the A10 run)
--------------------------------

Nothing in this module has ever executed against the real model — there is
no CUDA GPU in this development environment. `docs/nemotron_a10_runbook.md`
§10 is the full checklist; the code-level unknowns are:

* That the published `nemo_toolkit>=3.0.0` wheel behaves as its source reads
  here at runtime, and that `ASRModel.from_pretrained` resolves the HF id
  without falling back to a whole-repo `snapshot_download` behind a mirror
  (`common.py:1166` calls `HfApi().file_exists` first).
* fp32 vs `amp=True`. No `NotImplementedError` about cache-aware compute
  dtypes exists anywhere in the 3.0.0 wheel, so the widely repeated
  "cache-aware models are float32-only" claim is *unconfirmed for 3.0.0*.
  We ship `amp=False` and follow NVIDIA's pattern for the flag: fp32 weights
  and fp32 caches, autocast only around the step
  (`cache_aware_rnnt_inference_wrapper.py:59-60, 182-186`).
* Numeric equivalence of our chunk-wise features against whole-utterance
  features. The 320-sample look-back argument is sound for `center=True`
  STFT with `n_fft=512`, but the mel + log path has not been diffed.
* `greedy.use_cuda_graph_decoder` (default `True`,
  `rnnt_greedy_decoding.py:2474`) interacting with per-stream
  `partial_hypotheses` under the shared lock. If concurrent streams show
  cross-contamination on GPU day, turn it off first.
* Per-step latency, the real concurrency ceiling under the global lock,
  warm-up cost / lazy CUDA-graph capture, measured zh CER, and whether the
  `<xx-XX>` tag really appears in streaming output.
* `device: cpu` / `device: mps`: accepted by the constructor, never run.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import re
import threading
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import structlog

from stt_server.backends._audio import pcm16_bytes_to_float32
from stt_server.backends.base import (
    BackendCapabilities,
    BackendEvent,
    BackendUnavailableError,
    StreamConfig,
    SttBackend,
    SttStream,
)
from stt_server.backends.registry import register_backend
from stt_server.core.events import AudioChunk

logger = structlog.get_logger(__name__)

_SENTINEL = None

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # pcm16
#: Samples per mel frame: `window_stride` 0.01 s * 16 kHz (checkpoint
#: `model_config.yaml:570`, and `features.py:405-408`'s `seq_len // hop`).
SAMPLES_PER_FRAME = 160
#: Encoder sub-sampling factor (`config.json: subsampling_factor`), i.e. the
#: number of feature frames per encoder output frame, and also
#: `sampling_frames[1]` — the shortest chunk NeMo's own buffer will decode
#: (`streaming_utils.py:1637-1638`).
SUBSAMPLING_FACTOR = 8
#: Raw-audio left overlap handed to the preprocessor on every chunk. Must be
#: >= `n_fft // 2` = 256 (the `center=True` STFT pad) and a whole number of
#: frames; 320 = 2 frames is the smallest such value. See finding 1.
LOOKBACK_SAMPLES = 320
#: Right-context values the checkpoint was trained with (`model_config.yaml:
#: 602-611`). Anything else only *warns* inside NeMo, so we reject it here.
SUPPORTED_RIGHT_CONTEXTS = (0, 3, 6, 13)


# --------------------------------------------------------------------------
# Language helpers (pure stdlib: no torch, no nemo, importable anywhere)
# --------------------------------------------------------------------------

#: The 40 locales the model card documents, in its own three quality tiers:
#: transcription-ready, then broad-coverage, then adaptation-ready. The
#: checkpoint's `prompt_dictionary` additionally carries a few aliases
#: (`en`, `enGB`, `zh-TW`, `zh-ZH`, ...) that are deliberately not accepted
#: here — this tuple is the documented, supported surface.
SUPPORTED_LOCALES: tuple[str, ...] = (
    # transcription-ready
    "en-US",
    "en-GB",
    "es-US",
    "es-ES",
    "fr-FR",
    "fr-CA",
    "it-IT",
    "pt-BR",
    "pt-PT",
    "nl-NL",
    "de-DE",
    "tr-TR",
    "ru-RU",
    "ar-AR",
    "hi-IN",
    "ja-JP",
    "ko-KR",
    "vi-VN",
    "uk-UA",
    # broad-coverage
    "pl-PL",
    "sv-SE",
    "cs-CZ",
    "nb-NO",
    "da-DK",
    "bg-BG",
    "fi-FI",
    "hr-HR",
    "sk-SK",
    "zh-CN",
    "hu-HU",
    "ro-RO",
    "et-EE",
    # adaptation-ready
    "el-GR",
    "lt-LT",
    "lv-LV",
    "mt-MT",
    "sl-SI",
    "he-IL",
    "th-TH",
    "nn-NO",
)

#: `auto` is prompt index 101 in the checkpoint's dictionary; it is the
#: per-utterance automatic language-identification prompt, and the default.
AUTO_LANGUAGE = "auto"

#: Bare ISO 639-1 code -> the locale we pick for it. Note the model's prompt
#: dictionary has **no** bare `zh` key (only `zh-CN`/`zh-TW`/`zh-ZH`), so
#: this mapping is required, not merely convenient.
_DEFAULT_LOCALE_BY_ISO639: dict[str, str] = {
    "en": "en-US",
    "zh": "zh-CN",
    "es": "es-US",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "tr": "tr-TR",
    "ru": "ru-RU",
    "ar": "ar-AR",
    "hi": "hi-IN",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "vi": "vi-VN",
    "uk": "uk-UA",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "cs": "cs-CZ",
    "nb": "nb-NO",
    "no": "nb-NO",
    "da": "da-DK",
    "bg": "bg-BG",
    "fi": "fi-FI",
    "hr": "hr-HR",
    "sk": "sk-SK",
    "hu": "hu-HU",
    "ro": "ro-RO",
    "et": "et-EE",
    "el": "el-GR",
    "lt": "lt-LT",
    "lv": "lv-LV",
    "mt": "mt-MT",
    "sl": "sl-SI",
    "he": "he-IL",
    "th": "th-TH",
    "nn": "nn-NO",
}

#: ISO 639-1 code -> canonical English language name, used both for
#: `BackendEvent.language` and for the English-name aliases below.
_LANGUAGE_NAME_BY_ISO639: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "pl": "Polish",
    "sv": "Swedish",
    "cs": "Czech",
    "nb": "Norwegian",
    "nn": "Norwegian Nynorsk",
    "da": "Danish",
    "bg": "Bulgarian",
    "fi": "Finnish",
    "hr": "Croatian",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "et": "Estonian",
    "el": "Greek",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "sl": "Slovenian",
    "he": "Hebrew",
    "th": "Thai",
}


def _build_name_aliases() -> dict[str, str]:
    """English language names (plus a few common synonyms) -> locale.

    OpenAI clients routinely send `language="Chinese"` / `"english"` rather
    than an ISO code, so those spellings must resolve.
    """
    aliases = {
        name.casefold(): _DEFAULT_LOCALE_BY_ISO639[iso]
        for iso, name in _LANGUAGE_NAME_BY_ISO639.items()
        if iso in _DEFAULT_LOCALE_BY_ISO639
    }
    aliases.update(
        {
            "mandarin": "zh-CN",
            "chinese (simplified)": "zh-CN",
            "simplified chinese": "zh-CN",
            "putonghua": "zh-CN",
            "norwegian bokmal": "nb-NO",
            "norwegian bokmål": "nb-NO",
            "castilian": "es-ES",
            "brazilian portuguese": "pt-BR",
        }
    )
    return aliases


_LOCALE_BY_ENGLISH_NAME: dict[str, str] = _build_name_aliases()

_LOCALE_BY_CASEFOLD: dict[str, str] = {loc.casefold(): loc for loc in SUPPORTED_LOCALES}

#: The `<xx-XX>` locale tag the model emits in its transcript. Byte-identical
#: in shape to NeMo's own default `lang_tag_pattern`
#: (`rnnt_decoding.py:705`, `r'\s*<[a-z]{2}-[A-Z]{2}>'`), with the locale
#: captured so we can report it as the detected language.
LANG_TAG_RE = re.compile(r"\s*<([a-z]{2}-[A-Z]{2})>")


def normalize_language(code: str | None) -> str:
    """Normalize a client/config language value to a prompt-dictionary key.

    Returns a canonical locale (`"zh-CN"`) or `"auto"`. Accepts `None`,
    `""`, `"auto"`, a full locale in any case and with either separator
    (`zh-cn`, `ZH_CN`), a bare ISO 639-1 code (`zh`), or an English language
    name (`chinese`, `mandarin`). An unrecognized value is **never** an
    error — OpenAI treats `language` as a hint — it falls back to `"auto"`.
    """
    if code is None:
        return AUTO_LANGUAGE
    text = code.strip()
    if not text or text.casefold() == AUTO_LANGUAGE:
        return AUTO_LANGUAGE

    candidate = text.replace("_", "-").casefold()
    locale = _LOCALE_BY_CASEFOLD.get(candidate)
    if locale is not None:
        return locale
    locale = _DEFAULT_LOCALE_BY_ISO639.get(candidate)
    if locale is not None:
        return locale
    return _LOCALE_BY_ENGLISH_NAME.get(text.casefold(), AUTO_LANGUAGE)


def locale_to_language_name(locale: str | None) -> str | None:
    """`"zh-CN"` -> `"Chinese"`, `"en-GB"` -> `"English"`.

    Falls back to the locale itself for anything unmapped, and returns
    `None` for `None`/`"auto"` (there is no detected language to report).
    """
    if not locale or locale == AUTO_LANGUAGE:
        return None
    return _LANGUAGE_NAME_BY_ISO639.get(locale.split("-")[0].casefold(), locale)


def split_lang_tag(text: str, strip: bool = True) -> tuple[str, str | None]:
    """Split a raw hypothesis into `(text, detected_locale)`.

    The model emits its language-ID decision as a `<xx-XX>` vocabulary token
    inside the transcript (finding 4). The **last** tag wins — the model
    re-emits it as the utterance grows, and the latest one is its current
    decision. With `strip=False` the text is returned untouched, but the
    locale is still extracted.
    """
    matches = LANG_TAG_RE.findall(text)
    locale = matches[-1] if matches else None
    if strip:
        text = LANG_TAG_RE.sub("", text).strip()
    return text, locale


def chunk_samples_for(att_context_size: Iterable[int]) -> int:
    """Samples of new audio consumed by one cache-aware step.

    `streaming_cfg.chunk_size[1] = sampling_frames[1] * (1 + R) = 8 + 8R`
    feature frames (`conformer_encoder.py:1040-1046`), each 160 samples. So
    `[56, 6]` -> 8960 samples (560 ms) and `[56, 3]` -> 5120 (320 ms).
    """
    _left, right = tuple(att_context_size)
    return SUBSAMPLING_FACTOR * (1 + int(right)) * SAMPLES_PER_FRAME


def _capability_languages() -> tuple[str, ...]:
    """ISO 639-1 codes of every supported locale, deduped, `en`/`zh` first."""
    ordered = ["en", "zh"]
    for locale in SUPPORTED_LOCALES:
        iso = locale.split("-")[0]
        if iso not in ordered:
            ordered.append(iso)
    return tuple(ordered)


# --------------------------------------------------------------------------
# Tensor plumbing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    """Per-step sizes, all read from `encoder.streaming_cfg` at load time.

    Never hardcoded from the model card: `_build_model` derives every field
    from the model that was actually loaded, so a different checkpoint or a
    different `att_context_size` cannot silently desynchronize the buffering
    from what the encoder expects.
    """

    chunk_frames: int  # streaming_cfg.chunk_size[1]  == 8 + 8R
    pre_encode_frames: int  # streaming_cfg.pre_encode_cache_size[1] == 9
    drop_extra_pre_encoded: int  # streaming_cfg.drop_extra_pre_encoded == 2
    n_mels: int  # preprocessor.features == 128
    hop_samples: int = SAMPLES_PER_FRAME
    min_frames: int = SUBSAMPLING_FACTOR
    lookback_samples: int = LOOKBACK_SAMPLES

    @property
    def chunk_samples(self) -> int:
        return self.chunk_frames * self.hop_samples

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_samples * BYTES_PER_SAMPLE


class _TorchOps:
    """The complete set of tensor primitives the streaming step needs.

    Isolating them behind this tiny adapter is what lets the unit tests
    exercise the *real* step logic — look-back handling, feature slicing,
    cache threading, prompt-under-lock, the final flush — on a box with no
    torch installed, by substituting a list-based fake. `torch` is imported
    in `__init__`, i.e. only once the backend is really starting.
    """

    def __init__(self, device: str, amp: bool = False) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._device_type = device.split(":")[0]
        self._amp = amp

    # --- audio (1-D float32 waveform in [-1, 1)) ---------------------------
    def audio_tensor(self, samples: Any) -> Any:
        # `samples` is a numpy float32 array, or a plain list when numpy is
        # absent (see backends/_audio.py); `as_tensor` handles both.
        return self._torch.as_tensor(samples, dtype=self._torch.float32).to(self._device)

    def zeros_audio(self, n: int) -> Any:
        return self._torch.zeros(n, dtype=self._torch.float32, device=self._device)

    def cat_audio(self, left: Any, right: Any) -> Any:
        return self._torch.cat([left, right])

    def tail_audio(self, audio: Any, n: int) -> Any:
        return audio[-n:].clone()

    def audio_len(self, audio: Any) -> int:
        return int(audio.shape[-1])

    def batch_audio(self, audio: Any) -> Any:
        return audio.unsqueeze(0)

    def length_tensor(self, n: int) -> Any:
        return self._torch.tensor([n], dtype=self._torch.int64, device=self._device)

    # --- features ((1, n_mels, T) float32) ---------------------------------
    def cat_feats(self, left: Any, right: Any) -> Any:
        return self._torch.cat([left, right], dim=-1)

    def head_feats(self, feats: Any, n: int) -> Any:
        return feats[:, :, :n].contiguous()

    def tail_feats(self, feats: Any, n: int) -> Any:
        return feats[:, :, -n:].contiguous()

    def zeros_feats(self, like: Any, n_mels: int, n: int) -> Any:
        return self._torch.zeros((1, n_mels, n), dtype=like.dtype, device=like.device)

    def feats_width(self, feats: Any) -> int:
        return int(feats.shape[-1])

    # --- contexts ----------------------------------------------------------
    def inference_mode(self) -> Any:
        return self._torch.inference_mode()

    def autocast(self) -> Any:
        if not self._amp:
            return contextlib.nullcontext()
        # NVIDIA's own pattern: fp32 weights and fp32 caches, autocast only
        # around the step (cache_aware_rnnt_inference_wrapper.py:59-60,
        # 182-186). UNVERIFIED on real hardware.
        return self._torch.autocast(
            device_type=self._device_type, dtype=self._torch.bfloat16, enabled=True
        )


class _StreamState:
    """Everything one utterance owns. Nothing here is shared between streams.

    Session isolation lives entirely in this object: the encoder caches, the
    RNNT partial hypothesis, the raw-audio look-back and the left feature
    context. The only shared objects are the model itself and the decode
    lock.
    """

    __slots__ = (
        "locale",
        "cache_last_channel",
        "cache_last_time",
        "cache_last_channel_len",
        "prev_hyps",
        "pcm_lookback",
        "feat_lookback",
        "text",
        "steps",
    )

    def __init__(self, locale: str, caches: tuple[Any, Any, Any], pcm_lookback: Any) -> None:
        self.locale = locale
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = caches
        self.prev_hyps: Any = None
        self.pcm_lookback = pcm_lookback
        self.feat_lookback: Any = None
        self.text = ""
        self.steps = 0


class _NemoEngine:
    """Owns the model and turns `(state, samples) -> cumulative text`.

    Both `new_state` and `step` take the **process-wide** decode lock
    (`threading.Lock`, not asyncio, so exclusion survives cancellation of an
    in-flight decode: the executor thread keeps holding it until the step
    returns). It is mandatory for two independent reasons — the language
    prompt is a single model-global int (finding 2), and
    `cache_aware_stream_step` mutates the shared
    `encoder.streaming_cfg.drop_extra_pre_encoded` for the duration of the
    call (`mixins/streaming.py:53-74`). Consequence: all model compute for
    all sessions is serialized; the executor buys concurrency in feature
    extraction and queueing, not in the step.
    """

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        ops: Any,
        geometry: _Geometry,
        decode_lock: threading.Lock,
    ) -> None:
        self._model = model
        self._preprocessor = preprocessor
        self._ops = ops
        self._geometry = geometry
        self._decode_lock = decode_lock

    @property
    def geometry(self) -> _Geometry:
        return self._geometry

    def new_state(self, locale: str) -> _StreamState:
        """Allocate one utterance's caches. Called on the executor, under the
        lock, on the first step — never on the event loop, so `create_stream`
        stays cheap and never touches CUDA from the loop thread."""
        ops = self._ops
        with self._decode_lock:
            caches = self._model.encoder.get_initial_cache_state(batch_size=1)
        return _StreamState(locale, tuple(caches), ops.zeros_audio(self._geometry.lookback_samples))

    def _features(self, state: _StreamState, samples: Any) -> Any:
        """Raw PCM chunk -> exactly `len(samples) // hop` new feature frames.

        The look-back (finding 1) is prepended before the preprocessor runs
        and dropped afterwards. The head-slice to the *reported* frame count
        comes first because `torch.stft(center=True)` returns one extra
        column that NeMo zeroes rather than removes (`features.py:481-485`);
        taking the tail of the raw tensor would otherwise shift the window
        by one frame.
        """
        ops = self._ops
        geom = self._geometry
        audio = ops.audio_tensor(samples)
        n_new_samples = ops.audio_len(audio)
        fed = ops.cat_audio(state.pcm_lookback, audio)
        state.pcm_lookback = ops.tail_audio(audio, geom.lookback_samples)

        signal = ops.batch_audio(fed)
        n_fed = ops.audio_len(fed)
        with ops.inference_mode():
            feats, _lens = self._preprocessor(
                input_signal=signal, length=ops.length_tensor(n_fed)
            )
        feats = ops.head_feats(feats, n_fed // geom.hop_samples)
        return ops.tail_feats(feats, n_new_samples // geom.hop_samples)

    def step(
        self,
        state: _StreamState,
        samples: Any,
        last: bool,
        valid_samples: int | None = None,
    ) -> str:
        """One cache-aware step. Returns the cumulative hypothesis text.

        `valid_samples` is how many of `samples` are real audio rather than
        the zero padding the final flush adds to reach a full chunk; the
        feature tensor is truncated to it so `processed_signal_length` stays
        truthful.
        """
        ops = self._ops
        geom = self._geometry
        feats = self._features(state, samples)
        if last:
            # NeMo's own buffer drops any chunk shorter than
            # `sampling_frames[1]` (streaming_utils.py:1637-1638), so keep at
            # least that many frames even for a very short tail.
            n_valid = len(samples) if valid_samples is None else valid_samples
            wanted = max(geom.min_frames, n_valid // geom.hop_samples)
            if ops.feats_width(feats) > wanted:
                feats = ops.head_feats(feats, wanted)

        if state.feat_lookback is None:
            pre = ops.zeros_feats(feats, geom.n_mels, geom.pre_encode_frames)
        else:
            pre = state.feat_lookback
        signal = ops.cat_feats(pre, feats)
        length = ops.length_tensor(ops.feats_width(signal))
        state.feat_lookback = ops.tail_feats(feats, geom.pre_encode_frames)

        with self._decode_lock:
            # The prompt is model-global, so it must be (re)applied inside
            # the lock immediately before the step that consumes it —
            # otherwise an interleaved stream's language wins. See finding 2.
            self._model.set_inference_prompt(state.locale)
            with ops.inference_mode(), ops.autocast():
                result = self._model.conformer_stream_step(
                    processed_signal=signal,
                    processed_signal_length=length,
                    cache_last_channel=state.cache_last_channel,
                    cache_last_time=state.cache_last_time,
                    cache_last_channel_len=state.cache_last_channel_len,
                    keep_all_outputs=last,
                    previous_hypotheses=state.prev_hyps,
                    previous_pred_out=None,  # CTC-only (mixins.py:702-707)
                    drop_extra_pre_encoded=geom.drop_extra_pre_encoded,
                    return_transcription=True,
                )
        (
            _greedy_predictions,
            _transcribed,
            state.cache_last_channel,
            state.cache_last_time,
            state.cache_last_channel_len,
            best_hyp,
        ) = result
        state.prev_hyps = best_hyp
        state.steps += 1
        state.text = best_hyp[0].text or ""
        return state.text


# --------------------------------------------------------------------------
# Stream
# --------------------------------------------------------------------------


class NemotronStream(SttStream):
    """One utterance: its own encoder caches, RNNT hypothesis and buffers.

    Raw PCM16 is accumulated byte-wise (like `FunasrStream`); each time a
    whole `chunk_samples` step's worth is available, one cache-aware step
    runs on the backend's shared ThreadPoolExecutor. Partials carry the
    model's cumulative hypothesis (REPLACE semantics) with the `<xx-XX>`
    locale tag stripped, deduped on unchanged text. An asyncio.Lock
    serializes this stream's own decodes and gives `close()` the same
    race-safety invariant as the other backends: it waits out an in-flight
    decode so that decode's partial always lands before the sentinel.
    """

    def __init__(
        self,
        engine: _NemoEngine,
        executor: ThreadPoolExecutor,
        locale: str,
        strip_lang_tags: bool = True,
        requested_language: str | None = None,
    ) -> None:
        self._engine = engine
        self._executor = executor
        self._locale = locale
        self._strip_lang_tags = strip_lang_tags
        # Non-None only when the caller pinned a language; then the FINAL
        # reports what was requested rather than what the model tagged.
        self._requested_language = requested_language
        self._geometry = engine.geometry
        self._chunk_bytes = self._geometry.chunk_bytes
        self._state: _StreamState | None = None
        self._buffer = bytearray()
        self._last_text = ""
        self._detected_locale: str | None = None
        self._audio_ms = 0.0
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._done = False
        self._lock = asyncio.Lock()

    # --- sync helpers, always run on the executor --------------------------
    def _ensure_state(self) -> _StreamState:
        if self._state is None:
            self._state = self._engine.new_state(self._locale)
        return self._state

    def _step_sync(self, chunk_bytes: bytes, last: bool) -> str:
        state = self._ensure_state()
        return self._engine.step(state, pcm16_bytes_to_float32(chunk_bytes), last)

    def _flush_sync(self, remainder: bytes) -> str:
        """Final step: zero-pad the remainder to a full chunk, decode with
        `keep_all_outputs=True` so the encoder does not truncate the tail.

        Two shortcuts, both deliberate:
        * remainder empty and at least one step already ran -> return the
          hypothesis as it stands; there is no audio left to decode.
        * no audio ever arrived -> return `""` **without touching the
          model**. An empty utterance still emits exactly one FINAL (the
          plugin contract), but it costs no GPU work and cannot fail; this
          is the documented choice over running a padded silent step.
        """
        if not remainder:
            return self._state.text if self._state is not None else ""
        valid_samples = len(remainder) // BYTES_PER_SAMPLE
        padding = self._geometry.chunk_bytes - len(remainder)
        if padding > 0:
            remainder = remainder + bytes(padding)
        state = self._ensure_state()
        samples = pcm16_bytes_to_float32(remainder)
        return self._engine.step(state, samples, True, valid_samples=valid_samples)

    # --- SttStream ---------------------------------------------------------
    async def push_audio(self, chunk: AudioChunk) -> None:
        if self._done:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._done:  # close()/finalize() won the lock while we waited
                return
            self._buffer += chunk.data
            self._audio_ms += chunk.duration_ms
            while len(self._buffer) >= self._chunk_bytes:
                piece = bytes(self._buffer[: self._chunk_bytes])
                del self._buffer[: self._chunk_bytes]
                raw = await loop.run_in_executor(self._executor, self._step_sync, piece, False)
                text = self._observe(raw)
                if text != self._last_text:
                    self._last_text = text
                    await self._queue.put(
                        BackendEvent(kind="partial", text=text, audio_time_ms=self._audio_ms)
                    )

    def _observe(self, raw: str) -> str:
        """Strip the locale tag (when configured) and remember the last one
        the model emitted as its language-ID decision."""
        text, locale = split_lang_tag(raw, strip=self._strip_lang_tags)
        if locale is not None:
            self._detected_locale = locale
        return text

    def _final_language(self) -> str | None:
        if self._requested_language is not None:
            return self._requested_language
        return locale_to_language_name(self._detected_locale)

    async def events(self) -> AsyncIterator[BackendEvent]:
        while True:
            ev = await self._queue.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def finalize(self) -> None:
        if self._done:
            return
        # Flip _done before taking the lock: that flag, not the lock scope,
        # is what makes a concurrent push_audio a no-op.
        self._done = True
        loop = asyncio.get_running_loop()
        async with self._lock:
            remainder = bytes(self._buffer)
            self._buffer.clear()
            raw = await loop.run_in_executor(self._executor, self._flush_sync, remainder)
        text = self._observe(raw)
        await self._queue.put(
            BackendEvent(
                kind="final",
                text=text,
                audio_time_ms=self._audio_ms,
                language=self._final_language(),
            )
        )
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        # Lock-aware: wait out any in-flight decode so its partial (enqueued
        # inside the lock) always lands before the end-of-stream sentinel.
        async with self._lock:
            if not self._done:
                self._done = True
                await self._queue.put(_SENTINEL)


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


@register_backend("nemotron")
class NemotronBackend(SttBackend):
    name = "nemotron"
    # "en" and "zh" first: serving both from one streaming model is this
    # backend's reason to exist. The rest are the ISO 639-1 codes of the 40
    # locales the model card documents, across three quality tiers — so the
    # tuple is a supported-language list, not an accuracy promise (zh-CN is
    # "broad-coverage"; see docs/backends.md#nemotron).
    capabilities = BackendCapabilities(
        streaming=True,
        languages=_capability_languages(),
        native_endpointing=False,
        batch_decode=False,
    )

    def __init__(
        self,
        model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        att_context_size: tuple[int, int] | list[int] = (56, 6),
        language: str | None = None,
        strip_lang_tags: bool = True,
        device: str = "cuda",
        amp: bool = False,
        max_concurrent: int = 8,
        warmup: bool = True,
    ) -> None:
        # Validate every numeric/enumerated option BEFORE the availability
        # gate: a config typo must fail loudly at construction (server
        # startup) whether or not the nemotron extra happens to be
        # installed, mirroring FunasrBackend's chunk_size ordering.
        att = tuple(att_context_size)
        if len(att) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in att):
            raise ValueError(
                f"att_context_size must be two ints [left, right] (got {att!r})"
            )
        left, right = att
        if left <= 0:
            raise ValueError(f"att_context_size left context must be > 0 (got {left!r})")
        if right not in SUPPORTED_RIGHT_CONTEXTS:
            raise ValueError(
                f"att_context_size right context must be one of "
                f"{list(SUPPORTED_RIGHT_CONTEXTS)} (got {right!r}); these are the "
                "lookaheads the checkpoint was trained with — NeMo only warns for "
                "others and would decode out of distribution"
            )
        if not device.startswith(("cuda", "cpu", "mps")):
            raise ValueError(
                f"device must start with 'cuda', 'cpu' or 'mps' (got {device!r})"
            )
        if max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be > 0 (got {max_concurrent!r})")
        if importlib.util.find_spec("nemo") is None:
            raise BackendUnavailableError(
                "nemo_toolkit is not installed; pip install 'stt-server[nemotron]'"
            )

        self._model_name = model
        self._att_context_size = (int(left), int(right))
        self._language = language
        self._locale = normalize_language(language)
        self._strip_lang_tags = strip_lang_tags
        self._device = device
        self._amp = amp
        self._max_concurrent = max_concurrent
        self._warmup_enabled = warmup
        self._model_instance: Any = None
        self._engine: _NemoEngine | None = None
        self._executor: ThreadPoolExecutor | None = None
        # Process-wide serialization of prompt-set + model step. A
        # threading.Lock (not asyncio) so exclusion survives asyncio
        # cancellation of an in-flight decode: the executor thread holds it
        # until the step returns. See _NemoEngine for why it is mandatory.
        self._decode_lock = threading.Lock()

    # --- model construction (the only place nemo/torch are imported) -------
    def _build_model(self) -> _NemoEngine:
        import nemo.collections.asr as nemo_asr
        import torch
        from omegaconf import OmegaConf, open_dict

        map_location = torch.device(self._device)
        if os.path.exists(self._model_name):
            # Local `.nemo` file. `ASRModel.restore_from` resolves the
            # checkpoint's own target class (EncDecRNNTBPEModelWithPrompt),
            # which is exactly what that class's own restore_from override
            # delegates to — see the module docstring.
            model = nemo_asr.models.ASRModel.restore_from(
                self._model_name, map_location=map_location
            )
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._model_name, map_location=map_location
            )

        model.eval()
        model.to(self._device)
        freeze = getattr(model, "freeze", None)
        if callable(freeze):
            # ModelPT.freeze() == eval() + requires_grad_(False). Optional:
            # eval() plus inference_mode() is already sufficient.
            freeze()

        if not getattr(model, "concat", False):
            # `concat` is set by initialize_prompt_feature(); without it,
            # _apply_prompt_to_encoded is a no-op and every language
            # (including "auto") silently decodes unconditioned.
            logger.warning(
                "nemotron.prompt_feature_missing",
                model=self._model_name,
                detail="model.concat is False; language prompting will be a no-op",
            )

        model.encoder.set_default_att_context_size(list(self._att_context_size))

        decoding_cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy_batch"  # loop_labels=True -> partial_hypotheses OK
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = False
            decoding_cfg.fused_batch_size = -1  # disables joint.fuse_loss_wer (training path)
            if "greedy" in decoding_cfg:
                decoding_cfg.greedy.max_symbols = 10
        model.change_decoding_strategy(decoding_cfg)

        # Keep NeMo's own tag stripping OFF: the `<xx-XX>` tag is the
        # model's language-ID output and this adapter needs to read it
        # (finding 4). The method only exists on recent decodings.
        set_strip = getattr(model.decoding, "set_strip_lang_tags", None)
        if callable(set_strip):
            set_strip(False)

        # Private preprocessor clone, per NeMo's own recipe
        # (streaming_utils.py:1716-1726): dither off, no padding.
        pre_cfg = OmegaConf.create(
            OmegaConf.to_container(model._cfg.preprocessor, resolve=True)
        )
        OmegaConf.set_struct(pre_cfg, False)
        pre_cfg.dither = 0.0
        pre_cfg.pad_to = 0
        preprocessor = model.from_config_dict(pre_cfg).to(self._device)
        preprocessor.eval()

        geometry = self._read_geometry(model)
        self._validate_prompt_dictionary(model)
        self._model_instance = model
        logger.info(
            "nemotron.model_loaded",
            model=self._model_name,
            device=self._device,
            att_context_size=list(self._att_context_size),
            chunk_ms=geometry.chunk_samples / SAMPLE_RATE * 1000.0,
            chunk_frames=geometry.chunk_frames,
            pre_encode_frames=geometry.pre_encode_frames,
            drop_extra_pre_encoded=geometry.drop_extra_pre_encoded,
        )
        return _NemoEngine(
            model=model,
            preprocessor=preprocessor,
            ops=_TorchOps(self._device, self._amp),
            geometry=geometry,
            decode_lock=self._decode_lock,
        )

    def _read_geometry(self, model: Any) -> _Geometry:
        """Derive the per-step sizes from the loaded encoder, never from the
        model card. `set_default_att_context_size` has already re-run
        `setup_streaming_params`, so `streaming_cfg` is authoritative."""
        scfg = model.encoder.streaming_cfg
        chunk = scfg.chunk_size
        chunk_frames = int(chunk[1] if isinstance(chunk, (list, tuple)) else chunk)
        pre = scfg.pre_encode_cache_size
        pre_frames = int(pre[1] if isinstance(pre, (list, tuple)) else pre)
        expected = chunk_samples_for(self._att_context_size)
        actual = chunk_frames * SAMPLES_PER_FRAME
        if actual != expected:
            # Not fatal: the encoder wins. But it means this checkpoint's
            # geometry differs from what chunk_samples_for() predicts, which
            # every doc and test in this repo assumes.
            logger.warning(
                "nemotron.unexpected_chunk_geometry",
                expected_samples=expected,
                actual_samples=actual,
            )
        return _Geometry(
            chunk_frames=chunk_frames,
            pre_encode_frames=pre_frames,
            drop_extra_pre_encoded=int(scfg.drop_extra_pre_encoded),
            n_mels=int(model.cfg.preprocessor.features),
        )

    def _validate_prompt_dictionary(self, model: Any) -> None:
        """Warn (never raise) about locales this checkpoint cannot prompt.

        `set_inference_prompt` raises ValueError for an unknown key
        (`mixins.py:962-967`), which would surface mid-session on the first
        step. Surfacing it at load time instead makes a checkpoint/locale
        mismatch visible in the startup log.
        """
        prompt_dict = {}
        try:
            prompt_dict = dict(model.cfg.model_defaults.get("prompt_dictionary") or {})
        except Exception:  # pragma: no cover - defensive: config shape varies
            logger.warning("nemotron.prompt_dictionary_unreadable", model=self._model_name)
            return
        if not prompt_dict:
            logger.warning("nemotron.prompt_dictionary_empty", model=self._model_name)
            return
        missing = [
            key for key in (AUTO_LANGUAGE, *SUPPORTED_LOCALES) if key not in prompt_dict
        ]
        if missing:
            logger.warning(
                "nemotron.prompt_dictionary_missing_locales",
                missing=missing,
                available=len(prompt_dict),
            )
        if self._locale not in prompt_dict:
            logger.warning(
                "nemotron.configured_locale_unavailable",
                configured=self._locale,
                detail="streams using it will fail at the first step",
            )

    def _warmup(self) -> None:
        """Push ~1.2 s of silence through the real step path, final flush
        included, so the first client request does not pay CUDA graph
        capture / kernel autotune. Best effort: a failure is logged and
        swallowed (start() still succeeds; the first request then pays the
        cold start, which is the pre-existing behavior)."""
        engine = self._engine
        if engine is None:  # pragma: no cover - start() sets it first
            return
        try:
            geom = engine.geometry
            state = engine.new_state(self._locale)
            silence = [0.0] * geom.chunk_samples
            steps = max(1, int(1200 * SAMPLE_RATE / 1000 / geom.chunk_samples))
            for _ in range(steps):
                engine.step(state, silence, False)
            engine.step(state, silence, True)  # exercise the keep_all_outputs flush too
        except Exception:
            logger.warning("nemotron.warmup_failed", exc_info=True)

    # --- SttBackend --------------------------------------------------------
    async def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent)
        self._engine = await asyncio.to_thread(self._build_model)
        if self._warmup_enabled:
            await asyncio.to_thread(self._warmup)

    async def stop(self) -> None:
        if self._executor is not None:
            # shutdown(wait=True) blocks until in-flight decodes finish; run
            # it off-loop so awaiting stop() (e.g. from the FastAPI lifespan)
            # never starves the event loop.
            await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._executor = None
        self._engine = None
        self._model_instance = None
        self._release_cuda_memory()

    def _release_cuda_memory(self) -> None:
        """Best-effort `torch.cuda.empty_cache()` after dropping the model.

        Guarded and lazily imported: stop() must work on a box where torch
        was never imported (an ML-free test run) and where CUDA is absent.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - purely opportunistic
            logger.debug("nemotron.cuda_empty_cache_skipped", exc_info=True)

    async def create_stream(self, cfg: StreamConfig) -> NemotronStream:
        if self._executor is None or self._engine is None:
            raise RuntimeError("backend not started: await start() before create_stream()")
        # StreamConfig.language overrides the constructor default for this
        # one utterance; this is the only backend where that genuinely
        # switches the model's language rather than being a no-op hint.
        requested = cfg.language if cfg.language is not None else self._language
        locale = normalize_language(requested)
        asked = requested.strip() if isinstance(requested, str) else ""
        if asked and locale == AUTO_LANGUAGE and asked.casefold() != AUTO_LANGUAGE:
            # OpenAI treats `language` as a hint: an unrecognized value falls
            # back to automatic language ID, never an error.
            logger.debug(
                "stream.language_hint_ignored",
                requested=requested,
                supported=self.capabilities.languages,
            )
        return NemotronStream(
            self._engine,
            self._executor,
            locale=locale,
            strip_lang_tags=self._strip_lang_tags,
            # An explicitly selected language is reported back as-is on the
            # FINAL; only in auto mode is the model's own `<xx-XX>` tag the
            # source of truth.
            requested_language=locale_to_language_name(locale),
        )
