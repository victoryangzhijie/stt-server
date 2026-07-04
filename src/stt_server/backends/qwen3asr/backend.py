"""Qwen3-ASR backend via the official `qwen-asr` inference framework (vLLM backend).

`qwen_asr` (and its heavy `vllm`/`torch` dependencies) is imported lazily —
only inside `Qwen3AsrBackend.__init__` (via `importlib.util.find_spec`) and
`Qwen3AsrBackend._build_model` (the real `from qwen_asr import Qwen3ASRModel`)
— so importing this module, and `stt_server.backends` as a whole, never
requires the optional `qwen3asr` extra to be installed.

Research (2026-07-03, against the real QwenLM/Qwen3-ASR GitHub README and
`qwen_asr/inference/qwen3_asr.py` source, and the real PyPI `qwen-asr`
project page — see the task report for links):

* PyPI package: **`qwen-asr`** (import name `qwen_asr`), currently at 0.0.6.
  `pip install 'qwen-asr[vllm]'` pulls in `vllm==0.14.0` (the package's own
  pinned `extra == "vllm"` requirement) plus the framework's own
  `transformers`/`accelerate`/etc. This module targets the vLLM backend only
  (required for streaming — see below), so the `qwen3asr` extra in
  `pyproject.toml` is `qwen-asr[vllm]>=0.0.6`.

* Engine init: `Qwen3ASRModel.LLM(model=..., gpu_memory_utilization=...,
  max_inference_batch_size=..., max_new_tokens=...)` is a **synchronous**
  wrapper: internally it does `from vllm import LLM as vLLM; llm =
  vLLM(model=model, **kwargs)` — i.e. vLLM's offline, blocking `LLM` class,
  *not* `AsyncLLMEngine`. `model.generate(...)` calls made through it (both
  in `transcribe()` and in the streaming methods below) are ordinary
  blocking Python calls that occupy the calling thread until vLLM's
  internal batching scheduler returns.

  This contradicts the plan's working assumption that "vLLM's
  AsyncLLMEngine is itself async" and that decodes could simply be awaited
  in-loop with `max_concurrent` as an `asyncio.Semaphore`. The real,
  verified API forces the opposite: like `FunasrBackend`/`SherpaBackend`,
  every decode call MUST run on a background thread (`run_in_executor`) to
  avoid blocking the event loop, and `max_concurrent` is used exactly as
  `pool_workers` is in those two backends — the size of a shared
  `ThreadPoolExecutor` bounding how many `generate()` calls can be in
  flight at once.

* Native streaming interface: the framework exposes real per-utterance
  streaming state machinery — `model.init_streaming_state(context=...,
  language=..., chunk_size_sec=...)`, `model.streaming_transcribe(pcm16k,
  state)`, `model.finish_streaming_transcribe(state)` — documented as
  "vLLM backend only" in the README's Streaming Inference section and
  implemented in `qwen_asr/inference/qwen3_asr.py`. This is the *actual*
  documented adapter strategy (not a hand-rolled re-decode-growing-buffer
  loop): `streaming_transcribe` internally buffers incoming PCM and, each
  time a full `chunk_size_sec` chunk has accumulated, re-feeds the entire
  accumulated utterance audio (`state.audio_accum`) through the model and
  mutates `state.text`/`state.language` in place with the new cumulative
  hypothesis (REPLACE semantics, exactly as the plan anticipated for the
  stabilizer). `state.chunk_id` increments by exactly one each time a
  chunk-boundary decode actually runs, which is how this adapter detects
  "did this push_audio cross a redecode boundary" (`state.chunk_id` before
  vs. after the call) without re-implementing the framework's own
  buffering.

  `redecode_interval_ms` (this backend's config knob) is passed straight
  through as `chunk_size_sec = redecode_interval_ms / 1000.0` to
  `init_streaming_state` — the framework's own cadence knob *is* our
  redecode cadence, so there is no separate buffer/stride bookkeeping here
  (contrast with `FunasrBackend`, which must compute its own byte stride).

  `finish_streaming_transcribe` mirrors this: if `state.buffer` is already
  empty (the last `streaming_transcribe` call consumed everything exactly
  on a chunk boundary) it is a documented no-op that returns `state`
  unmodified rather than issuing a redundant decode call. This adapter
  still always emits exactly one FINAL event on `finalize()` regardless —
  using whatever `state.text` currently holds — matching the fixed
  plugin contract, not the underlying framework's decode-call count.

* Sample rate: the framework's streaming API is hardcoded to 16 kHz mono
  PCM (`ASRStreamingState`/`SAMPLE_RATE` in the source), consistent with
  this server's `AudioChunk` convention; `pcm16_bytes_to_float32` handles
  the PCM16->float32 conversion shared with the other backends.

If a future implementer needs to re-verify any of the above: the GitHub
README section "Streaming Inference" links
`examples/example_qwen3_asr_vllm_streaming.py`, which is the executable
form of everything described here.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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

_SENTINEL = None


class Qwen3AsrStream(SttStream):
    """One streaming-ASR session (the framework's `ASRStreamingState`) per
    utterance.

    All calls into `model.streaming_transcribe(...)` /
    `model.finish_streaming_transcribe(...)` happen on the backend's shared
    ThreadPoolExecutor via `run_in_executor` (see module docstring: the
    real `Qwen3ASRModel.LLM` wrapper is a synchronous/blocking vLLM offline
    engine, not `AsyncLLMEngine`); an asyncio.Lock serializes them
    per-instance since a single `ASRStreamingState` is mutated in place and
    is not safe for concurrent use, and to preserve the same
    close()/push_audio() race-safety invariant used by `FunasrStream` /
    `SherpaStream`: close() waits out any in-flight decode so its partial
    (enqueued inside the lock) always lands before the end-of-stream
    sentinel.
    """

    def __init__(self, model: Any, executor: ThreadPoolExecutor, state: Any) -> None:
        self._model = model
        self._executor = executor
        self._state = state
        self._last_text = ""
        self._audio_ms = 0.0
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._done = False
        self._lock = asyncio.Lock()

    def _streaming_transcribe_sync(self, samples: Any) -> bool:
        """Feed `samples` into the framework's own internal chunk buffer;
        returns True iff a chunk-boundary decode actually ran (detected via
        `state.chunk_id` before/after), i.e. `state.text` was just updated."""
        chunk_id_before = self._state.chunk_id
        self._model.streaming_transcribe(samples, self._state)
        return self._state.chunk_id != chunk_id_before

    def _finish_streaming_transcribe_sync(self) -> None:
        self._model.finish_streaming_transcribe(self._state)

    async def push_audio(self, chunk: AudioChunk) -> None:
        if self._done:
            return
        samples = pcm16_bytes_to_float32(chunk.data)
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._done:  # close()/finalize() won the lock while we waited
                return
            self._audio_ms += chunk.duration_ms
            decoded = await loop.run_in_executor(
                self._executor, self._streaming_transcribe_sync, samples
            )
            # Dedup on changed text (matches SherpaStream): a redecode tick
            # over pure silence legitimately leaves state.text unchanged,
            # and the stabilizer counts partials from actual events, so a
            # duplicate would inflate its stability confirmation count.
            if decoded and self._state.text != self._last_text:
                self._last_text = self._state.text
                await self._queue.put(
                    BackendEvent(
                        kind="partial", text=self._state.text, audio_time_ms=self._audio_ms
                    )
                )

    async def events(self) -> AsyncIterator[BackendEvent]:
        while True:
            ev = await self._queue.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def finalize(self) -> None:
        if self._done:
            return
        self._done = True
        loop = asyncio.get_running_loop()
        async with self._lock:
            await loop.run_in_executor(self._executor, self._finish_streaming_transcribe_sync)
        await self._queue.put(
            BackendEvent(kind="final", text=self._state.text, audio_time_ms=self._audio_ms)
        )
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        async with self._lock:
            if not self._done:
                self._done = True
                await self._queue.put(_SENTINEL)


@register_backend("qwen3asr")
class Qwen3AsrBackend(SttBackend):
    name = "qwen3asr"
    # languages: Qwen3-ASR supports 52 languages/dialects (see the README's
    # language table); ("en", "zh") are listed here as the two this repo's
    # fixtures/docs exercise, not an exhaustive capability limit.
    capabilities = BackendCapabilities(
        streaming=True,
        languages=("en", "zh"),
        native_endpointing=False,
        batch_decode=True,
    )

    def __init__(
        self,
        model: str = "Qwen/Qwen3-ASR-0.6B",
        gpu_memory_utilization: float = 0.8,
        max_concurrent: int = 8,
        language: str | None = None,
        redecode_interval_ms: float = 480,
    ) -> None:
        if redecode_interval_ms <= 0:
            raise ValueError(
                f"redecode_interval_ms must be > 0 (got {redecode_interval_ms!r})"
            )
        if importlib.util.find_spec("qwen_asr") is None:
            raise BackendUnavailableError(
                "qwen_asr is not installed; pip install 'stt-server[qwen3asr]'"
            )
        self._model_name = model
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_concurrent = max_concurrent
        self._language = language
        self._redecode_interval_ms = redecode_interval_ms
        self._model_instance: Any = None
        self._executor: ThreadPoolExecutor | None = None

    def _build_model(self) -> Any:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel.LLM(
            model=self._model_name,
            gpu_memory_utilization=self._gpu_memory_utilization,
        )

    async def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent)
        self._model_instance = await asyncio.to_thread(self._build_model)

    async def stop(self) -> None:
        if self._executor is not None:
            # shutdown(wait=True) blocks until in-flight decodes finish; run
            # it off-loop so awaiting stop() (e.g. from the FastAPI lifespan)
            # never starves the event loop.
            await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._executor = None
        self._model_instance = None

    async def create_stream(self, cfg: StreamConfig) -> Qwen3AsrStream:
        if self._executor is None or self._model_instance is None:
            raise RuntimeError("backend not started: await start() before create_stream()")
        # init_streaming_state() makes no GPU/vLLM generate() call — it
        # renders the HF processor's chat-template prompt (CPU, cheap) and
        # allocates a zeroed numpy buffer — so it's cheap enough to call
        # directly on the event loop rather than via the executor.
        state = self._model_instance.init_streaming_state(
            language=cfg.language or self._language,
            chunk_size_sec=self._redecode_interval_ms / 1000.0,
        )
        return Qwen3AsrStream(self._model_instance, self._executor, state)
