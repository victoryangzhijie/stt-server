"""sherpa-onnx streaming Zipformer backend.

`sherpa_onnx` (and `numpy`) are imported lazily — only inside
`SherpaBackend.__init__` / the PCM conversion helper — so that importing
this module, and `stt_server.backends` as a whole, never requires the
optional `sherpa` extra to be installed.
"""

from __future__ import annotations

import asyncio
import glob
import importlib.util
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
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

__all__ = [
    "SherpaBackend",
    "SherpaStream",
    "pcm16_bytes_to_float32",  # re-exported from stt_server.backends._audio
]

logger = structlog.get_logger(__name__)

_SENTINEL = None


def _find_one(model_dir: str, pattern: str) -> str:
    """Find exactly one file matching `pattern` under `model_dir`, preferring
    full-precision weights over quantized (`.int8.onnx`) ones when both are
    present (the streaming zipformer release ships both variants).

    Raises FileNotFoundError (not BackendUnavailableError) so a missing
    model *file* is clearly distinguishable from a missing *package* —
    mirroring the silero VAD convention. The lifespan startup handler
    catches Exception broadly, so backend.start() failures are handled
    identically either way."""
    matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
    non_int8 = [m for m in matches if ".int8." not in os.path.basename(m)]
    candidates = non_int8 or matches
    if not candidates:
        raise FileNotFoundError(
            f"no file matching {pattern!r} found under model_dir={model_dir!r}; "
            "run `python scripts/download_models.py sherpa-zipformer-en`"
        )
    return candidates[0]


class SherpaStream(SttStream):
    """One native sherpa-onnx OnlineStream per utterance.

    All calls into the native recognizer/stream happen on the backend's
    shared ThreadPoolExecutor via `run_in_executor`; an asyncio.Lock
    serializes them per-instance since a single native stream is not safe
    for concurrent decode calls.
    """

    def __init__(self, recognizer: Any, native_stream: Any, executor: ThreadPoolExecutor) -> None:
        self._recognizer = recognizer
        self._stream = native_stream
        self._executor = executor
        self._audio_ms = 0.0
        self._last_text = ""
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._done = False
        self._lock = asyncio.Lock()

    def _decode_sync(self) -> str:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        return self._recognizer.get_result(self._stream)

    def _accept_and_decode_sync(self, samples: Any) -> str:
        self._stream.accept_waveform(16000, samples)
        return self._decode_sync()

    def _finalize_sync(self) -> str:
        self._stream.input_finished()
        return self._decode_sync()

    async def push_audio(self, chunk: AudioChunk) -> None:
        if self._done:
            return
        samples = pcm16_bytes_to_float32(chunk.data)
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._done:  # close()/finalize() won the lock while we waited
                return
            text = await loop.run_in_executor(
                self._executor, self._accept_and_decode_sync, samples
            )
            self._audio_ms += chunk.duration_ms
            if text != self._last_text:
                self._last_text = text
                await self._queue.put(
                    BackendEvent(kind="partial", text=text, audio_time_ms=self._audio_ms)
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
            text = await loop.run_in_executor(self._executor, self._finalize_sync)
        await self._queue.put(
            BackendEvent(kind="final", text=text, audio_time_ms=self._audio_ms)
        )
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        # Lock-aware: wait out any in-flight push_audio decode so its partial
        # (enqueued inside the lock) always lands *before* the end-of-stream
        # sentinel — otherwise the event would be enqueued after the iterator
        # ended, onto a dead queue.
        async with self._lock:
            if not self._done:
                self._done = True
                await self._queue.put(_SENTINEL)


@register_backend("sherpa_onnx")
class SherpaBackend(SttBackend):
    name = "sherpa_onnx"
    capabilities = BackendCapabilities(streaming=True, languages=("en",), native_endpointing=False)

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 4,
        pool_workers: int = 8,
        language: str = "en",
    ) -> None:
        if importlib.util.find_spec("sherpa_onnx") is None:
            raise BackendUnavailableError(
                "sherpa_onnx is not installed; pip install 'stt-server[sherpa]'"
            )
        self._model_dir = model_dir
        self._num_threads = num_threads
        self._pool_workers = pool_workers
        self._language = language
        self._recognizer: Any = None
        self._executor: ThreadPoolExecutor | None = None

    def _build_recognizer(self) -> Any:
        import sherpa_onnx

        tokens = os.path.join(self._model_dir, "tokens.txt")
        encoder = _find_one(self._model_dir, "encoder-*.onnx")
        decoder = _find_one(self._model_dir, "decoder-*.onnx")
        joiner = _find_one(self._model_dir, "joiner-*.onnx")
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=self._num_threads,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=False,
        )

    async def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self._pool_workers)
        self._recognizer = await asyncio.to_thread(self._build_recognizer)

    async def stop(self) -> None:
        if self._executor is not None:
            # shutdown(wait=True) blocks until in-flight decodes finish; run
            # it off-loop so awaiting stop() (e.g. from the FastAPI lifespan)
            # never starves the event loop.
            await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._executor = None
        self._recognizer = None

    async def create_stream(self, cfg: StreamConfig) -> SherpaStream:
        if self._executor is None or self._recognizer is None:
            raise RuntimeError("backend not started: await start() before create_stream()")
        if cfg.language is not None and cfg.language not in self.capabilities.languages:
            # sherpa-onnx's recognizer is fixed to whatever model was loaded
            # at start() — there is no per-utterance language knob to honor.
            # OpenAI treats `language` as a hint, so this is a debug-level
            # no-op, never an error.
            logger.debug(
                "stream.language_hint_ignored",
                requested=cfg.language,
                supported=self.capabilities.languages,
            )
        loop = asyncio.get_running_loop()
        native_stream = await loop.run_in_executor(
            self._executor, self._recognizer.create_stream
        )
        return SherpaStream(self._recognizer, native_stream, self._executor)
