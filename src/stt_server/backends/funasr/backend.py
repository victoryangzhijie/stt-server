"""FunASR Paraformer streaming backend.

`funasr` (and its heavy `torch`/`torchaudio` dependencies) is imported
lazily — only inside `FunasrBackend.__init__` (via `importlib.util.find_spec`)
and `FunasrBackend._build_model` (the real `from funasr import AutoModel`) —
so importing this module, and `stt_server.backends` as a whole, never
requires the optional `funasr` extra to be installed.

FunASR streaming chunk convention (`chunk_size=[0, 10, 5]`, verified against
the upstream FunASR README streaming example): each element is a count of
60 ms units. `chunk_size[1]` (10) is the size of one decode chunk: 10 * 60ms
= 600 ms of audio per `generate()` call. In *samples* at 16 kHz that's
`chunk_size[1] * 960` (960 = 60ms * 16 samples/ms) = 9600 samples, i.e.
19200 bytes of PCM16. The brief's own comment ("chunk_size[1] * 960 bytes")
conflates samples and bytes; this module computes the stride in samples
first and only converts to bytes (`* 2`) at the end, which is the
dimensionally-correct version of the same 19200-byte result for the default
`[0, 10, 5]` config.
"""

from __future__ import annotations

import asyncio
import importlib.util
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

logger = structlog.get_logger(__name__)

_SENTINEL = None

_SAMPLES_PER_60MS_AT_16K = 960  # 60ms * 16 samples/ms
_BYTES_PER_SAMPLE = 2  # pcm16


def _extract_text(result: Any) -> str:
    """Pull the incremental text out of a `model.generate(...)` result.

    FunASR streaming `generate()` calls return a list of result dicts (one
    per input utterance batch element — always length <= 1 here since we
    feed one chunk at a time); on pure-silence chunks it may return an empty
    list `[]` or a dict with an empty `"text"` — both are handled as "no new
    text" rather than raised.
    """
    if not result:
        return ""
    first = result[0]
    if not isinstance(first, dict):
        return ""
    return first.get("text", "") or ""


class FunasrStream(SttStream):
    """One FunASR streaming session (cache dict) per utterance.

    All calls into `model.generate(...)` happen on the backend's shared
    ThreadPoolExecutor via `run_in_executor`; an asyncio.Lock serializes
    them per-instance (mirrors `SherpaStream`) since the FunASR `cache` dict
    is mutated in place by `generate()` and is not safe for concurrent use,
    and to preserve the same close()/push_audio() race-safety invariant:
    close() waits out any in-flight decode so its partial (enqueued inside
    the lock) always lands before the end-of-stream sentinel.
    """

    def __init__(
        self,
        model: Any,
        executor: ThreadPoolExecutor,
        chunk_size: tuple[int, int, int],
    ) -> None:
        self._model = model
        self._executor = executor
        self._chunk_size = chunk_size
        chunk_stride_samples = chunk_size[1] * _SAMPLES_PER_60MS_AT_16K
        self._chunk_stride_bytes = chunk_stride_samples * _BYTES_PER_SAMPLE
        if self._chunk_stride_bytes <= 0:
            # A chunk_size like (0, 0, 5) (config typo) would otherwise make
            # push_audio's whole-chunk while-loop spin forever on the event
            # loop the moment any audio arrives.
            raise ValueError(
                f"chunk_size[1] must be > 0 (got chunk_size={chunk_size!r}); "
                "it is the decode chunk length in 60 ms units"
            )
        self._cache: dict[str, Any] = {}
        self._buffer = bytearray()
        self._hypothesis = ""
        self._audio_ms = 0.0
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._done = False
        self._lock = asyncio.Lock()

    def _generate_sync(self, chunk_bytes: bytes, is_final: bool) -> str:
        samples = pcm16_bytes_to_float32(chunk_bytes)
        result = self._model.generate(
            input=samples,
            cache=self._cache,
            is_final=is_final,
            chunk_size=list(self._chunk_size),
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        return _extract_text(result)

    async def push_audio(self, chunk: AudioChunk) -> None:
        if self._done:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self._done:  # close()/finalize() won the lock while we waited
                return
            self._buffer += chunk.data
            self._audio_ms += chunk.duration_ms
            # Note: when a single push carries multiple whole decode chunks,
            # every partial emitted below tags the same end-of-push
            # audio_time_ms (accumulated above, once per push) rather than a
            # per-decode-chunk timestamp. Unreachable in practice today — the
            # session feeds 30 ms frames, far below one 600 ms stride — but a
            # future caller batching large pushes shouldn't be surprised that
            # intra-push partials aren't individually time-resolved.
            while len(self._buffer) >= self._chunk_stride_bytes:
                piece = bytes(self._buffer[: self._chunk_stride_bytes])
                del self._buffer[: self._chunk_stride_bytes]
                text_increment = await loop.run_in_executor(
                    self._executor, self._generate_sync, piece, False
                )
                if text_increment:
                    self._hypothesis += text_increment
                    await self._queue.put(
                        BackendEvent(
                            kind="partial", text=self._hypothesis, audio_time_ms=self._audio_ms
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
            remaining = bytes(self._buffer)
            self._buffer.clear()
            text_increment = await loop.run_in_executor(
                self._executor, self._generate_sync, remaining, True
            )
        if text_increment:
            self._hypothesis += text_increment
        await self._queue.put(
            BackendEvent(kind="final", text=self._hypothesis, audio_time_ms=self._audio_ms)
        )
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        async with self._lock:
            if not self._done:
                self._done = True
                await self._queue.put(_SENTINEL)


@register_backend("funasr")
class FunasrBackend(SttBackend):
    name = "funasr"
    capabilities = BackendCapabilities(
        streaming=True, languages=("zh", "en"), native_endpointing=False
    )

    def __init__(
        self,
        model: str = "paraformer-zh-streaming",
        pool_workers: int = 4,
        chunk_size: tuple[int, int, int] = (0, 10, 5),
    ) -> None:
        # Validate chunk_size before the availability check: a config typo
        # should fail loudly at construction (server startup) regardless of
        # whether the funasr package happens to be installed, not at the
        # first create_stream() mid-session.
        if len(chunk_size) != 3 or chunk_size[1] <= 0:
            raise ValueError(
                f"chunk_size must be a 3-tuple with chunk_size[1] > 0 "
                f"(got {tuple(chunk_size)!r}); chunk_size[1] is the decode "
                "chunk length in 60 ms units"
            )
        if importlib.util.find_spec("funasr") is None:
            raise BackendUnavailableError(
                "funasr is not installed; pip install 'stt-server[funasr]'"
            )
        self._model_name = model
        self._pool_workers = pool_workers
        self._chunk_size = tuple(chunk_size)
        self._model_instance: Any = None
        self._executor: ThreadPoolExecutor | None = None

    def _build_model(self) -> Any:
        from funasr import AutoModel

        return AutoModel(model=self._model_name)

    async def start(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self._pool_workers)
        self._model_instance = await asyncio.to_thread(self._build_model)

    async def stop(self) -> None:
        if self._executor is not None:
            # shutdown(wait=True) blocks until in-flight decodes finish; run
            # it off-loop so awaiting stop() (e.g. from the FastAPI lifespan)
            # never starves the event loop.
            await asyncio.to_thread(self._executor.shutdown, wait=True)
            self._executor = None
        self._model_instance = None

    async def create_stream(self, cfg: StreamConfig) -> FunasrStream:
        if self._executor is None or self._model_instance is None:
            raise RuntimeError("backend not started: await start() before create_stream()")
        if cfg.language is not None and cfg.language not in self.capabilities.languages:
            # FunASR's loaded model is pinned to whatever `model` was
            # configured at startup — there is no per-utterance language
            # knob to honor. OpenAI treats `language` as a hint, so this is
            # a debug-level no-op, never an error.
            logger.debug(
                "stream.language_hint_ignored",
                requested=cfg.language,
                supported=self.capabilities.languages,
            )
        return FunasrStream(self._model_instance, self._executor, self._chunk_size)
