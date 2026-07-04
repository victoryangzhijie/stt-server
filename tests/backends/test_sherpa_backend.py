import array
import asyncio
import importlib.util
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from stt_server.backends.base import BackendUnavailableError
from stt_server.backends.registry import create_backend
from stt_server.config.settings import BackendDef
from stt_server.core.events import AudioChunk

SHERPA_AVAILABLE = importlib.util.find_spec("sherpa_onnx") is not None


@pytest.mark.skipif(SHERPA_AVAILABLE, reason="only valid when sherpa-onnx is NOT installed")
def test_missing_sherpa_onnx_raises_unavailable_with_extras_hint():
    with pytest.raises(BackendUnavailableError) as exc_info:
        create_backend(BackendDef(type="sherpa_onnx", options={"model_dir": "models/whatever"}))
    message = str(exc_info.value)
    assert "sherpa" in message
    assert "pip install" in message
    assert "stt-server[sherpa]" in message


def test_pcm16_to_float32_conversion_pure_stdlib_fallback():
    from stt_server.backends.sherpa.backend import pcm16_bytes_to_float32

    samples = array.array("h", [0, 16384, -16384, 32767, -32768])
    data = struct.pack(f"<{len(samples)}h", *samples)

    result = pcm16_bytes_to_float32(data, use_numpy=False)

    assert isinstance(result, list)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.5, abs=1e-4)
    assert result[2] == pytest.approx(-0.5, abs=1e-4)
    assert result[3] == pytest.approx(32767 / 32768, abs=1e-4)
    assert result[4] == pytest.approx(-1.0, abs=1e-4)


def test_pcm16_to_float32_conversion_empty_bytes():
    from stt_server.backends.sherpa.backend import pcm16_bytes_to_float32

    result = pcm16_bytes_to_float32(b"", use_numpy=False)
    assert list(result) == []


def test_find_one_missing_model_file_raises_file_not_found(tmp_path):
    from stt_server.backends.sherpa.backend import _find_one

    with pytest.raises(FileNotFoundError) as exc_info:
        _find_one(str(tmp_path), "encoder-*.onnx")
    message = str(exc_info.value)
    assert "encoder-*.onnx" in message
    assert "download_models.py" in message


def _bare_backend():
    """A SherpaBackend without going through __init__ (which requires the
    sherpa_onnx package to be importable); attributes set to post-__init__
    defaults."""
    from stt_server.backends.sherpa.backend import SherpaBackend

    backend = SherpaBackend.__new__(SherpaBackend)
    backend._model_dir = "models/unused"
    backend._num_threads = 1
    backend._pool_workers = 1
    backend._language = "en"
    backend._recognizer = None
    backend._executor = None
    return backend


async def test_create_stream_before_start_raises_runtime_error():
    from stt_server.backends.base import StreamConfig

    backend = _bare_backend()
    with pytest.raises(RuntimeError, match="not started"):
        await backend.create_stream(StreamConfig())


class _FakeRecognizer:
    def create_stream(self):
        return object()


async def test_create_stream_language_mismatch_is_ignored_without_error():
    # sherpa-onnx is per-model-language; a client-supplied language that
    # doesn't match the loaded model's capability must be a silent (debug
    # logged) no-op, never an error — OpenAI treats `language` as a hint.
    from stt_server.backends.base import StreamConfig

    backend = _bare_backend()
    backend._executor = ThreadPoolExecutor(max_workers=1)
    backend._recognizer = _FakeRecognizer()
    try:
        stream = await backend.create_stream(StreamConfig(language="zh"))
        assert stream is not None
    finally:
        backend._executor.shutdown(wait=True)


async def test_stop_with_inflight_decode_does_not_block_event_loop():
    # stop() is awaited from the FastAPI lifespan; a synchronous
    # executor.shutdown(wait=True) with an in-flight decode would starve
    # the whole loop. The heartbeat must keep ticking while stop() waits.
    backend = _bare_backend()
    backend._executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    backend._executor.submit(release.wait)  # slow "decode" in flight
    threading.Timer(0.5, release.set).start()  # unblocks even if loop stalls

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    try:
        await asyncio.wait_for(backend.stop(), timeout=5.0)
        assert ticks >= 5, "event loop was starved while stop() waited for the executor"
    finally:
        hb.cancel()
        release.set()


class _SlowFakeRecognizer:
    """One slow decode cycle per accept_waveform; always returns 'hello'."""

    def __init__(self, decode_seconds: float) -> None:
        self._decode_seconds = decode_seconds
        self._ready = False

    def is_ready(self, stream) -> bool:
        ready, self._ready = self._ready, False
        return ready

    def decode_stream(self, stream) -> None:
        time.sleep(self._decode_seconds)

    def get_result(self, stream) -> str:
        return "hello"

    def mark_ready(self) -> None:
        self._ready = True


class _FakeNativeStream:
    def __init__(self, recognizer: _SlowFakeRecognizer) -> None:
        self._recognizer = recognizer

    def accept_waveform(self, sample_rate, waveform) -> None:
        self._recognizer.mark_ready()

    def input_finished(self) -> None: ...


async def test_close_waits_for_inflight_decode_and_sentinel_is_last():
    # close() during an in-flight push_audio decode must not put the
    # end-of-stream sentinel *before* the decode's partial lands on the
    # queue (events after the sentinel are dead: the iterator has ended).
    from stt_server.backends.sherpa.backend import SherpaStream

    recognizer = _SlowFakeRecognizer(decode_seconds=0.2)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        stream = SherpaStream(recognizer, _FakeNativeStream(recognizer), executor)

        push = asyncio.create_task(
            stream.push_audio(AudioChunk(data=b"\x00\x01" * 800, ingest_ts=0.0))
        )
        await asyncio.sleep(0.05)  # decode now in flight
        await stream.close()
        await push

        async def drain():
            return [ev async for ev in stream.events()]

        events = await asyncio.wait_for(drain(), timeout=5.0)
        assert [(e.kind, e.text) for e in events] == [("partial", "hello")]
        assert stream._queue.empty(), "an event was enqueued after the sentinel"
    finally:
        executor.shutdown(wait=True)
