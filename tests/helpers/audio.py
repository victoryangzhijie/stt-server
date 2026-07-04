"""Synthesize PCM16 mono 16 kHz test audio without numpy."""

import math
import struct

from stt_server.core.events import SAMPLE_RATE


def make_tone(ms: int, freq: float = 440.0, amplitude: float = 0.3) -> bytes:
    n = SAMPLE_RATE * ms // 1000
    samples = (
        int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)
    )
    return struct.pack(f"<{n}h", *samples)


def make_silence(ms: int) -> bytes:
    n = SAMPLE_RATE * ms // 1000
    return b"\x00\x00" * n


def make_wav(pcm: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
