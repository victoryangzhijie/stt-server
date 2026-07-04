"""File-mode (`POST /v1/audio/transcriptions`) benchmark client."""

from __future__ import annotations

import io
import time
import wave

import httpx

SAMPLE_RATE = 16000


def wrap_wav(pcm16: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a minimal WAV container (stdlib `wave`)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


async def transcribe_file(
    base_url: str,
    pcm16: bytes,
    model: str,
    token: str | None = None,
) -> tuple[str, float]:
    """POST `pcm16` (wrapped as a WAV) to `/v1/audio/transcriptions`.

    Returns `(text, wall_seconds)` — `wall_seconds` is the request's
    end-to-end wall-clock time, used for RTF (audio_seconds / wall_seconds)."""
    wav_bytes = wrap_wav(pcm16)
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"model": model, "response_format": "json"}
    headers = {"Authorization": f"Bearer {token}"} if token is not None else None

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/v1/audio/transcriptions", data=data, files=files, headers=headers
        )
    wall_seconds = time.monotonic() - start

    resp.raise_for_status()
    text = resp.json().get("text", "")
    return text, wall_seconds
