"""Minimal native-WS client: streams a 16 kHz mono PCM16 WAV at real-time pace.

Usage: uv run python examples/ws_client.py path/to/audio.wav [ws://127.0.0.1:8000]
The `websockets` package is already installed via uvicorn[standard]."""

import asyncio
import json
import sys
import wave

import websockets

CHUNK_MS = 100


async def run(wav_path: str, base: str = "ws://127.0.0.1:8000") -> None:
    with wave.open(wav_path, "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1, "need 16 kHz mono"
        audio = wav.readframes(wav.getnframes())

    async with websockets.connect(f"{base}/ws/transcribe?model=mock") as ws:
        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                if ev["type"] == "partial":
                    print(f"\r[{ev['utterance_id']}] {ev['stable_text']} | {ev['volatile_text']}",
                          end="", flush=True)
                elif ev["type"] == "final":
                    print(f"\r[{ev['utterance_id']}] FINAL: {ev['text']}")
                elif ev["type"] == "session.closed":
                    return

        recv_task = asyncio.create_task(receiver())
        chunk_bytes = 16000 * 2 * CHUNK_MS // 1000
        for i in range(0, len(audio), chunk_bytes):
            await ws.send(audio[i : i + chunk_bytes])
            await asyncio.sleep(CHUNK_MS / 1000)  # real-time pacing
        await ws.send(json.dumps({"type": "input_done"}))
        await recv_task


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], *sys.argv[2:]))
