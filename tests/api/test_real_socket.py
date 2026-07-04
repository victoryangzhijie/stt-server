"""One true-network smoke test: uvicorn on a real port, websockets client.

Everything else uses TestClient (in-process ASGI); this guards the
socket-level path (HTTP upgrade, binary frames over TCP)."""

import asyncio
import base64
import json
import socket
import threading
import time

import pytest
import uvicorn
from websockets.asyncio.client import connect

from stt_server.api.app import create_app
from tests.api.test_native_ws import make_test_settings
from tests.helpers.audio import make_silence, make_tone


@pytest.fixture(scope="module")
def server_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(create_app(make_test_settings()), host="127.0.0.1",
                            port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn failed to start")
        time.sleep(0.05)
    yield port
    server.should_exit = True
    thread.join(timeout=5)


async def test_realtime_over_real_socket(server_port):
    uri = f"ws://127.0.0.1:{server_port}/v1/realtime?intent=transcription&model=mock"
    async with connect(uri) as ws:
        created = json.loads(await ws.recv())
        assert created["type"] == "transcription_session.created"
        audio = make_tone(600) + make_silence(300)
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode(),
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        transcript = None
        async with asyncio.timeout(10):
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "conversation.item.input_audio_transcription.completed":
                    transcript = m["transcript"]
                    break
        assert transcript == "The quick brown fox."
