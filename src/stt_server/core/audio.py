"""Audio re-chunking utilities."""

from __future__ import annotations

from stt_server.core.events import BYTES_PER_SAMPLE, SAMPLE_RATE


class FrameSlicer:
    """Re-chunks arbitrarily sized PCM16 byte strings into fixed-duration frames."""

    def __init__(self, frame_ms: int) -> None:
        if frame_ms <= 0:
            raise ValueError(f"frame_ms must be positive, got {frame_ms}")
        self.frame_ms = frame_ms
        self.frame_bytes = SAMPLE_RATE * frame_ms // 1000 * BYTES_PER_SAMPLE
        self._buf = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        frames: list[bytes] = []
        while len(self._buf) >= self.frame_bytes:
            frames.append(bytes(self._buf[: self.frame_bytes]))
            del self._buf[: self.frame_bytes]
        return frames
