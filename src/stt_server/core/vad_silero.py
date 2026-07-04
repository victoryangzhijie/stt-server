"""Silero VAD (ONNX) detector — optional, behind the `silero` extra.

Silero VAD v5 expects 512-sample (32 ms @ 16 kHz) windows, but the pipeline
feeds this detector fixed 480-sample (30 ms @ 16 kHz) PCM16 frames. `SileroVad`
buffers incoming frames and runs inference once it has accumulated a full
512-sample window, so the reported decision lags the newest audio by at most
one window (~32 ms) — `is_speech()` returns the *previous* window's verdict
for any call that does not itself complete a new window.

ONNX I/O (v5 model, verified against the official snakers4/silero-vad
release): inputs `input` (1, 512) float32 in [-1, 1], `sr` int64 scalar
(16000), `state` (2, 1, 128) float32 (v5 merged the old h/c pair into a
single recurrent state tensor); output `output` (1, 1) float32 speech
probability and updated `stateN` (2, 1, 128) float32.
"""

from __future__ import annotations

import array
from pathlib import Path

from stt_server.core.vad import VadDetector

WINDOW_SAMPLES = 512
SAMPLE_RATE = 16000
_STATE_SHAPE = (2, 1, 128)

INSTALL_HINT = "pip install 'stt-server[silero]'"


class SileroVad(VadDetector):
    """Silero VAD (ONNX Runtime) speech detector.

    Decision latency is at most one 512-sample window (~32 ms): frames are
    buffered internally and inference only runs once a full window has
    accumulated, so `is_speech()` reflects the most recently *completed*
    window rather than the exact bytes just passed in.
    """

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise NotImplementedError(
                "Silero VAD requires onnxruntime, which is not installed. "
                f"Install it with: {INSTALL_HINT}"
            ) from exc
        import numpy as np  # ships with onnxruntime; only needed on this path

        self._np = np

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path!r}. "
                "Download it with: python scripts/download_models.py silero"
            )

        self.threshold = threshold
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._buffer = array.array("h")
        self._state = self._zeros_state()
        self._last_decision = False

    def _zeros_state(self):
        return self._np.zeros(_STATE_SHAPE, dtype="float32")

    def is_speech(self, frame: bytes) -> bool:
        np = self._np
        if len(frame) % 2:
            frame = frame[:-1]
        samples = array.array("h")
        samples.frombytes(frame)
        self._buffer.extend(samples)

        while len(self._buffer) >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            del self._buffer[:WINDOW_SAMPLES]

            audio = np.asarray(window, dtype="float32") / 32768.0
            audio = audio.reshape(1, WINDOW_SAMPLES)
            sr = np.array(SAMPLE_RATE, dtype="int64")

            outputs = self._session.run(
                None,
                {"input": audio, "sr": sr, "state": self._state},
            )
            prob, new_state = outputs[0], outputs[1]
            self._state = new_state
            self._last_decision = float(prob.reshape(-1)[0]) > self.threshold

        return self._last_decision

    def reset(self) -> None:
        self._buffer = array.array("h")
        self._state = self._zeros_state()
        self._last_decision = False
