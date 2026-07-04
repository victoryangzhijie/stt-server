import importlib.util
from pathlib import Path

import pytest

from stt_server.config.settings import VadConfig
from stt_server.core.vad import make_vad
from stt_server.core.vad_silero import INSTALL_HINT

ONNXRUNTIME_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
MODEL_PATH = "models/silero_vad.onnx"

# 1 second of real speech, PCM16 mono 16 kHz, from the Open Speech Repository
# (see tests/fixtures/README.md for full provenance, license terms, and the
# conversion procedure). A pure sine tone or white noise is NOT reliably
# classified as speech by Silero (verified experimentally), so real speech is
# used here instead of the synthetic tone/silence helpers of the energy VAD.
SPEECH_FIXTURE = Path(__file__).parent.parent / "fixtures" / "speech_16k_mono_s16le.pcm"
FRAME_BYTES = 960  # 480 samples * 2 bytes, the pipeline's 30 ms frame size


@pytest.mark.skipif(ONNXRUNTIME_AVAILABLE, reason="onnxruntime is installed; missing-dep path N/A")
def test_make_vad_silero_without_onnxruntime_raises_with_install_hint():
    with pytest.raises(NotImplementedError) as exc_info:
        make_vad(VadConfig(kind="silero"))
    assert INSTALL_HINT in str(exc_info.value)


def _frames(data: bytes):
    for i in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
        yield data[i : i + FRAME_BYTES]


@pytest.mark.model
def test_speech_is_speech_silence_is_not():
    from stt_server.core.vad_silero import SileroVad

    vad = SileroVad(model_path=MODEL_PATH)
    speech = SPEECH_FIXTURE.read_bytes()
    silence = b"\x00\x00" * (len(speech) // 2)

    decision = False
    for frame in _frames(speech):
        decision = vad.is_speech(frame)
    assert decision is True

    vad.reset()
    decision = True
    for frame in _frames(silence):
        decision = vad.is_speech(frame)
    assert decision is False


@pytest.mark.model
def test_reset_clears_state():
    from stt_server.core.vad_silero import SileroVad

    vad = SileroVad(model_path=MODEL_PATH)
    speech = SPEECH_FIXTURE.read_bytes()
    for frame in _frames(speech):
        vad.is_speech(frame)

    vad.reset()
    assert len(vad._buffer) == 0
    assert vad._last_decision is False

    # After reset, replaying silence must behave like a freshly-constructed
    # detector: state (not just the sample buffer) has to have been cleared,
    # not carried over from the speech that primed the LSTM/state tensor.
    fresh = SileroVad(model_path=MODEL_PATH)
    silence = b"\x00\x00" * (len(speech) // 2)
    decision_reset = None
    decision_fresh = None
    for frame in _frames(silence):
        decision_reset = vad.is_speech(frame)
        decision_fresh = fresh.is_speech(frame)
    assert decision_reset == decision_fresh is False


@pytest.mark.model
def test_480_sample_framing_buffers_across_calls():
    from stt_server.core.vad_silero import WINDOW_SAMPLES, SileroVad

    vad = SileroVad(model_path=MODEL_PATH)
    # A single 480-sample (960-byte) frame does not fill a 512-sample window.
    frame_480 = b"\x00\x00" * 480
    vad.is_speech(frame_480)
    assert len(vad._buffer) == 480

    # A second frame pushes the buffer past 512, triggering inference and
    # leaving the remainder (480 + 480 - 512 = 448 samples) buffered.
    vad.is_speech(frame_480)
    assert len(vad._buffer) == 960 - WINDOW_SAMPLES
