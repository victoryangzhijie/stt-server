from stt_server.config.settings import VadConfig
from stt_server.core.vad import EnergyVad, make_vad
from tests.helpers.audio import make_silence, make_tone


def test_tone_is_speech_silence_is_not():
    vad = EnergyVad(threshold_dbfs=-40.0)
    assert vad.is_speech(make_tone(30)) is True
    assert vad.is_speech(make_silence(30)) is False


def test_quiet_tone_below_threshold():
    vad = EnergyVad(threshold_dbfs=-20.0)
    assert vad.is_speech(make_tone(30, amplitude=0.01)) is False


def test_factory():
    assert isinstance(make_vad(VadConfig(kind="energy")), EnergyVad)


def test_empty_frame_is_not_speech():
    vad = EnergyVad(threshold_dbfs=-40.0)
    assert vad.is_speech(b"") is False


def test_odd_length_frame_drops_trailing_byte_and_never_raises():
    vad = EnergyVad(threshold_dbfs=-40.0)
    # A single odd trailing byte with no full sample must not raise.
    assert vad.is_speech(b"\x01") is False
    # A tone with one extra trailing byte: the trailing byte is dropped and
    # the remaining (even-length) frame is classified normally.
    tone = make_tone(30)
    assert vad.is_speech(tone + b"\x01") is True
    silence = make_silence(30)
    assert vad.is_speech(silence + b"\x01") is False
