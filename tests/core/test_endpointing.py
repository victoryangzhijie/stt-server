from stt_server.config.settings import EndpointingConfig
from stt_server.core.endpointing import (
    Endpointer,
    EndpointerState,
    EndUtterance,
    SpeechAudio,
    StartUtterance,
)

CFG = EndpointingConfig(
    frame_ms=30, pre_roll_ms=90, min_silence_ms=90, max_utterance_ms=600, speech_start_frames=2
)
SPEECH = b"\x01" * 960
SILENCE = b"\x00" * 960


def feed(ep, frames):
    out = []
    for frame, is_speech in frames:
        out.extend(ep.process(frame, is_speech))
    return out


def test_start_requires_consecutive_speech_frames_and_includes_preroll():
    ep = Endpointer(CFG)
    assert ep.process(SPEECH, True) == []          # streak 1 < 2
    actions = ep.process(SPEECH, True)             # streak 2 -> start
    assert isinstance(actions[0], StartUtterance)
    assert len(actions[0].frames) == 2             # both speech frames via pre-roll buffer
    assert ep.state is EndpointerState.SPEECH


def test_preroll_is_capped():
    ep = Endpointer(CFG)  # pre_roll_ms=90 -> 3 frames
    feed(ep, [(SILENCE, False)] * 10)
    actions = feed(ep, [(SPEECH, True)] * 2)
    (start,) = [a for a in actions if isinstance(a, StartUtterance)]
    assert len(start.frames) == 3  # 1 silence + 2 speech, capped at 3


def test_silence_endpoint_after_min_silence():
    ep = Endpointer(CFG)
    feed(ep, [(SPEECH, True)] * 2)                 # start
    actions = feed(ep, [(SILENCE, False)] * 3)     # 90 ms silence
    assert ep.state is EndpointerState.IDLE
    assert isinstance(actions[-1], EndUtterance)
    assert actions[-1].reason == "silence"
    # all 3 silence frames were still forwarded to the backend first
    assert sum(isinstance(a, SpeechAudio) for a in actions) == 3


def test_speech_resumes_cancels_endpointing():
    ep = Endpointer(CFG)
    feed(ep, [(SPEECH, True)] * 2)
    feed(ep, [(SILENCE, False)] * 2)               # 60 ms < 90 ms
    assert ep.state is EndpointerState.ENDPOINTING
    ep.process(SPEECH, True)
    assert ep.state is EndpointerState.SPEECH
    actions = feed(ep, [(SILENCE, False)] * 3)     # counter restarted
    assert actions[-1] == EndUtterance(reason="silence")


def test_max_utterance_forces_endpoint():
    ep = Endpointer(CFG)  # max 600 ms = 20 frames of continuous speech
    # 2 frames trigger start (60 ms via pre-roll); 18 more reach 600 ms exactly
    actions = feed(ep, [(SPEECH, True)] * 20)
    ends = [a for a in actions if isinstance(a, EndUtterance)]
    assert [e.reason for e in ends] == ["max_duration"]
    assert ep.state is EndpointerState.IDLE


def test_flush_ends_open_utterance_only():
    ep = Endpointer(CFG)
    assert ep.flush() == []                        # idle -> nothing
    feed(ep, [(SPEECH, True)] * 2)
    assert ep.flush() == [EndUtterance(reason="input_ended")]
    assert ep.state is EndpointerState.IDLE
