import pytest

from stt_server.core.audio import FrameSlicer


def test_slicer_rechunks_to_fixed_frames():
    s = FrameSlicer(frame_ms=30)  # 30 ms @ 16 kHz pcm16 = 960 bytes
    assert s.frame_bytes == 960
    assert s.push(b"\x01" * 500) == []          # not enough yet
    frames = s.push(b"\x01" * 1500)              # buffer now 2000 bytes
    assert [len(f) for f in frames] == [960, 960]
    assert s.push(b"\x01" * 880) == [b"\x01" * 960]  # 80 leftover + 880


def test_slicer_rejects_non_positive_frame_ms():
    with pytest.raises(ValueError, match="frame_ms"):
        FrameSlicer(frame_ms=0)
