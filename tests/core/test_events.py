from stt_server.core.events import AudioChunk, EventType, TranscriptEvent


def test_audio_chunk_duration():
    # 16000 samples/s * 2 bytes: 3200 bytes = 100 ms
    chunk = AudioChunk(data=b"\x00" * 3200, ingest_ts=1.0)
    assert chunk.duration_ms == 100.0


def test_transcript_event_defaults():
    ev = TranscriptEvent(
        type=EventType.PARTIAL,
        session_id="s1",
        utterance_id=0,
        seq=3,
        audio_time_ms=1200.0,
        emitted_ts=2.0,
        stable_text="hello",
        volatile_text="wor",
    )
    assert ev.text == ""
    assert ev.recoverable is True
    assert ev.latency == {}
