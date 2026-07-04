from pathlib import Path

import pydantic
import pytest

from stt_server.config.settings import ConfigError, LimitsConfig, Settings, load_settings


def test_defaults():
    s = Settings()
    assert s.server.port == 8000
    assert s.vad.kind == "energy"
    assert s.endpointing.min_silence_ms == 500
    assert s.stabilizer.min_partials == 2
    assert s.backends["mock"].type == "mock"
    assert s.models["mock"] == "mock"
    assert s.limits.max_sessions == 100
    assert s.limits.max_upload_bytes == 26_214_400
    assert s.limits.max_session_seconds == 3600.0
    assert s.limits.audio_queue_chunks == 64
    assert s.limits.audio_overflow_policy == "drop_oldest"


def test_limits_yaml_roundtrip_for_backpressure_fields(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "limits:\n  audio_queue_chunks: 8\n  audio_overflow_policy: error\n"
    )
    s = load_settings(cfg)
    assert s.limits.audio_queue_chunks == 8
    assert s.limits.audio_overflow_policy == "error"


def test_invalid_audio_overflow_policy_rejected():
    with pytest.raises(pydantic.ValidationError):
        LimitsConfig(audio_overflow_policy="explode")


def test_yaml_file_overrides_defaults(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("server:\n  port: 9001\nendpointing:\n  min_silence_ms: 700\n")
    s = load_settings(cfg)
    assert s.server.port == 9001
    assert s.endpointing.min_silence_ms == 700
    assert s.vad.kind == "energy"  # untouched defaults survive


def test_env_overrides_yaml(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("server:\n  port: 9001\n")
    monkeypatch.setenv("STT__SERVER__PORT", "9002")
    s = load_settings(cfg)
    assert s.server.port == 9002


def test_missing_config_file_raises_config_error(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("server: [unclosed")
    with pytest.raises(ConfigError):
        load_settings(cfg)
