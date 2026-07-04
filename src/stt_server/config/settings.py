"""Externalized configuration: YAML file + STT__-prefixed env overrides (env wins)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class VadConfig(BaseModel):
    kind: Literal["energy", "silero"] = "energy"
    threshold_dbfs: float = -40.0
    model_path: str = "models/silero_vad.onnx"
    threshold: float = 0.5


class EndpointingConfig(BaseModel):
    frame_ms: int = 30
    pre_roll_ms: int = 300
    min_silence_ms: int = 500
    max_utterance_ms: int = 30000
    speech_start_frames: int = 2


class StabilizerConfig(BaseModel):
    min_partials: int = 2
    min_stable_ms: float = 400.0


class BackendDef(BaseModel):
    type: str
    options: dict[str, Any] = Field(default_factory=dict)


class AuthConfig(BaseModel):
    tokens: list[str] = Field(default_factory=list)


class LimitsConfig(BaseModel):
    max_sessions: int = 100
    max_upload_bytes: int = 26_214_400  # 25 MiB, OpenAI's documented cap
    max_session_seconds: float = 3600.0
    # Bounded per-session input-audio queue depth, in chunks (spec §13
    # backpressure). 64 x ~100ms client chunks =~ 6.4s of audio buffered
    # before the shed policy kicks in.
    audio_queue_chunks: int = 64
    audio_overflow_policy: Literal["drop_oldest", "error"] = "drop_oldest"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT__", env_nested_delimiter="__")

    server: ServerConfig = ServerConfig()
    vad: VadConfig = VadConfig()
    endpointing: EndpointingConfig = EndpointingConfig()
    stabilizer: StabilizerConfig = StabilizerConfig()
    backends: dict[str, BackendDef] = Field(
        default_factory=lambda: {"mock": BackendDef(type="mock")}
    )
    models: dict[str, str] = Field(default_factory=lambda: {"mock": "mock"})
    auth: AuthConfig = AuthConfig()
    limits: LimitsConfig = LimitsConfig()

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Values from the YAML file arrive as init kwargs; env must override them.
        return (env_settings, init_settings)


class ConfigError(Exception):
    """Raised when the config file cannot be read or parsed."""


def load_settings(path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        try:
            text = path.read_text()
        except OSError as exc:
            raise ConfigError(f"cannot read config file: {path}") from exc
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in config file: {path}") from exc
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            raise ConfigError(
                f"config file must contain a YAML mapping at the top level: {path}"
            )
        data = parsed
    return Settings(**data)
