# Plan 1: Core Server + Mock Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable STT server with the full protocol-agnostic session pipeline (VAD → endpointing → backend stream → stabilizer → TranscriptEvent bus), the mock backend, and the native WebSocket API — fully tested with no ML dependencies.

**Architecture:** Single asyncio FastAPI process. The session core speaks only internal types (`AudioChunk` in, `TranscriptEvent` out); backends implement an async plugin interface; the native WS API is a thin encoder over the event stream. See `docs/superpowers/specs/2026-07-02-stt-server-design.md`.

**Tech Stack:** Python 3.12+, uv, FastAPI/Starlette, uvicorn, pydantic + pydantic-settings, PyYAML, structlog, pytest + pytest-asyncio, ruff.

**This is Plan 1 of 4.** Plan 2: OpenAI protocol adapters. Plan 3: real backends (Silero VAD, sherpa-onnx, FunASR, Qwen3-ASR) + metrics + Docker. Plan 4: benchmarks + docs.

**Spec items intentionally deferred** (config fields exist now; enforcement lands later): auth bearer tokens and `limits.max_sessions` (Plan 2, with the OpenAI adapters); Silero VAD, Prometheus metrics, per-session summary logs, backpressure/queue-shedding policy (Plan 3); full README and architecture docs (Plan 4).

## Global Constraints

- Python **3.12+**; all tooling through **uv** (`uv sync`, `uv run pytest`).
- Package name `stt_server`, src layout (`src/stt_server/`). Repo working name `stt-server`.
- License: MIT.
- Internal audio format: **PCM16 mono 16 kHz little-endian**, always.
- `src/stt_server/core/` and `src/stt_server/backends/` must never import from `src/stt_server/api/`.
- The server must always boot with the mock backend; tests in this plan require **no ML dependencies** (energy VAD only; Silero comes in Plan 3).
- All configuration externalized: `config.yaml` + `STT__`-prefixed env overrides (env wins over file).
- Async tests use pytest-asyncio in `asyncio_mode = "auto"` (no per-test markers needed).

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md`,
  `src/stt_server/__init__.py`, `src/stt_server/{api,core,backends,config}/__init__.py`,
  `tests/__init__.py`, `tests/test_package.py`

**Interfaces:**
- Produces: installable package `stt_server` with `stt_server.__version__ = "0.1.0"`; `uv run pytest` works.

- [ ] **Step 1: Initialize project files**

`pyproject.toml`:

```toml
[project]
name = "stt-server"
version = "0.1.0"
description = "OpenAI-compatible real-time STT inference server with pluggable open-source ASR backends"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "pyyaml>=6.0",
    "structlog>=24.4",
]

[project.scripts]
stt-server = "stt_server.__main__:main"

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.25",
    "httpx>=0.28",
    "ruff>=0.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stt_server"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ASYNC"]
```

`.gitignore`:

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
dist/
*.egg-info/
models/
```

`LICENSE`: standard MIT text, copyright 2026.

`README.md` (stub, expanded in Plan 4):

```markdown
# stt-server

OpenAI-compatible real-time speech-to-text inference server with pluggable
open-source ASR backends. Work in progress; see
`docs/superpowers/specs/2026-07-02-stt-server-design.md` for the design.

## Quickstart (dev)

    uv sync
    uv run pytest
```

`src/stt_server/__init__.py`:

```python
__version__ = "0.1.0"
```

Empty `__init__.py` for `api/`, `core/`, `backends/`, `config/`, and `tests/`.

- [ ] **Step 2: Write the smoke test**

`tests/test_package.py`:

```python
import stt_server


def test_version():
    assert stt_server.__version__ == "0.1.0"
```

- [ ] **Step 3: Sync and run tests**

Run: `uv sync && uv run pytest -v`
Expected: `test_version PASS` (1 passed).

- [ ] **Step 4: Lint**

Run: `uv run ruff check .`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: project scaffold (uv, src layout, pytest, ruff)"
```

---

### Task 2: Core types — AudioChunk, TranscriptEvent, FrameSlicer

**Files:**
- Create: `src/stt_server/core/events.py`, `src/stt_server/core/audio.py`
- Test: `tests/core/test_events.py`, `tests/core/test_audio.py` (and empty `tests/core/__init__.py`)

**Interfaces:**
- Produces:
  - `events.SAMPLE_RATE = 16000`, `events.BYTES_PER_SAMPLE = 2`
  - `AudioChunk(data: bytes, ingest_ts: float)` frozen dataclass, `.duration_ms -> float`
  - `EventType` enum: `SPEECH_START, SPEECH_END, PARTIAL, STABILIZED, FINAL, ERROR`
  - `TranscriptEvent(type, session_id, utterance_id, seq, audio_time_ms, emitted_ts, stable_text="", volatile_text="", text="", error_code=None, recoverable=True, latency={})` frozen dataclass
  - `audio.FrameSlicer(frame_ms: int)` with `.frame_bytes: int` and `.push(data: bytes) -> list[bytes]`

- [ ] **Step 1: Write failing tests**

`tests/core/test_events.py`:

```python
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
```

`tests/core/test_audio.py`:

```python
from stt_server.core.audio import FrameSlicer


def test_slicer_rechunks_to_fixed_frames():
    s = FrameSlicer(frame_ms=30)  # 30 ms @ 16 kHz pcm16 = 960 bytes
    assert s.frame_bytes == 960
    assert s.push(b"\x01" * 500) == []          # not enough yet
    frames = s.push(b"\x01" * 1500)              # buffer now 2000 bytes
    assert [len(f) for f in frames] == [960, 960]
    assert s.push(b"\x01" * 880) == [b"\x01" * 960]  # 80 leftover + 880
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/core -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_server.core.events'`.

- [ ] **Step 3: Implement**

`src/stt_server/core/events.py`:

```python
"""Internal data model: audio chunks and the unified transcript event stream."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # pcm16


@dataclass(frozen=True)
class AudioChunk:
    """PCM16 mono 16 kHz little-endian audio with its ingest timestamp."""

    data: bytes
    ingest_ts: float  # time.monotonic() at ingest

    @property
    def duration_ms(self) -> float:
        return len(self.data) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000.0


class EventType(enum.Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    PARTIAL = "partial"
    STABILIZED = "stabilized"
    FINAL = "final"
    ERROR = "error"


@dataclass(frozen=True)
class TranscriptEvent:
    """Unified event emitted by the session core; all protocol encoders consume only this."""

    type: EventType
    session_id: str
    utterance_id: int
    seq: int
    audio_time_ms: float  # position in the audio stream when the event was produced
    emitted_ts: float  # time.monotonic() when emitted
    stable_text: str = ""  # PARTIAL: committed prefix
    volatile_text: str = ""  # PARTIAL: uncommitted tail
    text: str = ""  # FINAL: final text; STABILIZED: newly committed delta
    error_code: str | None = None
    recoverable: bool = True
    latency: dict[str, float] = field(default_factory=dict)
```

`src/stt_server/core/audio.py`:

```python
"""Audio re-chunking utilities."""

from __future__ import annotations

from stt_server.core.events import BYTES_PER_SAMPLE, SAMPLE_RATE


class FrameSlicer:
    """Re-chunks arbitrarily sized PCM16 byte strings into fixed-duration frames."""

    def __init__(self, frame_ms: int) -> None:
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/core -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core tests/core
git commit -m "feat: core types — AudioChunk, TranscriptEvent, FrameSlicer"
```

---

### Task 3: Configuration system

**Files:**
- Create: `src/stt_server/config/settings.py`, `configs/mock.yaml`
- Test: `tests/config/test_settings.py` (and empty `tests/config/__init__.py`)

**Interfaces:**
- Produces (all pydantic models in `stt_server.config.settings`):
  - `ServerConfig(host="127.0.0.1", port=8000)`
  - `VadConfig(kind="energy", threshold_dbfs=-40.0)` — `kind: Literal["energy", "silero"]`
  - `EndpointingConfig(frame_ms=30, pre_roll_ms=300, min_silence_ms=500, max_utterance_ms=30000, speech_start_frames=2)`
  - `StabilizerConfig(min_partials=2, min_stable_ms=400.0)`
  - `BackendDef(type: str, options: dict[str, Any] = {})`
  - `AuthConfig(tokens: list[str] = [])`
  - `LimitsConfig(max_sessions=100)`
  - `Settings` with fields `server, vad, endpointing, stabilizer, backends: dict[str, BackendDef], models: dict[str, str], auth, limits`. Defaults include `backends={"mock": BackendDef(type="mock")}` and `models={"mock": "mock"}` (served model name → backend key).
  - `load_settings(path: str | Path | None = None) -> Settings` — YAML file + env overrides; **env wins over file**.

- [ ] **Step 1: Write failing tests**

`tests/config/test_settings.py`:

```python
from pathlib import Path

from stt_server.config.settings import Settings, load_settings


def test_defaults():
    s = Settings()
    assert s.server.port == 8000
    assert s.vad.kind == "energy"
    assert s.endpointing.min_silence_ms == 500
    assert s.stabilizer.min_partials == 2
    assert s.backends["mock"].type == "mock"
    assert s.models["mock"] == "mock"


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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/config -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/stt_server/config/settings.py`:

```python
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


def load_settings(path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    return Settings(**data)
```

`configs/mock.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8000
vad:
  kind: energy
  threshold_dbfs: -40.0
endpointing:
  pre_roll_ms: 300
  min_silence_ms: 500
  max_utterance_ms: 30000
stabilizer:
  min_partials: 2
  min_stable_ms: 400.0
backends:
  mock:
    type: mock
    options:
      partial_interval_ms: 240
models:
  mock: mock
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/config -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/config tests/config configs
git commit -m "feat: pydantic-settings config with YAML file and env overrides"
```

---

### Task 4: Backend plugin interface + registry

**Files:**
- Create: `src/stt_server/backends/base.py`, `src/stt_server/backends/registry.py`
- Test: `tests/backends/test_registry.py` (and empty `tests/backends/__init__.py`)

**Interfaces:**
- Consumes: `AudioChunk` from Task 2, `BackendDef` from Task 3.
- Produces (in `stt_server.backends.base`):
  - `BackendCapabilities(streaming: bool, languages: tuple[str, ...], native_endpointing: bool = False, batch_decode: bool = False)` frozen dataclass
  - `BackendEvent(kind: Literal["partial", "final"], text: str, audio_time_ms: float)` frozen dataclass
  - `StreamConfig(language: str | None = None)` frozen dataclass
  - `SttStream` ABC: `async push_audio(chunk: AudioChunk)`, `events() -> AsyncIterator[BackendEvent]`, `async finalize()`, `async close()`. Lifecycle: **one stream per utterance** — created on speech-start, finalized on endpoint, closed after FINAL.
  - `SttBackend` ABC: class attrs `name: str`, `capabilities: BackendCapabilities`; `async start()`, `async stop()`, `async create_stream(cfg: StreamConfig) -> SttStream`
  - `BackendUnavailableError(Exception)` with `.code = "backend_unavailable"`
- Produces (in `stt_server.backends.registry`):
  - `@register_backend(type_name: str)` decorator for `SttBackend` subclasses
  - `create_backend(defn: BackendDef) -> SttBackend` — instantiates `cls(**defn.options)`; raises `BackendUnavailableError` for unknown types

- [ ] **Step 1: Write failing tests**

`tests/backends/test_registry.py`:

```python
import pytest

from stt_server.backends.base import (
    BackendCapabilities,
    BackendUnavailableError,
    SttBackend,
    StreamConfig,
)
from stt_server.backends.registry import create_backend, register_backend
from stt_server.config.settings import BackendDef


@register_backend("dummy")
class DummyBackend(SttBackend):
    name = "dummy"
    capabilities = BackendCapabilities(streaming=True, languages=("en",))

    def __init__(self, greeting: str = "hi"):
        self.greeting = greeting

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def create_stream(self, cfg: StreamConfig):
        raise NotImplementedError


def test_create_backend_passes_options():
    backend = create_backend(BackendDef(type="dummy", options={"greeting": "yo"}))
    assert isinstance(backend, DummyBackend)
    assert backend.greeting == "yo"


def test_unknown_type_raises_unavailable():
    with pytest.raises(BackendUnavailableError):
        create_backend(BackendDef(type="nope"))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/backends -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/stt_server/backends/base.py`:

```python
"""The backend plugin contract every STT backend implements."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from stt_server.core.events import AudioChunk


@dataclass(frozen=True)
class BackendCapabilities:
    streaming: bool
    languages: tuple[str, ...]
    native_endpointing: bool = False
    batch_decode: bool = False


@dataclass(frozen=True)
class BackendEvent:
    kind: Literal["partial", "final"]
    text: str
    audio_time_ms: float


@dataclass(frozen=True)
class StreamConfig:
    language: str | None = None


class BackendUnavailableError(Exception):
    code = "backend_unavailable"


class SttStream(abc.ABC):
    """One stream per utterance: created on speech-start, finalized on endpoint,
    closed after the FINAL event is emitted."""

    @abc.abstractmethod
    async def push_audio(self, chunk: AudioChunk) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[BackendEvent]:
        """Yields partials then exactly one final; the iterator ends after finalize()."""

    @abc.abstractmethod
    async def finalize(self) -> None:
        """Endpoint reached: flush and emit the final event, then end the iterator."""

    @abc.abstractmethod
    async def close(self) -> None: ...


class SttBackend(abc.ABC):
    name: str
    capabilities: BackendCapabilities

    @abc.abstractmethod
    async def start(self) -> None:
        """Load models / warm up. Called once at server startup."""

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    async def create_stream(self, cfg: StreamConfig) -> SttStream: ...
```

`src/stt_server/backends/registry.py`:

```python
"""Backend registry: maps config `type` strings to backend classes."""

from __future__ import annotations

from stt_server.backends.base import BackendUnavailableError, SttBackend
from stt_server.config.settings import BackendDef

_REGISTRY: dict[str, type[SttBackend]] = {}


def register_backend(type_name: str):
    def decorator(cls: type[SttBackend]) -> type[SttBackend]:
        _REGISTRY[type_name] = cls
        return cls

    return decorator


def create_backend(defn: BackendDef) -> SttBackend:
    cls = _REGISTRY.get(defn.type)
    if cls is None:
        raise BackendUnavailableError(
            f"unknown backend type {defn.type!r}; registered: {sorted(_REGISTRY)}"
        )
    return cls(**defn.options)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/backends -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/backends tests/backends
git commit -m "feat: backend plugin interface and registry"
```

---

### Task 5: Mock backend + plugin conformance suite

**Files:**
- Create: `src/stt_server/backends/mock.py`
- Modify: `src/stt_server/backends/__init__.py` (import mock so registration runs)
- Test: `tests/backends/conformance.py`, `tests/backends/test_mock_backend.py`

**Interfaces:**
- Consumes: `SttBackend`, `SttStream`, `BackendEvent`, `StreamConfig`, `BackendCapabilities` from Task 4; `AudioChunk` from Task 2; `register_backend` from Task 4.
- Produces:
  - `MockUtteranceScript(partials: tuple[str, ...], final: str)` frozen dataclass
  - `MockBackend(partial_interval_ms: float = 240.0, scripts: list | None = None)` registered as type `"mock"`. Streams cycle through `scripts` round-robin. Partials are emitted deterministically from **audio time**, not wall clock: the Nth partial (1-based) is queued once `audio_ms >= N * partial_interval_ms`.
  - `DEFAULT_SCRIPTS: list[MockUtteranceScript]`
  - `tests/backends/conformance.py::BackendConformanceSuite` — pytest base class any backend must pass; requires subclasses to override the `backend` fixture.

- [ ] **Step 1: Write the conformance suite and mock tests (failing)**

`tests/backends/conformance.py`:

```python
"""Behavioral contract for SttBackend implementations.

Subclass and override the `backend` fixture. Run against mock in CI;
runnable against real backends locally (Plan 3)."""

import asyncio

import pytest

from stt_server.backends.base import SttBackend, StreamConfig
from stt_server.core.events import AudioChunk

ONE_SECOND_PCM = b"\x00\x01" * 16000  # 1 s of quiet non-zero pcm16


class BackendConformanceSuite:
    @pytest.fixture
    def backend(self) -> SttBackend:
        raise NotImplementedError("subclass must provide a backend fixture")

    async def _run_utterance(self, backend: SttBackend, audio: bytes) -> list:
        stream = await backend.create_stream(StreamConfig())
        events = []

        async def reader():
            async for ev in stream.events():
                events.append(ev)

        task = asyncio.create_task(reader())
        await stream.push_audio(AudioChunk(data=audio, ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        await stream.close()
        return events

    async def test_start_stop(self, backend):
        await backend.start()
        await backend.stop()

    async def test_finalize_yields_exactly_one_final_last(self, backend):
        await backend.start()
        try:
            events = await self._run_utterance(backend, ONE_SECOND_PCM)
            finals = [e for e in events if e.kind == "final"]
            assert len(finals) == 1
            assert events[-1].kind == "final"
            assert isinstance(finals[0].text, str) and finals[0].text
        finally:
            await backend.stop()

    async def test_partials_are_ordered_by_audio_time(self, backend):
        await backend.start()
        try:
            events = await self._run_utterance(backend, ONE_SECOND_PCM)
            times = [e.audio_time_ms for e in events]
            assert times == sorted(times)
        finally:
            await backend.stop()

    async def test_close_without_finalize_ends_iterator(self, backend):
        await backend.start()
        try:
            stream = await backend.create_stream(StreamConfig())
            await stream.push_audio(AudioChunk(data=ONE_SECOND_PCM, ingest_ts=0.0))
            await stream.close()

            async def drain():
                return [ev async for ev in stream.events()]

            await asyncio.wait_for(drain(), timeout=5.0)  # must not hang
        finally:
            await backend.stop()
```

`tests/backends/test_mock_backend.py`:

```python
import asyncio

import pytest

from stt_server.backends.base import StreamConfig
from stt_server.backends.mock import MockBackend, MockUtteranceScript
from stt_server.core.events import AudioChunk

from .conformance import BackendConformanceSuite


class TestMockConformance(BackendConformanceSuite):
    @pytest.fixture
    def backend(self):
        return MockBackend()


SCRIPT = MockUtteranceScript(partials=("i", "i want", "i want a coffee"), final="I want a coffee.")


async def test_partials_follow_audio_time():
    backend = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT])
    await backend.start()
    stream = await backend.create_stream(StreamConfig())
    events = []
    task = asyncio.create_task(_drain(stream, events))

    # 250 ms of audio -> partials at 100 ms and 200 ms boundaries = 2 partials
    await stream.push_audio(AudioChunk(data=b"\x00" * 8000, ingest_ts=0.0))
    await _until(lambda: len(events) == 2)
    assert [e.text for e in events] == ["i", "i want"]

    await stream.finalize()
    await asyncio.wait_for(task, timeout=5.0)
    assert [e.kind for e in events] == ["partial", "partial", "final"]
    assert events[-1].text == "I want a coffee."
    await stream.close()
    await backend.stop()


async def test_streams_cycle_scripts():
    s2 = MockUtteranceScript(partials=("ok",), final="OK.")
    backend = MockBackend(partial_interval_ms=100.0, scripts=[SCRIPT, s2])
    await backend.start()
    finals = []
    for _ in range(3):
        stream = await backend.create_stream(StreamConfig())
        events = []
        task = asyncio.create_task(_drain(stream, events))
        await stream.push_audio(AudioChunk(data=b"\x00" * 8000, ingest_ts=0.0))
        await stream.finalize()
        await asyncio.wait_for(task, timeout=5.0)
        await stream.close()
        finals.append(events[-1].text)
    assert finals == ["I want a coffee.", "OK.", "I want a coffee."]
    await backend.stop()


async def _drain(stream, sink):
    async for ev in stream.events():
        sink.append(ev)


async def _until(predicate, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/backends -v`
Expected: new tests FAIL with `ModuleNotFoundError: No module named 'stt_server.backends.mock'`; Task 4 tests still pass.

- [ ] **Step 3: Implement**

`src/stt_server/backends/mock.py`:

```python
"""Deterministic scripted backend: the reference implementation for all tests.

Partial timing is driven by accumulated *audio time*, never the wall clock,
so tests and benchmarks are fully deterministic."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator
from dataclasses import dataclass

from stt_server.backends.base import (
    BackendCapabilities,
    BackendEvent,
    SttBackend,
    SttStream,
    StreamConfig,
)
from stt_server.backends.registry import register_backend
from stt_server.core.events import AudioChunk


@dataclass(frozen=True)
class MockUtteranceScript:
    partials: tuple[str, ...]
    final: str


DEFAULT_SCRIPTS = [
    MockUtteranceScript(
        partials=("the", "the quick", "the quick brown", "the quick brown fox"),
        final="The quick brown fox.",
    ),
    MockUtteranceScript(
        partials=("hello", "hello world"),
        final="Hello world.",
    ),
]

_SENTINEL = None


class MockStream(SttStream):
    def __init__(self, script: MockUtteranceScript, partial_interval_ms: float) -> None:
        self._script = script
        self._interval = partial_interval_ms
        self._audio_ms = 0.0
        self._emitted = 0
        self._queue: asyncio.Queue[BackendEvent | None] = asyncio.Queue()
        self._done = False

    async def push_audio(self, chunk: AudioChunk) -> None:
        if self._done:
            return
        self._audio_ms += chunk.duration_ms
        while (
            self._emitted < len(self._script.partials)
            and self._audio_ms >= (self._emitted + 1) * self._interval
        ):
            await self._queue.put(
                BackendEvent(
                    kind="partial",
                    text=self._script.partials[self._emitted],
                    audio_time_ms=(self._emitted + 1) * self._interval,
                )
            )
            self._emitted += 1

    async def events(self) -> AsyncIterator[BackendEvent]:
        while True:
            ev = await self._queue.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def finalize(self) -> None:
        if self._done:
            return
        self._done = True
        await self._queue.put(
            BackendEvent(kind="final", text=self._script.final, audio_time_ms=self._audio_ms)
        )
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        if not self._done:
            self._done = True
            await self._queue.put(_SENTINEL)


@register_backend("mock")
class MockBackend(SttBackend):
    name = "mock"
    capabilities = BackendCapabilities(streaming=True, languages=("en",))

    def __init__(
        self,
        partial_interval_ms: float = 240.0,
        scripts: list[MockUtteranceScript] | None = None,
    ) -> None:
        self._interval = partial_interval_ms
        self._scripts = itertools.cycle(scripts or DEFAULT_SCRIPTS)

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def create_stream(self, cfg: StreamConfig) -> MockStream:
        return MockStream(next(self._scripts), self._interval)
```

Note: `events()` is declared `async def ... -> AsyncIterator[BackendEvent]` with `yield`, which satisfies the ABC's `def events()` signature (calling it returns an async iterator in both spellings).

`src/stt_server/backends/__init__.py`:

```python
from stt_server.backends import mock  # noqa: F401  (registers the mock backend)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/backends -v`
Expected: all pass (conformance x4 + 2 mock-specific + 2 registry).

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/backends tests/backends
git commit -m "feat: mock backend with deterministic audio-time partials + conformance suite"
```

---

### Task 6: VAD interface + energy detector + test audio helpers

**Files:**
- Create: `src/stt_server/core/vad.py`, `tests/helpers/__init__.py`, `tests/helpers/audio.py`
- Test: `tests/core/test_vad.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE`, `BYTES_PER_SAMPLE` from Task 2; `VadConfig` from Task 3.
- Produces:
  - `VadDetector` ABC: `is_speech(frame: bytes) -> bool` (one fixed-size PCM16 frame), `reset() -> None` (default no-op)
  - `EnergyVad(threshold_dbfs: float = -40.0)`
  - `make_vad(cfg: VadConfig) -> VadDetector` factory (raises `NotImplementedError` for `"silero"` until Plan 3)
  - Test helpers: `make_tone(ms: int, freq: float = 440.0, amplitude: float = 0.3) -> bytes`, `make_silence(ms: int) -> bytes` — PCM16 mono 16 kHz bytes.

- [ ] **Step 1: Write failing tests**

`tests/helpers/audio.py`:

```python
"""Synthesize PCM16 mono 16 kHz test audio without numpy."""

import math
import struct

from stt_server.core.events import SAMPLE_RATE


def make_tone(ms: int, freq: float = 440.0, amplitude: float = 0.3) -> bytes:
    n = SAMPLE_RATE * ms // 1000
    samples = (
        int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)
    )
    return struct.pack(f"<{n}h", *samples)


def make_silence(ms: int) -> bytes:
    n = SAMPLE_RATE * ms // 1000
    return b"\x00\x00" * n
```

`tests/core/test_vad.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/core/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_server.core.vad'`.

- [ ] **Step 3: Implement**

`src/stt_server/core/vad.py`:

```python
"""Voice activity detection interface and the dependency-free energy detector."""

from __future__ import annotations

import abc
import array
import math

from stt_server.config.settings import VadConfig


class VadDetector(abc.ABC):
    @abc.abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """Classify one fixed-size PCM16 frame."""

    def reset(self) -> None: ...


class EnergyVad(VadDetector):
    def __init__(self, threshold_dbfs: float = -40.0) -> None:
        self.threshold_dbfs = threshold_dbfs

    def is_speech(self, frame: bytes) -> bool:
        samples = array.array("h")
        samples.frombytes(frame)
        if not samples:
            return False
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        dbfs = 20 * math.log10(max(rms, 1e-9) / 32768.0)
        return dbfs > self.threshold_dbfs


def make_vad(cfg: VadConfig) -> VadDetector:
    if cfg.kind == "energy":
        return EnergyVad(threshold_dbfs=cfg.threshold_dbfs)
    raise NotImplementedError(f"VAD kind {cfg.kind!r} not available yet")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/core/test_vad.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core/vad.py tests/core/test_vad.py tests/helpers
git commit -m "feat: VAD interface, energy detector, synthetic test audio helpers"
```

---

### Task 7: Endpointing state machine

**Files:**
- Create: `src/stt_server/core/endpointing.py`
- Test: `tests/core/test_endpointing.py`

**Interfaces:**
- Consumes: `EndpointingConfig` from Task 3.
- Produces (in `stt_server.core.endpointing`):
  - `EndpointerState` enum: `IDLE, SPEECH, ENDPOINTING`
  - Actions (frozen dataclasses): `StartUtterance(frames: tuple[bytes, ...])` (pre-roll + triggering frames), `SpeechAudio(frame: bytes)`, `EndUtterance(reason: Literal["silence", "max_duration", "input_ended"])`
  - `EndpointAction = StartUtterance | SpeechAudio | EndUtterance`
  - `Endpointer(cfg: EndpointingConfig)` with `.state`, `.process(frame: bytes, is_speech: bool) -> list[EndpointAction]`, `.flush() -> list[EndpointAction]`
- Semantics: pure, clock-free — time advances only via frames (`cfg.frame_ms` each). `StartUtterance.frames` includes the pre-roll ring buffer. During `SPEECH`/`ENDPOINTING`, every frame (speech or silence) is forwarded as `SpeechAudio` *before* any `EndUtterance` in the same call's action list.

- [ ] **Step 1: Write failing tests**

`tests/core/test_endpointing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/core/test_endpointing.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/stt_server/core/endpointing.py`:

```python
"""VAD-frame-driven endpointing state machine. Pure and clock-free:
time advances only through frames, so behavior is fully deterministic."""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Literal

from stt_server.config.settings import EndpointingConfig


class EndpointerState(enum.Enum):
    IDLE = "idle"
    SPEECH = "speech"
    ENDPOINTING = "endpointing"


@dataclass(frozen=True)
class StartUtterance:
    frames: tuple[bytes, ...]  # pre-roll buffer, oldest first, incl. triggering frames


@dataclass(frozen=True)
class SpeechAudio:
    frame: bytes


@dataclass(frozen=True)
class EndUtterance:
    reason: Literal["silence", "max_duration", "input_ended"]


EndpointAction = StartUtterance | SpeechAudio | EndUtterance


class Endpointer:
    def __init__(self, cfg: EndpointingConfig) -> None:
        self.cfg = cfg
        self.state = EndpointerState.IDLE
        preroll_frames = max(1, cfg.pre_roll_ms // cfg.frame_ms)
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._speech_streak = 0
        self._silence_ms = 0
        self._utterance_ms = 0

    def process(self, frame: bytes, is_speech: bool) -> list[EndpointAction]:
        if self.state is EndpointerState.IDLE:
            return self._process_idle(frame, is_speech)
        return self._process_active(frame, is_speech)

    def _process_idle(self, frame: bytes, is_speech: bool) -> list[EndpointAction]:
        self._preroll.append(frame)
        self._speech_streak = self._speech_streak + 1 if is_speech else 0
        if self._speech_streak < self.cfg.speech_start_frames:
            return []
        frames = tuple(self._preroll)
        self.state = EndpointerState.SPEECH
        self._utterance_ms = len(frames) * self.cfg.frame_ms
        self._silence_ms = 0
        return [StartUtterance(frames=frames)]

    def _process_active(self, frame: bytes, is_speech: bool) -> list[EndpointAction]:
        actions: list[EndpointAction] = [SpeechAudio(frame=frame)]
        self._utterance_ms += self.cfg.frame_ms
        if is_speech:
            self.state = EndpointerState.SPEECH
            self._silence_ms = 0
        else:
            self.state = EndpointerState.ENDPOINTING
            self._silence_ms += self.cfg.frame_ms
        if self._silence_ms >= self.cfg.min_silence_ms:
            actions.append(EndUtterance(reason="silence"))
            self._reset()
        elif self._utterance_ms >= self.cfg.max_utterance_ms:
            actions.append(EndUtterance(reason="max_duration"))
            self._reset()
        return actions

    def flush(self) -> list[EndpointAction]:
        if self.state is EndpointerState.IDLE:
            return []
        self._reset()
        return [EndUtterance(reason="input_ended")]

    def _reset(self) -> None:
        self.state = EndpointerState.IDLE
        self._preroll.clear()
        self._speech_streak = 0
        self._silence_ms = 0
        self._utterance_ms = 0
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/core/test_endpointing.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core/endpointing.py tests/core/test_endpointing.py
git commit -m "feat: clock-free endpointing state machine with pre-roll"
```

---

### Task 8: Transcript stabilizer

**Files:**
- Create: `src/stt_server/core/stabilizer.py`
- Test: `tests/core/test_stabilizer.py`

**Interfaces:**
- Consumes: `StabilizerConfig` from Task 3.
- Produces:
  - `StabilizerUpdate(stable_text: str, volatile_text: str, newly_committed: str)` frozen dataclass
  - `Stabilizer(cfg: StabilizerConfig)` with `.update(text: str, now_ms: float) -> StabilizerUpdate` and `.reset() -> None`
- Semantics: token = whitespace-split word. A token is committed once it has appeared unchanged at the same position (token-level longest-common-prefix with the previous partial) in **≥ `min_partials` successive partials** and **≥ `min_stable_ms` ms** since first seen there. The committed prefix never shrinks, even if a later partial contradicts it. Clock-free: `now_ms` is a caller-supplied timestamp.

- [ ] **Step 1: Write failing tests**

`tests/core/test_stabilizer.py`:

```python
from stt_server.config.settings import StabilizerConfig
from stt_server.core.stabilizer import Stabilizer

CFG = StabilizerConfig(min_partials=2, min_stable_ms=400.0)


def test_commit_requires_survival_and_time():
    st = Stabilizer(CFG)
    u1 = st.update("i", now_ms=0)
    assert (u1.stable_text, u1.volatile_text) == ("", "i")   # seen once, 0 ms
    u2 = st.update("i want", now_ms=300)
    assert u2.stable_text == ""                              # "i" x2 but only 300 ms
    u3 = st.update("i want a", now_ms=500)
    assert u3.stable_text == "i"                             # x3 and 500 ms
    assert u3.newly_committed == "i"
    assert u3.volatile_text == "want a"
    u4 = st.update("i want a coffee", now_ms=900)
    assert u4.stable_text == "i want a"                      # "want","a" now qualify
    assert u4.newly_committed == "want a"


def test_changed_token_resets_survival():
    st = Stabilizer(CFG)
    st.update("i went", now_ms=0)
    st.update("i want", now_ms=500)      # "went"->"want": survival restarts
    u = st.update("i want", now_ms=600)  # "want" x2 but only 100 ms old
    assert u.stable_text == "i"
    u = st.update("i want more", now_ms=1000)
    assert u.stable_text == "i want"


def test_committed_prefix_never_shrinks():
    st = Stabilizer(CFG)
    st.update("alpha beta", now_ms=0)
    st.update("alpha beta", now_ms=500)
    u = st.update("alpha beta", now_ms=1000)
    assert u.stable_text == "alpha beta"
    u = st.update("alfa", now_ms=1500)   # backend contradicts committed text
    assert u.stable_text == "alpha beta"
    assert u.volatile_text == ""
    assert u.newly_committed == ""


def test_monotonic_stable_prefix_property():
    """Committed text is always a prefix-extension of the previous committed text."""
    partials = ["the", "the cat", "the can", "the can of", "the cat sat", "the cat sat down"]
    st = Stabilizer(CFG)
    prev = ""
    for i, p in enumerate(partials):
        u = st.update(p, now_ms=i * 450.0)
        assert u.stable_text.startswith(prev)
        prev = u.stable_text


def test_reset_clears_state():
    st = Stabilizer(CFG)
    st.update("hello world", now_ms=0)
    st.update("hello world", now_ms=500)
    st.reset()
    u = st.update("goodbye", now_ms=1000)
    assert (u.stable_text, u.volatile_text) == ("", "goodbye")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/core/test_stabilizer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/stt_server/core/stabilizer.py`:

```python
"""Partial-transcript stabilizer: maintains a committed prefix that only grows.

Reduces visible flicker by exposing {stable, volatile} text splits. Purely
functional over caller-supplied timestamps, so tests need no real clock."""

from __future__ import annotations

from dataclasses import dataclass

from stt_server.config.settings import StabilizerConfig


@dataclass(frozen=True)
class StabilizerUpdate:
    stable_text: str
    volatile_text: str
    newly_committed: str  # "" when the committed prefix did not grow


class Stabilizer:
    def __init__(self, cfg: StabilizerConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._committed: list[str] = []
        self._prev: list[str] = []
        self._seen_count: list[int] = []  # per token index: survived partials
        self._first_ms: list[float] = []  # per token index: first seen at this position

    def update(self, text: str, now_ms: float) -> StabilizerUpdate:
        tokens = text.split()

        # Survival tracking: tokens inside the LCP with the previous partial survive.
        lcp = 0
        limit = min(len(tokens), len(self._prev))
        while lcp < limit and tokens[lcp] == self._prev[lcp]:
            lcp += 1
        self._seen_count = [
            self._seen_count[i] + 1 if i < lcp else 1 for i in range(len(tokens))
        ]
        self._first_ms = [
            self._first_ms[i] if i < lcp else now_ms for i in range(len(tokens))
        ]
        self._prev = tokens

        # Grow the committed prefix (never shrink it).
        newly: list[str] = []
        i = len(self._committed)
        while (
            i < len(tokens)
            and self._seen_count[i] >= self._cfg.min_partials
            and now_ms - self._first_ms[i] >= self._cfg.min_stable_ms
        ):
            newly.append(tokens[i])
            i += 1
        self._committed.extend(newly)

        return StabilizerUpdate(
            stable_text=" ".join(self._committed),
            volatile_text=" ".join(tokens[len(self._committed) :]),
            newly_committed=" ".join(newly),
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/core/test_stabilizer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core/stabilizer.py tests/core/test_stabilizer.py
git commit -m "feat: prefix-commit transcript stabilizer"
```

---

### Task 9: Session pipeline

**Files:**
- Create: `src/stt_server/core/session.py`
- Test: `tests/core/test_session.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8 — `FrameSlicer`, `VadDetector`, `Endpointer` + actions, `Stabilizer`, `SttBackend`/`SttStream`/`StreamConfig`, `AudioChunk`, `TranscriptEvent`, `EventType`.
- Produces:
  - `Session(session_id: str, backend: SttBackend, vad: VadDetector, endpointer: Endpointer, stabilizer_factory: Callable[[], Stabilizer], stream_config: StreamConfig = StreamConfig())`
  - `async push_audio(chunk: AudioChunk) -> None`
  - `async end_input() -> None` — flush open utterance, then end the event stream
  - `async abort() -> None` — cancel/cleanup on client disconnect; also ends the event stream
  - `events() -> AsyncIterator[TranscriptEvent]` — yields until the session ends
- Event guarantees (documented + tested): `SPEECH_START` precedes all of its utterance's `PARTIAL`s; every `PARTIAL`/`STABILIZED` of an utterance precedes its `FINAL`; `FINAL.text` equals the backend's final verbatim; `FINAL.latency` contains `final_ms` (endpoint → FINAL) and `first_partial_ms` when a partial was seen. Trailing `PARTIAL`s may legitimately arrive after `SPEECH_END` (the backend is still flushing) — tests must not assert otherwise.

- [ ] **Step 1: Write failing tests**

`tests/core/test_session.py`:

```python
import asyncio

from stt_server.backends.mock import MockBackend, MockUtteranceScript
from stt_server.config.settings import EndpointingConfig, StabilizerConfig
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import EnergyVad
from tests.helpers.audio import make_silence, make_tone

EP_CFG = EndpointingConfig(
    frame_ms=30, pre_roll_ms=90, min_silence_ms=90, max_utterance_ms=60000, speech_start_frames=2
)
# Commit instantly so PARTIAL stable/volatile behavior is easy to assert.
STAB_CFG = StabilizerConfig(min_partials=1, min_stable_ms=0.0)
SCRIPTS = [
    MockUtteranceScript(partials=("hello", "hello world"), final="Hello world."),
    MockUtteranceScript(partials=("again",), final="Again."),
]


def make_session(scripts=SCRIPTS) -> Session:
    return Session(
        session_id="s-test",
        backend=MockBackend(partial_interval_ms=100.0, scripts=list(scripts)),
        vad=EnergyVad(threshold_dbfs=-40.0),
        endpointer=Endpointer(EP_CFG),
        stabilizer_factory=lambda: Stabilizer(STAB_CFG),
    )


async def run_session(session: Session, audio: bytes) -> list:
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=audio, ingest_ts=0.0))
    await session.end_input()
    await asyncio.wait_for(task, timeout=5.0)
    return events


async def test_single_utterance_event_flow():
    events = await run_session(make_session(), make_tone(600) + make_silence(300))
    types = [e.type for e in events]

    assert types[0] == EventType.SPEECH_START
    assert types.count(EventType.FINAL) == 1
    assert types[-1] == EventType.FINAL

    final = events[-1]
    assert final.text == "Hello world."          # backend final, verbatim
    assert final.utterance_id == 0
    assert "final_ms" in final.latency
    assert "first_partial_ms" in final.latency

    partials = [e for e in events if e.type == EventType.PARTIAL]
    assert len(partials) == 2
    assert partials[0].volatile_text or partials[0].stable_text  # carries text
    assert all(events.index(p) < events.index(final) for p in partials)

    stabilized = [e.text for e in events if e.type == EventType.STABILIZED]
    assert " ".join(stabilized).split() == ["hello", "world"]

    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_two_utterances_increment_utterance_id():
    audio = (
        make_tone(600) + make_silence(300) + make_tone(600) + make_silence(300)
    )
    events = await run_session(make_session(), audio)
    finals = [e for e in events if e.type == EventType.FINAL]
    assert [f.utterance_id for f in finals] == [0, 1]
    assert [f.text for f in finals] == ["Hello world.", "Again."]


async def test_end_input_mid_utterance_flushes_final():
    events = await run_session(make_session(), make_tone(600))  # no trailing silence
    finals = [e for e in events if e.type == EventType.FINAL]
    assert len(finals) == 1
    assert finals[0].text == "Hello world."


async def test_silence_only_produces_no_events():
    events = await run_session(make_session(), make_silence(600))
    assert events == []


async def test_abort_ends_event_stream():
    session = make_session()
    events = []

    async def collect():
        async for ev in session.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await session.push_audio(AudioChunk(data=make_tone(300), ingest_ts=0.0))
    await session.abort()
    await asyncio.wait_for(task, timeout=5.0)  # iterator must end
    assert not any(e.type == EventType.FINAL for e in events)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/core/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/stt_server/core/session.py`:

```python
"""Protocol-agnostic session: AudioChunks in, TranscriptEvents out.

Pipeline: FrameSlicer -> VAD -> Endpointer -> backend SttStream -> Stabilizer.
One backend stream per utterance (created on StartUtterance, closed after FINAL)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

from stt_server.backends.base import SttBackend, SttStream, StreamConfig
from stt_server.core.audio import FrameSlicer
from stt_server.core.endpointing import (
    EndpointAction,
    Endpointer,
    EndUtterance,
    SpeechAudio,
    StartUtterance,
)
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import VadDetector


class Session:
    def __init__(
        self,
        session_id: str,
        backend: SttBackend,
        vad: VadDetector,
        endpointer: Endpointer,
        stabilizer_factory: Callable[[], Stabilizer],
        stream_config: StreamConfig = StreamConfig(),
    ) -> None:
        self.session_id = session_id
        self._backend = backend
        self._vad = vad
        self._endpointer = endpointer
        self._stabilizer_factory = stabilizer_factory
        self._stream_config = stream_config

        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._slicer = FrameSlicer(endpointer.cfg.frame_ms)
        self._seq = 0
        self._utterance_id = 0
        self._audio_ms = 0.0
        self._stream: SttStream | None = None
        self._reader: asyncio.Task[str] | None = None
        self._stabilizer: Stabilizer | None = None
        self._utterance_started_ts = 0.0
        self._first_partial_ms: float | None = None
        self._ended = False

    async def push_audio(self, chunk: AudioChunk) -> None:
        for frame in self._slicer.push(chunk.data):
            self._audio_ms += self._endpointer.cfg.frame_ms
            is_speech = self._vad.is_speech(frame)
            for action in self._endpointer.process(frame, is_speech):
                await self._apply(action, chunk.ingest_ts)

    async def end_input(self) -> None:
        if self._ended:
            return
        for action in self._endpointer.flush():
            await self._apply(action, time.monotonic())
        self._end(None)

    async def abort(self) -> None:
        if self._ended:
            return
        if self._reader is not None:
            self._reader.cancel()
        if self._stream is not None:
            await self._stream.close()
            self._stream = None
        self._end(None)

    def _end(self, sentinel: None) -> None:
        self._ended = True
        self._queue.put_nowait(sentinel)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def _apply(self, action: EndpointAction, ingest_ts: float) -> None:
        if isinstance(action, StartUtterance):
            self._stream = await self._backend.create_stream(self._stream_config)
            self._stabilizer = self._stabilizer_factory()
            self._utterance_started_ts = time.monotonic()
            self._first_partial_ms = None
            self._emit(EventType.SPEECH_START)
            self._reader = asyncio.create_task(self._read_backend(self._stream))
            for frame in action.frames:
                await self._stream.push_audio(AudioChunk(data=frame, ingest_ts=ingest_ts))
        elif isinstance(action, SpeechAudio):
            assert self._stream is not None
            await self._stream.push_audio(AudioChunk(data=action.frame, ingest_ts=ingest_ts))
        elif isinstance(action, EndUtterance):
            assert self._stream is not None and self._reader is not None
            self._emit(EventType.SPEECH_END)
            endpoint_ts = time.monotonic()
            await self._stream.finalize()
            final_text = await self._reader
            latency = {"final_ms": (time.monotonic() - endpoint_ts) * 1000.0}
            if self._first_partial_ms is not None:
                latency["first_partial_ms"] = self._first_partial_ms
            self._emit(EventType.FINAL, text=final_text, latency=latency)
            await self._stream.close()
            self._stream = None
            self._reader = None
            self._utterance_id += 1

    async def _read_backend(self, stream: SttStream) -> str:
        """Consume backend events for one utterance; returns the final text."""
        assert self._stabilizer is not None
        final_text = ""
        async for bev in stream.events():
            if bev.kind == "partial":
                if self._first_partial_ms is None:
                    self._first_partial_ms = (
                        time.monotonic() - self._utterance_started_ts
                    ) * 1000.0
                upd = self._stabilizer.update(bev.text, time.monotonic() * 1000.0)
                self._emit(
                    EventType.PARTIAL,
                    stable_text=upd.stable_text,
                    volatile_text=upd.volatile_text,
                )
                if upd.newly_committed:
                    self._emit(EventType.STABILIZED, text=upd.newly_committed)
            else:
                final_text = bev.text
        return final_text

    def _emit(self, type_: EventType, **kwargs) -> None:
        ev = TranscriptEvent(
            type=type_,
            session_id=self.session_id,
            utterance_id=self._utterance_id,
            seq=self._seq,
            audio_time_ms=self._audio_ms,
            emitted_ts=time.monotonic(),
            **kwargs,
        )
        self._seq += 1
        self._queue.put_nowait(ev)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/core/test_session.py -v` then the full suite `uv run pytest -v`
Expected: 5 passed in the file; whole suite green.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/core/session.py tests/core/test_session.py
git commit -m "feat: session pipeline wiring VAD, endpointing, backend streams, stabilizer"
```

---

### Task 10: FastAPI app factory + native WebSocket API

**Files:**
- Create: `src/stt_server/api/app.py`, `src/stt_server/api/native_ws.py`
- Test: `tests/api/test_app.py`, `tests/api/test_native_ws.py` (and empty `tests/api/__init__.py`)

**Interfaces:**
- Consumes: `Settings`, `load_settings` (Task 3); `create_backend`, `BackendUnavailableError` (Task 4); `Session` (Task 9); `make_vad` (Task 6); `Endpointer` (Task 7); `Stabilizer` (Task 8); `TranscriptEvent`, `EventType`, `AudioChunk` (Task 2).
- Produces:
  - `app.create_app(settings: Settings) -> FastAPI` — lifespan starts/stops all configured backends; `app.state.backends: dict[str, SttBackend]`, `app.state.settings`, `app.state.ready: bool`; routes `/healthz`, `/readyz`, and the native WS router.
  - `app.resolve_backend(app: FastAPI, model: str) -> SttBackend` — maps served model name → backend via `settings.models`; raises `BackendUnavailableError`.
  - `native_ws.encode_native(ev: TranscriptEvent) -> dict` — the native JSON encoding.
  - WS endpoint `GET /ws/transcribe?model=<name>` — client sends **binary frames** of PCM16 mono 16 kHz audio and the **text frame** `{"type": "input_done"}` to flush; server sends JSON events and finally `{"type": "session.closed"}`, then closes. Unknown model → `{"type": "error", "code": "backend_unavailable", ...}` then close with code 4404.

Native event JSON shapes (documented here, asserted in tests):

```json
{"type": "speech_start", "session_id": "…", "utterance_id": 0, "seq": 0, "audio_time_ms": 60.0}
{"type": "partial", "…": "…", "stable_text": "hello", "volatile_text": "world"}
{"type": "stabilized", "…": "…", "text": "hello"}
{"type": "final", "…": "…", "text": "Hello world.", "latency": {"final_ms": 1.2, "first_partial_ms": 0.8}}
{"type": "error", "code": "backend_unavailable", "recoverable": false, "message": "…"}
{"type": "session.closed"}
```

- [ ] **Step 1: Write failing tests**

`tests/api/test_app.py`:

```python
from fastapi.testclient import TestClient

from stt_server.api.app import create_app
from stt_server.config.settings import Settings


def test_health_and_ready():
    app = create_app(Settings())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200
```

`tests/api/test_native_ws.py`:

```python
import json

from fastapi.testclient import TestClient

from stt_server.api.app import create_app
from stt_server.config.settings import (
    BackendDef,
    EndpointingConfig,
    Settings,
    StabilizerConfig,
)
from tests.helpers.audio import make_silence, make_tone


def make_test_settings() -> Settings:
    return Settings(
        endpointing=EndpointingConfig(
            frame_ms=30, pre_roll_ms=90, min_silence_ms=90, speech_start_frames=2
        ),
        stabilizer=StabilizerConfig(min_partials=1, min_stable_ms=0.0),
        backends={"mock": BackendDef(type="mock", options={"partial_interval_ms": 100.0})},
        models={"mock": "mock"},
    )


def drain(ws) -> list[dict]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg["type"] == "session.closed":
            return msgs


def test_ws_transcribe_end_to_end():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=mock") as ws:
            ws.send_bytes(make_tone(600) + make_silence(300))
            ws.send_text(json.dumps({"type": "input_done"}))
            msgs = drain(ws)

    types = [m["type"] for m in msgs]
    assert types[0] == "speech_start"
    assert "partial" in types
    final = next(m for m in msgs if m["type"] == "final")
    assert final["text"] == "The quick brown fox."  # first DEFAULT_SCRIPTS entry
    assert final["utterance_id"] == 0
    assert "final_ms" in final["latency"]
    partial = next(m for m in msgs if m["type"] == "partial")
    assert {"stable_text", "volatile_text", "seq", "session_id"} <= partial.keys()
    assert types[-1] == "session.closed"


def test_ws_unknown_model_sends_error():
    app = create_app(make_test_settings())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe?model=nope") as ws:
            msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["code"] == "backend_unavailable"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/api -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/stt_server/api/app.py`:

```python
"""FastAPI application factory."""

from __future__ import annotations

import contextlib

from fastapi import FastAPI, Response

from stt_server.api import native_ws
from stt_server.backends.base import BackendUnavailableError, SttBackend
from stt_server.backends.registry import create_backend
from stt_server.config.settings import Settings


def create_app(settings: Settings) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        backends = {key: create_backend(defn) for key, defn in settings.backends.items()}
        for backend in backends.values():
            await backend.start()
        app.state.backends = backends
        app.state.ready = True
        yield
        app.state.ready = False
        for backend in backends.values():
            await backend.stop()

    app = FastAPI(title="stt-server", lifespan=lifespan)
    app.state.settings = settings
    app.state.ready = False
    app.include_router(native_ws.router)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response):
        if not app.state.ready:
            response.status_code = 503
            return {"status": "not_ready"}
        return {"status": "ready"}

    return app


def resolve_backend(app: FastAPI, model: str) -> SttBackend:
    backend_key = app.state.settings.models.get(model)
    if backend_key is None or backend_key not in app.state.backends:
        raise BackendUnavailableError(f"unknown model {model!r}")
    return app.state.backends[backend_key]
```

`src/stt_server/api/native_ws.py`:

```python
"""Native WebSocket protocol: the internal TranscriptEvent stream as JSON.

Richer than the OpenAI protocol (exposes the stable/volatile split); used by
benchmark clients so instrumentation never depends on OpenAI framing."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from stt_server.backends.base import BackendUnavailableError
from stt_server.core.endpointing import Endpointer
from stt_server.core.events import AudioChunk, EventType, TranscriptEvent
from stt_server.core.session import Session
from stt_server.core.stabilizer import Stabilizer
from stt_server.core.vad import make_vad

logger = structlog.get_logger(__name__)
router = APIRouter()


def encode_native(ev: TranscriptEvent) -> dict:
    out: dict = {
        "type": ev.type.value,
        "session_id": ev.session_id,
        "utterance_id": ev.utterance_id,
        "seq": ev.seq,
        "audio_time_ms": ev.audio_time_ms,
    }
    if ev.type is EventType.PARTIAL:
        out["stable_text"] = ev.stable_text
        out["volatile_text"] = ev.volatile_text
    elif ev.type in (EventType.STABILIZED, EventType.FINAL):
        out["text"] = ev.text
    if ev.type is EventType.FINAL:
        out["latency"] = ev.latency
    if ev.type is EventType.ERROR:
        out["code"] = ev.error_code
        out["recoverable"] = ev.recoverable
    return out


@router.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket, model: str = "mock") -> None:
    from stt_server.api.app import resolve_backend  # local import: avoid cycle

    await ws.accept()
    try:
        backend = resolve_backend(ws.app, model)
    except BackendUnavailableError as exc:
        await ws.send_json(
            {"type": "error", "code": exc.code, "recoverable": False, "message": str(exc)}
        )
        await ws.close(code=4404)
        return

    settings = ws.app.state.settings
    session = Session(
        session_id=uuid.uuid4().hex,
        backend=backend,
        vad=make_vad(settings.vad),
        endpointer=Endpointer(settings.endpointing),
        stabilizer_factory=lambda: Stabilizer(settings.stabilizer),
    )
    log = logger.bind(session_id=session.session_id, model=model)
    log.info("session.opened")

    async def sender() -> None:
        async for ev in session.events():
            await ws.send_json(encode_native(ev))
        await ws.send_json({"type": "session.closed"})

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                await session.abort()
                break
            if msg.get("bytes") is not None:
                await session.push_audio(
                    AudioChunk(data=msg["bytes"], ingest_ts=time.monotonic())
                )
            elif msg.get("text"):
                control = json.loads(msg["text"])
                if control.get("type") == "input_done":
                    await session.end_input()
                    await send_task
                    await ws.close()
                    return
    except WebSocketDisconnect:
        await session.abort()
    finally:
        if not send_task.done():
            send_task.cancel()
        log.info("session.closed")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/api -v` then `uv run pytest`
Expected: 3 api tests pass; whole suite green.

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/api tests/api
git commit -m "feat: app factory, health endpoints, native WebSocket transcription API"
```

---

### Task 11: CLI entry point, logging, example client, CI

**Files:**
- Create: `src/stt_server/__main__.py`, `src/stt_server/logging.py`,
  `examples/ws_client.py`, `.github/workflows/ci.yml`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `create_app` (Task 10), `load_settings` (Task 3).
- Produces: `stt-server [--config PATH] [--host HOST] [--port PORT]` console command; `stt_server.logging.configure_logging()`; `build_settings(argv) -> Settings` (separated from `main()` so it is testable without starting uvicorn).

- [ ] **Step 1: Write failing test**

`tests/test_main.py`:

```python
from stt_server.__main__ import build_settings


def test_cli_overrides_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("server:\n  port: 9001\n")
    s = build_settings(["--config", str(cfg), "--port", "9100"])
    assert s.server.port == 9100

    s = build_settings(["--config", str(cfg)])
    assert s.server.port == 9001
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError` (no `stt_server.__main__`).

- [ ] **Step 3: Implement**

`src/stt_server/logging.py`:

```python
"""Structured JSON logging via structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level]
        ),
    )
```

`src/stt_server/__main__.py`:

```python
"""CLI entry point: `stt-server --config configs/mock.yaml`."""

from __future__ import annotations

import argparse

import uvicorn

from stt_server.api.app import create_app
from stt_server.config.settings import Settings, load_settings
from stt_server.logging import configure_logging


def build_settings(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(prog="stt-server")
    parser.add_argument("--config", default=None, help="path to config YAML")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.host is not None:
        settings.server.host = args.host
    if args.port is not None:
        settings.server.port = args.port
    return settings


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    settings = build_settings(argv)
    uvicorn.run(create_app(settings), host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
```

`examples/ws_client.py`:

```python
"""Minimal native-WS client: streams a 16 kHz mono PCM16 WAV at real-time pace.

Usage: uv run python examples/ws_client.py path/to/audio.wav [ws://127.0.0.1:8000]
The `websockets` package is already installed via uvicorn[standard]."""

import asyncio
import json
import sys
import wave

import websockets

CHUNK_MS = 100


async def run(wav_path: str, base: str = "ws://127.0.0.1:8000") -> None:
    with wave.open(wav_path, "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1, "need 16 kHz mono"
        audio = wav.readframes(wav.getnframes())

    async with websockets.connect(f"{base}/ws/transcribe?model=mock") as ws:
        async def receiver():
            async for raw in ws:
                ev = json.loads(raw)
                if ev["type"] == "partial":
                    print(f"\r[{ev['utterance_id']}] {ev['stable_text']} | {ev['volatile_text']}",
                          end="", flush=True)
                elif ev["type"] == "final":
                    print(f"\r[{ev['utterance_id']}] FINAL: {ev['text']}")
                elif ev["type"] == "session.closed":
                    return

        recv_task = asyncio.create_task(receiver())
        chunk_bytes = 16000 * 2 * CHUNK_MS // 1000
        for i in range(0, len(audio), chunk_bytes):
            await ws.send(audio[i : i + chunk_bytes])
            await asyncio.sleep(CHUNK_MS / 1000)  # real-time pacing
        await ws.send(json.dumps({"type": "input_done"}))
        await recv_task


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], *sys.argv[2:]))
```

`.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest -v
```

- [ ] **Step 4: Run tests and verify the server boots**

Run: `uv run pytest -v`
Expected: whole suite green.

Manual smoke check:

```bash
uv run stt-server --config configs/mock.yaml --port 8090 &
sleep 2
curl -s http://127.0.0.1:8090/healthz   # expect {"status":"ok"}
curl -s http://127.0.0.1:8090/readyz    # expect {"status":"ready"}
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add src/stt_server/__main__.py src/stt_server/logging.py examples .github tests/test_main.py
git commit -m "feat: CLI entry point, structured logging, example WS client, CI"
```

---

## Plan 1 exit criteria

- `uv run pytest` green; `uv run ruff check .` clean.
- `uv run stt-server --config configs/mock.yaml` boots; `/healthz`, `/readyz` respond; `examples/ws_client.py` streams a WAV and prints partials/finals against the mock backend.
- No module under `core/` or `backends/` imports from `api/` (spot-check with `grep -r "stt_server.api" src/stt_server/core src/stt_server/backends` → no matches).






