from __future__ import annotations

import importlib

from backends.base import ASRBackend

_BACKENDS: dict[str, str] = {
    "qwen3": "backends.qwen3.Qwen3Backend",
    "mock": "backends.mock.MockBackend",
    "whisper": "backends.whisper.WhisperBackend",
}


def create_backend(name: str = "qwen3", **kwargs: object) -> ASRBackend:
    dotted = _BACKENDS.get(name)
    if dotted is None:
        raise ValueError(f"Unknown backend: {name!r}. Available: {sorted(_BACKENDS)}")
    module_path, class_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)
