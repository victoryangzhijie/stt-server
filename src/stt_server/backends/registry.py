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
    try:
        return cls(**defn.options)
    except TypeError as exc:
        raise BackendUnavailableError(
            f"bad options for backend type {defn.type!r}: {exc}"
        ) from exc
