import pytest

from stt_server.backends.base import (
    BackendCapabilities,
    BackendUnavailableError,
    StreamConfig,
    SttBackend,
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


def test_bad_options_type_error_wrapped_as_unavailable():
    with pytest.raises(BackendUnavailableError) as exc_info:
        create_backend(BackendDef(type="dummy", options={"unexpected_kw": 1}))
    message = str(exc_info.value)
    assert "dummy" in message
    assert "unexpected_kw" in message
    assert isinstance(exc_info.value.__cause__, TypeError)
