"""Config-roundtrip coverage for every shipped backend profile: each config's
backend `options` must be loadable as-is by the real backend constructor's
signature (inspect.signature does not instantiate, so this holds even when a
backend's heavy extra isn't installed — the constructor only raises
BackendUnavailableError on instantiation, not on signature inspection)."""

from __future__ import annotations

import inspect

import pytest

from stt_server.config.settings import load_settings


def _mock_backend_cls():
    from stt_server.backends.mock import MockBackend

    return MockBackend


def _sherpa_backend_cls():
    from stt_server.backends.sherpa.backend import SherpaBackend

    return SherpaBackend


def _funasr_backend_cls():
    from stt_server.backends.funasr.backend import FunasrBackend

    return FunasrBackend


@pytest.mark.parametrize(
    ("config_path", "backend_key", "backend_type", "cls_factory"),
    [
        ("configs/mock.yaml", "mock", "mock", _mock_backend_cls),
        ("configs/sherpa.yaml", "sherpa", "sherpa_onnx", _sherpa_backend_cls),
        ("configs/funasr.yaml", "funasr", "funasr", _funasr_backend_cls),
    ],
)
def test_config_options_are_a_subset_of_the_backend_constructor_signature(
    config_path, backend_key, backend_type, cls_factory
):
    settings = load_settings(config_path)
    backend_def = settings.backends[backend_key]
    assert backend_def.type == backend_type

    cls = cls_factory()
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters) - {"self"}
    assert set(backend_def.options) <= accepted
