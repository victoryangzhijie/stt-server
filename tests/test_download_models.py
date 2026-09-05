"""Unmarked tests for scripts/download_models.py's prewarm handling.

The script is stdlib-only and importable without any optional extra; these
tests exercise the funasr prewarm entry's failure paths with `funasr` absent
(as it is in the default dev environment)."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_models.py"

FUNASR_AVAILABLE = importlib.util.find_spec("funasr") is not None


def _load_script_module():
    spec = importlib.util.spec_from_file_location("download_models_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


def test_script_is_importable_stdlib_only():
    mod = _load_script_module()
    assert "paraformer-zh-streaming" in mod.REGISTRY
    assert mod.REGISTRY["paraformer-zh-streaming"].prewarm is not None


@pytest.mark.skipif(FUNASR_AVAILABLE, reason="only valid when funasr is NOT installed")
def test_all_skips_failing_prewarm_and_continues(tmp_path, capsys, monkeypatch):
    mod = _load_script_module()

    funasr_artifact = dataclasses.replace(
        mod.REGISTRY["paraformer-zh-streaming"],
        dest=tmp_path / "funasr.prewarmed",
    )

    ran = []
    ok_artifact = mod.Artifact(
        name="later-artifact",
        url="prewarm://informational",
        dest=tmp_path / "later.prewarmed",
        min_size_bytes=0,
        prewarm=lambda: ran.append(True),
    )

    # funasr entry deliberately FIRST: the bug being pinned down was the
    # unhandled RuntimeError aborting the loop so later artifacts never ran.
    monkeypatch.setattr(
        mod, "REGISTRY", {"paraformer-zh-streaming": funasr_artifact, "later-artifact": ok_artifact}
    )

    rc = mod.main(["all"])

    assert rc == 0
    err = capsys.readouterr().err
    assert "[skip] paraformer-zh-streaming:" in err
    assert "funasr is not installed" in err
    assert "stt-server[funasr]" in err
    assert ran == [True], "artifact after the failing prewarm was never processed"
    assert ok_artifact.dest.exists()
    assert not funasr_artifact.dest.exists(), "failed prewarm must not leave a marker"


@pytest.mark.skipif(FUNASR_AVAILABLE, reason="only valid when funasr is NOT installed")
def test_named_prewarm_invocation_exits_nonzero_with_actionable_message(
    tmp_path, capsys, monkeypatch
):
    mod = _load_script_module()

    funasr_artifact = dataclasses.replace(
        mod.REGISTRY["paraformer-zh-streaming"],
        dest=tmp_path / "funasr.prewarmed",
    )
    monkeypatch.setattr(mod, "REGISTRY", {"paraformer-zh-streaming": funasr_artifact})

    rc = mod.main(["paraformer-zh-streaming"])

    assert rc != 0
    err = capsys.readouterr().err
    assert "[error] paraformer-zh-streaming:" in err
    assert "funasr is not installed" in err
    assert "stt-server[funasr]" in err
    assert not funasr_artifact.dest.exists()


def test_nemotron_registry_entry_is_a_prewarm():
    mod = _load_script_module()
    artifact = mod.REGISTRY["nemotron-3.5-asr-streaming-0.6b"]
    assert artifact.prewarm is not None
    # The prewarm writes the real .nemo here, so `dest` doubles as the
    # idempotency marker (see the artifact's comment in the script).
    assert artifact.dest.name == "nemotron-3.5-asr-streaming-0.6b.nemo"
    assert artifact.dest.parent.name == "nemotron-3.5-asr-streaming-0.6b"


def test_nemotron_prewarm_turns_missing_huggingface_hub_into_runtimeerror(monkeypatch):
    mod = _load_script_module()

    # Force the deferred `from huggingface_hub import hf_hub_download` to
    # raise ImportError regardless of whether the package is installed in
    # this environment: a None entry in sys.modules makes the import
    # machinery raise instead of finding/loading the real module.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(RuntimeError) as excinfo:
        mod.REGISTRY["nemotron-3.5-asr-streaming-0.6b"].prewarm()

    message = str(excinfo.value)
    assert "huggingface_hub is not installed" in message
    assert "stt-server[nemotron]" in message


def test_named_nemotron_prewarm_invocation_exits_nonzero_without_huggingface_hub(
    tmp_path, capsys, monkeypatch
):
    mod = _load_script_module()

    artifact = dataclasses.replace(
        mod.REGISTRY["nemotron-3.5-asr-streaming-0.6b"],
        dest=tmp_path / "nemotron.nemo",
    )
    monkeypatch.setattr(mod, "REGISTRY", {"nemotron-3.5-asr-streaming-0.6b": artifact})
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    rc = mod.main(["nemotron-3.5-asr-streaming-0.6b"])

    assert rc != 0
    err = capsys.readouterr().err
    assert "[error] nemotron-3.5-asr-streaming-0.6b:" in err
    assert "huggingface_hub is not installed" in err
    assert not artifact.dest.exists(), "failed prewarm must not leave a marker"
