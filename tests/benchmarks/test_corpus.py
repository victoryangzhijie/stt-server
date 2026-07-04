"""Tests for benchmarks/corpus.py: manifest building, text normalization,
and (extra-gated) FLAC decode. No network access — download_subset itself
is exercised manually per the plan, never in the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.corpus import Utterance, build_manifest, normalize_text


def _write_fake_librispeech(root: Path) -> Path:
    """Fabricate a tiny LibriSpeech-shaped tree under `root`:

        <root>/test-clean/<spk>/<chap>/<spk>-<chap>-<utt>.flac
        <root>/test-clean/<spk>/<chap>/<spk>-<chap>.trans.txt
    """
    split_dir = root / "test-clean"
    layout = {
        ("19", "198"): ["0000", "0001", "0002"],
        ("19", "227"): ["0000", "0001"],
        ("26", "495"): ["0000"],
    }
    for (spk, chap), utts in layout.items():
        chap_dir = split_dir / spk / chap
        chap_dir.mkdir(parents=True)
        trans_lines = []
        for utt in utts:
            utt_id = f"{spk}-{chap}-{utt}"
            (chap_dir / f"{utt_id}.flac").write_bytes(b"fake-flac-bytes")
            trans_lines.append(f"{utt_id} HELLO WORLD {utt}")
        (chap_dir / f"{spk}-{chap}.trans.txt").write_text("\n".join(trans_lines) + "\n")
    return split_dir


def test_build_manifest_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    split_dir = _write_fake_librispeech(tmp_path)

    first = build_manifest(split_dir, n=4, seed=42)
    second = build_manifest(split_dir, n=4, seed=42)

    assert [u.id for u in first] == [u.id for u in second]
    assert len(first) == 4


def test_build_manifest_seed_changes_selection(tmp_path: Path) -> None:
    split_dir = _write_fake_librispeech(tmp_path)

    a = build_manifest(split_dir, n=3, seed=1)
    b = build_manifest(split_dir, n=3, seed=2)

    assert [u.id for u in a] != [u.id for u in b]


def test_build_manifest_fields(tmp_path: Path) -> None:
    split_dir = _write_fake_librispeech(tmp_path)

    manifest = build_manifest(split_dir, n=6, seed=42)

    assert len(manifest) == 6
    for utt in manifest:
        assert isinstance(utt, Utterance)
        assert utt.flac_path.exists()
        assert utt.flac_path.name == f"{utt.id}.flac"
        assert utt.ref_text == "HELLO WORLD " + utt.id.rsplit("-", 1)[-1]
        assert utt.duration_s == 0.0  # fabricated .flac has no real audio; see load_pcm16


def test_build_manifest_n_larger_than_corpus_raises(tmp_path: Path) -> None:
    split_dir = _write_fake_librispeech(tmp_path)

    with pytest.raises(ValueError):
        build_manifest(split_dir, n=100, seed=42)


def test_normalize_text_lowercases_strips_punctuation_and_collapses_whitespace() -> None:
    assert normalize_text("Hello,  World!!  How's  it going?") == "hello world hows it going"


def test_normalize_text_empty() -> None:
    assert normalize_text("") == ""


def test_load_pcm16_decodes_and_resamples(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    import numpy as np
    from benchmarks.corpus import load_pcm16

    # Real FLAC at 8kHz mono to exercise both decode and resample paths.
    sr = 8000
    n = sr  # 1 second
    samples = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype("float32")
    flac_path = tmp_path / "19-198-0000.flac"
    soundfile.write(flac_path, samples, sr, format="FLAC")

    utt = Utterance(id="19-198-0000", flac_path=flac_path, ref_text="x", duration_s=1.0)
    pcm = load_pcm16(utt)

    assert isinstance(pcm, bytes)
    # 16kHz mono pcm16 for ~1 second of audio: roughly 32000 bytes.
    assert 30000 < len(pcm) < 34000
