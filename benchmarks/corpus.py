"""LibriSpeech corpus tooling: download, manifest sampling, FLAC decode.

Layout (OpenSLR LibriSpeech, e.g. `test-clean.tar.gz`):
    LibriSpeech/<split>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    LibriSpeech/<split>/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt

`download_subset` returns the path to `<dest>/LibriSpeech/<split>`, which is
the `split_dir` `build_manifest` expects.
"""

from __future__ import annotations

import logging
import random
import re
import string
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "benchmarks" / "data"

LIBRISPEECH_URLS: dict[str, str] = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}


@dataclass(frozen=True)
class Utterance:
    id: str
    flac_path: Path
    ref_text: str
    duration_s: float


def download_subset(split: str, dest: Path = DEFAULT_DATA_DIR) -> Path:
    """Download + extract a LibriSpeech split under `dest`.

    Idempotent: if `dest/LibriSpeech/<split>` already exists and is
    non-empty, skips the download. Stdlib-only (urllib + tarfile), with the
    same path-traversal guard as `scripts/download_models.py`."""
    if split not in LIBRISPEECH_URLS:
        raise ValueError(f"unknown split {split!r}; available: {sorted(LIBRISPEECH_URLS)}")

    split_dir = dest / "LibriSpeech" / split
    if split_dir.exists() and any(split_dir.iterdir()):
        return split_dir

    dest.mkdir(parents=True, exist_ok=True)
    url = LIBRISPEECH_URLS[split]
    tmp_archive = dest / f".{split}.tar.gz.part"
    try:
        urllib.request.urlretrieve(url, tmp_archive)  # noqa: S310 -- fixed OpenSLR URL

        with tarfile.open(tmp_archive, mode="r:gz") as tar:
            # Guard against path traversal from a malicious/corrupt archive:
            # every member must extract to a path under `dest`.
            for member in tar.getmembers():
                member_path = (dest / member.name).resolve()
                if not member_path.is_relative_to(dest.resolve()):
                    raise RuntimeError(
                        f"archive member {member.name!r} escapes {dest}; aborting"
                    )
            tar.extractall(dest)  # noqa: S202 -- paths validated above
    finally:
        tmp_archive.unlink(missing_ok=True)

    if not split_dir.exists():
        raise RuntimeError(
            f"extracted archive for {split!r} did not produce expected directory {split_dir}"
        )
    return split_dir


def _iter_utterances(split_dir: Path):
    for trans_path in sorted(split_dir.glob("*/*/*.trans.txt")):
        chap_dir = trans_path.parent
        for line in trans_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            flac_path = chap_dir / f"{utt_id}.flac"
            yield utt_id, flac_path, text


def build_manifest(split_dir: Path, n: int, seed: int) -> list[Utterance]:
    """Seeded sample of `n` utterances from `split_dir` (deterministic for a
    given seed: same seed + same corpus -> same subset, same order)."""
    all_utts = sorted(_iter_utterances(split_dir), key=lambda t: t[0])
    if n > len(all_utts):
        raise ValueError(
            f"requested n={n} utterances but split only has {len(all_utts)}"
        )

    try:
        import soundfile  # noqa: PLC0415 -- optional (bench extra) import
    except ImportError:
        soundfile = None
        logger.debug(
            "soundfile not installed (bench extra); manifest duration_s will be 0.0"
        )

    chosen = random.Random(seed).sample(all_utts, n)
    manifest: list[Utterance] = []
    for utt_id, flac_path, text in chosen:
        # duration_s is informational metadata, not required to build a
        # manifest: a missing `soundfile` or an unreadable FLAC yields 0.0
        # (logged, not raised).
        duration_s = 0.0
        if soundfile is not None:
            try:
                info = soundfile.info(flac_path)
                duration_s = info.frames / info.samplerate
            except Exception as exc:
                logger.warning(
                    "could not read duration of %s (%s); recording duration_s=0.0",
                    flac_path,
                    exc,
                )
        manifest.append(
            Utterance(id=utt_id, flac_path=flac_path, ref_text=text, duration_s=duration_s)
        )
    return manifest


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    lowered = s.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", lowered).strip()


def load_pcm16(utt: Utterance) -> bytes:
    """Decode `utt.flac_path` and resample to 16 kHz mono PCM16 bytes.

    Requires the `bench` extra (`soundfile`, which ships bundled libsndfile
    wheels so no FLAC decode plan needs a system ffmpeg/libsndfile install).
    Resampling uses stdlib `audioop.ratecv` — Python 3.13 removed the
    `audioop` module; this repo targets 3.12, where it's still available.
    """
    import audioop  # noqa: PLC0415 -- Python 3.12 only (removed in 3.13)

    import soundfile  # noqa: PLC0415 -- optional (bench extra) import

    data, samplerate = soundfile.read(utt.flac_path, dtype="int16", always_2d=False)
    if data.ndim > 1:
        # Downmix to mono by taking the first channel (LibriSpeech is mono
        # already; this is just a defensive fallback).
        data = data[:, 0]
    pcm16 = data.tobytes()

    if samplerate != 16000:
        pcm16, _ = audioop.ratecv(pcm16, 2, 1, samplerate, 16000, None)
    return pcm16
