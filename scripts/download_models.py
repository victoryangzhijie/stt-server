#!/usr/bin/env python3
"""Download model artifacts used by optional backends/detectors.

Usage:
    python scripts/download_models.py <name>
    python scripts/download_models.py all

Idempotent: skips any artifact whose destination file already exists and
passes the size sanity check. Uses only the stdlib (urllib) — no extra
runtime dependency is required just to fetch model weights.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    dest: Path
    min_size_bytes: int
    # When set, `dest` is treated as a directory: the downloaded file at
    # `url` is a tar archive (e.g. .tar.bz2) that gets extracted under
    # `models/`, and `archive_root` is the top-level directory name the
    # archive extracts to (used for the idempotency/presence check instead
    # of a single file's size).
    archive_root: str | None = None
    # When set, this artifact isn't fetched via `urllib` at all: `url` is
    # informational only (points at the model's page for reference) and
    # `download()` instead calls this zero-arg callable, then touches `dest`
    # as a marker file so re-running the script is a cheap no-op. Used for
    # FunASR models, which modelscope downloads into its own cache directory
    # automatically the first time `AutoModel(model=...)` is constructed —
    # there is no single artifact URL for this script to `urlretrieve`.
    prewarm: Callable[[], None] | None = None


def _prewarm_funasr_paraformer_zh_streaming() -> None:
    """Pre-warm FunASR's `paraformer-zh-streaming` model cache.

    FunASR pulls model weights from modelscope automatically the first time
    `AutoModel(model=...)` is constructed (into modelscope's own cache dir,
    not `models/`); this just triggers that download ahead of time so the
    first real request doesn't pay for it. The `funasr` import is deferred
    to this function body (not module level) so this script stays
    importable/runnable (`all`, other artifacts) with only the stdlib and
    without the optional `funasr` extra installed.
    """
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "funasr is not installed; pip install 'stt-server[funasr]' before "
            "pre-warming paraformer-zh-streaming"
        ) from exc

    AutoModel(model="paraformer-zh-streaming")


REGISTRY: dict[str, Artifact] = {
    "silero": Artifact(
        name="silero",
        url=(
            "https://github.com/snakers4/silero-vad/raw/v5.1/"
            "src/silero_vad/data/silero_vad.onnx"
        ),
        dest=MODELS_DIR / "silero_vad.onnx",
        min_size_bytes=1_000_000,  # official v5 asset is ~2.3 MB
    ),
    "sherpa-zipformer-en": Artifact(
        name="sherpa-zipformer-en",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
        ),
        dest=MODELS_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        min_size_bytes=300_000_000,  # official asset is ~310 MB
        archive_root="sherpa-onnx-streaming-zipformer-en-2023-06-26",
    ),
    "paraformer-zh-streaming": Artifact(
        name="paraformer-zh-streaming",
        url=(
            "https://modelscope.cn/models/iic/"
            "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
            "  (informational only — see `prewarm` below, not urlretrieve'd)"
        ),
        dest=MODELS_DIR / ".funasr-paraformer-zh-streaming.prewarmed",
        min_size_bytes=0,
        prewarm=_prewarm_funasr_paraformer_zh_streaming,
    ),
    # Future entries (added by later tasks): "paraformer-en", ...
}


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.0f}TB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download(artifact: Artifact, *, force: bool = False) -> None:
    if artifact.prewarm is not None:
        _prewarm(artifact, force=force)
        return

    if artifact.archive_root is not None:
        _download_archive(artifact, force=force)
        return

    if artifact.dest.exists() and not force:
        size = artifact.dest.stat().st_size
        if size >= artifact.min_size_bytes:
            print(
                f"[skip] {artifact.name}: already present at {artifact.dest} "
                f"({_human_size(size)})"
            )
            return
        print(
            f"[warn] {artifact.name}: existing file at {artifact.dest} is only "
            f"{_human_size(size)} (< {_human_size(artifact.min_size_bytes)}); re-downloading"
        )

    artifact.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = artifact.dest.with_suffix(artifact.dest.suffix + ".part")
    print(f"[fetch] {artifact.name}: {artifact.url} -> {artifact.dest}")
    try:
        urllib.request.urlretrieve(artifact.url, tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    size = tmp_path.stat().st_size
    if size < artifact.min_size_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded {artifact.name} is only {_human_size(size)}, "
            f"expected at least {_human_size(artifact.min_size_bytes)} "
            "(download likely failed or was truncated)"
        )

    tmp_path.replace(artifact.dest)
    print(f"[done] {artifact.name}: {_human_size(size)} written to {artifact.dest}")


def _prewarm(artifact: Artifact, *, force: bool = False) -> None:
    """Run `artifact.prewarm()` to populate a third-party cache (e.g.
    modelscope's), then touch `artifact.dest` as a marker file so a second
    invocation is a cheap no-op instead of re-running the (network-bound,
    possibly slow) prewarm callable.
    """
    assert artifact.prewarm is not None
    if artifact.dest.exists() and not force:
        print(f"[skip] {artifact.name}: already pre-warmed (marker at {artifact.dest})")
        return

    print(f"[prewarm] {artifact.name}: {artifact.url}")
    artifact.prewarm()

    artifact.dest.parent.mkdir(parents=True, exist_ok=True)
    artifact.dest.touch()
    print(f"[done] {artifact.name}: pre-warmed; marker written to {artifact.dest}")


def _download_archive(artifact: Artifact, *, force: bool = False) -> None:
    """Download a tar archive and extract it under `models/`.

    `artifact.dest` names the expected extracted directory; presence +
    on-disk size of that directory (not a single file) drives idempotency.
    """
    if artifact.dest.exists() and not force:
        size = _dir_size(artifact.dest)
        if size >= artifact.min_size_bytes:
            print(
                f"[skip] {artifact.name}: already present at {artifact.dest} "
                f"({_human_size(size)})"
            )
            return
        print(
            f"[warn] {artifact.name}: existing dir at {artifact.dest} is only "
            f"{_human_size(size)} (< {_human_size(artifact.min_size_bytes)}); re-downloading"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_archive = MODELS_DIR / f".{artifact.name}.tar.bz2.part"
    print(f"[fetch] {artifact.name}: {artifact.url} -> {tmp_archive}")
    try:
        urllib.request.urlretrieve(artifact.url, tmp_archive)

        size = tmp_archive.stat().st_size
        if size < artifact.min_size_bytes:
            raise RuntimeError(
                f"downloaded {artifact.name} is only {_human_size(size)}, "
                f"expected at least {_human_size(artifact.min_size_bytes)} "
                "(download likely failed or was truncated)"
            )

        print(f"[extract] {artifact.name}: {tmp_archive} -> {MODELS_DIR}")
        with tarfile.open(tmp_archive, mode="r:bz2") as tar:
            # Guard against path traversal from a malicious/corrupt archive:
            # every member must extract to a path under MODELS_DIR.
            for member in tar.getmembers():
                member_path = (MODELS_DIR / member.name).resolve()
                if not member_path.is_relative_to(MODELS_DIR.resolve()):
                    raise RuntimeError(
                        f"archive member {member.name!r} escapes {MODELS_DIR}; aborting"
                    )
            tar.extractall(MODELS_DIR)  # noqa: S202 -- paths validated above
    finally:
        tmp_archive.unlink(missing_ok=True)

    if not artifact.dest.exists():
        raise RuntimeError(
            f"extracted archive for {artifact.name} did not produce expected "
            f"directory {artifact.dest}"
        )

    final_size = _dir_size(artifact.dest)
    print(f"[done] {artifact.name}: {_human_size(final_size)} extracted to {artifact.dest}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        names = ", ".join(sorted(REGISTRY)) + ", all"
        print(f"usage: {Path(sys.argv[0]).name} <name>\n  available: {names}", file=sys.stderr)
        return 2

    requested = argv[0]
    if requested == "all":
        targets = list(REGISTRY.values())
    elif requested in REGISTRY:
        targets = [REGISTRY[requested]]
    else:
        names = ", ".join(sorted(REGISTRY)) + ", all"
        print(f"unknown artifact {requested!r}; available: {names}", file=sys.stderr)
        return 2

    for artifact in targets:
        try:
            download(artifact)
        except RuntimeError as exc:
            if artifact.prewarm is None:
                raise  # only prewarm failures get the clean-skip treatment
            if requested == "all":
                # A prewarm entry needing an optional extra (e.g. funasr)
                # that isn't installed must not abort the whole `all` run:
                # report it cleanly and keep downloading the rest.
                print(f"[skip] {artifact.name}: {exc}", file=sys.stderr)
                continue
            # Named single-artifact invocation: actionable message, non-zero
            # exit, no traceback (consistent with the unknown-artifact path).
            print(f"[error] {artifact.name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
