"""NVIDIA Nemotron cache-aware streaming backend (optional `nemotron` extra).

Importing this package (and its `backend` module) never requires
`nemo_toolkit` (or `torch`/`numpy`) to be installed — every heavy import is
deferred to backend construction (`importlib.util.find_spec` gate only) and
to `start()` / the decode path, so `import stt_server.backends` always
succeeds on an ML-free install. The module-level language helpers
(`normalize_language`, `split_lang_tag`, `locale_to_language_name`,
`chunk_samples_for`) are pure stdlib and are unit-tested without the extra.
"""

from __future__ import annotations

from stt_server.backends.nemotron.backend import (
    LANG_TAG_RE,
    SUPPORTED_LOCALES,
    NemotronBackend,
    NemotronStream,
    chunk_samples_for,
    locale_to_language_name,
    normalize_language,
    split_lang_tag,
)

__all__ = [
    "LANG_TAG_RE",
    "SUPPORTED_LOCALES",
    "NemotronBackend",
    "NemotronStream",
    "chunk_samples_for",
    "locale_to_language_name",
    "normalize_language",
    "split_lang_tag",
]
