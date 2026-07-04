"""Shared PCM16<->float32 conversion helper for streaming backends.

Extracted from the sherpa backend so both `sherpa` and `funasr` backends
(and any future one) share a single implementation. `numpy` stays lazy /
optional here too: importing this module never requires numpy, and the
pure-stdlib fallback keeps working without it.
"""

from __future__ import annotations

import array
import importlib.util
from typing import Any


def _numpy_available() -> bool:
    return importlib.util.find_spec("numpy") is not None


def pcm16_bytes_to_float32(data: bytes, use_numpy: bool | None = None) -> Any:
    """Convert little-endian PCM16 bytes to float32 samples in [-1, 1].

    Uses numpy when available (and not explicitly disabled) for speed;
    falls back to a pure-stdlib `array("h")` implementation so this helper
    (and everything that calls it) works without numpy installed.
    """
    if use_numpy is None:
        use_numpy = _numpy_available()

    if use_numpy:
        import numpy as np

        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0

    samples = array.array("h")
    samples.frombytes(data)
    return [s / 32768.0 for s in samples]
