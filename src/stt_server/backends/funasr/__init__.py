"""FunASR Paraformer streaming backend (optional `funasr` extra).

Importing this package (and its `backend` module) never requires `funasr`
(or `torch`) to be installed — that import is deferred to backend
construction / start() so `import stt_server.backends` always succeeds.
"""

from __future__ import annotations

from stt_server.backends.funasr.backend import FunasrBackend, FunasrStream

__all__ = ["FunasrBackend", "FunasrStream"]
