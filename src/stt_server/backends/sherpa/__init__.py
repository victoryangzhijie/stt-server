"""sherpa-onnx streaming Zipformer backend (optional `sherpa` extra).

Importing this package (and its `backend` module) never requires
`sherpa_onnx` to be installed — that import is deferred to backend
construction / start() so `import stt_server.backends` always succeeds.
"""

from __future__ import annotations

from stt_server.backends.sherpa.backend import SherpaBackend, SherpaStream

__all__ = ["SherpaBackend", "SherpaStream"]
