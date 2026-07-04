"""Qwen3-ASR backend (optional `qwen3asr` extra).

Importing this package (and its `backend` module) never requires `qwen_asr`
(or `vllm`/`torch`) to be installed — that import is deferred to backend
construction / start() so `import stt_server.backends` always succeeds.
"""

from __future__ import annotations

from stt_server.backends.qwen3asr.backend import Qwen3AsrBackend, Qwen3AsrStream

__all__ = ["Qwen3AsrBackend", "Qwen3AsrStream"]
