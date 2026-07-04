"""Prometheus metrics: a leaf package.

Nothing under `stt_server.metrics` imports from the API layer, the session
core, or the backends package — only `prometheus_client`. This keeps the
dependency direction one-way: other packages import metrics, never the
reverse.
"""

from __future__ import annotations
