"""Benchmark suite for stt-server (spec §10).

Top-level package (NOT under src/) so nothing in `src/stt_server` ever
depends on it — see the layering rule in the plan-4 constraints. Requires
the `bench` extra (`uv sync --extra bench`) for corpus/scoring tooling;
`benchmarks.server` / `benchmarks.client_ws` / `benchmarks.client_file` only
need the base dev environment (httpx, websockets already present via
uvicorn[standard])."""

from __future__ import annotations
