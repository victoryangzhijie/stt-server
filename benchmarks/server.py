"""Spawn a real `stt-server` subprocess for benchmarks (and their tests).

Runs the server out-of-process (not in-process via `TestClient`) because
benchmarks need a real listening socket for `websockets` streaming and, for
GPU/venv-pinned backends (sherpa, funasr), a specific interpreter — see the
`python` parameter."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


class ServerUnderTest:
    """Context manager: boots `python -m stt_server --config ... --host ... --port ...`
    and waits for `/readyz` to return 200 before `__enter__` returns.

    The child's stdout+stderr are captured to `.log_path` (a per-instance
    tempfile) so readiness failures and mid-benchmark crashes stay
    debuggable after the fact — the log path is included in every raised
    readiness error."""

    def __init__(
        self,
        config_path: str,
        port: int = 8100,
        python: str | None = None,
        env: dict | None = None,
        ready_timeout_s: float = 120.0,
    ) -> None:
        self.config_path = config_path
        self.port = port
        self.python = python or sys.executable
        # Merged over os.environ at launch (never replacing it): a bare
        # `env=` Popen kwarg would drop PATH/HOME/venv vars the child needs.
        self.env = env
        self.ready_timeout_s = ready_timeout_s
        self.log_path: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def base_ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def __enter__(self) -> ServerUnderTest:
        cmd = [
            self.python,
            "-m",
            "stt_server",
            "--config",
            self.config_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]
        fd, log_name = tempfile.mkstemp(prefix=f"stt-server-{self.port}-", suffix=".log")
        self.log_path = Path(log_name)
        self._log_file = os.fdopen(fd, "wb")
        # `start_new_session=True` puts the server in its own process
        # group/session (setsid). The qwen3asr backend's vLLM engine spawns an
        # `EngineCore` child process (multiprocessing spawn) that holds the
        # GPU's KV-cache memory; vLLM's `LLM` exposes no public close() and the
        # child is reaped only on clean interpreter exit, NOT on SIGTERM. A
        # signal to just the parent therefore orphans EngineCore -- measured
        # directly: a SIGTERM'd server left EngineCore alive holding ~16.6 GiB,
        # OOMing the next server boot ("Free memory ... less than desired GPU
        # memory utilization"). Starting in a new session lets `_terminate`
        # kill the whole group (server + EngineCore) via killpg.
        self._proc = subprocess.Popen(
            cmd,
            env={**os.environ, **(self.env or {})},
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self._wait_ready(timeout_s=self.ready_timeout_s)
        except BaseException:
            # __exit__ never runs when __enter__ raises — tear the child
            # down here (same SIGTERM -> wait -> SIGKILL shape) or it leaks
            # as a live orphan holding the port.
            self._terminate()
            raise
        return self

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        url = f"{self.base_url}/readyz"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"server process exited early (code {self._proc.returncode}) "
                    f"before becoming ready; config={self.config_path!r}; "
                    f"log={self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:  # noqa: S310
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
            time.sleep(0.1)
        raise TimeoutError(
            f"server did not become ready within {timeout_s}s at {url}; "
            f"log={self.log_path}"
        ) from last_error

    def _terminate(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            # Kill the whole process group (set up by start_new_session=True in
            # __enter__), not just the parent: the qwen3asr/vLLM backend spawns
            # an EngineCore child that survives a parent-only SIGTERM and leaks
            # GPU memory (see __enter__). killpg(getpgid(pid)) reaches it.
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pgid = None
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pgid, signal.SIGKILL)
                proc.kill()
                proc.wait(timeout=10.0)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self._terminate()
