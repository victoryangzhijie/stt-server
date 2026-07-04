"""Background CPU/RSS(/GPU) resource sampler for load-test rungs (spec §10.3).

`ResourceSampler(pid).start()` spawns a daemon thread that polls
`psutil.Process(pid)` (plus all descendants, summed, so a server that forks
worker processes is covered even though today's backends only spawn
threads) every `interval_s` seconds; `.stop()` joins the thread and returns
a dict of time series + peaks.

GPU sampling is opportunistic: `pynvml` is an optional (`bench` extra)
dependency, and even when installed there may be no NVIDIA device visible
(CPU-only dev boxes, CI). Both cases fail silently -- the returned dict
simply omits `gpu_util_pct`/`gpu_mem_mb` -- because GPU telemetry is a nice-
to-have, not a requirement, for a benchmark run.
"""

from __future__ import annotations

import threading
import time

import psutil

try:
    import pynvml
except ImportError:  # pragma: no cover -- exercised only when pynvml absent
    pynvml = None


def _init_nvml_handle():
    """Return an NVML device handle for GPU index 0, or `None` if `pynvml`
    isn't importable or no NVIDIA device/driver is present. Never raises --
    absence of GPU telemetry is expected on most dev/CI machines."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() < 1:
            return None
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def _sample_gpu(handle) -> tuple[float, float] | None:
    """`(util_pct, mem_used_mb)` for `handle`, or `None` on any NVML error
    (e.g. the device disappears mid-run)."""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(util.gpu), mem.used / (1024 * 1024)
    except Exception:
        return None


class ResourceSampler:
    """Samples `pid` (+ all live descendants) on a background thread.

    `psutil.Process.cpu_percent()` requires an initial "priming" call whose
    result is meaningless (it measures CPU since the process started, or
    since the *previous* call) -- so the constructor primes every process
    it can see once, and every sample after that reports the delta since the
    prior interval, which is what makes the returned series meaningful.
    """

    def __init__(self, pid: int, interval_s: float = 1.0) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[dict] = []
        self._nvml_handle = _init_nvml_handle()

    def _live_processes(self) -> list[psutil.Process]:
        """The root process plus all live descendants. Returns `[]` if the
        root has already exited (psutil.NoSuchProcess) -- the caller treats
        that as "nothing to sample this tick", not an error."""
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return []
        procs = [root]
        try:
            procs.extend(root.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
        return procs

    def _prime(self) -> None:
        for proc in self._live_processes():
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _take_sample(self) -> dict | None:
        """One sample point, summing CPU%/RSS across the root + all live
        descendants. Returns `None` if the root process is already gone (the
        run loop stops scheduling further samples once this happens, but the
        already-collected series is preserved)."""
        procs = self._live_processes()
        if not procs:
            return None

        cpu_pct = 0.0
        rss_mb = 0.0
        for proc in procs:
            try:
                cpu_pct += proc.cpu_percent(interval=None)
                rss_mb += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sample = {"t": time.monotonic(), "cpu_pct": cpu_pct, "rss_mb": rss_mb}

        if self._nvml_handle is not None:
            gpu = _sample_gpu(self._nvml_handle)
            if gpu is not None:
                sample["gpu_util_pct"], sample["gpu_mem_mb"] = gpu

        return sample

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._take_sample()
            if sample is None:
                # Root process died mid-sample: stop scheduling further
                # ticks (nothing left to sample) but leave already-collected
                # data for `.stop()` to summarize.
                return
            self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        self._prime()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and return the collected series + peaks:
        `{"samples": [...], "cpu_pct_peak", "rss_mb_peak",
        "gpu_util_pct_peak"?, "gpu_mem_mb_peak"?}` (GPU peak keys are
        omitted entirely if no sample ever had GPU data)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2 + 5.0)

        samples = self._samples
        result: dict = {
            "samples": samples,
            "cpu_pct_peak": max((s["cpu_pct"] for s in samples), default=0.0),
            "rss_mb_peak": max((s["rss_mb"] for s in samples), default=0.0),
        }

        gpu_utils = [s["gpu_util_pct"] for s in samples if "gpu_util_pct" in s]
        gpu_mems = [s["gpu_mem_mb"] for s in samples if "gpu_mem_mb" in s]
        if gpu_utils:
            result["gpu_util_pct_peak"] = max(gpu_utils)
        if gpu_mems:
            result["gpu_mem_mb_peak"] = max(gpu_mems)

        return result
