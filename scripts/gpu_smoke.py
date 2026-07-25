#!/usr/bin/env python3
"""GPU smoke test for the qwen3asr STT server (Docker `gpu` profile).

Run against a server already listening on http://localhost:8000 (the compose
`gpu` service publishes :8000). Builds a 16 kHz mono WAV from the committed PCM
fixture, uploads it to ``POST /v1/audio/transcriptions``, and prints the
transcript, the end-to-end latency, and the GPU memory/utilization peak during
the request (sampled from ``nvidia-smi`` on the host, which sees the same GPU
the container uses).

Self-contained — stdlib only, no venv needed:

    python3 scripts/gpu_smoke.py
    python3 scripts/gpu_smoke.py --url http://localhost:8000 --model qwen3-asr-0.6b

Exits nonzero if the request fails or returns an empty transcript.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PCM = REPO_ROOT / "tests" / "fixtures" / "speech_16k_mono_s16le.pcm"


# --------------------------------------------------------------------------- #
# Audio + multipart helpers (stdlib only).
# --------------------------------------------------------------------------- #
def pcm_to_wav(pcm: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def build_multipart(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str = "audio/wav",
) -> tuple[bytes, str]:
    boundary = "----stt-smoke-" + format(time.monotonic_ns(), "x")
    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(f"{value}\r\n".encode())
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), boundary


# --------------------------------------------------------------------------- #
# nvidia-smi sampling.
# --------------------------------------------------------------------------- #
def _query_gpu(gpu_index: int) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        mem, util = (int(x) for x in out.split(","))
        return mem, util
    except Exception:
        return None


class GpuSampler:
    """Poll nvidia-smi in a background thread; keep the peak mem/util seen."""

    def __init__(self, gpu_index: int = 0, interval: float = 0.04) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.peak_mem_mib = 0
        self.peak_util_pct = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            s = _query_gpu(self.gpu_index)
            if s:
                mem, util = s
                self.peak_mem_mib = max(self.peak_mem_mib, mem)
                self.peak_util_pct = max(self.peak_util_pct, util)
                self.samples += 1

    def __enter__(self) -> GpuSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# HTTP helpers.
# --------------------------------------------------------------------------- #
def http_get(url: str, headers: dict[str, str], timeout: float = 10) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - localhost
        return r.status, r.read()


def http_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - localhost
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def extract_text(resp: bytes, response_format: str) -> str:
    if response_format == "text":
        return resp.decode(errors="replace").strip()
    try:
        return json.loads(resp).get("text", "").strip()
    except Exception:
        return resp.decode(errors="replace").strip()


def scrape_metric(metrics_text: str, name: str) -> str | None:
    """Return the last sample line for a simple unaggregated Prometheus metric."""
    last = None
    for line in metrics_text.splitlines():
        if line.startswith(name) and not line.startswith(name + "_"):
            last = line
    return last


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://localhost:8000", help="server base URL")
    ap.add_argument("--model", default="qwen3-asr-0.6b", help="model id (config `models` key)")
    ap.add_argument("--pcm", default=str(DEFAULT_PCM), help="PCM16 mono 16k input file")
    ap.add_argument("--token", default=None, help="bearer token if auth.tokens is set")
    ap.add_argument("--language", default=None, help="optional language hint")
    ap.add_argument("--response-format", default="json", choices=["json", "text", "verbose_json"])
    ap.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index")
    ap.add_argument("--timeout", type=float, default=120.0, help="POST timeout seconds")
    args = ap.parse_args()

    pcm = Path(args.pcm).read_bytes()
    wav = pcm_to_wav(pcm)
    print(
        f"input: {args.pcm}  ({len(pcm)} bytes PCM -> {len(wav)} bytes WAV, "
        f"{len(pcm) // 2 / 16000:.2f}s @ 16 kHz mono)"
    )

    headers: dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    # Readiness check (also confirms the server is reachable before timing).
    try:
        st, body = http_get(f"{args.url}/readyz", headers)
        print(f"GET /readyz -> {st} {body.decode().strip()}")
    except Exception as e:
        print(f"!! GET /readyz failed: {e} (is the server up on {args.url}?)", file=sys.stderr)

    have_nvidia_smi = shutil.which("nvidia-smi") is not None
    if not have_nvidia_smi:
        print("!! nvidia-smi not found on host; GPU stats will be unavailable", file=sys.stderr)
    else:
        idle = _query_gpu(args.gpu)
        if idle:
            print(f"GPU idle: {idle[0]} MiB used, {idle[1]}% util")

    # Build + send the transcription request, sampling GPU around it.
    fields = {"model": args.model, "response_format": args.response_format}
    if args.language:
        fields["language"] = args.language
    body_bytes, boundary = build_multipart(fields, "file", "speech.wav", wav)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    url = f"{args.url}/v1/audio/transcriptions"
    print(f"POST {url}  (model={args.model}, response_format={args.response_format}) ...")
    t0 = time.perf_counter()
    if have_nvidia_smi:
        with GpuSampler(gpu_index=args.gpu) as sampler:
            status, resp = http_post(url, body_bytes, headers, timeout=args.timeout)
    else:
        sampler = None
        status, resp = http_post(url, body_bytes, headers, timeout=args.timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"-> HTTP {status} in {latency_ms:.0f} ms")
    if status != 200:
        print(f"!! response: {resp.decode(errors='replace')}", file=sys.stderr)
        return 2

    text = extract_text(resp, args.response_format)
    print(f"transcript: {text!r}")
    if have_nvidia_smi:
        print(
            f"GPU during request: peak {sampler.peak_mem_mib} MiB used, "
            f"{sampler.peak_util_pct}% util ({sampler.samples} samples)"
        )

    # Optional: server-side latency from /metrics.
    try:
        _, mb = http_get(f"{args.url}/metrics", headers)
        m = scrape_metric(mb.decode(errors="replace"), "stt_final_latency_ms")
        if m:
            print(f"server metric: {m}")
    except Exception:
        pass

    if not text:
        print("!! empty transcript", file=sys.stderr)
        return 3
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
