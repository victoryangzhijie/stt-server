#!/usr/bin/env python3
"""Live-server validation for the `nemotron` backend (target box: NVIDIA A10).

Runs against a server that is ALREADY listening (start it with
`uv run stt-server --config configs/nemotron.yaml`); it never boots one
itself. Everything it checks is a claim from the design that has never been
executed on real hardware -- see docs/nemotron_a10_runbook.md for the
surrounding procedure and the full UNVERIFIED checklist.

Cases (each labelled, each independently pass/fail):

  a. zh explicit     -- POST /v1/audio/transcriptions with language=zh
  b. en explicit     -- POST /v1/audio/transcriptions with language=en
  c. auto detect     -- the same two clips with NO language, over both the
                        native WS and the file HTTP endpoint (4 cases)
  d. final flush     -- stream a clip over WS and send {"type":"input_done"}
                        immediately after the last audio frame, with NO
                        trailing silence: a FINAL must still arrive with
                        non-empty text (this is the `finalize()` flush path;
                        the utterance tail is printed for manual review)
  e. concurrency     -- N parallel WS sessions at real-time pace, reporting
                        per-session first_partial_ms / final_ms (server-side,
                        from the FINAL event's `latency` object) and
                        client-side wall latency
  f. GPU + drops     -- nvidia-smi sampled in a background thread for the
                        whole run (peak VRAM / utilization), and
                        `stt_audio_dropped_total` scraped from /metrics
                        before and after: any increase FAILS the run, because
                        every number above would then describe behavior under
                        active audio shedding rather than the pipeline.

Language is NOT carried by the native WS protocol (see docs/backends.md §5),
so the explicit-language cases necessarily go through the file HTTP endpoint,
which does thread the `language` form field into `StreamConfig`.

Dependencies: the stdlib plus `websockets`, which the base install already
provides via `uvicorn[standard]` (the same dependency `examples/ws_client.py`
relies on). No `bench` extra required.

Usage:

    uv run python scripts/validate_nemotron_a10.py \
        --zh benchmarks/audio/zh_sample.wav \
        --en benchmarks/audio/en_sample.wav \
        --zh-ref "今天天气很好，我们去公园散步吧。" \
        --en-ref "the canoe slid on the smooth planks" \
        --concurrency 4 --json nemotron-a10.json

Exits nonzero if any case fails.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

try:
    import websockets
except ImportError:  # pragma: no cover - environment problem, not logic
    print(
        "!! the `websockets` package is missing. It normally ships with the base "
        "install via uvicorn[standard]; install it with "
        "`uv pip install --python .venv/bin/python websockets` (or install the "
        "`bench` extra, which also depends on it).",
        file=sys.stderr,
    )
    raise

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
DROPPED_METRIC = "stt_audio_dropped_total"


# --------------------------------------------------------------------------- #
# Audio I/O
# --------------------------------------------------------------------------- #
def load_pcm16(path: Path) -> bytes:
    """Read 16 kHz mono signed-16-bit-LE PCM from a `.wav` or raw `.pcm` file.

    WAV headers are validated rather than resampled: the server's pipeline
    assumes 16 kHz mono s16le end to end, and silently feeding it something
    else produces plausible-looking but meaningless transcripts. Convert
    first (`scripts/make_tts_fixtures.sh` does this for macOS `say` output;
    otherwise `ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav`).
    """
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(
                    f"{path}: need 16000 Hz mono 16-bit WAV, got "
                    f"{w.getframerate()} Hz / {w.getnchannels()} ch / "
                    f"{w.getsampwidth() * 8}-bit"
                )
            return w.readframes(w.getnframes())
    return path.read_bytes()


def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_duration_s(pcm: bytes) -> float:
    return len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)


# --------------------------------------------------------------------------- #
# Text scoring (tiny Levenshtein; no jiwer dependency)
# --------------------------------------------------------------------------- #
def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance over two token sequences (rolling-row DP, O(min) memory)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def _strip_punct(text: str) -> str:
    """Drop punctuation (Unicode category P*, both ASCII and CJK) and collapse
    case. ASR hypotheses from a streaming model routinely differ from a
    reference only in punctuation, which would otherwise dominate CER."""
    return "".join(c for c in text if not unicodedata.category(c).startswith("P")).lower()


def cer(reference: str, hypothesis: str) -> float | None:
    """Character error rate, whitespace and punctuation removed — the standard
    scoring unit for Chinese, which is not word-segmented."""
    ref = [c for c in _strip_punct(reference) if not c.isspace()]
    hyp = [c for c in _strip_punct(hypothesis) if not c.isspace()]
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float | None:
    """Word error rate over whitespace-split, punctuation-stripped tokens."""
    ref = _strip_punct(reference).split()
    hyp = _strip_punct(hypothesis).split()
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib only; same multipart builder shape as scripts/gpu_smoke.py)
# --------------------------------------------------------------------------- #
def build_multipart(fields: dict[str, str], filename: str, file_bytes: bytes) -> tuple[bytes, str]:
    boundary = "----stt-nemotron-" + format(time.monotonic_ns(), "x")
    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(f"{value}\r\n".encode())
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    body.write(b"Content-Type: audio/wav\r\n\r\n")
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), boundary


def http_get(url: str, headers: dict[str, str], timeout: float = 10.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-supplied
        return resp.status, resp.read()


def http_post(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Family-level totals from Prometheus text exposition (labels summed).

    A trimmed copy of `benchmarks/_drops.py::parse_prometheus_metrics`; this
    script deliberately does not import `benchmarks`, which is not part of
    the installed package (`pyproject.toml` ships only `src/stt_server`) and
    so is not importable from an installed/`uv run` context.
    """
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name = line.split("{", 1)[0]
            _, brace, after = line.partition("}")
            if not brace:
                continue
            value_tokens = after.split()
        else:
            tokens = line.split()
            name, value_tokens = tokens[0], tokens[1:]
        if not value_tokens:
            continue
        try:
            totals[name] = totals.get(name, 0.0) + float(value_tokens[0])
        except ValueError:
            continue
    return totals


def fetch_metrics(base_url: str, headers: dict[str, str]) -> dict[str, float]:
    try:
        _, body = http_get(f"{base_url}/metrics", headers)
    except Exception as exc:  # noqa: BLE001 - /metrics is best-effort here
        print(f"!! could not scrape {base_url}/metrics: {exc}", file=sys.stderr)
        return {}
    return parse_prometheus_metrics(body.decode("utf-8", errors="replace"))


# --------------------------------------------------------------------------- #
# GPU sampling
# --------------------------------------------------------------------------- #
def query_gpu(gpu_index: int) -> tuple[int, int] | None:
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
    except Exception:  # noqa: BLE001 - missing/failing nvidia-smi is non-fatal
        return None


class GpuSampler:
    """Poll nvidia-smi in a background thread and keep the peak seen.

    A no-op (`available=False`) when nvidia-smi is absent, so this script
    still runs — with GPU numbers reported as unavailable — on a box without
    the NVIDIA tooling on PATH.
    """

    def __init__(self, gpu_index: int = 0, interval: float = 0.1) -> None:
        self.gpu_index = gpu_index
        self.interval = interval
        self.available = shutil.which("nvidia-smi") is not None
        self.baseline_mem_mib: int | None = None
        self.peak_mem_mib = 0
        self.peak_util_pct = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            sample = query_gpu(self.gpu_index)
            if sample:
                mem, util = sample
                self.peak_mem_mib = max(self.peak_mem_mib, mem)
                self.peak_util_pct = max(self.peak_util_pct, util)
                self.samples += 1

    def __enter__(self) -> GpuSampler:
        if self.available:
            idle = query_gpu(self.gpu_index)
            self.baseline_mem_mib = idle[0] if idle else None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self) -> dict:
        return {
            "available": self.available,
            "baseline_mem_mib": self.baseline_mem_mib,
            "peak_mem_mib": self.peak_mem_mib if self.available else None,
            "peak_util_pct": self.peak_util_pct if self.available else None,
            "samples": self.samples,
        }


# --------------------------------------------------------------------------- #
# Wire clients
# --------------------------------------------------------------------------- #
@dataclass
class WsResult:
    text: str = ""
    finals: list[str] = field(default_factory=list)
    partial_count: int = 0
    last_partial: str = ""
    server_first_partial_ms: float | None = None
    server_final_ms: float | None = None
    client_final_ms: float | None = None
    wall_ms: float = 0.0
    errors: list[dict] = field(default_factory=list)
    audio_seconds: float = 0.0


async def ws_transcribe(
    base_url: str,
    model: str,
    pcm16: bytes,
    *,
    token: str | None = None,
    pace: float = 1.0,
    chunk_ms: int = 100,
    trailing_silence_ms: int = 0,
    # Not named `timeout`: ruff's ASYNC109 reserves that name on async
    # defs for the `asyncio.timeout` idiom, and this one is an internal
    # cap on waiting for `session.closed`, not a cancel scope the caller
    # composes with.
    event_timeout: float = 180.0,
) -> WsResult:
    """Stream `pcm16` to `/ws/transcribe?model=...` and collect the events.

    Wire format per `src/stt_server/api/native_ws.py::encode_native`: binary
    frames carry raw PCM16, a `{"type": "input_done"}` text frame ends the
    input, and the server emits partial/final/error events plus a terminal
    `session.closed`.

    Pacing uses the same cumulative-deadline scheme as
    `benchmarks/client_ws.py`, so per-chunk jitter never accumulates into
    drift. `pace=1.0` (the default here, and the binding methodology rule
    for real backends) is real time.

    `trailing_silence_ms` appends silence BEFORE `input_done`, letting the
    server-side VAD/endpointer close the utterance on its own. Passing 0 —
    the final-flush case — sends `input_done` immediately after the last
    audio frame, exercising the backend's `finalize()` flush path instead.
    """
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/ws/transcribe?model={model}"
    connect_kwargs: dict = {}
    if token:
        connect_kwargs["additional_headers"] = {"Authorization": f"Bearer {token}"}

    result = WsResult(audio_seconds=pcm_duration_s(pcm16))
    latencies: list[dict] = []
    final_recv_ts: list[float] = []

    speech_bytes = len(pcm16)
    if trailing_silence_ms > 0:
        pcm16 = pcm16 + b"\x00" * (SAMPLE_RATE * BYTES_PER_SAMPLE * trailing_silence_ms // 1000)

    t_start = time.monotonic()
    async with websockets.connect(url, **connect_kwargs) as ws:

        async def receiver() -> None:
            async for raw in ws:
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "partial":
                    result.partial_count += 1
                    result.last_partial = (
                        f"{event.get('stable_text', '')}{event.get('volatile_text', '')}"
                    )
                elif kind == "final":
                    result.finals.append(event.get("text", ""))
                    latencies.append(event.get("latency") or {})
                    final_recv_ts.append(time.monotonic())
                elif kind == "error":
                    result.errors.append(event)
                    if not event.get("recoverable", True):
                        return
                elif kind == "session.closed":
                    return

        recv_task = asyncio.create_task(receiver())
        chunk_bytes = max(1, SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_ms // 1000)
        t0 = time.monotonic()
        # Timestamp of the last frame containing real SPEECH, not the last
        # frame sent: the endpointer fires part-way into the appended
        # trailing silence, so a FINAL routinely arrives before the padding
        # has finished streaming. Measuring client latency from the end of
        # the padding would report a meaningless negative number.
        last_speech_send_ts = t0
        for index, offset in enumerate(range(0, len(pcm16), chunk_bytes)):
            if pace > 0:
                deadline = t0 + index * (chunk_ms / 1000.0) / pace
                delay = deadline - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(pcm16[offset : offset + chunk_bytes])
            if offset < speech_bytes:
                last_speech_send_ts = time.monotonic()

        await ws.send(json.dumps({"type": "input_done"}))
        try:
            await asyncio.wait_for(recv_task, timeout=event_timeout)
        except TimeoutError:
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task
            result.errors.append(
                {"code": "client_timeout", "message": f"no session.closed within {event_timeout}s"}
            )

    result.wall_ms = (time.monotonic() - t_start) * 1000.0
    result.text = " ".join(t for t in result.finals if t).strip()
    if latencies:
        result.server_first_partial_ms = latencies[0].get("first_partial_ms")
        result.server_final_ms = latencies[-1].get("final_ms")
    if final_recv_ts:
        result.client_final_ms = (final_recv_ts[-1] - last_speech_send_ts) * 1000.0
    return result


def http_transcribe(
    base_url: str,
    model: str,
    pcm16: bytes,
    *,
    language: str | None = None,
    token: str | None = None,
    timeout: float = 180.0,
) -> dict:
    """POST the clip to `/v1/audio/transcriptions` (the only surface that
    carries a per-request `language`), returning status/text/latency."""
    fields = {"model": model, "response_format": "json"}
    if language:
        fields["language"] = language
    body, boundary = build_multipart(fields, "clip.wav", pcm_to_wav(pcm16))
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    t0 = time.monotonic()
    status, raw = http_post(f"{base_url}/v1/audio/transcriptions", body, headers, timeout)
    wall_ms = (time.monotonic() - t0) * 1000.0

    text = ""
    if status == 200:
        try:
            text = (json.loads(raw).get("text") or "").strip()
        except Exception:  # noqa: BLE001 - fall back to the raw body below
            text = raw.decode("utf-8", errors="replace").strip()
    return {
        "status": status,
        "text": text,
        "wall_ms": wall_ms,
        "body": None if status == 200 else raw.decode("utf-8", errors="replace")[:500],
    }


# --------------------------------------------------------------------------- #
# Case bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    label: str
    status: str = "PASS"  # PASS | FAIL | SKIP
    detail: str = ""
    data: dict = field(default_factory=dict)

    def fail(self, reason: str) -> Case:
        self.status = "FAIL"
        self.detail = reason
        return self

    def skip(self, reason: str) -> Case:
        self.status = "SKIP"
        self.detail = reason
        return self


class Report:
    def __init__(self) -> None:
        self.cases: list[Case] = []

    def add(self, case: Case) -> Case:
        self.cases.append(case)
        marker = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[case.status]
        print(f"[{marker}] {case.label}" + (f" -- {case.detail}" if case.detail else ""))
        return case

    @property
    def failed(self) -> list[Case]:
        return [c for c in self.cases if c.status == "FAIL"]


def _score(label: str, reference: str | None, hypothesis: str, unit: str) -> dict:
    """CER/WER against an optional reference; printed, never asserted on (no
    accuracy threshold is defensible before the model has ever been run)."""
    if not reference:
        return {}
    rate = cer(reference, hypothesis) if unit == "cer" else wer(reference, hypothesis)
    if rate is None:
        return {}
    print(f"       {label}: {unit.upper()} {rate * 100:.2f}%  ref={reference!r}")
    return {unit: rate, "reference": reference}


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
async def run_explicit_language_cases(
    report: Report, args, clips: dict[str, bytes], refs: dict[str, str | None]
) -> None:
    """(a)/(b): explicit language via the file HTTP endpoint.

    The native WS protocol carries no language field (docs/backends.md §5),
    so `POST /v1/audio/transcriptions` with a `language` form field is the
    only way to pin the backend's language prompt per request.
    """
    for lang_key, lang_value, unit in (("zh", "zh", "cer"), ("en", "en", "wer")):
        case = Case(f"{lang_key}-explicit-http (language={lang_value})")
        pcm = clips.get(lang_key)
        if pcm is None:
            report.add(case.skip(f"no --{lang_key} clip supplied"))
            continue
        out = await asyncio.to_thread(
            http_transcribe,
            args.base_url,
            args.model,
            pcm,
            language=lang_value,
            token=args.token,
            timeout=args.timeout,
        )
        case.data = out
        if out["status"] != 200:
            report.add(case.fail(f"HTTP {out['status']}: {out['body']}"))
            continue
        if not out["text"]:
            report.add(case.fail("empty transcript"))
            continue
        print(f"       transcript: {out['text']!r}  ({out['wall_ms']:.0f} ms)")
        case.data.update(_score(case.label, refs.get(lang_key), out["text"], unit))
        report.add(case)


async def run_auto_language_cases(
    report: Report, args, clips: dict[str, bytes], refs: dict[str, str | None]
) -> None:
    """(c): automatic language identification on both clips, over both
    surfaces, with NO language supplied anywhere."""
    for lang_key, unit in (("zh", "cer"), ("en", "wer")):
        pcm = clips.get(lang_key)

        ws_case = Case(f"{lang_key}-auto-ws (no language)")
        if pcm is None:
            report.add(ws_case.skip(f"no --{lang_key} clip supplied"))
        else:
            res = await ws_transcribe(
                args.base_url,
                args.model,
                pcm,
                token=args.token,
                pace=args.pace,
                chunk_ms=args.chunk_ms,
                trailing_silence_ms=args.trailing_silence_ms,
                event_timeout=args.timeout,
            )
            ws_case.data = {
                "text": res.text,
                "partials": res.partial_count,
                "server_first_partial_ms": res.server_first_partial_ms,
                "server_final_ms": res.server_final_ms,
                "client_final_ms": res.client_final_ms,
                "errors": res.errors,
            }
            if res.errors:
                report.add(ws_case.fail(f"server errors: {res.errors}"))
            elif not res.text:
                report.add(ws_case.fail("empty transcript"))
            else:
                print(
                    f"       transcript: {res.text!r}  "
                    f"(partials={res.partial_count}, "
                    f"first_partial={_ms(res.server_first_partial_ms)}, "
                    f"final={_ms(res.server_final_ms)})"
                )
                ws_case.data.update(_score(ws_case.label, refs.get(lang_key), res.text, unit))
                report.add(ws_case)

        http_case = Case(f"{lang_key}-auto-http (no language)")
        if pcm is None:
            report.add(http_case.skip(f"no --{lang_key} clip supplied"))
            continue
        out = await asyncio.to_thread(
            http_transcribe,
            args.base_url,
            args.model,
            pcm,
            language=None,
            token=args.token,
            timeout=args.timeout,
        )
        http_case.data = out
        if out["status"] != 200:
            report.add(http_case.fail(f"HTTP {out['status']}: {out['body']}"))
        elif not out["text"]:
            report.add(http_case.fail("empty transcript"))
        else:
            print(f"       transcript: {out['text']!r}  ({out['wall_ms']:.0f} ms)")
            http_case.data.update(_score(http_case.label, refs.get(lang_key), out["text"], unit))
            report.add(http_case)


async def run_final_flush_cases(report: Report, args, clips: dict[str, bytes]) -> None:
    """(d): `input_done` immediately after the last audio frame, with NO
    trailing silence.

    This is the path where the server's VAD never sees the silence that
    would normally close the utterance, so the FINAL can only come from the
    backend's own `finalize()` flush (buffer remainder zero-padded to a full
    chunk, `keep_all_outputs=True`). The utterance TAIL is what gets dropped
    when that flush is wrong, so it is printed for manual review — asserting
    on exact tail text would need a reference transcript this script cannot
    assume.
    """
    for lang_key in ("zh", "en"):
        case = Case(f"{lang_key}-final-flush (input_done, no trailing silence)")
        pcm = clips.get(lang_key)
        if pcm is None:
            report.add(case.skip(f"no --{lang_key} clip supplied"))
            continue
        res = await ws_transcribe(
            args.base_url,
            args.model,
            pcm,
            token=args.token,
            pace=args.pace,
            chunk_ms=args.chunk_ms,
            trailing_silence_ms=0,
            event_timeout=args.timeout,
        )
        case.data = {
            "text": res.text,
            "finals": res.finals,
            "partials": res.partial_count,
            "last_partial": res.last_partial,
            "server_final_ms": res.server_final_ms,
            "client_final_ms": res.client_final_ms,
            "errors": res.errors,
        }
        if res.errors:
            report.add(case.fail(f"server errors: {res.errors}"))
            continue
        if not res.finals:
            report.add(case.fail("no final event arrived after input_done"))
            continue
        if not res.text:
            report.add(case.fail("final arrived but its text is empty"))
            continue
        print(f"       final: {res.text!r}")
        print(f"       TAIL CHECK (manual): last ~20 chars = {res.text[-20:]!r}")
        report.add(case)


async def run_concurrency_case(report: Report, args, clips: dict[str, bytes]) -> None:
    """(e): N parallel WS sessions at real-time pace.

    Every session gets its own connection and, per the design, its own
    encoder cache + decoder state; the clips are alternated across the
    available languages so the run also exercises concurrent per-stream
    language prompts (the model-global `set_inference_prompt` is the reason
    the backend serializes steps behind a process-wide lock).
    """
    case = Case(f"concurrency-{args.concurrency}-ws")
    available = [pcm for pcm in (clips.get("zh"), clips.get("en")) if pcm is not None]
    if not available:
        report.add(case.skip("no clips supplied"))
        return

    async def one(index: int) -> WsResult:
        return await ws_transcribe(
            args.base_url,
            args.model,
            available[index % len(available)],
            token=args.token,
            pace=args.pace,
            chunk_ms=args.chunk_ms,
            trailing_silence_ms=args.trailing_silence_ms,
            event_timeout=args.timeout,
        )

    t0 = time.monotonic()
    results = await asyncio.gather(
        *(one(i) for i in range(args.concurrency)), return_exceptions=True
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    sessions = []
    problems: list[str] = []
    for index, res in enumerate(results):
        if isinstance(res, BaseException):
            problems.append(f"session {index}: {type(res).__name__}: {res}")
            sessions.append({"session": index, "exception": repr(res)})
            continue
        if res.errors:
            problems.append(f"session {index}: server errors {res.errors}")
        elif not res.text:
            problems.append(f"session {index}: empty transcript")
        sessions.append(
            {
                "session": index,
                "text": res.text,
                "partials": res.partial_count,
                "server_first_partial_ms": res.server_first_partial_ms,
                "server_final_ms": res.server_final_ms,
                "client_final_ms": res.client_final_ms,
                "wall_ms": res.wall_ms,
                "audio_seconds": res.audio_seconds,
                "errors": res.errors,
            }
        )

    case.data = {"concurrency": args.concurrency, "elapsed_ms": elapsed_ms, "sessions": sessions}
    print(f"       {args.concurrency} sessions in {elapsed_ms:.0f} ms wall")
    print(
        f"       {'sess':>4}  {'first_partial_ms':>16}  {'final_ms':>9}  "
        f"{'client_final_ms':>15}"
    )
    for entry in sessions:
        if "exception" in entry:
            print(f"       {entry['session']:>4}  {entry['exception']}")
            continue
        print(
            f"       {entry['session']:>4}  {_ms(entry['server_first_partial_ms']):>16}  "
            f"{_ms(entry['server_final_ms']):>9}  {_ms(entry['client_final_ms']):>15}"
        )
    if problems:
        case.fail("; ".join(problems))
    report.add(case)


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zh", type=Path, default=None, help="Chinese clip (.wav 16k mono / .pcm)")
    parser.add_argument("--en", type=Path, default=None, help="English clip (.wav 16k mono / .pcm)")
    parser.add_argument("--zh-ref", default=None, help="reference transcript for the zh clip (CER)")
    parser.add_argument("--en-ref", default=None, help="reference transcript for the en clip (WER)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument(
        "--model",
        default="nemotron-3.5-asr-streaming-0.6b",
        help="model id (a key of the config's `models` map)",
    )
    parser.add_argument("--token", default=None, help="bearer token if auth.tokens is set")
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="streaming pace; 1.0 = real time. Do NOT raise this against a real "
        "backend: outrunning the decode trips drop_oldest backpressure and the "
        "run's own drops check will fail it",
    )
    parser.add_argument("--chunk-ms", type=int, default=100, help="WS audio frame size")
    parser.add_argument(
        "--trailing-silence-ms",
        type=int,
        default=800,
        help="silence appended before input_done for the non-flush WS cases, so "
        "the server-side endpointer closes the utterance naturally",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="parallel WS sessions in case e")
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index")
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout seconds")
    parser.add_argument("--json", type=Path, default=None, help="write all results to this file")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    clips: dict[str, bytes] = {}
    for key, path in (("zh", args.zh), ("en", args.en)):
        if path is not None:
            clips[key] = load_pcm16(path)
            print(f"input {key}: {path} ({pcm_duration_s(clips[key]):.2f}s @ 16 kHz mono)")
    if not clips:
        print("!! supply at least one of --zh / --en", file=sys.stderr)
        return 2
    refs = {"zh": args.zh_ref, "en": args.en_ref}

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    try:
        status, body = http_get(f"{args.base_url}/readyz", headers)
        print(f"GET /readyz -> {status} {body.decode(errors='replace').strip()}")
    except Exception as exc:  # noqa: BLE001
        print(f"!! GET /readyz failed: {exc} (is the server up on {args.base_url}?)",
              file=sys.stderr)
        return 2

    report = Report()
    metrics_before = fetch_metrics(args.base_url, headers)

    with GpuSampler(gpu_index=args.gpu) as gpu:
        if not gpu.available:
            print("!! nvidia-smi not found; GPU memory/utilization will be unavailable",
                  file=sys.stderr)
        await run_explicit_language_cases(report, args, clips, refs)
        await run_auto_language_cases(report, args, clips, refs)
        await run_final_flush_cases(report, args, clips)
        await run_concurrency_case(report, args, clips)

    metrics_after = fetch_metrics(args.base_url, headers)
    dropped_delta = metrics_after.get(DROPPED_METRIC, 0.0) - metrics_before.get(
        DROPPED_METRIC, 0.0
    )

    drops_case = Case(f"no-audio-drops ({DROPPED_METRIC} delta)")
    drops_case.data = {"dropped_delta": dropped_delta}
    if dropped_delta:
        report.add(
            drops_case.fail(
                f"server shed audio: {DROPPED_METRIC} increased by {dropped_delta:g} during "
                "the run, so every latency/accuracy number above describes behavior under "
                "active audio drop. Remedies: keep --pace 1.0, raise "
                "limits.audio_queue_chunks, or lower limits.max_sessions/--concurrency"
            )
        )
    else:
        report.add(drops_case)

    gpu_summary = gpu.summary()
    if gpu_summary["available"]:
        print(
            f"\nGPU peak during run: {gpu_summary['peak_mem_mib']} MiB used "
            f"(idle baseline {gpu_summary['baseline_mem_mib']} MiB), "
            f"{gpu_summary['peak_util_pct']}% util over {gpu_summary['samples']} samples"
        )
    else:
        print("\nGPU peak during run: unavailable (nvidia-smi missing)")

    print("\n== summary ==")
    width = max(len(c.label) for c in report.cases)
    for case in report.cases:
        print(f"  {case.label:<{width}}  {case.status:<4}  {case.detail}")
    print(
        f"  {'-' * width}  {len(report.cases)} cases, "
        f"{len(report.failed)} failed, "
        f"{sum(1 for c in report.cases if c.status == 'SKIP')} skipped"
    )

    if args.json:
        payload = {
            "base_url": args.base_url,
            "model": args.model,
            "pace": args.pace,
            "concurrency": args.concurrency,
            "gpu": gpu_summary,
            "audio_dropped_delta": dropped_delta,
            "cases": [
                {"label": c.label, "status": c.status, "detail": c.detail, "data": c.data}
                for c in report.cases
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
