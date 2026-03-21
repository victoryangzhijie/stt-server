from __future__ import annotations

import copy
import threading

import numpy as np
import structlog

from backends.base import ASRResult
from config import settings

logger = structlog.get_logger()


# ── HypothesisBuffer (ported from whisper_streaming) ─────────────────────────


class _HypothesisBuffer:
    """Tracks text stability using word-level local agreement.

    Each word is a tuple (start_time, end_time, text).
    Ported from whisper_streaming/whisper_online.py HypothesisBuffer.
    """

    def __init__(self) -> None:
        self.commited_in_buffer: list[tuple[float, float, str]] = []
        self.buffer: list[tuple[float, float, str]] = []
        self.new: list[tuple[float, float, str]] = []
        self.last_commited_time: float = 0.0
        self.last_commited_word: str | None = None

    def insert(self, new: list[tuple[float, float, str]], offset: float) -> None:
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, _b, _t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for _j in range(i):
                                self.new.pop(0)
                            break

    def flush(self) -> list[tuple[float, float, str]]:
        commit: list[tuple[float, float, str]] = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float) -> None:
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[tuple[float, float, str]]:
        return self.buffer
