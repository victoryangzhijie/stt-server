"""Partial-transcript stabilizer: maintains a committed prefix that only grows.

Reduces visible flicker by exposing {stable, volatile} text splits. Purely
functional over caller-supplied timestamps, so tests need no real clock."""

from __future__ import annotations

from dataclasses import dataclass

from stt_server.config.settings import StabilizerConfig


@dataclass(frozen=True)
class StabilizerUpdate:
    stable_text: str
    volatile_text: str
    newly_committed: str  # "" when the committed prefix did not grow


class Stabilizer:
    def __init__(self, cfg: StabilizerConfig) -> None:
        self._cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._committed: list[str] = []
        self._prev: list[str] = []
        self._seen_count: list[int] = []  # per token index: survived partials
        self._first_ms: list[float] = []  # per token index: first seen at this position

    def update(self, text: str, now_ms: float) -> StabilizerUpdate:
        tokens = text.split()

        # Survival tracking: tokens inside the LCP with the previous partial survive.
        lcp = 0
        limit = min(len(tokens), len(self._prev))
        while lcp < limit and tokens[lcp] == self._prev[lcp]:
            lcp += 1
        self._seen_count = [
            self._seen_count[i] + 1 if i < lcp else 1 for i in range(len(tokens))
        ]
        self._first_ms = [
            self._first_ms[i] if i < lcp else now_ms for i in range(len(tokens))
        ]
        self._prev = tokens

        # Grow the committed prefix (never shrink it).
        newly: list[str] = []
        i = len(self._committed)
        while (
            i < len(tokens)
            and self._seen_count[i] >= self._cfg.min_partials
            and now_ms - self._first_ms[i] >= self._cfg.min_stable_ms
        ):
            newly.append(tokens[i])
            i += 1
        self._committed.extend(newly)

        return StabilizerUpdate(
            stable_text=" ".join(self._committed),
            volatile_text=" ".join(tokens[len(self._committed) :]),
            newly_committed=" ".join(newly),
        )
