"""Auth and capacity guards shared by all protocol adapters."""

from __future__ import annotations

import secrets
import time

import structlog
from fastapi import FastAPI

from stt_server.config.settings import Settings
from stt_server.metrics.registry import REJECTIONS, SESSIONS_ACTIVE

logger = structlog.get_logger(__name__)


def _token_matches(presented: str, configured: list[str]) -> bool:
    # No early exit on match: every configured token is compared so the
    # function's running time doesn't leak which position (if any) matched.
    ok = False
    for t in configured:
        if secrets.compare_digest(presented.encode(), t.encode()):
            ok = True
    return ok


def _reject(reason: str) -> bool:
    # Logged reason can be more specific than the "unauthorized" the client
    # sees on the wire (bad_scheme, bad_token, ...); the metric label stays
    # coarse ("unauthorized") to match REJECTIONS' documented reason set.
    logger.info("auth.rejected", reason=reason)
    REJECTIONS.labels(reason="unauthorized").inc()
    return False


def check_token(app: FastAPI, authorization: str | None) -> bool:
    tokens = app.state.settings.auth.tokens
    if not tokens:
        return True
    if authorization is None:
        return _reject("missing_header")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return _reject("bad_scheme")
    if not _token_matches(credentials, tokens):
        # Never log `credentials` or any configured token here.
        return _reject("bad_token")
    return True


class SessionSlots:
    """Bounded counter for concurrent sessions. Single event loop: no locking."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.active >= self._limit:
            logger.info("capacity.rejected", active=self.active, limit=self._limit)
            REJECTIONS.labels(reason="capacity").inc()
            return False
        self.active += 1
        SESSIONS_ACTIVE.set(self.active)
        return True

    def release(self) -> None:
        if self.active <= 0:
            # A release without a matching acquire indicates a bug in a
            # caller's guard ordering (e.g. double-release). Still clamp to
            # zero so the counter never goes negative and wedges capacity.
            logger.warning("slots.release_underflow", active=self.active)
        self.active = max(0, self.active - 1)
        SESSIONS_ACTIVE.set(self.active)


def session_deadline(settings: Settings) -> float:
    """Absolute `time.monotonic()` deadline for one session's total duration.

    There is no background timer enforcing this: both WS receive loops check
    `time.monotonic() > deadline` only when a new message arrives, so a
    session that has gone quiet (no more client messages) only times out on
    its *next* message, not exactly at the deadline instant.
    """
    return time.monotonic() + settings.limits.max_session_seconds
