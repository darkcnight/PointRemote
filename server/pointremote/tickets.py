from __future__ import annotations

import secrets
import time
from collections.abc import Callable


class LoginTickets:
    """Single-use, short-lived tickets so the desktop app can open the web UI logged in
    without putting the API key in a URL."""

    def __init__(self, ttl_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._ttl, self._clock = ttl_s, clock
        self._issued: dict[str, float] = {}

    def issue(self) -> str:
        now = self._clock()
        self._issued = {t: exp for t, exp in self._issued.items() if exp > now}
        ticket = secrets.token_urlsafe(24)
        self._issued[ticket] = now + self._ttl
        return ticket

    def redeem(self, ticket: str) -> bool:
        expires = self._issued.pop(ticket, None)
        return expires is not None and expires > self._clock()
