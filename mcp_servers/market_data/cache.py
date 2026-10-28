"""In-memory TTL cache and daily quota guard for the market-data MCP server.

TTLCache.get_or_fetch() is the single entry point: it checks the cache first,
and only calls the upstream API on a miss. Concurrent requests for the same key
share one upstream call. Errors are never cached.

Limits: the cache lives in process memory, so it is per-container and resets on
restart. For multiple replicas on AWS, swap this for Redis/ElastiCache later.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from typing import Any

_MISSING = object()


class TTLCache:
    """Small async-safe cache where every entry expires after its own TTL."""

    def __init__(self, max_items: int = 500) -> None:
        self._max = max_items
        self._data: dict[str, tuple[float, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        """Return the cached value, or the _MISSING sentinel if absent/expired."""
        entry = self._data.get(key)
        if entry is None:
            return _MISSING
        expires_at, value = entry
        if expires_at <= time.monotonic():
            self._data.pop(key, None)
            return _MISSING
        return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        self._data[key] = (time.monotonic() + ttl, value)
        if len(self._data) > self._max:
            self._evict()

    def _evict(self) -> None:
        now = time.monotonic()
        for key in [k for k, (exp, _) in self._data.items() if exp <= now]:
            del self._data[key]
        while len(self._data) > self._max:  # still too big: drop soonest-to-expire
            del self._data[min(self._data, key=lambda k: self._data[k][0])]

    async def get_or_fetch(
        self, key: str, ttl: float, fetch: Callable[[], Awaitable[Any]]
    ) -> tuple[Any, bool]:
        """Return (value, from_cache). Calls `fetch()` only on a cache miss."""
        value = self.get(key)
        if value is not _MISSING:
            self.hits += 1
            return value, True

        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                value = self.get(key)  # another caller may have filled it while we waited
                if value is not _MISSING:
                    self.hits += 1
                    return value, True
                self.misses += 1
                value = await fetch()  # exceptions propagate and are NOT cached
                self.set(key, value, ttl)
                return value, False
        finally:
            self._locks.pop(key, None)


class DailyBudget:
    """Counts upstream calls per UTC day so a free-tier quota is never blown.

    This is a local, in-memory estimate: it resets on restart and the provider's
    own day boundary may differ slightly. The provider's rejection message is
    still handled separately as a safety net.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._day: date | None = None
        self._used = 0

    def _roll(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day = today
            self._used = 0

    def try_consume(self) -> bool:
        """Reserve one call. Returns False if the daily limit is already reached."""
        self._roll()
        if self._used >= self.limit:
            return False
        self._used += 1
        return True

    @property
    def used(self) -> int:
        self._roll()
        return self._used