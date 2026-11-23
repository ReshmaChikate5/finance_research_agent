"""TTLCache and DailyBudget (market-data server helpers)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import Any

import pytest

from tests.unit.helpers import SERVERS_DIR, run


@pytest.fixture
def cache_mod() -> Any:
    # cache.py is imported by the market-data server as a sibling module; load it by path.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "unit_cache", SERVERS_DIR / "market_data" / "cache.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch, cache_mod: Any) -> Clock:
    c = Clock()
    monkeypatch.setattr(cache_mod.time, "monotonic", c)
    return c


def test_get_returns_missing_sentinel_for_unknown_key(cache_mod):
    assert cache_mod.TTLCache().get("nope") is cache_mod._MISSING


def test_value_is_served_until_its_ttl_then_expires(cache_mod, clock):
    cache = cache_mod.TTLCache()
    cache.set("k", "v", ttl=60)
    assert cache.get("k") == "v"
    clock.now += 59
    assert cache.get("k") == "v"
    clock.now += 2
    assert cache.get("k") is cache_mod._MISSING


def test_each_entry_has_its_own_ttl(cache_mod, clock):
    cache = cache_mod.TTLCache()
    cache.set("short", 1, ttl=10)
    cache.set("long", 2, ttl=100)
    clock.now += 50
    assert cache.get("short") is cache_mod._MISSING
    assert cache.get("long") == 2


def test_falsy_values_are_cached_too(cache_mod):
    cache = cache_mod.TTLCache()
    cache.set("zero", 0, ttl=60)
    assert cache.get("zero") == 0


def test_eviction_keeps_the_cache_at_max_items(cache_mod, clock):
    cache = cache_mod.TTLCache(max_items=3)
    for i in range(5):
        cache.set(f"k{i}", i, ttl=100 + i)  # k0 expires first
    assert len(cache._data) == 3
    assert cache.get("k0") is cache_mod._MISSING and cache.get("k4") == 4


def test_eviction_drops_expired_entries_before_live_ones(cache_mod, clock):
    cache = cache_mod.TTLCache(max_items=2)
    cache.set("old", 1, ttl=1)
    cache.set("live", 2, ttl=100)
    clock.now += 5
    cache.set("new", 3, ttl=100)  # over the limit: "old" is expired and goes first
    assert cache.get("live") == 2 and cache.get("new") == 3


def test_get_or_fetch_calls_upstream_once_then_hits(cache_mod):
    cache = cache_mod.TTLCache()
    calls = []

    async def fetch():
        calls.append(1)
        return {"price": 10}

    async def scenario():
        first = await cache.get_or_fetch("q", 60, fetch)
        second = await cache.get_or_fetch("q", 60, fetch)
        return first, second

    first, second = run(scenario())
    assert first == ({"price": 10}, False) and second == ({"price": 10}, True)
    assert len(calls) == 1 and (cache.hits, cache.misses) == (1, 1)


def test_concurrent_requests_for_one_key_share_one_upstream_call(cache_mod):
    cache = cache_mod.TTLCache()
    calls = []

    async def fetch():
        calls.append(1)
        await asyncio.sleep(0.01)
        return "value"

    async def scenario():
        return await asyncio.gather(*(cache.get_or_fetch("k", 60, fetch) for _ in range(5)))

    results = run(scenario())
    assert len(calls) == 1
    assert [hit for _, hit in results].count(False) == 1  # one miss, four hits


def test_upstream_errors_are_not_cached(cache_mod):
    cache = cache_mod.TTLCache()
    attempts = []

    async def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return "ok"

    async def scenario():
        with pytest.raises(RuntimeError):
            await cache.get_or_fetch("k", 60, flaky)
        return await cache.get_or_fetch("k", 60, flaky)

    assert run(scenario()) == ("ok", False)
    assert len(attempts) == 2


def test_locks_are_cleaned_up(cache_mod):
    cache = cache_mod.TTLCache()

    async def fetch():
        return 1

    run(cache.get_or_fetch("k", 60, fetch))
    assert cache._locks == {}


# ---------------------------------------------------------------------------
# DailyBudget
# ---------------------------------------------------------------------------
class FakeDatetime:
    """Stands in for the `datetime` class inside cache.py; the test chooses today's date."""

    today = date(2026, 1, 1)

    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return datetime(cls.today.year, cls.today.month, cls.today.day, 12, tzinfo=UTC)


@pytest.fixture
def fake_day(monkeypatch: pytest.MonkeyPatch, cache_mod: Any) -> type[FakeDatetime]:
    FakeDatetime.today = date(2026, 1, 1)
    monkeypatch.setattr(cache_mod, "datetime", FakeDatetime)
    return FakeDatetime


def test_budget_allows_exactly_the_limit(cache_mod, fake_day):
    budget = cache_mod.DailyBudget(3)
    assert [budget.try_consume() for _ in range(5)] == [True, True, True, False, False]
    assert budget.used == 3


def test_budget_resets_on_a_new_utc_day(cache_mod, fake_day):
    budget = cache_mod.DailyBudget(1)
    assert budget.try_consume() is True and budget.try_consume() is False
    fake_day.today = date(2026, 1, 2)
    assert budget.used == 0
    assert budget.try_consume() is True


def test_zero_budget_never_allows_a_call(cache_mod, fake_day):
    assert cache_mod.DailyBudget(0).try_consume() is False


def test_market_server_wires_the_budget_from_env(make_market):
    mod = make_market(AV_DAILY_LIMIT="7")
    assert mod.av_budget.limit == 7
