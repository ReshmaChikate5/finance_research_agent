"""Fixtures for the MCP server unit tests (Step 17). Helpers live in helpers.py.

Rules these tests follow:
  * No real network: every HTTP call goes to a fake transport, and any attempt to reach the
    internet fails the test.
  * No real secrets: each server reads settings from env vars on import, so fixtures set fake
    values first and switch off python-dotenv (a real .env must never leak into a test).
  * Each test loads a FRESH copy of the server, so caches, dedupe tables and quota counters
    never carry over between tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tests.unit.helpers import Router, attach, load_server


@pytest.fixture
def router() -> Any:
    r = Router()
    yield r
    assert not r.unmatched, f"Requests no test route answered: {r.unmatched}"


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any HTTP request that is not served by a fake transport fails the test."""

    async def refuse(self: Any, request: Any) -> Any:
        raise AssertionError(f"Real network call attempted in a unit test: {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Servers call load_dotenv() on import; make that a no-op so a real .env is never read."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make asyncio.sleep instant and record the delays that were requested."""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *a: Any, **k: Any) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


# ---------------------------------------------------------------------------
# One fixture per server
# ---------------------------------------------------------------------------
ENV_KEYS = (
    "SEC_USER_AGENT",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "AV_DAILY_LIMIT",
    "SLACK_WEBHOOK_URL",
    "SLACK_DRY_RUN",
    "SLACK_CHANNEL_NAME",
    "SLACK_DEDUPE_TTL_SECONDS",
    "PORTFOLIO_DATABASE_URL",
    "QUOTE_TTL_SECONDS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def edgar(monkeypatch: pytest.MonkeyPatch, router: Router) -> Any:
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    mod = load_server("sec_edgar")
    mod._limiter = mod._RateLimiter(per_second=100_000)
    attach(mod, router)
    return mod


@pytest.fixture
def make_market(monkeypatch: pytest.MonkeyPatch, router: Router) -> Callable[..., Any]:
    """Build a market-data server. Pass finnhub=False / alpha=False to leave a key unset."""

    def build(finnhub: bool = True, alpha: bool = True, **env: str) -> Any:
        if finnhub:
            monkeypatch.setenv("FINNHUB_API_KEY", "fh-test-key")
        if alpha:
            monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "av-test-key")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        mod = load_server("market_data")
        mod._finnhub_limiter = mod._RateLimiter(per_second=100_000)
        attach(mod, router)
        return mod

    return build


@pytest.fixture
def market(make_market: Callable[..., Any]) -> Any:
    return make_market()


@pytest.fixture
def make_notify(monkeypatch: pytest.MonkeyPatch, router: Router) -> Callable[..., Any]:
    def build(dry_run: bool = False, **env: str) -> Any:
        monkeypatch.setenv("SLACK_CHANNEL_NAME", "#finance-reports")
        if dry_run:
            monkeypatch.setenv("SLACK_DRY_RUN", "true")
        else:
            monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T0/B0/secret")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        mod = load_server("notify_slack")
        attach(mod, router)
        return mod

    return build


@pytest.fixture
def notify(make_notify: Callable[..., Any]) -> Any:
    return make_notify()


@pytest.fixture
def portfolio() -> Any:
    return load_server("portfolio")
