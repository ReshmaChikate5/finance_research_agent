"""Plain helpers for the MCP server unit tests (Step 17). Fixtures live in conftest.py.

Rules these tests follow:
  * No real network. Every HTTP call goes to a fake transport; any attempt to reach the
    internet fails the test.
  * No real secrets. Each server reads its settings from env vars when it is imported, so the
    fixtures set fake values first and switch off python-dotenv (a developer's real .env must
    never leak into a test).
  * Each test loads a FRESH copy of the server, so caches, dedupe tables and quota counters
    never carry over from one test to the next.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import httpx

SERVERS_DIR = Path(__file__).resolve().parents[2] / "mcp_servers"

# Names the servers import as siblings ("import db", "from cache import ..."). They are loaded
# fresh with each server and removed afterwards so servers cannot see each other's modules.
_SIBLINGS = ("db", "cache")


def run(coro: Any) -> Any:
    """Run a coroutine to completion (the tests need no pytest plugin for async)."""
    return asyncio.run(coro)


def load_server(folder: str) -> Any:
    """Import mcp_servers/<folder>/server.py as a fresh module."""
    path = SERVERS_DIR / folder
    for name in _SIBLINGS:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(f"unit_{folder}", path / "server.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path))
        for name in _SIBLINGS:
            sys.modules.pop(name, None)
    return module


class Router:
    """A fake internet. Register answers by URL fragment; read `calls` afterwards.

    router.add("company_tickers", json={...})              always answers this
    router.add("api/quote", status=429, times=1)           answers once, then falls through
    router.add("hooks.slack.com", exc=httpx.ConnectError("x"))   raises instead of answering
    """

    def __init__(self) -> None:
        self._routes: list[dict[str, Any]] = []
        self.calls: list[httpx.Request] = []
        self.unmatched: list[str] = []

    def add(
        self,
        needle: str,
        *,
        json: Any = None,
        text: str | None = None,
        status: int = 200,
        exc: Exception | None = None,
        times: int | None = None,
    ) -> Router:
        self._routes.append(
            {
                "needle": needle,
                "json": json,
                "text": text,
                "status": status,
                "exc": exc,
                "times": times,
                "used": 0,
            }
        )
        return self

    def urls(self, needle: str = "") -> list[str]:
        return [str(r.url) for r in self.calls if needle in str(r.url)]

    def count(self, needle: str) -> int:
        return len(self.urls(needle))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
        for route in self._routes:
            if route["needle"] not in url:
                continue
            if route["times"] is not None and route["used"] >= route["times"]:
                continue
            route["used"] += 1
            if route["exc"] is not None:
                raise route["exc"]
            if route["text"] is not None:
                return httpx.Response(route["status"], text=route["text"])
            if route["json"] is not None:
                return httpx.Response(route["status"], json=route["json"])
            return httpx.Response(route["status"])
        self.unmatched.append(url)
        return httpx.Response(500, text="unmatched request in test")


def attach(module: Any, router: Router, **client_kwargs: Any) -> None:
    """Point a server's shared HTTP client at the fake internet."""
    module._http = httpx.AsyncClient(transport=httpx.MockTransport(router), **client_kwargs)
