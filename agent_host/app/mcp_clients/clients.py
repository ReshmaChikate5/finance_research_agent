"""Thin MCP clients, one per tool server.

Each call opens a short-lived streamable-HTTP session: simple and robust for a first version
(no connection state to lose when a container restarts). Every client has an ALLOWLIST of
tool names, so even if the model asks for a tool that exists on the server but is not meant
for the agent, the call is refused here, before any network traffic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Protocol

from app.config import Settings, get_settings

# server key -> tools the agent may call. Notify is separate: only the notify node calls it.
ALLOWED_TOOLS: dict[str, frozenset[str]] = {
    "secedgar": frozenset({"search_company_filings", "get_filing_document", "get_company_facts"}),
    "marketdata": frozenset({"get_quote", "get_fundamentals", "get_technical_indicator"}),
    "portfolio": frozenset({"get_holdings", "get_cost_basis"}),
    "notify": frozenset({"post_report"}),
}


def _root_cause(exc: BaseException) -> BaseException:
    """anyio wraps connection errors in an ExceptionGroup; show the real one."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


class ToolClientError(Exception):
    """A tool call could not be made (server down, refused tool). Message is safe to show."""


class ToolClient(Protocol):
    async def call(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]: ...

    async def list_tools(self) -> list[dict[str, Any]]: ...


class McpToolClient:
    def __init__(self, name: str, base_url: str, timeout: float = 60.0) -> None:
        self.name = name
        self._url = base_url.rstrip("/") + "/mcp"
        self._timeout = timeout

    async def _session_call(self, fn):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with (
            streamablehttp_client(self._url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await fn(session)

    async def call(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
        if tool not in ALLOWED_TOOLS[self.name]:
            raise ToolClientError(f"Tool '{tool}' is not allowed on server '{self.name}'.")
        try:
            result = await asyncio.wait_for(
                self._session_call(lambda s: s.call_tool(tool, dict(args))), self._timeout
            )
        except TimeoutError as exc:
            raise ToolClientError(
                f"{self.name}.{tool} timed out after {self._timeout:.0f}s."
            ) from exc
        except Exception as exc:
            raise ToolClientError(
                f"Could not reach the {self.name} server at {self._url} ({type(_root_cause(exc)).__name__}). "
                "Is `docker compose up` running?"
            ) from exc

        text = "".join(getattr(c, "text", "") for c in result.content)
        if getattr(result, "isError", False):
            return {"error": text or f"{tool} failed."}
        try:
            data = json.loads(text)
        except ValueError:
            return {"error": f"{tool} returned something that was not JSON."}
        return data if isinstance(data, dict) else {"result": data}

    async def list_tools(self) -> list[dict[str, Any]]:
        listing = await asyncio.wait_for(
            self._session_call(lambda s: s.list_tools()), self._timeout
        )
        return [
            {"name": t.name, "description": t.description or "", "schema": t.inputSchema}
            for t in listing.tools
            if t.name in ALLOWED_TOOLS[self.name]
        ]


def build_clients(settings: Settings | None = None) -> dict[str, ToolClient]:
    s = settings or get_settings()
    t = s.mcp_timeout_seconds
    return {
        "secedgar": McpToolClient("secedgar", s.sec_edgar_url, t),
        "marketdata": McpToolClient("marketdata", s.market_data_url, t),
        "portfolio": McpToolClient("portfolio", s.portfolio_url, t),
        "notify": McpToolClient("notify", s.notify_slack_url, t),
    }


async def describe_tools(clients: Mapping[str, ToolClient]) -> str:
    """Tool menu for the planner prompt, pulled from the live servers.

    If a server is unreachable its section says so, and the planner will see errors if it
    tries to use it. The notify server is never listed: the model cannot post to Slack.
    """
    lines: list[str] = []
    for server in ("secedgar", "marketdata", "portfolio"):
        try:
            tools = await clients[server].list_tools()
        except Exception as exc:  # noqa: BLE001
            lines.append(f"[{server}] UNAVAILABLE ({type(_root_cause(exc)).__name__})")
            continue
        for t in tools:
            props = (t["schema"] or {}).get("properties", {})
            required = set((t["schema"] or {}).get("required", []))
            params = ", ".join(f"{k}{'' if k in required else '?'}" for k in props)
            first = (t["description"] or "").strip().splitlines()[0] if t["description"] else ""
            lines.append(f"[{server}] {t['name']}({params}): {first}")
    return "\n".join(lines)
