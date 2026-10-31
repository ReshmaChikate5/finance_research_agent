"""Everything the nodes need from the outside world, passed in explicitly.

Nodes never import an SDK, read env vars or open sockets themselves. Tests build a Deps with
fakes; the real app builds one from settings (see build_real_deps).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.llm import ChatModel
from app.mcp_clients import ToolClient
from app.vectorstore.base import VectorStore

Approver = Callable[[str], Awaitable[bool]]


@dataclass
class Deps:
    llm: ChatModel
    clients: Mapping[str, ToolClient]  # keys: secedgar, marketdata, portfolio, notify
    store: VectorStore
    approver: Approver
    tool_menu: str  # text listing the tools the planner may use
    max_iterations: int = 8
    slack_channel: str = "#finance-reports"
    llm_retry_delay: float = 2.0  # seconds to wait before retrying a failed model call
    menu_warnings: list[str] = field(default_factory=list)  # e.g. servers that were down at start
    today: Callable[[], str] = field(default=lambda: dt.date.today().isoformat())


async def console_approver(report: str) -> bool:
    """Ask on the terminal. Step 16 replaces this with a LangGraph interrupt()."""
    import asyncio

    print("\n" + "=" * 70 + "\nREPORT FOR APPROVAL\n" + "=" * 70)
    print(report)
    print("=" * 70)
    answer = await asyncio.to_thread(input, "Post this report to Slack? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


async def build_real_deps(
    settings: Settings | None = None, approver: Approver | None = None
) -> Deps:
    from app.llm import get_chat_model
    from app.mcp_clients import build_clients, describe_tools
    from app.vectorstore import get_vector_store

    s = settings or get_settings()
    clients = build_clients(s)
    menu = await describe_tools(clients)
    down = [ln.split("]")[0].lstrip("[") for ln in menu.splitlines() if "UNAVAILABLE" in ln]
    return Deps(
        llm=get_chat_model(s),
        clients=clients,
        store=get_vector_store(s),
        approver=approver or console_approver,
        tool_menu=menu,
        menu_warnings=[
            f"The {n} server was unreachable at start, so its data is missing." for n in down
        ],
        max_iterations=s.max_iterations,
        slack_channel=s.slack_channel_name,
    )
