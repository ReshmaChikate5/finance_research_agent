"""Where the graph saves its state (Step 15).

A "checkpointer" stores the full graph state after every node finishes. With the Postgres
checkpointer, a run that is killed halfway can be resumed from its last finished node by
starting a new process and passing the same thread_id.

Usage (the connection stays open for the whole `async with` block):

    async with open_checkpointer(settings) as saver:
        graph = build_graph(deps, checkpointer=saver)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from psycopg import ProgrammingError

from app.config import Settings


class CheckpointError(RuntimeError):
    """The checkpoint database could not be reached or prepared. Message is safe to print."""


def describe_target(url: str) -> str:
    """host:port/dbname from a connection string, with the password left out."""
    from psycopg.conninfo import conninfo_to_dict

    try:
        parts = conninfo_to_dict(url)
    except ProgrammingError:
        return "(unreadable CHECKPOINT_DATABASE_URL)"
    return f"{parts.get('host', 'localhost')}:{parts.get('port', 5432)}/{parts.get('dbname', '?')}"


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[Any]:
    """Yield a checkpointer (or None) and close its connection afterwards."""
    if settings.checkpointer == "none":
        yield None
        return
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.conninfo import make_conninfo

    url = make_conninfo(settings.checkpoint_database_url, connect_timeout=5)
    async with AsyncExitStack() as stack:
        try:
            saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(url))
            await saver.setup()  # creates the checkpoint tables on first use; safe to repeat
        except Exception as exc:
            raise CheckpointError(
                f"Could not open the checkpoint database at {describe_target(url)}: "
                f"{type(exc).__name__}: {str(exc).strip()[:200]}\n"
                "Check that the Postgres container is up and that you ran sql/03_agent_state.sql "
                "(Step 15 walkthrough), or set CHECKPOINTER=memory to run without it."
            ) from exc
        yield saver
