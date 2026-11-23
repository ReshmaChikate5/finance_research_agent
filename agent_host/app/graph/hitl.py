"""Human-in-the-loop helpers (Step 16).

When a node calls interrupt(payload), the graph saves its state and stops. The caller sees the
payload, asks a human, and resumes by sending the answer back: Command(resume=answer).
The answer is always a bool here: True = go ahead, False = stop.

Payload "kind" values:
    approve_post     a finished report waits for permission to be posted to Slack
    review_conflict  the draft looks unreliable (sources disagree, thin evidence, cut off)
    confirm_write    the agent wants to call a tool that changes something
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.types import Command

Decide = Callable[[dict[str, Any]], Awaitable[bool | None]]  # None = leave it paused


def describe_interrupt(payload: dict[str, Any]) -> tuple[str, str]:
    """(text to show the human, the yes/no question to ask)."""
    kind = payload.get("kind")
    if kind == "approve_post":
        return payload["report"], f"Post this report to {payload.get('channel', 'Slack')}? [y/N] "
    if kind == "review_conflict":
        text = f"{payload['report']}\n\nFLAGGED: {payload['conflict']}"
        return text, "The draft was flagged. Continue to the approval step anyway? [y/N] "
    if kind == "confirm_write":
        text = (
            f"The agent wants to run {payload['server']}.{payload['tool']} with {payload['args']}"
        )
        return text, "Allow this action? [y/N] "
    return str(payload), "Continue? [y/N] "


async def drive(
    graph: Any, graph_input: Any, config: dict[str, Any], decide: Decide
) -> dict[str, Any] | None:
    """Run the graph; whenever it pauses, ask `decide` and resume. Returns the final state,
    or None if `decide` returned None (leave the run paused for a later --resume)."""
    result = await graph.ainvoke(graph_input, config, durability="sync")
    while result.get("__interrupt__"):
        answer = await decide(result["__interrupt__"][0].value)
        if answer is None:
            return None
        result = await graph.ainvoke(Command(resume=bool(answer)), config, durability="sync")
    return (await graph.aget_state(config)).values
