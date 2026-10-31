"""The graph's state: everything the agent knows during one request.

Fields that several nodes add to (observations, rag_hits, warnings) use `operator.add` as a
"reducer": when a node returns {"observations": [x]}, LangGraph APPENDS x instead of
replacing the list. Every other field is simply overwritten by the node that returns it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- input (set by the caller) ---
    question: str
    user_id: int  # comes from the caller/auth, NEVER from the model

    # --- planning loop ---
    iteration: int  # planner runs so far; capped by deps.max_iterations
    pending: dict[str, Any] | None  # the planner's validated next action
    route: str  # name of the next node, set by the planner, read by tool_router

    # --- evidence collected so far (append-only) ---
    observations: Annotated[list[dict[str, Any]], operator.add]  # tool results
    rag_hits: Annotated[list[dict[str, Any]], operator.add]  # retrieved filing passages
    warnings: Annotated[list[str], operator.add]  # caveats to show the user

    # --- output ---
    report: str  # body text, with citations (no disclaimer; Slack server adds its own)
    final_report: str  # what the user sees: report + sources + disclaimer
    approved: bool | None
    notify_result: dict[str, Any] | None
