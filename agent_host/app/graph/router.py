"""Conditional edges: small pure functions that read state and name the next node.

They contain no I/O, so routing can be tested by feeding in a state dict.
"""

from __future__ import annotations

from langgraph.graph import END

from app.graph.state import AgentState

# planner "route" value -> node name
_TOOL_ROUTES = {
    "mcp_secedgar": "mcp_secedgar_node",
    "mcp_marketdata": "mcp_marketdata_node",
    "mcp_portfolio": "mcp_portfolio_node",
    "rag": "rag_node",
    "synthesizer": "synthesizer_node",
}


def tool_router(state: AgentState) -> str:
    """After the planner: where next? Unknown values fall back to synthesis (safe default)."""
    return _TOOL_ROUTES.get(state.get("route", ""), "synthesizer_node")


def after_synthesis(state: AgentState) -> str:
    """No report: stop. A flagged draft goes to a human review first; otherwise to approval."""
    if not state.get("report"):
        return END
    return "conflict_review_node" if state.get("conflict") else "human_approval_node"


def after_review(state: AgentState) -> str:
    """Carry on to the approval step only if the reviewer said to go on."""
    return "human_approval_node" if state.get("review") is True else END


def after_approval(state: AgentState) -> str:
    """Post to Slack only on an explicit yes."""
    return "mcp_notify_node" if state.get("approved") is True else END
