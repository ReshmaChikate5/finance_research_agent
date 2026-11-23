"""Assemble the graph.

START -> planner -> (tool_router) -> one of: secedgar / marketdata / portfolio / rag
                                       each of those loops back to the planner
                                  -> synthesizer -> (flagged?) -> conflict_review (pauses)
                                       -> human_approval (pauses)
                                       -> (approved?) -> notify -> END
The loop is bounded by deps.max_iterations (the planner forces synthesis at the cap).
"""

from __future__ import annotations

from app.graph.deps import Deps
from app.graph.nodes import make_nodes
from app.graph.router import after_approval, after_review, after_synthesis, tool_router
from app.graph.state import AgentState
from langgraph.graph import END, START, StateGraph

LOOP_NODES = ("mcp_secedgar_node", "mcp_marketdata_node", "mcp_portfolio_node", "rag_node")


def build_graph(deps: Deps, checkpointer=None):
    nodes = make_nodes(deps)
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "planner_node")
    g.add_conditional_edges(
        "planner_node",
        tool_router,
        {n: n for n in (*LOOP_NODES, "synthesizer_node")},
    )
    for n in LOOP_NODES:
        g.add_edge(n, "planner_node")
    g.add_conditional_edges(
        "synthesizer_node",
        after_synthesis,
        {
            "conflict_review_node": "conflict_review_node",
            "human_approval_node": "human_approval_node",
            END: END,
        },
    )
    g.add_conditional_edges(
        "conflict_review_node",
        after_review,
        {"human_approval_node": "human_approval_node", END: END},
    )
    g.add_conditional_edges(
        "human_approval_node", after_approval, {"mcp_notify_node": "mcp_notify_node", END: END}
    )
    g.add_edge("mcp_notify_node", END)
    # interrupt() needs a checkpointer: the paused state has to be saved somewhere.
    return g.compile(checkpointer=checkpointer)


def initial_state(question: str, user_id: int, warnings: list[str] | None = None) -> AgentState:
    return {
        "question": question,
        "user_id": user_id,
        "iteration": 0,
        "observations": [],
        "rag_hits": [],
        "warnings": list(warnings or []),
    }
