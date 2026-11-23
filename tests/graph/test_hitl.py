"""Human-in-the-loop tests: the graph pauses at the right places and obeys the answer."""

from __future__ import annotations

import asyncio

from app.graph.build_graph import build_graph, initial_state
from app.graph.hitl import describe_interrupt, drive
from app.graph.nodes import split_conflict_line
from app.mcp_clients import ALLOWED_TOOLS, WRITE_TOOLS
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.graph.test_graph import FakeClient, FakeLLM, j, make_deps, run

SEARCH = [j(action="search_filings", query="risks", ticker="NVDA"), j(action="synthesize")]


def kinds(deps) -> list[str]:
    return [c["kind"] for c in deps.checkpoints]


# ---------------------------------------------------------------------------
# The conflict marker
# ---------------------------------------------------------------------------
def test_split_conflict_line():
    assert split_conflict_line("Body.\nCONFLICTS: none") == ("Body.", "")
    assert split_conflict_line("Body.\nCONFLICTS: None.") == ("Body.", "")
    assert split_conflict_line("Body.\nCONFLICTS: prices differ") == ("Body.", "prices differ")
    assert split_conflict_line("Body only.") == ("Body only.", "")
    assert split_conflict_line("Body.\n`CONFLICTS: x`")[1] == "x"


# ---------------------------------------------------------------------------
# Pause before posting
# ---------------------------------------------------------------------------
def test_pauses_for_approval_and_posts_once_after_yes():
    notify = FakeClient({"post_report": {"status": "posted"}})
    deps = make_deps(FakeLLM(list(SEARCH)), clients={"notify": notify}, approve=True)
    final = run(deps)
    assert kinds(deps) == ["approve_post"]
    assert len(notify.calls) == 1 and final["approved"] is True


def test_nothing_is_posted_while_paused():
    notify = FakeClient()
    deps = make_deps(FakeLLM(list(SEARCH)), clients={"notify": notify})
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "p"}, "recursion_limit": 60}
    first = asyncio.run(graph.ainvoke(initial_state("Brief on NVDA", 1), config))
    payload = first["__interrupt__"][0].value
    assert payload["kind"] == "approve_post" and "not financial advice" in payload["report"]
    assert notify.calls == []

    async def later():
        snap = await graph.aget_state(config)
        assert snap.next == ("human_approval_node",)
        return await graph.ainvoke(Command(resume=False), config)

    final = asyncio.run(later())
    assert final["approved"] is False and notify.calls == []


def test_no_flag_means_no_review_step():
    deps = make_deps(FakeLLM(list(SEARCH), synth="All good [S1].\nCONFLICTS: none"))
    run(deps)
    assert kinds(deps) == ["approve_post"]


# ---------------------------------------------------------------------------
# Pause on a conflict / low confidence
# ---------------------------------------------------------------------------
def test_conflict_pauses_for_review_before_approval():
    notify = FakeClient({"post_report": {"status": "posted"}})
    llm = FakeLLM(list(SEARCH), synth="NVDA looks fine [S1].\nCONFLICTS: quote and filing disagree")
    deps = make_deps(llm, clients={"notify": notify})
    final = run(deps, confirm_conflicts=True)
    assert kinds(deps) == ["review_conflict", "approve_post"]
    assert "quote and filing disagree" in deps.checkpoints[0]["conflict"]
    assert "CONFLICTS:" not in final["report"]  # the marker line never reaches the reader
    assert "Possible conflict" in final["report"]  # but the reader is told in the caveats
    assert len(notify.calls) == 1


def test_stopping_at_review_never_asks_to_post():
    notify = FakeClient()
    llm = FakeLLM(list(SEARCH), synth="Draft [S1].\nCONFLICTS: sources disagree")
    deps = make_deps(llm, clients={"notify": notify})
    final = run(deps, confirm_conflicts=False)
    assert kinds(deps) == ["review_conflict"]  # no approval question after a stop
    assert notify.calls == [] and deps.approvals == []
    assert "Stopped at your request" in final["final_report"] and final["report"] == ""


def test_cut_off_answer_also_triggers_review():
    llm = FakeLLM(list(SEARCH))
    llm.last_stop_reason = "max_tokens"
    deps = make_deps(llm)
    run(deps)
    assert kinds(deps)[0] == "review_conflict"
    assert "max_tokens" in deps.checkpoints[0]["conflict"]


# ---------------------------------------------------------------------------
# Pause before a tool that changes something
# ---------------------------------------------------------------------------
def with_write_tool(monkeypatch):
    monkeypatch.setitem(ALLOWED_TOOLS, "portfolio", ALLOWED_TOOLS["portfolio"] | {"propose_trade"})
    monkeypatch.setitem(WRITE_TOOLS, "portfolio", frozenset({"propose_trade"}))


WRITE_PLAN = [
    j(action="tool", server="portfolio", tool="propose_trade", args={"ticker": "NVDA"}),
    j(action="synthesize"),
]


def test_write_tool_waits_for_a_yes(monkeypatch):
    with_write_tool(monkeypatch)
    portfolio = FakeClient({"propose_trade": {"status": "queued"}})
    deps = make_deps(FakeLLM(list(WRITE_PLAN)), clients={"portfolio": portfolio}, approve=False)
    run(deps, confirm_conflicts=True)
    assert deps.checkpoints[0]["kind"] == "confirm_write"
    assert deps.checkpoints[0]["args"]["user_id"] == 1  # shown with the forced identity
    assert portfolio.calls == [("propose_trade", {"ticker": "NVDA", "user_id": 1})]


def test_write_tool_declined_is_never_called(monkeypatch):
    with_write_tool(monkeypatch)
    portfolio = FakeClient({"propose_trade": {"status": "queued"}})
    llm = FakeLLM(list(WRITE_PLAN), synth="Could not act.")
    deps = make_deps(llm, clients={"portfolio": portfolio})
    final = run(deps, confirm_conflicts=False)
    assert portfolio.calls == []
    assert deps.checkpoints[0]["kind"] == "confirm_write"
    assert "declined" in final["final_report"]  # the reader is told it did not happen


def test_read_tools_never_pause():
    market = FakeClient({"get_quote": {"price": 1}})
    plan = [j(action="tool", server="marketdata", tool="get_quote", args={"ticker": "A"})]
    deps = make_deps(FakeLLM([*plan, j(action="synthesize")]), clients={"marketdata": market})
    run(deps)
    assert "confirm_write" not in kinds(deps)


def test_describe_interrupt_has_text_for_every_kind():
    for payload in (
        {"kind": "approve_post", "report": "R", "channel": "#c"},
        {"kind": "review_conflict", "report": "R", "conflict": "why"},
        {"kind": "confirm_write", "server": "s", "tool": "t", "args": {}},
    ):
        text, question = describe_interrupt(payload)
        assert text and question.endswith("[y/N] ")


def test_drive_returns_none_when_left_paused():
    deps = make_deps(FakeLLM(list(SEARCH)))
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "d"}, "recursion_limit": 60}

    async def leave(payload):
        return None

    assert asyncio.run(drive(graph, initial_state("q", 1), config, leave)) is None
