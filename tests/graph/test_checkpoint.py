"""Checkpoint tests: a run stopped halfway can be resumed by a "new process".

The in-memory saver always runs. The Postgres saver runs only when you point it at a scratch
database (it creates its tables there):

    $env:CHECKPOINT_TEST_DATABASE_URL="postgresql://agent_state_rw:agent_state_local@localhost:5432/agent_state"
    uv run --project agent_host --extra local python -m pytest tests/graph/test_checkpoint.py -q
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager

import pytest
from app.config import Settings
from app.graph.build_graph import build_graph, initial_state
from app.graph.checkpointing import CheckpointError, describe_target, open_checkpointer
from app.graph.hitl import drive
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.graph.test_graph import FakeClient, FakeLLM, j, make_deps


def arun(coro):
    """asyncio.run, but on Windows use the selector loop that psycopg's async mode requires."""
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    return asyncio.run(coro, loop_factory=factory)


PG_URL = os.environ.get("CHECKPOINT_TEST_DATABASE_URL")
needs_pg = pytest.mark.skipif(not PG_URL, reason="set CHECKPOINT_TEST_DATABASE_URL to run")

PLAN = [
    j(action="tool", server="marketdata", tool="get_quote", args={"ticker": "NVDA"}),
    j(action="search_filings", query="supply risk", ticker="NVDA"),
    j(action="synthesize"),
]


def new_run(market, *, approve=False):
    """A fresh Deps + fakes, like a newly started process."""
    llm = FakeLLM(list(PLAN), synth="NVDA is fine [S1].")
    return make_deps(llm, clients={"marketdata": market}, approve=approve), llm


def cfg():
    return {"configurable": {"thread_id": f"t-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 60}


async def stop_after(graph, state, config, n):
    """Run the graph but abandon it after n finished nodes (the 'crash')."""
    done = 0
    async for update in graph.astream(state, config, stream_mode="updates", durability="sync"):
        done += len(update)
        if done >= n:
            return done
    return done


@asynccontextmanager
async def memory_saver():
    yield InMemorySaver()


@asynccontextmanager
async def pg_saver():
    s = Settings(checkpointer="postgres", checkpoint_database_url=PG_URL)
    async with open_checkpointer(s) as saver:
        yield saver


def savers():
    return [
        pytest.param(("memory", memory_saver), id="memory"),
        pytest.param(("postgres", pg_saver), id="postgres", marks=needs_pg),
    ]


@pytest.mark.parametrize("kind", savers())
def test_resume_continues_without_repeating_finished_work(kind):
    name, opener = kind
    config = cfg()
    market = FakeClient({"get_quote": {"ticker": "NVDA", "price": 100.0}})

    async def scenario():
        shared = None
        # --- process 1: dies after planner + market-data node (2 nodes) ---
        async with opener() as saver:
            shared = saver  # the memory saver must outlive this block; Postgres reopens
            deps1, _ = new_run(market)
            g1 = build_graph(deps1, checkpointer=saver)
            await stop_after(g1, initial_state("Brief on NVDA", 1), config, 2)
            snap = await g1.aget_state(config)
            assert snap.next == ("planner_node",)
            assert len(snap.values["observations"]) == 1
            assert snap.values["iteration"] == 1
        # --- process 2: new connection, new graph, new fakes; the plan restarts at step 2 ---
        async with opener() as saver2:
            if name == "memory":
                saver2 = shared
            deps2, llm2 = new_run(market)
            llm2.planner = PLAN[1:]  # the new process asks the model for what is left
            g2 = build_graph(deps2, checkpointer=saver2)
            snap = await g2.aget_state(config)
            assert snap.next == ("planner_node",)

            async def no_post(payload):
                return False  # the human declines to post; we only test the resumed work

            return await drive(g2, None, config, no_post), deps2

    final, deps2 = arun(scenario())
    assert len(market.calls) == 1  # the price was fetched once, before the crash
    assert deps2.store.queries == [("supply risk", {"ticker": "NVDA"})]
    assert "[S1]" in final["report"] and final["final_report"].endswith("not financial advice._")
    assert final["iteration"] == 3  # the counter carried over, so the cap still holds


@pytest.mark.parametrize("kind", savers())
def test_threads_are_isolated(kind):
    _, opener = kind
    a, b = cfg(), cfg()

    async def scenario():
        async with opener() as saver:
            for config, question in ((a, "Brief on NVDA"), (b, "Brief on AMD")):
                deps, _ = new_run(FakeClient({"get_quote": {"price": 1}}))
                g = build_graph(deps, checkpointer=saver)
                await g.ainvoke(initial_state(question, 1), config)
            g = build_graph(new_run(FakeClient())[0], checkpointer=saver)
            return (await g.aget_state(a)).values, (await g.aget_state(b)).values

    va, vb = arun(scenario())
    assert va["question"] == "Brief on NVDA" and vb["question"] == "Brief on AMD"


def test_finished_run_has_nothing_left_to_resume():
    config = cfg()

    async def scenario():
        saver = InMemorySaver()
        deps, _ = new_run(FakeClient({"get_quote": {"price": 1}}))
        g = build_graph(deps, checkpointer=saver)

        async def no_post(payload):
            return False

        await drive(g, initial_state("Brief on NVDA", 1), config, no_post)
        return await g.aget_state(config)

    snap = arun(scenario())
    assert snap.next == () and snap.values["final_report"]


def test_memory_mode_gives_an_in_memory_saver():
    async def scenario():
        async with open_checkpointer(Settings(checkpointer="memory")) as saver:
            return saver

    assert isinstance(arun(scenario()), InMemorySaver)


@pytest.mark.parametrize("kind", savers())
def test_a_run_paused_for_approval_survives_a_restart(kind):
    """The point of Step 16 on top of Step 15: a question waiting for a human is not lost
    when the process ends. A new connection and new graph can answer it."""
    name, opener = kind
    config = cfg()
    notify = FakeClient({"post_report": {"status": "posted"}})
    market = FakeClient({"get_quote": {"price": 1}})

    async def scenario():
        async with opener() as saver:
            shared = saver
            deps1, _ = new_run(market)
            deps1.clients["notify"] = notify
            g1 = build_graph(deps1, checkpointer=saver)
            first = await g1.ainvoke(initial_state("Brief on NVDA", 1), config, durability="sync")
            assert first["__interrupt__"][0].value["kind"] == "approve_post"
        assert notify.calls == []  # paused: nothing posted yet
        async with opener() as saver2:
            deps2, _ = new_run(market)
            deps2.clients["notify"] = notify
            g2 = build_graph(deps2, checkpointer=shared if name == "memory" else saver2)
            snap = await g2.aget_state(config)
            assert snap.next == ("human_approval_node",)
            return await g2.ainvoke(Command(resume=True), config, durability="sync")

    final = arun(scenario())
    assert final["notify_result"] == {"status": "posted"}
    assert len(notify.calls) == 1  # approved once, posted once


def test_unreachable_database_gives_a_readable_error_without_the_password():
    url = "postgresql://agent_state_rw:s3cret@127.0.0.1:1/agent_state"

    async def scenario():
        async with open_checkpointer(
            Settings(checkpointer="postgres", checkpoint_database_url=url)
        ):
            pass

    with pytest.raises(CheckpointError) as err:
        arun(scenario())
    text = str(err.value)
    assert "127.0.0.1:1/agent_state" in text and "s3cret" not in text
    assert "CHECKPOINTER=memory" in text


def test_describe_target_hides_password():
    assert describe_target("postgresql://u:pw@db:6543/x") == "db:6543/x"
