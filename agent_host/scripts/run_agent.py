"""Run the agent once from the terminal, with saved state (Step 15).

Run from agent_host/ with the Docker stack up (docker compose up -d):

    uv run --extra local python -m scripts.run_agent "Give me a research brief on NVDA" --no-notify
    uv run --extra local python -m scripts.run_agent "Brief on NVDA" --no-notify --crash-after 3
    uv run --extra local python -m scripts.run_agent --resume <thread-id> --no-notify
    uv run --extra local python -m scripts.run_agent --history <thread-id>

Every run has a thread id (printed to stderr). The graph saves its state under that id after
each node, so --resume continues a run that was killed or crashed. --crash-after N kills this
process on purpose after N nodes, so you can see a restart for yourself.
--no-notify answers "no" to the approval prompt automatically, so nothing is ever posted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings
from app.graph.build_graph import build_graph, initial_state
from app.graph.checkpointing import CheckpointError, open_checkpointer
from app.graph.deps import build_real_deps


def new_thread_id() -> str:
    return f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask the finance research agent a question.")
    p.add_argument("question", nargs="?", help="what to ask (not needed with --resume/--history)")
    p.add_argument(
        "--user-id", type=int, default=None, help="portfolio owner (default: DEMO_USER_ID)"
    )
    p.add_argument("--no-notify", action="store_true", help="never post to Slack")
    p.add_argument("--verbose", action="store_true", help="log each planner decision")
    p.add_argument("--thread-id", default=None, help="name for this run (default: generated)")
    p.add_argument("--resume", metavar="THREAD_ID", help="continue a run from its last saved node")
    p.add_argument("--history", metavar="THREAD_ID", help="list the saved checkpoints of a run")
    p.add_argument(
        "--crash-after",
        type=int,
        metavar="N",
        help="TEST ONLY: kill this process after N nodes to try --resume",
    )
    args = p.parse_args(argv)
    if not (args.question or args.resume or args.history):
        p.error("give a question, or --resume THREAD_ID, or --history THREAD_ID")
    if args.resume and args.question:
        p.error("--resume continues the saved question; do not pass a new one")
    return args


async def show_history(saver: Any, thread_id: str) -> int:
    if saver is None:
        print("Checkpointing is off (CHECKPOINTER=none), so there is no history.")
        return 1
    rows = [t async for t in saver.alist({"configurable": {"thread_id": thread_id}})]
    if not rows:
        print(f"No checkpoints found for thread {thread_id!r}.")
        return 1
    print(f"{len(rows)} checkpoints for {thread_id} (newest first):")
    for t in rows:
        values = t.checkpoint.get("channel_values", {})
        meta = t.metadata or {}
        print(
            f"  step {meta.get('step', '?'):>3}  source={meta.get('source', '?'):<6}"
            f" iteration={values.get('iteration', '-')}"
            f" observations={len(values.get('observations', []))}"
            f" rag_hits={len(values.get('rag_hits', []))}"
            f" report={'yes' if values.get('report') else 'no'}"
        )
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    try:
        async with open_checkpointer(settings) as saver:
            if args.history:
                return await show_history(saver, args.history)
            return await run(args, settings, saver)
    except CheckpointError as exc:
        print(exc, file=sys.stderr)
        return 2


async def run(args: argparse.Namespace, settings: Any, saver: Any) -> int:
    if args.resume and saver is None:
        print("Cannot resume: checkpointing is off (CHECKPOINTER=none).", file=sys.stderr)
        return 2
    thread_id = args.resume or args.thread_id or new_thread_id()

    async def deny(_: str) -> bool:
        return False

    deps = await build_real_deps(settings, approver=deny if args.no_notify else None)
    if args.no_notify:
        original = deps.approver

        async def show_then_deny(report: str) -> bool:
            print(report)
            return await original(report)

        deps.approver = show_then_deny

    graph = build_graph(deps, checkpointer=saver)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 6 * settings.max_iterations + 10,
    }

    if args.resume:
        snap = await graph.aget_state(config)
        if not snap.values:
            print(f"No saved run found for thread {thread_id!r}.", file=sys.stderr)
            return 2
        if not snap.next:
            print(f"Thread {thread_id} already finished.", file=sys.stderr)
            print(snap.values.get("final_report", "No report was produced."))
            return 0
        print(
            f"Resuming thread {thread_id} before node(s): {', '.join(snap.next)}", file=sys.stderr
        )
        graph_input = None  # None = continue from the last saved checkpoint
    else:
        print(f"thread_id={thread_id}", file=sys.stderr)
        user_id = args.user_id if args.user_id is not None else settings.demo_user_id
        graph_input = initial_state(args.question, user_id, deps.menu_warnings)

    if saver is None:  # checkpointing off: plain run, nothing is saved
        final = await graph.ainvoke(graph_input, config=config)
    else:
        nodes_done = 0
        # durability="sync": each checkpoint is written BEFORE the next node starts, so a kill
        # at any moment loses at most the node that was running.
        async for update in graph.astream(
            graph_input, config=config, stream_mode="updates", durability="sync"
        ):
            nodes_done += len(update)
            if args.verbose:
                print(f"[saved] {', '.join(update)}", file=sys.stderr)
            if args.crash_after is not None and nodes_done >= args.crash_after:
                print(
                    f"\n*** Simulated crash after {nodes_done} nodes. Resume with:\n"
                    f"    python -m scripts.run_agent --resume {thread_id}"
                    f"{' --no-notify' if args.no_notify else ''}",
                    file=sys.stderr,
                )
                sys.stderr.flush()
                os._exit(137)  # no cleanup, like a power cut: only Postgres remembers
        final = (await graph.aget_state(config)).values

    if final.get("notify_result"):
        print("\nSlack:", final["notify_result"])
    elif final.get("approved") is False and final.get("report"):
        print("\nNot posted to Slack.")
    elif not final.get("report"):
        print(final.get("final_report", "No report was produced."))
    return 0


if __name__ == "__main__":
    # Windows: psycopg's async mode cannot run on the default (Proactor) event loop.
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    raise SystemExit(asyncio.run(main(), loop_factory=factory))
