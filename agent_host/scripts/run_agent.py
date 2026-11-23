"""Run the agent once from the terminal, with saved state and human checkpoints (Steps 15-16).

Run from agent_host/ with the Docker stack up (docker compose up -d):

    uv run --extra local python -m scripts.run_agent "Give me a research brief on NVDA"
    uv run --extra local python -m scripts.run_agent "Brief on NVDA" --no-notify
    uv run --extra local python -m scripts.run_agent "Brief on NVDA" --pause
    uv run --extra local python -m scripts.run_agent --resume <thread-id> --answer yes
    uv run --extra local python -m scripts.run_agent --history <thread-id>

The graph PAUSES (and saves its state) when it needs a person: before posting to Slack, when a
draft is flagged as unreliable, and before any tool that changes data. This script asks you on
the terminal. With --pause it does not ask: it exits, and you answer later with
--resume THREAD --answer yes|no, possibly from a new terminal or after a restart.
--no-notify answers "no" to the Slack question automatically, so nothing is ever posted.
--crash-after N kills the process on purpose after N nodes (for trying --resume).
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
from app.graph.hitl import describe_interrupt
from langgraph.types import Command

YES = {"y", "yes", "true", "approve", "go"}
NO = {"n", "no", "false", "reject", "stop"}


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
    p.add_argument("--resume", metavar="THREAD_ID", help="continue a saved or paused run")
    p.add_argument("--answer", metavar="yes|no", help="with --resume: answer the pending question")
    p.add_argument("--pause", action="store_true", help="do not ask; exit and wait for --resume")
    p.add_argument("--history", metavar="THREAD_ID", help="list the saved checkpoints of a run")
    p.add_argument("--crash-after", type=int, metavar="N", help="TEST ONLY: kill after N nodes")
    args = p.parse_args(argv)
    if not (args.question or args.resume or args.history):
        p.error("give a question, or --resume THREAD_ID, or --history THREAD_ID")
    if args.resume and args.question:
        p.error("--resume continues the saved question; do not pass a new one")
    if args.answer and not args.resume:
        p.error("--answer only makes sense with --resume")
    if args.answer and args.answer.strip().lower() not in YES | NO:
        p.error("--answer must be yes or no")
    return args


async def show_history(saver: Any, thread_id: str) -> int:
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


def make_decider(args: argparse.Namespace, presets: list[bool]):
    """Return decide(payload) -> True/False, or None to leave the run paused."""

    async def decide(payload: dict[str, Any]) -> bool | None:
        text, question = describe_interrupt(payload)
        kind = payload.get("kind")
        print("\n" + "=" * 70 + f"\nHUMAN CHECKPOINT: {kind}\n" + "=" * 70)
        print(text)
        print("=" * 70)
        if presets:  # --answer applies to the first question only
            answer = presets.pop()
            print(f"(answered {'yes' if answer else 'no'} from --answer)")
            return answer
        if args.no_notify and kind == "approve_post":
            return False
        if args.no_notify and kind == "review_conflict":
            return True  # only shows the draft; nothing is posted
        if args.pause or not sys.stdin.isatty():
            return None
        answer = await asyncio.to_thread(input, question)
        return answer.strip().lower() in YES

    return decide


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
    thread_id = args.resume or args.thread_id or new_thread_id()
    deps = await build_real_deps(settings)
    graph = build_graph(deps, checkpointer=saver)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 6 * settings.max_iterations + 10,
    }
    presets = [args.answer.strip().lower() in YES] if args.answer else []

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
        graph_input: Any = None  # None = continue; a pending question is asked again
    else:
        print(f"thread_id={thread_id}", file=sys.stderr)
        user_id = args.user_id if args.user_id is not None else settings.demo_user_id
        graph_input = initial_state(args.question, user_id, deps.menu_warnings)

    decide = make_decider(args, presets)
    nodes_done = 0
    while True:
        asked: dict[str, Any] | None = None
        # durability="sync": each checkpoint is written BEFORE the next node starts, so a kill
        # at any moment loses at most the node that was running.
        async for update in graph.astream(
            graph_input, config=config, stream_mode="updates", durability="sync"
        ):
            if "__interrupt__" in update:
                asked = update["__interrupt__"][0].value
                continue
            nodes_done += len(update)
            if args.verbose:
                print(f"[saved] {', '.join(update)}", file=sys.stderr)
            if args.crash_after is not None and nodes_done >= args.crash_after:
                print(
                    f"\n*** Simulated crash after {nodes_done} nodes. Resume with:\n"
                    f"    python -m scripts.run_agent --resume {thread_id}",
                    file=sys.stderr,
                )
                sys.stderr.flush()
                os._exit(137)  # no cleanup, like a power cut: only Postgres remembers
        if asked is None:
            break
        answer = await decide(asked)
        if answer is None:
            print(
                f"\nPaused at a human checkpoint. Answer later with:\n"
                f"    python -m scripts.run_agent --resume {thread_id} --answer yes   (or no)",
                file=sys.stderr,
            )
            return 0
        graph_input = Command(resume=answer)

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
