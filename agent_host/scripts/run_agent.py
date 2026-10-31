"""Run the agent once from the terminal.

Run from agent_host/ with the Docker stack up (docker compose up -d):

    uv run --extra local python -m scripts.run_agent "Give me a research brief on NVDA"
    uv run --extra local python -m scripts.run_agent "How is my portfolio doing?" --user-id 1
    uv run --extra local python -m scripts.run_agent "Brief on NVDA" --no-notify

--no-notify answers "no" to the approval prompt automatically, so nothing is ever posted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from app.config import get_settings
from app.graph.build_graph import build_graph, initial_state
from app.graph.deps import build_real_deps


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ask the finance research agent a question.")
    p.add_argument("question")
    p.add_argument(
        "--user-id", type=int, default=None, help="portfolio owner (default: DEMO_USER_ID)"
    )
    p.add_argument("--no-notify", action="store_true", help="never post to Slack")
    p.add_argument("--verbose", action="store_true", help="log each planner decision")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    async def deny(_: str) -> bool:
        return False

    deps = await build_real_deps(settings, approver=deny if args.no_notify else None)
    if args.no_notify:
        original = deps.approver

        async def show_then_deny(report: str) -> bool:
            print(report)
            return await original(report)

        deps.approver = show_then_deny

    graph = build_graph(deps)
    user_id = args.user_id if args.user_id is not None else settings.demo_user_id
    final = await graph.ainvoke(
        initial_state(args.question, user_id, deps.menu_warnings),
        config={"recursion_limit": 6 * settings.max_iterations + 10},
    )

    if final.get("notify_result"):
        print("\nSlack:", final["notify_result"])
    elif final.get("approved") is False and final.get("report"):
        print("\nNot posted to Slack.")
    elif not final.get("report"):
        print(final.get("final_report", "No report was produced."))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
