"""Full local end-to-end check (Step 19).

With the whole stack up (docker compose up -d --build), from agent_host/:

    uv run --extra local python -m scripts.e2e_check            # approval answered "no"
    uv run --extra local python -m scripts.e2e_check --post     # approval answered "yes"

It checks that every service is healthy, asks for an NVDA research brief including the user's
NVDA holding and average cost, and then checks each link of the chain: filings -> market data ->
portfolio -> RAG passages -> cited synthesis -> human approval prompt -> Slack post (only with
--post). The run is saved in the Postgres checkpointer under a thread id, so you can inspect it
afterwards with `python -m scripts.run_agent --history THREAD_ID`.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.graph.build_graph import build_graph, initial_state
from app.graph.checkpointing import CheckpointError, open_checkpointer
from app.graph.deps import build_real_deps
from app.graph.hitl import drive

QUESTION = "Give me a research brief on NVDA, including my portfolio's NVDA holding and average cost."
E2E_MIN_ITERATIONS = 12


@dataclass
class Link:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


def _calls(state: dict[str, Any], server: str) -> list[dict[str, Any]]:
    return [o for o in state.get("observations", []) if o["server"] == server]


def verify_chain(
    state: dict[str, Any], asked: list[dict[str, Any]], *, posted_expected: bool, user_id: int
) -> list[Link]:
    """Turn the final graph state into one pass/fail line per link of the chain. Pure."""
    links: list[Link] = []

    for server, label, tool_hint in (
        ("secedgar", "1. SEC filings/financials retrieved", "check the sec-edgar container logs"),
        (
            "marketdata",
            "2. Market data retrieved",
            "check FINNHUB_API_KEY and the market-data logs",
        ),
        ("portfolio", "3. Portfolio context retrieved", "check the portfolio logs and DB seed"),
    ):
        got = _calls(state, server)
        good = [o for o in got if "error" not in o["result"]]
        if good:
            links.append(Link(label, True, ", ".join(sorted({o["tool"] for o in good}))))
        elif got:
            err = str(got[0]["result"].get("error", ""))[:140]
            links.append(Link(label, False, f"call failed: {err}", tool_hint))
        else:
            links.append(Link(label, False, "the planner never called it", "see prompts.py"))

    uids = {o["args"].get("user_id") for o in _calls(state, "portfolio")}
    if uids:
        links.append(
            Link(
                "3b. Portfolio used the right user",
                uids == {user_id},
                f"user_id seen: {sorted(uids, key=str)}",
            )
        )

    passages = len(state.get("rag_hits", []))
    links.append(
        Link(
            "4. RAG passages retrieved",
            passages > 0,
            f"{passages} passages",
            "run ingestion: is the NVDA 10-K in Chroma (CHROMA_PATH)?",
        )
    )

    report = state.get("report", "")
    final = state.get("final_report", "")
    cited = sorted({int(n) for n in re.findall(r"\[S(\d+)\]", report)})
    links.append(
        Link(
            "5. Synthesis is grounded and cited",
            bool(report) and bool(cited) and "not financial advice" in final.lower(),
            f"{len(report)} chars, citations {cited}" if report else "no report written",
            "if there is no report, read final_report below for the reason",
        )
    )

    approval = [p for p in asked if p.get("kind") == "approve_post"]
    links.append(
        Link(
            "6. Human approval was requested",
            bool(approval),
            f"checkpoints seen: {[p.get('kind') for p in asked]}",
            "the graph must pause before posting (Step 16)",
        )
    )

    result = state.get("notify_result")
    if posted_expected:
        ok = bool(result) and result.get("status") in ("ok", "duplicate_skipped", "dry_run")
        links.append(Link("7. Slack post after approval", ok, str(result), "check notify logs"))
    else:
        links.append(
            Link("7. Nothing posted without --post", not result, f"notify_result={result}")
        )
    return links


async def preflight(settings: Any) -> list[Link]:
    import httpx

    out: list[Link] = []
    urls = {
        "sec-edgar": settings.sec_edgar_url,
        "market-data": settings.market_data_url,
        "portfolio": settings.portfolio_url,
        "notify-slack": settings.notify_slack_url,
    }
    async with httpx.AsyncClient(timeout=5) as http:
        for name, base in urls.items():
            try:
                r = await http.get(f"{base}/health")
                body = r.json() if r.status_code == 200 else {}
                extra = " (SLACK_DRY_RUN: nothing will really post)" if body.get("dry_run") else ""
                out.append(
                    Link(
                        f"service {name}",
                        r.status_code == 200,
                        f"HTTP {r.status_code}{extra}",
                        "docker compose ps; docker compose logs " + name,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                out.append(
                    Link(f"service {name}", False, type(exc).__name__, "docker compose up -d")
                )
    return out


def show(links: list[Link]) -> bool:
    for ln in links:
        print(f"  {'PASS' if ln.ok else 'FAIL'}  {ln.name:<42} {ln.detail}")
        if not ln.ok and ln.hint:
            print(f"        hint: {ln.hint}")
    return all(ln.ok for ln in links)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end check of the whole local stack.")
    p.add_argument("--question", default=QUESTION)
    p.add_argument("--user-id", type=int, default=None)
    p.add_argument("--post", action="store_true", help="answer YES to the Slack approval")
    p.add_argument("--skip-preflight", action="store_true")
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    user_id = args.user_id if args.user_id is not None else settings.demo_user_id

    if not args.skip_preflight:
        print("Services:")
        if not show(await preflight(settings)):
            print("\nFix the services above first (docker compose up -d --build).")
            return 2

    thread_id = f"e2e-{uuid.uuid4().hex[:8]}"
    asked: list[dict[str, Any]] = []

    async def decide(payload: dict[str, Any]) -> bool:
        asked.append(payload)
        print(f"\n>> human checkpoint: {payload.get('kind')}")
        if payload.get("kind") == "approve_post":
            print(f">> answering {'YES' if args.post else 'no'}")
            return args.post
        return True

    print(f"\nAsking: {args.question!r} as user {user_id} (thread {thread_id}) ...")
    try:
        async with open_checkpointer(settings) as saver:
            deps = await build_real_deps(settings)
            deps.max_iterations = max(deps.max_iterations, E2E_MIN_ITERATIONS)
            graph = build_graph(deps, checkpointer=saver)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 6 * settings.max_iterations + 10,
            }
            state = await drive(
                graph, initial_state(args.question, user_id, deps.menu_warnings), config, decide
            )
    except CheckpointError as exc:
        print(exc, file=sys.stderr)
        return 2

    state = state or {}
    print("\n" + "=" * 70 + "\n" + (state.get("final_report") or "(no report)") + "\n" + "=" * 70)
    for w in state.get("warnings", []):
        print("warning:", w)
    print("\nChain:")
    ok = show(verify_chain(state, asked, posted_expected=args.post, user_id=user_id))
    print(f"\nThread {thread_id} is saved: python -m scripts.run_agent --history {thread_id}")
    print("RESULT:", "ALL LINKS PASSED" if ok else "SOMETHING BROKE - fix before Part B")
    return 0 if ok else 1


if __name__ == "__main__":
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    raise SystemExit(asyncio.run(main(), loop_factory=factory))
