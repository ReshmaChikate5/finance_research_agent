"""Run the agent against the eval dataset and score it (Step 18).

From the repo root, with the Docker stack up (docker compose up -d) and ANTHROPIC_API_KEY set:

    uv run --project agent_host --extra local python evals/run_evals.py --no-judge --limit 3
    uv run --project agent_host --extra local python evals/run_evals.py
    uv run --project agent_host --extra local python evals/run_evals.py --save-baseline
    uv run --project agent_host --extra local python evals/run_evals.py --compare

Every case runs on a fresh in-memory graph. The fake "human" never approves a Slack post, so
nothing is ever sent. Results are written to evals/results/<timestamp>.json (git-ignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "agent_host", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.graph.build_graph import build_graph, initial_state
from app.graph.hitl import drive
from app.graph.nodes import render_observations
from langgraph.checkpoint.memory import InMemorySaver

from evals import scoring

log = logging.getLogger("evals")
EVALS_DIR = ROOT / "evals"
DATASET = EVALS_DIR / "dataset.jsonl"
BASELINE = EVALS_DIR / "baseline.json"
RESULTS_DIR = EVALS_DIR / "results"


class NoPostClient:
    """Stands in for the Slack client. If anything ever reaches it, the run is marked failed."""

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"eval safety stop: tried to call notify.{tool}")

    async def list_tools(self) -> list[Any]:
        return []


def render_evidence(state: dict[str, Any]) -> str:
    """What the judge sees: tool results plus the FULL filing passages the agent was given."""
    parts = [render_observations(state.get("observations", []), [], per_item=2500, total=12000)]
    for i, h in enumerate(state.get("rag_hits", []), 1):
        m = h["metadata"]
        label = m.get("section_title") or m.get("section", "")
        parts.append(
            f"[passage {i}] {m.get('ticker', '?')} {m.get('form_type', '')} {label}:\n{h['text']}"
        )
    return "\n\n".join(parts)


async def run_agent_once(case: dict[str, Any], deps: Any, timeout: float) -> scoring.Trace:
    """Run one question through a fresh graph and describe what happened."""
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": f"eval-{case['id']}-{time.monotonic_ns()}"},
        "recursion_limit": 6 * deps.max_iterations + 10,
    }
    asked: list[str] = []

    async def decide(payload: dict[str, Any]) -> bool:
        kind = str(payload.get("kind"))
        asked.append(kind)
        return kind == "review_conflict"  # look at flagged drafts; never post, never write

    started = time.monotonic()
    state: dict[str, Any] = {}
    error = ""
    try:
        state = (
            await asyncio.wait_for(
                drive(
                    graph,
                    initial_state(case["question"], case["user_id"], list(deps.menu_warnings)),
                    config,
                    decide,
                ),
                timeout,
            )
            or {}
        )
    except TimeoutError:
        error = f"timed out after {timeout:.0f}s"
        state = (await graph.aget_state(config)).values or {}
    except Exception as exc:
        log.warning("case %s crashed", case["id"], exc_info=True)
        error = f"{type(exc).__name__}: {exc}"
        state = (await graph.aget_state(config)).values or {}
    evidence = render_evidence(state)
    return scoring.trace_from_state(
        state,
        question=case["question"],
        user_id=case["user_id"],
        checkpoints=asked,
        seconds=time.monotonic() - started,
        max_iterations=deps.max_iterations,
        evidence=evidence,
        error=error,
    )


async def judge_case(case: dict[str, Any], trace: scoring.Trace, judge: Any) -> list[scoring.Check]:
    system, user = scoring.judge_prompt(case, trace)
    try:
        reply = await judge.complete(system, user, max_tokens=900)
    except Exception as exc:
        log.warning("judge failed for %s", case["id"], exc_info=True)
        reply = f"judge error: {exc}"
    return scoring.parse_judge(reply, case)


async def evaluate(
    cases: list[dict[str, Any]],
    deps: Any,
    judge: Any | None,
    *,
    repeats: int = 1,
    timeout: float = 240.0,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Run every case `repeats` times, one after another (free-tier APIs have rate limits)."""
    deps.clients = {**deps.clients, "notify": NoPostClient()}
    rows: list[dict[str, Any]] = []
    for case in cases:
        for rep in range(1, repeats + 1):
            trace = await run_agent_once(case, deps, timeout)
            checks = scoring.scripted_checks(case, trace)
            if judge is not None and case.get("rubric"):
                checks += await judge_case(case, trace, judge)
            result = scoring.score_checks(checks)
            row = {
                "id": case["id"],
                "category": case["category"],
                "repeat": rep,
                "seconds": round(trace.seconds, 1),
                "tools": [c.name for c in trace.calls],
                "rag_passages": trace.rag_hits,
                "iterations": trace.iterations,
                "checkpoints": trace.checkpoints,
                "error": trace.error,
                "answer": trace.text[:3000],
                "checks": scoring.asdict_checks(checks),
                **result,
            }
            rows.append(row)
            if progress:
                progress(row)
    return rows


def build_report(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": meta,
        "summary": scoring.summarize(rows),
        "cases": scoring.per_case(rows),
        "rows": rows,
    }


def print_row(row: dict[str, Any]) -> None:
    mark = "PASS" if row["passed"] else "FAIL"
    print(
        f"  {mark}  {row['id']:<28} score={row['score']:.2f}  {row['seconds']:>5.1f}s", flush=True
    )
    for name in row["failed_checks"]:
        print(f"        x {name}")


def print_summary(report: dict[str, Any], cmp: dict[str, Any] | None) -> None:
    s = report["summary"]
    print("\nBy category:")
    for cat, a in s["by_category"].items():
        print(
            f"  {cat:<12} pass {a['pass_rate']:>5.0%}   mean score {a['mean_score']:.2f}"
            f"   ({a['runs']} runs)"
        )
    o = s["overall"]
    print(
        f"\nOVERALL  pass rate {o['pass_rate']:.0%}   mean score {o['mean_score']:.2f}   "
        f"avg {s['mean_seconds']}s per run"
    )
    if s["not_judged_checks"]:
        print(
            f"  note: {s['not_judged_checks']} judge checks could not be read and were not counted"
        )
    if cmp:
        print(f"\nVs baseline: overall score {cmp['overall_delta']:+.3f}")
        for w in cmp["worse"]:
            print(f"  WORSE   {w['id']}: {w['before']:.2f} -> {w['now']:.2f}")
        for b in cmp["better"]:
            print(f"  better  {b['id']}: {b['before']:.2f} -> {b['now']:.2f}")
        if cmp["new_cases"]:
            print("  new cases (no baseline):", ", ".join(cmp["new_cases"]))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the finance research agent.")
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--ids", help="comma-separated case ids to run")
    p.add_argument("--category", help="run only this category (rag, financials, market, ...)")
    p.add_argument("--limit", type=int, help="run only the first N matching cases")
    p.add_argument(
        "--repeats", type=int, default=1, help="run each case N times (model output varies)"
    )
    p.add_argument("--no-judge", action="store_true", help="scripted checks only (cheaper, faster)")
    p.add_argument(
        "--judge-model", default=None, help="model for the judge (default: same as agent)"
    )
    p.add_argument("--timeout", type=float, default=240.0, help="seconds allowed per run")
    p.add_argument("--save-baseline", action="store_true", help="write evals/baseline.json")
    p.add_argument("--compare", action="store_true", help="compare with evals/baseline.json")
    p.add_argument("--tolerance", type=float, default=0.05, help="score drop that counts as worse")
    p.add_argument(
        "--min-pass-rate", type=float, default=None, help="exit 1 if overall pass rate is lower"
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    return args


def select(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in wanted]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")
    return cases


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr)
    from app.config import get_settings
    from app.graph.deps import build_real_deps
    from app.llm import get_chat_model

    cases = select(scoring.load_dataset(args.dataset), args)
    settings = get_settings()
    deps = await build_real_deps(settings)
    judge = None
    judge_name = "none"
    if not args.no_judge:
        update = {}
        if args.judge_model:
            update = (
                {"llm_model": args.judge_model}
                if settings.llm_provider == "anthropic"
                else {"bedrock_model_id": args.judge_model}
            )
        judge_settings = settings.model_copy(update=update)
        judge = get_chat_model(judge_settings)
        judge_name = args.judge_model or (
            settings.llm_model
            if settings.llm_provider == "anthropic"
            else settings.bedrock_model_id
        )

    print(
        f"Running {len(cases)} cases x {args.repeats} (judge: {judge_name}). "
        "Nothing is posted to Slack.\n"
    )
    rows = await evaluate(
        cases, deps, judge, repeats=args.repeats, timeout=args.timeout, progress=print_row
    )

    meta = {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_model": settings.llm_model
        if settings.llm_provider == "anthropic"
        else settings.bedrock_model_id,
        "judge_model": judge_name,
        "repeats": args.repeats,
        "cases": len(cases),
        "dataset": str(args.dataset.name),
    }
    report = build_report(rows, meta)

    cmp = None
    if args.compare:
        if BASELINE.exists():
            cmp = scoring.compare(json.loads(BASELINE.read_text("utf-8")), report, args.tolerance)
        else:
            print("\n(no evals/baseline.json yet; run once with --save-baseline)")
    print_summary(report, cmp)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull results: {out.relative_to(ROOT)}")
    if args.save_baseline:
        slim = {"meta": meta, "summary": report["summary"], "cases": report["cases"]}
        BASELINE.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        print(f"Baseline saved: {BASELINE.relative_to(ROOT)} (commit it)")

    if (
        args.min_pass_rate is not None
        and report["summary"]["overall"]["pass_rate"] < args.min_pass_rate
    ):
        print(f"FAILED: pass rate below {args.min_pass_rate:.0%}", file=sys.stderr)
        return 1
    if cmp and cmp["worse"]:
        print("FAILED: some cases are worse than the baseline", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.exit(asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop))
    sys.exit(asyncio.run(main()))