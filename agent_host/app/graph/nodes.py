"""The graph's nodes. Each is a small async function: state in, state UPDATE out.

make_nodes(deps) returns them as closures over the injected dependencies. Nothing here
talks to the network directly, so every node can be unit-tested with fakes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.graph.deps import Deps
from app.graph.prompts import PLANNER_SYSTEM, PLANNER_USER, SYNTH_SYSTEM, SYNTH_USER
from app.graph.state import AgentState
from app.llm import LLMError
from app.mcp_clients import ALLOWED_TOOLS, ToolClientError

logger = logging.getLogger("agent.graph")

DISCLAIMER = "_This report is for informational purposes only and is not financial advice._"
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
_CITE_RE = re.compile(r"\[S(\d+)\]")
_PLANNABLE_SERVERS = ("secedgar", "marketdata", "portfolio")  # notify is NOT plannable
_SCALAR = (str, int, float, bool, type(None))

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Helpers (pure functions, tested directly)
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict[str, Any] | None:
    """Pull one JSON object out of model text (tolerates code fences and chatter)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_action(obj: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    """Check a planner decision. Returns (clean_action, "") or (None, reason)."""
    if not obj:
        return None, "reply was not a JSON object"
    action = obj.get("action")
    if action == "synthesize":
        return {"action": "synthesize"}, ""
    if action == "search_filings":
        query = obj.get("query")
        if not isinstance(query, str) or not query.strip():
            return None, "search_filings needs a non-empty string 'query'"
        ticker = obj.get("ticker")
        if isinstance(ticker, str):
            ticker = ticker.strip().upper()
            if not _TICKER_RE.match(ticker):
                return None, f"'{ticker}' is not a valid ticker"
        else:
            ticker = None
        return {"action": "search_filings", "query": query.strip()[:300], "ticker": ticker}, ""
    if action == "tool":
        server, tool, args = obj.get("server"), obj.get("tool"), obj.get("args", {})
        if server not in _PLANNABLE_SERVERS:
            return None, f"server must be one of {', '.join(_PLANNABLE_SERVERS)}"
        if tool not in ALLOWED_TOOLS[server]:
            return None, f"'{tool}' is not a tool on {server}"
        if not isinstance(args, dict):
            return None, "args must be an object"
        for k, v in args.items():
            if not isinstance(k, str) or not (
                isinstance(v, _SCALAR)
                or (isinstance(v, list) and all(isinstance(i, _SCALAR) for i in v))
            ):
                return None, f"argument '{k}' must be a simple value"
        args = {k: v for k, v in args.items() if not (server == "portfolio" and k == "user_id")}
        return {"action": "tool", "server": server, "tool": tool, "args": args}, ""
    return None, "unknown action"


def _clip(text: str, limit: int) -> str:
    return (
        text if len(text) <= limit else text[:limit] + f"... [truncated, {len(text)} chars total]"
    )


def _compact_facts(result: dict[str, Any]) -> str:
    """One short line per metric, so a long EDGAR facts result fits whole in the prompt."""
    lines = [f"{result.get('ticker', '')} {result.get('frequency', '')}".strip()]
    for name, m in result.get("metrics", {}).items():
        vals = ", ".join(f"{v['period_end']}={v['value']:,}" for v in m.get("values", []))
        lines.append(f"{name} ({(m.get('values') or [{}])[0].get('unit', '')}): {vals}")
    if result.get("missing_metrics"):
        lines.append("not found: " + ", ".join(result["missing_metrics"]))
    return "\n".join(lines)


def _render_result(o: dict[str, Any], per_item: int) -> str:
    r = o["result"]
    if o["tool"] == "get_company_facts" and isinstance(r, dict) and "metrics" in r:
        try:
            return _clip(_compact_facts(r), per_item)
        except (KeyError, TypeError):
            pass
    return _clip(json.dumps(r, default=str), per_item)


def render_observations(
    observations: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    per_item: int = 1800,
    total: int = 9000,
) -> str:
    parts: list[str] = []
    for i, o in enumerate(observations, 1):
        body = _render_result(o, per_item)
        parts.append(
            f"{i}. {o['server']}.{o['tool']}({json.dumps(o['args'], default=str)}) -> {body}"
        )
    for h in hits:
        m = h["metadata"]
        snippet = _clip(h["text"].replace("\n", " "), 160)
        label = m.get("section_title") or m.get("section", "")
        parts.append(
            f"- passage: {m.get('ticker', '?')} {m.get('form_type', '')} {label}: {snippet}"
        )
    return _clip("\n".join(parts), total) if parts else "(nothing yet)"


def _same_call(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["server"] == b["server"] and a["tool"] == b["tool"] and a["args"] == b["args"]


def number_passages(hits: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for h in sorted(hits, key=lambda x: x["score"], reverse=True):
        if h["id"] not in seen:
            seen.add(h["id"])
            unique.append(h)
    return unique[:limit]


def apply_citations(text: str, passages: list[dict[str, Any]]) -> tuple[str, list[int], list[int]]:
    """Return (clean_text, used_numbers, removed_invalid_numbers). Invalid [S#] are deleted."""
    valid = set(range(1, len(passages) + 1))
    removed: list[int] = []

    def fix(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if n in valid:
            return m.group(0)
        removed.append(n)
        return ""

    clean = _CITE_RE.sub(fix, text)
    used = sorted({int(n) for n in _CITE_RE.findall(clean)})
    return clean, used, sorted(set(removed))


def sources_block(passages: list[dict[str, Any]], used: list[int]) -> str:
    lines = []
    for n in used:
        m = passages[n - 1]["metadata"]
        label = m.get("section_title") or m.get("section", "")
        body = passages[n - 1]["text"].splitlines()
        text = " ".join(body[1:] or body).strip()
        snippet = text[:90] + ("..." if len(text) > 90 else "")
        lines.append(
            f"[S{n}] {m.get('ticker', '?')} {m.get('form_type', '')} filed "
            f'{m.get("filing_date", "?")}, {label}: "{snippet}" ({m.get("source_url", "no URL")})'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------
def make_nodes(deps: Deps) -> dict[str, NodeFn]:
    async def ask(system: str, user: str, max_tokens: int) -> str:
        """One model call, retried once: a single timeout should not end the whole run."""
        try:
            return await deps.llm.complete(system, user, max_tokens=max_tokens)
        except LLMError as exc:
            logger.info("model call failed (%s); retrying once", exc)
            await asyncio.sleep(deps.llm_retry_delay)
            return await deps.llm.complete(system, user, max_tokens=max_tokens)

    async def planner_node(state: AgentState) -> dict[str, Any]:
        step = state.get("iteration", 0)
        if step >= deps.max_iterations:
            return {
                "pending": {"action": "synthesize"},
                "route": "synthesizer",
                "warnings": [
                    f"Stopped gathering after {deps.max_iterations} steps (iteration cap)."
                ],
            }

        obs, hits = state.get("observations", []), state.get("rag_hits", [])
        user = PLANNER_USER.format(
            question=state["question"],
            tools=deps.tool_menu,
            n_obs=len(obs),
            n_hits=len(hits),
            observations=render_observations(obs, hits),
            step=step + 1,
            limit=deps.max_iterations,
        )
        feedback = ""
        for attempt in range(2):  # one retry with the validation error as feedback
            try:
                reply = await ask(PLANNER_SYSTEM, user + feedback, 500)
            except LLMError as exc:
                return {
                    "pending": {"action": "synthesize"},
                    "route": "synthesizer",
                    "iteration": step + 1,
                    "warnings": [f"Planning failed: {exc}"],
                }
            action, problem = validate_action(extract_json(reply))
            if action is not None:
                break
            logger.info("planner reply rejected (%s)", problem)
            feedback = (
                f"\n\nYour previous reply was rejected: {problem}. Reply with valid JSON only."
            )
        else:
            return {
                "pending": {"action": "synthesize"},
                "route": "synthesizer",
                "iteration": step + 1,
                "warnings": [f"The planner could not produce a valid action ({problem})."],
            }

        if action["action"] == "tool" and any(_same_call(action, o) for o in obs):
            return {
                "pending": {"action": "synthesize"},
                "route": "synthesizer",
                "iteration": step + 1,
                "warnings": ["The planner repeated a call it had already made; moving on."],
            }

        route = {"synthesize": "synthesizer", "search_filings": "rag"}.get(
            action["action"], f"mcp_{action.get('server', '')}"
        )
        return {"pending": action, "route": route, "iteration": step + 1}

    def tool_node(server: str) -> NodeFn:
        async def node(state: AgentState) -> dict[str, Any]:
            p = state["pending"] or {}
            args = dict(p.get("args", {}))
            if server == "portfolio":
                args["user_id"] = state["user_id"]  # never trust the model for identity
            try:
                result = await deps.clients[server].call(p["tool"], args)
            except ToolClientError as exc:
                result = {"error": str(exc)}
            return {
                "observations": [
                    {"server": server, "tool": p["tool"], "args": args, "result": result}
                ]
            }

        return node

    async def rag_node(state: AgentState) -> dict[str, Any]:
        p = state["pending"] or {}
        filters = {"ticker": p["ticker"]} if p.get("ticker") else None
        try:
            found = await deps.store.query(p["query"], k=5, filters=filters)
        except Exception as exc:  # noqa: BLE001 - store errors must not crash the run
            return {"warnings": [f"Filing search failed ({type(exc).__name__}: {exc})."]}
        if not found:
            msg = f"Filing search for '{p['query']}' found nothing (is the index loaded? run ingestion)."
            return {"warnings": [msg]}
        return {
            "rag_hits": [
                {"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata}
                for h in found
            ]
        }

    async def synthesizer_node(state: AgentState) -> dict[str, Any]:
        obs = state.get("observations", [])
        warnings = list(state.get("warnings", []))
        passages = number_passages(state.get("rag_hits", []))
        usable = [o for o in obs if "error" not in o["result"]]
        for o in obs:
            if "error" in o["result"]:
                warnings.append(f"{o['server']}.{o['tool']} failed: {o['result']['error']}")

        if not usable and not passages:
            msg = "I could not gather any usable data for this question.\n" + "\n".join(
                f"- {w}" for w in warnings
            )
            return {"report": "", "final_report": msg.strip(), "approved": False}

        passages_text = (
            "\n\n".join(
                f"[S{i}] {p['metadata'].get('ticker', '?')} {p['metadata'].get('form_type', '')} "
                f"{p['metadata'].get('section_title') or p['metadata'].get('section', '')}\n{_clip(p['text'], 1200)}"
                for i, p in enumerate(passages, 1)
            )
            or "(none)"
        )
        user = SYNTH_USER.format(
            question=state["question"],
            observations=render_observations(usable, [], per_item=6000, total=24000),
            passages=passages_text,
            warnings="\n".join(f"- {w}" for w in warnings) or "(none)",
        )
        try:
            draft = await ask(SYNTH_SYSTEM, user, 6000)
        except LLMError as exc:
            return {
                "report": "",
                "final_report": f"The report could not be written: {exc}",
                "approved": False,
            }

        stop = getattr(deps.llm, "last_stop_reason", "end_turn")
        if stop not in ("end_turn", "stop_sequence"):
            warnings.append(f"The model's answer may be incomplete (stop reason: {stop}).")
        body, used, removed = apply_citations(draft.strip(), passages)
        if removed:
            warnings.append(
                f"Removed {len(removed)} citation(s) that did not match any retrieved passage."
            )
        report = body
        if used:
            report += "\n\n*Sources*\n" + sources_block(passages, used)
        shown = [w for w in warnings if w]
        if shown:
            report += "\n\n*Data caveats*\n" + "\n".join(f"- {w}" for w in shown)
        return {"report": report, "final_report": f"{report}\n\n{DISCLAIMER}"}

    async def human_approval_node(state: AgentState) -> dict[str, Any]:
        return {"approved": bool(await deps.approver(state["final_report"]))}

    async def mcp_notify_node(state: AgentState) -> dict[str, Any]:
        key = f"brief-{deps.today()}-{hashlib.sha256(state['report'].encode()).hexdigest()[:12]}"
        try:
            result = await deps.clients["notify"].call(
                "post_report",
                {"channel": deps.slack_channel, "report_text": state["report"], "dedupe_key": key},
            )
        except ToolClientError as exc:
            result = {"error": str(exc)}
        return {"notify_result": result}

    return {
        "planner_node": planner_node,
        "mcp_secedgar_node": tool_node("secedgar"),
        "mcp_marketdata_node": tool_node("marketdata"),
        "mcp_portfolio_node": tool_node("portfolio"),
        "rag_node": rag_node,
        "synthesizer_node": synthesizer_node,
        "human_approval_node": human_approval_node,
        "mcp_notify_node": mcp_notify_node,
    }
