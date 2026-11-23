"""Tests for the eval framework itself (Step 18). No network, no API key.

uv run --project agent_host --extra local python -m pytest tests/evals -q
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from app.graph.deps import Deps
from app.vectorstore.base import RetrievedChunk

from evals import run_evals, scoring

ROOT = Path(__file__).resolve().parents[2]
DISC = "_This report is for informational purposes only and is not financial advice._"


def trace(**kw) -> scoring.Trace:
    base = {"question": "q", "user_id": 1, "report": "NVDA grew [S1].\n\n[S1] x", "iterations": 2}
    base["final_report"] = base["report"] + "\n\n" + DISC
    base.update(kw)
    return scoring.Trace(**base)


def call(server, tool, **args):
    return scoring.ToolCall(server, tool, args, True)


def result(checks, name):
    return next(c for c in checks if c.name == name)


# ---------------------------------------------------------------- dataset
def test_shipped_dataset_is_valid():
    cases = scoring.load_dataset(ROOT / "evals" / "dataset.jsonl")
    assert len(cases) >= 20
    assert {c["category"] for c in cases} >= {"rag", "financials", "market", "portfolio", "safety"}
    for c in cases:
        known = {
            "tools_called",
            "tools_called_any",
            "tools_not_called",
            "min_distinct_tools",
            "max_tool_calls",
            "tool_args",
            "portfolio_user_id",
            "used_rag",
            "text_contains_all",
            "text_contains_any",
            "text_not_contains",
            "has_citations",
        }
        assert set(c["expect"]) <= known, c["id"]
        assert c["expect"] or c["rubric"], f"{c['id']} checks nothing"


def test_load_dataset_rejects_bad_files(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"id": "a"}\n')
    with pytest.raises(ValueError, match="missing category, question"):
        scoring.load_dataset(f)
    f.write_text("not json\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        scoring.load_dataset(f)
    row = '{"id":"a","category":"c","question":"q"}\n'
    f.write_text(row + row)
    with pytest.raises(ValueError, match="duplicate"):
        scoring.load_dataset(f)
    f.write_text("# comment\n\n" + row)
    assert scoring.load_dataset(f)[0]["user_id"] == 1


# ---------------------------------------------------------------- scripted checks
def test_default_checks_pass_for_a_clean_run():
    checks = scoring.scripted_checks({"expect": {}}, trace())
    assert all(c.passed for c in checks)


def test_safety_checks_catch_a_slack_post_and_missing_disclaimer():
    t = trace(calls=[call("notify", "post_report")], final_report="no disclaimer here")
    checks = scoring.scripted_checks({"expect": {}}, t)
    assert result(checks, "nothing_posted_to_slack").passed is False
    assert result(checks, "disclaimer_present").passed is False


def test_citation_to_missing_source_fails():
    t = trace(report="claim [S2]\n\n[S1] only", final_report="claim [S2]\n\n[S1] only " + DISC)
    assert (
        result(scoring.scripted_checks({"expect": {}}, t), "citations_listed_in_sources").passed
        is False
    )


def test_tool_expectations():
    t = trace(
        calls=[
            call("marketdata", "get_quote", ticker="aapl"),
            call("portfolio", "get_holdings", user_id=1),
        ]
    )
    expect = {
        "tools_called": ["marketdata.get_quote"],
        "tools_called_any": ["secedgar.*", "portfolio.*"],
        "tools_not_called": ["notify.*"],
        "tool_args": [{"call": "marketdata.get_quote", "arg": "ticker", "equals": "AAPL"}],
        "portfolio_user_id": 1,
        "max_tool_calls": 2,
        "min_distinct_tools": 2,
    }
    assert all(c.passed for c in scoring.scripted_checks({"expect": expect}, t))
    bad = {
        "tools_called": ["secedgar.get_company_facts"],
        "max_tool_calls": 1,
        "portfolio_user_id": 2,
        "tool_args": [{"call": "marketdata.get_quote", "arg": "ticker", "equals": "MSFT"}],
    }
    failed = [c for c in scoring.scripted_checks({"expect": bad}, t) if not c.passed]
    assert len(failed) == 4


def test_isolation_check_sees_a_wrong_user_id():
    t = trace(calls=[call("portfolio", "get_holdings", user_id=2)])
    assert (
        result(
            scoring.scripted_checks({"expect": {"portfolio_user_id": 1}}, t),
            "portfolio calls used only user_id=1",
        ).passed
        is False
    )


def test_text_and_rag_expectations():
    t = trace(rag_hits=3)
    expect = {
        "used_rag": True,
        "has_citations": True,
        "text_contains_all": ["nvda"],
        "text_contains_any": ["grew", "fell"],
        "text_not_contains": ["AMD"],
    }
    assert all(c.passed for c in scoring.scripted_checks({"expect": expect}, t))
    assert not all(
        c.passed for c in scoring.scripted_checks({"expect": {"used_rag": True}}, trace())
    )
    assert not all(
        c.passed for c in scoring.scripted_checks({"expect": {"text_not_contains": ["nvda"]}}, t)
    )


def test_no_report_skips_disclaimer_and_run_error_fails():
    t = trace(report="", final_report="", error="timed out")
    checks = scoring.scripted_checks({"expect": {}}, t)
    assert "disclaimer_present" not in {c.name for c in checks}
    assert result(checks, "run_completed").passed is False


# ---------------------------------------------------------------- judge parsing
CASE = {"rubric": ["a", "b", "c"]}


def test_parse_judge_reads_clean_and_fenced_json():
    text = (
        '{"results": [{"n":1,"pass":true,"reason":"ok"},'
        '{"n":2,"pass":false,"reason":"bad"},{"n":3,"pass":true}]}'
    )
    for reply in (text, "```json\n" + text + "\n```", "Here you go: " + text):
        got = scoring.parse_judge(reply, CASE)
        assert [c.passed for c in got] == [True, False, True]
        assert got[0].kind == "judge"


def test_parse_judge_marks_unreadable_items_as_not_judged():
    got = scoring.parse_judge("garbage", CASE)
    assert [c.passed for c in got] == [None, None, None]
    got = scoring.parse_judge('{"results": [{"n":1,"pass":"yes"},{"n":2,"pass":true}]}', CASE)
    assert [c.passed for c in got] == [None, True, None]


def test_judge_prompt_contains_question_evidence_answer_and_numbered_rubric():
    system, user = scoring.judge_prompt(
        {"rubric": ["is grounded", "no advice"]}, trace(evidence="EV-TEXT")
    )
    assert "JSON" in system and "data" in system
    for part in ("QUESTION:", "EV-TEXT", "ANSWER:", "1. is grounded", "2. no advice"):
        assert part in user


# ---------------------------------------------------------------- scoring maths
def test_score_checks_ignores_unjudged():
    checks = [scoring.Check("a", True), scoring.Check("b", False), scoring.Check("c", None)]
    r = scoring.score_checks(checks)
    assert r["score"] == 0.5 and r["passed"] is False
    assert r["checks_not_judged"] == 1 and r["failed_checks"] == ["b"]
    assert scoring.score_checks([scoring.Check("a", True)])["passed"] is True
    assert scoring.score_checks([])["passed"] is False


def rows():
    return [
        {
            "id": "a",
            "category": "rag",
            "score": 1.0,
            "passed": True,
            "seconds": 10,
            "checks_not_judged": 0,
        },
        {
            "id": "a",
            "category": "rag",
            "score": 0.5,
            "passed": False,
            "seconds": 20,
            "checks_not_judged": 1,
        },
        {
            "id": "b",
            "category": "mkt",
            "score": 1.0,
            "passed": True,
            "seconds": 3,
            "checks_not_judged": 0,
        },
    ]


def test_summarize_and_per_case():
    s = scoring.summarize(rows())
    assert s["overall"]["pass_rate"] == round(2 / 3, 4)
    assert s["by_category"]["rag"] == {"runs": 2, "pass_rate": 0.5, "mean_score": 0.75}
    assert s["not_judged_checks"] == 1
    assert scoring.per_case(rows())["a"] == {"pass_rate": 0.5, "mean_score": 0.75}


def test_compare_flags_regressions_improvements_and_new_cases():
    def rep(cases, overall):
        return {"cases": cases, "summary": {"overall": {"mean_score": overall}}}

    base = rep(
        {"a": {"mean_score": 1.0}, "b": {"mean_score": 0.5}, "gone": {"mean_score": 1.0}}, 0.8
    )
    now = rep(
        {"a": {"mean_score": 0.6}, "b": {"mean_score": 0.9}, "new": {"mean_score": 1.0}}, 0.75
    )
    c = scoring.compare(base, now, tolerance=0.05)
    assert [w["id"] for w in c["worse"]] == ["a"]
    assert [b["id"] for b in c["better"]] == ["b"]
    assert c["new_cases"] == ["new"] and c["removed_cases"] == ["gone"]
    assert c["overall_delta"] == -0.05
    assert scoring.compare(base, base, 0.05)["worse"] == []


# ---------------------------------------------------------------- end to end with fakes
class FakeLLM:
    def __init__(self, planner, synth, judge_reply=None):
        self.planner, self.synth, self.judge_reply = list(planner), synth, judge_reply

    async def complete(self, system, user, max_tokens=1500):
        if system == scoring.JUDGE_SYSTEM:
            return self.judge_reply
        if "planning step" in system:
            return self.planner.pop(0) if self.planner else '{"action": "synthesize"}'
        return self.synth


class FakeClient:
    def __init__(self, responses=None):
        self.responses, self.calls = responses or {}, []

    async def call(self, tool, args):
        self.calls.append((tool, dict(args)))
        return self.responses.get(tool, {"ok": True})

    async def list_tools(self):
        return []


class Store:
    async def upsert(self, chunks):
        return 0

    async def query(self, text, k=5, filters=None):
        return [
            RetrievedChunk(
                "c1",
                "Supply is concentrated.",
                0.9,
                {
                    "ticker": "NVDA",
                    "form_type": "10-K",
                    "filing_date": "2024-02-21",
                    "section": "1A",
                    "section_title": "Risk Factors",
                    "source_url": "https://sec.example/x",
                },
            )
        ]


def deps_for(llm, clients=None):
    async def approver(report):  # would post if ever asked; the eval human never says yes
        return True

    full = {
        k: (clients or {}).get(k, FakeClient())
        for k in ("secedgar", "marketdata", "portfolio", "notify")
    }
    return Deps(
        llm=llm,
        clients=full,
        store=Store(),
        approver=approver,
        tool_menu="(menu)",
        max_iterations=6,
        llm_retry_delay=0,
        today=lambda: "2026-10-28",
    )


QUOTE = '{"action":"tool","server":"marketdata","tool":"get_quote","args":{"ticker":"AAPL"}}'
CASES = [
    {
        "id": "quote",
        "category": "market",
        "question": "AAPL price?",
        "user_id": 1,
        "expect": {"tools_called": ["marketdata.get_quote"], "tools_not_called": ["notify.*"]},
        "rubric": ["grounded"],
    },
]
SYN = '{"action":"synthesize"}'
GOOD_JUDGE = '{"results": [{"n": 1, "pass": true, "reason": "fine"}]}'


def test_runner_end_to_end_passes_and_never_posts():
    market = FakeClient({"get_quote": {"price": 190.5}})
    llm = FakeLLM([QUOTE, SYN, QUOTE, SYN], "AAPL trades at 190.5.\nCONFLICTS: none", GOOD_JUDGE)
    deps = deps_for(llm, {"marketdata": market})
    got = asyncio.run(run_evals.evaluate(CASES, deps, llm, repeats=2, timeout=30))
    assert [r["passed"] for r in got] == [True, True]
    assert got[0]["tools"] == ["marketdata.get_quote"] and got[0]["checkpoints"] == ["approve_post"]
    assert market.calls[0] == ("get_quote", {"ticker": "AAPL"})
    assert any(c["kind"] == "judge" for c in got[0]["checks"])


def test_runner_reports_failures_and_unreadable_judge():
    llm = FakeLLM(['{"action":"synthesize"}'], "I do not know.\nCONFLICTS: none", "not json")
    got = asyncio.run(run_evals.evaluate(CASES, deps_for(llm), llm, timeout=30))[0]
    assert got["passed"] is False
    assert "called marketdata.get_quote" in got["failed_checks"]
    assert got["checks_not_judged"] == 1


def test_runner_survives_a_crashing_case_and_a_judge_error():
    class Boom(FakeLLM):
        async def complete(self, system, user, max_tokens=1500):
            if system == scoring.JUDGE_SYSTEM:
                raise RuntimeError("judge down")
            raise RuntimeError("model down")

    llm = Boom([], "")
    got = asyncio.run(run_evals.evaluate(CASES * 1, deps_for(llm), llm, timeout=30))[0]
    assert got["passed"] is False  # reported, not raised


def test_runner_blocks_slack_even_if_the_graph_tried():
    deps = deps_for(FakeLLM([], ""))
    asyncio.run(run_evals.evaluate([], deps, None))
    with pytest.raises(RuntimeError, match="eval safety stop"):
        asyncio.run(deps.clients["notify"].call("post_report", {}))


def test_runner_without_judge_skips_rubric():
    llm = FakeLLM([QUOTE, '{"action":"synthesize"}'], "AAPL 190.\nCONFLICTS: none")
    got = asyncio.run(run_evals.evaluate(CASES, deps_for(llm), None, timeout=30))[0]
    assert not any(c["kind"] == "judge" for c in got["checks"])


def test_timeout_is_recorded_as_a_failure():
    class Slow(FakeLLM):
        async def complete(self, system, user, max_tokens=1500):
            await asyncio.sleep(5)

    llm = Slow([], "")
    got = asyncio.run(run_evals.evaluate(CASES, deps_for(llm), None, timeout=0.2))[0]
    assert "timed out" in got["error"] and got["passed"] is False


def test_select_and_report_helpers(tmp_path):
    args = run_evals.parse_args(["--ids", "a", "--no-judge"])
    cases = [{"id": "a", "category": "x"}, {"id": "b", "category": "y"}]
    assert run_evals.select(cases, args) == [cases[0]]
    with pytest.raises(SystemExit):
        run_evals.select(cases, run_evals.parse_args(["--ids", "zzz"]))
    with pytest.raises(SystemExit):
        run_evals.select(cases, run_evals.parse_args(["--category", "none"]))
    rep = run_evals.build_report(rows(), {"m": 1})
    assert set(rep) == {"meta", "summary", "cases", "rows"}
    json.dumps(rep)


# ---------------------------------------------------------------- the command line
def test_main_saves_baseline_then_compares_and_gates(tmp_path, monkeypatch, capsys):
    import app.config
    import app.graph.deps
    import app.llm

    llm = FakeLLM([QUOTE, SYN] * 10, "AAPL trades at 190.5.\nCONFLICTS: none", GOOD_JUDGE)

    class S:
        llm_provider = "anthropic"
        llm_model = "fake-model"
        bedrock_model_id = "fake-bedrock"

        def model_copy(self, update):
            return self

    async def fake_deps(settings):
        return deps_for(llm, {"marketdata": FakeClient({"get_quote": {"price": 1}})})

    monkeypatch.setattr(app.config, "get_settings", lambda: S())
    monkeypatch.setattr(app.graph.deps, "build_real_deps", fake_deps)
    monkeypatch.setattr(app.llm, "get_chat_model", lambda s: llm)
    monkeypatch.setattr(run_evals, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(run_evals, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_evals, "ROOT", tmp_path)
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(CASES[0]) + "\n")
    base = ["--dataset", str(data), "--timeout", "30", "--judge-model", "other-model"]

    assert asyncio.run(run_evals.main([*base, "--save-baseline"])) == 0
    assert (tmp_path / "baseline.json").exists() and list((tmp_path / "results").glob("*.json"))
    assert asyncio.run(run_evals.main([*base, "--compare", "--min-pass-rate", "0.9"])) == 0
    assert "Vs baseline" in capsys.readouterr().out

    llm.planner = []  # the agent now skips the tool: a regression
    llm.synth = "No idea.\nCONFLICTS: none"
    assert asyncio.run(run_evals.main([*base, "--compare"])) == 1
    assert asyncio.run(run_evals.main([*base, "--min-pass-rate", "0.5"])) == 1


def test_judge_evidence_includes_full_passages():
    long = "x" * 900 + " END-OF-PASSAGE"
    state = {
        "observations": [],
        "rag_hits": [{"text": long, "metadata": {"ticker": "NVDA", "form_type": "10-K"}}],
    }
    assert "END-OF-PASSAGE" in run_evals.render_evidence(state)