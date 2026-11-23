"""verify_chain turns a final graph state into pass/fail lines (Step 19)."""

from __future__ import annotations

from scripts.e2e_check import verify_chain


def obs(server, tool, result=None, **args):
    return {"server": server, "tool": tool, "args": args, "result": result or {"ok": 1}}


GOOD = {
    "observations": [
        obs("secedgar", "get_company_facts", ticker="NVDA"),
        obs("marketdata", "get_quote", ticker="NVDA"),
        obs("portfolio", "get_holdings", user_id=1),
    ],
    "rag_hits": [{"text": "t", "metadata": {}}],
    "report": "NVDA is big [S1].\n\n[S1] x",
    "final_report": "NVDA is big [S1].\n\n_not financial advice._",
}
ASKED = [{"kind": "approve_post"}]


def by_name(links):
    return {ln.name.split(" ", 1)[0]: ln for ln in links}


def test_full_chain_passes_without_posting():
    links = verify_chain(GOOD, ASKED, posted_expected=False, user_id=1)
    assert all(ln.ok for ln in links), [ln for ln in links if not ln.ok]


def test_post_expected_needs_a_notify_result():
    bad = verify_chain(GOOD, ASKED, posted_expected=True, user_id=1)
    assert by_name(bad)["7."].ok is False
    state = {**GOOD, "notify_result": {"status": "ok"}}
    assert all(ln.ok for ln in verify_chain(state, ASKED, posted_expected=True, user_id=1))


def test_unexpected_post_is_flagged():
    state = {**GOOD, "notify_result": {"status": "ok"}}
    assert by_name(verify_chain(state, ASKED, posted_expected=False, user_id=1))["7."].ok is False


def test_missing_and_failed_tools_are_reported_differently():
    state = {
        **GOOD,
        "observations": [obs("secedgar", "get_company_facts", {"error": "429 rate limited"})],
    }
    links = by_name(verify_chain(state, ASKED, posted_expected=False, user_id=1))
    assert "429" in links["1."].detail and links["1."].ok is False
    assert "never called" in links["2."].detail
    assert "3b." not in links


def test_missing_portfolio_call_is_reported():
    state = {
        **GOOD,
        "observations": [obs("marketdata", "get_quote", ticker="NVDA")],
    }
    link = by_name(verify_chain(state, ASKED, posted_expected=False, user_id=1))["3."]
    assert not link.ok and link.detail == "the planner never called it"


def test_wrong_user_rag_and_approval_failures():
    state = {**GOOD, "rag_hits": [], "observations": [obs("portfolio", "get_holdings", user_id=2)]}
    links = by_name(verify_chain(state, [], posted_expected=False, user_id=1))
    assert not links["3b."].ok and not links["4."].ok and not links["6."].ok


def test_uncited_or_missing_report_fails_synthesis():
    for report in ("", "no citations here"):
        state = {**GOOD, "report": report, "final_report": report + " not financial advice"}
        links = by_name(verify_chain(state, ASKED, posted_expected=False, user_id=1))
        assert not links["5."].ok
