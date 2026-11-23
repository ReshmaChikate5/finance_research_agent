"""Slack notify server: channel lock, idempotency, truncation, failure handling, dry run."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.unit.helpers import load_server, run

REPORT = "*NVDA brief*\n- Supply is concentrated at a few foundries [S1]\n- Price is near highs."
CHANNEL = "#finance-reports"
WEBHOOK_SECRET = "hooks.slack.com/services/T0/B0/secret"


@pytest.fixture
def slack(notify, router):
    router.add("hooks.slack.com", text="ok")
    return notify


def post(mod, text=REPORT, channel=CHANNEL, key=None):
    return run(mod.post_report(channel, text, key))


def sent_text(router) -> str:
    import json

    return json.loads(router.calls[-1].content)["text"]


# ---------------------------------------------------------------------------
# Start-up and health
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", ["", "http://hooks.slack.com/x", "https://evil.example/hook"])
def test_refuses_to_start_with_a_non_slack_webhook(monkeypatch, url):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", url)
    with pytest.raises(RuntimeError, match=r"hooks\.slack\.com"):
        load_server("notify_slack")


def test_dry_run_needs_no_webhook(make_notify):
    assert make_notify(dry_run=True).DRY_RUN is True


def test_health_never_exposes_the_webhook(notify):
    body = run(notify.health(None)).body.decode()
    assert "mcp-notify" in body and "secret" not in body and "hooks.slack.com" not in body


# ---------------------------------------------------------------------------
# Input rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "has space", "UPPER CASE!", "#" + "a" * 90])
def test_invalid_channel_names(slack, router, bad):
    assert "not a valid channel name" in post(slack, channel=bad)["error"]
    assert router.calls == []


def test_only_the_configured_channel_is_allowed(slack, router):
    error = post(slack, channel="#general")["error"]
    assert "not allowed" in error and CHANNEL in error and router.calls == []


@pytest.mark.parametrize("channel", ["#finance-reports", "finance-reports", "#Finance-Reports"])
def test_channel_match_ignores_hash_and_case(slack, channel):
    assert post(slack, channel=channel)["status"] == "posted"


@pytest.mark.parametrize("short", ["", "   ", "too short"])
def test_short_reports_are_rejected(slack, router, short):
    assert "too short" in post(slack, text=short)["error"]
    assert router.calls == []


@pytest.mark.parametrize("bad", ["has space", "semi;colon", "x" * 129, ""])
def test_invalid_dedupe_keys_are_rejected(slack, router, bad):
    assert "dedupe_key may only contain" in post(slack, key=bad)["error"]
    assert router.calls == []


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------
def test_a_report_is_posted_with_the_disclaimer_footer(slack, router):
    result = post(slack, key="NVDA-brief-1")
    assert (
        result["status"] == "posted"
        and result["channel"] == CHANNEL
        and result["dedupe_key"] == "NVDA-brief-1"
    )
    text = sent_text(router)
    assert text.startswith("*NVDA brief*") and text.endswith(slack.DISCLAIMER)
    assert result["chars_posted"] == len(text)
    assert str(router.calls[0].url) == "https://hooks.slack.com/services/T0/B0/secret"


def test_the_same_key_never_posts_twice(slack, router):
    first, second = post(slack, key="k1"), post(slack, key="k1")
    assert first["status"] == "posted" and second["status"] == "duplicate_skipped"
    assert second["chars_posted"] == first["chars_posted"] and router.count("hooks.slack.com") == 1


def test_a_different_key_posts_again(slack, router):
    post(slack, key="k1")
    assert post(slack, key="k2")["status"] == "posted" and router.count("hooks.slack.com") == 2


def test_without_a_key_identical_text_is_still_deduplicated(slack, router):
    a, b = post(slack), post(slack)
    assert a["dedupe_key"].startswith("sha256:") and a["dedupe_key"] == b["dedupe_key"]
    assert b["status"] == "duplicate_skipped" and router.count("hooks.slack.com") == 1
    assert post(slack, text=REPORT + " Edited.")["status"] == "posted"


def test_the_dedupe_window_expires(slack, router):
    post(slack, key="k1")
    key, (_, state, details) = next(iter(slack._sent.items()))
    slack._sent[key] = (0.0, state, details)  # expired long ago
    assert post(slack, key="k1")["status"] == "posted" and router.count("hooks.slack.com") == 2


def test_very_long_reports_are_truncated_with_a_note(slack, router):
    post(slack, text="x" * 50_000)
    text = sent_text(router)
    assert "truncated" in text and text.endswith(slack.DISCLAIMER)
    assert len(text) < 38_500


def test_simultaneous_identical_posts_send_only_once(make_notify, router):
    mod = make_notify()

    async def slow_slack(request: httpx.Request) -> httpx.Response:
        router.calls.append(request)
        await asyncio.sleep(0.05)
        return httpx.Response(200, text="ok")

    mod._http = httpx.AsyncClient(transport=httpx.MockTransport(slow_slack))

    async def both():
        return await asyncio.gather(
            mod.post_report(CHANNEL, REPORT, "same"), mod.post_report(CHANNEL, REPORT, "same")
        )

    results = run(both())
    assert sorted("error" in r for r in results) == [False, True]
    assert any("being sent right now" in r.get("error", "") for r in results)
    assert len(router.calls) == 1


# ---------------------------------------------------------------------------
# Failure handling: retry only when it is safe
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "hint"),
    [
        (404, "webhook was deleted"),
        (410, "archived"),
        (403, "not allowed to post"),
        (400, "message format"),
        (429, "rate limiting"),
        (500, "Try again later"),
    ],
)
def test_slack_rejections_are_explained_and_can_be_retried(make_notify, router, status, hint):
    mod = make_notify()
    router.add("hooks.slack.com", status=status, times=1)
    router.add("hooks.slack.com", text="ok")
    error = post(mod, key="k")["error"]
    assert f"HTTP {status}" in error and hint in error
    assert WEBHOOK_SECRET not in error
    assert post(mod, key="k")["status"] == "posted"  # the key was released


def test_connection_failure_before_sending_allows_a_retry(make_notify, router):
    mod = make_notify()
    router.add("hooks.slack.com", exc=httpx.ConnectError("refused"), times=1)
    router.add("hooks.slack.com", text="ok")
    assert "Nothing was posted" in post(mod, key="k")["error"]
    assert post(mod, key="k")["status"] == "posted"


def test_failure_after_sending_blocks_blind_retries(make_notify, router):
    mod = make_notify()
    router.add("hooks.slack.com", exc=httpx.ReadTimeout("no reply"), times=1)
    first = post(mod, key="k")["error"]
    assert "may or may not have been posted" in first and "NEW dedupe_key" in first
    second = post(mod, key="k")["error"]
    assert "may or may not have been delivered" in second
    assert router.count("hooks.slack.com") == 1  # the second call never sent anything
    router.add("hooks.slack.com", text="ok")
    assert post(mod, key="k2")["status"] == "posted"  # a NEW key goes through


def test_unexpected_bugs_release_the_claim_and_hide_details(slack, monkeypatch):
    calls = []

    async def boom(text):
        calls.append(1)
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(slack, "_send", boom)
    error = post(slack, key="k")["error"]
    assert "Unexpected internal error" in error and "secret" not in error
    assert "k" not in slack._sent  # claim released, so a retry is possible


def test_the_webhook_url_is_never_logged(slack, caplog):
    caplog.set_level("DEBUG")
    post(slack, key="k")
    assert "secret" not in caplog.text and "hooks.slack.com" not in caplog.text


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def test_dry_run_sends_nothing(make_notify, router):
    mod = make_notify(dry_run=True)
    result = post(mod, key="k")
    assert result["status"] == "dry_run" and result["chars_posted"] > len(REPORT)
    assert router.calls == []


def test_dry_run_still_deduplicates(make_notify):
    mod = make_notify(dry_run=True)
    post(mod, key="k")
    assert post(mod, key="k")["status"] == "duplicate_skipped"
