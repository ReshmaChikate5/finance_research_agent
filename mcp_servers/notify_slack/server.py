"""Slack notify MCP server (side-effecting, idempotent).

Tool:
    - post_report: post a finished research report to Slack via an incoming webhook.

Design notes:
    * An incoming webhook is bound to ONE channel (chosen when the webhook was created).
      `channel` is therefore validated against SLACK_CHANNEL_NAME instead of redirecting the post,
      so the model can never believe it posted somewhere it did not.
    * Idempotent: every post has a dedupe key (caller-supplied, or a hash of channel + text).
      A key is claimed BEFORE sending, so concurrent or retried calls cannot double-post.
    * A disclaimer footer is appended structurally (not left to the model to remember).
    * The webhook URL is a secret: it is never logged or returned.
    * SLACK_DRY_RUN=true skips the real POST, handy for testing with MCP Inspector.

The dedupe store is in memory (per container, resets on restart). For several replicas on AWS,
move it to a shared store such as DynamoDB or Redis.

Transports (set MCP_TRANSPORT): stdio (default, MCP Inspector) | streamable-http (Docker/ECS).
With stdio, never print() to stdout; logging goes to stderr.
"""

import hashlib
import logging
import os
import re
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs full URLs; ours is a secret
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("notify_slack")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
DRY_RUN = os.getenv("SLACK_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
CHANNEL_NAME = os.getenv("SLACK_CHANNEL_NAME", "#finance-reports").strip()
DEDUPE_TTL = int(os.getenv("SLACK_DEDUPE_TTL_SECONDS", str(24 * 3600)))

if not DRY_RUN and not WEBHOOK_URL.startswith("https://hooks.slack.com/"):
    raise RuntimeError(
        "SLACK_WEBHOOK_URL must be a https://hooks.slack.com/... URL. "
        "Set SLACK_DRY_RUN=true to run without posting."
    )

MAX_TEXT = 38_000  # Slack truncates message text around 40,000 characters
MIN_TEXT = 20
DISCLAIMER = "_This report is for informational purposes only and is not financial advice._"

_CHANNEL_RE = re.compile(r"^#?[a-z0-9][a-z0-9_\-]{0,79}$")
_KEY_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


def _norm_channel(name: str) -> str:
    return "#" + name.strip().lstrip("#").lower()


mcp = FastMCP(
    "notify-slack",
    instructions=(
        "Posts a finished report to Slack. This has a real-world side effect: only call "
        "post_report after the user has approved the report. Safe to retry with the same "
        "dedupe_key: duplicates are skipped."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Liveness probe used by docker compose and ECS (never exposes the webhook)."""
    return JSONResponse(
        {"status": "ok", "service": "mcp-notify", "dry_run": DRY_RUN, "channel": CHANNEL_NAME}
    )


# ---------------------------------------------------------------------------
# Idempotency store: key -> (expires_at, state, details)
# ---------------------------------------------------------------------------
# state: "pending" (being sent), "posted", "unknown" (send may or may not have arrived)
_sent: dict[str, tuple[float, str, dict[str, Any]]] = {}


def _purge() -> None:
    now = time.monotonic()
    for key in [k for k, (exp, _, _) in _sent.items() if exp <= now]:
        del _sent[key]


class NotifyError(Exception):
    """An error whose message is safe and useful to show to the model."""


def _fail(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, NotifyError):
        return {"error": str(exc)}
    logger.exception("Unexpected error in post_report")
    return {"error": "Unexpected internal error in post_report. The message was not confirmed as sent."}


_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    return _http


def _build_text(report_text: str) -> str:
    body = report_text.strip()
    note = ""
    if len(body) > MAX_TEXT:
        body = body[:MAX_TEXT].rstrip()
        note = "\n\n_(Report truncated to fit Slack's message size limit.)_"
    return f"{body}{note}\n\n{DISCLAIMER}"


async def _send(text: str) -> tuple[bool, str]:
    """POST to the webhook. Returns (delivered, detail). Raises httpx errors for the caller."""
    resp = await _client().post(WEBHOOK_URL, json={"text": text})
    if resp.status_code == 200:
        return True, "ok"
    body = resp.text.strip()[:100]  # Slack returns short codes like "invalid_payload"
    return False, f"HTTP {resp.status_code} {body}".strip()


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@mcp.tool()
async def post_report(
    channel: str, report_text: str, dedupe_key: str | None = None
) -> dict[str, Any]:
    """Post a finished report to the team's Slack channel (idempotent).

    This has a real-world side effect, so call it only after the user approved the report.
    Calling it again with the same dedupe_key (or the same channel and text) will NOT post
    a second time, so retries are safe. A disclaimer footer is added automatically.

    Args:
        channel: Slack channel to post to, e.g. "#finance-reports". The webhook is bound to
            one channel, so this must match it; otherwise an error names the right one.
        report_text: The report body in Slack mrkdwn (*bold*, _italic_, bullet lists).
            20 to about 38,000 characters; longer text is truncated.
        dedupe_key: Optional stable id for this report, e.g. "NVDA-brief-2026-10-23".
            Letters, digits and . _ : - only (max 128). Defaults to a hash of channel + text.

    Returns:
        On success: {"status": "posted" | "duplicate_skipped" | "dry_run", "channel",
        "dedupe_key", "chars_posted"}. "duplicate_skipped" means it was already posted
        within the last 24 hours, which counts as success; do not retry.
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        post_report(channel="#finance-reports", report_text="*NVDA brief* ...",
        dedupe_key="NVDA-brief-2026-10-23")
    """
    key: str | None = None
    claimed = False
    try:
        if not _CHANNEL_RE.match(channel.strip().lower()):
            raise NotifyError(f"'{channel}' is not a valid channel name. Example: '#finance-reports'.")
        if _norm_channel(channel) != _norm_channel(CHANNEL_NAME):
            raise NotifyError(
                f"Channel '{channel}' is not allowed. This server only posts to {CHANNEL_NAME} "
                "(the channel its Slack webhook is bound to). Retry with that channel."
            )
        channel = _norm_channel(CHANNEL_NAME)

        if not isinstance(report_text, str) or len(report_text.strip()) < MIN_TEXT:
            raise NotifyError(f"report_text is too short (minimum {MIN_TEXT} characters).")

        text = _build_text(report_text)
        if dedupe_key is not None:
            if not _KEY_RE.match(dedupe_key):
                raise NotifyError(
                    "dedupe_key may only contain letters, digits and . _ : - (max 128 characters)."
                )
            key = dedupe_key
        else:
            key = "sha256:" + hashlib.sha256(f"{channel}\n{text}".encode()).hexdigest()[:32]

        _purge()
        existing = _sent.get(key)
        if existing:
            _, state, details = existing
            if state == "posted":
                return {"status": "duplicate_skipped", "channel": channel, "dedupe_key": key, **details}
            if state == "pending":
                raise NotifyError(
                    "An identical post is being sent right now. Do not retry; check Slack in a moment."
                )
            raise NotifyError(
                "An earlier attempt with this dedupe_key may or may not have been delivered "
                "(the connection failed mid-send). Check the Slack channel first. "
                "If it is missing, retry with a NEW dedupe_key."
            )

        # Claim the key BEFORE sending. No await sits between the check and this line, so
        # concurrent calls on the event loop cannot both pass the check above.
        expiry = time.monotonic() + DEDUPE_TTL
        _sent[key] = (expiry, "pending", {})
        claimed = True

        if DRY_RUN:
            _sent[key] = (expiry, "posted", {"chars_posted": len(text), "dry_run": True})
            logger.info("DRY RUN: would post %d chars to %s", len(text), channel)
            return {"status": "dry_run", "channel": channel, "dedupe_key": key, "chars_posted": len(text)}

        try:
            delivered, detail = await _send(text)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            del _sent[key]  # request never reached Slack: safe to retry
            claimed = False
            raise NotifyError("Could not connect to Slack. Nothing was posted; you can retry.") from exc
        except httpx.HTTPError as exc:
            _sent[key] = (expiry, "unknown", {})  # may have been delivered
            claimed = False
            raise NotifyError(
                f"The connection to Slack failed after sending ({type(exc).__name__}). The message "
                "may or may not have been posted. Check the channel before retrying; if it is "
                "missing, retry with a NEW dedupe_key."
            ) from exc

        if delivered:
            _sent[key] = (expiry, "posted", {"chars_posted": len(text)})
            logger.info("Posted %d chars to %s", len(text), channel)
            return {"status": "posted", "channel": channel, "dedupe_key": key, "chars_posted": len(text)}

        del _sent[key]  # Slack rejected it: nothing posted, retry allowed after a fix
        claimed = False
        hints = {
            "404": "The webhook was deleted or the URL is wrong. Create a new webhook.",
            "410": "The channel is archived. Pick another channel and create a new webhook.",
            "403": "The app is not allowed to post to this channel.",
            "400": "Slack rejected the message format. Simplify the formatting and retry.",
            "429": "Slack is rate limiting this webhook. Wait a minute and retry.",
        }
        code = detail.split()[1] if detail.startswith("HTTP ") else ""
        raise NotifyError(f"Slack rejected the post ({detail}). {hints.get(code, 'Try again later.')}")
    except Exception as exc:  # noqa: BLE001 - tools must never leak stack traces
        if claimed and key is not None and _sent.get(key, (0, "", {}))[1] == "pending":
            del _sent[key]  # unexpected failure before sending: release the claim
        return _fail(exc)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "streamable-http", "sse"):
        raise SystemExit(f"Unsupported MCP_TRANSPORT '{transport}'")
    logger.info(
        "Starting notify-slack MCP server (transport=%s, channel=%s, dry_run=%s)",
        transport, CHANNEL_NAME, DRY_RUN,
    )
    mcp.run(transport=transport)  # type: ignore[arg-type]