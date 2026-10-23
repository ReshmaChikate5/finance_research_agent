"""Portfolio MCP server (READ-ONLY).

Tools:
    - get_holdings:   a user's positions with total cost and average cost
    - get_cost_basis: purchase lots and cost totals for one ticker

Safety: parameterized queries only (db.py), a SELECT-only Postgres role, read-only sessions,
and strict input validation. There are deliberately NO write tools here. Any future write tool
must sit behind a human-approval interrupt in the LangGraph agent (Step 16).

Transports (set MCP_TRANSPORT): stdio (default, MCP Inspector) | streamable-http (Docker/ECS).
With stdio, never print() to stdout; logging goes to stderr.
"""

import logging
import os
import re
import sys
from typing import Any

import db
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
logger = logging.getLogger("portfolio")

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")
MAX_USER_ID = 2**63 - 1

mcp = FastMCP(
    "portfolio",
    instructions=(
        "Read-only access to a user's portfolio. Call get_holdings first to see positions, "
        "then get_cost_basis for the purchase lots of one ticker. user_id is a number."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Liveness probe used by docker compose and ECS."""
    return JSONResponse({"status": "ok", "service": "mcp-portfolio"})


class PortfolioError(Exception):
    """An error whose message is safe and useful to show to the model."""


def _fail(tool: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, PortfolioError):
        return {"error": str(exc)}
    logger.exception("Unexpected error in %s", tool)  # details go to logs, not to the model
    return {
        "error": f"Could not complete {tool} because the portfolio database is unavailable "
        "or returned an unexpected error. Try again later."
    }


def _valid_user_id(user_id: int) -> int:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or not 1 <= user_id <= MAX_USER_ID:
        raise PortfolioError("user_id must be a positive whole number, e.g. 1.")
    return user_id


@mcp.tool()
async def get_holdings(user_id: int) -> dict[str, Any]:
    """List all positions a user currently holds, with cost information.

    Read-only. Use this first to learn which tickers the user owns before researching them.
    Values are in the user's base currency; this tool does not fetch live prices (use the
    market-data server's get_quote for that).

    Args:
        user_id: Numeric id of the user, e.g. 1.

    Returns:
        On success: {"user_id", "holding_count", "total_cost", "holdings": [{"ticker",
        "shares", "total_cost" (all lots incl. fees), "average_cost_per_share",
        "lot_count"}]}. A user with no positions returns an empty list.
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        get_holdings(user_id=1)
    """
    try:
        uid = _valid_user_id(user_id)
        if not await db.user_exists(uid):
            raise PortfolioError(f"No user with id {uid} exists.")
        holdings = await db.fetch_holdings(uid)
        return {
            "user_id": uid,
            "holding_count": len(holdings),
            "total_cost": round(sum(h["total_cost"] or 0 for h in holdings), 2),
            "holdings": holdings,
        }
    except Exception as exc:  # noqa: BLE001 - tools must never leak stack traces
        return _fail("get_holdings", exc)


@mcp.tool()
async def get_cost_basis(user_id: int, ticker: str) -> dict[str, Any]:
    """Get the purchase lots and cost basis for one holding of a user.

    Read-only. Use it to compute gains or losses together with a live price from get_quote.

    Args:
        user_id: Numeric id of the user, e.g. 1.
        ticker: Ticker the user holds, e.g. "NVDA" (case-insensitive).

    Returns:
        On success: {"user_id", "ticker", "total_shares", "total_cost" (incl. fees),
        "average_cost_per_share", "lots": [{"purchase_date", "shares", "price_per_share",
        "fees", "lot_cost"}]}, oldest lot first.
        On failure: {"error": "<explanation and suggestion>"}; if the user does not hold
        the ticker, the error lists the tickers they do hold.

    Example:
        get_cost_basis(user_id=1, ticker="NVDA")
    """
    try:
        uid = _valid_user_id(user_id)
        t = ticker.strip().upper()
        if not _TICKER_RE.match(t):
            raise PortfolioError(f"'{ticker}' is not a valid ticker. Use letters/digits, e.g. 'NVDA'.")
        if not await db.user_exists(uid):
            raise PortfolioError(f"No user with id {uid} exists.")
        if not await db.holding_exists(uid, t):
            owned = await db.list_tickers(uid)
            held = ", ".join(owned) if owned else "none"
            raise PortfolioError(f"User {uid} does not hold {t}. Tickers held: {held}.")

        lots = await db.fetch_cost_basis(uid, t)
        shares = sum(lot["shares"] for lot in lots)
        cost = sum(lot["lot_cost"] for lot in lots)
        return {
            "user_id": uid,
            "ticker": t,
            "total_shares": round(shares, 6),
            "total_cost": round(cost, 2),
            "average_cost_per_share": round(cost / shares, 4) if shares else None,
            "lots": lots,
        }
    except Exception as exc:  # noqa: BLE001
        return _fail("get_cost_basis", exc)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "streamable-http", "sse"):
        raise SystemExit(f"Unsupported MCP_TRANSPORT '{transport}'")
    logger.info("Starting portfolio MCP server (transport=%s)", transport)
    mcp.run(transport=transport)  # type: ignore[arg-type]