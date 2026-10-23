"""Portfolio database access."""
"""Read-only database access for the portfolio MCP server.

All queries use asyncpg positional parameters ($1, $2): user input is NEVER formatted into SQL.
The connection role (portfolio_ro) only has SELECT, and the session is also forced read-only.
"""

import os
from decimal import Decimal
from typing import Any

import asyncpg

_pool: asyncpg.Pool | None = None


def _dsn() -> str:
    dsn = os.getenv("PORTFOLIO_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "PORTFOLIO_DATABASE_URL is not set. Use the read-only role, e.g. "
            "postgresql://portfolio_ro:<password>@localhost:5432/portfolio"
        )
    return dsn


async def get_pool() -> asyncpg.Pool:
    """Create the connection pool lazily, once."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            _dsn(),
            min_size=1,
            max_size=5,
            command_timeout=10,
            server_settings={"default_transaction_read_only": "on"},
        )
    return _pool


def _f(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


async def user_exists(user_id: int) -> bool:
    pool = await get_pool()
    return await pool.fetchval("SELECT EXISTS (SELECT 1 FROM users WHERE id = $1)", user_id)


async def fetch_holdings(user_id: int) -> list[dict[str, Any]]:
    """Holdings with total cost, aggregated from cost_basis lots."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT h.ticker,
               h.shares,
               COALESCE(SUM(c.shares * c.price_per_share + c.fees), 0) AS total_cost,
               COUNT(c.id)                                             AS lot_count
        FROM holdings h
        LEFT JOIN cost_basis c ON c.holding_id = h.id
        WHERE h.user_id = $1
        GROUP BY h.id, h.ticker, h.shares
        ORDER BY h.ticker
        """,
        user_id,
    )
    result = []
    for r in rows:
        shares, cost = r["shares"], r["total_cost"]
        result.append(
            {
                "ticker": r["ticker"],
                "shares": _f(shares),
                "total_cost": _f(cost),
                "average_cost_per_share": _f(cost / shares) if shares else None,
                "lot_count": r["lot_count"],
            }
        )
    return result


async def fetch_cost_basis(user_id: int, ticker: str) -> list[dict[str, Any]]:
    """Individual purchase lots for one holding."""
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT c.purchase_date, c.shares, c.price_per_share, c.fees
        FROM cost_basis c
        JOIN holdings h ON h.id = c.holding_id
        WHERE h.user_id = $1 AND h.ticker = $2
        ORDER BY c.purchase_date, c.id
        """,
        user_id,
        ticker,
    )
    return [
        {
            "purchase_date": r["purchase_date"].isoformat(),
            "shares": _f(r["shares"]),
            "price_per_share": _f(r["price_per_share"]),
            "fees": _f(r["fees"]),
            "lot_cost": _f(r["shares"] * r["price_per_share"] + r["fees"]),
        }
        for r in rows
    ]


async def holding_exists(user_id: int, ticker: str) -> bool:
    pool = await get_pool()
    return await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM holdings WHERE user_id = $1 AND ticker = $2)", user_id, ticker
    )


async def list_tickers(user_id: int) -> list[str]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT ticker FROM holdings WHERE user_id = $1 ORDER BY ticker", user_id)
    return [r["ticker"] for r in rows]