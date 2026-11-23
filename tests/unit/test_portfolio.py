"""Portfolio server (tool validation and shaping) and its read-only database layer."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.unit.helpers import run

HOLDINGS = [
    {
        "ticker": "AMD",
        "shares": 10.0,
        "total_cost": 1000.0,
        "average_cost_per_share": 100.0,
        "lot_count": 1,
    },
    {
        "ticker": "NVDA",
        "shares": 5.0,
        "total_cost": 2500.5,
        "average_cost_per_share": 500.1,
        "lot_count": 2,
    },
]
LOTS = [
    {
        "purchase_date": "2024-01-02",
        "shares": 2.0,
        "price_per_share": 400.0,
        "fees": 1.0,
        "lot_cost": 801.0,
    },
    {
        "purchase_date": "2024-06-03",
        "shares": 3.0,
        "price_per_share": 500.0,
        "fees": 0.5,
        "lot_cost": 1500.5,
    },
]


@pytest.fixture
def pf(portfolio: Any) -> Any:
    """The server with the database layer replaced by async mocks."""
    db = portfolio.db
    db.user_exists = AsyncMock(return_value=True)
    db.fetch_holdings = AsyncMock(return_value=HOLDINGS)
    db.fetch_cost_basis = AsyncMock(return_value=LOTS)
    db.holding_exists = AsyncMock(return_value=True)
    db.list_tickers = AsyncMock(return_value=["AMD", "NVDA"])
    return portfolio


def test_health_route(portfolio):
    assert b"mcp-portfolio" in run(portfolio.health(None)).body


# ---------------------------------------------------------------------------
# user_id validation: the model supplies it, so it is never trusted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, -1, True, False, "1", 1.5, None, 2**63, [1]])
def test_bad_user_ids_are_rejected_before_any_database_call(pf, bad):
    for result in (run(pf.get_holdings(bad)), run(pf.get_cost_basis(bad, "NVDA"))):
        assert "user_id must be a positive whole number" in result["error"]
    pf.db.user_exists.assert_not_awaited()


def test_largest_valid_user_id_is_accepted(pf):
    assert run(pf.get_holdings(2**63 - 1))["user_id"] == 2**63 - 1


# ---------------------------------------------------------------------------
# get_holdings
# ---------------------------------------------------------------------------
def test_holdings_are_totalled(pf):
    result = run(pf.get_holdings(2))
    assert result == {"user_id": 2, "holding_count": 2, "total_cost": 3500.5, "holdings": HOLDINGS}
    pf.db.fetch_holdings.assert_awaited_once_with(2)


def test_user_with_no_positions_gets_an_empty_list(pf):
    pf.db.fetch_holdings.return_value = []
    assert run(pf.get_holdings(3)) == {
        "user_id": 3,
        "holding_count": 0,
        "total_cost": 0,
        "holdings": [],
    }


def test_unknown_user_is_explained_and_nothing_else_is_queried(pf):
    pf.db.user_exists.return_value = False
    assert run(pf.get_holdings(99))["error"] == "No user with id 99 exists."
    pf.db.fetch_holdings.assert_not_awaited()


def test_missing_costs_count_as_zero(pf):
    pf.db.fetch_holdings.return_value = [{**HOLDINGS[0], "total_cost": None}]
    assert run(pf.get_holdings(1))["total_cost"] == 0


@pytest.mark.parametrize("tool", ["get_holdings", "get_cost_basis"])
def test_database_failures_return_a_generic_message(pf, tool):
    pf.db.user_exists.side_effect = ConnectionError(
        "password authentication failed for portfolio_ro"
    )
    args = (1,) if tool == "get_holdings" else (1, "NVDA")
    error = run(getattr(pf, tool)(*args))["error"]
    assert (
        "database is unavailable" in error
        and "password" not in error
        and "portfolio_ro" not in error
    )


# ---------------------------------------------------------------------------
# get_cost_basis
# ---------------------------------------------------------------------------
def test_cost_basis_totals_and_average(pf):
    result = run(pf.get_cost_basis(2, " nvda "))
    assert result["ticker"] == "NVDA" and result["total_shares"] == 5.0
    assert result["total_cost"] == 2301.5 and result["average_cost_per_share"] == 460.3
    assert result["lots"] == LOTS
    pf.db.fetch_cost_basis.assert_awaited_once_with(2, "NVDA")


def test_zero_share_lots_give_no_average(pf):
    pf.db.fetch_cost_basis.return_value = []
    result = run(pf.get_cost_basis(2, "NVDA"))
    assert result["average_cost_per_share"] is None and result["total_cost"] == 0


def test_a_ticker_the_user_does_not_hold_lists_what_they_do_hold(pf):
    pf.db.holding_exists.return_value = False
    assert (
        run(pf.get_cost_basis(2, "TSLA"))["error"]
        == "User 2 does not hold TSLA. Tickers held: AMD, NVDA."
    )
    pf.db.list_tickers.return_value = []
    assert "Tickers held: none." in run(pf.get_cost_basis(2, "TSLA"))["error"]


def test_unknown_user_for_cost_basis(pf):
    pf.db.user_exists.return_value = False
    assert "No user with id 5 exists" in run(pf.get_cost_basis(5, "NVDA"))["error"]


@pytest.mark.parametrize(
    "evil", ["NVDA'; DROP TABLE holdings;--", "NVDA OR 1=1", "", "../etc", "N" * 16, "NV DA"]
)
def test_suspicious_tickers_never_reach_the_database(pf, evil):
    assert "not a valid ticker" in run(pf.get_cost_basis(1, evil))["error"]
    pf.db.user_exists.assert_not_awaited()
    pf.db.holding_exists.assert_not_awaited()


def test_dotted_and_dashed_tickers_are_allowed(pf):
    assert run(pf.get_cost_basis(1, "brk-b"))["ticker"] == "BRK-B"
    assert run(pf.get_cost_basis(1, "RELIANCE.BSE"))["ticker"] == "RELIANCE.BSE"


# ---------------------------------------------------------------------------
# db.py: parameters, shaping, pool settings
# ---------------------------------------------------------------------------
class Row(dict):
    """asyncpg rows act like dicts for the ways db.py reads them."""


class FakePool:
    def __init__(self, rows: list[Row] | None = None, scalar: Any = None) -> None:
        self.rows, self.scalar, self.queries = rows or [], scalar, []

    async def fetch(self, sql: str, *args: Any) -> list[Row]:
        self.queries.append((sql, args))
        return self.rows

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.queries.append((sql, args))
        return self.scalar


@pytest.fixture
def dbmod(portfolio: Any) -> Any:
    return portfolio.db


def use_pool(dbmod: Any, pool: FakePool) -> FakePool:
    async def get_pool() -> FakePool:
        return pool

    dbmod.get_pool = get_pool
    return pool


def test_holdings_are_aggregated_and_converted_to_floats(dbmod):
    pool = use_pool(
        dbmod,
        FakePool(
            [
                Row(ticker="NVDA", shares=Decimal(5), total_cost=Decimal("2500.50"), lot_count=2),
                Row(ticker="ZERO", shares=Decimal(0), total_cost=Decimal(0), lot_count=0),
            ]
        ),
    )
    result = run(dbmod.fetch_holdings(7))
    assert result[0] == {
        "ticker": "NVDA",
        "shares": 5.0,
        "total_cost": 2500.5,
        "average_cost_per_share": 500.1,
        "lot_count": 2,
    }
    assert result[1]["average_cost_per_share"] is None  # no division by zero shares
    sql, args = pool.queries[0]
    assert args == (7,) and "$1" in sql and "7" not in sql


def test_cost_basis_lots_are_shaped_and_ticker_is_a_parameter(dbmod):
    pool = use_pool(
        dbmod,
        FakePool(
            [
                Row(
                    purchase_date=dt.date(2024, 1, 2),
                    shares=Decimal(2),
                    price_per_share=Decimal(400),
                    fees=Decimal("1.5"),
                )
            ]
        ),
    )
    lots = run(dbmod.fetch_cost_basis(1, "NVDA'; DROP TABLE x;--"))
    assert lots == [
        {
            "purchase_date": "2024-01-02",
            "shares": 2.0,
            "price_per_share": 400.0,
            "fees": 1.5,
            "lot_cost": 801.5,
        }
    ]
    sql, args = pool.queries[0]
    assert (
        args == (1, "NVDA'; DROP TABLE x;--") and "DROP TABLE" not in sql
    )  # never in the SQL text


def test_existence_checks_and_ticker_list(dbmod):
    pool = use_pool(dbmod, FakePool([Row(ticker="AMD"), Row(ticker="NVDA")], scalar=True))
    assert run(dbmod.user_exists(1)) is True
    assert run(dbmod.holding_exists(1, "AMD")) is True
    assert run(dbmod.list_tickers(1)) == ["AMD", "NVDA"]
    assert all("$1" in sql for sql, _ in pool.queries)


def test_every_query_in_the_module_is_a_read(dbmod):
    source = Path(dbmod.__file__).read_text(encoding="utf-8").upper()
    for word in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "TRUNCATE ", "ALTER "):
        assert word not in source


def test_missing_database_url_has_a_helpful_error(dbmod):
    with pytest.raises(RuntimeError, match="PORTFOLIO_DATABASE_URL"):
        dbmod._dsn()


def test_pool_is_created_once_with_read_only_settings(dbmod, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DATABASE_URL", "postgresql://portfolio_ro:pw@db/portfolio")
    created: list[tuple[tuple, dict]] = []

    async def fake_create_pool(*args: Any, **kwargs: Any) -> str:
        created.append((args, kwargs))
        return "POOL"

    monkeypatch.setattr(dbmod.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(dbmod, "_pool", None)

    async def twice() -> tuple:
        return await dbmod.get_pool(), await dbmod.get_pool()

    assert run(twice()) == ("POOL", "POOL")
    assert len(created) == 1
    args, kwargs = created[0]
    assert args == ("postgresql://portfolio_ro:pw@db/portfolio",)
    assert kwargs["server_settings"] == {"default_transaction_read_only": "on"}
    assert kwargs["command_timeout"] == 10 and kwargs["max_size"] <= 5


def test_float_helper_keeps_none(dbmod):
    assert dbmod._f(None) is None and dbmod._f(Decimal("1.25")) == 1.25
