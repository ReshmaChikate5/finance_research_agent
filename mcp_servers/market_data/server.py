"""Market data MCP server: Finnhub first, Alpha Vantage as fallback.

Tools:
    - get_quote:               latest price (Finnhub -> Alpha Vantage fallback)
    - get_fundamentals:        valuation/profitability ratios (Finnhub -> Alpha Vantage)
    - get_technical_indicator: SMA/EMA/RSI/... (Alpha Vantage)

Every external call goes through cache.TTLCache first. Alpha Vantage's free tier
is only 25 requests/day, so its calls are also counted by a local DailyBudget.

Transports (set MCP_TRANSPORT):
    stdio            (default) for MCP Inspector / local clients
    streamable-http  for docker compose / ECS, served at /mcp, health at /health

IMPORTANT: with stdio, never print() to stdout. All logging goes to stderr.
"""

import asyncio
import logging
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from cache import DailyBudget, TTLCache
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

load_dotenv()  # local dev: finds .env in a parent folder. Docker injects env directly.

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx logs full request URLs at INFO, and Alpha Vantage needs the key in the URL.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("market_data")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
AV_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
if not FINNHUB_KEY and not AV_KEY:
    raise RuntimeError("Set FINNHUB_API_KEY and/or ALPHA_VANTAGE_API_KEY in your environment.")
if not FINNHUB_KEY:
    logger.warning("FINNHUB_API_KEY not set: Alpha Vantage will be used for everything.")
if not AV_KEY:
    logger.warning("ALPHA_VANTAGE_API_KEY not set: no fallback and no technical indicators.")

FINNHUB_BASE = "https://finnhub.io/api/v1"
AV_BASE = "https://www.alphavantage.co/query"

QUOTE_TTL = int(os.getenv("QUOTE_TTL_SECONDS", "60"))
FUNDAMENTALS_TTL = int(os.getenv("FUNDAMENTALS_TTL_SECONDS", str(6 * 3600)))
TECHNICAL_TTL = int(os.getenv("TECHNICAL_TTL_SECONDS", str(3600)))
AV_DAILY_LIMIT = int(os.getenv("AV_DAILY_LIMIT", "25"))
MAX_POINTS = 30

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

cache = TTLCache(max_items=500)
av_budget = DailyBudget(AV_DAILY_LIMIT)

# indicator -> default time_period (None = not applicable) and whether it takes series_type
INDICATORS: dict[str, dict[str, Any]] = {
    "SMA": {"period": 50, "series_type": True},
    "EMA": {"period": 20, "series_type": True},
    "RSI": {"period": 14, "series_type": True},
    "BBANDS": {"period": 20, "series_type": True},
    "ATR": {"period": 14, "series_type": False},
    "ADX": {"period": 14, "series_type": False},
    "MACD": {"period": None, "series_type": True},
}

# Normalised field -> Finnhub /stock/metric keys (first one present wins)
FINNHUB_FIELDS: dict[str, list[str]] = {
    "pe_ratio": ["peTTM", "peBasicExclExtraTTM"],
    "eps_ttm": ["epsTTM", "epsBasicExclExtraItemsTTM"],
    "beta": ["beta"],
    "dividend_yield_pct": ["dividendYieldIndicatedAnnual", "currentDividendYieldTTM"],
    "price_to_book": ["pbQuarterly", "pbAnnual"],
    "price_to_sales_ttm": ["psTTM"],
    "profit_margin_pct": ["netProfitMarginTTM"],
    "operating_margin_pct": ["operatingMarginTTM"],
    "roe_pct": ["roeTTM"],
    "roa_pct": ["roaTTM"],
    "week_52_high": ["52WeekHigh"],
    "week_52_low": ["52WeekLow"],
}

# Normalised field -> (Alpha Vantage OVERVIEW key, multiplier). AV gives ratios as decimals,
# Finnhub as percents, so AV decimals are multiplied by 100 to match the *_pct fields.
AV_FIELDS: dict[str, tuple[str, float]] = {
    "pe_ratio": ("PERatio", 1),
    "eps_ttm": ("EPS", 1),
    "beta": ("Beta", 1),
    "dividend_yield_pct": ("DividendYield", 100),
    "price_to_book": ("PriceToBookRatio", 1),
    "price_to_sales_ttm": ("PriceToSalesRatioTTM", 1),
    "profit_margin_pct": ("ProfitMargin", 100),
    "operating_margin_pct": ("OperatingMarginTTM", 100),
    "roe_pct": ("ReturnOnEquityTTM", 100),
    "roa_pct": ("ReturnOnAssetsTTM", 100),
    "week_52_high": ("52WeekHigh", 1),
    "week_52_low": ("52WeekLow", 1),
}

mcp = FastMCP(
    "market-data",
    instructions=(
        "Market data tools. get_quote for the latest price, get_fundamentals for valuation "
        "and profitability ratios, get_technical_indicator for SMA/EMA/RSI/etc. Results are "
        "cached; check 'cached' and 'as_of'. Alpha Vantage has a tiny daily quota, so avoid "
        "repeating technical-indicator calls."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Liveness probe used by docker compose and ECS (never exposes keys)."""
    return JSONResponse(
        {
            "status": "ok",
            "service": "mcp-marketdata",
            "providers": {"finnhub": bool(FINNHUB_KEY), "alpha_vantage": bool(AV_KEY)},
        }
    )


# ---------------------------------------------------------------------------
# Errors and small helpers
# ---------------------------------------------------------------------------
class MarketDataError(Exception):
    """An error whose message is safe and useful to show to the model."""


class ProviderError(Exception):
    """A single provider failed. kind: not_found | rate_limited | auth | premium |
    unavailable | not_configured."""

    def __init__(self, provider: str, message: str, kind: str = "unavailable") -> None:
        super().__init__(message)
        self.provider = provider
        self.message = message
        self.kind = kind


def _fail(tool: str, exc: Exception) -> dict[str, Any]:
    """Convert any exception into a model-readable error dict (no stack traces)."""
    if isinstance(exc, MarketDataError):
        return {"error": str(exc)}
    logger.exception("Unexpected error in %s", tool)
    return {"error": f"Unexpected internal error in {tool}. Check the arguments and try again."}


def _clean_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise MarketDataError(
            f"'{raw}' is not a valid ticker. Use letters/digits, e.g. 'AAPL'. "
            "Non-US listings need an exchange suffix, e.g. 'RELIANCE.BSE'."
        )
    return ticker


def _num(value: Any) -> float | None:
    """Parse numbers from JSON values or Alpha Vantage strings ('None', '-', '')."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in ("", "None", "-", "N/A", "null"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _fh_symbol(ticker: str) -> str:
    return ticker.replace("-", ".")  # Finnhub writes class shares as BRK.B


class _RateLimiter:
    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._last + self._interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


_finnhub_limiter = _RateLimiter(per_second=5)  # free plan: 60/min, 30/sec hard cap
_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(15.0), follow_redirects=True)
    return _http


# ---------------------------------------------------------------------------
# Provider HTTP calls
# ---------------------------------------------------------------------------
async def _finnhub_get(path: str, params: dict[str, str]) -> Any:
    if not FINNHUB_KEY:
        raise ProviderError("finnhub", "FINNHUB_API_KEY is not configured.", "not_configured")
    await _finnhub_limiter.wait()
    try:
        resp = await _client().get(
            f"{FINNHUB_BASE}{path}", params=params, headers={"X-Finnhub-Token": FINNHUB_KEY}
        )
    except httpx.TimeoutException as exc:
        raise ProviderError("finnhub", "request timed out.") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("finnhub", f"network error ({type(exc).__name__}).") from exc

    if resp.status_code == 429:
        raise ProviderError("finnhub", "rate limit reached (free plan: 60 calls/min).", "rate_limited")
    if resp.status_code in (401, 403):
        raise ProviderError(
            "finnhub",
            f"request rejected (HTTP {resp.status_code}): invalid API key, or this "
            "endpoint needs a paid plan.",
            "auth",
        )
    if resp.status_code != 200:
        raise ProviderError("finnhub", f"returned HTTP {resp.status_code}.")
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError("finnhub", "returned invalid JSON.") from exc


async def _av_get(params: dict[str, str]) -> dict[str, Any]:
    if not AV_KEY:
        raise ProviderError("alpha_vantage", "ALPHA_VANTAGE_API_KEY is not configured.", "not_configured")
    if not av_budget.try_consume():
        raise ProviderError(
            "alpha_vantage",
            f"local daily budget of {av_budget.limit} Alpha Vantage requests is used up "
            "(resets at 00:00 UTC). Cached data may still be available.",
            "rate_limited",
        )
    logger.info("Alpha Vantage request %d/%d today", av_budget.used, av_budget.limit)
    try:
        resp = await _client().get(AV_BASE, params={**params, "apikey": AV_KEY})
    except httpx.TimeoutException as exc:
        raise ProviderError("alpha_vantage", "request timed out.") from exc
    except httpx.HTTPError as exc:
        raise ProviderError("alpha_vantage", f"network error ({type(exc).__name__}).") from exc
    if resp.status_code != 200:
        raise ProviderError("alpha_vantage", f"returned HTTP {resp.status_code}.")
    try:
        data = resp.json()
    except ValueError as exc:
        raise ProviderError("alpha_vantage", "returned invalid JSON.") from exc

    if isinstance(data, dict):
        if "Error Message" in data:
            raise ProviderError("alpha_vantage", "does not recognise this symbol or request.", "not_found")
        notice = str(data.get("Note") or data.get("Information") or "")
        if notice:
            if "premium endpoint" in notice.lower():
                raise ProviderError("alpha_vantage", "this data requires a paid Alpha Vantage plan.", "premium")
            raise ProviderError(
                "alpha_vantage",
                "rate limit reached (free plan: 25 requests/day, 5/minute). Try later.",
                "rate_limited",
            )
    return data


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
async def _quote_finnhub(ticker: str) -> dict[str, Any]:
    data = await _finnhub_get("/quote", {"symbol": _fh_symbol(ticker)})
    if not isinstance(data, dict) or (not data.get("c") and not data.get("t")):
        raise ProviderError(
            "finnhub", "has no quote for this symbol (free plan covers US stocks).", "not_found"
        )
    ts = data.get("t")
    return {
        "ticker": ticker,
        "price": _r(_num(data.get("c"))),
        "change": _r(_num(data.get("d"))),
        "change_percent": _r(_num(data.get("dp"))),
        "open": _r(_num(data.get("o"))),
        "high": _r(_num(data.get("h"))),
        "low": _r(_num(data.get("l"))),
        "previous_close": _r(_num(data.get("pc"))),
        "as_of": datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else None,
        "source": "finnhub",
    }


async def _quote_av(ticker: str) -> dict[str, Any]:
    data = await _av_get({"function": "GLOBAL_QUOTE", "symbol": ticker})
    q = data.get("Global Quote") or {}
    price = _num(q.get("05. price"))
    if price is None:
        raise ProviderError("alpha_vantage", "has no quote for this symbol.", "not_found")
    volume = _num(q.get("06. volume"))
    return {
        "ticker": ticker,
        "price": _r(price),
        "change": _r(_num(q.get("09. change"))),
        "change_percent": _r(_num(str(q.get("10. change percent", "")).rstrip("%"))),
        "open": _r(_num(q.get("02. open"))),
        "high": _r(_num(q.get("03. high"))),
        "low": _r(_num(q.get("04. low"))),
        "previous_close": _r(_num(q.get("08. previous close"))),
        "volume": int(volume) if volume is not None else None,
        "as_of": q.get("07. latest trading day"),
        "source": "alpha_vantage",
    }


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
def _finish_fundamentals(base: dict[str, Any], fields: dict[str, float | None]) -> dict[str, Any]:
    result = {k: v for k, v in base.items() if v is not None}
    result.update({k: _r(v) for k, v in fields.items() if v is not None})
    result["unavailable_fields"] = [k for k, v in fields.items() if v is None]
    return result


async def _fundamentals_finnhub(ticker: str) -> dict[str, Any]:
    symbol = _fh_symbol(ticker)
    metric_res, profile_res = await asyncio.gather(
        _finnhub_get("/stock/metric", {"symbol": symbol, "metric": "all"}),
        _finnhub_get("/stock/profile2", {"symbol": symbol}),
        return_exceptions=True,
    )
    if isinstance(metric_res, BaseException):
        raise metric_res
    metrics = metric_res.get("metric") if isinstance(metric_res, dict) else None
    if not metrics:
        raise ProviderError("finnhub", "has no fundamentals for this symbol.", "not_found")
    profile = profile_res if isinstance(profile_res, dict) else {}  # profile is optional

    cap_millions = _num(profile.get("marketCapitalization")) or _num(metrics.get("marketCapitalization"))
    fields: dict[str, float | None] = {}
    for name, keys in FINNHUB_FIELDS.items():
        fields[name] = next((v for k in keys if (v := _num(metrics.get(k))) is not None), None)
    base = {
        "ticker": ticker,
        "source": "finnhub",
        "company_name": profile.get("name"),
        "exchange": profile.get("exchange"),
        "industry": profile.get("finnhubIndustry"),
        "currency": profile.get("currency"),
        "market_cap": cap_millions * 1_000_000 if cap_millions else None,
    }
    return _finish_fundamentals(base, fields)


async def _fundamentals_av(ticker: str) -> dict[str, Any]:
    data = await _av_get({"function": "OVERVIEW", "symbol": ticker})
    if not data.get("Symbol"):
        raise ProviderError("alpha_vantage", "has no fundamentals for this symbol.", "not_found")
    fields: dict[str, float | None] = {}
    for name, (key, scale) in AV_FIELDS.items():
        value = _num(data.get(key))
        fields[name] = value * scale if value is not None else None
    base = {
        "ticker": ticker,
        "source": "alpha_vantage",
        "company_name": data.get("Name"),
        "exchange": data.get("Exchange"),
        "industry": data.get("Industry"),
        "currency": data.get("Currency"),
        "market_cap": _num(data.get("MarketCapitalization")),
    }
    return _finish_fundamentals(base, fields)


# ---------------------------------------------------------------------------
# Technical indicators (Alpha Vantage only)
# ---------------------------------------------------------------------------
async def _technical_av(ticker: str, indicator: str, period: int | None) -> dict[str, Any]:
    spec = INDICATORS[indicator]
    params = {"function": indicator, "symbol": ticker, "interval": "daily"}
    if spec["series_type"]:
        params["series_type"] = "close"
    if period is not None:
        params["time_period"] = str(period)

    data = await _av_get(params)
    key = next((k for k in data if k.startswith("Technical Analysis")), None)
    if not key or not data[key]:
        raise ProviderError("alpha_vantage", "returned no indicator data for this symbol.", "not_found")

    series = data[key]
    dates = sorted(series, reverse=True)[:MAX_POINTS]
    points = [
        {"date": d, **{_snake(name): _r(_num(val)) for name, val in series[d].items()}}
        for d in dates
    ]
    return {
        "ticker": ticker,
        "indicator": indicator,
        "time_period": period,
        "interval": "daily",
        "source": "alpha_vantage",
        "as_of": dates[0],
        "points": points,
    }


# ---------------------------------------------------------------------------
# Provider fallback orchestration
# ---------------------------------------------------------------------------
async def _try_providers(
    what: str,
    ticker: str,
    attempts: list[Callable[[str], Awaitable[dict[str, Any]]]],
) -> dict[str, Any]:
    """Try each provider in order; return the first success, noting why earlier ones failed."""
    failures: list[ProviderError] = []
    for attempt in attempts:
        try:
            result = await attempt(ticker)
        except ProviderError as exc:
            logger.info("%s for %s failed via %s (%s): %s", what, ticker, exc.provider, exc.kind, exc.message)
            failures.append(exc)
            continue
        if failures:
            result["fallback_reason"] = "; ".join(f"{f.provider} {f.message}" for f in failures)
        return result

    detail = " ".join(f"[{f.provider}] {f.message}" for f in failures)
    hint = ""
    if failures and all(f.kind == "not_found" for f in failures):
        hint = " Check the ticker (US tickers like AAPL; other exchanges need a suffix like RELIANCE.BSE)."
    raise MarketDataError(f"Could not get {what} for {ticker}. {detail}{hint}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_quote(ticker: str) -> dict[str, Any]:
    """Get the latest stock price and day statistics for a ticker.

    Tries Finnhub first and falls back to Alpha Vantage if Finnhub fails. Results are
    cached for about a minute, so repeated calls are cheap.

    Args:
        ticker: Stock ticker, e.g. "AAPL" or "NVDA". Non-US listings need an exchange
            suffix, e.g. "RELIANCE.BSE" (these usually resolve via Alpha Vantage).

    Returns:
        On success: {"ticker", "price", "change", "change_percent", "open", "high",
        "low", "previous_close", "as_of", "source", "cached"} plus "volume" when the
        source is Alpha Vantage, and "fallback_reason" if the fallback was used.
        "as_of" is the last trade time; outside market hours it is the last close.
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        get_quote(ticker="AAPL")
    """
    try:
        t = _clean_ticker(ticker)
        value, hit = await cache.get_or_fetch(
            f"quote:{t}",
            QUOTE_TTL,
            lambda: _try_providers("a quote", t, [_quote_finnhub, _quote_av]),
        )
        return {**value, "cached": hit}
    except Exception as exc:  # noqa: BLE001 - tools must never leak stack traces
        return _fail("get_quote", exc)


@mcp.tool()
async def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Get valuation, profitability and risk ratios for a company.

    Tries Finnhub first and falls back to Alpha Vantage. Values are normalised so both
    sources use the same field names and units. Cached for several hours.

    Args:
        ticker: Stock ticker, e.g. "AAPL".

    Returns:
        On success: {"ticker", "source", "company_name", "exchange", "industry",
        "currency", "market_cap" (raw currency units), "pe_ratio", "eps_ttm", "beta",
        "dividend_yield_pct", "price_to_book", "price_to_sales_ttm",
        "profit_margin_pct", "operating_margin_pct", "roe_pct", "roa_pct",
        "week_52_high", "week_52_low", "unavailable_fields", "cached"}.
        Fields ending in _pct are percents (25.3 means 25.3%). Missing fields are
        omitted and listed in "unavailable_fields".
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        get_fundamentals(ticker="NVDA")
    """
    try:
        t = _clean_ticker(ticker)
        value, hit = await cache.get_or_fetch(
            f"fundamentals:{t}",
            FUNDAMENTALS_TTL,
            lambda: _try_providers("fundamentals", t, [_fundamentals_finnhub, _fundamentals_av]),
        )
        return {**value, "cached": hit}
    except Exception as exc:  # noqa: BLE001
        return _fail("get_fundamentals", exc)


@mcp.tool()
async def get_technical_indicator(
    ticker: str,
    indicator: str,
    time_period: int | None = None,
    points: int = 5,
) -> dict[str, Any]:
    """Get a daily technical indicator series for a ticker (Alpha Vantage).

    Alpha Vantage's free plan allows only 25 requests per day, so results are cached
    for an hour and the call is refused once the local daily budget is used up.

    Args:
        ticker: Stock ticker, e.g. "AAPL".
        indicator: One of SMA, EMA, RSI, BBANDS, ATR, ADX, MACD (case-insensitive).
            Some indicators may require a paid Alpha Vantage plan; the error says so.
        time_period: Look-back window in days, 2 to 200. Defaults: SMA 50, EMA 20,
            RSI 14, BBANDS 20, ATR 14, ADX 14. Ignored for MACD (fixed 12/26/9).
        points: How many most recent daily values to return, 1 to 30. Defaults to 5.

    Returns:
        On success: {"ticker", "indicator", "time_period", "interval", "source",
        "as_of", "points": [{"date", <indicator values>}], "cached"}, newest first.
        Value names depend on the indicator, e.g. "rsi", or "macd"/"macd_signal"/
        "macd_hist", or "real_upper_band"/"real_middle_band"/"real_lower_band".
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        get_technical_indicator(ticker="AAPL", indicator="RSI", time_period=14, points=3)
    """
    try:
        t = _clean_ticker(ticker)
        ind = indicator.strip().upper()
        if ind not in INDICATORS:
            raise MarketDataError(
                f"Unsupported indicator '{indicator}'. Supported: {', '.join(INDICATORS)}."
            )
        default_period = INDICATORS[ind]["period"]
        period = None if default_period is None else max(2, min(int(time_period or default_period), 200))
        points = max(1, min(int(points), MAX_POINTS))

        value, hit = await cache.get_or_fetch(
            f"tech:{t}:{ind}:{period}",
            TECHNICAL_TTL,
            lambda: _try_providers(
                "a technical indicator", t, [lambda tk: _technical_av(tk, ind, period)]
            ),
        )
        return {**value, "points": value["points"][:points], "cached": hit}
    except Exception as exc:  # noqa: BLE001
        return _fail("get_technical_indicator", exc)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "streamable-http", "sse"):
        raise SystemExit(f"Unsupported MCP_TRANSPORT '{transport}'")
    logger.info("Starting market-data MCP server (transport=%s)", transport)
    mcp.run(transport=transport)  # type: ignore[arg-type]