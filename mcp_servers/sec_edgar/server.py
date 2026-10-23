"""SEC EDGAR MCP server.

Exposes three tools over MCP:
    - search_company_filings: list recent filings for a ticker
    - get_filing_document:    read a filing's text, in pages
    - get_company_facts:      key annual XBRL financials

Transports (set MCP_TRANSPORT):
    stdio            (default) for MCP Inspector / local clients
    streamable-http  for docker compose / ECS, served at /mcp, health at /health

IMPORTANT: with stdio, never print() to stdout; it corrupts the protocol.
All logging goes to stderr.
"""

import asyncio
import collections
import difflib
import logging
import os
import re
import sys
import time
from datetime import date
from typing import Any

import httpx
from bs4 import BeautifulSoup
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
logger = logging.getLogger("sec_edgar")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()
if "@" not in USER_AGENT:
    raise RuntimeError(
        'SEC_USER_AGENT must be set to "Your Name your@email.com". '
        "SEC requires a contact email in the User-Agent header."
    )

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

TICKERS_TTL = 24 * 3600
SUBMISSIONS_TTL = 10 * 60
FACTS_TTL = 60 * 60

MAX_FILINGS = 20
MAX_CHARS = 50_000
DEFAULT_CHARS = 20_000

_TICKER_RE = re.compile(r"^[A-Z0-9\-]{1,10}$")
_FORM_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-/ ]{0,14}$")
_ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")

# Friendly metric name -> candidate us-gaap tags (earlier tags win on conflicts)
FACT_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}

mcp = FastMCP(
    "sec-edgar",
    instructions=(
        "Tools for SEC EDGAR filings. Typical flow: call search_company_filings to find "
        "a filing's accession_number, then get_filing_document to read it (page with "
        "offset), or get_company_facts for headline annual financials."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8000")),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Liveness probe used by docker compose and ECS."""
    return JSONResponse({"status": "ok", "service": "mcp-secedgar"})


# ---------------------------------------------------------------------------
# HTTP plumbing: shared client, rate limit (SEC allows <=10 req/s), retries, cache
# ---------------------------------------------------------------------------
class EdgarError(Exception):
    """An error whose message is safe and useful to show to the model."""


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


_limiter = _RateLimiter(per_second=8)  # stay under SEC's 10 req/s limit
_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
    return _http


async def _fetch(url: str) -> httpx.Response:
    """GET a URL with rate limiting and retries. Raises EdgarError on failure."""
    client = _client()
    for attempt in range(3):
        await _limiter.wait()
        try:
            resp = await client.get(url)
        except httpx.TimeoutException as exc:
            if attempt == 2:
                raise EdgarError("SEC EDGAR timed out. Try again in a few seconds.") from exc
            await asyncio.sleep(1 + attempt)
            continue
        except httpx.HTTPError as exc:
            raise EdgarError(
                f"Network error contacting SEC EDGAR ({type(exc).__name__}). Try again later."
            ) from exc

        if resp.status_code in (429, 503) and attempt < 2:
            await asyncio.sleep(2**attempt)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 404:
            raise EdgarError(
                "SEC EDGAR returned 404 Not Found. Check the ticker or accession number."
            )
        if resp.status_code == 403:
            raise EdgarError(
                "SEC EDGAR returned 403 Forbidden. The User-Agent may be invalid "
                "or the request rate was too high."
            )
        if resp.status_code in (429, 503):
            raise EdgarError("SEC EDGAR is rate limiting requests. Wait a few seconds and retry.")
        raise EdgarError(f"SEC EDGAR returned HTTP {resp.status_code}.")
    raise EdgarError("SEC EDGAR request failed after retries.")


_CACHE_MAX = 40
_cache: dict[str, tuple[float, Any]] = {}


async def _get_json_cached(url: str, ttl: float) -> Any:
    """Fetch JSON with a small in-memory TTL cache (be kind to SEC's servers)."""
    now = time.monotonic()
    hit = _cache.get(url)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        data = (await _fetch(url)).json()
    except ValueError as exc:
        raise EdgarError("SEC EDGAR returned a response that was not valid JSON.") from exc
    _cache[url] = (now, data)
    if len(_cache) > _CACHE_MAX:
        _cache.pop(min(_cache, key=lambda k: _cache[k][0]), None)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fail(tool: str, exc: Exception) -> dict[str, Any]:
    """Convert any exception into a model-readable error dict (no stack traces)."""
    if isinstance(exc, EdgarError):
        return {"error": str(exc)}
    logger.exception("Unexpected error in %s", tool)
    return {
        "error": f"Unexpected internal error in {tool}. "
        "Check the arguments and try again."
    }


async def _resolve_ticker(raw: str) -> tuple[str, int, str]:
    """Return (normalized_ticker, cik, company_name) or raise EdgarError."""
    ticker = raw.strip().upper().replace(".", "-")
    if not _TICKER_RE.match(ticker):
        raise EdgarError(
            f"'{raw}' is not a valid ticker. Use letters/digits only, e.g. 'NVDA' or 'BRK-B'."
        )
    data = await _get_json_cached(TICKERS_URL, TICKERS_TTL)
    by_ticker = {row["ticker"].upper(): row for row in data.values()}
    row = by_ticker.get(ticker)
    if row is None:
        close = difflib.get_close_matches(ticker, list(by_ticker), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise EdgarError(f"No SEC-registered company found for ticker '{ticker}'.{hint}")
    return ticker, int(row["cik_str"]), row["title"]


def _recent_rows(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the column-oriented 'recent' filings block into row dicts (newest first)."""
    recent = submissions.get("filings", {}).get("recent", {})
    cols = (
        "accessionNumber",
        "form",
        "filingDate",
        "reportDate",
        "primaryDocument",
        "primaryDocDescription",
    )
    return [dict(zip(cols, vals)) for vals in zip(*(recent.get(c, []) for c in cols))]


def _doc_url(cik: int, accession: str, doc: str) -> str:
    return ARCHIVES_URL.format(cik=cik, acc_nodash=accession.replace("-", ""), doc=doc)


def _html_to_text(raw: str) -> str:
    """Strip an HTML/iXBRL filing down to readable plain text."""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all("ix:header"):  # hidden XBRL metadata
        tag.decompose()
    lines = (
        re.sub(r"[ \t\r\f\v\xa0]+", " ", ln).strip() for ln in soup.get_text("\n").splitlines()
    )
    return "\n".join(ln for ln in lines if ln)


_DOC_CACHE_MAX = 6
_doc_cache: dict[str, str] = {}


def _annual_points(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep full-year facts from 10-K filings, one per period end (latest filing wins)."""
    best: dict[str, dict[str, Any]] = {}
    for p in points:
        if p.get("form") not in ("10-K", "10-K/A") or p.get("fp") != "FY":
            continue
        end, val = p.get("end"), p.get("val")
        if not end or val is None:
            continue
        start = p.get("start")
        if start:  # flow items (income/cash flow) must span roughly one year
            try:
                days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            except ValueError:
                continue
            if not 340 <= days <= 380:
                continue
        prev = best.get(end)
        if prev is None or p.get("filed", "") > prev.get("filed", ""):
            best[end] = p
    return best


def _extract_metric(us_gaap: dict[str, Any], tags: list[str], periods: int) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    used: list[str] = []
    for tag in tags:
        units = us_gaap.get(tag, {}).get("units", {})
        unit_key = next((u for u in ("USD", "USD/shares") if u in units), None)
        if unit_key is None:
            continue
        added = False
        for end, p in _annual_points(units[unit_key]).items():
            if end not in merged:  # earlier tags take priority
                merged[end] = {
                    "period_end": end,
                    "value": p["val"],
                    "unit": unit_key,
                    "form": p["form"],
                    "filed": p.get("filed"),
                    "accession_number": p.get("accn"),
                }
                added = True
        if added:
            used.append(tag)
    values = sorted(merged.values(), key=lambda v: v["period_end"], reverse=True)[:periods]
    return {"tags_used": used, "values": values}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def search_company_filings(
    ticker: str, form_type: str = "10-K", limit: int = 5
) -> dict[str, Any]:
    """Find a company's most recent SEC filings of a given form type.

    Use this first to discover accession numbers, then pass one to get_filing_document.

    Args:
        ticker: US stock ticker, e.g. "NVDA" or "BRK-B". Case-insensitive.
        form_type: SEC form type, e.g. "10-K" (annual), "10-Q" (quarterly),
            "8-K" (events), "10-K/A" (amended annual), "20-F" (foreign issuer annual).
            Defaults to "10-K".
        limit: How many filings to return, newest first. 1 to 20. Defaults to 5.

    Returns:
        On success: {"ticker", "company_name", "cik", "form_type", "count",
        "filings": [{"accession_number", "form", "filing_date", "report_date",
        "primary_document", "description", "url"}]}.
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        search_company_filings(ticker="NVDA", form_type="10-K", limit=3)
    """
    try:
        ticker, cik, name = await _resolve_ticker(ticker)
        form = form_type.strip().upper()
        if not _FORM_RE.match(form):
            raise EdgarError(f"'{form_type}' is not a valid form type. Try '10-K', '10-Q' or '8-K'.")
        limit = max(1, min(int(limit), MAX_FILINGS))

        sub = await _get_json_cached(SUBMISSIONS_URL.format(cik=cik), SUBMISSIONS_TTL)
        rows = _recent_rows(sub)
        matches = [r for r in rows if str(r["form"]).upper() == form][:limit]

        if not matches:
            counts = collections.Counter(r["form"] for r in rows).most_common(8)
            available = ", ".join(f"{f} ({n})" for f, n in counts) or "none"
            raise EdgarError(
                f"No '{form}' filings found for {ticker} ({name}) among its recent filings. "
                f"Form types available: {available}. Foreign issuers file 20-F/6-K "
                "instead of 10-K/10-Q."
            )

        return {
            "ticker": ticker,
            "company_name": name,
            "cik": cik,
            "form_type": form,
            "count": len(matches),
            "filings": [
                {
                    "accession_number": r["accessionNumber"],
                    "form": r["form"],
                    "filing_date": r["filingDate"],
                    "report_date": r["reportDate"] or None,
                    "primary_document": r["primaryDocument"],
                    "description": r["primaryDocDescription"] or None,
                    "url": _doc_url(cik, r["accessionNumber"], r["primaryDocument"]),
                }
                for r in matches
            ],
        }
    except Exception as exc:  # noqa: BLE001 - tools must never leak stack traces
        return _fail("search_company_filings", exc)


@mcp.tool()
async def get_filing_document(
    accession_number: str,
    ticker: str,
    offset: int = 0,
    max_chars: int = DEFAULT_CHARS,
) -> dict[str, Any]:
    """Read the text of an SEC filing, one page at a time.

    Filings are very large, so text is returned in pages. Check "has_more" and call
    again with offset=next_offset to continue reading.

    Args:
        accession_number: Accession number from search_company_filings, format
            "0001045810-24-000029" (10 digits, 2 digits, 6 digits).
        ticker: Ticker of the company that filed it (needed to locate the document).
        offset: Character position to start reading from. Defaults to 0.
        max_chars: Page size in characters, 1000 to 50000. Defaults to 20000.

    Returns:
        On success: {"accession_number", "ticker", "form", "filing_date", "url",
        "total_chars", "offset", "returned_chars", "has_more", "next_offset", "text"}.
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        get_filing_document(accession_number="0001045810-24-000029", ticker="NVDA")
    """
    try:
        accession = accession_number.strip()
        if not _ACCESSION_RE.match(accession):
            raise EdgarError(
                f"'{accession_number}' is not a valid accession number. "
                "Expected format 0001045810-24-000029. Get one from search_company_filings."
            )
        if offset < 0:
            raise EdgarError("offset must be 0 or greater.")
        max_chars = max(1000, min(int(max_chars), MAX_CHARS))

        ticker, cik, _ = await _resolve_ticker(ticker)
        sub = await _get_json_cached(SUBMISSIONS_URL.format(cik=cik), SUBMISSIONS_TTL)
        row = next((r for r in _recent_rows(sub) if r["accessionNumber"] == accession), None)
        if row is None:
            raise EdgarError(
                f"Accession {accession} was not found among {ticker}'s recent filings. "
                "Check the number and the ticker, or use search_company_filings. "
                "Very old filings are not supported."
            )

        doc = row["primaryDocument"]
        url = _doc_url(cik, accession, doc)
        text = _doc_cache.get(url)
        if text is None:
            lower = doc.lower()
            if lower.endswith((".htm", ".html")):
                resp = await _fetch(url)
                text = await asyncio.to_thread(_html_to_text, resp.text)
            elif lower.endswith(".txt"):
                text = (await _fetch(url)).text
            else:
                raise EdgarError(
                    f"The primary document '{doc}' is not an HTML or text file, so it "
                    f"cannot be read here. Open it directly: {url}"
                )
            _doc_cache[url] = text
            if len(_doc_cache) > _DOC_CACHE_MAX:
                _doc_cache.pop(next(iter(_doc_cache)))

        total = len(text)
        if offset >= total:
            raise EdgarError(f"offset {offset} is past the end of the document ({total} chars).")
        chunk = text[offset : offset + max_chars]
        end = offset + len(chunk)
        return {
            "accession_number": accession,
            "ticker": ticker,
            "form": row["form"],
            "filing_date": row["filingDate"],
            "url": url,
            "total_chars": total,
            "offset": offset,
            "returned_chars": len(chunk),
            "has_more": end < total,
            "next_offset": end if end < total else None,
            "text": chunk,
        }
    except Exception as exc:  # noqa: BLE001
        return _fail("get_filing_document", exc)


@mcp.tool()
async def get_company_facts(
    ticker: str, periods: int = 4, metrics: list[str] | None = None
) -> dict[str, Any]:
    """Get headline annual financials for a company from SEC XBRL data.

    Values come from 10-K filings (full fiscal years only) and are in raw units
    (USD, not thousands; EPS is USD per share). Newest period first.

    Args:
        ticker: US stock ticker, e.g. "NVDA".
        periods: Number of fiscal years to return, 1 to 10. Defaults to 4.
        metrics: Optional subset of: revenue, gross_profit, operating_income,
            net_income, eps_diluted, total_assets, total_liabilities,
            stockholders_equity, cash_and_equivalents, operating_cash_flow.
            Defaults to all of them.

    Returns:
        On success: {"ticker", "company_name", "cik", "metrics": {name:
        {"tags_used", "values": [{"period_end", "value", "unit", "form", "filed",
        "accession_number"}]}}, "missing_metrics": [...]}.
        On failure: {"error": "<explanation and suggestion>"}.

    Example:
        get_company_facts(ticker="NVDA", periods=3, metrics=["revenue", "net_income"])
    """
    try:
        ticker, cik, name = await _resolve_ticker(ticker)
        periods = max(1, min(int(periods), 10))
        wanted = metrics or list(FACT_CONCEPTS)
        unknown = [m for m in wanted if m not in FACT_CONCEPTS]
        if unknown:
            raise EdgarError(
                f"Unknown metrics: {', '.join(unknown)}. "
                f"Valid metrics: {', '.join(FACT_CONCEPTS)}."
            )

        facts = await _get_json_cached(FACTS_URL.format(cik=cik), FACTS_TTL)
        us_gaap = facts.get("facts", {}).get("us-gaap")
        if not us_gaap:
            raise EdgarError(
                f"No US-GAAP XBRL facts for {ticker}. The company may report under IFRS "
                "(foreign issuer). Try search_company_filings with form_type '20-F'."
            )

        result: dict[str, Any] = {}
        missing: list[str] = []
        for metric in wanted:
            data = _extract_metric(us_gaap, FACT_CONCEPTS[metric], periods)
            if data["values"]:
                result[metric] = data
            else:
                missing.append(metric)

        return {
            "ticker": ticker,
            "company_name": facts.get("entityName", name),
            "cik": cik,
            "frequency": "annual (10-K)",
            "metrics": result,
            "missing_metrics": missing,
        }
    except Exception as exc:  # noqa: BLE001
        return _fail("get_company_facts", exc)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport not in ("stdio", "streamable-http", "sse"):
        raise SystemExit(f"Unsupported MCP_TRANSPORT '{transport}'")
    logger.info("Starting sec-edgar MCP server (transport=%s)", transport)
    mcp.run(transport=transport)  # type: ignore[arg-type]