"""Data sources for ingestion: SEC EDGAR filings and (optionally) Finnhub company news.

This is a batch script, so it uses plain synchronous httpx and talks to SEC directly rather
than through the MCP server (no running containers needed). It follows the same SEC rules as
the EDGAR MCP server: a descriptive User-Agent with a contact email, and at most ~6 requests
per second. Downloaded filings are cached on disk (.cache/edgar/), so re-runs are free.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import REPO_ROOT

logger = logging.getLogger("ingestion.sources")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "edgar"


class SourceError(Exception):
    """A readable problem with a data source (bad config, network, not found)."""


class IngestSettings(BaseSettings):
    """Reads the same repo-root .env as everything else."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")
    sec_user_agent: str = ""
    finnhub_api_key: str = ""


@dataclass(frozen=True)
class Filing:
    ticker: str
    company_name: str
    cik: int
    form_type: str
    accession_number: str
    filing_date: str
    report_date: str | None
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVES_URL.format(
            cik=self.cik,
            acc_nodash=self.accession_number.replace("-", ""),
            doc=self.primary_document,
        )


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path = DEFAULT_CACHE_DIR, min_interval: float = 0.17):
        if "@" not in user_agent:
            raise SourceError(
                'SEC_USER_AGENT must be set in .env as "Your Name your@email.com". '
                "SEC requires a contact email in the User-Agent header."
            )
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)
        self._interval = min_interval
        self._last = 0.0
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )

    # ---- HTTP -------------------------------------------------------------
    def _get(self, url: str) -> httpx.Response:
        for attempt in range(3):
            time.sleep(max(0.0, self._last + self._interval - time.monotonic()))
            try:
                resp = self._http.get(url)
            except httpx.HTTPError as exc:
                self._last = time.monotonic()
                if attempt == 2:
                    raise SourceError(f"Network error contacting SEC ({type(exc).__name__}).") from exc
                time.sleep(1 + attempt)
                continue
            self._last = time.monotonic()
            if resp.status_code in (429, 503) and attempt < 2:
                time.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403:
                raise SourceError(
                    "SEC returned 403 Forbidden: check SEC_USER_AGENT (name + real email) "
                    "and wait a minute if you were rate limited."
                )
            raise SourceError(f"SEC returned HTTP {resp.status_code} for {url}")
        raise SourceError("SEC request failed after retries.")

    def _json_cached(self, url: str, name: str, ttl_seconds: float) -> dict:
        path = self._cache / name
        if path.exists() and time.time() - path.stat().st_mtime < ttl_seconds:
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._get(url).json()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    # ---- lookups ----------------------------------------------------------
    def resolve(self, ticker: str) -> tuple[int, str]:
        data = self._json_cached(TICKERS_URL, "company_tickers.json", 24 * 3600)
        wanted = ticker.upper().replace(".", "-")
        for row in data.values():
            if row["ticker"].upper() == wanted:
                return int(row["cik_str"]), row["title"]
        raise SourceError(f"No SEC-registered company found for ticker '{ticker}'.")

    def list_filings(self, ticker: str, form_type: str, limit: int) -> list[Filing]:
        cik, name = self.resolve(ticker)
        sub = self._json_cached(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}.json", 3600)
        recent = sub.get("filings", {}).get("recent", {})
        cols = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
        rows = [dict(zip(cols, vals)) for vals in zip(*(recent.get(c, []) for c in cols))]
        wanted = form_type.upper()
        return [
            Filing(
                ticker=ticker.upper(), company_name=name, cik=cik, form_type=r["form"],
                accession_number=r["accessionNumber"], filing_date=r["filingDate"],
                report_date=r["reportDate"] or None, primary_document=r["primaryDocument"],
            )
            for r in rows
            if str(r["form"]).upper() == wanted and r["primaryDocument"]
        ][:limit]

    def download_html(self, filing: Filing, refresh: bool = False) -> str:
        if not filing.primary_document.lower().endswith((".htm", ".html", ".txt")):
            raise SourceError(
                f"{filing.accession_number}: primary document '{filing.primary_document}' "
                "is not HTML or text."
            )
        path = self._cache / filing.ticker / f"{filing.accession_number}.html"
        if path.exists() and not refresh:
            return path.read_text(encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self._get(filing.url).text
        path.write_text(text, encoding="utf-8")
        return text

    def close(self) -> None:
        self._http.close()


# ---------------------------------------------------------------------------
# News (optional, Finnhub free plan)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewsItem:
    id: str
    ticker: str
    headline: str
    summary: str
    url: str
    source: str
    published: str  # YYYY-MM-DD


def fetch_news(ticker: str, days: int, api_key: str, limit: int = 30) -> list[NewsItem]:
    if not api_key:
        raise SourceError("FINNHUB_API_KEY is not set in .env, so news cannot be fetched.")
    today = date.today()
    params = {
        "symbol": ticker.upper().replace("-", "."),
        "from": (today - timedelta(days=max(1, days))).isoformat(),
        "to": today.isoformat(),
    }
    try:
        resp = httpx.get(FINNHUB_NEWS_URL, params=params, headers={"X-Finnhub-Token": api_key}, timeout=15.0)
    except httpx.HTTPError as exc:
        raise SourceError(f"Network error contacting Finnhub ({type(exc).__name__}).") from exc
    if resp.status_code == 429:
        raise SourceError("Finnhub rate limit reached; wait a minute and retry.")
    if resp.status_code in (401, 403):
        raise SourceError("Finnhub rejected the API key (or this endpoint needs a paid plan).")
    if resp.status_code != 200:
        raise SourceError(f"Finnhub returned HTTP {resp.status_code}.")

    items: dict[str, NewsItem] = {}
    for row in sorted(resp.json(), key=lambda r: r.get("datetime", 0), reverse=True):
        news_id = str(row.get("id") or row.get("url") or "")
        headline = (row.get("headline") or "").strip()
        summary = (row.get("summary") or "").strip()
        if not news_id or not (headline or summary) or news_id in items:
            continue
        published = datetime.fromtimestamp(row.get("datetime", 0), tz=timezone.utc).date().isoformat()
        items[news_id] = NewsItem(
            news_id, ticker.upper(), headline, summary, row.get("url", ""),
            row.get("source", "") or "unknown", published,
        )
        if len(items) >= limit:
            break
    return list(items.values())
