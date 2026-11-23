"""SEC EDGAR server: ticker lookup, filing search, document paging, XBRL facts, HTTP errors."""

from __future__ import annotations

import httpx
import pytest

from tests.unit.helpers import load_server, run

TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
    "2": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
}
ACC_10K = "0001045810-24-000029"
ACC_10Q = "0001045810-24-000124"
ACC_PDF = "0001045810-23-000017"


def submissions(rows=None):
    rows = rows or [
        (ACC_10K, "10-K", "2024-02-21", "2024-01-28", "nvda-20240128.htm", "Annual report"),
        (ACC_10Q, "10-Q", "2024-08-28", "2024-07-28", "nvda-20240728.htm", ""),
        (ACC_PDF, "8-K", "2023-05-01", "", "scan.pdf", ""),
    ]
    cols = (
        "accessionNumber",
        "form",
        "filingDate",
        "reportDate",
        "primaryDocument",
        "primaryDocDescription",
    )
    return {"filings": {"recent": {c: [r[i] for r in rows] for i, c in enumerate(cols)}}}


@pytest.fixture
def sec(edgar, router):
    """The server with the ticker list and NVDA's filing index available."""
    router.add("company_tickers.json", json=TICKERS)
    router.add("submissions/CIK0001045810", json=submissions())
    return edgar


# ---------------------------------------------------------------------------
# Start-up and health
# ---------------------------------------------------------------------------
def test_refuses_to_start_without_a_contact_email(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "no-email-here")
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        load_server("sec_edgar")


def test_health_route(edgar):
    response = run(edgar.health(None))
    assert response.status_code == 200 and b"mcp-secedgar" in response.body


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------
def test_ticker_is_normalised_to_upper_case_and_dash(sec):
    assert run(sec._resolve_ticker(" brk.b ")) == ("BRK-B", 1067983, "BERKSHIRE HATHAWAY INC")


@pytest.mark.parametrize("bad", ["", "N VDA", "NVDA;DROP", "WAYTOOLONGTICKER", "$$"])
def test_invalid_ticker_text_is_rejected_before_any_request(sec, router, bad):
    result = run(sec.search_company_filings(bad))
    assert "not a valid ticker" in result["error"]
    assert router.calls == []


def test_unknown_ticker_suggests_close_matches(sec):
    result = run(sec.search_company_filings("NVDX"))
    assert "No SEC-registered company" in result["error"] and "NVDA" in result["error"]


# ---------------------------------------------------------------------------
# search_company_filings
# ---------------------------------------------------------------------------
def test_search_returns_filings_newest_first_with_urls(sec):
    result = run(sec.search_company_filings("nvda", form_type="10-k"))
    assert result["ticker"] == "NVDA" and result["cik"] == 1045810 and result["count"] == 1
    filing = result["filings"][0]
    assert filing["accession_number"] == ACC_10K and filing["description"] == "Annual report"
    assert filing["url"] == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm"
    )


def test_empty_optional_fields_become_none(sec):
    result = run(sec.search_company_filings("NVDA", form_type="10-Q"))
    assert result["filings"][0]["description"] is None


def test_form_with_no_matches_lists_what_is_available(sec):
    result = run(sec.search_company_filings("NVDA", form_type="20-F"))
    assert "No '20-F' filings found" in result["error"] and "10-K (1)" in result["error"]


def test_invalid_form_type_is_rejected(sec):
    assert (
        "not a valid form type" in run(sec.search_company_filings("NVDA", form_type="@@"))["error"]
    )


@pytest.mark.parametrize("limit", [0, -5, 1000])
def test_limit_is_clamped_instead_of_failing(sec, limit):
    result = run(sec.search_company_filings("NVDA", form_type="10-K", limit=limit))
    assert result["count"] == 1


def test_repeated_searches_reuse_cached_json(sec, router):
    run(sec.search_company_filings("NVDA"))
    run(sec.search_company_filings("NVDA"))
    assert router.count("company_tickers") == 1 and router.count("submissions") == 1


# ---------------------------------------------------------------------------
# HTTP errors become readable messages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "fragment"),
    [(404, "404 Not Found"), (403, "403 Forbidden"), (500, "HTTP 500")],
)
def test_http_status_errors_are_explained(edgar, router, status, fragment):
    router.add("company_tickers.json", status=status)
    assert fragment in run(edgar.search_company_filings("NVDA"))["error"]


def test_rate_limit_is_retried_then_succeeds(edgar, router, no_sleep):
    router.add("company_tickers.json", status=429, times=2)
    router.add("company_tickers.json", json=TICKERS)
    router.add("submissions", json=submissions())
    assert run(edgar.search_company_filings("NVDA"))["count"] == 1
    assert [d for d in no_sleep if d >= 1] == [1, 2]  # backs off 1s then 2s (ignores the limiter's tiny pauses)


def test_persistent_rate_limit_gives_up_with_advice(edgar, router, no_sleep):
    router.add("company_tickers.json", status=503)
    assert "rate limiting" in run(edgar.search_company_filings("NVDA"))["error"]
    assert router.count("company_tickers") == 3


def test_timeouts_retry_then_report_a_timeout(edgar, router, no_sleep):
    router.add("company_tickers.json", exc=httpx.ReadTimeout("slow"))
    assert "timed out" in run(edgar.search_company_filings("NVDA"))["error"]
    assert router.count("company_tickers") == 3


def test_connection_failure_is_reported_without_a_traceback(edgar, router):
    router.add("company_tickers.json", exc=httpx.ConnectError("dns"))
    error = run(edgar.search_company_filings("NVDA"))["error"]
    assert "Network error" in error and "ConnectError" in error and "Traceback" not in error


def test_non_json_response_is_reported(edgar, router):
    router.add("company_tickers.json", text="<html>maintenance</html>")
    assert "not valid JSON" in run(edgar.search_company_filings("NVDA"))["error"]


def test_unexpected_bugs_never_leak_details(sec, monkeypatch):
    async def boom(raw):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(sec, "_resolve_ticker", boom)
    error = run(sec.search_company_filings("NVDA"))["error"]
    assert "Unexpected internal error" in error and "secret internal detail" not in error


# ---------------------------------------------------------------------------
# get_filing_document
# ---------------------------------------------------------------------------
def filing_html(body_chars: int = 0) -> str:
    filler = (
        "<p>" + ("Risk factor sentence. " * (body_chars // 22 + 1)) + "</p>" if body_chars else ""
    )
    return (
        "<html><head><title>x</title><style>.a{}</style></head><body>"
        "<script>track()</script><ix:header>HIDDEN XBRL</ix:header>"
        "<h1>NVIDIA  Annual&nbsp;Report</h1><p>Supply   is concentrated.</p>"
        + filler
        + "</body></html>"
    )


DOC_URL = "000104581024000029/nvda-20240128.htm"


def test_html_to_text_strips_scripts_styles_and_hidden_xbrl(edgar):
    text = edgar._html_to_text(filing_html())
    assert "track()" not in text and "HIDDEN XBRL" not in text and ".a{}" not in text
    assert "NVIDIA Annual Report" in text and "Supply is concentrated." in text


def test_reads_the_first_page_of_a_filing(sec, router):
    router.add(DOC_URL, text=filing_html())
    result = run(sec.get_filing_document(ACC_10K, "nvda"))
    assert result["form"] == "10-K" and result["offset"] == 0 and result["has_more"] is False
    assert result["next_offset"] is None and "Supply is concentrated." in result["text"]


def test_long_filings_are_paged_with_next_offset(sec, router):
    router.add(DOC_URL, text=filing_html(body_chars=3000))
    first = run(sec.get_filing_document(ACC_10K, "NVDA", max_chars=1000))
    assert first["returned_chars"] == 1000 and first["has_more"] is True
    second = run(
        sec.get_filing_document(ACC_10K, "NVDA", offset=first["next_offset"], max_chars=1000)
    )
    assert second["offset"] == 1000 and second["text"] != first["text"]
    assert first["total_chars"] == second["total_chars"]


def test_page_size_is_clamped_to_at_least_1000(sec, router):
    router.add(DOC_URL, text=filing_html(body_chars=3000))
    assert run(sec.get_filing_document(ACC_10K, "NVDA", max_chars=5))["returned_chars"] == 1000


def test_offset_past_the_end_is_an_error(sec, router):
    router.add(DOC_URL, text=filing_html())
    assert "past the end" in run(sec.get_filing_document(ACC_10K, "NVDA", offset=10**6))["error"]


def test_negative_offset_is_rejected(sec):
    assert "offset must be" in run(sec.get_filing_document(ACC_10K, "NVDA", offset=-1))["error"]


@pytest.mark.parametrize("bad", ["", "123", "0001045810-24-00002", "abc-de-fghijk", "../../etc"])
def test_malformed_accession_numbers_are_rejected_before_any_request(sec, router, bad):
    assert "not a valid accession number" in run(sec.get_filing_document(bad, "NVDA"))["error"]
    assert router.calls == []


def test_unknown_accession_number_is_explained(sec):
    result = run(sec.get_filing_document("0000000000-24-000001", "NVDA"))
    assert "was not found among NVDA's recent filings" in result["error"]


def test_documents_are_cached_between_pages(sec, router):
    router.add(DOC_URL, text=filing_html(body_chars=3000))
    run(sec.get_filing_document(ACC_10K, "NVDA", max_chars=1000))
    run(sec.get_filing_document(ACC_10K, "NVDA", offset=1000, max_chars=1000))
    assert router.count(DOC_URL) == 1


def test_plain_text_filings_are_returned_as_is(edgar, router):
    router.add("company_tickers.json", json=TICKERS)
    router.add(
        "submissions/CIK0001045810",
        json=submissions([(ACC_10K, "10-K", "2024-02-21", "", "full-submission.txt", "")]),
    )
    router.add("full-submission.txt", text="PLAIN TEXT FILING")
    assert run(edgar.get_filing_document(ACC_10K, "NVDA"))["text"] == "PLAIN TEXT FILING"


def test_non_text_documents_point_to_the_url(sec):
    result = run(sec.get_filing_document(ACC_PDF, "NVDA"))
    assert "cannot be read here" in result["error"] and "scan.pdf" in result["error"]


def test_document_cache_is_bounded(sec, router):
    sec._DOC_CACHE_MAX = 1
    sec._doc_cache["https://example.test/old"] = "old document"
    router.add(DOC_URL, text=filing_html())
    run(sec.get_filing_document(ACC_10K, "NVDA"))
    assert list(sec._doc_cache) == [
        "https://www.sec.gov/Archives/edgar/data/1045810/" + DOC_URL
    ]  # the oldest entry made room


# ---------------------------------------------------------------------------
# get_company_facts
# ---------------------------------------------------------------------------
def point(end, val, *, start=None, form="10-K", fp="FY", filed="2024-02-21", accn="0001-24-1"):
    p = {"end": end, "val": val, "form": form, "fp": fp, "filed": filed, "accn": accn}
    if start:
        p["start"] = start
    return p


def facts(us_gaap):
    return {"entityName": "NVIDIA Corp", "facts": {"us-gaap": us_gaap}}


def usd(*points):
    return {"units": {"USD": list(points)}}


@pytest.fixture
def with_facts(sec, router):
    def install(us_gaap):
        router.add("companyfacts/CIK0001045810", json=facts(us_gaap))
        return sec

    return install


def test_returns_annual_values_newest_first(with_facts):
    mod = with_facts(
        {
            "Revenues": usd(
                point("2023-01-29", 26974, start="2022-01-31"),
                point("2024-01-28", 60922, start="2023-01-30"),
            )
        }
    )
    result = run(mod.get_company_facts("NVDA", metrics=["revenue"]))
    values = result["metrics"]["revenue"]["values"]
    assert [v["period_end"] for v in values] == ["2024-01-28", "2023-01-29"]
    assert values[0]["value"] == 60922 and values[0]["unit"] == "USD"
    assert result["company_name"] == "NVIDIA Corp" and result["frequency"] == "annual (10-K)"


def test_quarterly_and_partial_year_points_are_ignored(with_facts):
    mod = with_facts(
        {
            "Revenues": usd(
                point("2024-01-28", 60922, start="2023-01-30"),
                point(
                    "2023-10-29", 18120, start="2023-07-31", form="10-Q", fp="Q3"
                ),  # quarter filing
                point("2023-10-29", 44000, start="2023-01-30"),  # 9 months, wrong length
                point("2022-01-30", 1, start="2021-01-31", fp="Q4"),  # not a full-year period
            )
        }
    )
    values = run(mod.get_company_facts("NVDA", metrics=["revenue"]))["metrics"]["revenue"]["values"]
    assert [v["period_end"] for v in values] == ["2024-01-28"]


def test_balance_sheet_items_need_no_start_date(with_facts):
    mod = with_facts({"Assets": usd(point("2024-01-28", 65728))})
    values = run(mod.get_company_facts("NVDA", metrics=["total_assets"]))["metrics"][
        "total_assets"
    ]["values"]
    assert values[0]["value"] == 65728


def test_the_latest_filing_wins_for_a_period(with_facts):
    mod = with_facts(
        {
            "Revenues": usd(
                point("2024-01-28", 1, start="2023-01-30", filed="2024-02-21"),
                point("2024-01-28", 2, start="2023-01-30", filed="2024-12-01", form="10-K/A"),
            )
        }
    )
    values = run(mod.get_company_facts("NVDA", metrics=["revenue"]))["metrics"]["revenue"]["values"]
    assert len(values) == 1 and values[0]["value"] == 2


def test_earlier_tags_take_priority_and_later_tags_fill_gaps(with_facts):
    mod = with_facts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": usd(
                point("2024-01-28", 100, start="2023-01-30")
            ),
            "Revenues": usd(
                point("2024-01-28", 999, start="2023-01-30"),
                point("2023-01-29", 50, start="2022-01-31"),
            ),
        }
    )
    data = run(mod.get_company_facts("NVDA", metrics=["revenue"]))["metrics"]["revenue"]
    by_end = {v["period_end"]: v["value"] for v in data["values"]}
    assert by_end == {"2024-01-28": 100, "2023-01-29": 50}  # 100 not 999; older year filled in
    assert len(data["tags_used"]) == 2


def test_eps_uses_the_per_share_unit(with_facts):
    mod = with_facts(
        {
            "EarningsPerShareDiluted": {
                "units": {"USD/shares": [point("2024-01-28", 11.93, start="2023-01-30")]}
            }
        }
    )
    values = run(mod.get_company_facts("NVDA", metrics=["eps_diluted"]))["metrics"]["eps_diluted"][
        "values"
    ]
    assert values[0]["unit"] == "USD/shares"


def test_periods_limit_and_clamp(with_facts):
    pts = [point(f"20{y}-12-31", y, start=f"20{y}-01-01") for y in range(10, 20)]
    mod = with_facts({"Revenues": usd(*pts)})
    assert (
        len(
            run(mod.get_company_facts("NVDA", periods=3, metrics=["revenue"]))["metrics"][
                "revenue"
            ]["values"]
        )
        == 3
    )
    assert (
        len(
            run(mod.get_company_facts("NVDA", periods=-4, metrics=["revenue"]))["metrics"][
                "revenue"
            ]["values"]
        )
        == 1
    )
    assert (
        len(
            run(mod.get_company_facts("NVDA", periods=99, metrics=["revenue"]))["metrics"][
                "revenue"
            ]["values"]
        )
        == 10
    )


def test_missing_metrics_are_listed(with_facts):
    mod = with_facts({"Revenues": usd(point("2024-01-28", 1, start="2023-01-30"))})
    result = run(mod.get_company_facts("NVDA", metrics=["revenue", "net_income"]))
    assert list(result["metrics"]) == ["revenue"] and result["missing_metrics"] == ["net_income"]


def test_unknown_metric_names_are_rejected(with_facts):
    mod = with_facts({})
    error = run(mod.get_company_facts("NVDA", metrics=["revenue", "vibes"]))["error"]
    assert "Unknown metrics: vibes" in error and "net_income" in error


def test_companies_without_us_gaap_data_get_an_ifrs_hint(sec, router):
    router.add("companyfacts/CIK0001045810", json={"entityName": "X", "facts": {"ifrs-full": {}}})
    assert "IFRS" in run(sec.get_company_facts("NVDA"))["error"]


def test_all_metrics_are_returned_by_default(with_facts):
    mod = with_facts({"Revenues": usd(point("2024-01-28", 1, start="2023-01-30"))})
    result = run(mod.get_company_facts("NVDA"))
    assert len(result["metrics"]) + len(result["missing_metrics"]) == len(mod.FACT_CONCEPTS)


def test_bad_dates_in_the_data_do_not_crash(with_facts):
    mod = with_facts(
        {
            "Revenues": usd(
                point("2024-01-28", 5, start="not-a-date"),
                {"end": None, "val": 3, "form": "10-K", "fp": "FY"},
                {"end": "2024-01-28", "val": None, "form": "10-K", "fp": "FY"},
            )
        }
    )
    assert run(mod.get_company_facts("NVDA", metrics=["revenue"]))["missing_metrics"] == ["revenue"]
