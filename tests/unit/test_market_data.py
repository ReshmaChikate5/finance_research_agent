"""Market-data server: Finnhub first, Alpha Vantage fallback, caching, quota, normalised fields."""

from __future__ import annotations

import httpx
import pytest

from tests.unit.helpers import run

FH_QUOTE = {
    "c": 189.5,
    "d": 1.25,
    "dp": 0.66,
    "o": 188.0,
    "h": 190.1,
    "l": 187.2,
    "pc": 188.25,
    "t": 1_700_000_000,
}
AV_QUOTE = {
    "Global Quote": {
        "02. open": "188.00",
        "03. high": "190.10",
        "04. low": "187.20",
        "05. price": "189.50",
        "06. volume": "51234567",
        "07. latest trading day": "2026-10-30",
        "08. previous close": "188.25",
        "09. change": "1.25",
        "10. change percent": "0.6640%",
    }
}
FH_METRIC = {
    "metric": {
        "peTTM": 60.5,
        "epsTTM": 2.1,
        "beta": 1.7,
        "netProfitMarginTTM": 55.0,
        "52WeekHigh": 950.0,
        "52WeekLow": 400.0,
        "roeTTM": 90.0,
    }
}
FH_PROFILE = {
    "name": "NVIDIA Corp",
    "exchange": "NASDAQ",
    "finnhubIndustry": "Semiconductors",
    "currency": "USD",
    "marketCapitalization": 2_500_000,
}  # millions
AV_OVERVIEW = {
    "Symbol": "NVDA",
    "Name": "NVIDIA Corp",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Industry": "Semiconductors",
    "MarketCapitalization": "2500000000000",
    "PERatio": "60.5",
    "EPS": "2.1",
    "Beta": "1.7",
    "ProfitMargin": "0.55",
    "DividendYield": "0.0003",
    "52WeekHigh": "950",
    "52WeekLow": "400",
    "ReturnOnEquityTTM": "0.9",
    "OperatingMarginTTM": "None",
}


def av_series(days):
    return {
        "Meta Data": {},
        "Technical Analysis: RSI": {d: {"RSI": f"{50 + i}.5"} for i, d in enumerate(days)},
    }


# ---------------------------------------------------------------------------
# Start-up, health, small helpers
# ---------------------------------------------------------------------------
def test_refuses_to_start_with_no_api_keys(make_market):
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        make_market(finnhub=False, alpha=False)


def test_health_reports_providers_but_never_keys(make_market):
    body = run(make_market(alpha=False).health(None)).body.decode()
    assert '"finnhub":true' in body.replace(" ", "") and '"alpha_vantage":false' in body.replace(
        " ", ""
    )
    assert "test-key" not in body


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        (True, None),
        (3, 3.0),
        (2.5, 2.5),
        ("1,234.5", 1234.5),
        ("None", None),
        ("-", None),
        ("", None),
        ("N/A", None),
        ("abc", None),
        ("  7 ", 7.0),
    ],
)
def test_num_parses_provider_values(market, raw, expected):
    assert market._num(raw) == expected


def test_small_formatters(market):
    assert market._r(1.234567) == 1.2346 and market._r(None) is None
    assert (
        market._snake("Real Upper Band") == "real_upper_band"
        and market._snake("MACD_Hist") == "macd_hist"
    )
    assert market._fh_symbol("BRK-B") == "BRK.B"


@pytest.mark.parametrize("bad", ["", "a b", "AAPL;", "WAYTOOLONGTICKERSYMBOL", "-X"])
def test_invalid_tickers_never_reach_a_provider(market, router, bad):
    assert "not a valid ticker" in run(market.get_quote(bad))["error"]
    assert router.calls == []


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------
def test_quote_from_finnhub_is_normalised(market, router):
    router.add("finnhub.io/api/v1/quote", json=FH_QUOTE)
    result = run(market.get_quote(" aapl "))
    assert result["ticker"] == "AAPL" and result["price"] == 189.5 and result["source"] == "finnhub"
    assert result["change_percent"] == 0.66 and result["previous_close"] == 188.25
    assert result["as_of"].startswith("2023-11-14") and result["cached"] is False
    assert "fallback_reason" not in result


def test_finnhub_receives_the_key_in_a_header_not_the_url(market, router):
    router.add("finnhub.io/api/v1/quote", json=FH_QUOTE)
    run(market.get_quote("AAPL"))
    request = router.calls[0]
    assert request.headers["X-Finnhub-Token"] == "fh-test-key"
    assert "fh-test-key" not in str(request.url)


@pytest.mark.parametrize("symbol", ["RELIANCE.BSE", "BRK-B", "TCS.BSE", "VOD.LON"])
def test_documented_non_us_symbols_are_accepted(market, router, symbol):
    router.add("finnhub.io/api/v1/quote", json=FH_QUOTE)
    assert run(market.get_quote(symbol))["ticker"] == symbol


def test_class_shares_use_a_dot_for_finnhub(market, router):
    router.add("finnhub.io/api/v1/quote", json=FH_QUOTE)
    run(market.get_quote("BRK-B"))
    assert "symbol=BRK.B" in router.urls("quote")[0]


def test_second_quote_comes_from_the_cache(market, router):
    router.add("finnhub.io/api/v1/quote", json=FH_QUOTE)
    first, second = run(market.get_quote("AAPL")), run(market.get_quote("aapl"))
    assert first["cached"] is False and second["cached"] is True and second["price"] == 189.5
    assert router.count("finnhub.io") == 1


def test_empty_finnhub_quote_falls_back_to_alpha_vantage(market, router):
    router.add("finnhub.io/api/v1/quote", json={"c": 0, "t": 0})
    router.add("alphavantage.co", json=AV_QUOTE)
    result = run(market.get_quote("AAPL"))
    assert (
        result["source"] == "alpha_vantage"
        and result["price"] == 189.5
        and result["volume"] == 51234567
    )
    assert result["change_percent"] == 0.664
    assert "finnhub has no quote" in result["fallback_reason"]


@pytest.mark.parametrize(
    ("status", "words"), [(429, "rate limit"), (401, "rejected"), (500, "HTTP 500")]
)
def test_finnhub_errors_fall_back_with_a_reason(market, router, status, words):
    router.add("finnhub.io/api/v1/quote", status=status)
    router.add("alphavantage.co", json=AV_QUOTE)
    result = run(market.get_quote("AAPL"))
    assert result["source"] == "alpha_vantage" and words in result["fallback_reason"]


def test_finnhub_timeout_and_network_errors_fall_back(market, router):
    router.add("finnhub.io/api/v1/quote", exc=httpx.ReadTimeout("slow"), times=1)
    router.add("finnhub.io/api/v1/quote", exc=httpx.ConnectError("dns"))
    router.add("alphavantage.co", json=AV_QUOTE)
    assert "timed out" in run(market.get_quote("AAA"))["fallback_reason"]
    assert "ConnectError" in run(market.get_quote("BBB"))["fallback_reason"]


def test_both_providers_failing_explains_each_failure(market, router):
    router.add("finnhub.io/api/v1/quote", status=500)
    router.add("alphavantage.co", json={"Error Message": "Invalid API call"})
    error = run(market.get_quote("ZZZZ"))["error"]
    assert (
        "Could not get a quote for ZZZZ" in error
        and "[finnhub]" in error
        and "[alpha_vantage]" in error
    )


def test_unknown_symbol_everywhere_suggests_checking_the_ticker(market, router):
    router.add("finnhub.io/api/v1/quote", json={"c": 0, "t": 0})
    router.add("alphavantage.co", json={"Global Quote": {}})
    assert "Check the ticker" in run(market.get_quote("NOPE"))["error"]


def test_failures_are_not_cached(market, router):
    router.add("finnhub.io/api/v1/quote", status=500, times=1)
    router.add("alphavantage.co", status=500, times=1)
    router.add("finnhub.io/api/v1/quote", json=FH_QUOTE)
    assert "error" in run(market.get_quote("AAPL"))
    assert run(market.get_quote("AAPL"))["price"] == 189.5


def test_only_alpha_vantage_configured(make_market, router):
    mod = make_market(finnhub=False)
    router.add("alphavantage.co", json=AV_QUOTE)
    result = run(mod.get_quote("AAPL"))
    assert (
        result["source"] == "alpha_vantage"
        and "FINNHUB_API_KEY is not configured" in result["fallback_reason"]
    )
    assert router.count("finnhub.io") == 0


def test_only_finnhub_configured_has_no_fallback(make_market, router):
    mod = make_market(alpha=False)
    router.add("finnhub.io/api/v1/quote", json={"c": 0, "t": 0})
    error = run(mod.get_quote("AAPL"))["error"]
    assert "ALPHA_VANTAGE_API_KEY is not configured" in error


def test_alpha_vantage_daily_budget_stops_calls(make_market, router):
    mod = make_market(finnhub=False, AV_DAILY_LIMIT="1")
    router.add("alphavantage.co", json=AV_QUOTE)
    assert run(mod.get_quote("AAA"))["price"] == 189.5
    error = run(mod.get_quote("BBB"))["error"]
    assert "daily budget of 1" in error and router.count("alphavantage.co") == 1


@pytest.mark.parametrize(
    ("payload", "words"),
    [
        ({"Note": "Thank you for using Alpha Vantage! rate limit"}, "rate limit"),
        ({"Information": "This is a premium endpoint"}, "paid Alpha Vantage plan"),
    ],
)
def test_alpha_vantage_notices_are_translated(make_market, router, payload, words):
    mod = make_market(finnhub=False)
    router.add("alphavantage.co", json=payload)
    assert words in run(mod.get_quote("AAPL"))["error"]


def test_alpha_vantage_key_is_not_logged_or_returned(make_market, router, caplog):
    mod = make_market(finnhub=False)
    router.add("alphavantage.co", json=AV_QUOTE)
    caplog.set_level("DEBUG")
    result = run(mod.get_quote("AAPL"))
    assert "av-test-key" not in str(result) and "av-test-key" not in caplog.text


def test_alpha_vantage_bad_http_and_json(make_market, router):
    mod = make_market(finnhub=False)
    router.add("alphavantage.co", status=502, times=1)
    router.add("alphavantage.co", text="not json", times=1)
    assert "HTTP 502" in run(mod.get_quote("AAA"))["error"]
    assert "invalid JSON" in run(mod.get_quote("BBB"))["error"]


def test_unexpected_bugs_never_leak_details(market, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(market, "_try_providers", boom)
    error = run(market.get_quote("AAPL"))["error"]
    assert "Unexpected internal error" in error and "secret" not in error


# ---------------------------------------------------------------------------
# get_fundamentals
# ---------------------------------------------------------------------------
def test_fundamentals_from_finnhub_are_normalised(market, router):
    router.add("stock/metric", json=FH_METRIC)
    router.add("stock/profile2", json=FH_PROFILE)
    result = run(market.get_fundamentals("NVDA"))
    assert result["source"] == "finnhub" and result["company_name"] == "NVIDIA Corp"
    assert result["market_cap"] == 2_500_000_000_000  # millions -> raw dollars
    assert result["pe_ratio"] == 60.5 and result["profit_margin_pct"] == 55.0
    assert (
        "dividend_yield_pct" in result["unavailable_fields"]
        and "pe_ratio" not in result["unavailable_fields"]
    )


def test_finnhub_field_fallback_keys(market, router):
    router.add("stock/metric", json={"metric": {"peBasicExclExtraTTM": 33.3}})
    router.add("stock/profile2", json={})
    assert run(market.get_fundamentals("NVDA"))["pe_ratio"] == 33.3


def test_a_failing_profile_call_does_not_fail_fundamentals(market, router):
    router.add("stock/metric", json=FH_METRIC)
    router.add("stock/profile2", status=500)
    result = run(market.get_fundamentals("NVDA"))
    assert result["source"] == "finnhub" and "company_name" not in result


def test_fundamentals_fall_back_to_alpha_vantage_with_percent_scaling(market, router):
    router.add("stock/metric", json={"metric": {}})
    router.add("stock/profile2", json={})
    router.add("alphavantage.co", json=AV_OVERVIEW)
    result = run(market.get_fundamentals("NVDA"))
    assert result["source"] == "alpha_vantage"
    assert result["profit_margin_pct"] == 55.0 and result["roe_pct"] == 90.0  # 0.55 -> 55
    assert result["dividend_yield_pct"] == 0.03 and result["market_cap"] == 2_500_000_000_000.0
    assert "operating_margin_pct" in result["unavailable_fields"]  # "None" text is not a number
    assert "has no fundamentals" in result["fallback_reason"]


def test_fundamentals_unknown_symbol(market, router):
    router.add("stock/metric", json={"metric": {}})
    router.add("stock/profile2", json={})
    router.add("alphavantage.co", json={})
    error = run(market.get_fundamentals("NOPE"))["error"]
    assert "Could not get fundamentals" in error and "Check the ticker" in error


def test_fundamentals_are_cached(market, router):
    router.add("stock/metric", json=FH_METRIC)
    router.add("stock/profile2", json=FH_PROFILE)
    run(market.get_fundamentals("NVDA"))
    assert run(market.get_fundamentals("NVDA"))["cached"] is True
    assert router.count("stock/metric") == 1


# ---------------------------------------------------------------------------
# get_technical_indicator
# ---------------------------------------------------------------------------
DAYS = [f"2026-10-{d:02d}" for d in range(30, 20, -1)]


def test_indicator_is_requested_with_the_default_period(market, router):
    router.add("alphavantage.co", json=av_series(DAYS))
    result = run(market.get_technical_indicator("aapl", "rsi"))
    url = router.urls("alphavantage")[0]
    assert "function=RSI" in url and "time_period=14" in url and "series_type=close" in url
    assert (
        result["time_period"] == 14
        and result["indicator"] == "RSI"
        and result["source"] == "alpha_vantage"
    )
    assert result["as_of"] == "2026-10-30" and len(result["points"]) == 5
    assert result["points"][0] == {"date": "2026-10-30", "rsi": 50.5}


def test_points_are_limited_and_clamped(market, router):
    router.add("alphavantage.co", json=av_series(DAYS))
    assert len(run(market.get_technical_indicator("AAPL", "RSI", points=2))["points"]) == 2
    assert len(run(market.get_technical_indicator("AAPL", "RSI", points=0))["points"]) == 1
    assert (
        len(run(market.get_technical_indicator("AAPL", "RSI", points=500))["points"]) == 10
    )  # data has 10


def test_period_is_clamped_to_2_200(market, router):
    router.add("alphavantage.co", json=av_series(DAYS))
    run(market.get_technical_indicator("AAPL", "SMA", time_period=9999))
    run(market.get_technical_indicator("AAPL", "SMA", time_period=1))
    urls = router.urls("alphavantage")
    assert "time_period=200" in urls[0] and "time_period=2&" in urls[1] + "&"


def test_macd_ignores_the_period(market, router):
    router.add(
        "alphavantage.co",
        json={
            "Technical Analysis: MACD": {
                "2026-10-30": {"MACD": "1.0", "MACD_Signal": "0.5", "MACD_Hist": "0.5"}
            }
        },
    )
    result = run(market.get_technical_indicator("AAPL", "MACD", time_period=30))
    assert "time_period" not in router.urls("alphavantage")[0]
    assert result["time_period"] is None
    assert result["points"][0] == {
        "date": "2026-10-30",
        "macd": 1.0,
        "macd_signal": 0.5,
        "macd_hist": 0.5,
    }


def test_atr_has_no_series_type(market, router):
    router.add("alphavantage.co", json={"Technical Analysis: ATR": {"2026-10-30": {"ATR": "2.0"}}})
    run(market.get_technical_indicator("AAPL", "ATR"))
    assert "series_type" not in router.urls("alphavantage")[0]


def test_unsupported_indicator_is_rejected_without_a_call(market, router):
    error = run(market.get_technical_indicator("AAPL", "VIBES"))["error"]
    assert "Unsupported indicator" in error and "RSI" in error and router.calls == []


def test_indicator_results_are_cached_per_period(market, router):
    router.add("alphavantage.co", json=av_series(DAYS))
    run(market.get_technical_indicator("AAPL", "RSI"))
    assert run(market.get_technical_indicator("AAPL", "RSI", points=3))["cached"] is True
    run(market.get_technical_indicator("AAPL", "RSI", time_period=21))
    assert router.count("alphavantage") == 2


def test_indicator_needs_the_alpha_vantage_key(make_market):
    error = run(make_market(alpha=False).get_technical_indicator("AAPL", "RSI"))["error"]
    assert "ALPHA_VANTAGE_API_KEY is not configured" in error


def test_empty_indicator_data_is_reported(market, router):
    router.add("alphavantage.co", json={"Meta Data": {}})
    assert "no indicator data" in run(market.get_technical_indicator("AAPL", "RSI"))["error"]
