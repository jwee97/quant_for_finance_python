"""The valuation yardsticks: the perpetuity formula, the limits on the two inputs, each bar against a hand calculation, the comparison with the price only in one currency, and the route that serves them."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.fundamentals import valuation as V
from src.fundamentals.errors import FundamentalsError
from src.webapp.api import ApiError
from tests.test_fund_api import app, get  # noqa: F401  (the module's simulated world)


def _rows(**change) -> pd.DataFrame:
    """Five fiscal years of a company whose bars are a few lines of arithmetic."""
    data = dict(fiscal_year=[2016, 2017, 2018, 2019, 2020], reporting_currency="USD", revenue=[1000.0, 1100.0, 1200.0, 1300.0, 1400.0], operating_income=[100.0, 121.0, 120.0, 156.0, 154.0],
                income_tax=[25.0] * 5, income_before_tax=[100.0] * 5, cash_and_equivalents=[200.0] * 5, total_debt=[300.0] * 5, minority_interest=[20.0] * 5, preferred_equity=[0.0] * 5,
                shares_outstanding=[100.0] * 5, graham_number=[18.0] * 5, eps_avg_3y=[1.8] * 5, book_value_per_share=[10.0] * 5, ncav_per_share=[6.0] * 5, owner_earnings=[80.0, 90.0, 90.0, 100.0, 110.0],
                fcf=[70.0, 80.0, 90.0, 100.0, 110.0], eps_diluted=[2.0] * 5, eps_cagr_3y=[0.2] * 5)
    data.update(change)
    return pd.DataFrame(data)


def _by_key(result: dict) -> dict:
    return {r["key"]: r for r in result["rows"]}


def test_a_perpetuity_is_next_years_flow_over_the_rate_less_the_growth():
    assert V.perpetuity(100.0, 0.09, 0.02) == pytest.approx(100 * 1.02 / 0.07, rel=1e-12)
    assert V.perpetuity(100.0, 0.09, 0.0) == pytest.approx(100 / 0.09, rel=1e-12) and V.perpetuity(100.0, 0.09, -0.03) == pytest.approx(100 * 0.97 / 0.12, rel=1e-12)
    assert V.perpetuity(0.0, 0.09, 0.02) is None and V.perpetuity(-5.0, 0.09, 0.02) is None and V.perpetuity(float("nan"), 0.09, 0.02) is None
    assert V.perpetuity(100.0, 0.02, 0.02) is None and V.perpetuity(100.0, 0.02, 0.05) is None             # a growth as fast as the rate has no finite value


def test_the_two_inputs_are_limited_and_the_rate_must_stay_two_points_over_the_growth():
    V.check_inputs(0.09, 0.02)
    V.check_inputs(0.09, 0.07)                                                                              # exactly two points: allowed
    for rate, growth, why in ((0.02, 0.0, "discount rate must be between"), (0.26, 0.0, "discount rate must be between"), (0.09, -0.06, "growth must be between"), (0.09, 0.09, "growth must be between"),
                              (0.09, 0.075, "at least 2 points above")):
        with pytest.raises(FundamentalsError, match=why):
            V.check_inputs(rate, growth)


def test_every_bar_is_the_rule_worked_by_hand():
    out = V.build(_rows(), 25.0, "2021-06-30", "USD", 0.09, 0.02)
    rows = _by_key(out)
    assert out["currency"] == "USD" and out["fiscal_year"] == 2020 and out["price"] == {"value": 25.0, "date": "2021-06-30", "currency": "USD"} and out["price_note"] is None
    assert rows["graham_number"]["point"] == 18.0 and rows["book_value"]["point"] == 10.0
    assert (rows["ncav"]["low"], rows["ncav"]["high"]) == pytest.approx((4.0, 6.0), rel=1e-12)                # two-thirds of 6 up to 6
    margin = (100 / 1000 + 121 / 1100 + 120 / 1200 + 156 / 1300 + 154 / 1400) / 5
    nopat = margin * 1400 * 0.75                                                                            # a tax of 25% in every year
    assert margin == pytest.approx(0.108, rel=1e-12)
    epv = rows["epv"]
    assert epv["point"] == pytest.approx((nopat / 0.09 + 200 - 300 - 20) / 100, rel=1e-12)
    assert epv["low"] == pytest.approx((nopat / 0.10 - 120) / 100, rel=1e-12) and epv["high"] == pytest.approx((nopat / 0.08 - 120) / 100, rel=1e-12)
    oe = rows["owner_earnings"]                                                                             # the last three years: (90 + 100 + 110) / 3
    assert oe["point"] == pytest.approx(100 * 1.02 / 0.07 / 100, rel=1e-12) and oe["low"] == pytest.approx(100 * 1.02 / 0.08 / 100, rel=1e-12) and oe["high"] == pytest.approx(100 * 1.02 / 0.06 / 100, rel=1e-12)
    assert rows["free_cash_flow"]["point"] == pytest.approx(100 * 1.02 / 0.07 / 100, rel=1e-12)
    assert rows["lynch"]["point"] == pytest.approx(2.0 * 20.0, rel=1e-12)                                    # earnings per share 2.0, growth 20 per cent: a P/E of 20
    assert out["inputs"] == {"rate": 0.09, "growth": 0.02, "band": 0.01, "min_spread": 0.02}


def test_the_lynch_line_never_asks_for_more_than_forty_times_earnings():
    assert _by_key(V.build(_rows(eps_cagr_3y=[0.9] * 5), None, None, "USD"))["lynch"]["point"] == pytest.approx(2.0 * 40.0, rel=1e-12)


def test_graham_price_tests_use_the_three_year_average_and_book_value():
    tests = {t["key"]: t for t in V.build(_rows(), 25.0, "2021-06-30", "USD")["graham_tests"]}
    assert tests["pe"]["value"] == pytest.approx(25 / 1.8, rel=1e-12) and tests["pe"]["ok"] is True and tests["pe"]["limit"] == 15.0
    assert tests["pb"]["value"] == pytest.approx(2.5, rel=1e-12) and tests["pb"]["ok"] is False and tests["pb"]["limit"] == 1.5
    assert tests["product"]["value"] == pytest.approx(25 / 1.8 * 2.5, rel=1e-12) and tests["product"]["ok"] is False and tests["product"]["limit"] == 22.5
    cheap = {t["key"]: t for t in V.build(_rows(), 14.0, "2021-06-30", "USD")["graham_tests"]}
    assert cheap["pe"]["ok"] is True and cheap["pb"]["ok"] is True and cheap["product"]["ok"] is True      # 7.78 x 1.4 = 10.9


def test_the_price_is_set_against_the_bars_only_when_it_is_in_the_currency_of_the_statements():
    other = V.build(_rows(), 25.0, "2021-06-30", "GBP")
    assert other["price"] is None and other["graham_tests"] == [] and "GBP" in other["price_note"] and "USD" in other["price_note"]
    assert V.build(_rows(), None, None, "USD")["price_note"] == "no price is loaded for this company"
    assert V.build(_rows(), 25.0, "2021-06-30", None)["price"] is None
    assert math.isfinite(_by_key(other)["graham_number"]["point"])                                          # the bars are still there, in the statements' currency


def test_a_bar_that_cannot_be_made_says_why_and_is_empty():
    rows = _by_key(V.build(_rows(ncav_per_share=[-2.0] * 5, graham_number=[float("nan")] * 5, owner_earnings=[-10.0] * 5, eps_cagr_3y=[-0.1] * 5), 25.0, "2021-06-30", "USD"))
    assert rows["ncav"]["low"] is None and "do not cover the liabilities" in rows["ncav"]["missing"]
    assert rows["graham_number"]["point"] is None and "three years of earnings" in rows["graham_number"]["missing"]
    assert rows["owner_earnings"]["point"] is None and "average is positive" in rows["owner_earnings"]["missing"]
    assert rows["lynch"]["point"] is None and "earnings per share now and three years ago" in rows["lynch"]["missing"]
    shareless = _by_key(V.build(_rows(shares_outstanding=[float("nan")] * 5), 25.0, None, "USD"))
    assert shareless["epv"]["missing"] == shareless["owner_earnings"]["missing"] == "the number of shares is not in the statements"
    assert shareless["graham_number"]["point"] == 18.0                                                      # the per-share bars that were already per share do not need the count


def test_the_earnings_power_value_needs_three_years_of_margin_and_a_positive_profit():
    short = _by_key(V.build(_rows().tail(2).reset_index(drop=True), 25.0, None, "USD"))
    assert short["epv"]["point"] is None and "3 years" in short["epv"]["missing"]
    losing = _by_key(V.build(_rows(operating_income=[-100.0] * 5), 25.0, None, "USD"))
    assert losing["epv"]["point"] is None and "not positive" in losing["epv"]["missing"]
    no_balance = _by_key(V.build(_rows(cash_and_equivalents=[float("nan")] * 5, total_debt=[float("nan")] * 5), 25.0, None, "USD"))
    assert no_balance["epv"]["point"] is None and "neither debt nor cash" in no_balance["epv"]["missing"]
    no_debt_line = _by_key(V.build(_rows(total_debt=[float("nan")] * 5), 25.0, None, "USD"))["epv"]                # debt not reported counts as none
    assert no_debt_line["point"] == pytest.approx((0.108 * 1400 * 0.75 / 0.09 + 200 - 20) / 100, rel=1e-12)


def test_the_tax_rate_of_the_earnings_power_value_is_the_companys_own_when_it_is_plausible():
    odd = _rows(income_tax=[60.0] * 5)                                                                      # 60 per cent: outside 0-45, so the default of 25 per cent
    assert _by_key(V.build(odd, None, None, "USD"))["epv"]["point"] == pytest.approx((0.108 * 1400 * 0.75 / 0.09 + 200 - 300 - 20) / 100, rel=1e-12)
    low = _rows(income_tax=[10.0] * 5)
    assert _by_key(V.build(low, None, None, "USD"))["epv"]["point"] == pytest.approx((0.108 * 1400 * 0.90 / 0.09 + 200 - 300 - 20) / 100, rel=1e-12)


def test_a_company_with_no_statements_is_refused():
    with pytest.raises(FundamentalsError, match="no annual statements"):
        V.build(_rows().iloc[0:0], None, None, "USD")


def test_the_route_serves_the_bars_of_a_simulated_company_beside_its_last_close(app):  # noqa: F811
    page = get(app, "/api/fund/companies", db="demo", limit="5", q="")
    sid = str(page["companies"][0]["security_id"])
    out = get(app, "/api/fund/valuation", db="demo", id=sid)
    keys = [r["key"] for r in out["rows"]]
    assert keys == ["graham_number", "ncav", "epv", "owner_earnings", "free_cash_flow", "lynch", "book_value"]
    assert out["inputs"]["rate"] == 0.09 and out["inputs"]["growth"] == 0.02 and out["price"]["value"] > 0 and out["price"]["currency"] == out["currency"]
    assert sum(1 for r in out["rows"] if r["point"] is not None) >= 4
    for r in out["rows"]:
        if r["point"] is not None:
            assert r["low"] <= r["point"] + 1e-9 <= r["high"] + 2e-9, r["key"]                                  # a bar never has its point outside it
    steeper = get(app, "/api/fund/valuation", db="demo", id=sid, rate="0.12", growth="0.03")
    base, more = _by_key(out)["owner_earnings"], _by_key(steeper)["owner_earnings"]
    if base["point"] is not None and more["point"] is not None:
        assert more["point"] < base["point"]                                                                 # a higher rate is worth less
    assert [t["key"] for t in out["graham_tests"]] == ["pe", "pb", "product"]
    assert any("Greenwald" in r["cite"] for r in out["references"]) and any("Buffett (1986)" in r["cite"] for r in out["references"])


def test_the_route_refuses_what_it_cannot_value(app):  # noqa: F811
    page = get(app, "/api/fund/companies", db="demo", limit="1", q="")
    sid = str(page["companies"][0]["security_id"])
    with pytest.raises(ApiError, match="say which company"):
        get(app, "/api/fund/valuation", db="demo")
    with pytest.raises(ApiError, match="no company with id"):
        get(app, "/api/fund/valuation", db="demo", id="999999")
    with pytest.raises(ApiError, match="discount rate must be between"):
        get(app, "/api/fund/valuation", db="demo", id=sid, rate="0.5")
    with pytest.raises(ApiError, match="perpetual growth must be between"):
        get(app, "/api/fund/valuation", db="demo", id=sid, growth="0.2")
    with pytest.raises(ApiError, match="perpetual growth must be a number"):
        get(app, "/api/fund/valuation", db="demo", id=sid, growth="abc")
    with pytest.raises(ApiError, match="at least 2 points"):
        get(app, "/api/fund/valuation", db="demo", id=sid, rate="0.05", growth="0.04")


def test_a_company_that_reports_in_another_currency_than_its_price_is_never_set_against_it(app):  # noqa: F811
    companies = get(app, "/api/fund/companies", db="demo", limit="100", q="")["companies"]
    mismatched = 0
    for c in companies:
        out = get(app, "/api/fund/valuation", db="demo", id=str(c["security_id"]))
        if out["currency"] != c["currency"]:
            mismatched += 1
            assert out["price"] is None and out["graham_tests"] == [] and c["currency"] in out["price_note"] and out["currency"] in out["price_note"], c["ticker"]
            assert any(r["point"] is not None for r in out["rows"]), c["ticker"]                 # the bars are still made, in the statements' currency
        else:
            assert out["price"] is not None and out["price_note"] is None, c["ticker"]
    assert mismatched >= 1                                                                      # the demo world has companies that report in US dollars whatever they trade in
