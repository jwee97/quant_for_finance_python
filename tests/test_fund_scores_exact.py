"""The scores from the books on financial statements, each against a hand calculation: Altman's Z-score family, the Magic Formula's return on capital, owner earnings, sustainable and internal growth,
the Graham number, net current asset value and the five Graham tests, and net operating assets. The inputs are chosen so that every expected value is a few lines of arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.fundamentals import metrics, scores
from src.fundamentals.pit import PeriodTables
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement

BASE = dict(revenue=1800.0, cost_of_revenue=1000.0, gross_profit=800.0, selling_general_admin=400.0, operating_income=250.0, interest_expense=30.0, income_before_tax=220.0, income_tax=70.0,
            net_income=150.0, depreciation_amortization=50.0, operating_cash_flow=210.0, capital_expenditure=80.0, total_assets=2000.0, current_assets=900.0, current_liabilities=500.0,
            inventory=100.0, accounts_receivable=150.0, accounts_payable=80.0, ppe=700.0, total_debt=500.0, long_term_debt=400.0, short_term_debt=100.0, cash_and_equivalents=200.0,
            shareholders_equity=1000.0, total_liabilities=1000.0, retained_earnings=600.0, shares_outstanding=100.0, dividends_paid=45.0, eps_diluted=1.5)


def _company(tmp_path, years: dict[int, dict], name="AAA"):
    folder = tmp_path / f"db{sum(1 for _ in tmp_path.iterdir())}"                              # a test may build several companies, each in a database of its own
    folder.mkdir()
    store = FundamentalsStore(folder / "f.db")
    sid = store.upsert_security(CompanyProfile("simulated", name, name, name, exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement("FY", f"{y}-12-31", dict(v), fiscal_year=y, filing_date=f"{y + 1}-02-20") for y, v in years.items()])
    return store, sid


def _last(tmp_path, years: dict[int, dict]):
    """The derived metrics of the company's last fiscal year, and the tables they came from."""
    store, sid = _company(tmp_path, years)
    tables = PeriodTables(store, [sid])
    values = metrics.compute(tables, "fy")
    table = tables.table("fy")
    return values[table["period_end"].dt.year == max(years)].iloc[0], tables


def _span(first: int, last: int, **change) -> dict[int, dict]:
    """The same statements every year from ``first`` to ``last``, with ``change`` applied to the last one (a value, or a function of the year to set it year by year)."""
    out = {}
    for y in range(first, last + 1):
        row = dict(BASE)
        for key, value in change.items():
            row[key] = value(y) if callable(value) else value
        out[y] = row
    return out


def test_altman_z_models_are_the_published_weights_on_five_ratios(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE, revenue=1800.0)})
    x1, x2, x3, x4, x5 = 400 / 2000, 600 / 2000, 250 / 2000, 1000 / 1000, 1800 / 2000
    assert (r["working_capital_to_assets"], r["retained_earnings_to_assets"], r["ebit_to_assets"], r["equity_to_liabilities"], r["asset_turnover"]) == pytest.approx((x1, x2, x3, x4, x5), rel=1e-12)
    assert r["altman_z_double_prime"] == pytest.approx(6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4, rel=1e-12)
    assert r["altman_z_double_prime"] == pytest.approx(4.18, abs=1e-12)                           # 1.312 + 0.978 + 0.84 + 1.05
    assert r["altman_z_prime"] == pytest.approx(0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5, rel=1e-12)
    assert r["altman_z_prime"] == pytest.approx(2.104075, abs=1e-12)
    assert scores.altman_zone(r["altman_z_double_prime"], "double_prime") == "safe" and scores.altman_zone(r["altman_z_prime"], "prime") == "grey"


def test_altman_zones_use_the_cut_offs_of_each_model_and_a_missing_score_has_no_zone():
    assert [scores.altman_zone(v, "double_prime") for v in (1.09, 1.10, 1.5, 2.60, 2.61)] == ["distress", "grey", "grey", "grey", "safe"]
    assert [scores.altman_zone(v, "prime") for v in (1.22, 1.23, 2.90, 2.91)] == ["distress", "grey", "grey", "safe"]
    assert [scores.altman_zone(v, "original") for v in (1.80, 1.81, 2.99, 3.0)] == ["distress", "grey", "grey", "safe"]
    assert scores.altman_zone(None, "prime") is None and scores.altman_zone(float("nan"), "prime") is None


def test_altman_keeps_a_negative_equity_and_never_guesses_a_missing_figure(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE, shareholders_equity=-200.0, total_liabilities=2200.0, retained_earnings=-300.0)})
    x4 = -200 / 2200
    assert r["equity_to_liabilities"] == pytest.approx(x4, rel=1e-12) and r["retained_earnings_to_assets"] == pytest.approx(-0.15, rel=1e-12)
    assert r["altman_z_double_prime"] == pytest.approx(6.56 * 0.2 + 3.26 * -0.15 + 6.72 * 0.125 + 1.05 * x4, rel=1e-12)
    bare = {k: v for k, v in BASE.items() if k != "retained_earnings"}
    r2, _ = _last(tmp_path, {2021: bare})
    assert math.isnan(r2["altman_z_double_prime"]) and math.isnan(r2["altman_z_prime"]) and math.isnan(r2["retained_earnings_to_assets"])
    assert not math.isnan(r2["equity_to_liabilities"])                                         # the ratios that do not need it are still there


def test_greenblatt_return_on_capital_takes_cash_and_borrowings_out_of_working_capital(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE)})
    working = (900 - 200) - (500 - 100)                                                        # 300
    assert r["greenblatt_roc"] == pytest.approx(250 / (working + 700), rel=1e-12) and r["greenblatt_roc"] == pytest.approx(0.25, rel=1e-12)


def test_greenblatt_capital_is_floored_at_a_twentieth_of_assets_and_a_missing_debt_line_is_none(tmp_path):
    paid_ahead = dict(BASE, current_assets=300.0, current_liabilities=1200.0, short_term_debt=0.0, ppe=100.0)            # working capital (300 - 200) - 1200 = -1100, capital -1000
    r, _ = _last(tmp_path, {2021: paid_ahead})
    assert r["greenblatt_roc"] == pytest.approx(250 / (0.05 * 2000), rel=1e-12)                # floored at 100
    no_debt_line = {k: v for k, v in BASE.items() if k != "short_term_debt"}
    r2, _ = _last(tmp_path, {2021: no_debt_line})
    assert r2["greenblatt_roc"] == pytest.approx(250 / ((900 - 200) - 500 + 700), rel=1e-12)


def test_owner_earnings_is_profit_plus_the_non_cash_charge_less_capital_spending(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE)})
    assert r["owner_earnings"] == pytest.approx(150 + 50 - 80, rel=1e-12) and r["owner_earnings_margin"] == pytest.approx(120 / 1800, rel=1e-12)
    r2, _ = _last(tmp_path, {2021: {k: v for k, v in BASE.items() if k != "capital_expenditure"}})
    assert math.isnan(r2["owner_earnings"])                                                    # an unknown capital spending is not zero


def test_sustainable_and_internal_growth_compound_the_retained_return(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE)})
    b, roe, roa = 1 - 45 / 150, 150 / 1000, 150 / 2000
    assert r["retention_ratio"] == pytest.approx(0.7, rel=1e-12)
    assert r["sustainable_growth"] == pytest.approx(b * roe / (1 - b * roe), rel=1e-12) and r["sustainable_growth"] == pytest.approx(0.105 / 0.895, rel=1e-12)
    assert r["internal_growth"] == pytest.approx(b * roa / (1 - b * roa), rel=1e-12) and r["internal_growth"] == pytest.approx(0.0525 / 0.9475, rel=1e-12)
    assert r["internal_growth"] < r["sustainable_growth"]                                       # assets are paid for by borrowing as well as by equity


def test_a_statement_with_no_dividend_line_retains_everything_and_no_statement_retains_nothing_known(tmp_path):
    none_paid = {k: v for k, v in BASE.items() if k != "dividends_paid"}
    r, _ = _last(tmp_path, {2021: none_paid})
    assert r["retention_ratio"] == 1.0 and r["sustainable_growth"] == pytest.approx(0.15 / 0.85, rel=1e-12)
    no_statement = {k: v for k, v in none_paid.items() if k != "operating_cash_flow"}
    r2, _ = _last(tmp_path, {2021: no_statement})
    assert math.isnan(r2["retention_ratio"]) and math.isnan(r2["sustainable_growth"])


def test_growth_the_profit_kept_cannot_pay_for_is_missing_not_infinite(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE, net_income=1200.0, dividends_paid=0.0)})          # retention 1, return on equity 1.2: the rate has no finite value
    assert math.isnan(r["sustainable_growth"])
    loss, _ = _last(tmp_path, {2021: dict(BASE, net_income=-50.0)})
    assert math.isnan(loss["retention_ratio"]) and math.isnan(loss["sustainable_growth"]) and math.isnan(loss["internal_growth"])


def test_eps_growth_over_three_years_is_compound_and_needs_both_ends_positive(tmp_path):
    years = {y: dict(BASE, eps_diluted=e) for y, e in {2018: 1.0, 2019: 1.2, 2020: 1.5, 2021: 2.0}.items()}
    r, _ = _last(tmp_path, years)
    assert r["eps_cagr_3y"] == pytest.approx(2.0 ** (1 / 3) - 1, rel=1e-12)
    assert r["eps_avg_3y"] == pytest.approx((1.2 + 1.5 + 2.0) / 3, rel=1e-12)
    years[2021]["eps_diluted"] = -0.4
    r2, _ = _last(tmp_path, years)
    assert math.isnan(r2["eps_cagr_3y"])


def test_the_graham_number_is_the_root_of_22_5_times_average_earnings_times_book_value(tmp_path):
    years = {y: dict(BASE, eps_diluted=e) for y, e in {2019: 1.5, 2020: 2.0, 2021: 2.5}.items()}
    r, _ = _last(tmp_path, years)
    eps3, bvps = 2.0, 1000 / 100
    assert r["book_value_per_share"] == pytest.approx(10.0, rel=1e-12)
    assert r["graham_number"] == pytest.approx(math.sqrt(22.5 * eps3 * bvps), rel=1e-12) and r["graham_number"] == pytest.approx(math.sqrt(450.0), rel=1e-12)
    lossy = {y: dict(v, eps_diluted=-1.0 if y == 2020 else v["eps_diluted"]) for y, v in years.items()}          # the average is (1.5 - 1 + 2.5) / 3 = 1, still positive
    r2, _ = _last(tmp_path, lossy)
    assert r2["graham_number"] == pytest.approx(math.sqrt(22.5 * 1.0 * 10.0), rel=1e-12)
    lossy[2019]["eps_diluted"] = -5.0
    r3, _ = _last(tmp_path, lossy)
    assert math.isnan(r3["graham_number"])                                                     # a negative average has no root


def test_net_current_asset_value_deducts_every_liability_and_the_preferred_stock(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE, current_assets=900.0, total_liabilities=500.0, preferred_equity=100.0)})
    assert r["ncav"] == pytest.approx(900 - 500 - 100, rel=1e-12) and r["ncav_per_share"] == pytest.approx(3.0, rel=1e-12)
    r2, _ = _last(tmp_path, {2021: dict(BASE)})
    assert r2["ncav"] == pytest.approx(-100.0, rel=1e-12) and r2["ncav_per_share"] == pytest.approx(-1.0, rel=1e-12)       # no preferred line: none


def test_the_current_ratio_and_debt_tests(tmp_path):
    r, _ = _last(tmp_path, {2021: dict(BASE)})
    assert r["graham_current_ratio_ok"] == 0.0 and r["graham_debt_ok"] == 1.0                  # 900 / 500 = 1.8; long-term debt 400 against working capital 400
    strong, _ = _last(tmp_path, {2021: dict(BASE, current_assets=1000.0)})
    assert strong["graham_current_ratio_ok"] == 1.0 and strong["graham_debt_ok"] == 1.0         # exactly 2.0 passes
    heavy, _ = _last(tmp_path, {2021: dict(BASE, long_term_debt=401.0)})
    assert heavy["graham_debt_ok"] == 0.0
    split, _ = _last(tmp_path, {2021: {k: v for k, v in BASE.items() if k != "long_term_debt"}})
    assert split["graham_debt_ok"] == 1.0                                                      # total debt 500 less short-term debt 100 = 400


def test_the_earnings_and_dividend_tests_judge_the_years_held_and_need_five_of_them(tmp_path):
    ten = _span(2012, 2021)
    r, _ = _last(tmp_path, ten)
    assert r["graham_earnings_ok"] == 1.0 and r["graham_dividends_ok"] == 1.0
    one_loss = _span(2012, 2021, net_income=lambda y: -10.0 if y == 2016 else 150.0)
    r2, _ = _last(tmp_path, one_loss)
    assert r2["graham_earnings_ok"] == 0.0
    skipped = _span(2012, 2021, dividends_paid=lambda y: 0.0 if y == 2013 else 45.0)
    r3, _ = _last(tmp_path, skipped)
    assert r3["graham_dividends_ok"] == 0.0 and r3["graham_earnings_ok"] == 1.0
    short, _ = _last(tmp_path, _span(2018, 2021))
    assert math.isnan(short["graham_earnings_ok"]) and math.isnan(short["graham_dividends_ok"])                  # four years is not enough to judge
    five, _ = _last(tmp_path, _span(2017, 2021))
    assert five["graham_earnings_ok"] == 1.0 and five["graham_dividends_ok"] == 1.0
    old_loss = _span(2008, 2021, net_income=lambda y: -10.0 if y == 2010 else 150.0)
    r4, _ = _last(tmp_path, old_loss)
    assert r4["graham_earnings_ok"] == 1.0                                                     # a loss eleven years back is outside the ten years looked at


def test_the_growth_test_compares_three_year_averages_seven_years_apart(tmp_path):
    def eps(now: float, then: float):
        return lambda y: now if y >= 2019 else then if y <= 2014 else 1.2                    # the last three years against the first three of ten (2012-2014)
    r, _ = _last(tmp_path, _span(2012, 2021, eps_diluted=eps(1.4, 1.0)))
    assert r["graham_growth_ok"] == 1.0                                                        # 1.4 >= 4/3 * 1.0
    r2, _ = _last(tmp_path, _span(2012, 2021, eps_diluted=eps(1.3, 1.0)))
    assert r2["graham_growth_ok"] == 0.0
    r3, _ = _last(tmp_path, _span(2012, 2021, eps_diluted=eps(1.0, 0.0)))
    assert math.isnan(r3["graham_growth_ok"])                                                  # no growth rate from a base of zero
    r4, _ = _last(tmp_path, _span(2013, 2021, eps_diluted=eps(1.4, 1.0)))
    assert math.isnan(r4["graham_growth_ok"])                                                  # nine years do not reach the earlier window


def test_graham_counts_the_tests_it_could_judge_and_the_tests_passed(tmp_path):
    r, _ = _last(tmp_path, _span(2012, 2021, eps_diluted=lambda y: 1.4 if y >= 2019 else 1.0))
    assert r["graham_criteria_judged"] == 5.0 and r["graham_criteria_met"] == 4.0               # only the current ratio (1.8) fails
    r2, _ = _last(tmp_path, _span(2017, 2021))
    assert r2["graham_criteria_judged"] == 4.0 and r2["graham_criteria_met"] == 3.0             # no ten-year growth test on five years


def test_net_operating_assets_are_the_capital_that_funds_the_business_over_last_years_assets(tmp_path):
    prior = dict(BASE, total_assets=1800.0, total_debt=450.0, shareholders_equity=900.0, minority_interest=10.0, cash_and_equivalents=150.0)
    now = dict(BASE, minority_interest=20.0)
    r, _ = _last(tmp_path, {2020: prior, 2021: now})
    noa_now, noa_prior = 500 + 1000 + 20 - 200, 450 + 900 + 10 - 150
    assert r["net_operating_assets"] == pytest.approx(noa_now, rel=1e-12)
    assert r["noa_to_assets"] == pytest.approx(noa_now / 1800.0, rel=1e-12)
    assert r["noa_change"] == pytest.approx((noa_now - noa_prior) / 1800.0, rel=1e-12)


def test_the_new_scores_are_in_the_catalogue_with_a_text_each_and_per_share_amounts_stay_out_of_the_formula_language():
    names = {m["name"]: m for m in metrics.catalogue()}
    new = ("altman_z_double_prime", "altman_z_prime", "retained_earnings_to_assets", "ebit_to_assets", "equity_to_liabilities", "greenblatt_roc", "owner_earnings", "owner_earnings_margin",
           "retention_ratio", "sustainable_growth", "internal_growth", "eps_cagr_3y", "eps_avg_3y", "book_value_per_share", "ncav", "ncav_per_share", "graham_number", "graham_current_ratio_ok",
           "graham_debt_ok", "graham_earnings_ok", "graham_dividends_ok", "graham_growth_ok", "graham_criteria_judged", "graham_criteria_met", "net_operating_assets", "noa_to_assets", "noa_change")
    assert all(n in names and len(names[n]["text"]) > 20 for n in new)
    assert {n for n in new if names[n]["unit"] == "per_share"} == {"eps_avg_3y", "book_value_per_share", "ncav_per_share", "graham_number"}


def test_every_figure_is_a_plain_float_after_the_whole_pipeline(tmp_path):
    r, _ = _last(tmp_path, _span(2012, 2021))
    for name in ("altman_z_double_prime", "greenblatt_roc", "sustainable_growth", "graham_number", "noa_to_assets"):
        assert np.isfinite(r[name]), name


def test_retained_earnings_is_read_from_each_provider_and_a_deficit_stays_negative():
    from src.fundamentals import canonical

    sec, unmapped = canonical.normalise_values("sec_edgar", {"us-gaap:RetainedEarningsAccumulatedDeficit": -250.0})
    assert sec["retained_earnings"] == {"value": -250.0, "source_metric": "us-gaap:RetainedEarningsAccumulatedDeficit", "flag": "ok"} and unmapped == []
    ifrs, _ = canonical.normalise_values("sec_edgar", {"ifrs-full:RetainedEarnings": 75.0})
    assert ifrs["retained_earnings"]["value"] == 75.0 and ifrs["retained_earnings"]["source_metric"] == "ifrs-full:RetainedEarnings"
    assert canonical.SEC_FIELDS["us-gaap:RetainedEarningsAccumulatedDeficit"] == "instant" and canonical.SEC_FIELDS["ifrs-full:RetainedEarnings"] == "instant"      # a balance at the period end
    of, unmapped = canonical.normalise_values("openfilings", {"retained_earnings": -40.0, "accumulated_deficit": 40.0})
    assert of["retained_earnings"]["value"] == -40.0 and unmapped == ["accumulated_deficit"]                                    # its other key's sign is not documented: not read, and said to be unread
    assert "retained_earnings" in canonical.OPENFILINGS_KEYS and "accumulated_deficit" not in canonical.OPENFILINGS_KEYS
    assert canonical.METRICS["retained_earnings"].nonnegative is False and canonical.METRICS["retained_earnings"].kind == "stock"


def test_the_simulated_world_reports_retained_earnings_that_move_with_equity(tmp_path):
    from src.fundamentals.service import FundamentalsService

    svc = FundamentalsService(tmp_path)
    svc.load_simulated(n_companies=30, start_year=2012, end_year=2016, seed=3)
    tables = PeriodTables(svc.store, None)
    values = metrics.compute(tables, "fy")
    fy = tables.table("fy").sort_values(["security_id", "period_end"])
    first = fy.groupby("security_id")["shareholders_equity"].transform("first")
    expected = fy["shareholders_equity"] - 0.35 * first                                          # the capital paid in stays at 35% of the first year's equity
    ok = fy["retained_earnings"].notna() & expected.notna()
    assert ok.sum() > 100 and np.allclose(fy["retained_earnings"][ok], expected[ok], rtol=1e-9)
    assert values["altman_z_double_prime"].notna().mean() > 0.9 and values["altman_z_prime"].notna().mean() > 0.9                 # so the Altman scores exist for the demo world
