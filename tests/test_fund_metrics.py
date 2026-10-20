"""Derived metrics and company analytics: every ratio against a hand calculation, the rules for denominators that cannot divide, the DuPont split, the F-score and the anomaly score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import analytics, metrics, quality
from src.fundamentals.pit import PeriodTables
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement

Y19 = dict(revenue=1000.0, cost_of_revenue=600.0, gross_profit=400.0, selling_general_admin=150.0, research_and_development=50.0, operating_income=200.0, interest_expense=20.0,
           income_before_tax=180.0, income_tax=45.0, net_income=135.0, depreciation_amortization=40.0, operating_cash_flow=160.0, capital_expenditure=60.0, total_assets=2000.0,
           current_assets=800.0, current_liabilities=400.0, inventory=100.0, accounts_receivable=150.0, accounts_payable=80.0, ppe=900.0, total_debt=500.0, long_term_debt=400.0,
           short_term_debt=100.0, cash_and_equivalents=200.0, shareholders_equity=1100.0, total_liabilities=900.0, shares_outstanding=100.0, stock_based_compensation=10.0,
           dividends_paid=30.0, eps_diluted=1.35, ebitda=240.0, free_cash_flow=100.0)
Y20 = dict(revenue=1100.0, cost_of_revenue=630.0, gross_profit=470.0, selling_general_admin=160.0, research_and_development=55.0, operating_income=255.0, interest_expense=22.0,
           income_before_tax=233.0, income_tax=58.25, net_income=174.75, depreciation_amortization=44.0, operating_cash_flow=190.0, capital_expenditure=70.0, total_assets=2200.0,
           current_assets=900.0, current_liabilities=420.0, inventory=105.0, accounts_receivable=160.0, accounts_payable=85.0, ppe=950.0, total_debt=480.0, long_term_debt=400.0,
           short_term_debt=80.0, cash_and_equivalents=260.0, shareholders_equity=1300.0, total_liabilities=900.0, shares_outstanding=98.0, stock_based_compensation=11.0,
           dividends_paid=35.0, eps_diluted=1.78, ebitda=299.0, free_cash_flow=120.0)


def _company(tmp_path, years: dict[int, dict], name="AAA", store=None):
    store = store or FundamentalsStore(tmp_path / "f.db")
    sid = store.upsert_security(CompanyProfile("simulated", name, name, name, exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement("FY", f"{y}-12-31", dict(v), fiscal_year=y, filing_date=f"{y + 1}-02-20") for y, v in years.items()])
    return store, sid


def _row(store, sid, year, basis="fy"):
    tables = PeriodTables(store, [sid])
    values = metrics.compute(tables, basis)
    table = tables.table(basis)
    return values[table["period_end"].dt.year == year].iloc[0], tables


def test_every_ratio_matches_a_hand_calculation(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    r, _ = _row(store, sid, 2020)
    tax = 58.25 / 233.0
    nopat = 255.0 * (1 - tax)
    invested = 480.0 + 1300.0 - 260.0
    expected = {
        "gross_margin": 470 / 1100, "operating_margin": 255 / 1100, "net_margin": 174.75 / 1100, "fcf_margin": 120 / 1100, "roe": 174.75 / 1300, "roa": 174.75 / 2200,
        "roe_avg": 174.75 / ((1300 + 1100) / 2), "roa_avg": 174.75 / ((2200 + 2000) / 2), "roic": nopat / invested, "gross_profitability": 470 / 2200,
        "operating_profitability": (1100 - 630 - 160 - 22) / 1300, "cash_return_on_assets": 190 / 2200,
        "tax_burden": 174.75 / 233, "interest_burden": 233 / 255, "asset_turnover": 1100 / 2200, "equity_multiplier": 2200 / 1300,
        "debt_to_equity": 480 / 1300, "debt_to_assets": 480 / 2200, "liabilities_to_assets": 900 / 2200, "equity_ratio": 1300 / 2200, "net_debt": 480 - 260,
        "net_debt_to_ebitda": 220 / 299, "interest_coverage": 255 / 22,
        "current_ratio": 900 / 420, "quick_ratio": (900 - 105) / 420, "cash_ratio": 260 / 420, "working_capital": 480, "working_capital_to_assets": 480 / 2200,
        "fixed_asset_turnover": 1100 / 950, "inventory_turnover": 630 / 105, "receivables_turnover": 1100 / 160, "payables_turnover": 630 / 85,
        "days_sales_outstanding": 365 / (1100 / 160), "days_inventory": 365 / (630 / 105), "days_payables": 365 / (630 / 85),
        "fcf": 120, "fcf_conversion": 120 / 174.75, "cfo_to_net_income": 190 / 174.75, "accruals_ratio": (174.75 - 190) / 2200, "sloan_accruals": (174.75 - 190) / 2100,
        "capex_to_revenue": 70 / 1100, "capex_to_depreciation": 70 / 44, "rd_intensity": 55 / 1100, "sga_intensity": 160 / 1100, "sbc_to_revenue": 11 / 1100, "dividend_payout": 35 / 174.75,
        "revenue_growth": 1100 / 1000 - 1, "gross_profit_growth": 470 / 400 - 1, "operating_income_growth": 255 / 200 - 1, "net_income_growth": 174.75 / 135 - 1, "eps_growth": 1.78 / 1.35 - 1,
        "asset_growth": 2200 / 2000 - 1, "equity_growth": 1300 / 1100 - 1, "capex_growth": 70 / 60 - 1, "share_growth": 98 / 100 - 1,
        "capex_to_assets": 70 / 2200, "net_investment_to_assets": (70 - 44) / 2200,
    }
    for name, value in expected.items():
        assert r[name] == pytest.approx(value, rel=1e-9), name
    assert r["cash_conversion_cycle"] == pytest.approx(365 / (1100 / 160) + 365 / (630 / 105) - 365 / (630 / 85))
    assert np.isnan(r["revenue_cagr_3y"])                                                      # three years back there is no row


def test_a_denominator_that_cannot_divide_gives_a_missing_value_never_a_number(tmp_path):
    bad = dict(Y20, shareholders_equity=-50.0, revenue=0.0, current_liabilities=0.0, interest_expense=0.0, net_income=-20.0, income_before_tax=-10.0, income_tax=0.0)
    store, sid = _company(tmp_path, {2019: Y19, 2020: bad})
    r, _ = _row(store, sid, 2020)
    for name in ("roe", "roe_avg", "debt_to_equity", "gross_margin", "operating_margin", "net_margin", "current_ratio", "quick_ratio", "interest_coverage", "fcf_conversion",
                 "dividend_payout", "equity_multiplier"):
        assert np.isnan(r[name]), name
    assert r["roa"] == pytest.approx(-20.0 / 2200)                                              # a loss over positive assets is a real (negative) return
    assert r["revenue_growth"] == pytest.approx(-1.0) and r["net_income_growth"] == pytest.approx(-20.0 / 135.0 - 1.0)        # a fall from a positive base is a growth rate; from a loss it is not (next test)
    assert np.isnan(r["tax_burden"])                                                            # net income over a pre-tax loss is not a burden


def test_growth_from_a_loss_and_a_missing_input_are_missing_not_zero(tmp_path):
    loss = dict(Y19, net_income=-5.0)
    gap = {k: v for k, v in Y20.items() if k not in ("research_and_development", "selling_general_admin")}
    store, sid = _company(tmp_path, {2019: loss, 2020: gap})
    r, _ = _row(store, sid, 2020)
    assert np.isnan(r["net_income_growth"]) and np.isnan(r["rd_intensity"]) and np.isnan(r["sga_intensity"]) and np.isnan(r["operating_profitability"])
    assert r["revenue_growth"] == pytest.approx(0.1)


def test_averages_need_both_ends_and_a_missing_year_gives_a_missing_growth_rate(tmp_path):
    store, sid = _company(tmp_path, {2018: Y19, 2020: Y20})                                       # no 2019: the row a year before 2020 does not exist
    first, _ = _row(store, sid, 2018)
    second, _ = _row(store, sid, 2020)
    assert np.isnan(first["roe_avg"]) and not np.isnan(first["roe"])                              # the first year has no opening balance
    assert np.isnan(second["revenue_growth"]) and np.isnan(second["roe_avg"])                     # found by date, not by row position: two years apart is not a year


def test_the_year_before_is_found_for_a_fiscal_year_that_ends_a_few_weeks_off(tmp_path):
    store = FundamentalsStore(tmp_path / "f.db")
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement("FY", "2019-09-28", dict(Y19), filing_date="2019-11-20"), PeriodStatement("FY", "2020-09-26", dict(Y20), filing_date="2020-11-20")])
    r, _ = _row(store, sid, 2020)
    assert r["revenue_growth"] == pytest.approx(0.1)                                              # a 52-week year ends on a different date each year


# ------------------------------------------------------------------------------------------------------------------------ DuPont
def test_the_dupont_steps_multiply_to_the_roe_and_the_contributions_add_up_exactly(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    tables = PeriodTables(store, [sid])
    d = analytics.dupont(tables, "fy").iloc[-1]
    assert d["roe"] == pytest.approx(d["roe_from_steps"]) and d["roe"] == pytest.approx(174.75 / 1300)
    assert d["d_roe"] == pytest.approx(174.75 / 1300 - 135.0 / 1100)
    assert d["c_total"] == pytest.approx(d["d_roe"], abs=1e-15) and d["method"] == "log-mean (exact)"
    assert d["c_ebit_margin"] > 0 and d["c_equity_multiplier"] < 0                                 # the margin rose and leverage fell: the split says which mattered
    steps = ("tax_burden", "interest_burden", "ebit_margin", "asset_turnover", "equity_multiplier")
    by_hand = {k: 0.0 for k in steps}
    prior = analytics.dupont(tables, "fy").iloc[0]
    now = d
    mean = (now["roe"] - prior["roe"]) / np.log(now["roe"] / prior["roe"])
    for k in steps:
        by_hand[k] = mean * np.log(now[k] / prior[k])
    assert all(d[f"c_{k}"] == pytest.approx(by_hand[k]) for k in steps)
    assert analytics.dupont(tables, "fy").iloc[0]["method"] == "no prior year"


def test_the_dupont_split_is_refused_when_a_step_is_not_positive(tmp_path):
    loss = dict(Y20, net_income=-20.0, income_before_tax=-10.0, income_tax=10.0)
    store, sid = _company(tmp_path, {2019: Y19, 2020: loss})
    d = analytics.dupont(PeriodTables(store, [sid]), "fy").iloc[-1]
    assert np.isnan(d["c_total"]) and d["method"] == "needs every step positive in both years"


# ------------------------------------------------------------------------------------------------------------------------ Piotroski
def test_the_f_score_counts_the_nine_signs_and_is_missing_when_one_cannot_be_judged(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    r, tables = _row(store, sid, 2020)
    signs = {name: r[name] for name, _ in analytics.PIOTROSKI}
    assert signs == {"pio_roa_positive": 1.0, "pio_cfo_positive": 1.0, "pio_roa_improved": 1.0, "pio_cash_beats_profit": 1.0, "pio_leverage_fell": 1.0, "pio_liquidity_rose": 1.0,
                     "pio_no_dilution": 1.0, "pio_gross_margin_rose": 1.0, "pio_turnover_rose": 0.0}                # asset turnover is 0.5 in both years: unchanged is not "rose"
    assert r["piotroski_f"] == 8.0
    worse = dict(Y20, net_income=-30.0, operating_cash_flow=-10.0, shares_outstanding=110.0, current_assets=700.0, gross_profit=380.0, total_debt=600.0)
    store2, sid2 = _company(tmp_path / "b", {2019: Y19, 2020: worse}, name="BBB", store=FundamentalsStore(tmp_path / "g.db"))
    r2, _ = _row(store2, sid2, 2020)
    assert r2["piotroski_f"] == 1.0 and r2["pio_roa_positive"] == 0.0 and r2["pio_no_dilution"] == 0.0 and r2["pio_cash_beats_profit"] == 1.0
    no_shares = {k: v for k, v in Y20.items() if k not in ("shares_outstanding",)}
    store3, sid3 = _company(tmp_path / "c", {2019: Y19, 2020: no_shares}, name="CCC", store=FundamentalsStore(tmp_path / "h.db"))
    r3, _ = _row(store3, sid3, 2020)
    assert np.isnan(r3["pio_no_dilution"]) and np.isnan(r3["piotroski_f"])                        # no share count: the sign, and so the score, is not judged
    assert np.isnan(_row(store, sid, 2019)[0]["piotroski_f"])                                       # the first year has no change to judge


# ------------------------------------------------------------------------------------------------------------------------ Beneish
def test_the_anomaly_score_follows_the_published_formula(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    r, tables = _row(store, sid, 2020)
    a, b = Y20, Y19
    dsri = (a["accounts_receivable"] / a["revenue"]) / (b["accounts_receivable"] / b["revenue"])
    gmi = (b["gross_profit"] / b["revenue"]) / (a["gross_profit"] / a["revenue"])
    aqi = (1 - (a["current_assets"] + a["ppe"]) / a["total_assets"]) / (1 - (b["current_assets"] + b["ppe"]) / b["total_assets"])
    sgi = a["revenue"] / b["revenue"]
    depi = (b["depreciation_amortization"] / (b["depreciation_amortization"] + b["ppe"])) / (a["depreciation_amortization"] / (a["depreciation_amortization"] + a["ppe"]))
    sgai = (a["selling_general_admin"] / a["revenue"]) / (b["selling_general_admin"] / b["revenue"])
    lvgi = ((a["current_liabilities"] + a["long_term_debt"]) / a["total_assets"]) / ((b["current_liabilities"] + b["long_term_debt"]) / b["total_assets"])
    tata = (a["net_income"] - a["operating_cash_flow"]) / a["total_assets"]
    m = -4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi
    assert r["accounting_anomaly_score"] == pytest.approx(m) and r["accounting_anomaly_flag"] == float(m > analytics.BENEISH_THRESHOLD)
    parts = analytics.anomaly_parts(tables, "fy").iloc[-1]
    assert parts["dsri"] == pytest.approx(dsri) and parts["tata"] == pytest.approx(tata)
    stretched = dict(Y20, accounts_receivable=600.0, operating_cash_flow=-100.0, net_income=300.0)         # receivables ballooning while cash flow turns negative
    store2, sid2 = _company(tmp_path / "b", {2019: Y19, 2020: stretched}, name="BBB", store=FundamentalsStore(tmp_path / "g.db"))
    r2, _ = _row(store2, sid2, 2020)
    assert r2["accounting_anomaly_flag"] == 1.0 and r2["accounting_anomaly_score"] > r["accounting_anomaly_score"]
    assert "look closer" in analytics.interpret_anomaly(r2["accounting_anomaly_score"]) and "not evidence" in analytics.interpret_anomaly(r2["accounting_anomaly_score"])
    assert "missing" in analytics.interpret_anomaly(float("nan"))


# ------------------------------------------------------------------------------------------------------------------------ trailing twelve months
def _quarterly(tmp_path, quarters: list[tuple[str, str, dict]], name="QQQ"):
    store = FundamentalsStore(tmp_path / "q.db")
    sid = store.upsert_security(CompanyProfile("simulated", name, name, name, exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement(fp, end, dict(v), filing_date=str((pd.Timestamp(end) + pd.Timedelta(days=30)).date())) for fp, end, v in quarters])
    return store, sid


def test_ttm_adds_four_quarters_of_flows_and_takes_the_balance_sheet_of_the_latest(tmp_path):
    q = [("Q1", "2019-03-31", dict(revenue=10.0, net_income=1.0, total_assets=100.0)), ("Q2", "2019-06-30", dict(revenue=11.0, net_income=1.1, total_assets=101.0)),
         ("Q3", "2019-09-30", dict(revenue=12.0, net_income=1.2, total_assets=102.0)), ("Q4", "2019-12-31", dict(revenue=13.0, net_income=1.3, total_assets=103.0)),
         ("Q1", "2020-03-31", dict(revenue=20.0, net_income=2.0, total_assets=110.0))]
    store, sid = _quarterly(tmp_path, q)
    tables = PeriodTables(store, [sid])
    ttm = tables.ttm.set_index("period_end")
    assert ttm.loc["2019-12-31", "revenue"] == 46.0 and ttm.loc["2019-12-31", "total_assets"] == 103.0
    assert ttm.loc["2020-03-31", "revenue"] == 11 + 12 + 13 + 20 and ttm.loc["2020-03-31", "net_income"] == pytest.approx(1.1 + 1.2 + 1.3 + 2.0) and ttm.loc["2020-03-31", "total_assets"] == 110.0
    assert len(ttm) == 2                                                                           # three quarters are not a year: no TTM row for them


def test_a_missing_quarter_makes_the_trailing_figure_missing_not_a_partial_sum(tmp_path):
    q = [("Q1", "2019-03-31", dict(revenue=10.0)), ("Q2", "2019-06-30", dict(revenue=11.0)), ("Q3", "2019-09-30", dict()), ("Q4", "2019-12-31", dict(revenue=13.0)),
         ("Q1", "2020-03-31", dict(revenue=20.0)), ("Q2", "2020-06-30", dict(revenue=21.0))]
    store, sid = _quarterly(tmp_path, [(a, b, dict(c, total_assets=1.0)) for a, b, c in q])
    ttm = PeriodTables(store, [sid]).ttm.set_index("period_end")
    assert len(ttm) == 3 and ttm["revenue"].isna().all()                                           # every window of four contains the gap


def test_two_half_years_make_a_year_and_a_gap_between_them_restarts_the_count(tmp_path):
    h = [("H1", "2018-06-30", dict(revenue=50.0, total_assets=1.0)), ("H2", "2018-12-31", dict(revenue=55.0, total_assets=1.0)), ("H1", "2019-06-30", dict(revenue=60.0, total_assets=1.0)),
         ("H1", "2020-06-30", dict(revenue=70.0, total_assets=1.0))]                               # the second half of 2019 never arrived
    store, sid = _quarterly(tmp_path, h)
    ttm = PeriodTables(store, [sid]).ttm.set_index("period_end")
    assert ttm.loc["2018-12-31", "revenue"] == 105.0 and ttm.loc["2019-06-30", "revenue"] == 115.0 and "2020-06-30" not in ttm.index


def test_quarters_that_do_not_add_up_to_the_year_are_not_used(tmp_path):
    rows = []
    for y, annual in ((2018, 400.0), (2019, 440.0)):
        rows.append(("FY", f"{y}-12-31", dict(revenue=annual, total_assets=1000.0)))
        for k, end in enumerate((f"{y}-03-31", f"{y}-06-30", f"{y}-09-30", f"{y}-12-31")):
            rows.append((f"Q{k + 1}", end, dict(revenue=annual * (k + 1) / 4 * 0.9, total_assets=1000.0)))        # cumulative year-to-date figures labelled as quarters
    store, sid = _quarterly(tmp_path, rows)
    tables = PeriodTables(store, [sid])
    assert any("do not add up" in n for n in tables.notes) and set(tables.ttm["fiscal_period"]) == {"TTM"} and len(tables.ttm) == 2        # the two annual rows stand in


# ------------------------------------------------------------------------------------------------------------------------ keeping them
def test_metrics_are_persisted_with_their_version_and_a_second_run_changes_nothing(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    tables = PeriodTables(store, [sid])
    first = metrics.persist(store, tables, ("fy",))
    again = metrics.persist(store, PeriodTables(store, [sid]), ("fy",))
    assert first == again and store.counts()["derived_metrics"] == first and first > 60
    with store.db.transaction() as conn:
        version, = conn.execute("SELECT DISTINCT calculation_version FROM fundamental_metrics").fetchone()
        missing = conn.execute("SELECT COUNT(*) FROM fundamental_metrics WHERE metric_value IS NULL").fetchone()[0]
        roe = conn.execute("SELECT metric_value FROM fundamental_metrics m JOIN financial_periods p ON p.id = m.financial_period_id WHERE metric_name = 'roe' AND p.period_end = '2020-12-31'").fetchone()[0]
    assert version == metrics.VERSION and missing == 0 and roe == pytest.approx(174.75 / 1300)
    names = [m["name"] for m in metrics.catalogue()]
    assert len(names) == len(set(names)) > 80 and {m["family"] for m in metrics.catalogue()} >= {"profitability", "dupont", "leverage", "liquidity", "efficiency", "cashflow", "growth", "momentum", "quality"}


def test_fundamental_momentum_is_the_change_from_a_year_ago_and_its_second_difference(tmp_path):
    third = dict(Y20, operating_income=270.0, revenue=1250.0, gross_profit=520.0)
    store, sid = _company(tmp_path, {2018: Y19, 2019: Y19, 2020: Y20, 2021: third})
    tables = PeriodTables(store, [sid])
    values = metrics.compute(tables, "fy").set_index(tables.fy["period_end"].dt.year)
    assert values.loc[2020, "d_operating_margin"] == pytest.approx(255 / 1100 - 200 / 1000)
    assert values.loc[2021, "d2_operating_margin"] == pytest.approx((270 / 1250 - 255 / 1100) - (255 / 1100 - 200 / 1000))
    assert values.loc[2021, "revenue_growth_acceleration"] == pytest.approx((1250 / 1100 - 1) - (1100 / 1000 - 1))


# ------------------------------------------------------------------------------------------------------------------------ the unit-error check
def test_a_block_of_bad_quarters_is_found_as_a_block_and_fast_growth_is_not_mistaken_for_an_error(tmp_path):
    store = FundamentalsStore(tmp_path / "u.db")
    sid = store.upsert_security(CompanyProfile("simulated", "AAA", "AAA", "A", exchange="NYSE"))
    quarters = []
    for i in range(16):
        end = (pd.Timestamp("2016-03-31") + pd.offsets.QuarterEnd(i)).strftime("%Y-%m-%d")
        value = 100.0 * 1.04 ** i * (1000.0 if 6 <= i <= 9 else 1.0)
        quarters.append(PeriodStatement(f"Q{i % 4 + 1}", end, {"revenue": value, "total_assets": 5000.0}, filing_date=str((pd.Timestamp(end) + pd.Timedelta(days=30)).date())))
    store.store_financials(sid, "simulated", quarters)
    quality.check_security(store, sid)
    flagged = store.inspect(sid, "revenue").sort_values("period_end")
    bad = flagged[flagged["quality_flag"] == "suspicious_jump"]
    assert list(bad["period_end"]) == [q.period_end for q in quarters[6:10]]
    fast = FundamentalsStore(tmp_path / "v.db")
    sid2 = fast.upsert_security(CompanyProfile("simulated", "BBB", "BBB", "B", exchange="NYSE"))
    fast.store_financials(sid2, "simulated", [PeriodStatement("FY", f"{2010 + i}-12-31", {"revenue": 10.0 * 1.7 ** i, "total_assets": 100.0}, filing_date=f"{2011 + i}-03-01") for i in range(10)])
    assert "suspicious_jump" not in quality.check_security(fast, sid2)                              # 1.7x a year for ten years is a 200-fold rise and not an error
