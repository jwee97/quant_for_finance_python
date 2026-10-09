"""Point-in-time statements and the factor library: no figure is used before it was public, and each factor is the arithmetic its definition says (checked by hand on a table with known numbers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import factors as F
from src.equity.fundamentals import FIELDS, FactorInputs, Fundamentals, FundamentalsError, ratio

IDX = pd.bdate_range("2014-01-01", periods=1800)                        # to the end of 2020
SCALE = {"A": 1.0, "B": 2.0, "C": 0.5}
PRICE = {"A": 10.0, "B": 15.0, "C": 40.0}
YEARS = range(7)                                                         # fiscal years 2013 .. 2019, filed 60 days after the year end


def statements(m: float, y: int) -> dict:
    """The figures of a firm ``m`` times the size of A for fiscal year ``2013 + y``. Everything is a simple multiple, so the expected factors can be written out by hand."""
    sales, assets = 2000.0 * 1.1 ** y, 1500.0 * 1.05 ** y
    return {"sales": m * sales, "cogs": m * 0.5 * sales, "sga": m * 0.2 * sales, "depreciation": m * 0.05 * sales, "ebitda": m * 0.3 * sales, "operating_income": m * 0.25 * sales,
            "net_income": m * 0.15 * sales, "cfo": m * 0.18 * sales, "capex": m * 100.0 * 1.2 ** y, "dividends": m * 40.0, "equity_issuance": m * 10.0, "equity_repurchase": m * 20.0,
            "debt_issuance": m * 30.0, "debt_repayment": m * 10.0, "tax_rate": 0.25, "total_assets": m * assets, "current_assets": m * 0.4 * assets, "cash": m * 0.1 * assets,
            "current_liabilities": m * 0.15 * assets, "short_term_debt": m * 0.03 * assets, "long_term_debt": m * 0.2 * assets, "total_liabilities": m * 0.5 * assets,
            "book_equity": m * 0.5 * assets, "long_term_investments": m * 0.05 * assets, "gross_plant": m * 0.9 * assets, "net_plant": m * 0.5 * assets,
            "interest_expense": m * 0.05 * 0.23 * assets, "shares_outstanding": 100.0 * m * 1.02 ** y, "preferred": 0.0, "minority_interest": 0.0}


def build_table() -> pd.DataFrame:
    rows = [{"ticker": t.lower() if t == "B" else t, "period_end": pd.Timestamp(2013 + y, 12, 31), **statements(m, y)} for t, m in SCALE.items() for y in YEARS]     # one ticker in lower case on purpose
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def inputs():
    prices = pd.DataFrame({t: np.full(len(IDX), p) for t, p in PRICE.items()}, index=IDX)
    return FactorInputs(prices, Fundamentals.from_table(build_table(), lag_days=60))


T = pd.Timestamp("2019-07-01")            # the latest filing public is fiscal 2018 (y = 5, filed 2019-03-01); a year earlier it was y = 4, and so on


def at(inputs, name, **kw):
    return F.compute(name, inputs, **kw).loc[T]


def firm(t):
    """Fiscal-year statements of ticker t at the dates the panel sees: y = 5 now, 4 a year ago, 3 two years ago, 2 three years ago."""
    m = SCALE[t]
    return {y: statements(m, y) for y in (2, 3, 4, 5)}


# ------------------------------------------------------------------------------------------------------------------ point-in-time loading
def table(rows, **kw):
    return Fundamentals.from_table(pd.DataFrame(rows), **kw)


def test_a_figure_is_used_from_its_filing_date_and_not_a_day_before():
    idx = pd.bdate_range("2020-01-01", periods=120)
    f = table([{"ticker": "A", "period_end": "2019-12-31", "sales": 100.0}], lag_days=60)             # public on 2020-02-29, a Saturday: first usable on Monday 2020-03-02
    p = f.panel("sales", idx, ["A"])["A"]
    assert p.loc[:"2020-02-28"].isna().all() and (p.loc["2020-03-02":] == 100.0).all()
    on = table([{"ticker": "A", "period_end": "2019-12-31", "available": "2020-02-12", "sales": 100.0}])         # an explicit availability date beats the default lag
    q = on.panel("sales", idx, ["A"])["A"]
    assert q.loc[:"2020-02-11"].isna().all() and q.loc["2020-02-12"] == 100.0


def test_a_later_filing_replaces_the_earlier_one_and_a_gap_in_it_does_not_erase_the_old_value():
    idx = pd.bdate_range("2020-01-01", periods=200)
    f = table([{"ticker": "A", "available": "2020-01-10", "sales": 100.0, "cash": 7.0}, {"ticker": "A", "available": "2020-04-10", "sales": 120.0, "cash": np.nan},
               {"ticker": "A", "available": "2020-04-10", "sales": 125.0, "cash": np.nan}])
    assert f.panel("sales", idx, ["A"])["A"].loc["2020-04-10":].eq(125.0).all()                       # of two filings on one day the last wins
    assert f.panel("sales", idx, ["A"])["A"].loc["2020-01-10":"2020-04-09"].eq(100.0).all()
    assert f.panel("cash", idx, ["A"])["A"].loc["2020-04-10":].eq(7.0).all()                         # the later filing did not report cash: the old figure stands


def test_a_figure_goes_stale_and_filings_outside_the_calendar_are_handled():
    idx = pd.bdate_range("2020-01-01", periods=300)
    f = table([{"ticker": "A", "available": "2019-06-03", "sales": 100.0}, {"ticker": "B", "available": "2020-06-01", "sales": 5.0}, {"ticker": "A", "available": "2030-01-01", "sales": 9.0}], expiry=100)
    p = f.panel("sales", idx, ["A", "B", "Z"])
    assert p["A"].iloc[0] == 100.0 and p["A"].iloc[100] == 100.0 and p["A"].iloc[101:].isna().all()    # held for `expiry` trading days after the first day it could be used
    assert p["B"].loc[:"2020-05-29"].isna().all() and p["B"].loc["2020-06-01"] == 5.0                  # a later ticker starts later
    assert p["Z"].isna().all()                                                                          # a ticker with no filings is missing, not zero
    assert 9.0 not in p.to_numpy()                                                                      # a filing after the last date is never used


def test_nothing_after_a_date_can_change_what_was_known_on_it():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rows = [{"ticker": "A", "available": "2020-01-10", "sales": 100.0}, {"ticker": "A", "available": "2020-05-01", "sales": 120.0}]
    base = table(rows).panel("sales", idx, ["A"])
    rows[1]["sales"] = -999.0
    changed = table(rows).panel("sales", idx, ["A"])
    assert base.loc[:"2020-04-30"].equals(changed.loc[:"2020-04-30"]) and base.loc["2020-05-01":].ne(changed.loc["2020-05-01":]).all().all()


def test_tickers_are_matched_without_regard_to_case_and_unused_columns_are_listed():
    f = table([{"ticker": " abc ", "available": "2020-01-02", "sales": 3.0, "my_note": "x"}])
    assert f.tickers == ["ABC"] and f.extra == ["my_note"] and f.fields == ["sales"]
    assert f.panel("sales", pd.bdate_range("2020-01-01", periods=5), ["ABC"])["ABC"].iloc[-1] == 3.0


def test_bad_files_are_refused_with_the_reason(tmp_path):
    with pytest.raises(KeyError, match="need the file"):
        Fundamentals.from_csv(tmp_path / "missing.csv")
    with pytest.raises(FundamentalsError, match="ticker"):
        Fundamentals.from_table(pd.DataFrame({"period_end": ["2020-01-01"], "sales": [1.0]}))
    with pytest.raises(FundamentalsError, match="period_end"):
        Fundamentals.from_table(pd.DataFrame({"ticker": ["A"], "sales": [1.0]}))
    with pytest.raises(FundamentalsError, match="not numbers"):
        table([{"ticker": "A", "available": "2020-01-02", "sales": "lots"}])
    with pytest.raises(FundamentalsError, match="lag_days"):
        table([{"ticker": "A", "period_end": "2020-01-02", "sales": 1.0}], lag_days=-1)
    with pytest.raises(FundamentalsError, match="expiry"):
        table([{"ticker": "A", "available": "2020-01-02", "sales": 1.0}], expiry=0)
    with pytest.raises(KeyError, match="no column"):
        table([{"ticker": "A", "available": "2020-01-02", "sales": 1.0}]).panel("cash", pd.bdate_range("2020-01-01", periods=3), ["A"])
    path = tmp_path / "fundamentals.csv"
    build_table().to_csv(path, index=False)
    loaded = Fundamentals.from_csv(path)
    assert set(loaded.fields) == set(statements(1.0, 0)) and loaded.tickers == ["A", "B", "C"]
    na = tmp_path / "na.csv"
    na.write_text("ticker,period_end,sales\nNA,2020-01-01,5\nX,2020-01-01,\n")                       # NA is a ticker, an empty cell is missing
    assert Fundamentals.from_csv(na).tickers == ["NA", "X"] and Fundamentals.from_csv(na).table["sales"].isna().sum() == 1


def test_every_documented_field_has_a_description_and_the_example_uses_only_known_fields():
    assert all(isinstance(v, str) and v for v in FIELDS.values())
    assert set(statements(1.0, 0)) <= set(FIELDS)


# ------------------------------------------------------------------------------------------------------------------ market value and enterprise value
def test_market_value_and_enterprise_value_follow_their_definitions(inputs):
    s = firm("A")[5]
    mcap = PRICE["A"] * s["shares_outstanding"]
    assert inputs.market_cap.loc[T, "A"] == pytest.approx(mcap)
    assert inputs.enterprise_value.loc[T, "A"] == pytest.approx(mcap + s["short_term_debt"] + s["long_term_debt"] - s["cash"])
    assert np.isnan(inputs.enterprise_value.loc[IDX[10], "A"])                                        # nothing was public on day ten: no enterprise value
    rich = FactorInputs(pd.DataFrame({"A": np.full(len(IDX), 1.0)}, index=IDX), Fundamentals.from_table(pd.DataFrame([{**statements(1.0, 5), "cash": 1e6, "ticker": "A", "period_end": "2013-12-31"}])))
    assert rich.enterprise_value.dropna().empty                                                      # a company whose cash exceeds its market value and debt has no meaningful EV
    assert ratio(pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"a": [0.0, np.nan]})).isna().all().all()


def test_the_lagged_and_averaged_panels_look_back_a_year_of_trading_days(inputs):
    sales = inputs.field("sales")
    assert inputs.lag("sales").loc[T, "A"] == sales.iloc[IDX.get_loc(T) - 252]["A"]
    assert inputs.lag("sales").loc[T, "A"] == pytest.approx(firm("A")[4]["sales"]) and inputs.lag("sales", 3).loc[T, "A"] == pytest.approx(firm("A")[2]["sales"])
    assert inputs.average("total_assets").loc[T, "A"] == pytest.approx((firm("A")[5]["total_assets"] + firm("A")[4]["total_assets"]) / 2)


# ------------------------------------------------------------------------------------------------------------------ the factors, by hand
@pytest.mark.parametrize("t", ["A", "B", "C"])
def test_the_value_factors_are_the_ratios_they_are_named_for(inputs, t):
    s, mc = firm(t)[5], PRICE[t] * firm(t)[5]["shares_outstanding"]
    ev = mc + s["short_term_debt"] + s["long_term_debt"] - s["cash"]
    nxf = s["equity_issuance"] - s["equity_repurchase"] - s["dividends"] + s["debt_issuance"] - s["debt_repayment"]
    assert at(inputs, "cfo2ev")[t] == pytest.approx(s["cfo"] / ev)
    assert at(inputs, "ebitda2ev")[t] == pytest.approx(s["ebitda"] / ev)
    assert at(inputs, "earnings_yield")[t] == pytest.approx(s["net_income"] / mc)
    assert at(inputs, "payout_yield")[t] == pytest.approx((s["dividends"] + s["equity_repurchase"] - s["equity_issuance"]) / mc)
    assert at(inputs, "nxf2ev")[t] == pytest.approx(nxf / ev)
    assert at(inputs, "b2p")[t] == pytest.approx(s["book_equity"] / mc)
    assert at(inputs, "s2ev")[t] == pytest.approx(s["sales"] / ev)


@pytest.mark.parametrize("t", ["A", "B", "C"])
def test_the_quality_factors_are_the_ratios_they_are_named_for(inputs, t):
    f = firm(t)
    now, ago, ago2, ago3 = f[5], f[4], f[3], f[2]
    noa = lambda s: (s["total_assets"] - s["cash"] - s["long_term_investments"]) - (s["total_liabilities"] - s["short_term_debt"] - s["long_term_debt"])
    wc = lambda s: (s["current_assets"] - s["cash"]) - (s["current_liabilities"] - s["short_term_debt"])
    nco = lambda s: (s["total_assets"] - s["current_assets"] - s["long_term_investments"]) - (s["total_liabilities"] - s["current_liabilities"] - s["long_term_debt"])
    avg_assets = (now["total_assets"] + ago["total_assets"]) / 2
    assert at(inputs, "rnoa")[t] == pytest.approx(now["operating_income"] * 0.75 / ((noa(now) + noa(ago)) / 2))
    assert at(inputs, "operating_leverage")[t] == pytest.approx((now["cogs"] + now["sga"]) / now["total_assets"])
    assert at(inputs, "operating_leverage_change")[t] == pytest.approx((now["cogs"] + now["sga"]) / now["total_assets"] - (ago["cogs"] + ago["sga"]) / ago["total_assets"])
    assert at(inputs, "wc_inc")[t] == pytest.approx((wc(now) - wc(ago)) / avg_assets)
    assert at(inputs, "nco_inc")[t] == pytest.approx((nco(now) - nco(ago)) / avg_assets)
    assert at(inputs, "capxg")[t] == pytest.approx(now["capex"] / ago["capex"] - 1)
    assert at(inputs, "capxg")[t] == pytest.approx(0.2)
    assert at(inputs, "icapx")[t] == pytest.approx(now["capex"] / ((ago["capex"] + ago2["capex"] + ago3["capex"]) / 3) - 1)
    nxf = now["equity_issuance"] - now["equity_repurchase"] - now["dividends"] + now["debt_issuance"] - now["debt_repayment"]
    assert at(inputs, "xf")[t] == pytest.approx(nxf / avg_assets)
    assert at(inputs, "share_inc")[t] == pytest.approx(1.02 - 1)


def test_a_tax_rate_in_the_file_is_used_and_otherwise_the_default_applies(inputs):
    without = table([{k: v for k, v in r.items() if k != "tax_rate"} for r in build_table().to_dict("records")], lag_days=60)
    x = FactorInputs(inputs.prices, without)
    s4, s5 = firm("A")[4], firm("A")[5]
    noa = lambda s: (s["total_assets"] - s["cash"] - s["long_term_investments"]) - (s["total_liabilities"] - s["short_term_debt"] - s["long_term_debt"])
    assert F.compute("rnoa", x, tax=0.4).loc[T, "A"] == pytest.approx(s5["operating_income"] * 0.6 / ((noa(s5) + noa(s4)) / 2))
    assert F.compute("rnoa", x).loc[T, "A"] == pytest.approx(F.compute("rnoa", inputs).loc[T, "A"])      # the default 25% is the same as the file's


def test_cfroi_solves_the_return_that_equates_the_investment_with_the_cash_it_earns():
    rng = np.random.default_rng(0)
    gcf, life, nda = rng.uniform(5, 30, 200), rng.uniform(4, 30, 200), rng.uniform(0, 40, 200)
    gi = gcf * rng.uniform(3, 9, 200) + nda
    r = F.cfroi_irr(gcf, gi, life, nda)
    ok = np.isfinite(r)
    assert ok.mean() > 0.9
    pv = gcf[ok] * (1 - (1 + r[ok]) ** -life[ok]) / r[ok] + nda[ok] * (1 + r[ok]) ** -life[ok]
    assert np.abs(pv - gi[ok]).max() < 1e-8                                                           # the equation holds at the solution
    assert abs(F.cfroi_irr(10.0, 100.0, 1e6, 0.0) - 0.10) < 1e-6                                       # a perpetuity with no recovery: the return is cash flow over investment
    assert F.cfroi_irr(10.0, 100.0, 20.0, 0.0) < F.cfroi_irr(10.0, 100.0, 200.0, 0.0)                  # the same cash flow earned for longer is a higher return on the same investment
    assert F.cfroi_irr(20.0, 100.0, 15.0, 10.0) > F.cfroi_irr(10.0, 100.0, 15.0, 10.0)                 # more cash flow for the same investment, a higher return
    assert np.isnan(F.cfroi_irr(-5.0, 100.0, 15.0, 10.0)) and np.isnan(F.cfroi_irr(10.0, 5.0, 15.0, 10.0))      # no positive cash flow, or assets alone worth more than the investment: no root
    assert np.isnan(F.cfroi_irr(10.0, 100.0, 0.0, 10.0))


def test_cfroi_factor_uses_gross_cash_flow_over_gross_investment_and_the_plant_life(inputs):
    s = firm("A")[5]
    gcf = s["net_income"] + s["depreciation"] + s["interest_expense"]
    life = float(np.clip(s["gross_plant"] / s["depreciation"], 3, 40))
    expected = float(F.cfroi_irr(gcf, s["total_assets"] + (s["gross_plant"] - s["net_plant"]), life, s["total_assets"] - s["net_plant"]))
    assert at(inputs, "cfroi")["A"] == pytest.approx(expected)
    assert 0.0 < expected < 0.5


def test_the_price_factors_and_the_estimate_factors():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0.0005, 0.01, (900, 3)), index=pd.bdate_range("2016-01-01", periods=900), columns=list("ABC"))
    prices = 100 * (1 + r).cumprod()
    months = np.arange(0, 900, 21)
    rows = []
    for k, ticker in enumerate("ABC"):
        for m, pos in enumerate(months):
            rows.append({"ticker": ticker, "available": prices.index[pos], "eps_fy1": (5.0 + k) * 1.01 ** m, "ltg": 0.05 + 0.001 * m, "n_up": 3 + k, "n_down": 1, "n_estimates": 10})
    x = FactorInputs(prices, Fundamentals.from_table(pd.DataFrame(rows)))
    t = 21 * 30 + 5                                                                                      # five days into month 30: the same phase 9 months (189 days) earlier
    d = prices.index[t]
    assert F.compute("ret1", x).iloc[t].tolist() == pytest.approx((prices.iloc[t] / prices.iloc[t - 21] - 1).tolist())
    assert F.compute("ret9", x).iloc[t].tolist() == pytest.approx((prices.iloc[t - 21] / prices.iloc[t - 21 - 189] - 1).tolist())
    vol = r.shift(21).iloc[t - 188: t + 1].std() * np.sqrt(189)
    assert F.compute("adj_ret9", x).iloc[t].tolist() == pytest.approx((F.compute("ret9", x).iloc[t] / vol).tolist())
    eps = lambda k, m: (5.0 + k) * 1.01 ** m
    assert F.compute("earn_rev9", x).loc[d].tolist() == pytest.approx([(eps(k, 30) - eps(k, 21)) / prices.loc[d, c] for k, c in enumerate("ABC")])
    assert F.compute("earn_diff9", x).loc[d].tolist() == pytest.approx([(3 + k - 1) / 10 for k in range(3)])
    assert F.compute("ltg_rev9", x).loc[d].tolist() == pytest.approx([0.009] * 3)


# ------------------------------------------------------------------------------------------------------------------ what is available
def test_factors_that_the_file_cannot_support_are_skipped_with_the_columns_they_lack_and_refused_by_name(inputs):
    thin = FactorInputs(inputs.prices, table([{"ticker": t, "period_end": "2013-12-31", "sales": 100.0, "shares_outstanding": 10.0, "cash": 1.0, "short_term_debt": 0.0} for t in "ABC"], lag_days=60))
    ok, lacking = F.available(thin)
    assert "s2ev" in ok and "ret9" in ok and "b2p" not in ok and lacking["b2p"] == ["book_equity"]
    assert lacking["xf"] == ["total_assets", F.FINANCING]                                             # an any-of requirement is reported as the set
    with pytest.raises(KeyError, match="book_equity"):
        F.compute("b2p", thin)
    with pytest.raises(KeyError, match="unknown factor"):
        F.compute("nope", thin)
    got, skipped = F.compute_all(thin, "value")
    assert set(got) == {"s2ev"} and "b2p" in skipped
    assert set(F.FACTORS) >= set(F.names("value")) | set(F.names("quality")) | set(F.names("momentum")) | set(F.names("estimates"))


def test_the_library_covers_every_named_factor_with_a_sign_and_a_text():
    named = ["cfo2ev", "ebitda2ev", "earnings_yield", "forward_earnings_yield", "payout_yield", "nxf2ev", "b2p", "s2ev", "rnoa", "cfroi", "operating_leverage", "operating_leverage_change",
             "wc_inc", "nco_inc", "icapx", "capxg", "xf", "share_inc", "ret1", "ret9", "adj_ret9", "earn_rev9", "earn_diff9", "ltg_rev9"]
    assert set(named) <= set(F.FACTORS)
    for n in named:
        spec = F.FACTORS[n]
        assert spec.sign in (-1, 1) and len(spec.text) > 20 and spec.style in ("value", "quality", "momentum", "estimates")
    assert {n for n in named if F.FACTORS[n].sign == -1} == {"nxf2ev", "operating_leverage_change", "wc_inc", "nco_inc", "icapx", "capxg", "xf", "share_inc", "ret1"}
