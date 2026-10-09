"""Discounted cash flow and its multipath version: closed forms, monotonicity, the effect of uncertainty, and the factors built on them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import dcf
from src.equity import factors as F
from src.equity.fundamentals import YEAR, FactorInputs, Fundamentals


# ------------------------------------------------------------------------------------------------------------------ the present value
@pytest.mark.parametrize("g,r", [(0.0, 0.08), (0.03, 0.08), (0.05, 0.12), (-0.02, 0.06)])
def test_a_constant_growth_rate_collapses_to_the_gordon_formula(g, r):
    assert dcf.dcf_value(100.0, g, g, r, 10) == pytest.approx(100.0 * (1 + g) / (r - g), rel=1e-12)
    assert dcf.dcf_value(100.0, g, g, r, 3) == pytest.approx(100.0 * (1 + g) / (r - g), rel=1e-12)          # however many years are called explicit


def test_a_two_year_valuation_by_hand():
    f0, g1, gT, r = 100.0, 0.10, 0.02, 0.08
    f1 = f0 * (1 + g1)                                   # year 1 grows at the first-stage rate
    f2 = f1 * (1 + gT)                                   # year 2 (the last explicit year) grows at the terminal rate
    expected = f1 / (1 + r) + f2 / (1 + r) ** 2 + f2 * (1 + gT) / (r - gT) / (1 + r) ** 2
    assert dcf.dcf_value(f0, g1, gT, r, years=2) == pytest.approx(expected, rel=1e-12)
    three = f1 / (1 + r) + f1 * (1 + 0.06) / (1 + r) ** 2 + f1 * 1.06 * 1.02 / (1 + r) ** 3 + f1 * 1.06 * 1.02 * 1.02 / (r - gT) / (1 + r) ** 3
    assert dcf.dcf_value(f0, g1, gT, r, years=3) == pytest.approx(three, rel=1e-12)       # growth fades 0.10, 0.06, 0.02


def test_growth_fades_linearly_from_the_first_stage_rate_to_the_terminal_rate():
    path = dcf.growth_path(0.10, 0.02, 5)
    assert path.ravel() == pytest.approx([0.10, 0.08, 0.06, 0.04, 0.02])
    assert dcf.growth_path(np.array([0.1, 0.2]), 0.02, 3).shape == (3, 2)
    with pytest.raises(ValueError, match="years"):
        dcf.growth_path(0.1, 0.02, 1)
    with pytest.raises(ValueError, match="years"):
        dcf.dcf_value(1.0, 0.1, 0.02, 0.08, years=1)


def test_value_rises_with_growth_and_cash_flow_and_falls_with_the_discount_rate():
    base = dcf.dcf_value(100.0, 0.05, 0.025, 0.08)
    assert dcf.dcf_value(100.0, 0.08, 0.025, 0.08) > base > dcf.dcf_value(100.0, 0.02, 0.025, 0.08)
    assert dcf.dcf_value(100.0, 0.05, 0.025, 0.07) > base > dcf.dcf_value(100.0, 0.05, 0.025, 0.09)
    assert dcf.dcf_value(200.0, 0.05, 0.025, 0.08) == pytest.approx(2 * base)                 # linear in the starting cash flow
    assert dcf.dcf_value(100.0, 0.05, 0.03, 0.08) > base                                       # a higher terminal rate is worth more
    v = dcf.dcf_value(np.array([[100.0, 50.0]]), 0.05, 0.025, np.array([0.08, 0.10]))
    assert v.shape == (1, 2) and v[0, 0] > v[0, 1]
    assert np.isnan(dcf.dcf_value(100.0, 0.05, 0.05, 0.05)) and np.isnan(dcf.dcf_value(100.0, 0.05, 0.06, 0.05))      # a discount rate that does not exceed the terminal rate is not a valuation


def test_equity_value_per_share_is_enterprise_value_less_net_debt_over_shares():
    assert dcf.per_share(1000.0, 200.0, 50.0) == pytest.approx(16.0)
    assert dcf.per_share(1000.0, -100.0, 10.0) == pytest.approx(110.0)                         # net cash adds to the value
    assert np.isnan(dcf.per_share(1000.0, 0.0, 0.0)) and np.isnan(dcf.per_share(1000.0, 0.0, -5.0))


# ------------------------------------------------------------------------------------------------------------------ the multipath valuation
def test_without_uncertainty_every_path_is_the_point_estimate():
    v = dcf.mdcf_values(np.full((3, 4), 100.0), 0.05, 0.025, 0.08, years=10, paths=50, growth_sd=0.0, rate_sd=0.0, fcf_sd=0.0, terminal_sd=0.0)
    assert v.shape == (50, 3, 4) and np.ptp(v) == 0.0 and v[0, 0, 0] == pytest.approx(dcf.dcf_value(100.0, 0.05, 0.025, 0.08))


def test_uncertainty_in_the_discount_rate_adds_value_because_value_is_convex_in_it():
    kw = dict(years=10, paths=20000, growth_sd=0.0, fcf_sd=0.0, terminal_sd=0.0)
    point = dcf.dcf_value(100.0, 0.05, 0.025, 0.08)
    wide = dcf.mdcf_values(100.0, 0.05, 0.025, 0.08, rate_sd=0.015, **kw)
    narrow = dcf.mdcf_values(100.0, 0.05, 0.025, 0.08, rate_sd=0.005, **kw)
    assert wide.mean() > narrow.mean() > point                                                 # the average of the draws beats the value at the average rate, and by more the wider the spread
    assert abs(np.median(wide) - point) / point < 0.03                                          # the median stays near the point estimate: only the mean is lifted by the long right tail
    assert wide.std() > narrow.std() > 0


def test_the_draws_are_common_random_numbers_and_the_seed_controls_them():
    a = dcf.mdcf_values(np.array([100.0, 100.0]), 0.05, 0.025, 0.08, paths=100, seed=3)
    assert np.array_equal(a[:, 0], a[:, 1])                                                    # two companies with the same inputs get the same distribution
    assert np.array_equal(a, dcf.mdcf_values(np.array([100.0, 100.0]), 0.05, 0.025, 0.08, paths=100, seed=3))
    assert not np.array_equal(a, dcf.mdcf_values(np.array([100.0, 100.0]), 0.05, 0.025, 0.08, paths=100, seed=4))


def test_the_terminal_rate_is_kept_below_the_discount_rate_in_every_draw():
    v = dcf.mdcf_values(100.0, 0.05, 0.04, 0.045, paths=5000, rate_sd=0.02, terminal_sd=0.02)
    assert np.isfinite(v).all() and (v > 0).all()


# ------------------------------------------------------------------------------------------------------------------ the factors
def world(ltg=True, fcf=(80.0, 40.0), seed=0, n=1700):
    """Two companies with constant statements and random-walk prices."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    prices = pd.DataFrame(50 * np.cumprod(1 + rng.normal(0.0003, 0.01, (n, 2)), axis=0), index=idx, columns=["A", "B"])
    rows = []
    for k, t in enumerate("AB"):
        for y in range(8):
            rows.append({"ticker": t, "period_end": pd.Timestamp(2012 + y, 12, 31), "sales": 1000.0 * 1.06 ** y, "cfo": fcf[k] + 20.0, "capex": 20.0, "shares_outstanding": 10.0, "cash": 50.0,
                         "short_term_debt": 20.0, "long_term_debt": 80.0, "total_assets": 1500.0, **({"ltg": 0.08} if ltg else {})})
    return FactorInputs(prices, Fundamentals.from_table(pd.DataFrame(rows), lag_days=60))


def test_the_beta_that_prices_the_discount_rate_is_shrunk_toward_one():
    x = world()
    b = dcf.beta_estimate(x.returns)
    t = 800
    window = x.returns.iloc[t - YEAR + 1: t + 1]
    market = window.mean(axis=1)
    raw = window.apply(lambda col: np.cov(col, market)[0, 1] / market.var())
    assert b.iloc[t].tolist() == pytest.approx((raw * 2 / 3 + 1 / 3).tolist())


def test_the_point_estimate_factor_is_value_per_share_over_price_less_one():
    x = world()
    up = F.compute("dcf_upside", x)
    d = x.index[1000]
    month = dcf._monthly_rows(x)
    row = month[np.searchsorted(month, 1000, side="right") - 1]                                  # the factor is refreshed monthly: the value is the last refresh's
    rate = float(np.clip(0.03 + dcf.beta_estimate(x.returns).iloc[row]["A"] * 0.05, 0.045, 0.25))
    ev = dcf.dcf_value(80.0, 0.08, 0.025, rate, 10)
    value = (ev - (100.0 - 50.0)) / 10.0
    assert up.loc[d, "A"] == pytest.approx(value / x.prices.loc[d, "A"] - 1, rel=1e-9)
    value = lambda c: (up.loc[d, c] + 1) * x.prices.loc[d, c]
    assert value("A") > value("B")                                                             # A earns twice B's free cash flow: it is worth more per share, whatever the price
    assert F.FACTORS["dcf_upside"].style == "valuation" and "dcf_upside" in F.names("valuation")


def test_growth_comes_from_the_analysts_when_given_and_from_three_years_of_sales_otherwise():
    with_ltg, without = world(ltg=True), world(ltg=False)
    rows = np.array([1200])
    kw = dict(risk_free=0.03, equity_premium=0.05, terminal_growth=0.025, years=10, growth_cap=0.30)
    assert dcf._inputs_at(with_ltg, rows, **kw)["growth"][0] == pytest.approx([0.08, 0.08])
    assert dcf._inputs_at(without, rows, **kw)["growth"][0] == pytest.approx([0.06, 0.06], abs=1e-9)          # sales grew 6% a year
    capped = dcf._inputs_at(with_ltg, rows, **{**kw, "growth_cap": 0.05})["growth"][0]
    assert capped == pytest.approx([0.05, 0.05])


def test_a_company_with_no_free_cash_flow_has_no_dcf_value():
    x = world(fcf=(-30.0, 40.0))
    up = F.compute("dcf_upside", x)
    assert up["A"].dropna().empty and up["B"].dropna().size > 500
    assert F.compute("mdcf_upside", x, paths=50)["A"].dropna().empty and F.compute("mdcf_prob", x, paths=50)["A"].dropna().empty


def test_the_multipath_factors_agree_with_the_point_estimate_and_respond_to_the_price():
    x = world()
    point, median = F.compute("dcf_upside", x), F.compute("mdcf_upside", x, paths=300)
    d = x.index[1200]
    assert abs(median.loc[d, "B"] - point.loc[d, "B"]) < 0.25 * (1 + abs(point.loc[d, "B"]))      # the median of the draws sits near the point estimate
    prob = F.compute("mdcf_prob", x, paths=300)
    assert prob.dropna(how="all").stack().between(0, 1).all()
    month = dcf._monthly_rows(x)
    block = prob.iloc[month[20]: month[21]]
    assert block["A"].nunique() == 1                                                          # the probability is refreshed once a month and held
    cheap = FactorInputs(x.prices * 0.5, x.fund)
    assert F.compute("mdcf_prob", cheap, paths=300).loc[d, "A"] >= prob.loc[d, "A"]            # halve the price and more of the plausible values exceed it
    assert F.compute("mdcf_upside", cheap, paths=300).loc[d, "A"] > median.loc[d, "A"]


def test_the_multipath_distribution_has_the_dimensions_the_dates_and_the_inputs_it_used():
    x = world()
    values, rows, v = dcf.mdcf_distribution(x, paths=60)
    assert values.shape == (60, len(rows), 2) and v["fcf0"].shape == (len(rows), 2) and rows[0] == min(dcf.WARMUP, len(x.index) - 1) and (np.diff(rows) == 21).all()
    assert np.nanmin(values) > -1e6
