"""Exact values and edges that a planted-bug audit found nothing checking. The audit changed one operator, comparison, sign or number at a time in the research code (a hundred and fifty places drawn at
random) and ran the tests: about half the changes went unnoticed. Most of those were harmless (a bound that no data touches, a word in a message); these are the ones that were not: a compound growth
rate, a second difference, the fall-back of a free cash flow, the tax rate and the floor of invested capital behind a return on capital, an average of balances, a signal that is exactly zero, the
lags of an error that allows for overlap, a ratio of an information coefficient to a spread of nothing, the default bars of a search and the weights of its fitness, the sign and edge of a verdict."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import discovery as D
from src.fundamentals import evaluate as E
from src.fundamentals import metrics, transforms
from tests.test_fund_metrics import Y19, Y20, _company, _row


def at(base, name: str, years: dict, year: int):
    """The derived metrics of one company in one year, from a database of its own."""
    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)
    store, sid = _company(folder, years)
    return _row(store, sid, year)[0]


# ------------------------------------------------------------------------------------------------------------------------ derived metrics
def test_three_year_growth_is_the_compound_rate_and_a_change_of_a_change_is_the_second_difference(tmp_path):
    grown = {y: dict(Y19, revenue=r) for y, r in zip((2017, 2018, 2019, 2020), (1000.0, 1100.0, 1210.0, 1331.0))}
    assert at(tmp_path / "a", "cagr", grown, 2020)["revenue_cagr_3y"] == pytest.approx(0.10)                  # (1331 / 1000) ** (1 / 3) - 1
    speeding = {2018: dict(Y19, revenue=1000.0), 2019: dict(Y19, revenue=1100.0), 2020: dict(Y19, revenue=1320.0)}
    r = at(tmp_path / "b", "accel", speeding, 2020)
    assert r["d_revenue_growth"] == pytest.approx(0.20 - 0.10) and r["revenue_growth_acceleration"] == pytest.approx(0.10)
    margins = {y: dict(Y19, revenue=1000.0, operating_income=oi) for y, oi in zip((2018, 2019, 2020), (100.0, 120.0, 150.0))}
    assert at(tmp_path / "c", "d2om", margins, 2020)["d2_operating_margin"] == pytest.approx((0.15 - 0.12) - (0.12 - 0.10))
    steps = {y: dict(Y19, operating_income=oi) for y, oi in zip((2018, 2019, 2020), (150.0, 200.0, 290.0))}
    roic = [at(tmp_path / f"d{y}", f"roic{y}", steps, y)["roic"] for y in (2018, 2019, 2020)]
    assert at(tmp_path / "d", "d2roic", steps, 2020)["d2_roic"] == pytest.approx((roic[2] - roic[1]) - (roic[1] - roic[0]))
    assert roic[0] < roic[1] < roic[2]


def test_a_free_cash_flow_the_provider_gives_is_used_and_otherwise_it_is_cash_from_operations_less_capital_spending(tmp_path):
    assert at(tmp_path / "a", "given", {2019: Y19, 2020: dict(Y20, free_cash_flow=999.0)}, 2020)["fcf"] == 999.0
    absent = {k: v for k, v in Y20.items() if k != "free_cash_flow"}
    assert at(tmp_path / "b", "absent", {2019: Y19, 2020: absent}, 2020)["fcf"] == 190.0 - 70.0


def test_the_tax_rate_behind_the_return_on_capital_is_the_firms_own_from_zero_to_45_percent_and_the_default_otherwise(tmp_path):
    invested = 480.0 + 1300.0 - 260.0
    for case, (tax, pretax, rate) in enumerate(((0.0, 200.0, 0.0), (90.0, 200.0, 0.45), (100.0, 200.0, metrics.DEFAULT_TAX_RATE), (-10.0, 200.0, metrics.DEFAULT_TAX_RATE), (50.0, 200.0, 0.25))):
        r = at(tmp_path / f"t{case}", f"tax{case}", {2019: Y19, 2020: dict(Y20, income_tax=tax, income_before_tax=pretax)}, 2020)
        assert r["roic"] == pytest.approx(255.0 * (1.0 - rate) / invested), (tax, pretax)                  # a rate of exactly 0 or 45% is the firm's own; 50% and a refund are not


def test_invested_capital_is_floored_at_a_share_of_assets_and_has_no_value_without_assets(tmp_path):
    hoard = dict(Y20, cash_and_equivalents=5000.0)                                                       # more cash than debt and equity together
    r = at(tmp_path / "a", "hoard", {2019: Y19, 2020: hoard}, 2020)
    assert metrics.MIN_INVESTED_CAPITAL == 0.05 and r["roic"] == pytest.approx(255.0 * (1.0 - 58.25 / 233.0) / (0.05 * 2200.0))
    none = at(tmp_path / "b", "noassets", {2019: Y19, 2020: dict(Y20, total_assets=0.0)}, 2020)
    assert math.isnan(none["roic"])


def test_an_average_of_an_opening_and_a_closing_balance_needs_both_to_be_positive(tmp_path):
    ok = at(tmp_path / "a", "both", {2019: Y19, 2020: Y20}, 2020)
    assert ok["roe_avg"] == pytest.approx(174.75 / ((1300.0 + 1100.0) / 2.0))
    assert math.isnan(at(tmp_path / "b", "open0", {2019: dict(Y19, shareholders_equity=0.0), 2020: Y20}, 2020)["roe_avg"])           # an opening equity of exactly zero
    assert math.isnan(at(tmp_path / "c", "close0", {2019: Y19, 2020: dict(Y20, shareholders_equity=0.0)}, 2020)["roe_avg"])         # and a closing one


def test_each_sign_of_the_f_score_is_no_when_the_change_is_exactly_zero(tmp_path):
    flat = dict(Y19, net_income=0.0, operating_cash_flow=0.0)                       # no profit, cash flow equal to it, the same current ratio and the same number of shares as a year before
    r = at(tmp_path, "zero", {2019: Y19, 2020: flat}, 2020)
    assert r["pio_roa_positive"] == 0.0 and r["pio_cfo_positive"] == 0.0 and r["pio_cash_beats_profit"] == 0.0 and r["pio_liquidity_rose"] == 0.0
    assert r["pio_no_dilution"] == 1.0                                                                       # "no new shares" holds when the count did not change
    assert r["pio_roa_improved"] == 0.0


# ------------------------------------------------------------------------------------------------------------------------ transformations
def test_a_clip_holds_both_sides_and_a_historical_score_uses_the_defaults_it_documents():
    panel = pd.DataFrame({"a": [-5.0, -2.0, 0.0, 2.0, 5.0]})
    assert transforms.apply(panel, [{"op": "clip", "limit": 2.0}], None, investable=False)["a"].tolist() == [-2.0, -2.0, 0.0, 2.0, 2.0]
    assert transforms.apply(panel, [{"op": "clip"}], None, investable=False)["a"].tolist() == [-3.0, -2.0, 0.0, 2.0, 3.0]            # the default limit is three
    p = pd.DataFrame(np.random.default_rng(0).standard_normal((50, 2)))
    z = transforms.hist_zscore(p)
    assert z.iloc[:11].isna().all().all() and z.iloc[11].notna().all()                                       # twelve values are needed before there is a past to compare with
    window = p.iloc[5:41]                                                                                    # thirty-six dates ending today
    assert z.iloc[40, 0] == pytest.approx((p.iloc[40, 0] - window[0].mean()) / window[0].std(ddof=1))


# ------------------------------------------------------------------------------------------------------------------------ the statistics of a study
def test_the_error_that_allows_for_overlap_uses_the_lags_the_rule_of_thumb_gives_and_never_more_than_the_data_allow():
    x = np.random.default_rng(0).standard_normal(300)
    assert E.newey_west(x)["lags"] == 5 and E.newey_west(x[:100])["lags"] == 4 and E.newey_west(x[:30])["lags"] == 3       # floor(4 (n / 100) ** (2 / 9))
    assert E.newey_west(x[:5], lags=10)["lags"] == 3 and E.newey_west(x[:5], lags=-3)["lags"] == 0                          # at most n - 2, at least 0
    ic = pd.Series(np.random.default_rng(1).normal(0.03, 0.08, 120))
    assert E.ic_stats(ic, 21, 21, 12)["hac_lags"] == 4 and E.ic_stats(ic, 63, 21, 12)["hac_lags"] == 4 and E.ic_stats(ic, 252, 21, 12)["hac_lags"] == 11    # the larger of the overlap and the rule


def test_a_series_that_never_moves_has_no_information_ratio():
    flat = E.ic_stats(pd.Series([0.5] * 40), 21, 21, 12)
    assert flat["std_ic"] == 0.0 and math.isnan(flat["icir"]) and math.isnan(flat["icir_annualised"]) and flat["mean_ic"] == 0.5


def test_a_verdict_is_given_from_24_dates_and_says_nothing_about_buckets_that_cannot_be_measured():
    def fake(t, mean, sharpe, mono, n=60):
        return {"ic": {"n_dates": n, "t_stat": t, "mean_ic": mean}, "portfolio": {"net": {"sharpe": sharpe}}, "quantiles": {"monotonicity": {"spearman": mono}}}

    assert E.MIN_DATES == 24
    assert not E.verdict(fake(4.0, 0.04, 1.0, 0.9, n=24)).startswith("Too little data") and E.verdict(fake(4.0, 0.04, 1.0, 0.9, n=23)).startswith("Too little data")
    for mono in (None, float("nan")):
        said = E.verdict(fake(4.0, 0.04, 1.0, mono))
        assert said.startswith("Strong in-sample evidence") and "bucket" not in said
    assert "buckets rise in order" in E.verdict(fake(4.0, 0.04, 1.0, 0.81)) and "buckets do not rise cleanly" in E.verdict(fake(4.0, 0.04, 1.0, 0.49))
    assert "buckets" not in E.verdict(fake(4.0, 0.04, 1.0, 0.8)) and "buckets" not in E.verdict(fake(4.0, 0.04, 1.0, 0.5))      # exactly on the edges: neither praise nor warning


# ------------------------------------------------------------------------------------------------------------------------ the bars and the fitness of a search
def test_the_default_bars_and_the_weights_of_the_fitness_are_what_the_documentation_says():
    assert asdict(D.Thresholds()) == {"min_coverage": 0.40, "min_dates": 36, "t_ic": 1.65, "min_abs_ic": 0.005, "positive_share": 0.55, "min_icir": 0.10, "best_year_t": 1.0, "monotonicity": 0.5,
                                      "ls_t": 1.5, "ls_sharpe": 0.3, "net_sharpe": 0.15, "max_turnover": 0.7, "val_t": 1.0, "oos_t": 1.0, "max_q": 0.25, "max_correlation": 0.9}
    assert D.FITNESS == {"ic": 0.30, "stability": 0.20, "sharpe": 0.25, "monotonic": 0.10, "decorrelation": 0.15, "turnover": -0.15, "complexity": -0.10}
    middle = dict(ic=0.025, val_ic=0.025, positive_share=0.65, sharpe=0.75, val_sharpe=0.75, monotonicity=0.5, turnover=0.35, nodes=9)       # every part at one half of its scale
    score, parts = D.fitness(middle, decorrelation=0.5)
    assert parts == pytest.approx({"ic": 0.5, "stability": 0.5, "sharpe": 0.5, "monotonic": 0.5, "decorrelation": 0.5, "turnover": 0.5, "complexity": 0.5})
    assert score == pytest.approx(0.5 * (0.30 + 0.20 + 0.25 + 0.10 + 0.15 - 0.15 - 0.10))
    top, parts = D.fitness(dict(middle, ic=0.2, val_ic=0.3, positive_share=1.0, sharpe=9.0, val_sharpe=9.0, monotonicity=2.0, turnover=3.0, nodes=40), decorrelation=7.0)
    assert all(v == 1.0 for v in parts.values()) and top == pytest.approx(0.30 + 0.20 + 0.25 + 0.10 + 0.15 - 0.15 - 0.10)               # no part is more than its scale
    low, parts = D.fitness(dict(ic=0.0, val_ic=0.0, positive_share=0.1, sharpe=-3.0, val_sharpe=-3.0, monotonicity=-1.0, turnover=0.0, nodes=1), decorrelation=-2.0)
    assert all(v == 0.0 for v in parts.values()) and low == 0.0
