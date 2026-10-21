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
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.pit import PeriodTables, PitSettings, availability, step_panel
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement
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


# ------------------------------------------------------------------------------------------------------------------------ what a study may see, and when
def test_the_assumed_filing_lags_and_the_age_limits_are_the_documented_ones_and_a_missing_filing_date_uses_them():
    s = PitSettings()
    assert dict(s.estimated_lag_days) == {"FY": 120, "H1": 90, "H2": 90, "Q1": 60, "Q2": 60, "Q3": 60, "Q4": 60} and s.lag_for("T1") == 120 and s.info_lag_days == 1
    assert s.age_for("fy") == 640 and s.age_for("ttm") == 540
    ends = pd.to_datetime(["2020-12-31", "2020-06-30", "2020-03-31", "2020-09-30"])
    nothing = pd.NaT
    periods = pd.DataFrame({"period_end": ends, "fiscal_period": ["FY", "H1", "Q1", "Q3"], "filing_date": nothing, "accepted_date": nothing, "publication_date": nothing, "pit_quality": "ok"})
    available, estimated = availability(periods, s)
    assert estimated.all() and list((available - ends).dt.days) == [121, 91, 61, 61]                               # the lag, then the day after: a filing may have come after the close
    known = periods.assign(filing_date=pd.to_datetime(["2021-02-10", "2020-08-01", "2020-05-01", "2020-11-01"]))
    available, estimated = availability(known, s)
    assert not estimated.any() and list(available.dt.strftime("%Y-%m-%d")) == ["2021-02-11", "2020-08-02", "2020-05-02", "2020-11-02"]
    assert list(availability(known, PitSettings(info_lag_days=30))[0].dt.strftime("%Y-%m-%d")) == ["2021-03-12", "2020-08-31", "2020-05-31", "2020-12-01"]


def test_a_figure_is_kept_until_the_day_its_period_is_the_limit_old_and_dropped_the_day_after():
    end = pd.Timestamp("2020-01-01")
    ev = pd.DataFrame({"security_id": [1], "available_date": [pd.Timestamp("2020-03-01")], "period_end": [end], "value": [5.0]})
    dates = pd.DatetimeIndex([end + pd.Timedelta(days=d) for d in (59, 60, 599, 600, 601)])
    panel = step_panel(ev, dates, [1], max_age_days=600)[1]
    assert np.isnan(panel.iloc[0]) and panel.iloc[1] == 5.0 and panel.iloc[2] == 5.0 and panel.iloc[3] == 5.0 and np.isnan(panel.iloc[4])     # known from the 60th day, expired after the 600th


def _quarters(store, name, rows, filing_after=40):
    """A company whose statements are given as ``(fiscal period, period end, revenue, net income, total assets)``."""
    sid = store.upsert_security(CompanyProfile("simulated", name, name, name, exchange="NYSE"))
    store.store_financials(sid, "simulated", [PeriodStatement(fp, end, dict(revenue=rev, net_income=ni, total_assets=assets), fiscal_year=int(end[:4]),
                                                              filing_date=(pd.Timestamp(end) + pd.Timedelta(days=filing_after)).strftime("%Y-%m-%d")) for fp, end, rev, ni, assets in rows])
    return sid


def test_trailing_twelve_months_add_four_consecutive_reported_quarters_or_two_halves_and_restart_after_a_gap(tmp_path):
    store = FundamentalsStore(tmp_path / "ttm.db")
    ends = ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31", "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31", "2020-06-30"]
    quarters = [(f"Q{i % 4 + 1}", end, 100.0 + 10 * i, 10.0 + i, 1000.0 + 5 * i) for i, end in enumerate(ends)]
    full = _quarters(store, "FULL", quarters)
    gap = _quarters(store, "GAP", [q for q in quarters if q[1] != "2019-06-30"])                                  # one quarter was never reported
    halves = _quarters(store, "HALF", [("H1", "2018-06-30", 200.0, 20.0, 2000.0), ("H2", "2018-12-31", 220.0, 22.0, 2100.0), ("H1", "2019-06-30", 240.0, 24.0, 2200.0)])
    ttm = PeriodTables(store, [full, gap, halves]).ttm

    def revenue(sid):
        sub = ttm[ttm["security_id"] == sid]
        return {d.strftime("%Y-%m-%d"): v for d, v in zip(sub["period_end"], sub["revenue"])}

    assert revenue(full) == {ends[i]: sum(100.0 + 10 * j for j in range(i - 3, i + 1)) for i in range(3, 10)}               # 460 at the end of the first year, then a window that slides by a quarter
    assert revenue(gap) == {"2018-12-31": 460.0, "2019-03-31": 500.0, "2020-06-30": 160.0 + 170.0 + 180.0 + 190.0}        # the missing quarter breaks the run: four new quarters are needed after it
    assert revenue(halves) == {"2018-12-31": 420.0, "2019-06-30": 460.0}                                                    # two halves of 182 days make a year
    sub = ttm[(ttm["security_id"] == full) & (ttm["period_end"] == pd.Timestamp("2018-12-31"))].iloc[0]
    assert sub["net_income"] == 10.0 + 11 + 12 + 13 and sub["total_assets"] == 1015.0 and sub["fiscal_period"] == "TTM"         # a flow is summed, a balance is the latest one


def test_quarters_that_do_not_add_up_to_the_year_by_more_than_a_tenth_are_not_used_and_the_year_stands_in(tmp_path):
    store = FundamentalsStore(tmp_path / "ttm2.db")
    ends = ["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31"]
    quarters = [(f"Q{i + 1}", end, 100.0 + 10 * i, 10.0, 1000.0) for i, end in enumerate(ends)]            # 100 + 110 + 120 + 130 = 460
    ids = {}
    for name, annual in (("CLOSE", 505.0), ("FAR", 520.0)):                                                     # 8.9% and 11.5% below the year
        sid = _quarters(store, name, quarters)
        store.store_financials(sid, "simulated", [PeriodStatement("FY", "2018-12-31", dict(revenue=annual, net_income=40.0, total_assets=1000.0), fiscal_year=2018, filing_date="2019-02-10")])
        ids[name] = sid
    tables = PeriodTables(store, list(ids.values()))
    ttm = tables.ttm.set_index("security_id")
    assert ttm.loc[ids["CLOSE"], "revenue"] == 460.0 and ttm.loc[ids["FAR"], "revenue"] == 520.0                    # the first keeps its quarters, the second takes the annual figure
    assert any("1 companies' quarterly figures do not add up" in n for n in tables.notes)


# ------------------------------------------------------------------------------------------------------------------------ the legs, the groups and the look-ahead audit
def test_a_leg_weighted_by_its_signal_weighs_each_name_by_its_distance_from_the_middle_and_a_thin_leg_is_empty():
    dates = pd.to_datetime(["2020-01-31"])
    signal = pd.DataFrame([np.arange(10.0)], index=dates, columns=range(10))                                    # ranks 0.1 ... 1.0
    by_signal = E.leg_weights(signal, 0.25, "signal", None, min_names=2)
    assert by_signal[0].iloc[0].round(6).tolist() == [0, 0, 0, 0, 0, 0, 0, 0.25, 0.333333, 0.416667]                # |rank - 0.5| = 0.3, 0.4, 0.5 over 1.2
    assert by_signal[1].iloc[0].round(6).tolist() == [0.571429, 0.428571, 0, 0, 0, 0, 0, 0, 0, 0]                   # 0.4 and 0.3 over 0.7
    equal = E.leg_weights(signal, 0.25, "equal", None, min_names=2)
    assert equal[0].iloc[0].round(6).tolist() == [0] * 7 + [round(1 / 3, 6)] * 3 and equal[1].iloc[0].round(6).tolist() == [0.5, 0.5] + [0] * 8
    thin = E.leg_weights(signal, 0.25, "equal", None)                                                           # three names are needed in a leg: the bottom one has two
    assert thin[0].iloc[0].sum() == pytest.approx(1.0) and thin[1].iloc[0].sum() == 0.0
    with pytest.raises(FundamentalsError, match="equal, signal or inv_vol"):
        E.leg_weights(signal, 0.25, "vibes", None)


def test_a_result_is_called_flattered_when_early_knowledge_adds_over_half_of_the_honest_ic_or_when_the_honest_ic_is_nothing_and_the_early_one_is_not():
    flat = E.flattered_by_early_knowledge
    assert flat(0.04, 0.07) and not flat(0.04, 0.059) and flat(-0.04, -0.07) and not flat(-0.04, -0.059)       # +0.03 is over half of 0.04, +0.019 is not, whichever way the signal points
    assert flat(0.02, 0.031) and not flat(0.02, 0.029) and flat(-0.02, -0.031) and not flat(-0.02, -0.029)
    assert not flat(0.04, 0.02) and not flat(0.04, 0.04)                                                     # an honest figure that is better than the early one is not flattered
    assert flat(0.005, -0.031) and not flat(0.005, -0.029)                                                   # next to nothing honestly, and over 0.03 either way with early knowledge
    assert flat(0.0099, -0.05) and not flat(0.0101, -0.05)                                                   # "next to nothing" is under 0.01
    assert flat(0.0, 0.031) and flat(0.0, -0.031) and not flat(0.0, 0.03) and not flat(0.0, 0.0)


def test_a_signal_is_said_to_fade_when_the_first_wait_of_a_month_or_more_leaves_less_than_half_of_an_ic_worth_more_than_a_hundredth():
    def lags(*pairs):
        return [{"lag_days": d, "mean_ic": ic, "t_stat": 1.0, "n_dates": 50} for d, ic in pairs]

    fade = E.fades_when_waiting
    assert fade(lags((1, 0.04), (30, 0.019), (60, 0.01), (90, 0.0)), 1) and not fade(lags((1, 0.04), (30, 0.021), (60, 0.0), (90, 0.0)), 1)         # it is the first wait that counts
    assert fade(lags((1, -0.04), (30, -0.019)), 1) and not fade(lags((1, -0.04), (30, -0.021)), 1) and fade(lags((1, 0.04), (30, -0.01)), 1)       # an IC that turned over has gone
    assert not fade(lags((1, 0.01), (30, 0.0)), 1) and fade(lags((1, 0.0101), (30, 0.0)), 1)                  # an IC of a hundredth or less is too small to say it faded
    assert not fade(lags((1, 0.04)), 1) and not fade([], 1) and not fade(lags((1, 0.04), (30, None)), 1) and not fade(lags((1, 0.04), (30, float("nan"))), 1)      # nothing to compare with
    assert not fade(lags((30, 0.04), (60, 0.01)), 30)                                                          # the study's own lag is a month: the wait it is compared with is itself
    assert fade(lags((1, 0.04), (30, 0.01)), 7)                                                              # no entry for the study's own lag: the first one stands in for it


def test_a_group_is_reported_with_eight_companies_and_twelve_dates_and_a_regime_with_twelve_dates():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2015-01-31", periods=40, freq="30D")
    cols = list(range(20))
    signal = pd.DataFrame(rng.standard_normal((40, 20)), index=dates, columns=cols)
    forward = signal * 0.5 + pd.DataFrame(rng.standard_normal((40, 20)), index=dates, columns=cols)
    labels = pd.Series(["a"] * 8 + ["b"] * 7 + ["c"] * 5, index=cols)
    out = E._group_ic(signal, forward, labels, 21, 21, 12)
    assert sorted(out) == ["a"] and out["a"]["companies"] == 8 and out["a"]["n_dates"] == 40 and out["a"]["mean_ic"] > 0
    assert sorted(E._group_ic(signal, forward, labels, 21, 21, 12, 7)) == ["a", "b"] and sorted(E._group_ic(signal, forward, labels, 21, 21, 12, 5)) == ["a", "b", "c"]
    for dates_with_a_return, kept in ((11, []), (12, ["a"])):
        cut = forward.copy()
        cut.iloc[dates_with_a_return:] = np.nan
        assert sorted(E._group_ic(signal, cut, labels, 21, 21, 12)) == kept
    ic = pd.Series(rng.normal(0.02, 0.05, 40), index=dates)
    regime = pd.Series(["up"] * 12 + ["down"] * 11 + [None] * 17, index=dates)
    got = E._regime_ic(ic, regime, 21, 21, 12)
    assert list(got) == ["up"] and got["up"]["n_dates"] == 12                                                  # twelve dates are enough, eleven are not


def test_the_reports_put_the_right_article_before_the_name_of_the_bars():
    from src.fundamentals import report
    assert [report._a(w) for w in ("exploratory", "standard", "strict")] == ["an exploratory", "a standard", "a strict"]
    pf = {"strictness": "exploratory", "formulas": 80, "reading": "r", "table": [], "blocks": {"train": ["a", "b"], "validation": ["c", "d"], "oos": ["e", "f"], "dates": {"train": 1, "validation": 2, "oos": 3}, "embargo": 1}}
    assert report.format_preflight(pf).startswith("What an exploratory search of 80 formulas could find")
    placebo = {"strictness": "strict", "reading": "x", "summary": {"runs": 1, "tested": 5, "tail": {}, "funnel_mean": {}}}
    assert report.format_placebo(placebo).startswith("Calibration of a strict search on shuffled returns (1 shuffle, 5 formulas tested)")
    assert report.format_placebo(dict(placebo, strictness="exploratory")).startswith("Calibration of an exploratory search")
