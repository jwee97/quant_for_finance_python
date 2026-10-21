"""Exact values and edges that a planted-bug audit found nothing checking. The audit changed one operator, comparison, sign or number at a time in the research code (a hundred and fifty places drawn at
random) and ran the tests: about half the changes went unnoticed. Most of those were harmless (a bound that no data touches, a word in a message); these are the ones that were not: a compound growth
rate, a second difference, the fall-back of a free cash flow, the tax rate and the floor of invested capital behind a return on capital, an average of balances, a signal that is exactly zero, the
lags of an error that allows for overlap, a ratio of an information coefficient to a spread of nothing, the default bars of a search and the weights of its fitness, the sign and edge of a verdict."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import discovery as D
from src.fundamentals import evaluate as E
from src.fundamentals import expr as X
from src.fundamentals import library as L
from src.fundamentals import metrics, transforms
from src.fundamentals.errors import CreditsExhausted, FundamentalsError, InvalidKey, MissingKey, NotFound, PlanError, ProviderError, QuotaExhausted, RateLimited, SyncPending
from src.fundamentals.keys import MAX_NUMBERED, KeyPool, discover
from src.fundamentals.providers import openfilings as of
from src.fundamentals.pit import PeriodTables, PitSettings, availability, step_panel
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement
from src.fundamentals.usage import Allowance, UsageLog, allowance
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


def test_a_free_cash_flow_falls_back_on_cash_from_operations_less_capital_spending_row_by_row():
    class Table(metrics.Ctx):                                                                              # the table of figures, without a database behind it
        def __init__(self, **columns):
            self.t = pd.DataFrame(columns)

    got = metrics._fcf(Table(free_cash_flow=[np.nan, 5.0, np.nan], operating_cash_flow=[190.0, 1.0, np.nan], capital_expenditure=[70.0, 1.0, 3.0]))
    assert got.iloc[0] == 120.0 and got.iloc[1] == 5.0 and np.isnan(got.iloc[2])                           # the provider's figure where there is one, else the difference, else nothing


def test_no_infinity_leaves_the_derived_metrics_and_the_definitions_cannot_be_changed(tmp_path):
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.DEFINITIONS[0].name = "changed"
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    tables = PeriodTables(store, [sid])

    @metrics.metric("zz_plus_infinity", "A test", "quality", "ratio", "An infinite figure.", stage=9)
    def _plus(c):
        return pd.Series(np.inf, index=c.t.index)

    @metrics.metric("zz_minus_infinity", "A test", "quality", "ratio", "A figure that falls to minus infinity.", stage=9)
    def _minus(c):
        return pd.Series(-np.inf, index=c.t.index)

    try:
        values = metrics.compute(tables, "fy")
        assert values["zz_plus_infinity"].isna().all() and values["zz_minus_infinity"].isna().all()
    finally:
        del metrics.DEFINITIONS[-2:]


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


def test_the_buckets_are_judged_from_three_populated_ones_and_an_empty_one_is_not_counted():
    three = E.monotonicity(pd.Series([1.0, 2.0, 3.0]))
    assert three["spearman"] == pytest.approx(1.0) and three["rising_pairs"] == 1.0 and three["buckets"] == 3                    # three are enough
    gap = E.monotonicity(pd.Series([1.0, np.nan, 3.0, 2.0]))
    assert gap["buckets"] == 3 and gap["spearman"] == pytest.approx(0.5) and gap["rising_pairs"] == 0.5                            # an empty bucket is left out, not read as a fall
    assert np.isnan(E.monotonicity(pd.Series([1.0, np.nan, 3.0]))["spearman"])                                                     # two populated ones are not


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


def test_a_table_lists_its_figures_without_the_bookkeeping_and_gives_the_figures_of_the_year_before(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    tables = PeriodTables(store, [sid])
    figures = tables.columns("fy")
    assert {"revenue", "net_income", "total_assets"} <= set(figures) and not {"period_id", "security_id", "period_end", "filing_date", "available_date", "provider", "basis", "pit_estimated"} & set(figures)
    assert "index" not in tables.fy.columns and list(tables.fy.index) == [0, 1]                                      # the rows are numbered afresh, the old numbers do not become a column
    prior = tables.prior("fy", 1)
    assert list(prior.index) == [0, 1] and prior.loc[1, "revenue"] == Y19["revenue"] and prior.loc[1, "net_income"] == Y19["net_income"] and np.isnan(prior.loc[0, "revenue"])


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


def test_quarters_follow_each_other_when_their_period_ends_are_61_to_121_days_apart_and_halves_152_to_212(tmp_path):
    store = FundamentalsStore(tmp_path / "spacing.db")

    def run(name, labels, days):
        return _quarters(store, name, [(label, (pd.Timestamp("2018-03-31") + pd.Timedelta(days=days * i)).strftime("%Y-%m-%d"), 100.0, 10.0, 1000.0) for i, label in enumerate(labels)])

    made = {("quarters", d): run(f"Q{d}", ["Q1", "Q2", "Q3", "Q4"], d) for d in (60, 61, 121, 122)}
    made.update({("halves", d): run(f"H{d}", ["H1", "H2"], d) for d in (151, 152, 212, 213)})
    ttm = PeriodTables(store, list(made.values())).ttm
    has_a_year = {key: bool((ttm["security_id"] == sid).any()) for key, sid in made.items()}
    assert has_a_year == {("quarters", 60): False, ("quarters", 61): True, ("quarters", 121): True, ("quarters", 122): False,
                          ("halves", 151): False, ("halves", 152): True, ("halves", 212): True, ("halves", 213): False}                       # a quarter is 91 days and a half 182, give or take 30


def test_a_set_of_tables_prints_briefly_and_is_compared_by_identity_not_by_its_tables(tmp_path):
    store, sid = _company(tmp_path, {2019: Y19, 2020: Y20})
    tables = PeriodTables(store, [sid])
    assert len(repr(tables)) < 120 and hash(tables) == hash(tables) and tables == tables and tables != PeriodTables(store, [sid])


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


# ------------------------------------------------------------------------------------------------------------------------ the library and its monitor
def test_the_monitor_calls_an_alpha_broken_decaying_weakening_or_healthy_exactly_at_the_bars():
    ok = dict(original=0.04, recent=0.03, two_years=0.03, recent_t=1.5, coverage_now=100.0, coverage_then=100.0)

    def judged(**changes):
        return L.judge(**{**ok, **changes})

    assert judged() == "Healthy"
    assert judged(recent_t=-2.0) == "Broken" and judged(recent_t=-1.99) == "Healthy" and judged(recent_t=None) == "Healthy" and judged(recent_t=float("nan")) == "Healthy"      # -2 is already reversed with evidence
    assert judged(coverage_now=29.99) == "Broken" and judged(coverage_now=30.0) == "Healthy"                                      # under 30% of the companies it once covered, not 30% itself
    assert judged(coverage_then=0.0, coverage_now=0.0) == "Healthy"                                                                  # there was nothing to lose
    assert judged(recent=0.0) == "Decaying" and judged(recent=-0.01) == "Decaying" and judged(recent=0.001) == "Weakening"          # zero is already no information
    assert judged(two_years=0.01) == "Healthy" and judged(two_years=0.0099) == "Decaying"                                           # a quarter of the original is not yet decaying
    assert judged(recent=0.02) == "Healthy" and judged(recent=0.0199) == "Weakening"                                               # nor is half of it weakening
    assert judged(original=0.0, recent=0.001, two_years=0.001) == "Healthy" and judged(original=0.0, recent=0.0, two_years=0.0) == "Decaying"       # with nothing to compare with, only a recent IC of zero or less is bad
    assert judged(recent=-0.01, recent_t=-3.0) == "Broken" and judged(recent=-0.01, coverage_now=1.0) == "Broken"                  # the worse verdict wins


def test_a_candidate_is_validated_at_its_bars_and_not_below_them_and_the_alphas_and_their_reasons_are_what_they_say(tmp_path):
    import re

    lib = L.AlphaLibrary(FundamentalsStore(tmp_path / "lib.db"))
    count = iter(range(1000))

    def alpha(**metrics_):
        settings = {"freq": "M", "horizon": "1m", "tag": next(count)}                                         # different conditions make a different alpha of the same formula
        return lib.add_candidate(X.parse("div(fcf,market_cap)"), settings, {"oos_t": 1.5, "q_value": 0.05, "deflated_sharpe": 0.5, **metrics_})

    assert (L.VALIDATE_MIN_OOS_T, L.VALIDATE_MAX_Q, L.VALIDATE_MIN_DSR) == (1.0, 0.10, 0.90)
    assert lib.validate(alpha())["status"] == "Validated" and lib.validate(alpha(oos_t=1.0))["status"] == "Validated"                  # a locked-dates t of exactly 1.0 will do
    for weak in (0.99, None):
        with pytest.raises(FundamentalsError, match="out-of-sample t-statistic"):
            lib.validate(alpha(oos_t=weak))
    assert lib.validate(alpha(q_value=0.10, deflated_sharpe=0.1))["status"] == "Validated"                                           # a q-value of exactly 0.10 will do, with a poor deflated Sharpe
    assert lib.validate(alpha(q_value=None, deflated_sharpe=0.90))["status"] == "Validated"                                          # and so will a probability of exactly 0.90 with no q-value
    for q, dsr in ((0.11, 0.89), (None, 0.89), (0.11, None), (None, None)):
        with pytest.raises(FundamentalsError, match="neither the false-discovery q-value"):
            lib.validate(alpha(q_value=q, deflated_sharpe=dsr))
    forced = lib.validate(alpha(q_value=0.5, deflated_sharpe=0.1), force_reason="a person looked at it")
    assert forced["status"] == "Validated" and forced["status_reason"].startswith("approved by a person despite") and "a person looked at it" in forced["status_reason"]
    fresh = alpha()
    assert re.fullmatch(r"A[0-9a-f]{8}-[0-9a-f]{4}", fresh)                                                                         # eight characters of the formula, four of the conditions
    assert len(lib.set_status(fresh, "Rejected", "x" * 400)["status_reason"]) == 300                                                # a reason is kept to 300 characters


def test_a_refused_move_names_where_the_alpha_may_go_and_the_log_says_why_each_one_was_added_and_whether_a_move_was_forced(tmp_path):
    import re

    lib = L.AlphaLibrary(FundamentalsStore(tmp_path / "lib.db"))
    plain = lib.add_candidate(X.parse("div(fcf,market_cap)"), {"freq": "M", "tag": 1}, {})
    reasoned = lib.add_candidate(X.parse("div(fcf,market_cap)"), {"freq": "M", "tag": 2}, {}, reason="found by a search")
    assert [e["reason"] for e in lib.status_log(plain)] == ["added"] and [e["reason"] for e in lib.status_log(reasoned)] == ["found by a search"]
    assert lib.get(plain)["status_reason"] == "passed every screen" and lib.get(reasoned)["status_reason"] == "found by a search"
    with pytest.raises(FundamentalsError, match=re.escape("is Candidate: it cannot go straight to Production (allowed: Validated, Rejected, Retired)")):
        lib.set_status(plain, "Production", "too soon")
    lib.set_status(plain, "Rejected", "no good")
    with pytest.raises(FundamentalsError, match=re.escape("it cannot go straight to Candidate (allowed: nothing)")):
        lib.set_status(plain, "Candidate", "second thoughts")
    lib.set_status(plain, "Candidate", "second thoughts", force=True)
    assert [(e["from_status"], e["to_status"], e["forced"]) for e in lib.status_log(plain)] == [(None, "Candidate", 0), ("Candidate", "Rejected", 0), ("Rejected", "Candidate", 1)]


def _watched(tmp_path, freq, signs, thin_after=None, keep=12, **settings):
    """A library with one alpha and a context made by hand: thirty companies whose signal is 1 ... 30 on every date, and whose return on each date is that signal times ``signs[t]`` (so the rank
    information coefficient of the date is exactly +1 or -1). From the date ``thin_after`` on, only the first ``keep`` companies have a signal."""
    n = len(signs)
    dates = pd.date_range("2012-01-31", periods=n, freq="ME")
    signal = pd.DataFrame(np.tile(np.arange(1.0, 31.0), (n, 1)), index=dates, columns=range(30))
    forward = signal.mul(np.array(signs, float) * 0.01, axis=0)
    if thin_after is not None:
        signal.iloc[thin_after:, keep:] = np.nan
    ctx = SimpleNamespace(freq=freq, forward=lambda horizon: forward, investable=lambda: pd.DataFrame(True, index=dates, columns=range(30)))
    lib = L.AlphaLibrary(FundamentalsStore(tmp_path / f"watched_{freq}_{n}_{thin_after}_{keep}.db"))
    alpha = lib.add_candidate(X.parse("div(fcf,market_cap)"), {"freq": freq, "horizon": "1m", "min_assets": 10, **settings}, {"ic": 0.8})
    return lib, ctx, signal, alpha, dates


def test_the_monitor_compares_the_last_window_with_the_one_before_and_the_whole_past_and_counts_how_many_companies_it_still_covers(tmp_path, monkeypatch):
    signs = [1] * 16 + [1] * 6 + [-1] * 6 + [1] * 9 + [-1] * 3                                          # the last twelve dates average +0.5, the twelve before them 0
    lib, ctx, signal, alpha, dates = _watched(tmp_path, "M", signs, thin_after=len(signs) - 12)
    monkeypatch.setattr(L, "build_signal", lambda c, expression, settings: signal)
    out = lib.monitor(alpha, ctx)
    d = out["detail"]
    assert out["monitor"] == "Healthy" and out["as_of"] == dates[-1].strftime("%Y-%m-%d")
    assert (d["recent_window"], d["recent_ic"], d["original_ic"], d["two_year_ic"]) == (12, pytest.approx(0.5), 0.8, pytest.approx(0.25))
    assert d["ratio"] == pytest.approx(0.5 / 0.8) and (d["coverage_now"], d["coverage_before"]) == (12.0, 30.0)         # twelve of the thirty companies now, thirty before
    assert lib.monitor_history(alpha)[0]["coverage"] == 12.0 and lib.monitor(alpha, ctx, recent=6)["detail"]["recent_window"] == 6
    lib, ctx, signal, alpha, _ = _watched(tmp_path, "M", signs, thin_after=len(signs) - 12, keep=4)                                        # four companies now: under 30% of thirty
    monkeypatch.setattr(L, "build_signal", lambda c, expression, settings: signal)
    gone = lib.monitor(alpha, ctx)
    assert gone["monitor"] == "Broken" and gone["detail"]["coverage_now"] == 4.0 and gone["detail"]["coverage_before"] == 30.0


def test_the_monitor_looks_back_a_year_of_monthly_dates_a_year_of_weekly_ones_and_four_quarters_and_wants_twice_that_to_judge(tmp_path, monkeypatch):
    for freq, window in (("M", 12), ("W", 52), ("Q", 4)):
        lib, ctx, signal, alpha, _ = _watched(tmp_path, freq, [1] * (2 * window - 1))
        monkeypatch.setattr(L, "build_signal", lambda c, expression, settings, signal=signal: signal)
        short = lib.monitor(alpha, ctx)
        assert short["monitor"] == "Weakening" and short["detail"]["note"] == f"only {2 * window - 1} dates of IC: too few to compare a recent window with the past"
        lib, ctx, signal, alpha, _ = _watched(tmp_path, freq, [1] * (2 * window))
        monkeypatch.setattr(L, "build_signal", lambda c, expression, settings, signal=signal: signal)
        enough = lib.monitor(alpha, ctx)
        assert enough["monitor"] == "Healthy" and enough["detail"]["recent_window"] == window


def test_the_library_finds_the_same_idea_in_another_spelling_under_any_or_the_given_conditions_and_remembers_a_failure_once(tmp_path):
    lib = L.AlphaLibrary(FundamentalsStore(tmp_path / "lib.db"))
    monthly, quarterly = {"freq": "M", "horizon": "1m"}, {"freq": "M", "horizon": "3m"}
    kept = lib.add_candidate(X.parse("div(fcf,market_cap)"), monthly, {"oos_t": 1.5})
    assert lib.add_graveyard(X.parse("rank(div(net_income,market_cap))"), quarterly, "too weak", "fast_ic") is True
    assert lib.add_graveyard(X.parse("rank(div(net_income,market_cap))"), quarterly, "a second reason", "stability") is False        # nothing is overwritten
    twin = X.parse("neg(rank(div(fcf,market_cap)))")                                                                                  # a sign and a rank on the outside change nothing
    assert [(r["where"], r["alpha_id"]) for r in lib.same_idea(twin)] == [("library", kept)]
    assert len(lib.same_idea(twin, L.settings_hash(monthly))) == 1 and lib.same_idea(twin, L.settings_hash(quarterly)) == []        # under these conditions only, when they are given
    gone = lib.same_idea(X.parse("div(net_income,market_cap)"))
    assert [(r["where"], r["stage"], r["reason"]) for r in gone] == [("graveyard", "fast_ic", "too weak")]
    assert lib.same_idea(X.parse("div(net_income,market_cap)"), L.settings_hash(monthly)) == [] and lib.same_idea(X.parse("div(revenue,market_cap)")) == []
    assert lib.seen(X.expression_hash(X.parse("div(fcf,market_cap)")), L.settings_hash(monthly)) == {"where": "library", "alpha_id": kept, "status": "Candidate"}
    assert lib.seen(X.expression_hash(X.parse("div(fcf,market_cap)")), L.settings_hash(quarterly)) is None
    failed = lib.seen(X.expression_hash(X.parse("rank(div(net_income,market_cap))")), L.settings_hash(quarterly))
    assert failed["where"] == "graveyard" and failed["reason"] == "too weak" and failed["stage"] == "fast_ic"


# ------------------------------------------------------------------------------------------------------------------------ the credit quota
def test_the_allowance_comes_from_the_environment_then_the_plan_then_the_free_plan_and_is_not_changed_afterwards(monkeypatch):
    for name in ("OPENFILINGS_DAILY_LIMIT", "OPENFILINGS_MONTHLY_LIMIT", "OPENFILINGS_PLAN", "SIMULATED_DAILY_LIMIT", "SIMULATED_MONTHLY_LIMIT", "YAHOO_DAILY_LIMIT", "YAHOO_MONTHLY_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    free = allowance("openfilings")
    assert (free.unit, free.daily, free.monthly) == ("credits", 100, 1000) and "assumed" in free.source
    pro = allowance("openfilings", "PRO")
    assert (pro.daily, pro.monthly) == (2500, 25000) and "reported by the account" in pro.source
    assert allowance("openfilings", "no such plan").daily == 100
    monkeypatch.setenv("OPENFILINGS_PLAN", "business")
    business = allowance("openfilings", "free")
    assert (business.daily, business.monthly) == (10000, 100000) and "set by OPENFILINGS_PLAN" in business.source                     # what the owner says beats what the account reported
    monkeypatch.setenv("OPENFILINGS_DAILY_LIMIT", "7")
    own = allowance("openfilings")
    assert (own.daily, own.monthly, own.source) == (7, 100000, "environment limits")
    monkeypatch.setenv("OPENFILINGS_MONTHLY_LIMIT", "0")
    assert allowance("openfilings").monthly is None                                                                                    # zero says there is no limit
    monkeypatch.setenv("OPENFILINGS_DAILY_LIMIT", "seven")
    assert allowance("openfilings").daily == 10000                                                                                     # words are not a limit: the plan's daily allowance stands
    simulated = allowance("simulated")
    assert (simulated.unit, simulated.daily, simulated.monthly, simulated.source) == ("requests", 1_000_000, None, "none")
    monkeypatch.setenv("SIMULATED_DAILY_LIMIT", "5")
    assert (allowance("simulated").daily, allowance("simulated").source) == (5, "environment limits")
    other = allowance("yahoo")
    assert (other.unit, other.daily, other.monthly, other.source) == ("requests", None, None, "none")
    monkeypatch.setenv("YAHOO_MONTHLY_LIMIT", "40")
    assert (allowance("yahoo").daily, allowance("yahoo").monthly, allowance("yahoo").source) == (None, 40, "environment limits")
    with pytest.raises(FrozenInstanceError):
        free.daily = 1


def test_the_quota_counts_by_the_utc_day_and_month_and_a_refusal_ends_when_its_day_or_month_does(tmp_path):
    clock = {"now": datetime(2026, 10, 3, 9, 30, tzinfo=timezone.utc)}
    usage = UsageLog(FundamentalsStore(tmp_path / "u.db").db, lambda: clock["now"])
    assert (usage.today(), usage.month_start(), usage.next_day(), usage.next_month()) == ("2026-10-03", "2026-10-01", "2026-10-04 00:00 UTC", "2026-11-01 00:00 UTC")
    for now, tomorrow, next_month in ((datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc), "2027-01-01 00:00 UTC", "2027-01-01 00:00 UTC"),
                                      (datetime(2028, 2, 29, 12, 0, tzinfo=timezone.utc), "2028-03-01 00:00 UTC", "2028-03-01 00:00 UTC"),
                                      (datetime(2026, 1, 31, 0, 0, tzinfo=timezone.utc), "2026-02-01 00:00 UTC", "2026-02-01 00:00 UTC")):
        clock["now"] = now
        assert (usage.next_day(), usage.next_month()) == (tomorrow, next_month)
    clock["now"] = datetime(2026, 10, 3, 9, 30, tzinfo=timezone.utc)
    usage.record("openfilings", "financials", True, key_id="k1", credits=10)
    usage.record("openfilings", "search", False, key_id="k1", error="limit:month the monthly allowance is spent")
    usage.record("openfilings", "financials", True, key_id="k2", credits=4, requests=2)
    assert (usage.used_today("openfilings"), usage.used_month("openfilings"), usage.requests_today("openfilings")) == (14, 14, 4)         # a failed call costs nothing, a request counts as one unless it says more
    assert usage.used_today("openfilings", "k1") == 10 and usage.used_today("openfilings", "k2") == 4 and usage.used_today("itick") == 0
    clock["now"] = datetime(2026, 10, 21, 9, 30, tzinfo=timezone.utc)                                                                    # later in the month: the day's count starts again, the month's goes on
    assert (usage.used_today("openfilings"), usage.used_month("openfilings"), usage.requests_today("openfilings")) == (0, 14, 0)
    month = usage.blocked("openfilings", "k1")
    assert month.kind == "month" and month.until == "2026-11-01 00:00 UTC" and month.message.startswith("limit:month")                   # the refusal of the third still stands on the twenty-first
    assert usage.blocked("openfilings", "k2") is None and usage.blocked("openfilings", "nobody") is None
    clock["now"] = datetime(2026, 11, 1, 0, 5, tzinfo=timezone.utc)
    assert usage.blocked("openfilings", "k1") is None and usage.used_month("openfilings") == 0
    usage.record("openfilings", "search", False, key_id="k3", error="limit:day the daily allowance is spent")
    assert usage.blocked("openfilings", "k3").kind == "day" and usage.blocked("openfilings", "k3").until == "2026-11-02 00:00 UTC"
    clock["now"] = datetime(2026, 11, 1, 23, 59, tzinfo=timezone.utc)
    assert usage.blocked("openfilings", "k3").kind == "day"
    clock["now"] = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    assert usage.blocked("openfilings", "k3") is None                                                                                    # the day rolled over
    usage.record("openfilings", "search", False, key_id="k4", error="balance no credits left")
    usage.record("openfilings", "search", False, key_id="k5", error="auth invalid key")
    usage.record("openfilings", "search", False, key_id="k6", error="a timeout")
    assert (usage.blocked("openfilings", "k4").kind, usage.blocked("openfilings", "k4").until) == ("balance", None) and usage.blocked("openfilings", "k5").kind == "auth" and usage.blocked("openfilings", "k6") is None
    usage.record("openfilings", "search", True, key_id="k4", credits=1)
    assert usage.blocked("openfilings", "k4") is None                                                                                    # one call that works lifts it at once
    clock["now"] = datetime(2026, 11, 3, 0, 0, tzinfo=timezone.utc)
    assert usage.blocked("openfilings", "k5") is None                                                                                    # a bad key is a problem of the day it was found


def test_the_usage_summary_the_history_and_the_activity_log_say_what_was_spent_and_when(tmp_path):
    clock = {"now": datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)}
    usage = UsageLog(FundamentalsStore(tmp_path / "u.db").db, lambda: clock["now"])
    for day in (19, 20, 21):
        clock["now"] = datetime(2026, 10, day, 9, 0, tzinfo=timezone.utc)
        usage.record("openfilings", "financials", True, key_id="k1", key_slot="key 1", credits=10, symbol=f"us:A{day}")
    usage.record("openfilings", "search", False, key_id="k1", key_slot="key 1", error="a timeout", symbol="us:ZZZ")
    summary = usage.summary("openfilings", [{"slot": "key 1", "key_id": "k1", "state": "ready", "role": "active"}], "pro")
    assert (summary.requests, summary.successful, summary.failed, summary.used_today, summary.used_month) == (2, 1, 1, 10, 30)
    assert (summary.daily_limit, summary.monthly_limit, summary.unit) == (2500, 25000, "credits")
    assert summary.keys == ({"slot": "key 1", "used_today": 10, "used_month": 30, "state": "ready", "role": "active"},)
    assert [(h["date"], h["requests"], h["ok"], h["credits"]) for h in usage.history("openfilings")] == [("2026-10-19", 1, 1, 10), ("2026-10-20", 1, 1, 10), ("2026-10-21", 2, 1, 10)]
    assert [h["date"] for h in usage.history("openfilings", days=2)] == ["2026-10-20", "2026-10-21"]
    assert usage.last_update("openfilings") is not None and usage.last_update("itick") is None
    usage.record("simulated")                                                                                                     # a call with nothing said about it is one successful request
    assert usage.used_today("simulated") == 1 and usage.requests_today("simulated") == 1 and usage.summary("simulated").successful == 1
    for field, value in (("daily_limit", 1), ("used_today", 1)):
        with pytest.raises(FrozenInstanceError):
            setattr(summary, field, value)
    usage.record("openfilings", "x", False, key_id="kb", error="balance none left")
    block = usage.blocked("openfilings", "kb")
    with pytest.raises(FrozenInstanceError):
        block.kind = "day"
    for i in range(39):
        usage.record("simulated", "x", True, symbol=f"S{i}")
    assert len(usage.recent("simulated")) == 30 and usage.recent("simulated")[0]["symbol"] == "S38" and len(usage.recent("simulated", 5)) == 5      # newest first, thirty by default
    usage.record("simulated", "x", False, error="e" * 400)
    assert usage.recent("simulated", 1)[0]["error"] == "e" * 300                                                                       # a long message is kept to 300 characters


def _keys(tmp_path, keys=("of_first_key_aaaa",), daily=100, monthly=1000, name="k.db"):
    usage = UsageLog(FundamentalsStore(tmp_path / name).db, lambda: datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc))
    return KeyPool("openfilings", [(f"OPENFILINGS_API_KEY_{i + 1}", k) for i, k in enumerate(keys)], usage, "salt", Allowance("credits", daily, monthly, "test")), usage


def test_a_pool_can_make_a_request_if_it_holds_a_key_or_needs_none_and_says_how_much_a_call_may_still_spend(tmp_path):
    assert not KeyPool("openfilings", []).configured and KeyPool("openfilings", [("OPENFILINGS_API_KEY", "of_x")]).configured and KeyPool.without_key("simulated").configured
    pool, usage = _keys(tmp_path, daily=100)
    usage.record("openfilings", "financials", True, key_id=pool.slots()[0]["key_id"], credits=90)
    pool.ensure(10)                                                                                       # 90 spent and 10 asked is exactly the 100 there is: allowed
    assert pool.remaining_today() == 10 and pool.remaining_month() == 910
    with pytest.raises(QuotaExhausted) as short:
        pool.ensure(11)
    assert "this call needs 11, 10 left" in str(short.value) and "standbys" not in str(short.value)           # one key has no standbys to speak of
    usage.record("openfilings", "financials", True, key_id=pool.slots()[0]["key_id"], credits=10)
    with pytest.raises(QuotaExhausted) as spent:
        pool.ensure(1)
    assert "100 of 100 credits today" in str(spent.value) and pool.remaining_today() == 0
    two, _ = _keys(tmp_path, keys=("of_first_key_aaaa", "of_second_key_bbbb"), daily=5, name="two.db")
    with pytest.raises(QuotaExhausted) as several:
        two.ensure(10)
    assert "standbys" in str(several.value) and "does not rotate keys" in str(several.value)               # with spares the message says why they are not used


def test_a_burst_a_bad_key_and_a_spent_allowance_are_each_logged_as_what_they_were_and_only_the_last_two_stop_the_key(tmp_path):
    pool, usage = _keys(tmp_path, keys=("of_first_key_aaaa", "of_second_key_bbbb"))

    def refuse(error):
        def fn(key):
            raise error
        return fn

    with pytest.raises(RateLimited):
        pool.call(refuse(RateLimited("too many at once", dimension="burst")), "search", "us:AAA", credits=1)
    last = usage.recent("openfilings", 1)[0]
    assert last["success"] == 0 and last["credits_used"] == 0 and last["error"].startswith("burst:")                    # a failed call, free, and no reason to set the key aside
    assert [s["state"] for s in pool.slots()] == ["ready", "ready"] and pool.slots()[0]["role"] == "active"
    used = []
    pool.call(lambda key: used.append(key) or 1, "search", "us:AAA", credits=1)
    assert used == ["of_first_key_aaaa"]
    calls = []

    def bad_then_good(key):
        calls.append(key)
        if key == "of_first_key_aaaa":
            raise InvalidKey("revoked")
        return 7

    assert pool.call(bad_then_good, "search", "us:BBB", credits=1) == 7 and calls == ["of_first_key_aaaa", "of_second_key_bbbb"]
    rows = usage.recent("openfilings", 3)
    assert rows[1]["success"] == 0 and rows[1]["error"].startswith("auth:") and rows[1]["credits_used"] == 0 and rows[0]["success"] == 1 and rows[0]["credits_used"] == 1
    assert [(s["state"], s["role"]) for s in pool.slots()] == [("invalid", "unusable"), ("ready", "active")]
    with pytest.raises(QuotaExhausted):
        pool.call(refuse(RateLimited("daily allowance", dimension="daily")), "financials", "us:CCC", credits=10)
    assert usage.recent("openfilings", 1)[0]["error"].startswith("limit:day") and pool.slots()[1]["state"] == "spent_day"


def test_a_pool_whose_every_key_was_rejected_says_so_and_has_nothing_left_to_spend(tmp_path):
    pool, usage = _keys(tmp_path, keys=("of_only_key_cccc",), name="last.db")

    def revoked(key):
        raise InvalidKey("revoked")

    with pytest.raises(InvalidKey, match="every OPENFILINGS key was rejected"):
        pool.call(revoked, "search", "us:AAA", credits=1)
    assert pool.remaining_today() == 0 and pool.remaining_month() is None and [s["role"] for s in pool.slots()] == ["unusable"]
    with pytest.raises(InvalidKey):
        pool.ensure(1)
    assert usage.recent("openfilings", 1)[0]["error"].startswith("auth:")


def test_numbered_key_slots_are_read_up_to_the_twentieth_and_a_plan_error_is_logged_without_setting_the_key_aside(tmp_path, monkeypatch):
    for name in [f"OPENFILINGS_API_KEY_{i}" for i in range(1, MAX_NUMBERED + 3)] + ["OPENFILINGS_API_KEY", "OPENFILINGS_API_KEYS"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(f"OPENFILINGS_API_KEY_{MAX_NUMBERED}", "of_twentieth_key_1234")
    monkeypatch.setenv(f"OPENFILINGS_API_KEY_{MAX_NUMBERED + 1}", "of_twenty_first_key_5678")
    found, _ = discover("openfilings", tmp_path)
    assert MAX_NUMBERED == 20 and found == [(f"OPENFILINGS_API_KEY_{MAX_NUMBERED}", "of_twentieth_key_1234")]            # the twenty-first slot is not looked at
    pool, usage = _keys(tmp_path, name="plan.db")

    def not_on_the_plan(key):
        raise PlanError("this needs the pro plan", feature="screener", min_plan="pro")

    with pytest.raises(PlanError):
        pool.call(not_on_the_plan, "screen", "us:AAA", credits=10)
    last = usage.recent("openfilings", 1)[0]
    assert last["success"] == 0 and last["credits_used"] == 0 and last["error"].startswith("plan:")                       # a refusal of the plan is free and says what it was
    assert pool.slots()[0]["state"] == "ready" and pool.remaining_today() == 100                                        # and every key on the plan would say the same, so none is set aside


# ------------------------------------------------------------------------------------------------------------------------ what OpenFilings says, and what each answer means here
def _answer(status, detail=None, headers=None, text=None):
    import json as _json

    body = text if text is not None else ("" if detail is None else _json.dumps({"detail": detail}))
    return of.OpenFilingsProvider._interpret(status, headers or {}, body.encode(), "of_secret_key_1234")


def test_each_answer_of_openfilings_is_told_apart_by_its_status_and_what_it_says_and_never_repeats_the_key():
    assert _answer(200, text='{"a": 1}') == {"a": 1} and _answer(204, text='{"b": 2}') == {"b": 2} and _answer(299, text="[1]") == [1]
    for bad in ("", "not json", "   "):
        with pytest.raises(ProviderError, match="not JSON"):
            _answer(200, text=bad)
    with pytest.raises(SyncPending) as pending:
        _answer(202, {"retry_after": 90})
    assert pending.value.retry_after == 90.0
    with pytest.raises(SyncPending) as later:
        _answer(202, None, {"Retry-After": "45"}, text="{}")
    assert later.value.retry_after == 45.0
    with pytest.raises(SyncPending) as unsaid:
        _answer(202, None, {}, text="{}")
    assert unsaid.value.retry_after is None
    with pytest.raises(InvalidKey, match="HTTP 401"):
        _answer(401, "no")
    with pytest.raises(CreditsExhausted) as credits:
        _answer(402, {"credit_balance": 3}, {"X-Credits-Required": "10"})
    assert credits.value.balance == 3.0 and credits.value.required == 10.0
    with pytest.raises(CreditsExhausted) as headers_only:
        _answer(402, None, {"x-credits-balance": "0.5", "x-credits-required": "1"}, text="{}")
    assert headers_only.value.balance == 0.5 and headers_only.value.required == 1.0
    with pytest.raises(PlanError) as upgrade:
        _answer(403, {"error": "upgrade_required", "feature": "screener", "min_plan": "pro"})
    assert (upgrade.value.feature, upgrade.value.min_plan) == ("screener", "pro") and "needs the pro plan" in str(upgrade.value) and "does not include screener" in str(upgrade.value)
    with pytest.raises(PlanError, match=r"refused the request \(HTTP 403\): not for you"):
        _answer(403, {"message": "not for you"})
    with pytest.raises(NotFound, match="nothing for this request: no such company"):
        _answer(404, {"error": "no such company"})
    with pytest.raises(NotFound, match="nothing for this request: plain text"):
        _answer(404, "plain text")
    for dimension, kind in (("daily", "daily"), ("per day", "daily"), ("monthly", "monthly"), ("month", "monthly"), ("minute", "burst"), ("", "daily")):
        body = {"error": "quota_exceeded", "dimension": dimension, "daily_limit": 100, "daily_used": 100, "monthly_remaining": 900.0, "ignored": "text"}
        with pytest.raises(RateLimited) as limited:
            _answer(429, body, {"Retry-After": "7"})
        assert limited.value.dimension == kind and limited.value.retry_after == 7.0
        if kind != "burst":
            assert f"{kind} allowance is spent" in str(limited.value) and "daily limit 100" in str(limited.value) and "monthly remaining 900" in str(limited.value) and "ignored" not in str(limited.value)
            assert limited.value.detail == {"daily_limit": 100, "daily_used": 100, "monthly_remaining": 900.0}
    with pytest.raises(RateLimited) as slow:
        _answer(429, "slow down")
    assert slow.value.dimension == "burst" and slow.value.retry_after is None
    for status in (400, 422):
        with pytest.raises(ProviderError, match=rf"rejected the request \(HTTP {status}\): bad ticker"):
            _answer(status, {"message": "bad ticker"})
    for status in (500, 502, 503):
        with pytest.raises(ProviderError, match=rf"temporarily unavailable \(HTTP {status}\)"):
            _answer(status, "boom")
    with pytest.raises(ProviderError, match=r"answered HTTP 418: " + "x" * 120 + "$"):
        _answer(418, "x" * 300)
    with pytest.raises(ProviderError) as leaked:
        _answer(400, {"message": "your key of_secret_key_1234 is not allowed here"})
    assert "of_secret_key_1234" not in str(leaked.value) and "your key" in str(leaked.value)
    with pytest.raises(ProviderError) as cut:
        _answer(400, {"message": "y" * 500})
    assert str(cut.value).endswith("y" * 200) and "y" * 201 not in str(cut.value)


def test_a_search_answer_becomes_one_row_per_company_under_the_listing_asked_for_or_the_primary_one():
    nvidia = {"display_names": ["NVIDIA Corporation", "NVIDIA"], "isin": "US67066G1040", "lei": "549300S4KLFTLO7GSQ80",
              "listings": [{"market_id": "UK", "ticker": " nvda ", "company_name": "", "score": "high", "registry": "x" * 60, "hint": "h" * 300},
                           {"market_id": "us", "ticker": "nvda", "company_name": "NVIDIA Corp", "primary": True, "score": 0.97, "isin": "US67066G1040"},
                           {"market_id": "de", "ticker": "nvd"}, {"market_id": "us"}, {"ticker": "x"}, "not a listing"]}
    for market, symbol in ((None, "us:NVDA"), ("us", "us:NVDA"), ("uk", "uk:NVDA")):
        (row,) = of._search_profiles({"entity": nvidia}, market)
        assert row.provider_symbol == symbol
    (primary,) = of._search_profiles({"entity": nvidia})
    assert (primary.name, primary.isin, primary.country, primary.exchange, primary.extra["primary"], primary.extra["score"]) == ("NVIDIA Corp", "US67066G1040", "US", "US", True, 0.97)
    assert primary.extra["also_listed"] == ["uk:NVDA", "de:NVD"] and primary.extra["has_financials"] is True                          # the other listings, so the same company is not added twice
    (uk,) = of._search_profiles({"entity": nvidia}, "uk")
    assert uk.name == "NVIDIA Corporation" and uk.country == "GB" and uk.extra["score"] is None and uk.extra["primary"] is False         # a name from the list when the listing has none; "uk" is Great Britain
    assert len(uk.extra["registry"]) == 40 and len(uk.extra["hint"]) == 200 and uk.extra["also_listed"] == ["us:NVDA", "de:NVD"]
    many = {"entity": {"listings": [{"market_id": "us", "ticker": f"T{i}", "primary": i == 0} for i in range(10)]}}
    assert len(of._search_profiles(many)[0].extra["also_listed"]) == 6                                                              # no more than six others are kept
    several = of._search_profiles({"entity": {"listing": "jp:7203", "display_names": ["", "Toyota Motor"]},
                                   "entities": [{"entity": {"listings": [{"market_id": "zz", "ticker": "q1", "company_name": "C" * 300}], "isin": "XX1234567890"}},
                                                {"listings": [{"market_id": "jp", "ticker": "7203"}]},                                    # the same listing again
                                                {"listing": "nocolon"}, {"listings": []}, "junk", None, {"entity": "not a mapping"}]})
    assert [r.provider_symbol for r in several] == ["jp:7203", "zz:Q1"]
    assert several[0].name == "Toyota Motor" and several[0].country == "JP" and several[0].extra["primary"] is True                   # a listing given as a string is taken as the primary one
    assert several[1].extra["has_financials"] is False and len(several[1].name) == 200 and several[1].country == "ZZ"
    assert of._search_profiles({}) == [] and of._search_profiles({"entity": None, "entities": None}) == []
    eu = of._search_profiles({"entity": {"isin": "DE0007164600", "listings": [{"market_id": "eu", "ticker": "sap", "primary": True}]}})[0]
    assert eu.country == "DE" and of._country("eu", None) is None and of._country("eu", "1") is None and of._country("fr", None) == "FR"          # a pan-European listing has the country of its ISIN


def _provider(tmp_path, answers, name="of.db"):
    """An OpenFilings client whose transport gives the listed ``(status, headers, body)`` answers in turn and notes every address and every pause."""
    import json as _json

    pool, usage = _keys(tmp_path, name=name)
    asked, slept = [], []

    def transport(url, headers, timeout):
        asked.append((url, headers))
        status, extra, body = answers[min(len(asked) - 1, len(answers) - 1)]
        return status, extra, (body if isinstance(body, bytes) else _json.dumps(body).encode())

    return of.OpenFilingsProvider(pool, transport=transport, sleep=slept.append), asked, slept, usage


def test_a_financials_request_asks_for_the_years_the_caller_named_or_the_default_history_and_only_annual_ones(tmp_path):
    from urllib import parse as urlparse

    page = {"entity": {"market_id": "us", "ticker": "AAA", "name": "A Co"}, "currency_latest": "USD",
            "periods": [{"fiscal_period": "FY", "fiscal_year": 2020, "period_end": "2020-12-31", "filing_date": "2021-02-20", "values": {"revenue": 10.0}}]}
    provider, asked, _, _ = _provider(tmp_path, [(200, {}, page)])
    this_year = datetime.now(timezone.utc).year

    def params():
        return dict(urlparse.parse_qsl(urlparse.urlparse(asked[-1][0]).query))

    provider.get_financials("us:aaa")
    assert params()["fiscal_year_from"] == str(this_year - of.DEFAULT_HISTORY_YEARS) and of.DEFAULT_HISTORY_YEARS == 25
    assert "fiscal_year_to" not in params() and params()["fiscal_period"] == "fy" and params()["ticker"] == "aaa" and params()["market_id"] == "us"
    provider.get_financials("us:aaa", limit=3)
    assert params()["fiscal_year_from"] == str(this_year - 3)
    provider.get_financials("us:aaa", start_year=2005, end_year=2012, limit=3)
    assert params()["fiscal_year_from"] == "2005" and params()["fiscal_year_to"] == "2012"                              # the years named win over a count of years
    assert asked[-1][1]["X-API-Key"] == "of_first_key_aaaa" and asked[-1][1]["Accept"] == "application/json"
    result = provider.get_financials("us:aaa", start_year=2010)
    assert result.cost == 10 and result.requests == 1 and result.exhausted is True and [p.fiscal_year for p in result.periods] == [2020]      # nothing before 2020 came back for a start in 2010
    assert provider.get_financials("us:aaa", start_year=2020).exhausted is False
    with pytest.raises(PlanError, match="quarterly figures are not supported"):
        provider.get_financials("us:aaa", period="quarter")
    assert len(asked) == 5                                                                                              # the refused request spent nothing and asked nothing


def test_a_burst_of_requests_is_waited_out_twice_for_the_time_asked_or_a_growing_pause_and_never_longer_than_five_seconds(tmp_path):
    ok = (200, {}, {"entity": {}, "periods": []})
    slow = (429, {"Retry-After": "3"}, {"detail": "slow down"})
    unsaid = (429, {}, {"detail": "slow down"})
    for answers, pauses in (([slow, ok], [3.0]), ([unsaid, unsaid, ok], [2.0, 4.0]), ([(429, {"Retry-After": "99"}, "x"), ok], [5.0]), ([(429, {"Retry-After": "0.5"}, "x"), ok], [0.5])):
        provider, asked, slept, _ = _provider(tmp_path, answers, name=f"b{len(answers)}{pauses[0]}.db")
        provider._get("/api/v1/markets", {}, "markets")
        assert slept == pauses and len(asked) == len(answers)
    provider, asked, slept, usage = _provider(tmp_path, [unsaid], name="three.db")
    with pytest.raises(RateLimited) as burst:
        provider._get("/api/v1/markets", {}, "markets")
    assert burst.value.dimension == "burst" and slept == [2.0, 4.0] and len(asked) == of.BURST_RETRIES + 1              # two pauses, three tries, then it says so
    assert usage.used_today("openfilings") == 0 and usage.recent("openfilings", 1)[0]["error"].startswith("burst:")


def test_the_shape_of_a_search_text_decides_ticker_or_name_at_five_capitals_and_the_other_reading_is_offered_up_to_eight_letters():
    plan = of.search_plan
    assert plan("GOOGL")["kind"] == "symbol" and plan("GOOGLE")["kind"] == "name" and plan("Googl")["kind"] == "name"               # five capitals are a ticker, six letters or any lower case a name
    assert plan("Zzyzxabc")["alternatives"] == ["symbol"] and plan("Zzyzxabcd")["alternatives"] == [] and plan("Zzyzx Corp")["alternatives"] == []     # a single word of eight letters could be a ticker, nine could not
    forced = plan("us:NVDA", kind="symbol")
    assert (forced["kind"], forced["market"], forced["value"], forced["params"]) == ("symbol", "us", "NVDA", {"ticker": "NVDA", "market_id": "us"})   # a market written before a ticker is the market, also when the kind was said
    assert plan("nvda", kind="symbol", market="jp")["market"] == "jp" and plan("jp:7203", kind="symbol", market="us")["market"] == "jp"                  # what is written in the text wins over the box


def test_the_country_of_a_company_comes_from_its_market_or_for_the_pan_european_one_from_the_first_two_letters_of_its_isin():
    assert [of._country("eu", i) for i in ("DE0005140008", "US", "D", "1234567", "", None)] == ["DE", "US", None, None, None, None]
    assert of._country("uk", "FR0000000000") == "GB" and of._country("jp", None) == "JP" and of._country("eu", "de0005140008") == "DE"


def test_a_provider_with_no_pool_or_a_pool_without_a_key_cannot_be_asked_and_says_where_to_put_one(tmp_path):
    def never(url, headers, timeout):
        raise AssertionError("a request was sent")

    for pool in (None, _keys(tmp_path, keys=())[0]):
        provider = of.OpenFilingsProvider(pool, transport=never)
        ok, why = provider.available()
        assert ok is False and "no OPENFILINGS_API_KEY is set" in why and "Tools > Secrets" in why
        with pytest.raises(MissingKey, match="no OPENFILINGS_API_KEY is set"):
            provider.get_company("us:AAPL")
    assert of.OpenFilingsProvider(_keys(tmp_path)[0], transport=never).available() == (True, "")


def test_the_account_test_reads_the_markets_it_is_told_and_nothing_else_and_a_search_is_cut_at_ten_by_default(tmp_path):
    answer = {"plan": "PRO", "markets": [{"id": "us"}, "junk", {"label": "no id"}, {"id": ""}, {"id": "jp"}], "allowed_market_ids": [], "all_markets": 1}
    provider, _, _, _ = _provider(tmp_path, [(200, {}, answer)], name="markets.db")
    tested = provider.test()
    assert tested["markets"] == ["us", "jp"] and tested["allowed_markets"] == ["us", "jp"] and tested["plan"] == "pro" and tested["all_markets"] is True and tested["with_filings"] == ["us", "jp"]
    twelve = {"entity": {"display_names": ["Zed Corp 0"], "listings": [{"market_id": "us", "ticker": "ZED0", "primary": True}]},
              "entities": [{"display_names": [f"Zed Corp {i}"], "listings": [{"market_id": "us", "ticker": f"ZED{i}", "primary": True}]} for i in range(1, 12)]}
    searcher, _, _, _ = _provider(tmp_path, [(200, {}, twelve)], name="search.db")
    assert len(searcher.search_detail("Zed Corp")["profiles"]) == 10 and len(searcher.search_detail("Zed Corp", limit=3)["profiles"]) == 3 and len(searcher.search("Zed Corp")) == 10


def test_a_company_profile_is_made_only_when_both_the_market_and_the_ticker_are_known():
    assert of.parse_financials({"periods": []}, "", "")[1] is None and of.parse_financials({"periods": []}, "us", "")[1] is None and of.parse_financials({"periods": []}, "", "aaa")[1] is None
    profile = of.parse_financials({"periods": [], "currency_latest": "USD"}, "us", "aaa")[1]
    assert profile.provider_symbol == "us:AAA" and profile.currency == "USD" and profile.country == "US"
    named = of.parse_financials({"entity": {"market_id": "JP", "ticker": "7203", "name": "Toyota"}, "periods": [], "isin": "JP3633400001"}, "", "")[1]
    assert (named.provider_symbol, named.name, named.isin, named.country) == ("jp:7203", "Toyota", "JP3633400001", "JP")
