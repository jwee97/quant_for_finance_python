"""What the futures and crypto result pages draw from a pipeline result (``assets.strategy_runs``): the risk views are worked out again here with plain loops from a result whose answers are known."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.assets import strategy_runs as sr
from src.risk import performance as rs


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2021-01-04", periods=500)
    net = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    weights = pd.DataFrame({"A": rng.uniform(-1.0, 1.0, 500), "B": rng.uniform(0.0, 0.5, 500), "C": 0.0}, index=idx)
    bench = pd.Series(0.5 * net.to_numpy() + rng.normal(0.0001, 0.008, 500), index=idx)
    table = pd.DataFrame({"contribution_pct": [4.5, -1.25, 0.0], "average_weight": [0.31, 0.05, 0.0], "asset_class": ["metal", "commodity", "rates"]}, index=["A", "B", "C"])
    result = SimpleNamespace(benchmarks={"equal_weight": bench}, tables={"attribution_by_asset": table}, weights=weights)
    sampled = net.iloc[::5].index
    return result, net, sampled, sr._risk_views(result, net, sampled)


def test_the_rolling_volatility_is_the_annualised_deviation_of_the_last_63_days_at_the_dates_drawn(case):
    _, net, sampled, views = case
    got = views["rolling_vol"]
    assert len(got) == len(sampled)
    for date, value in zip(sampled, got):
        pos = net.index.get_loc(date)
        if pos < 62:
            assert value is None
        else:
            window = net.iloc[pos - 62:pos + 1].to_numpy()
            assert value == pytest.approx(window.std(ddof=1) * math.sqrt(252.0), abs=6e-5)           # rounded to four decimals


def test_the_weekly_exposure_is_the_last_trading_day_of_each_week(case):
    result, net, _, views = case
    gross = result.weights.abs().sum(axis=1)
    weeks = {}
    for date in net.index:
        end = (date + pd.Timedelta(days=(6 - date.weekday()))).normalize()                              # the Sunday that closes the week
        weeks[end] = date
    last_days = [weeks[k] for k in sorted(weeks)]
    assert views["exposure"]["dates"] == [str(k.date()) for k in sorted(weeks)]                         # a week is labelled by the Sunday that closes it, as on the Backtest page
    assert views["exposure"]["gross"] == pytest.approx([float(gross[d]) for d in last_days], abs=5e-4)
    assert views["exposure"]["net"] == pytest.approx([float(result.weights.loc[d].sum()) for d in last_days], abs=5e-4)


def test_the_positions_are_each_months_average_weight_of_the_busiest_assets_first(case):
    result, net, _, views = case
    pos = views["positions"]
    months = sorted({(d.year, d.month) for d in net.index})
    assert pos["months"] == [f"{y}-{m:02d}" for y, m in months]
    assert pos["assets"] == ["A", "B", "C"] and [a for a in pos["assets"]] == list(result.weights.abs().mean().sort_values(ascending=False).index)
    for asset, row in zip(pos["assets"], pos["values"]):
        for (y, m), value in zip(months, row):
            mask = [(d.year, d.month) == (y, m) for d in net.index]
            assert value == pytest.approx(float(result.weights[asset][mask].mean()), abs=6e-5)


def test_the_contribution_table_and_the_benchmark_relative_measures_are_those_of_their_sources(case):
    result, net, _, views = case
    assert views["contribution"] == [{"asset": "A", "contribution_pct": 4.5, "average_weight": 0.31, "asset_class": "metal"}, {"asset": "B", "contribution_pct": -1.25, "average_weight": 0.05, "asset_class": "commodity"},
                                     {"asset": "C", "contribution_pct": 0.0, "average_weight": 0.0, "asset_class": "rates"}]
    bench = result.benchmarks["equal_weight"]
    rel = views["benchmark_relative"]["equal_weight"]
    assert rel["beta"] == pytest.approx(np.cov(net, bench, ddof=1)[0, 1] / np.var(bench, ddof=1), rel=1e-12) and rel == rs.benchmark_relative(net, bench)


def test_the_risk_measures_the_months_and_the_distribution_are_those_of_the_performance_module(case):
    _, net, _, views = case
    assert views["risk"] == rs.risk_metrics(net) and views["drawdown_episodes"] == rs.drawdown_episodes(net, 5)
    assert views["calendar_months"] == rs.calendar_month_profile(net) and views["monthly"] == rs.monthly_returns(net) and views["distribution"] == rs.return_distribution(net)


def test_a_result_without_an_attribution_table_or_benchmarks_still_has_its_views():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=120)
    net = pd.Series(rng.normal(0.0, 0.01, 120), index=idx)
    result = SimpleNamespace(benchmarks={}, tables={}, weights=pd.DataFrame({"X": np.sign(rng.normal(size=120))}, index=idx))
    views = sr._risk_views(result, net, idx)
    assert views["contribution"] == [] and views["benchmark_relative"] == {} and views["positions"]["assets"] == ["X"] and len(views["rolling_vol"]) == 120
    assert all(v is None for v in views["rolling_vol"][:62]) and views["rolling_vol"][62] is not None
