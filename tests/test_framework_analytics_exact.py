"""Attribution, cost and capacity tables of a pipeline result, and the regime volatility target, checked against numbers worked out by hand.

``framework/analytics.py`` builds the tables the Backtest tab shows under "Where the return came from", "Costs" and "Capacity"; ``framework/risk.py`` sets the volatility target from a regime. Only a
pipeline run reached them before, each in minutes, so no planted-bug audit could be pointed at them. These build a small result by hand and state what each table holds.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import brinson_fachler, cumulative, monthly_inputs
from src.backtest.execution import vol_target_scaling
from src.backtest.impact import ImpactSettings, capacity_from_curve, sharpe
from src.framework.analytics import DEFAULT_GRID, brinson_vs_benchmark, capacity_by_aum, cost_breakdown, model_attribution, standalone_shares
from src.framework.risk import RegimeRiskPolicy, vol_target_series
from src.framework.types import RegimeSeries


def days(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ------------------------------------------------------------------------------------------------------------------ each model's share of the combined forecast
def panels() -> dict:
    index = days(3)
    return {"a": SimpleNamespace(mean=pd.DataFrame(0.0, index=index, columns=["X", "Y"]), confidence=pd.DataFrame([[0.2, 0.4], [0.6, 0.6], [np.nan, np.nan]], index=index, columns=["X", "Y"]),
                                 std=pd.DataFrame([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]], index=index, columns=["X", "Y"])),
            "b": SimpleNamespace(mean=pd.DataFrame(0.0, index=index, columns=["X", "Y"]), confidence=pd.DataFrame([[0.6, 0.6], [0.2, 0.2], [np.nan, np.nan]], index=index, columns=["X", "Y"]),
                                 std=pd.DataFrame([[0.5, 0.5], [1.0, 1.0], [np.inf, np.inf]], index=index, columns=["X", "Y"]))}


def test_a_models_share_is_one_over_the_number_of_models_or_its_confidence_or_its_precision_or_the_weight_the_rule_gave_it():
    p = panels()
    equal = standalone_shares(p, "equal", None)
    assert equal.to_numpy().tolist() == [[0.5, 0.5]] * 3 and list(equal.columns) == ["a", "b"] and equal.index.equals(p["a"].mean.index)
    confidence = standalone_shares(p, "confidence", None)
    assert confidence.iloc[0].tolist() == pytest.approx([0.3 / 0.9, 0.6 / 0.9]) and confidence.iloc[1].tolist() == pytest.approx([0.6 / 0.8, 0.2 / 0.8]), "each model's mean confidence over the sum of them"
    assert confidence.iloc[2].tolist() == [0.5, 0.5], "no confidence anywhere: an equal split, not a division by zero"
    precision = standalone_shares(p, "precision", None)
    assert precision.iloc[0].tolist() == pytest.approx([1.0 / 5.0, 4.0 / 5.0]), "mean of 1 / std^2: 1 and 4"
    assert precision.iloc[1].tolist() == pytest.approx([0.25 / 1.25, 1.0 / 1.25]) and precision.iloc[2].tolist() == [0.5, 0.5], "a standard deviation of zero is an infinite precision and is left out, as is infinity itself"
    given = pd.DataFrame({"a": [0.9, np.nan], "b": [0.1, np.nan]}, index=p["a"].mean.index[:2])
    shares = standalone_shares(p, "ic_weighted", given)
    assert shares.iloc[0].tolist() == pytest.approx([0.9, 0.1]) and shares.iloc[1].tolist() == pytest.approx([0.9, 0.1]), "a weight carried forward over a gap"
    assert shares.iloc[2].tolist() == pytest.approx([0.9, 0.1]), "and the last one held to the end"
    lead = pd.DataFrame({"a": [np.nan, 0.7, 0.7], "b": [np.nan, 0.3, 0.3]}, index=p["a"].mean.index)
    assert standalone_shares(p, "ic_weighted", lead).iloc[0].tolist() == [0.5, 0.5], "before the rule has a view: equal"
    with pytest.raises(ValueError, match="no share definition for combination rule 'optimal_ic'"):
        standalone_shares(p, "optimal_ic", None)


# ------------------------------------------------------------------------------------------------------------------ where the combined return came from
def test_each_models_contribution_is_its_share_the_day_before_times_its_standalone_return_and_the_rows_add_up_to_the_net_return():
    index = days(6)
    gross = pd.DataFrame({"a": [0.010, 0.020, -0.010, 0.000, 0.010, 0.020], "b": [0.000, -0.010, 0.030, 0.010, 0.000, 0.010]}, index=index)
    shares = pd.DataFrame({"a": [0.5, 0.5, 0.8, 0.8, 0.2, 0.2], "b": [0.5, 0.5, 0.2, 0.2, 0.8, 0.8]}, index=index)
    net = pd.Series([np.nan, 0.012, 0.010, 0.004, 0.006, 0.014], index=index)
    costs = pd.Series([0.0, 0.001, 0.002, 0.001, 0.000, 0.002], index=index)
    result = SimpleNamespace(net_returns=net, costs=costs, start=index[1])
    table = model_attribution(result, gross, shares)
    window = index[1:]                                                                   # the days from the start with a net return
    a = (shares["a"].shift(1).loc[window] * gross["a"].loc[window]).sum() * 100.0
    b = (shares["b"].shift(1).loc[window] * gross["b"].loc[window]).sum() * 100.0
    assert a == pytest.approx((0.5 * 0.02 + 0.5 * -0.01 + 0.8 * 0.0 + 0.8 * 0.01 + 0.2 * 0.02) * 100.0), "the share on the decision date earns the next day's return"
    assert list(table.index) == ["model: a", "model: b", "interaction (stack, risk scaling, limits)", "costs", "NET RETURN (sum of daily net returns)"] and list(table.columns) == ["return_points"]
    assert table.loc["model: a", "return_points"] == pytest.approx(a) and table.loc["model: b", "return_points"] == pytest.approx(b)
    net_points, cost_points = 100.0 * net.loc[window].sum(), 100.0 * costs.loc[window].sum()
    assert net_points == pytest.approx(4.6) and cost_points == pytest.approx(0.6)
    assert table.loc["costs", "return_points"] == pytest.approx(-0.6) and table.loc["NET RETURN (sum of daily net returns)", "return_points"] == pytest.approx(4.6)
    assert table.loc["interaction (stack, risk scaling, limits)", "return_points"] == pytest.approx(net_points + cost_points - a - b), "what the models alone do not explain: the stack, the risk scaling, the limits"
    assert table["return_points"].iloc[:-1].sum() == pytest.approx(net_points), "the rows add up to the net return exactly"


# ------------------------------------------------------------------------------------------------------------------ costs: spread and commission against impact
def test_the_cost_table_splits_the_annual_cost_into_the_linear_part_and_the_square_root_impact():
    index = days(4)
    trades = pd.DataFrame({"A": [0.1, 0.0, 0.0, 0.0]}, index=index)
    sigma, adv = pd.DataFrame(0.02, index=index, columns=["A"]), pd.DataFrame(1e9, index=index, columns=["A"])
    result = SimpleNamespace(net_returns=pd.Series([0.0, 0.001, 0.001, 0.001], index=index), trades=trades, start=index[0])
    table = cost_breakdown(result, pd.Series({"A": 0.0005}), sigma, adv, 1e8, ImpactSettings())
    linear_day, impact_day = 0.1 * 0.0005, 0.02 * np.sqrt(0.1 * 1e8 / 1e9) * 0.1                      # 0.00005 and 0.0002 on the day of the trade, nothing on the others
    assert list(table.index) == ["spread and commission (linear)", "market impact (square root)", "total"] and list(table.columns) == ["annual_bps", "mean_participation"]
    assert table.loc["spread and commission (linear)", "annual_bps"] == pytest.approx(1e4 * linear_day / 4 * 252) and table.loc["market impact (square root)", "annual_bps"] == pytest.approx(1e4 * impact_day / 4 * 252)
    assert table.loc["total", "annual_bps"] == pytest.approx(1e4 * (linear_day + impact_day) / 4 * 252) and table.loc["total", "annual_bps"] == pytest.approx(157.5)
    assert table.loc["total", "mean_participation"] == pytest.approx(0.01), "the share of the day's volume, over the days that traded"
    assert np.isnan(table.loc["spread and commission (linear)", "mean_participation"])
    quad = cost_breakdown(result, pd.Series({"A": 0.0005}), sigma, adv, 1e8, ImpactSettings(coefficient=2.0))
    assert quad.loc["market impact (square root)", "annual_bps"] == pytest.approx(2 * table.loc["market impact (square root)", "annual_bps"]) and quad.loc["spread and commission (linear)", "annual_bps"] == pytest.approx(31.5)
    late = cost_breakdown(SimpleNamespace(net_returns=pd.Series([np.nan, 0.001, 0.001, 0.001], index=index), trades=trades, start=index[1]), pd.Series({"A": 0.0005}), sigma, adv, 1e8, ImpactSettings())
    assert late.loc["total", "annual_bps"] == 0.0, "a trade before the evaluation window is not in it"


def stub_market(n: int = 160):
    rng = np.random.default_rng(3)
    index = days(n)
    cols = ["A", "B"]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)), index=index, columns=cols)
    return SimpleNamespace(prices=prices, returns=prices.pct_change(), volume=None, index=index), cols


def test_capacity_is_the_net_sharpe_at_each_size_and_falls_with_it_from_the_linear_cost_figure_at_a_tiny_size():
    bundle, cols = stub_market()
    index = bundle.index
    rng = np.random.default_rng(4)
    trades = pd.DataFrame(rng.normal(0, 0.05, (len(index), 2)), index=index, columns=cols)
    gross = pd.Series(rng.normal(0.0008, 0.004, len(index)), index=index)
    result = SimpleNamespace(gross_returns=gross, trades=trades, net_returns=gross.where(index > index[20]), start=index[21])
    engine = SimpleNamespace(cost_model=SimpleNamespace(rates=lambda columns: pd.Series({"A": 0.0004, "B": 0.0008}).reindex(columns)))
    grid = (1.0, 1e9, 1e10, 1e11, 1e12)
    curve, capacity = capacity_by_aum(result, bundle, engine, grid)
    assert list(curve.index) == [float(g) for g in grid] and curve.is_monotonic_decreasing, "the bigger the book the more its own trading costs"
    window = index[21:]
    linear_cost = (trades.loc[window].abs() * np.array([0.0004, 0.0008])[None, :]).sum(axis=1)
    linear = sharpe(gross.loc[window] - linear_cost)
    assert curve.iloc[0] == pytest.approx(linear, abs=1e-3), "a book of one dollar pays only the linear cost"
    assert capacity == pytest.approx(capacity_from_curve(curve, linear)) and isinstance(capacity, float) and 1e10 < capacity < 1e11, "the size at which the Sharpe falls to half its linear-cost value, between two points of the grid"
    assert DEFAULT_GRID == (1e7, 1e8, 1e9, 1e10, 1e11)
    harder = capacity_by_aum(result, bundle, engine, grid, coefficient=5.0)[0]
    assert (harder.iloc[1:] < curve.iloc[1:]).all() and harder.iloc[0] == pytest.approx(curve.iloc[0], abs=1e-3), "a larger impact coefficient lowers the curve at every size"
    liquid = SimpleNamespace(**{**bundle.__dict__, "volume": bundle.prices * 0.0 + 1e9})
    assert capacity_by_aum(result, liquid, engine, grid)[0].tolist() == pytest.approx(curve.tolist()), "with no volume in the data a volume of a billion a day is assumed"
    thin = SimpleNamespace(**{**bundle.__dict__, "volume": bundle.prices * 0.0 + 1e5})
    assert (capacity_by_aum(result, thin, engine, grid)[0].iloc[1:] < curve.iloc[1:]).all(), "thinner markets make the same book more expensive"


# ------------------------------------------------------------------------------------------------------------------ the regime volatility target
def regimes(index, low, high, crisis) -> RegimeSeries:
    return RegimeSeries(pd.DataFrame({"LowVol": low, "HighVol": high, "Crisis": crisis}, index=index))


def test_the_volatility_target_is_the_probability_weighted_average_of_the_targets_of_the_regimes():
    index = days(4)
    policy = RegimeRiskPolicy()
    assert policy.targets == {"LowVol": 0.10, "HighVol": 0.08, "Crisis": 0.05} and (policy.default_target, policy.lookback, policy.max_leverage) == (0.10, 63, 1.5)
    assert policy.target_series(None, index).tolist() == [0.10] * 4, "no regimes: the default"
    probs = regimes(index, [0.5, 1.0, 0.0, np.nan], [0.3, 0.0, 0.0, np.nan], [0.2, 0.0, 1.0, np.nan])
    target = policy.target_series(probs, index)
    assert target.iloc[0] == pytest.approx(0.5 * 0.10 + 0.3 * 0.08 + 0.2 * 0.05) and target.iloc[1] == pytest.approx(0.10) and target.iloc[2] == pytest.approx(0.05)
    assert target.iloc[3] == pytest.approx(0.10), "no view: the default"
    partial = regimes(index, [0.2] * 4, [0.2] * 4, [0.2] * 4)
    assert policy.target_series(partial, index).iloc[0] == pytest.approx(0.2 * 0.10 + 0.2 * 0.08 + 0.2 * 0.05 + 0.4 * 0.10), "what the view does not cover goes to the default"
    custom = RegimeRiskPolicy(targets={"LowVol": 0.20}, default_target=0.06)
    assert custom.target_series(probs, index).iloc[0] == pytest.approx(0.5 * 0.20 + 0.3 * 0.06 + 0.2 * 0.06), "a regime with no target of its own takes the default"
    shorter = policy.target_series(probs, days(6))
    assert len(shorter) == 6 and shorter.iloc[4:].tolist() == pytest.approx([0.10, 0.10]), "dates the detector never saw take the default"


def test_the_policy_scales_the_book_with_the_engines_own_volatility_targeting_and_returns_the_scalar_and_the_target():
    rng = np.random.default_rng(12)
    n = 200
    index = days(n)
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=index, columns=list("ABC"))
    weights = pd.DataFrame(1 / 3, index=index, columns=list("ABC"))
    probs = regimes(index, np.full(n, 0.6), np.full(n, 0.3), np.full(n, 0.1))
    policy = RegimeRiskPolicy(lookback=40, max_leverage=2.0)
    scaled, scalar, target = policy.apply(weights, returns, probs)
    assert target.iloc[0] == pytest.approx(0.6 * 0.10 + 0.3 * 0.08 + 0.1 * 0.05)
    expected, expected_scalar = vol_target_scaling(weights, returns, target, 40, 2.0)
    pd.testing.assert_frame_equal(scaled, expected)
    pd.testing.assert_series_equal(scalar, expected_scalar)
    quiet = returns * 0.3                                                                # a volatility of about 3%: the target of 10% needs three times the book, and the cap is what stops it
    default_scaled, default_scalar = vol_target_series(weights, quiet, 0.10)
    pd.testing.assert_frame_equal(default_scaled, vol_target_scaling(weights, quiet, 0.10, 63, 1.5)[0])
    pd.testing.assert_series_equal(default_scalar, vol_target_scaling(weights, quiet, 0.10, 63, 1.5)[1])
    assert default_scalar.max() == 1.5 and default_scalar.iloc[-1] == 1.5, "a cap of one and a half by default"
    constant, same = vol_target_series(weights, returns, 0.10, 40, 2.0)
    base, base_scalar = vol_target_scaling(weights, returns, 0.10, 40, 2.0)
    pd.testing.assert_frame_equal(constant, base)
    pd.testing.assert_series_equal(same, base_scalar)
    assert scalar.max() <= 2.0 and scalar.iloc[0] == 1.0, "never more than the cap, and 1 until there is history"
    unscaled, no_regime_scalar, default_target = policy.apply(weights, returns)
    assert default_target.tolist() == [0.10] * n and unscaled.shape == weights.shape
    pd.testing.assert_series_equal(no_regime_scalar, vol_target_scaling(weights, returns, 0.10, 40, 2.0)[1], check_names=False)


# ------------------------------------------------------------------------------------------------------------------ Brinson-Fachler against equal weight
def brinson_fixture(n: int = 520, held_equal: bool = True):
    rng = np.random.default_rng(31)
    index = days(n)
    cols = ["A", "B", "C", "D"]
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n, 4)), index=index, columns=cols)
    investable = pd.DataFrame(True, index=index, columns=cols)
    bench = pd.DataFrame(0.25, index=index, columns=cols)
    weights = bench.copy() if held_equal else pd.DataFrame([[0.4, 0.4, 0.1, 0.1]] * n, index=index, columns=cols)
    gross = (weights.shift(1) * returns).sum(axis=1)
    bundle = SimpleNamespace(index=index, investable=investable, returns=returns, assets=cols, asset_class={"A": "equity", "B": "equity", "C": "bonds", "D": "bonds"})
    result = SimpleNamespace(weights=weights, gross_returns=gross, start=index[0])
    return result, bundle


def test_a_book_that_is_the_benchmark_has_no_active_effect_and_the_table_has_a_row_for_cash_and_the_reconciliation():
    result, bundle = brinson_fixture()
    table = brinson_vs_benchmark(result, bundle)
    assert set(table.index) >= {"equity", "bonds", "cash / leverage", "TOTAL (cumulative active return, points)"}
    assert "reconciliation (daily drift and rebalancing vs start-of-month weights)" in table.index
    effects = table.loc[["equity", "bonds", "cash / leverage"], ["allocation", "selection", "interaction"]]
    assert np.abs(effects.to_numpy()).max() < 1e-9, "the same weights as the benchmark: nothing to attribute"
    body = table.drop(index="TOTAL (cumulative active return, points)")
    assert table.loc["TOTAL (cumulative active return, points)"].tolist() == pytest.approx(body.sum(axis=0).tolist()), "the last row is the sum of the others"
    reconciliation = table.loc["reconciliation (daily drift and rebalancing vs start-of-month weights)"]
    assert reconciliation["allocation"] == 0.0 and reconciliation["selection"] == 0.0 and reconciliation["interaction"] == 0.0


def test_a_tilted_book_has_an_active_effect_that_adds_up_to_the_active_return_and_a_short_history_has_no_table():
    result, bundle = brinson_fixture(held_equal=False)
    table = brinson_vs_benchmark(result, bundle)
    total = table.loc["TOTAL (cumulative active return, points)", "total"]
    rows = table.drop(index="TOTAL (cumulative active return, points)")
    assert rows["total"].sum() == pytest.approx(total) and abs(rows.loc[["equity", "bonds"], "allocation"]).max() > 0.0, "an overweight in equities is an allocation decision"
    assert (rows[["allocation", "selection", "interaction"]].sum(axis=1) - rows["total"]).abs().iloc[:-1].max() < 1e-9, "each class's effects add up to its row"
    short, small = brinson_fixture(n=200)
    assert brinson_vs_benchmark(short, small).empty, "fewer than twelve full months: no table"


def independent_brinson(result, bundle, since):
    """The linked effects and the reconciliation, rebuilt from the start-of-month weights and the monthly returns."""
    held = result.weights.reindex(bundle.index)
    bench_held = pd.DataFrame(0.25, index=bundle.index, columns=bundle.assets)
    a = monthly_inputs(held, bundle.returns, result.gross_returns)
    b = monthly_inputs(bench_held, bundle.returns)
    months = [m for m in a["weights"].index if m >= pd.Period(since, "M") + 1 and a["weights"].loc[m].notna().any()]
    wp, wb, ret = a["weights"].loc[months].copy(), b["weights"].loc[months].copy(), a["asset_returns"].loc[months].copy()
    wp["CASH"], wb["CASH"], ret["CASH"] = 1.0 - wp.sum(axis=1), 1.0 - wb.sum(axis=1), 0.0
    sector_of = {c: bundle.asset_class[c] for c in bundle.assets}
    sector_of["CASH"] = "cash / leverage"
    br = brinson_fachler(wp, wb, ret, sector_of)
    return br.linked() * 100.0, 100.0 * (cumulative(a["portfolio_return"].loc[months]) - cumulative(br.portfolio_return)), months


def test_the_table_is_in_return_points_from_the_first_full_month_after_the_start_and_reconciles_with_the_engines_own_return():
    result, bundle = brinson_fixture(held_equal=False)
    table = brinson_vs_benchmark(result, bundle)
    effects, reconciliation, months = independent_brinson(result, bundle, result.start)
    assert months[0] == pd.Period("2021-02", "M"), "the start is in January: the first full month is February"
    for sector in ("equity", "bonds", "cash / leverage"):
        assert table.loc[sector, ["allocation", "selection", "interaction", "total"]].tolist() == pytest.approx(effects.loc[sector, ["allocation", "selection", "interaction", "total"]].tolist(), abs=1e-8)
    assert table.loc["reconciliation (daily drift and rebalancing vs start-of-month weights)", "total"] == pytest.approx(reconciliation, abs=1e-8)
    assert table.loc["TOTAL (cumulative active return, points)", "total"] == pytest.approx(effects["total"].sum() + reconciliation, abs=1e-8)
    later = SimpleNamespace(weights=result.weights, gross_returns=result.gross_returns, start=bundle.index[130])
    table_later = brinson_vs_benchmark(later, bundle)
    effects_later, reconciliation_later, months_later = independent_brinson(later, bundle, later.start)
    assert months_later[0] == pd.Period(bundle.index[130], "M") + 1 and len(months_later) < len(months)
    assert table_later.loc["TOTAL (cumulative active return, points)", "total"] == pytest.approx(effects_later["total"].sum() + reconciliation_later, abs=1e-8) and not np.isclose(table_later.loc["equity", "allocation"], table.loc["equity", "allocation"])


def test_twelve_full_months_after_the_start_are_the_least_that_make_a_table():
    eleven, bundle = brinson_fixture(n=len(pd.bdate_range("2021-01-04", "2021-12-31")))
    assert brinson_vs_benchmark(eleven, bundle).empty, "February to December is eleven months"
    twelve, bundle = brinson_fixture(n=len(pd.bdate_range("2021-01-04", "2021-12-31")) + 1)
    assert not brinson_vs_benchmark(twelve, bundle).empty, "and the first day of January makes a twelfth"


def test_a_day_the_table_of_what_can_be_held_does_not_cover_is_not_a_day_anything_can_be_held():
    result, full_bundle = brinson_fixture(held_equal=False)
    full = brinson_vs_benchmark(result, full_bundle)
    half_bundle = brinson_fixture(held_equal=False)[1]
    half_bundle.investable = half_bundle.investable.iloc[: len(half_bundle.index) // 2]     # nothing recorded for the second year: the benchmark holds nothing there
    half = brinson_vs_benchmark(result, half_bundle)
    assert not half.empty and abs(half.loc["equity", "allocation"]) < 0.7 * abs(full.loc["equity", "allocation"]), "February of the first year to the January after: twelve months of the twenty-three"
    quarter_bundle = brinson_fixture(held_equal=False)[1]
    quarter_bundle.investable = quarter_bundle.investable.iloc[: int(len(quarter_bundle.index) * 0.75)]
    quarter = brinson_vs_benchmark(result, quarter_bundle)
    assert abs(half.loc["equity", "allocation"]) < abs(quarter.loc["equity", "allocation"]) < abs(full.loc["equity", "allocation"])


def test_a_month_for_which_the_book_has_no_weights_is_left_out_not_counted_as_cash():
    result, bundle = brinson_fixture(held_equal=False)
    march_end = bundle.index[bundle.index <= "2021-03-31"][-1]
    result.weights = result.weights.copy()
    result.weights.loc[march_end] = np.nan                                                # April's starting weights are the ones held at March's last close
    effects, reconciliation, months = independent_brinson(result, bundle, result.start)
    assert pd.Period("2021-04", "M") not in months and pd.Period("2021-05", "M") in months
    table = brinson_vs_benchmark(result, bundle)
    assert table.loc["TOTAL (cumulative active return, points)", "total"] == pytest.approx(effects["total"].sum() + reconciliation, abs=1e-8)
    intact = brinson_vs_benchmark(*brinson_fixture(held_equal=False))
    assert not np.isclose(table.loc["equity", "allocation"], intact.loc["equity", "allocation"]), "one month fewer"
