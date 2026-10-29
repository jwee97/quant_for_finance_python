"""The cost model, the volatility target, drift and the engine's switches, checked against numbers worked out by hand.

A planted-bug audit (``experiments/plant_bugs.py``) found these modules weakly pinned: of the places where one operator, comparison or number of ``backtest/costs.py`` was changed, the tests noticed 18%
(7 of 38), of ``backtest/engine.py`` 23% (12 of 52) and of ``backtest/execution.py`` 48% (11 of 23). The tests that existed showed that the engine runs and that costs lower returns; none said what a cost
table, a turnover figure or a volatility scalar *is*. Each test below states the number it expects and how it was reached, so a change to the formula fails a test and not only a review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import LinearCostModel, breakeven_cost, cost_sensitivity, trades_from_weights, turnover, turnover_statistics
from src.backtest.engine import BacktestEngine, buy_and_hold
from src.backtest.execution import build_held_weights, drift_weights, vol_target_scaling


class Config:
    """The two things ``BacktestEngine.from_config`` reads of a configuration: ``get(key, default)``."""

    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


def days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", periods=n)


# ------------------------------------------------------------------------------------------------------------------ what a trade costs
def test_a_trade_costs_its_rate_times_the_weight_it_changes_asset_by_asset():
    model = LinearCostModel(cost_bps=10.0, per_asset_bps={"B": 40.0})
    assert model.rates(["A", "B", "C"]).tolist() == [0.001, 0.004, 0.001], "an asset without a figure of its own pays the default"
    index = days(3)
    trades = pd.DataFrame({"A": [0.5, -0.1], "B": [-0.25, 0.2], "C": [0.0, 0.3]}, index=index[:2])
    assert model.trade_costs(trades).tolist() == pytest.approx([0.5 * 0.001 + 0.25 * 0.004, 0.1 * 0.001 + 0.2 * 0.004 + 0.3 * 0.001]), "a sale costs as much as a purchase of the same size"
    gross = pd.Series([0.01, 0.002, 0.004], index=index)                                     # a third date on which nothing traded
    net, costs = model.apply(gross, trades)
    assert costs.tolist() == pytest.approx([0.0015, 0.0012, 0.0]) and net.tolist() == pytest.approx([0.0085, 0.0008, 0.004])
    assert model.describe() == {"model": "linear", "cost_bps": 10.0, "n_overrides": 1, "mean_effective_bps": 40.0}
    assert LinearCostModel().describe()["mean_effective_bps"] == 10.0 and LinearCostModel().cost_bps == 10.0, "the default is ten basis points one way"


def test_turnover_counts_the_first_row_as_the_cost_of_building_the_book():
    weights = pd.DataFrame({"A": [0.5, 0.6, 0.6], "B": [0.5, 0.4, 0.4]}, index=days(3))
    assert turnover(weights).tolist() == pytest.approx([1.0, 0.2, 0.0])
    plain = trades_from_weights(weights)
    assert plain.iloc[0].tolist() == [0.5, 0.5] and plain.iloc[1].tolist() == pytest.approx([0.1, -0.1]) and plain.iloc[2].tolist() == pytest.approx([0.0, 0.0])
    held = pd.DataFrame({"A": [0.0, 0.55, 0.6], "B": [0.0, 0.45, 0.4]}, index=days(3))
    against_book = trades_from_weights(weights, held)
    assert against_book.iloc[0].tolist() == [0.5, 0.5], "the first row is the book being built, whatever the held book says"
    assert against_book.iloc[1].tolist() == pytest.approx([0.05, -0.05]) and against_book.iloc[2].tolist() == pytest.approx([0.0, 0.0]), "later rows trade what separates the target from what is held"


def test_turnover_statistics_are_the_mean_the_share_of_days_trading_and_the_average_trade():
    trades = pd.DataFrame({"A": [0.6, 0.0, -0.1, 0.0, 0.0], "B": [0.4, 0.0, 0.1, 0.0, 0.0]}, index=days(5))        # daily turnover 1.0, 0, 0.2, 0, 0
    stats = turnover_statistics(trades, trades)
    assert stats == pytest.approx({"mean_daily_turnover": 0.24, "annualised_turnover": 0.24 * 252, "median_daily_turnover": 0.0, "max_daily_turnover": 1.0, "share_of_days_trading": 0.4,
                                   "mean_turnover_on_trading_days": 0.6})
    assert turnover_statistics(trades, trades, periods_per_year=52)["annualised_turnover"] == pytest.approx(0.24 * 52)
    unsaid = turnover_statistics(trades)                                                       # no trades given: the frame is read as weights and differenced, the first row being the build
    assert unsaid["mean_daily_turnover"] == pytest.approx((1.0 + 1.0 + 0.2 + 0.2 + 0.0) / 5) and unsaid["max_daily_turnover"] == pytest.approx(1.0)
    boundary = pd.DataFrame({"A": [1e-8, 1.2e-8, 0.0]}, index=days(3))
    s = turnover_statistics(boundary, boundary)
    assert s["share_of_days_trading"] == pytest.approx(1 / 3) and s["mean_turnover_on_trading_days"] == pytest.approx(1.2e-8), "a trade of exactly 1e-8 is rounding noise and is not a day of trading; 1.2e-8 is"
    idle = turnover_statistics(pd.DataFrame({"A": [0.0, 0.0]}, index=days(2)), pd.DataFrame({"A": [0.0, 0.0]}, index=days(2)))
    assert idle["share_of_days_trading"] == 0.0 and idle["mean_turnover_on_trading_days"] == 0.0
    noise = pd.DataFrame({"A": [1e-8, 0.0]}, index=days(2))
    quiet = turnover_statistics(noise, noise)
    assert quiet["share_of_days_trading"] == 0.0 and quiet["mean_turnover_on_trading_days"] == 0.0, "nothing but rounding noise is no trading, and its average is 0, not not-a-number"


GROSS = pd.Series([0.003, -0.001, 0.003, -0.001], index=days(4))                              # mean 0.001 a day, standard deviation 0.0023094
BOOK = pd.DataFrame({"A": [1.0, 0.0, 0.5, 0.0]}, index=days(4))                              # daily turnover 1, 0, 0.5, 0: mean 0.375


def annual(series: pd.Series) -> tuple[float, float]:
    """Mean times 252 and standard deviation times the square root of 252, in plain numpy."""
    x = series.to_numpy()
    return float(x.mean() * 252), float(x.std(ddof=1) * np.sqrt(252))


def test_the_cost_grid_reports_the_net_return_volatility_sharpe_and_what_costs_took():
    grid = cost_sensitivity(GROSS, BOOK, levels=(0.0, 10.0))
    assert list(grid.columns) == ["net_ann_return", "net_vol", "net_sharpe", "annual_cost_drag", "cost_share_of_gross"] and grid.index.name == "cost_bps"
    free = grid.loc[0.0]
    ret, vol = annual(GROSS)
    assert free["net_ann_return"] == pytest.approx(ret) and free["net_vol"] == pytest.approx(vol) and free["net_sharpe"] == pytest.approx(ret / vol) and free["annual_cost_drag"] == 0.0 and free["cost_share_of_gross"] == 0.0
    cost = pd.Series([0.001, 0.0, 0.0005, 0.0], index=GROSS.index)                           # ten basis points of the daily turnover
    ret, vol = annual(GROSS - cost)
    paid = grid.loc[10.0]
    assert paid["net_ann_return"] == pytest.approx(ret) and paid["net_vol"] == pytest.approx(vol) and paid["net_sharpe"] == pytest.approx(ret / vol)
    assert paid["annual_cost_drag"] == pytest.approx(0.000375 * 252) and paid["cost_share_of_gross"] == pytest.approx(0.375), "costs took 0.000375 a day of the 0.001 earned"
    assert list(cost_sensitivity(GROSS, BOOK).index) == [0.0, 5.0, 10.0, 25.0, 50.0] and cost_sensitivity(GROSS, BOOK, (5.0,)).index.tolist() == [5.0]
    assert cost_sensitivity(GROSS, BOOK, (10.0,), periods_per_year=52).loc[10.0, "annual_cost_drag"] == pytest.approx(0.000375 * 52), "the year length is a setting, not 252"


def test_the_cost_share_of_a_strategy_that_earns_nothing_is_not_a_number_and_a_flat_series_has_no_sharpe():
    losing = pd.Series([-0.002, 0.001, -0.002, 0.001], index=days(4))
    assert np.isnan(cost_sensitivity(losing, BOOK, (10.0,)).loc[10.0, "cost_share_of_gross"])
    assert np.isnan(cost_sensitivity(pd.Series(0.0, index=days(4)), BOOK, (10.0,)).loc[10.0, "cost_share_of_gross"]), "a mean of exactly zero is not positive either"
    idle = pd.DataFrame({"A": np.zeros(30)}, index=days(30))
    flat = pd.Series(0.001, index=days(30))                                                # the same return every day: a standard deviation of rounding noise, not a Sharpe ratio of 1e16
    row = cost_sensitivity(flat, idle, (0.0,)).loc[0.0]
    assert np.isnan(row["net_sharpe"]) and row["net_vol"] < 1e-9


def test_a_per_asset_cost_profile_is_scaled_by_the_level_and_keeps_its_ranking():
    book = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 0.0]}, index=days(4))        # A is bought on day 0 and B on day 2, each a turnover of 1
    profile = {"A": 5.0, "B": 15.0}                                                         # mean 10 bps: a level of 20 bps doubles each figure, to 10 and 30
    row = cost_sensitivity(GROSS, book, (20.0,), per_asset_bps=profile).loc[20.0]
    assert row["annual_cost_drag"] == pytest.approx(((10.0 + 30.0) / 1e4 / 4) * 252)
    assert cost_sensitivity(GROSS, book, (0.0,), per_asset_bps=profile).loc[0.0, "annual_cost_drag"] == 0.0
    uniform = cost_sensitivity(GROSS, book, (20.0,)).loc[20.0, "annual_cost_drag"]
    assert uniform == pytest.approx(((20.0 + 20.0) / 1e4 / 4) * 252) and row["annual_cost_drag"] == pytest.approx(uniform), "the same mean level costs the same on a book that trades each asset once"
    free = cost_sensitivity(GROSS, book, (20.0,), per_asset_bps={"A": 0.0, "B": 0.0}).loc[20.0]
    assert free["annual_cost_drag"] == pytest.approx(uniform), "a profile whose mean is zero cannot be scaled; the level applies to every asset"


def test_the_breakeven_cost_is_the_mean_gross_return_over_the_mean_turnover_in_basis_points():
    index = days(10)
    gross = pd.Series([0.001] * 5 + [0.0002] * 5, index=index)                               # mean 0.0006
    book = pd.DataFrame({"A": [0.6, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0]}, index=index)        # mean turnover 0.12
    assert breakeven_cost(gross, book) == pytest.approx(0.0006 / 0.12 * 1e4)
    shorter = book.iloc[:6]                                                                   # trades missing for the last four dates count as no trading
    assert breakeven_cost(gross, shorter) == pytest.approx(0.0006 / 0.12 * 1e4)
    none = pd.DataFrame({"A": np.zeros(10)}, index=index)
    assert breakeven_cost(gross, none) == float("inf"), "a strategy that never trades has no cost to break even on"
    assert np.isnan(breakeven_cost(-gross, none)) and np.isnan(breakeven_cost(gross * 0.0, none)), "no gain and no trading: not infinite, not anything"
    assert net_is_zero_at(gross, book)


def net_is_zero_at(gross: pd.Series, book: pd.DataFrame) -> bool:
    bps = breakeven_cost(gross, book)
    net, _ = LinearCostModel(bps).apply(gross, book)
    return abs(float(net.mean())) < 1e-12


# ------------------------------------------------------------------------------------------------------------------ the volatility target
def vol_book(n: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (n, 3)), index=days(n), columns=list("ABC"))
    weights = pd.DataFrame([[0.5, -0.3, 0.2]] * n, index=days(n), columns=list("ABC"))
    return weights, returns


def expected_scalar(weights, returns, k, target, lookback, cap, per_year=252) -> float:
    """The scalar applied on row ``k + 1``: the target over the volatility of the book's last ``lookback`` daily returns (known at the close of row ``k``), at most ``cap``; 1 until the window holds enough."""
    w, r = weights.to_numpy(), returns.to_numpy()
    book = np.array([np.nan] + [float(w[j - 1] @ r[j]) for j in range(1, len(w))])             # row j earns the return of day j on the book held at the close of j - 1
    window = book[max(1, k - lookback + 1): k + 1]
    if len(window) < min(lookback, max(lookback // 2, 20)):
        return 1.0
    return min(target / (np.std(window, ddof=1) * np.sqrt(per_year)), cap)


def test_the_volatility_scalar_is_the_target_over_the_books_realised_volatility_and_applies_a_day_later():
    weights, returns = vol_book()
    scaled, scalar = vol_target_scaling(weights, returns, target_vol=0.05, lookback=40, max_leverage=3.0)
    for k in (21, 22, 39, 40, 60, 100, 118):
        assert scalar.iloc[k + 1] == pytest.approx(expected_scalar(weights, returns, k, 0.05, 40, 3.0)), k
    assert (scalar.iloc[:21] == 1.0).all() and scalar.iloc[21] != 1.0 and scalar.iloc[21] == pytest.approx(expected_scalar(weights, returns, 20, 0.05, 40, 3.0)), "the first scalar needs twenty daily returns of the book (rows 1 to 20), so it first applies on row 21"
    assert scaled.equals(weights.mul(scalar, axis=0)) and scaled.iloc[60].tolist() == pytest.approx((weights.iloc[60] * scalar.iloc[60]).tolist()), "every asset of a row is scaled by that row's scalar"
    defaults = vol_target_scaling(weights, returns)[1]
    assert defaults.equals(vol_target_scaling(weights, returns, 0.10, 63, 1.5, 252)[1]), "the defaults are a 10% target, a 63-day window, a cap of 1.5 and 252 days a year"
    assert vol_target_scaling(weights, returns * 0.01)[1].iloc[80] == 1.5, "the default cap is 1.5, which a calm book reaches"
    wide = vol_target_scaling(weights, returns, 0.05, 63, 3.0)[1]
    for k in (29, 30, 31, 32, 80):                                                            # a 63-day window waits for 31 returns, half of it
        assert wide.iloc[k + 1] == pytest.approx(expected_scalar(weights, returns, k, 0.05, 63, 3.0)), k
    assert (wide.iloc[:31] == 1.0).all() and wide.iloc[32] != 1.0
    assert vol_target_scaling(weights, returns, 0.05, 40, 3.0, 52)[1].iloc[60] == pytest.approx(expected_scalar(weights, returns, 59, 0.05, 40, 3.0, per_year=52))


def test_the_scalar_never_sees_the_day_it_sizes_and_is_capped_at_the_maximum_leverage():
    weights, returns = vol_book()
    _, base = vol_target_scaling(weights, returns, 0.05, 40, 3.0)
    shocked = returns.copy()
    shocked.iloc[71:] = shocked.iloc[71:] * 7.0 + 0.02                                        # a different future from row 71 on
    _, other = vol_target_scaling(weights, shocked, 0.05, 40, 3.0)
    assert base.iloc[:72].equals(other.iloc[:72]), "a scalar through row 71 uses returns up to row 70 only"
    assert base.iloc[72] != other.iloc[72]
    quiet = returns * 0.01                                                                    # a book so calm that the target asks for more than the cap
    assert vol_target_scaling(weights, quiet, 0.10, 40, 1.5)[1].iloc[60] == 1.5 and vol_target_scaling(weights, quiet, 0.10, 40, 4.0)[1].iloc[60] == 4.0
    wild = returns * 30.0
    assert vol_target_scaling(weights, wild, 0.10, 40, 1.5)[1].iloc[60] < 1.0, "a volatile book is scaled down, with no floor"


def test_a_book_with_no_volatility_or_a_short_window_still_gets_a_scalar():
    weights, returns = vol_book()
    flat_scaled, flat = vol_target_scaling(weights * 0.0, returns, 0.05, 40, 3.0)
    assert (flat == 1.0).all() and (flat_scaled == 0.0).all().all(), "an empty book has nothing to scale, and no division by zero"
    _, short = vol_target_scaling(weights, returns, 0.05, 10, 3.0)
    for k in (9, 10, 11, 50):
        assert short.iloc[k + 1] == pytest.approx(expected_scalar(weights, returns, k, 0.05, 10, 3.0)), "a window of ten days waits for ten returns, not twenty"
    assert (short.iloc[:10] == 1.0).all() and short.iloc[11] != 1.0


# ------------------------------------------------------------------------------------------------------------------ drift
def test_a_long_short_book_drifts_with_the_returns_and_keeps_its_gross_exposure():
    index = days(3)
    targets = pd.DataFrame({"A": [0.8, np.nan, np.nan], "B": [-0.4, np.nan, np.nan]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, -0.05, 0.0]}, index=index)
    held, pre = drift_weights(targets, returns, return_pre_trade=True)
    grown = np.array([0.8 * 1.10, -0.4 * 0.95])                                                  # 0.88 and -0.38: gross 1.26 against 1.2 before
    expected = grown * (1.2 / 1.26)
    assert held.iloc[0].tolist() == [0.8, -0.4] and held.iloc[1].tolist() == pytest.approx(expected.tolist()) and held.iloc[2].tolist() == pytest.approx(expected.tolist())
    assert np.abs(held.iloc[1]).sum() == pytest.approx(1.2) and pre.iloc[1].tolist() == pytest.approx(expected.tolist()) and pre.iloc[0].isna().all(), "nothing is held before the first target, so nothing drifts"
    empty = drift_weights(pd.DataFrame({"A": [0.0, np.nan], "B": [0.0, np.nan]}, index=days(2)), pd.DataFrame({"A": [0.1, 0.1], "B": [0.1, 0.1]}, index=days(2)))
    assert empty.iloc[1].tolist() == [0.0, 0.0], "a flat book stays flat"


def test_the_default_is_a_book_that_drifts_and_a_book_that_does_not_is_held_as_set():
    index = days(15)                                                                          # three weeks, rebalanced on each Friday (rows 4, 9 and 14)
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0.0, 0.02, (15, 2)), index=index, columns=["A", "B"])
    targets = pd.DataFrame(0.5, index=index, columns=["A", "B"])
    drifted, traded = build_held_weights(targets, returns, "weekly", 0)
    default, _ = build_held_weights(targets, returns, "weekly", 0, True)
    fixed, fixed_traded = build_held_weights(targets, returns, "weekly", 0, False)
    assert drifted.equals(default) and not drifted.iloc[5:9].equals(fixed.iloc[5:9]), "a book drifts unless told not to"
    assert drifted.iloc[:4].isna().all().all() and drifted.iloc[4].tolist() == [0.5, 0.5] and abs(drifted.iloc[6, 0] - 0.5) > 1e-6 and drifted.iloc[9].tolist() == [0.5, 0.5]
    assert fixed.iloc[4:].eq(0.5).all().all() and fixed_traded.iloc[4].tolist() == [0.5, 0.5] and fixed_traded.iloc[5:].abs().to_numpy().max() == 0.0, "held as set: built once, then nothing to trade"
    assert traded.iloc[9].abs().sum() > 0.0, "a drifted book has to be brought back to its target"
    first, _ = build_held_weights(targets, returns, "daily")                                    # the default lag is one day: nothing is held on the first row
    assert first.iloc[0].isna().all() and first.iloc[1].tolist() == [0.5, 0.5]


# ------------------------------------------------------------------------------------------------------------------ the engine
def test_the_engine_reads_its_settings_from_the_configuration_and_has_stated_defaults():
    plain = BacktestEngine()
    assert (plain.signal_lag, plain.rebalance, plain.weight_drift, plain.annualisation, plain.min_assets, plain.target_vol, plain.vol_lookback, plain.max_leverage) == (1, "monthly", True, 252, 5, None, 63, 1.5)
    bare = BacktestEngine.from_config(Config())
    assert (bare.signal_lag, bare.rebalance, bare.weight_drift, bare.annualisation, bare.min_assets, bare.target_vol, bare.vol_lookback, bare.max_leverage) == (1, "monthly", True, 252, 5, None, 63, 1.5)
    assert bare.cost_model.cost_bps == 10.0 and bare.cost_model.per_asset_bps == {}
    on = BacktestEngine.from_config(Config(**{"portfolio.volatility.vol_target_enabled": True}))
    assert on.target_vol == 0.10, "a volatility target switched on without a level is 10%"
    given = BacktestEngine.from_config(Config(**{"backtest.engine.signal_lag": 2, "backtest.engine.rebalance": "weekly", "backtest.engine.weight_drift": False, "backtest.engine.annualisation": 260,
                                                  "backtest.engine.min_assets": 3, "backtest.costs.cost_bps": 7.0, "backtest.costs.per_asset_bps": {"A": 3.0}, "portfolio.volatility.vol_target_enabled": True,
                                                  "portfolio.volatility.target_vol": 0.2, "portfolio.volatility.lookback": 40, "portfolio.volatility.max_leverage_from_vol_target": 2.5}))
    assert (given.signal_lag, given.rebalance, given.weight_drift, given.annualisation, given.min_assets, given.target_vol, given.vol_lookback, given.max_leverage) == (2, "weekly", False, 260, 3, 0.2, 40, 2.5)
    assert given.cost_model.cost_bps == 7.0 and given.cost_model.per_asset_bps == {"A": 3.0}
    off = BacktestEngine.from_config(Config(**{"portfolio.volatility.vol_target_enabled": False, "portfolio.volatility.target_vol": 0.3}))
    assert off.target_vol is None, "a level without the switch is not a target"


def two_asset_run(**engine_args):
    index = days(80)
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0.0005, 0.02, (80, 2)), index=index, columns=["A", "B"])
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=index)
    engine = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False, min_assets=1, cost_model=LinearCostModel(0.0), **engine_args)
    return engine, weights, returns


def test_a_volatility_target_applies_only_when_asked_for_and_only_when_there_is_a_level():
    engine, weights, returns = two_asset_run(target_vol=0.05)
    assert engine.run(weights, returns, "t").vol_scalar is not None, "a level and no instruction: the target applies"
    assert engine.run(weights, returns, "t", apply_vol_target=None).vol_scalar is not None
    assert engine.run(weights, returns, "t", apply_vol_target=False).vol_scalar is None, "switched off for this run"
    bare, weights, returns = two_asset_run()
    assert bare.run(weights, returns, "t").vol_scalar is None
    assert bare.run(weights, returns, "t", apply_vol_target=True).vol_scalar is None, "asked for, with no level to aim at: nothing to apply"
    scaled = engine.run(weights, returns, "t")
    scalar = scaled.vol_scalar
    assert scaled.weights.iloc[50].tolist() == pytest.approx((weights.iloc[50] * scalar.iloc[50]).tolist())
    cut = scaled.slice(returns.index[10], returns.index[40])
    assert cut.vol_scalar.index[0] == returns.index[10] and cut.vol_scalar.index[-1] == returns.index[40] and bare.run(weights, returns, "t").slice(returns.index[10]).vol_scalar is None


def test_a_run_can_be_limited_to_a_window_and_the_window_is_what_the_full_run_had_there():
    engine, weights, returns = two_asset_run()
    full = engine.run(weights, returns, "full")
    start, end = returns.index[20], returns.index[60]
    cut = engine.run(weights, returns, "cut", start=start, end=end)
    assert cut.net_returns.index[0] == start and cut.net_returns.index[-1] == end and cut.net_returns.equals(full.net_returns.loc[start:end])
    assert cut.gross_returns.equals(full.gross_returns.loc[start:end]) and cut.weights.index[0] == start and cut.trades.index[-1] == end and cut.turnover.index[0] == start and cut.costs.index[-1] == end
    only_start = engine.run(weights, returns, "s", start=start)
    only_end = engine.run(weights, returns, "e", end=end)
    assert only_start.net_returns.index[0] == start and only_start.net_returns.index[-1] == full.net_returns.index[-1]
    assert only_end.net_returns.index[0] == full.net_returns.index[0] and only_end.net_returns.index[-1] == end


def test_an_asset_that_is_not_investable_or_unknown_holds_no_weight_and_too_few_assets_hold_none_at_all():
    engine, weights, returns = two_asset_run()
    investable = pd.DataFrame(True, index=weights.index, columns=weights.columns, dtype=object)
    investable.loc[weights.index[10:20], "B"] = False
    investable.loc[weights.index[30:35], "A"] = np.nan                                         # not known to be investable is not investable
    res = engine.run(weights, returns, "mask", investable)
    assert (res.weights.loc[weights.index[10:20], "B"] == 0.0).all() and (res.weights.loc[weights.index[30:35], "A"] == 0.0).all() and (res.weights.loc[weights.index[10:20], "A"] == 0.6).all()
    two = BacktestEngine(signal_lag=0, rebalance="daily", weight_drift=False, min_assets=2, cost_model=LinearCostModel(0.0))
    need = two.run(weights, returns, "need", investable)
    assert (need.weights.loc[weights.index[10:20]] == 0.0).all().all() and (need.weights.loc[weights.index[30:35]] == 0.0).all().all(), "with fewer than two assets investable the book is flat"
    assert (need.weights.loc[weights.index[40:]].to_numpy() == [0.6, 0.4]).all()


def test_a_missing_return_of_one_asset_does_not_hide_the_return_of_the_others():
    engine, weights, returns = two_asset_run()
    broken = returns.copy()
    broken.iloc[10, 0] = np.nan                                                                # A has no return on day 10
    res = engine.run(weights, broken, "gap")
    assert res.gross_returns[returns.index[10]] == pytest.approx(0.4 * broken.iloc[10, 1]), "the book earned what B earned"
    both = broken.copy()
    both.iloc[20] = np.nan                                                                     # no asset has a return on day 20: no return at all, not a zero
    assert pd.isna(engine.run(weights, both, "none").gross_returns.reindex(returns.index).iloc[20])


def test_a_result_carries_its_cost_of_trading_writes_its_files_and_prices_other_costs():
    index = days(120)
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(rng.normal(0.0006, 0.01, (120, 2)), index=index, columns=["A", "B"])
    targets = pd.DataFrame(rng.dirichlet([1, 1], 120), index=index, columns=["A", "B"])
    engine = BacktestEngine(signal_lag=1, rebalance="weekly", weight_drift=True, min_assets=1, cost_model=LinearCostModel(25.0))
    res = engine.run(targets, returns, "weekly_test")
    summary = res.summary()
    assert summary["cost_of_trading_sharpe"] == pytest.approx(summary["gross_sharpe"] - summary["sharpe"]) and summary["cost_of_trading_sharpe"] > 0 and summary["name"] == "weekly_test"
    assert res.cost_sensitivity().index.tolist() == [0.0, 5.0, 10.0, 25.0, 50.0] and res.breakeven_cost_bps() == pytest.approx(breakeven_cost(res.gross_returns, res.trades))
    assert res.cost_sensitivity().loc[25.0, "annual_cost_drag"] == pytest.approx(summary["ann_cost_drag"]), "the grid row at the engine's own cost is the engine's own cost drag"


def test_a_result_writes_its_returns_and_weights_to_a_folder_it_creates(tmp_path):
    engine, weights, returns = two_asset_run()
    res = engine.run(weights, returns, "saved")
    out = tmp_path / "a" / "b"
    written = res.write(out)
    assert set(written) == {"returns", "weights"} and written["returns"].name == "saved_returns.csv" and written["weights"].name == "saved_weights.csv"
    frame = pd.read_csv(written["returns"], index_col="date")
    assert list(frame.columns) == ["gross", "net", "costs", "turnover"] and len(frame) == len(res.weights.index) and pd.read_csv(written["weights"], index_col="date").columns.tolist() == ["A", "B"]
    assert res.write(out)["returns"].exists(), "writing again into a folder that exists is allowed"


def test_the_engine_logs_the_run_in_percent_and_basis_points(caplog):
    index = days(120)
    rng = np.random.default_rng(8)
    returns = pd.DataFrame(rng.normal(0.0006, 0.01, (120, 2)), index=index, columns=["A", "B"])
    targets = pd.DataFrame(rng.dirichlet([1, 1], 120), index=index, columns=["A", "B"])
    engine = BacktestEngine(signal_lag=1, rebalance="weekly", weight_drift=True, min_assets=1, cost_model=LinearCostModel(25.0))
    with caplog.at_level("INFO", logger="src.backtest.engine"):
        res = engine.run(targets, returns, "logged")
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("logged"))
    summary = res.summary()
    assert f"net CAGR={100 * summary['cagr']:+.2f}%" in line and f"vol={100 * summary['ann_vol']:.2f}%" in line and f"Sharpe={summary['sharpe']:+.2f}" in line
    assert f"costs={1e4 * res.costs.mean() * 252:.0f}bp/yr" in line and f"obs={len(res.net_returns):4d}" in line


def test_a_buy_and_hold_reference_is_named_after_its_ticker_unless_named():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.0, 0.0]}, index=days(3))
    assert buy_and_hold(returns, "A").name == "buy_hold_A" and buy_and_hold(returns, "A", "benchmark").name == "benchmark"


def test_on_the_day_the_account_is_lost_the_book_still_traded_and_paid_for_it():
    idx = days(60)
    ret = pd.DataFrame({"VOL": np.where(np.arange(60) == 30, 1.0, 0.002)}, index=idx)
    weights = pd.DataFrame({"VOL": np.where(np.arange(60) % 2 == 0, -1.5, -1.0)}, index=idx)      # a book that changes every day, so every day has a trade to pay for
    engine = BacktestEngine(rebalance="daily", min_assets=1, cost_model=LinearCostModel(cost_bps=10.0))
    res = engine.run(weights, ret, "short", pd.DataFrame(True, index=idx, columns=["VOL"]))
    day = idx[30]
    assert res.net_returns[day] == -1.0 and res.weights.loc[day, "VOL"] != 0.0 and res.turnover[day] > 0.0 and res.costs[day] > 0.0, "the book was in force at the close that ruined it; only what follows is cancelled"
    assert (res.weights.loc[idx[31:]] == 0.0).all().all() and (res.costs[idx[31:]] == 0.0).all() and (res.turnover[idx[31:]] == 0.0).all()
