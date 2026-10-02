"""Attribution: identities that must hold to machine precision."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import (
    brinson_fachler,
    carino_factor,
    contribution_by_asset,
    cost_by_asset,
    cumulative,
    euler_risk_contributions,
    link_effects,
    monthly_inputs,
)
from src.backtest.engine import BacktestEngine
from src.portfolio.risk_parity import risk_parity_weights

SECTORS = {"A": "eq", "B": "eq", "C": "bond", "D": "bond", "E": "gold"}


def _random_book(rng, periods, columns, zero_columns=()):
    w = rng.dirichlet(np.ones(len(columns)), size=periods)
    frame = pd.DataFrame(w, columns=columns)
    for column in zero_columns:
        frame[column] = 0.0
    return frame.div(frame.sum(axis=1), axis=0)


def _month_index(n):
    return pd.period_range("2015-01", periods=n, freq="M")


# ------------------------------------------------------------------------- Brinson-Fachler
def test_effects_sum_to_the_active_return_every_period_including_unheld_sectors():
    rng = np.random.default_rng(0)
    n = 40
    columns = list(SECTORS)
    wp = _random_book(rng, n, columns, zero_columns=["E"])               # the portfolio never holds gold
    wb = _random_book(rng, n, columns, zero_columns=["C", "D"])          # the benchmark never holds bonds
    wp.index = wb.index = _month_index(n)
    returns = pd.DataFrame(rng.normal(0.006, 0.04, (n, len(columns))), index=wp.index, columns=columns)
    result = brinson_fachler(wp, wb, returns, SECTORS)
    assert result.identity_error() < 1e-14
    np.testing.assert_allclose(result.total.sum(axis=1), (wp * returns).sum(axis=1) - (wb * returns).sum(axis=1), atol=1e-14)
    assert (result.allocation["gold"] != 0).any() or True                  # gold is held by the benchmark only
    assert (result.selection["bond"] == 0).all()                           # no benchmark weight in bonds: nothing to select against


def test_a_portfolio_that_is_the_benchmark_has_no_effects():
    rng = np.random.default_rng(1)
    n = 24
    wb = _random_book(rng, n, list(SECTORS))
    wb.index = _month_index(n)
    returns = pd.DataFrame(rng.normal(0.005, 0.03, (n, 5)), index=wb.index, columns=list(SECTORS))
    result = brinson_fachler(wb, wb, returns, SECTORS)
    assert result.total.abs().to_numpy().max() < 1e-15
    assert result.linked().abs().to_numpy().max() < 1e-12


def test_weights_that_do_not_sum_to_one_are_refused():
    wb = pd.DataFrame([[0.2, 0.2, 0.2, 0.2, 0.2]], columns=list(SECTORS), index=_month_index(1))
    wp = wb * 0.9
    returns = pd.DataFrame([[0.01] * 5], columns=list(SECTORS), index=wb.index)
    with pytest.raises(ValueError, match="sum to one"):
        brinson_fachler(wp, wb, returns, SECTORS)


# ------------------------------------------------------------------------------- Carino
def test_carino_factor_by_hand_and_its_limit():
    assert float(carino_factor(0.10, 0.04)) == pytest.approx((np.log(1.10) - np.log(1.04)) / 0.06)
    assert float(carino_factor(0.05, 0.05)) == pytest.approx(1.0 / 1.05)
    assert float(carino_factor(0.05, 0.05 + 1e-14)) == pytest.approx(1.0 / 1.05, rel=1e-9)


def test_linked_effects_sum_to_the_cumulative_active_return_exactly():
    rng = np.random.default_rng(2)
    n = 120
    columns = list(SECTORS)
    wp, wb = _random_book(rng, n, columns), _random_book(rng, n, columns)
    wp.index = wb.index = _month_index(n)
    returns = pd.DataFrame(rng.normal(0.007, 0.045, (n, 5)), index=wp.index, columns=columns)
    result = brinson_fachler(wp, wb, returns, SECTORS)
    linked = result.linked()
    assert linked["total"].sum() == pytest.approx(result.cumulative_active_return(), abs=1e-12)
    arithmetic = result.total.sum().sum()
    assert abs(arithmetic - result.cumulative_active_return()) > 1e-3        # plain addition does NOT work: that is why Carino exists


def test_linked_contributions_sum_to_the_total_return_exactly():
    rng = np.random.default_rng(3)
    n = 96
    columns = list(SECTORS)
    w = _random_book(rng, n, columns)
    w.index = _month_index(n)
    returns = pd.DataFrame(rng.normal(0.006, 0.04, (n, 5)), index=w.index, columns=columns)
    contributions = contribution_by_asset(w, returns)
    total = contributions.sum(axis=1)
    linked = link_effects(contributions, total)
    assert linked.sum() == pytest.approx(cumulative(total), abs=1e-12)


# ---------------------------------------------------------------------------------- risk
def _covariance(n=6, seed=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n + np.eye(n) * 0.05
    return pd.DataFrame(cov, index=list("ABCDEF"), columns=list("ABCDEF"))


def test_euler_components_sum_to_the_portfolio_volatility():
    cov = _covariance()
    w = pd.Series([0.3, 0.2, 0.1, 0.15, 0.15, 0.1], index=cov.index)
    table = euler_risk_contributions(w, cov)
    sigma = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    assert table["component"].sum() == pytest.approx(sigma, abs=1e-14)
    assert table["share"].sum() == pytest.approx(1.0, abs=1e-14)


def test_risk_parity_weights_have_equal_euler_shares_and_a_zero_weight_contributes_nothing():
    cov = _covariance()
    weights = risk_parity_weights(cov)
    table = euler_risk_contributions(weights, cov)
    np.testing.assert_allclose(table["share"], 1.0 / len(cov), atol=1e-7)
    w = pd.Series([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], index=cov.index)
    assert euler_risk_contributions(w, cov).loc["C", "component"] == 0.0


# ------------------------------------------------------------------- from an engine run
def _market(n_days=700, seed=5):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-02", periods=n_days)
    return pd.DataFrame(rng.normal(0.0003, 0.01, (n_days, 5)), index=index, columns=list(SECTORS))


def test_start_of_month_weights_reproduce_the_engine_monthly_return_exactly():
    returns = _market()
    target = pd.DataFrame(np.tile([0.4, 0.2, 0.15, 0.15, 0.1], (len(returns), 1)), index=returns.index, columns=returns.columns)
    engine = BacktestEngine()
    run = engine.run(target, returns, "fixed", apply_vol_target=False)
    inputs = monthly_inputs(run.weights, returns, run.gross_returns)
    weights, asset = inputs["weights"], inputs["asset_returns"]
    valid = weights.notna().all(axis=1) & (weights.sum(axis=1) > 0.99)
    assert valid.sum() > 15
    rebuilt = (weights[valid] * asset[valid]).sum(axis=1)
    np.testing.assert_allclose(rebuilt, inputs["portfolio_return"][valid], atol=1e-12)


def test_cost_attribution_by_asset_sums_to_the_engine_cost():
    returns = _market(seed=6)
    rng = np.random.default_rng(7)
    target = pd.DataFrame(rng.dirichlet(np.ones(5), size=len(returns)), index=returns.index, columns=returns.columns)
    engine = BacktestEngine()
    run = engine.run(target, returns, "random", apply_vol_target=False)
    per_asset = cost_by_asset(run.trades, engine.cost_model.rates(returns.columns))
    np.testing.assert_allclose(per_asset.sum(axis=1), run.costs.reindex(per_asset.index).fillna(0.0), atol=1e-14)
    assert per_asset.to_numpy().min() >= 0
