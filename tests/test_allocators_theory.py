"""The portfolio-theory allocators: the tangency portfolio on the capital market line, mean-variance under a CVaR limit and best-K selection by annealing."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.equity import qubo as qb
from src.equity.cvar_portfolio import scenario_cvar
from src.equity.synthetic import simulate_fundamental_world
from src.features.sleeves import month_end_dates
from src.framework import ALLOCATORS, MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.portfolio.covariance import estimate_covariance
from src.utils.config import load_config

load_library()


@pytest.fixture(scope="module")
def env():
    w = simulate_fundamental_world(n_assets=12, n_years=7, seed=4, premia={"momentum": 0.01, "reversal": 0.006})
    bundle = bundle_from_prices(w.prices, min_history=60, name="theory-world")
    model = MODELS.create("fundamental_momentum", factor="ret9")
    forecast = model.forecast(bundle, min_observations=252)
    config = load_config()
    return {"bundle": bundle, "forecast": forecast, "model": model, "config": config, "ctx": Context(bundle, config, forecasts=forecast, models=[model])}


def rebalances(env, w):
    rows = [d for d in month_end_dates(env["bundle"].index) if d in w.index and w.loc[d].abs().sum() > 1e-12]
    assert len(rows) > 20
    return w.loc[rows]


def build(env, name, **params):
    return ALLOCATORS.create(name, **params).build(env["ctx"])


# ------------------------------------------------------------------------------------------------------------------ tangency and the capital market line
def test_the_tangency_allocator_holds_the_tangency_weights_scaled_by_the_risk_aversion(env):
    w = rebalances(env, build(env, "tangency_cml", risk_aversion=3.0, long_only=True, max_leverage=3.0))
    assert (w >= -1e-9).all().all() and (w.sum(axis=1) <= 3.0 + 1e-9).all()
    from src.equity.theory import tangency_portfolio

    bundle, fc = env["bundle"], env["forecast"]
    date = w.index[10]
    pos = bundle.index.get_loc(date)
    window = bundle.returns.iloc[pos - 251:pos + 1].fillna(0.0)
    cov = estimate_covariance(window, "shrinkage", 252, annualise=True) * (fc.horizon / 252.0)
    mu = fc.mean.loc[date]
    t = tangency_portfolio(mu, cov, 0.0, long_only=True)
    held = w.loc[date]
    assert np.abs(held / held.sum() - t.weights.to_numpy()).max() < 1e-6                           # the mix of risky assets is the tangency portfolio
    gamma = 3.0
    assert held.sum() == pytest.approx(min((t.expected_return) / (gamma * t.volatility ** 2), 3.0), rel=1e-6)       # and the amount held is the CML allocation


def test_more_risk_aversion_means_less_invested_and_leverage_is_capped(env):
    shares = {g: rebalances(env, build(env, "tangency_cml", risk_aversion=g, max_leverage=10.0)).sum(axis=1).mean() for g in (1.0, 5.0, 25.0)}
    assert shares[1.0] > shares[5.0] > shares[25.0]
    capped = rebalances(env, build(env, "tangency_cml", risk_aversion=0.5, max_leverage=1.0)).sum(axis=1)
    assert capped.max() <= 1.0 + 1e-9 and (capped >= 1.0 - 1e-6).mean() > 0.3                    # a keen investor is fully invested, not more, when borrowing is not allowed


def test_with_a_risk_free_rate_above_every_forecast_the_tangency_allocator_holds_cash(env):
    w = build(env, "tangency_cml", risk_free=5.0, long_only=True)                                  # 500% a year: nothing is expected to beat it
    assert (w.abs().sum(axis=1) < 1e-12).all()


# ------------------------------------------------------------------------------------------------------------------ mean-variance with a CVaR limit
def test_the_cvar_allocator_respects_its_limit_at_every_rebalance(env):
    limit, bundle = 0.04, env["bundle"]
    w = rebalances(env, build(env, "mv_cvar", cvar_limit=limit, max_weight=0.5, scenarios=250))
    assert (w >= -1e-9).all().all() and (w.sum(axis=1) <= 1.0 + 1e-9).all() and (w.max(axis=1) <= 0.5 + 1e-9).all()
    h = env["forecast"].horizon
    for date in w.index[::8]:
        pos = bundle.index.get_loc(date)
        daily = bundle.returns.iloc[pos - 755:pos + 1].fillna(0.0).to_numpy()
        cs = np.vstack([np.zeros((1, daily.shape[1])), np.cumsum(daily, axis=0)])
        R = (cs[h:] - cs[:-h])
        R = R[np.linspace(0, len(R) - 1, 250).astype(int)]
        assert scenario_cvar(w.loc[date].reindex(bundle.assets).to_numpy(), R, 0.95) <= limit + 1e-6, date
    tight = rebalances(env, build(env, "mv_cvar", cvar_limit=0.01, max_weight=0.5, scenarios=250))
    assert tight.sum(axis=1).mean() < w.sum(axis=1).mean()                                         # a tighter limit holds more cash


def test_a_fully_invested_book_under_a_loose_limit_is_fully_invested_and_an_impossible_limit_holds_nothing(env):
    full = rebalances(env, build(env, "mv_cvar", cvar_limit=0.5, fully_invested=True, max_weight=0.6, scenarios=200))
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-6)
    impossible = build(env, "mv_cvar", cvar_limit=1e-4, fully_invested=True, max_weight=0.6, scenarios=200)
    assert (impossible.abs().sum(axis=1) < 1e-12).all()                                            # no fully invested book is this safe: the allocator does not invent one


# ------------------------------------------------------------------------------------------------------------------ QUBO selection
def test_the_annealing_allocator_holds_exactly_k_names_in_equal_weights_and_they_are_the_best_k(env):
    k = 3
    w = rebalances(env, build(env, "qubo_select", k=k, risk_aversion=5.0, sweeps=150, restarts=16))
    assert ((w > 1e-12).sum(axis=1) == k).all() and np.allclose(w.max(axis=1), 1.0 / k) and np.allclose(w.sum(axis=1), 1.0)
    bundle, fc = env["bundle"], env["forecast"]
    hits = 0
    dates = w.index[::12][:6]
    for date in dates:
        pos = bundle.index.get_loc(date)
        window = bundle.returns.iloc[pos - 251:pos + 1].fillna(0.0)
        cov = estimate_covariance(window, "shrinkage", 252, annualise=True) * (fc.horizon / 252.0)
        mu = fc.mean.loc[date].to_numpy()
        cols = list(bundle.assets)
        best = min(itertools.combinations(range(len(cols)), k), key=lambda c: qb.selection_objective(mu, cov.to_numpy(), np.isin(np.arange(len(cols)), c), 5.0))
        chosen = np.flatnonzero(w.loc[date].reindex(cols).to_numpy() > 1e-12)
        hits += int(set(chosen) == set(best))
    assert hits >= len(dates) - 1                                                                   # exhaustive search over all 220 triples agrees (allowing one miss)


# ------------------------------------------------------------------------------------------------------------------ common properties
@pytest.mark.parametrize("name,params", [("tangency_cml", {}), ("mv_cvar", {"scenarios": 150}), ("qubo_select", {"k": 3, "sweeps": 100, "restarts": 8})])
def test_the_weights_on_a_date_do_not_depend_on_data_after_it(env, name, params):
    bundle, cut = env["bundle"], env["bundle"].index[1100]
    noisy = bundle.perturbed_after(cut)
    a = ALLOCATORS.create(name, **params).build(env["ctx"])
    b = ALLOCATORS.create(name, **params).build(Context(noisy, env["config"], forecasts=env["forecast"], models=[env["model"]]))
    assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), atol=1e-9)


@pytest.mark.parametrize("name,params", [("tangency_cml", {"risk_aversion": 0.0}), ("tangency_cml", {"max_leverage": 0.0}), ("mv_cvar", {"cvar_limit": 0.0}), ("mv_cvar", {"alpha": 1.0}),
                                         ("mv_cvar", {"max_weight": 0.0}), ("mv_cvar", {"window": 100}), ("qubo_select", {"k": 0}), ("qubo_select", {"sweeps": 5})])
def test_bad_parameters_are_refused(name, params):
    with pytest.raises(ValueError):
        ALLOCATORS.create(name, **params)


def test_forecasts_are_required_and_qubo_needs_more_assets_than_it_picks(env):
    for name in ("tangency_cml", "mv_cvar", "qubo_select"):
        with pytest.raises(ValueError, match="forecasts"):
            ALLOCATORS.create(name).build(Context(env["bundle"], env["config"], forecasts=None, models=[]))
    assert ALLOCATORS.create("qubo_select", k=7).required_assets() == 8


@pytest.mark.parametrize("name,params", [("tangency_cml", {}), ("mv_cvar", {"scenarios": 150}), ("qubo_select", {"k": 4, "sweeps": 100, "restarts": 8})])
def test_the_pipeline_runs_each_allocator(env, name, params):
    spec = {"name": name, "models": [{"name": "fundamental_momentum", "params": {"factor": "ret9"}}], "allocation": {"allocator": name, "params": params}, "risk": {"mode": "none"},
            "evaluation": {"causality": False}}
    r = Pipeline(spec, env["config"], env["bundle"]).run(validate=False)
    assert np.isfinite(r.metrics["sharpe"]) and r.weights.loc[r.start:].abs().sum(axis=1).mean() > 0.1
