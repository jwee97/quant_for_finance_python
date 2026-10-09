"""The multi-period allocator: the book's limits hold at every rebalance, costs and persistence move the trading the right way, small universes and bad parameters are handled, and nothing looks ahead."""

from __future__ import annotations

import numpy as np
import pytest

from src.equity.synthetic import simulate_fundamental_world
from src.features.sleeves import month_end_dates
from src.framework import ALLOCATORS, MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.utils.config import load_config

load_library()


@pytest.fixture(scope="module")
def env():
    w = simulate_fundamental_world(n_assets=20, n_years=6, seed=3, premia={"momentum": 0.008, "reversal": 0.006})
    bundle = bundle_from_prices(w.prices, min_history=60, name="trading-world")
    model = MODELS.create("fundamental_momentum", factor="ret9")
    forecast = model.forecast(bundle, min_observations=252)
    config = load_config()
    return {"bundle": bundle, "forecast": forecast, "model": model, "config": config, "ctx": Context(bundle, config, forecasts=forecast, models=[model])}


def build(env, **params):
    params = {"horizon": 4, "plan_horizon": 2, **params}
    return ALLOCATORS.create("multi_period", **params).build(env["ctx"])


def rebalances(env, weights):
    rows = [d for d in month_end_dates(env["bundle"].index) if d in weights.index and weights.loc[d].abs().sum() > 1e-9]
    assert len(rows) > 20
    return weights.loc[rows]


def test_a_long_only_book_is_fully_invested_within_its_position_limit(env):
    w = rebalances(env, build(env, book="long_only", max_weight=0.15))
    assert (w >= -1e-7).all().all() and np.allclose(w.sum(axis=1), 1.0, atol=1e-5) and w.max().max() <= 0.15 + 1e-6


def test_a_130_30_book_holds_its_net_exposure_and_keeps_gross_within_the_limit(env):
    w = rebalances(env, build(env, book="130_30", max_weight=0.15))
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5) and (w.abs().sum(axis=1) <= 1.6 + 1e-5).all() and (w < -1e-6).any().any()
    assert w.abs().sum(axis=1).mean() > 1.2                                                              # the limit is used, not just respected
    n = rebalances(env, build(env, book="market_neutral", max_weight=0.15))
    assert np.allclose(n.sum(axis=1), 0.0, atol=1e-5) and (n.abs().sum(axis=1) <= 2.0 + 1e-5).all()


def test_dearer_trading_means_less_of_it(env):
    moved = lambda w: w.diff().abs().sum(axis=1).iloc[1:].mean()
    free = moved(rebalances(env, build(env, book="130_30", max_weight=0.15, cost_bps=0.0)))
    dear = moved(rebalances(env, build(env, book="130_30", max_weight=0.15, cost_bps=60.0)))
    assert dear < 0.85 * free


def test_a_forecast_that_lasts_is_worth_building_a_position_for(env):
    first = lambda w: w.loc[w.abs().sum(axis=1) > 1e-9].iloc[0].abs().sum()
    lasting = first(build(env, book="130_30", max_weight=0.15, cost_bps=150.0, persistence=1.0))
    fleeting = first(build(env, book="130_30", max_weight=0.15, cost_bps=150.0, persistence=0.0))
    assert lasting > fleeting + 0.15                                                                      # a position that will earn for several months is worth a cost that one earning for a single month is not
    traded = lambda w: w.diff().abs().sum(axis=1).iloc[1:].sum()
    assert traded(build(env, book="130_30", max_weight=0.15, cost_bps=40.0, persistence=1.0)) > traded(build(env, book="130_30", max_weight=0.15, cost_bps=40.0, persistence=0.0))      # and with a fading forecast the plan trades less


def test_a_universe_too_small_for_the_limit_has_it_widened(env):
    names = list(env["bundle"].assets)[:6]
    small = bundle_from_prices(env["bundle"].prices[names], min_history=60, name="small")
    ctx = Context(small, env["config"], forecasts=env["model"].forecast(small, min_observations=252), models=[env["model"]])
    w = ALLOCATORS.create("multi_period", book="long_only", max_weight=0.1, horizon=3, plan_horizon=2).build(ctx)
    last = w.iloc[-1]
    assert last.sum() == pytest.approx(1.0, abs=1e-5) and last.max() >= 1.0 / 6 - 1e-6                    # six names at 10% cannot be fully invested


def test_the_weights_on_a_date_do_not_depend_on_data_after_it(env):
    bundle, cut = env["bundle"], env["bundle"].index[900]
    noisy = bundle.perturbed_after(cut)
    a = build(env, book="130_30", max_weight=0.15)
    b = ALLOCATORS.create("multi_period", book="130_30", max_weight=0.15, horizon=4, plan_horizon=2).build(Context(noisy, env["config"], forecasts=env["forecast"], models=[env["model"]]))
    assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), atol=1e-9)


def test_bad_parameters_and_missing_forecasts_are_refused(env):
    for params in ({"book": "150_50"}, {"risk_aversion": 0.0}, {"horizon": 0}, {"plan_horizon": 0}, {"persistence": 1.5}, {"persistence": "sometimes"}, {"cost_bps": -1.0}, {"impact": -1.0}, {"max_weight": 0.0},
                   {"lookback": 10}, {"discount": 0.0}):
        with pytest.raises(ValueError):
            ALLOCATORS.create("multi_period", **params)
    with pytest.raises(ValueError, match="forecasts"):
        ALLOCATORS.create("multi_period").build(Context(env["bundle"], env["config"], forecasts=None, models=[]))
    assert ALLOCATORS.create("multi_period").required_assets() == 4


def test_the_pipeline_runs_the_book(env):
    spec = {"name": "mp", "models": [{"name": "fundamental_momentum", "params": {"factor": "ret9"}}], "allocation": {"allocator": "multi_period", "params": {"book": "long_only", "horizon": 4, "plan_horizon": 2}},
            "risk": {"mode": "none"}, "evaluation": {"causality": False}}
    r = Pipeline(spec, env["config"], env["bundle"]).run(validate=False)
    held = r.weights.loc[r.start:]
    held = held[held.abs().sum(axis=1) > 1e-9]
    assert 0.9 < held.sum(axis=1).mean() < 1.1 and np.isfinite(r.metrics["sharpe"])
