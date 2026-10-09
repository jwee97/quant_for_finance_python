"""The constrained long-short allocator: named books hold their exposures at every rebalance, neutrality and turnover limits bind, small universes are handled, and nothing looks ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity.synthetic import simulate_fundamental_world
from src.features.sleeves import month_end_dates
from src.framework import ALLOCATORS, MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.utils.config import load_config

load_library()


@pytest.fixture(scope="module")
def env():
    w = simulate_fundamental_world(n_assets=24, n_years=7, seed=2, premia={"momentum": 0.008, "reversal": 0.006})
    groups = {a: f"G{i % 4}" for i, a in enumerate(w.prices.columns)}
    bundle = bundle_from_prices(w.prices, min_history=60, name="book-world", asset_class=dict(groups))
    model = MODELS.create("fundamental_momentum", factor="ret9")
    forecast = model.forecast(bundle, min_observations=252)
    config = load_config()
    return {"bundle": bundle, "forecast": forecast, "model": model, "config": config, "ctx": Context(bundle, config, forecasts=forecast, models=[model])}


def build(env, **params):
    return ALLOCATORS.create("constrained_long_short", **params).build(env["ctx"])


def rebalances(env, weights):
    rows = [d for d in month_end_dates(env["bundle"].index) if d in weights.index and weights.loc[d].abs().sum() > 0]
    assert len(rows) > 30
    return weights.loc[rows]


def test_a_130_30_book_holds_its_exposures_at_every_rebalance(env):
    w = rebalances(env, build(env, book="130_30", max_weight=0.15))
    assert np.allclose(w.clip(lower=0).sum(axis=1), 1.3, atol=1e-5) and np.allclose((-w.clip(upper=0)).sum(axis=1), 0.3, atol=1e-5) and np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert (w.abs() <= 0.15 + 1e-6).all().all() and (w < -1e-6).any().any()                              # it really sells some names


@pytest.mark.parametrize("book,long,short", [("long_only", 1.0, 0.0), ("120_20", 1.2, 0.2), ("market_neutral", 1.0, 1.0), ("dollar_neutral", 0.5, 0.5)])
def test_every_named_book_has_its_gross_exposures(env, book, long, short):
    w = rebalances(env, build(env, book=book, max_weight=0.3))
    assert np.allclose(w.clip(lower=0).sum(axis=1), long, atol=1e-5) and np.allclose((-w.clip(upper=0)).sum(axis=1), short, atol=1e-5)


def test_the_beta_neutral_book_has_the_beta_it_was_asked_for_at_every_rebalance(env):
    bundle = env["bundle"]
    w = rebalances(env, build(env, book="market_neutral", max_weight=0.2, beta_neutral=True, beta_target=0.0, beta_tolerance=0.02))
    market = bundle.returns.where(bundle.investable).mean(axis=1)
    for date in w.index[::6]:
        pos = bundle.index.get_loc(date)
        m = market.iloc[pos - 251:pos + 1]
        beta = bundle.returns.iloc[pos - 251:pos + 1].fillna(0.0).apply(lambda col: col.cov(m) / m.var())
        assert abs(float(beta @ w.loc[date])) <= 0.02 + 1e-5, date
    assert (w.sum(axis=1).abs() < 1e-5).all()


def test_the_sector_neutral_book_nets_to_each_groups_share_of_the_names(env):
    w = rebalances(env, build(env, book="130_30", max_weight=0.15, sector_neutral=True, group_tolerance=0.0))
    groups = pd.Series(env["bundle"].asset_class)
    net = w.T.groupby(groups).sum().T
    share = groups.value_counts() / len(groups)
    assert np.abs(net - share * 1.0).max().max() < 1e-5


def test_a_turnover_limit_lowers_the_trading_and_the_limit_is_respected_by_the_targets(env):
    free = rebalances(env, build(env, book="130_30", max_weight=0.15, cost_bps=0.0))
    limited = rebalances(env, build(env, book="130_30", max_weight=0.15, cost_bps=0.0, max_turnover=0.2))
    moved = lambda w: w.diff().abs().sum(axis=1).iloc[1:]
    assert moved(limited).mean() < moved(free).mean() * 0.7 and moved(limited).max() <= 0.2 + 0.25      # the limit is on the trade from what drifted, which differs from the last target by the month's moves


def test_trading_costs_cut_the_turnover(env):
    cheap = rebalances(env, build(env, book="130_30", max_weight=0.15, cost_bps=0.0))
    dear = rebalances(env, build(env, book="130_30", max_weight=0.15, cost_bps=40.0))
    assert dear.diff().abs().sum(axis=1).mean() < cheap.diff().abs().sum(axis=1).mean()


def test_a_universe_too_small_for_the_position_limit_has_it_widened_just_enough(env):
    bundle = env["bundle"]
    names = list(bundle.assets)[:6]
    small = bundle_from_prices(bundle.prices[names], min_history=60, name="small", asset_class={a: bundle.asset_class[a] for a in names})
    model = MODELS.create("fundamental_momentum", factor="ret9")
    ctx = Context(small, env["config"], forecasts=model.forecast(small, min_observations=252), models=[model])
    w = ALLOCATORS.create("constrained_long_short", book="130_30", max_weight=0.1).build(ctx)
    last = w.iloc[-1]
    assert last.clip(lower=0).sum() == pytest.approx(1.3, abs=1e-5) and last.abs().max() >= 1.3 / 6 - 1e-6 and last.abs().max() < 0.4        # six names at 10% cannot be 130% long


def test_the_weights_on_a_date_do_not_depend_on_data_after_it(env):
    bundle, cut = env["bundle"], env["bundle"].index[1100]
    noisy = bundle.perturbed_after(cut)
    a = ALLOCATORS.create("constrained_long_short", book="130_30", max_weight=0.15).build(env["ctx"])
    b = ALLOCATORS.create("constrained_long_short", book="130_30", max_weight=0.15).build(Context(noisy, env["config"], forecasts=env["forecast"], models=[env["model"]]))
    assert np.allclose(a.loc[:cut].to_numpy(), b.loc[:cut].to_numpy(), atol=1e-9)


def test_bad_parameters_and_missing_forecasts_are_refused(env):
    for params in ({"book": "150_50"}, {"risk_aversion": 0.0}, {"max_weight": 0.0}, {"max_weight": 1.5}, {"group_tolerance": -1.0}, {"max_turnover": -0.1}, {"lookback": 10}):
        with pytest.raises(ValueError):
            ALLOCATORS.create("constrained_long_short", **params)
    with pytest.raises(ValueError, match="forecasts"):
        ALLOCATORS.create("constrained_long_short").build(Context(env["bundle"], env["config"], forecasts=None, models=[]))
    assert ALLOCATORS.create("constrained_long_short").required_assets() == 4


def test_the_pipeline_runs_the_book_and_reports_a_long_short_portfolio(env):
    spec = {"name": "ls", "models": [{"name": "fundamental_momentum", "params": {"factor": "ret9"}}], "allocation": {"allocator": "constrained_long_short", "params": {"book": "130_30", "max_weight": 0.15}},
            "risk": {"mode": "none"}, "evaluation": {"causality": False}}
    r = Pipeline(spec, env["config"], env["bundle"]).run(validate=False)
    held = r.weights.loc[r.start:]
    assert 1.4 < held.abs().sum(axis=1).mean() < 1.7 and 0.9 < held.sum(axis=1).mean() < 1.1 and np.isfinite(r.metrics["sharpe"])
