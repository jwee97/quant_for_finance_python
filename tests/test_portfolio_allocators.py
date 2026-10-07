"""Forecast-covariance and Bayesian allocators: planted answers, no look-ahead, invalid input, pipeline integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import ALLOCATORS, Pipeline, bundle_from_prices
from src.framework.allocation import Context
from src.utils.config import load_config

N_DAYS, K = 2300, 6


@pytest.fixture(scope="module")
def bundle():
    """Six assets with a common factor; asset V0 is by far the calmest and V5 by far the wildest."""
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2005-01-03", periods=N_DAYS)
    vol = np.array([0.004, 0.008, 0.009, 0.010, 0.012, 0.030])
    common = rng.normal(0, 0.004, (N_DAYS, 1))
    r = common * np.linspace(0.5, 1.2, K) + rng.normal(0, 1, (N_DAYS, K)) * vol + 0.0002
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"V{i}" for i in range(K)])
    return bundle_from_prices(prices, name="alloc")


def _build(name: str, bundle, **params) -> pd.DataFrame:
    return ALLOCATORS.create(name, **params).build(Context(bundle=bundle, config=load_config()))


@pytest.mark.parametrize("model", ["dcc", "ogarch", "static"])
def test_dynamic_cov_minimum_variance_loads_on_the_calm_asset_and_respects_the_budget(bundle, model):
    w = _build("dynamic_cov", bundle, model=model)
    live = w.loc[w.abs().sum(axis=1) > 0]
    assert len(live) > 500
    last = live.iloc[-1]
    assert last["V0"] == last.max() and last["V5"] == last.min()
    assert np.isclose(last.sum(), 1.0, atol=1e-6) and (last >= -1e-9).all()


def test_dynamic_cov_holds_nothing_before_the_first_training_window_has_elapsed(bundle):
    w = _build("dynamic_cov", bundle, model="static", min_train=800)
    assert (w.iloc[:800].abs().sum(axis=1) == 0).all()


def test_dynamic_cov_weights_do_not_depend_on_later_data(bundle):
    cut = bundle.index[1900]
    rng = np.random.default_rng(3)
    prices = bundle.prices.copy()
    later = prices.index > cut
    prices.loc[later] = prices.loc[later] * np.exp(rng.normal(0, 0.05, prices.loc[later].shape))
    noisy = bundle_from_prices(prices, name="noisy")
    for model in ("static", "dcc"):
        real = _build("dynamic_cov", bundle, model=model).loc[:cut]
        fake = _build("dynamic_cov", noisy, model=model).loc[:cut]
        assert np.allclose(real.to_numpy(), fake.to_numpy(), atol=1e-8), model


@pytest.mark.parametrize("kwargs", [{"model": "garch"}, {"min_train": 100}, {"refit_every": 5}, {"horizon": 0}, {"n_factors": 0}, {"lookback": 10}])
def test_dynamic_cov_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ALLOCATORS.create("dynamic_cov", **kwargs)


def test_dynamic_cov_needs_enough_assets():
    idx = pd.bdate_range("2010-01-01", periods=1200)
    r = np.random.default_rng(0).normal(0, 0.01, (1200, 2))
    two = bundle_from_prices(pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["A", "B"]), name="two")
    with pytest.raises(ValueError, match="at least 3"):
        _build("dynamic_cov", two, model="static")


@pytest.mark.parametrize("kind", ["mvo_sample", "bayes_stein", "bayes_predictive"])
def test_bayesian_allocator_produces_fully_invested_causal_weights(bundle, kind):
    w = _build("bayesian", bundle, kind=kind, n_draws=10)
    live = w.loc[w.abs().sum(axis=1) > 0]
    assert len(live) > 1000 and np.allclose(live.sum(axis=1), 1.0, atol=1e-6)
    cut = bundle.index[1500]
    prices = bundle.prices.copy()
    prices.loc[prices.index > cut] *= 1.5
    shocked = _build("bayesian", bundle_from_prices(prices, name="s"), kind=kind, n_draws=10).loc[:cut]
    assert np.allclose(w.loc[:cut].to_numpy(), shocked.to_numpy(), atol=1e-8)


def test_bayesian_posterior_mean_is_more_stable_than_the_plug_in_optimum(bundle):
    plug, bayes = _build("bayesian", bundle, kind="mvo_sample"), _build("bayesian", bundle, kind="bayes_stein")
    turnover = lambda w: w.diff().abs().sum(axis=1).loc["2010":].mean()
    assert turnover(bayes) <= turnover(plug) * 1.05


@pytest.mark.parametrize("kwargs", [{"kind": "hierarchical"}, {"lookback": 10}, {"risk_aversion": 0.0}, {"nu0": -1.0}, {"n_draws": 0}])
def test_bayesian_allocator_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ALLOCATORS.create("bayesian", **kwargs)


@pytest.mark.parametrize("allocation", [{"allocator": "dynamic_cov", "params": {"model": "static"}}, {"allocator": "bayesian", "params": {"kind": "bayes_stein"}}])
def test_the_allocators_run_through_the_pipeline_and_report_net_metrics(bundle, allocation):
    spec = {"models": [], "allocation": allocation, "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), bundle).run(validate=False)
    assert out.metrics["n_days"] > 300 and np.isfinite(out.metrics["sharpe"]) and out.metrics["ann_vol"] > 0


# ------------------------------------------------------------------------------------------------ evolution-strategy policy
@pytest.fixture(scope="module")
def trending():
    """Asset T0 drifts up strongly every month while the others do not: a policy that can read momentum should tilt to it."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2004-01-01", periods=2800)
    r = rng.normal(0.0001, 0.008, (2800, 5))
    r[:, 0] += 0.0012
    return bundle_from_prices(pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"T{i}" for i in range(5)]), name="trend")


def test_es_policy_is_long_only_fully_invested_and_flat_before_training(trending):
    w = _build("es_policy", trending, pairs=8, iterations=15)
    assert (w.iloc[:1260].abs().sum(axis=1) == 0).all()
    live = w.iloc[1400:]
    assert np.allclose(live.sum(axis=1), 1.0, atol=1e-9) and (live >= 0).all().all()


def test_es_policy_learns_to_tilt_toward_the_asset_that_pays_a_persistent_premium(trending):
    w = _build("es_policy", trending, pairs=16, iterations=40, risk_aversion=1.0, cost=0.0)
    assert w["T0"].iloc[1400:].mean() > 0.25                          # equal weight would be 0.20


def test_es_policy_does_not_depend_on_later_data(trending):
    cut = trending.index[1800]
    prices = trending.prices.copy()
    later = prices.index > cut
    prices.loc[later] = prices.loc[later] * np.exp(np.random.default_rng(2).normal(0, 0.05, prices.loc[later].shape))
    kwargs = dict(pairs=8, iterations=10)
    real = _build("es_policy", trending, **kwargs).loc[:cut]
    fake = _build("es_policy", bundle_from_prices(prices, name="n"), **kwargs).loc[:cut]
    assert np.allclose(real.to_numpy(), fake.to_numpy(), atol=1e-10)


@pytest.mark.parametrize("kwargs", [{"seeds": ()}, {"min_train": 100}, {"refit_every": 5}, {"cost": -1.0}, {"risk_aversion": 0.0}, {"pairs": 1}, {"sigma": 0.0}, {"lr": 0.0}, {"iterations": 0}])
def test_es_policy_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ALLOCATORS.create("es_policy", **kwargs)


def test_es_policy_runs_through_the_pipeline(trending):
    spec = {"models": [], "allocation": {"allocator": "es_policy", "params": {"pairs": 8, "iterations": 10}}, "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), trending).run(validate=False)
    assert out.metrics["n_days"] > 300 and np.isfinite(out.metrics["sharpe"])


# ------------------------------------------------------------------------------------------ min variance, max diversification
@pytest.mark.parametrize("name", ["min_variance", "max_diversification"])
def test_risk_based_books_are_fully_invested_flat_before_history_and_causal(bundle, name):
    w = _build(name, bundle)
    assert (w.iloc[:250].abs().sum(axis=1) == 0).all()
    live = w.iloc[300:]
    assert np.allclose(live.sum(axis=1), 1.0, atol=1e-6) and (live >= -1e-9).all().all()
    cut = bundle.index[1700]
    prices = bundle.prices.copy()
    prices.loc[prices.index > cut] *= np.exp(np.random.default_rng(1).normal(0, 0.05, prices.loc[prices.index > cut].shape))
    shocked = _build(name, bundle_from_prices(prices, name="s")).loc[:cut]
    assert np.allclose(w.loc[:cut].to_numpy(), shocked.to_numpy(), atol=1e-7)


def test_min_variance_favours_the_calm_asset_and_max_diversification_spreads_more_widely(bundle):
    mv, md = _build("min_variance", bundle).iloc[-1], _build("max_diversification", bundle).iloc[-1]
    assert mv["V0"] == mv.max() and mv["V5"] == mv.min()
    assert (md ** 2).sum() < (mv ** 2).sum() + 1e-9                           # max diversification is less concentrated in the calmest asset


def test_the_diversification_ratio_is_one_for_one_asset_and_the_optimiser_beats_equal_weight():
    from src.portfolio.diversification import diversification_ratio, maximum_diversification_weights

    rng = np.random.default_rng(3)
    a = rng.normal(size=(500, 5))
    cov = pd.DataFrame(np.cov(a.T) * np.array([1, 2, 3, 4, 5])[:, None] ** 0.5 * 0.01, columns=list("ABCDE"), index=list("ABCDE"))
    cov = (cov + cov.T) / 2 + np.eye(5) * 0.05
    assert diversification_ratio([1, 0, 0, 0, 0], cov.to_numpy()) == pytest.approx(1.0)
    result = maximum_diversification_weights(cov)
    assert result.success and result.weights.sum() == pytest.approx(1.0)
    assert diversification_ratio(result.weights, cov.to_numpy()) >= diversification_ratio(np.full(5, 0.2), cov.to_numpy()) - 1e-9


@pytest.mark.parametrize("name", ["min_variance", "max_diversification"])
def test_risk_based_books_reject_invalid_parameters_and_run_in_the_pipeline(bundle, name):
    for kwargs in ({"lookback": 10}, {"min_assets": 1}):
        with pytest.raises(ValueError):
            ALLOCATORS.create(name, **kwargs)
    spec = {"models": [], "allocation": {"allocator": name}, "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), bundle).run(validate=False)
    assert out.metrics["n_days"] > 300 and np.isfinite(out.metrics["sharpe"])


# ------------------------------------------------------------------------------------------------------------ sleeves
def test_sleeves_give_each_asset_an_equal_slice_times_its_own_signal_and_never_resize_each_other(bundle):
    from src.framework import MODELS

    ctx = Context(bundle=bundle, config=load_config(), models=[MODELS.create("sell_in_may", classes=())])
    w = ALLOCATORS.create("sleeves").build(ctx)
    live = w.dropna(how="all").iloc[400:]
    n = bundle.investable.loc[live.index].sum(axis=1)
    top = live.abs().max(axis=1)
    assert ((top == 0) | np.isclose(top, 1.0 / n)).all()
    flat = live.abs().sum(axis=1) == 0
    assert flat.any() and (~flat).any()                                        # cash when the rule says flat
    model = MODELS.create("rsi2")
    ctx2 = Context(bundle=bundle, config=load_config(), models=[model])
    w2 = ALLOCATORS.create("sleeves").build(ctx2)
    score = model.score(bundle)
    change = w2.diff().abs()
    stable = (score.diff().abs() == 0)                                         # an asset whose signal did not change keeps exactly the same weight
    assert (change.where(stable).fillna(0.0).max().max() < 1e-12)


def test_sleeves_options_and_invalid_input(bundle):
    from src.framework import MODELS

    model = MODELS.create("tsmom")
    ctx = Context(bundle=bundle, config=load_config(), models=[model])
    plain, scaled, long_only = (ALLOCATORS.create("sleeves", **p).build(ctx) for p in ({}, {"vol_scale": True}, {"long_only": True}))
    assert (long_only.stack().dropna() >= 0).all() and plain.abs().sum(axis=1).max() <= 3.0 + 1e-9
    assert not np.allclose(plain.fillna(0.0).to_numpy(), scaled.fillna(0.0).to_numpy())
    for bad in ({"gross": 0}, {"target_vol": 0}, {"max_scale": -1}, {"clip": 0}):
        with pytest.raises(ValueError):
            ALLOCATORS.create("sleeves", **bad)
    with pytest.raises(ValueError, match="exactly one"):
        ALLOCATORS.create("sleeves").build(Context(bundle=bundle, config=load_config(), models=[model, MODELS.create("rsi2")]))


def test_sleeves_run_through_the_pipeline_with_the_model_declared_rebalance(bundle):
    spec = {"models": [{"name": "rsi2"}], "allocation": {"allocator": "sleeves"}, "evaluation": {"causality": False, "benchmarks": []}}
    out = Pipeline(spec, load_config(), bundle).run(validate=False)
    assert out.metrics["n_days"] > 300 and out.metrics["ann_turnover"] > 0
