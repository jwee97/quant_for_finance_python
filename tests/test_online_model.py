"""The online-learning forecast model: it recovers a planted signal, adapts to a drift, uses only matured labels and rejects bad input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies.online import OnlineLinear
from src.utils.config import load_config

load_library()


def _world(flip_at: int | None = None, n: int = 3400, k: int = 8, seed: int = 3):
    """Daily prices whose next-month return loads on the trailing 63-day momentum (sign flipping at ``flip_at``)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    r = rng.normal(0.0002, 0.01, (n, k))
    for t in range(64, n):
        mom = r[t - 63:t].sum(axis=0)
        sign = -1.0 if (flip_at is not None and t >= flip_at) else 1.0
        r[t] += sign * 0.05 * np.tanh(mom / 0.3) / 21
    prices = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=cols)
    return bundle_from_prices(prices, name="planted")


def _ic(model, bundle, start="2012-01-01") -> float:
    f = model.forecast(bundle).mean
    fwd = bundle.prices.shift(-21) / bundle.prices - 1.0
    ics = []
    for d in f.index[f.index >= start][::21]:
        m = f.loc[d].notna() & fwd.loc[d].notna()
        if m.sum() > 4 and f.loc[d][m].nunique() > 1:
            ics.append(np.corrcoef(f.loc[d][m], fwd.loc[d][m])[0, 1])
    return float(np.nanmean(ics))


@pytest.mark.parametrize("updater", ["ridge", "nlms", "kalman"])
def test_each_updater_recovers_a_planted_momentum_relationship(updater):
    assert _ic(OnlineLinear(updater=updater), _world()) > 0.02


def test_forgetting_lets_the_model_follow_a_sign_flip_that_a_long_memory_does_not():
    bundle = _world(flip_at=2200)
    fast = _ic(OnlineLinear(forgetting=0.9), bundle, start="2014-06-01")
    slow = _ic(OnlineLinear(forgetting=1.0), bundle, start="2014-06-01")
    assert fast > slow


def test_no_forecast_before_enough_months_have_matured():
    bundle = _world()
    f = OnlineLinear(min_updates=36).forecast(bundle).mean
    first = f.dropna(how="all").index[0]
    assert first >= bundle.index[0] + pd.DateOffset(months=36)


def test_online_forecast_is_causal_in_the_library_check():
    bundle = _world(n=2200)
    for updater in ("ridge", "nlms", "kalman"):
        check = check_causality(OnlineLinear(updater=updater, min_updates=12), bundle, cutoff=bundle.index[1700])
        assert check["ok"], check


def test_data_after_an_origin_cannot_change_that_origin_forecast():
    bundle = _world(n=2200)
    model = OnlineLinear(min_updates=12)
    origin = bundle.prices.index[bundle.prices.index.to_series().groupby(bundle.prices.index.to_period("M")).transform("max") == bundle.prices.index][1800 // 21]
    base = model.forecast(bundle).mean.loc[origin].dropna()
    rng = np.random.default_rng(0)
    prices = bundle.prices.copy()
    later = prices.index > origin
    prices.loc[later] = prices.loc[later] * np.exp(rng.normal(0, 0.05, prices.loc[later].shape))
    shocked = model.forecast(bundle_from_prices(prices, name="shock")).mean.loc[origin].reindex(base.index)
    assert len(base) > 0 and np.allclose(base, shocked, atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"updater": "sgd"}, {"forgetting": 0.0}, {"forgetting": 1.5}, {"ridge": -1.0}, {"features": "all"}, {"min_updates": 0}, {"step": 0.0}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OnlineLinear(**kwargs)


def test_macro_features_are_demanded_by_name_and_otherwise_run():
    bundle = _world(n=1800)
    macro = pd.DataFrame({"VIX": 20 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, len(bundle.index)))}, index=bundle.index)
    with_macro = bundle_from_prices(bundle.prices, macro=macro, name="m")
    f = OnlineLinear(features="price_macro", min_updates=12).forecast(with_macro).mean
    assert f.stack().notna().sum() > 100


def test_the_model_is_registered_and_runs_through_the_pipeline():
    assert "online_ridge" in [e.name for e in MODELS.entries()]
    bundle = _world(n=2600)
    spec = {"models": [{"name": "online_ridge", "params": {"min_updates": 12}}], "allocation": {"allocator": "forecast_stack"}, "evaluation": {"causality": False}}
    out = Pipeline(spec, load_config(), bundle).run(validate=False)
    assert out.metrics["n_days"] > 200 and np.isfinite(out.metrics["sharpe"])
