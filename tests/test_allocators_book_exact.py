"""The risk-budget and resampled mean-variance allocators against what their sources define: risk contributions that follow the budget, and the average of the bootstrap optima."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.framework import ALLOCATORS, load_library
from src.framework.allocation import Context
from src.framework.allocators_portfolio import class_budget
from src.portfolio.covariance import estimate_covariance
from src.portfolio.mean_variance import resampled_weights
from src.portfolio.risk_parity import risk_contributions
from src.utils.config import load_config

load_library()
CLASSES = {"A": "equity", "B": "equity", "C": "rates", "D": "rates", "E": "commodity", "F": "commodity"}


def market(n: int = 700, seed: int = 4):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-02", periods=n)
    common = rng.normal(0.0, 0.006, (n, 1))
    scale = np.array([1.0, 1.1, 0.8, 0.9, 1.3, 1.2])
    returns = pd.DataFrame((common + rng.normal(0.0, 0.008, (n, 6))) * scale + 0.0002, index=index, columns=list("ABCDEF"))
    return SimpleNamespace(returns=returns, investable=pd.DataFrame(True, index=index, columns=returns.columns), index=index, assets=list("ABCDEF"), group={}, asset_class=CLASSES, market=None, name="test")


class Forecasts:
    """What the forecast-based allocators read of a ForecastPanel: a mean, a confidence and a horizon."""

    def __init__(self, bundle, mu):
        self.mean = pd.DataFrame(np.tile(mu, (len(bundle.index), 1)), index=bundle.index, columns=bundle.assets)
        self.confidence = pd.DataFrame(0.7, index=bundle.index, columns=bundle.assets)
        self.horizon = 21


def last_origin(allocator_weights, bundle):
    months = bundle.index.to_series().groupby([bundle.index.year, bundle.index.month]).max()
    origin = months.iloc[-2]                                                            # the last full month-end: the next month's weights are filled forward from it
    return origin, allocator_weights.loc[origin]


# ------------------------------------------------------------------------------------------------------------------ the budget
def test_class_budget_shares_each_class_budget_equally_over_its_members():
    got = class_budget(list("ABCD"), {"A": "equity", "B": "equity", "C": "rates", "D": "commodity"}, {"equity": 0.6, "rates": 0.3, "commodity": 0.1})
    assert got.to_dict() == pytest.approx({"A": 0.3, "B": 0.3, "C": 0.3, "D": 0.1})
    assert got.sum() == pytest.approx(1.0)


def test_a_class_that_is_not_named_gets_the_smallest_budget_and_the_budgets_are_renormalised():
    got = class_budget(list("ABC"), {"A": "equity", "B": "rates", "C": "crypto"}, {"equity": 0.7, "rates": 0.2})
    raw = np.array([0.7, 0.2, 0.2])                                                     # crypto is not named: it gets the smallest named budget, 0.2
    assert got.to_numpy() == pytest.approx(raw / raw.sum())


def test_without_budgets_every_asset_has_the_same_one():
    assert class_budget(list("ABCD"), CLASSES, None).to_numpy() == pytest.approx([0.25] * 4)


def test_the_risk_budget_book_gives_each_class_the_risk_it_was_budgeted():
    bundle = market()
    ctx = Context(bundle, load_config())
    budgets = {"equity": 0.5, "rates": 0.3, "commodity": 0.2}
    weights = ALLOCATORS.create("risk_budget", budgets=budgets, lookback=252).build(ctx)
    origin, w = last_origin(weights, bundle)
    position = bundle.index.get_loc(origin)
    cov = estimate_covariance(bundle.returns.iloc[position - 251:position + 1], "shrinkage", 252, annualise=True)
    contributions = pd.Series(risk_contributions(w.to_numpy(), cov.to_numpy(), normalise=True), index=w.index)
    by_class = contributions.groupby(pd.Series(CLASSES)).sum()
    assert w.sum() == pytest.approx(1.0, abs=1e-6) and (w > 0).all()
    assert by_class["equity"] == pytest.approx(0.5, abs=2e-3) and by_class["rates"] == pytest.approx(0.3, abs=2e-3) and by_class["commodity"] == pytest.approx(0.2, abs=2e-3)
    assert contributions["A"] == pytest.approx(contributions["B"], abs=2e-3)               # the members of a class share its budget equally


def test_with_no_budget_it_is_equal_risk_contribution_and_zero_before_the_lookback():
    bundle = market()
    weights = ALLOCATORS.create("risk_budget", lookback=252).build(Context(bundle, load_config()))
    origin, w = last_origin(weights, bundle)
    position = bundle.index.get_loc(origin)
    cov = estimate_covariance(bundle.returns.iloc[position - 251:position + 1], "shrinkage", 252, annualise=True)
    rc = risk_contributions(w.to_numpy(), cov.to_numpy(), normalise=True)
    assert rc == pytest.approx([1 / 6] * 6, abs=2e-3)
    assert (weights.iloc[:200] == 0).all().all()


@pytest.mark.parametrize("kwargs", [dict(budgets={}), dict(budgets={"equity": -0.1, "rates": 0.5}), dict(budgets={"equity": 0.0}), dict(budgets="equal"), dict(lookback=10), dict(min_assets=1)])
def test_risk_budget_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ALLOCATORS.create("risk_budget", **kwargs)


# ------------------------------------------------------------------------------------------------------------------ resampling
def test_resampled_mean_variance_is_the_average_of_the_bootstrap_optima_on_the_same_inputs():
    bundle = market()
    mu = np.array([0.08, 0.06, 0.03, 0.04, 0.07, 0.05])
    ctx = Context(bundle, load_config(), forecasts=Forecasts(bundle, mu))
    alloc = ALLOCATORS.create("resampled_mv", n_resamples=20, lookback=252, seed=3)
    weights = alloc.build(ctx)
    origin, w = last_origin(weights, bundle)
    position = bundle.index.get_loc(origin)
    window = bundle.returns.iloc[position - 251:position + 1]
    expected = resampled_weights(pd.Series(mu * 252 / 21, index=list("ABCDEF")), window, 5.0, 20, ctx.constraints(), "shrinkage", 3)
    assert w.to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-9) and (weights.iloc[:200] == 0).all().all()


def test_resampling_is_deterministic_in_its_seed_and_changes_with_it():
    bundle = market()
    ctx = Context(bundle, load_config(), forecasts=Forecasts(bundle, np.array([0.08, 0.06, 0.03, 0.04, 0.07, 0.05])))
    a = ALLOCATORS.create("resampled_mv", n_resamples=15, seed=1).build(ctx).iloc[-1]
    b = ALLOCATORS.create("resampled_mv", n_resamples=15, seed=1).build(ctx).iloc[-1]
    c = ALLOCATORS.create("resampled_mv", n_resamples=15, seed=2).build(ctx).iloc[-1]
    assert a.to_numpy() == pytest.approx(b.to_numpy(), abs=1e-15) and not np.allclose(a.to_numpy(), c.to_numpy(), atol=1e-9)


@pytest.mark.parametrize("kwargs", [dict(risk_aversion=0.0), dict(n_resamples=5), dict(lookback=10)])
def test_resampled_mean_variance_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ALLOCATORS.create("resampled_mv", **kwargs)


def test_the_resampled_book_needs_forecasts():
    with pytest.raises(ValueError):
        ALLOCATORS.create("resampled_mv").build(Context(market(), load_config()))
