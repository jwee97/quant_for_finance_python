"""The exposure-by-Q-learning allocator: states are causal, the fitted Q-function behaves as dynamic programming says it should, and the allocator holds the exposure the data justify without looking ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.sleeves import month_end_dates
from src.framework import ALLOCATORS, bundle_from_prices, load_library
from src.framework.allocation import Context
from src.rl import exposure as ex
from src.utils.config import load_config

load_library()


def trending_market(seed=0, months=180, drift=0.01, vol=0.02):
    rng = np.random.default_rng(seed)
    return drift + vol * rng.normal(size=months)


# ------------------------------------------------------------------------------------------------------------------ states
def test_market_states_use_only_the_past():
    r = trending_market()
    base = ex.market_states(r)
    r2 = r.copy()
    r2[100:] = 0.5                                                                                     # change the future
    changed = ex.market_states(r2)
    assert (base[:100] == changed[:100]).all()
    assert (base[:11] == -1).all() and (base[11:] >= 0).all() and base.max() <= 3


def test_market_states_read_trend_and_volatility():
    up = np.r_[np.full(30, 0.01), np.full(3, 0.01)]
    assert ex.market_states(up)[-1] & 1 == 1                                                          # rising: the trend bit is set
    down = np.full(40, -0.01)
    assert ex.market_states(down)[-1] & 1 == 0
    r = np.r_[np.full(30, 0.005), [0.1, -0.1, 0.1]]                                                   # calm, then three violent months
    assert ex.market_states(r)[-1] & 2 == 2


# ------------------------------------------------------------------------------------------------------------------ fitted Q
def test_q_prefers_full_exposure_when_the_market_pays_and_none_when_it_loses():
    r = trending_market(drift=0.02)
    s = ex.market_states(r)
    Q = ex.fit_exposure_policy(s[:-1], r[1:], (0.0, 0.5, 1.0), cost=0.0, risk_aversion=2.0)
    seen = [m for m in range(4) if (s[:-1] == m).sum() > 20]                                           # a state that is never visited teaches nothing
    assert seen and (Q[seen].argmax(axis=2) == 2).all()
    r = trending_market(1, drift=-0.02)
    s = ex.market_states(r)
    Q = ex.fit_exposure_policy(s[:-1], r[1:], (0.0, 0.5, 1.0), cost=0.0, risk_aversion=2.0)
    seen = [m for m in range(4) if (s[:-1] == m).sum() > 20]
    assert seen and (Q[seen].argmax(axis=2) == 0).all()


def test_trading_cost_makes_the_policy_stick_to_what_it_holds():
    rng = np.random.default_rng(0)
    r = 0.0004 + 0.02 * rng.normal(size=300)                                                           # a weak premium: almost nothing to gain from moving
    s = ex.market_states(r)
    free = ex.fit_exposure_policy(s[:-1], r[1:], (0.0, 0.5, 1.0), cost=0.0, risk_aversion=5.0)
    dear = ex.fit_exposure_policy(s[:-1], r[1:], (0.0, 0.5, 1.0), cost=0.05, risk_aversion=5.0)
    stay = lambda Q: np.mean([Q[m, p].argmax() == p for m in range(4) for p in range(3)])
    assert stay(dear) > stay(free) and stay(dear) == 1.0


def test_a_state_never_seen_falls_back_on_cost_only_and_inputs_are_checked():
    Q = ex.fit_exposure_policy(np.zeros(20, dtype=int), np.full(20, 0.01), (0.0, 1.0), cost=0.001, gamma=0.0)
    assert Q.shape == (4, 2, 2) and Q[3, 0, 0] == pytest.approx(0.0, abs=1e-12) and Q[3, 0, 1] == pytest.approx(-0.001)     # unseen: only the cost of moving is known
    assert Q[0, 0, 1] > Q[0, 0, 0]                                                                       # seen, and the market paid: hold more


# ------------------------------------------------------------------------------------------------------------------ allocator
@pytest.fixture(scope="module")
def bundle():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2012-01-02", periods=2000)
    common = rng.normal(0.0004, 0.01, size=(len(idx), 1))
    prices = 100 * np.exp(np.cumsum(common + rng.normal(0, 0.006, size=(len(idx), 6)), axis=0))
    return bundle_from_prices(pd.DataFrame(prices, index=idx, columns=list("ABCDEF")), min_history=60, name="rl")


def test_the_allocator_holds_an_equal_weighted_book_at_a_chosen_exposure(bundle):
    alloc = ALLOCATORS.create("q_learning_exposure")
    w = alloc.build(Context(bundle, load_config()))
    months = month_end_dates(bundle.index)
    rows = w.loc[months]
    gross = rows.sum(axis=1)
    assert set(np.round(gross.unique(), 6)) <= {0.0, 0.5, 1.0}
    assert (rows.nunique(axis=1) <= 2).all()                                                           # equal weights (zero for assets not held)
    assert (w >= 0).all().all() and np.allclose(gross.iloc[3:30], 1.0)                                     # fully invested until it has learned something


def test_the_allocator_does_not_look_ahead(bundle):
    alloc = ALLOCATORS.create("q_learning_exposure")
    w = alloc.build(Context(bundle, load_config()))
    cut = bundle.index[1500]
    prices = bundle.prices.copy()
    prices.loc[prices.index > cut] *= np.linspace(1, 3, int((prices.index > cut).sum()))[:, None]
    other = bundle_from_prices(prices, min_history=60, name="rl2")
    w2 = alloc.build(Context(other, load_config()))
    before = w.index[w.index <= cut]
    assert np.allclose(w.loc[before].to_numpy(), w2.loc[before].to_numpy(), atol=1e-12)


def test_the_allocator_validates_parameters():
    for bad in ({"exposures": (1.0,)}, {"exposures": (0.0, 1.5)}, {"min_months": 6}, {"gamma": 1.0}, {"risk_aversion": 0.0}):
        with pytest.raises(ValueError):
            ALLOCATORS.create("q_learning_exposure", **bad)
