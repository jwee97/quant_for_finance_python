"""The combination engine: scales agree, ICs are only used once known, trust weights are well formed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.alpha_engine import (
    combine_alphas, forward_returns, matured_ic, standardise_alpha, trailing_ir, trust_weights)


def _panel(n=600, k=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, k)), idx, cols)
    return idx, cols, returns


def test_standardised_alphas_have_zero_mean_unit_scale_and_are_clipped():
    idx, cols, _ = _panel()
    raw = pd.DataFrame(np.random.default_rng(1).standard_t(3, (len(idx), len(cols))) * 50 + 7, idx, cols)
    z = standardise_alpha(raw)
    assert z.mean(axis=1).abs().max() < 1e-9
    assert z.abs().max().max() <= 3.0 + 1e-12


def test_forward_return_is_the_return_after_the_date_and_ic_is_known_only_h_days_later():
    idx, cols, returns = _panel()
    fwd = forward_returns(returns, 5)
    t = 100
    expected = (1.0 + returns.iloc[t + 1:t + 6]).prod() - 1.0
    assert np.allclose(fwd.iloc[t].to_numpy(), expected.to_numpy())
    signal = fwd.shift(5).fillna(0.0)                         # a signal built from the past, to have an IC
    ic = matured_ic(signal, fwd, 5)
    # the IC value that sits on date t was computed for date t-5 and is therefore based on returns through t
    raw_ic = matured_ic(signal, fwd, 0)
    assert np.allclose(ic.iloc[200:260].dropna().to_numpy(), raw_ic.shift(5).iloc[200:260].dropna().to_numpy())


def test_trailing_ir_does_not_see_the_future():
    s = pd.Series(np.random.default_rng(2).normal(0.05, 1, 800), pd.bdate_range("2015-01-01", periods=800))
    a = trailing_ir(s, 200, 100)
    b = s.copy()
    b.iloc[600:] += 5.0
    assert np.allclose(a.iloc[:600].dropna().to_numpy(), trailing_ir(b, 200, 100).iloc[:600].dropna().to_numpy())


def test_trust_weights_are_a_distribution_fall_back_to_equal_and_shrink():
    idx = pd.bdate_range("2020-01-01", periods=10)
    scores = pd.DataFrame({"a": [np.nan] * 3 + [1.0] * 7, "b": [np.nan] * 3 + [3.0] * 7, "c": [np.nan] * 3 + [-1.0] * 7}, idx)
    w = trust_weights(scores, idx[[3, 6]], shrink=0.0)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w.iloc[0], 1 / 3)                                   # before any score exists: equal
    assert np.allclose(w.iloc[5].to_numpy(), [0.25, 0.75, 0.0])           # positive part, proportional
    shrunk = trust_weights(scores, idx[[3]], shrink=0.5)
    assert np.allclose(shrunk.iloc[5].to_numpy(), 0.5 * np.array([0.25, 0.75, 0.0]) + 0.5 / 3)
    losing = scores.copy()
    losing[["a", "b", "c"]] = -1.0
    assert np.allclose(trust_weights(losing, idx[[3]]).iloc[5], 1 / 3)     # nothing to trust: stay diversified


def test_combining_forecasts_nets_opposing_trades_before_they_are_sized():
    idx, cols, returns = _panel(400, 6, 3)
    z = standardise_alpha(pd.DataFrame(np.random.default_rng(4).standard_normal((400, 6)), idx, cols))
    opposite = -z
    weights = pd.DataFrame(0.5, idx, ["m", "r"])
    combined = combine_alphas({"m": z, "r": opposite}, weights)
    assert np.allclose(combined.fillna(0).to_numpy(), 0.0)                 # fully offsetting forecasts: no trade at all
    one = combine_alphas({"m": z, "r": z * np.nan}, weights)           # a missing alpha: the other carries full weight
    assert np.allclose(one.fillna(0).to_numpy(), z.fillna(0).to_numpy())
