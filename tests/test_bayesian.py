"""Bayesian portfolio construction: the posterior is right, the prior behaves, the book is causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.bayesian import (
    bayesian_book,
    bayesian_weights,
    jorion_prior,
    niw_posterior,
    predictive_quantiles,
    weight_distribution,
)
from src.portfolio.constraints import Constraints


def _returns(n=400, k=6, seed=0, mean=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n, 1))
    data = mean + vol * (0.6 * common + 0.8 * rng.standard_normal((n, k)))
    return pd.DataFrame(data, index=pd.bdate_range("2015-01-01", periods=n), columns=[f"A{i}" for i in range(k)])


def _constraints():
    return Constraints(min_weight=0.0, max_weight=0.6, gross_leverage=1.0, net_exposure=1.0, group_limits={}, group_map={})


def test_posterior_update_matches_the_closed_form():
    window = _returns(300, 4, seed=1)
    post = niw_posterior(window, nu0=50.0)
    values = window.to_numpy()
    n_obs, k = values.shape
    xbar = values.mean(axis=0)
    # kappa_n and nu_n add the sample size to the prior's
    assert post.kappa == pytest.approx(post.kappa0 + n_obs)
    assert post.nu == pytest.approx(50.0 + n_obs)
    # the posterior mean is the kappa-weighted average of the prior mean and the sample mean
    m0 = (post.mean * post.kappa - n_obs * xbar) / post.kappa0
    assert np.allclose(m0, m0[0])                                    # the prior mean is a common value
    assert np.allclose(post.scale, post.scale.T)
    assert np.all(np.linalg.eigvalsh(post.scale) > 0)


def test_jorion_shrinks_hard_when_means_are_indistinguishable_and_softly_when_they_are_not():
    rng = np.random.default_rng(2)
    n_obs, k = 250, 8
    cov = np.eye(k) * 1e-4
    same = np.full(k, 3e-4) + rng.normal(0, 1e-6, k)
    spread = np.linspace(-2e-3, 2e-3, k)
    _, w_same = jorion_prior(same, cov, n_obs)
    _, w_spread = jorion_prior(spread, cov, n_obs)
    assert w_same > 0.99 and w_spread < 0.3


def test_posterior_draws_have_the_posterior_mean_and_covariance():
    post = niw_posterior(_returns(500, 4, seed=3), nu0=60.0)
    mus, sigmas = post.draw(4000, np.random.default_rng(4))
    assert np.allclose(sigmas.mean(axis=0), post.mean_covariance(), rtol=0.08, atol=2e-6)
    assert np.allclose(mus.mean(axis=0), post.mean, atol=4 * np.sqrt(np.diag(post.mean_covariance()).max() / post.kappa / 4000) + 1e-5)


def test_more_data_means_a_tighter_weight_distribution():
    short = _returns(120, 5, seed=5)
    long = _returns(1200, 5, seed=5).iloc[-1000:]
    con = _constraints()
    d_short = weight_distribution(short, con, n_draws=60)
    d_long = weight_distribution(long, con, n_draws=60)
    assert (d_short["p95"] - d_short["p05"]).mean() >= (d_long["p95"] - d_long["p05"]).mean() - 1e-9


def test_every_kind_returns_feasible_weights():
    window = _returns(260, 6, seed=6)
    con = _constraints()
    for kind in ("mvo_sample", "bayes_stein", "bayes_predictive"):
        w = bayesian_weights(window, kind, con, n_draws=20)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-9).all() and (w <= 0.6 + 1e-6).all()


def test_the_bayesian_book_is_causal():
    returns = _returns(420, 5, seed=7)
    investable = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    marks = returns.index[[300, 330, 360]]
    con = _constraints()
    kwargs = dict(constraints=con, rebalance_index=marks, lookback=252, n_draws=10)
    a = bayesian_book(returns, investable, "bayes_predictive", **kwargs)
    changed = returns.copy()
    changed.iloc[350:] = changed.iloc[350:] * 3.0 + 0.01             # replace the future of the third mark's window... and beyond
    b = bayesian_book(changed, investable, "bayes_predictive", **kwargs)
    assert np.allclose(a.loc[:returns.index[349]].to_numpy(), b.loc[:returns.index[349]].to_numpy())


def test_predictive_intervals_are_nested_and_ordered():
    window = _returns(300, 4, seed=8)
    w = pd.Series(0.25, index=window.columns)
    q = predictive_quantiles(window, w, levels=(0.5, 0.9), n_draws=800)
    assert q["lo_0.9"] < q["lo_0.5"] < q["median"] < q["hi_0.5"] < q["hi_0.9"]
