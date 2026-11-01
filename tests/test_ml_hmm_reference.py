"""P0-10: the HMM's filtered probabilities against an independent forward recursion in log space, causality of the filter, and state ordering and reload continuity."""

import pickle

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.ml.regimes import STATE_COLUMNS, HMMEngine

START = np.array([0.7, 0.3])
TRANS = np.array([[0.95, 0.05], [0.10, 0.90]])
MEANS = np.array([[0.0005, -3.0, -4.0], [-0.001, -1.5, -3.2]])
COVARS = np.array([np.diag([1e-4, 0.05, 0.05]), np.diag([4e-4, 0.08, 0.08])])


def sample(n, seed):
    rng = np.random.default_rng(seed)
    s, Z = rng.choice(2, p=START), []
    for _ in range(n):
        Z.append(rng.multivariate_normal(MEANS[s], COVARS[s]))
        s = rng.choice(2, p=TRANS[s])
    return np.array(Z)


def engine():
    e = HMMEngine(2)
    e.means_, e.covars_, e.trans_, e.start_ = MEANS.copy(), COVARS.copy(), TRANS.copy(), START.copy()
    return e


def forward_log(Z):
    """alpha_t(j) = log p(z_1..z_t, s_t=j) by the textbook recursion in log space, then normalised at each t: the FILTERED distribution."""
    logb = np.array([[multivariate_normal(MEANS[j], COVARS[j]).logpdf(z) for j in range(2)] for z in Z])
    la = np.log(START) + logb[0]
    out = [np.exp(la - logsumexp(la))]
    for t in range(1, len(Z)):
        la = logsumexp(la[:, None] + np.log(TRANS), axis=0) + logb[t]
        out.append(np.exp(la - logsumexp(la)))
    return np.array(out)


def test_the_filter_matches_an_independent_log_space_forward_recursion():
    Z = sample(300, 1)
    assert np.allclose(engine().filter(Z), forward_log(Z), atol=1e-10)


def test_the_filter_is_stable_for_a_very_long_run_and_for_extreme_observations():
    Z = sample(3000, 2)
    Z[1500] = [5.0, 8.0, 8.0]                                                            # far in the tail of both states: the likelihoods underflow in plain space
    got = engine().filter(Z)
    assert np.isfinite(got).all() and np.allclose(got.sum(axis=1), 1.0) and np.allclose(got, forward_log(Z), atol=1e-8)


def smoothed(Z):
    """Forward-backward posterior of every day given ALL the data: retrospective, so diagnostic only."""
    logb = np.array([[multivariate_normal(MEANS[j], COVARS[j]).logpdf(z) for j in range(2)] for z in Z])
    n = len(Z)
    la = np.zeros((n, 2))
    la[0] = np.log(START) + logb[0]
    for t in range(1, n):
        la[t] = logsumexp(la[t - 1][:, None] + np.log(TRANS), axis=0) + logb[t]
    lb = np.zeros((n, 2))
    for t in range(n - 2, -1, -1):
        lb[t] = logsumexp(np.log(TRANS) + logb[t + 1] + lb[t + 1], axis=1)
    g = la + lb
    return np.exp(g - logsumexp(g, axis=1, keepdims=True))


def test_a_probability_on_day_t_depends_on_observations_through_t_only_and_smoothing_would_not():
    Z = sample(200, 3)
    t = 100
    Z[t] = 0.5 * (MEANS[0] + MEANS[1])                                                   # an ambiguous day, so that what comes after it could change a retrospective belief
    calm, stressed = Z.copy(), Z.copy()
    calm[t + 1:] = MEANS[0]
    stressed[t + 1:] = MEANS[1]
    assert np.allclose(engine().filter(calm)[:t + 1], engine().filter(stressed)[:t + 1])
    assert abs(smoothed(calm)[t, 0] - smoothed(stressed)[t, 0]) > 0.05, "a smoothed belief moves with the future, which is why it is never used to trade"
    assert np.allclose(smoothed(calm)[-1], engine().filter(calm)[-1], atol=1e-9), "on the last day the two agree: nothing later to learn from"


def test_fitted_states_are_ordered_by_volatility_and_a_reloaded_engine_filters_identically():
    pytest.importorskip("hmmlearn")
    Z = sample(1500, 4)
    fitted = HMMEngine(2, seed=0).fit(Z)
    vol = STATE_COLUMNS.index("log_vol")
    assert fitted.means_[0, vol] < fitted.means_[1, vol]
    again = pickle.loads(pickle.dumps(fitted))
    assert np.array_equal(fitted.filter(Z[:400]), again.filter(Z[:400]))
    assert np.allclose(fitted.filter(Z).sum(axis=1), 1.0)
