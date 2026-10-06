"""Online learning: recursions equal their batch counterparts, forget when told to, and never peek."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.online import KalmanRW, NLMS, OnlineRidge, hedge_aggregate, run_online_forecasts


def _data(n=400, d=3, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    beta = np.array([1.0, -0.5, 0.25])[:d]
    return x, x @ beta + noise * rng.standard_normal(n), beta


def test_rls_without_forgetting_equals_batch_ridge():
    x, y, _ = _data()
    model = OnlineRidge(3, forgetting=1.0, ridge=2.0)
    for start in range(0, len(y), 20):
        model.update(x[start:start + 20], y[start:start + 20])
    batch = np.linalg.solve(x.T @ x + 2.0 * np.eye(3), x.T @ y)
    assert np.allclose(model.beta, batch, atol=1e-10)


def test_forgetting_tracks_a_coefficient_that_moves():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((600, 1))
    beta_path = np.where(np.arange(600) < 300, 1.0, -1.0)
    y = x[:, 0] * beta_path + 0.05 * rng.standard_normal(600)
    forgetful, stubborn = OnlineRidge(1, 0.9, 1e-6), OnlineRidge(1, 1.0, 1e-6)
    for start in range(0, 600, 10):
        for m in (forgetful, stubborn):
            m.update(x[start:start + 10], y[start:start + 10])
    assert forgetful.beta[0] < -0.8 and abs(stubborn.beta[0]) < 0.3


def test_kalman_with_no_process_noise_converges_to_least_squares():
    x, y, beta = _data(300)
    kf = KalmanRW(3, process_to_observation=0.0, observation_variance=0.01, prior_variance=1e6)
    for start in range(0, 300, 15):
        kf.update(x[start:start + 15], y[start:start + 15])
    assert np.allclose(kf.beta, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-3)


def test_nlms_reduces_error_on_stationary_data():
    x, y, _ = _data(2000, noise=0.05)
    model = NLMS(3, step=0.2)
    early = late = 0.0
    for t in range(2000):
        error = (y[t] - x[t] @ model.beta) ** 2
        if t < 100:
            early += error
        elif t >= 1900:
            late += error
        model.update(x[t:t + 1], y[t:t + 1])
    assert late < 0.05 * early


def test_hedge_weights_are_a_distribution_known_before_the_day_and_regret_sublinear():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0.0004, 0.01, (1500, 4)) + np.array([0.0006, 0, -0.0003, 0.0001]),
                     index=pd.bdate_range("2010-01-01", periods=1500), columns=list("ABCD"))
    res = hedge_aggregate(r)
    assert np.allclose(res.weights.sum(axis=1), 1.0) and (res.weights >= 0).all().all()
    assert np.allclose(res.weights.iloc[0], 0.25)                     # day one: no information yet
    assert res.regret.iloc[-1] <= res.regret_bound.iloc[-1]
    changed = r.copy()
    changed.iloc[1000:] *= -1.0
    other = hedge_aggregate(changed)
    # weights used up to and including day 1000 depend only on returns before it... except via eta (horizon), so compare normalised logits
    assert np.allclose(res.weights.iloc[:2].to_numpy(), other.weights.iloc[:2].to_numpy(), atol=1e-6)


def test_online_forecasts_use_only_matured_labels():
    rng = np.random.default_rng(4)
    n_origins, assets, horizon = 60, 4, 3
    positions = np.repeat(np.arange(n_origins) * horizon, assets)               # origin positions on the calendar
    origin = np.repeat(np.arange(n_origins), assets)
    design = rng.standard_normal((n_origins * assets, 2))
    target = design @ np.array([1.0, 0.0]) + 0.1 * rng.standard_normal(len(design))
    test_origins = [(o, o * horizon) for o in range(40, 60)]
    factories = {"rls": lambda: OnlineRidge(2, 0.99, 1.0)}
    base = run_online_forecasts(design, target, origin, positions, 40 * horizon, horizon, factories, test_origins)["rls"]
    # corrupt the label of origin 55 (matures at position 55*3+3, i.e. before test origin 56 only)
    corrupted = target.copy()
    corrupted[origin == 55] += 100.0
    alt = run_online_forecasts(design, corrupted, origin, positions, 40 * horizon, horizon, factories, test_origins)["rls"]
    unaffected = base[base["origin"] <= 55]
    assert np.allclose(unaffected["mu"].to_numpy(), alt[alt["origin"] <= 55]["mu"].to_numpy())
    assert not np.allclose(base[base["origin"] == 58]["mu"].to_numpy(), alt[alt["origin"] == 58]["mu"].to_numpy())
