"""Dynamic covariance: filters, parameter recovery, forecasts, causality.

A DCC-GARCH fitted by two-step QMLE has to recover the parameters it was
simulated with before any forecast it produces means anything. After that the
tests pin down the properties a forecast needs: positive definiteness,
mean reversion at the stated rate, IGARCH safety, and, as everywhere in this
platform, that replacing the future leaves earlier forecasts untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.dynamic_covariance import (
    DCC,
    Garch11,
    average_covariance,
    dcc_correlation_forecast,
    dcc_q_path,
    fit_dcc,
    fit_garch11,
    fit_ogarch,
    garch_filter,
    garch_variance_path,
    gmv_realised_variance,
    ogarch_forecast,
    qlike_loss,
    realised_second_moment,
    static_forecasts,
    walk_forward_dynamic_covariance,
)

RBAR = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])


def simulate_dcc(T: int, seed: int, omega=0.02, alpha=0.08, beta=0.90, a=0.04, b=0.94) -> np.ndarray:
    """Daily returns (decimals) from a DCC-GARCH with known parameters."""
    rng = np.random.default_rng(seed)
    N = RBAR.shape[0]
    Q = RBAR.copy()
    sigma2 = np.full(N, omega / (1 - alpha - beta))
    eps = np.zeros(N)
    out = np.empty((T, N))
    for t in range(T):
        d = np.sqrt(np.diag(Q))
        R = Q / np.outer(d, d)
        z = np.linalg.cholesky(R) @ rng.standard_normal(N)
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2
        eps = np.sqrt(sigma2) * z
        out[t] = eps
        Q = (1 - a - b) * RBAR + a * np.outer(z, z) + b * Q
    return out / 100.0


def _garch(omega=0.05, alpha=0.08, beta=0.90, mu=0.0, sigma2_last=1.0, eps_last=0.5, scale=1.0) -> Garch11:
    return Garch11(mu=mu, omega=omega, alpha=alpha, beta=beta, scale=scale, sigma2_last=sigma2_last,
                   eps_last=eps_last, std_resid=np.zeros(1), sigma2=np.ones(1))


# ------------------------------------------------------------------------ GARCH pieces
def test_garch_filter_equals_the_direct_recursion():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    g = _garch(mu=0.1)
    sigma2, z = garch_filter(x, g)
    s2, e = g.sigma2_last, g.eps_last
    for t in range(len(x)):
        s2 = g.omega + g.alpha * e ** 2 + g.beta * s2
        assert sigma2[t] == pytest.approx(s2, rel=1e-12)
        e = x[t] - g.mu
        assert z[t] == pytest.approx(e / np.sqrt(s2), rel=1e-12)


def test_garch_variance_path_starts_at_the_next_day_decays_to_the_long_run_and_igarch_does_not_explode():
    g = _garch(omega=0.05, alpha=0.08, beta=0.90)
    path = garch_variance_path(g, 4.0, 500)
    assert path[0] == 4.0
    assert path[-1] == pytest.approx(g.omega / (1 - g.persistence), rel=1e-3)
    assert np.all(np.diff(path) < 0)                               # a high start decays monotonically
    integrated = _garch(omega=0.01, alpha=0.10, beta=0.90)          # persistence exactly one
    grown = garch_variance_path(integrated, 2.0, 21)
    assert np.all(np.isfinite(grown)) and grown[-1] == pytest.approx(2.0 + 20 * 0.01)
    assert grown[-1] < 2.5                                         # linear in omega, nothing like a divergence


def test_fit_garch11_recovers_persistence():
    rng = np.random.default_rng(1)
    r = simulate_dcc(4000, seed=1)[:, 0]
    g = fit_garch11(r)
    assert 0.95 < g.persistence < 1.0
    assert len(g.std_resid) == 4000 and abs(np.std(g.std_resid) - 1.0) < 0.05


# ------------------------------------------------------------------------------ DCC
def test_q_path_equals_a_loop_and_stays_positive_definite():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((200, 3))
    Qbar = np.cov(z, rowvar=False)
    a, b = 0.05, 0.9
    Q = dcc_q_path(z, a, b, Qbar)
    assert Q.shape == (201, 3, 3)
    current = Qbar.copy()
    for t in range(200):
        current = (1 - a - b) * Qbar + a * np.outer(z[t], z[t]) + b * current
        np.testing.assert_allclose(Q[t + 1], current, atol=1e-12)
    assert all(np.linalg.eigvalsh(q).min() > 0 for q in Q)


def test_dcc_recovers_the_parameters_it_was_simulated_with():
    r = simulate_dcc(5000, seed=3, a=0.04, b=0.94)
    fits = [fit_garch11(r[:, i]) for i in range(3)]
    z = np.column_stack([g.std_resid for g in fits])
    dcc = fit_dcc(z)
    assert dcc.success
    assert abs(dcc.a - 0.04) < 0.015 and abs(dcc.b - 0.94) < 0.025
    assert abs((dcc.a + dcc.b) - 0.98) < 0.02


def test_dcc_correlation_forecast_starts_at_the_next_day_and_reverts_to_the_unconditional_matrix():
    Qbar = RBAR.copy()
    dcc = DCC(a=0.04, b=0.94, Qbar=Qbar, loglik=0.0, success=True)
    Q_next = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    forecast = dcc_correlation_forecast(Q_next, dcc, 1000)
    d = np.sqrt(np.diag(Q_next))
    np.testing.assert_allclose(forecast[0], Q_next / np.outer(d, d), atol=1e-12)
    np.testing.assert_allclose(forecast[-1], Qbar, atol=1e-6)
    distance = np.abs(forecast - Qbar).max(axis=(1, 2))
    assert np.all(np.diff(distance) <= 1e-12)                       # monotone reversion
    np.testing.assert_allclose(np.diagonal(forecast, axis1=1, axis2=2), 1.0, atol=1e-12)


def test_average_covariance_is_d_r_d_at_horizon_one_and_an_average_beyond():
    sigma2 = np.array([[4.0, 1.0, 9.0]])
    R = np.array([RBAR])
    cov = average_covariance(sigma2, R, scale=np.array([2.0, 1.0, 1.0]))
    sd = np.sqrt([4.0 / 4.0, 1.0, 9.0])                              # variance / scale^2 in return units
    np.testing.assert_allclose(cov, np.outer(sd, sd) * RBAR, atol=1e-12)
    two = average_covariance(np.vstack([sigma2, 4 * sigma2]), np.vstack([R, R]), scale=np.ones(3))
    np.testing.assert_allclose(two, 0.5 * (np.outer(np.sqrt(sigma2[0]), np.sqrt(sigma2[0])) * RBAR
                                           + np.outer(2 * np.sqrt(sigma2[0]), 2 * np.sqrt(sigma2[0])) * RBAR))


# ------------------------------------------------------------------------- O-GARCH
def test_ogarch_forecast_is_symmetric_positive_definite_and_recovers_the_covariance_with_all_factors():
    r = simulate_dcc(3000, seed=4)
    model = fit_ogarch(r, n_factors=3)
    assert np.all(model.psi > 0)
    np.testing.assert_allclose(model.weights.T @ model.weights, np.eye(3), atol=1e-10)
    long_run = np.array([g.omega / (1 - g.persistence) * 1.0 for g in model.garch])
    forecast = ogarch_forecast(model, long_run, horizon=21)
    np.testing.assert_allclose(forecast, forecast.T, atol=1e-15)
    assert np.linalg.eigvalsh(forecast).min() > 0
    sample = np.cov(r, rowvar=False)
    assert np.linalg.norm(forecast - sample) / np.linalg.norm(sample) < 0.25


# --------------------------------------------------------------- walk-forward causality
def _origins(index: pd.DatetimeIndex, start: int, step: int = 21) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index[start::step])


def test_forecasts_do_not_depend_on_anything_after_the_origin():
    r = simulate_dcc(1500, seed=5)
    index = pd.bdate_range("2010-01-04", periods=1500)
    frame = pd.DataFrame(r, index=index, columns=["a", "b", "c"])
    origins = _origins(index, 600)
    settings = dict(min_train=600, refit_every=300, horizon=21, n_factors=2)
    base = walk_forward_dynamic_covariance(frame, origins, **settings)
    cut = 1100
    altered = frame.copy()
    altered.iloc[cut + 1:] = 3.0 * np.random.default_rng(9).standard_normal(altered.iloc[cut + 1:].shape) * 0.01
    changed = walk_forward_dynamic_covariance(altered, origins, **settings)
    before = np.flatnonzero(base.origins <= index[cut])
    after = np.flatnonzero(base.origins > index[cut])
    assert len(before) > 10 and len(after) > 5
    for model in ("dcc", "ogarch"):
        np.testing.assert_allclose(base.forecasts[model][before], changed.forecasts[model][before], atol=1e-12)
        assert not np.allclose(base.forecasts[model][after], changed.forecasts[model][after])
        for forecast in base.forecasts[model]:
            assert np.linalg.eigvalsh(forecast).min() > 0
    assert (base.diagnostics["dcc_persistence"] < 1.0).all()


def test_each_refit_uses_only_rows_before_its_window():
    """Origins before min_train get no forecast; the first window's parameters equal a direct fit on rows < start."""
    r = simulate_dcc(1300, seed=6)
    index = pd.bdate_range("2010-01-04", periods=1300)
    frame = pd.DataFrame(r, index=index, columns=["a", "b", "c"])
    origins = _origins(index, 300)
    result = walk_forward_dynamic_covariance(frame, origins, min_train=600, refit_every=400, horizon=21,
                                             models=("dcc",))
    assert (result.origins >= index[600]).all()
    fits = [fit_garch11(r[:600, i]) for i in range(3)]
    direct = fit_dcc(np.column_stack([g.std_resid for g in fits]))
    row = result.diagnostics.iloc[0]
    assert row["dcc_a"] == pytest.approx(direct.a, abs=1e-6) and row["dcc_b"] == pytest.approx(direct.b, abs=1e-6)


# --------------------------------------------------------------- static estimators and losses
def test_static_forecasts_use_data_through_the_origin_and_no_later():
    r = simulate_dcc(900, seed=7)
    index = pd.bdate_range("2010-01-04", periods=900)
    frame = pd.DataFrame(r, index=index, columns=["a", "b", "c"])
    origins = _origins(index, 300, 50)
    spec = {"sample": {"lookback": 252}, "ewma": {"halflife": 60}, "shrinkage": {"lookback": 252}}
    base = static_forecasts(frame, origins, spec)
    altered = frame.copy()
    altered.iloc[501:] *= 5.0
    changed = static_forecasts(altered, origins, spec)
    early = np.flatnonzero(origins <= index[500])
    for name in spec:
        np.testing.assert_allclose(base[name][early], changed[name][early], atol=1e-14)
        assert np.linalg.eigvalsh(base[name][0]).min() > 0
    position = index.get_loc(origins[0])
    window = frame.iloc[position - 251: position + 1]                 # 252 rows, the origin day included
    np.testing.assert_allclose(base["sample"][0], np.cov(window.to_numpy(), rowvar=False), atol=1e-8)


def test_realised_second_moment_averages_exactly_the_next_h_days():
    index = pd.bdate_range("2020-01-01", periods=12)
    frame = pd.DataFrame({"a": np.arange(1.0, 13.0), "b": np.arange(1.0, 13.0) * 2}, index=index)
    out = realised_second_moment(frame, index[[2, 10]], horizon=3)
    future = frame.iloc[3:6].to_numpy()
    np.testing.assert_allclose(out[0], future.T @ future / 3)
    assert np.isnan(out[1]).all()                                     # fewer than 3 future days


def test_qlike_prefers_the_truth_and_gmv_variance_prefers_the_better_forecast():
    rng = np.random.default_rng(3)
    truth = np.array([[1.0, 0.4, 0.1], [0.4, 2.0, 0.3], [0.1, 0.3, 0.5]])
    draws = rng.multivariate_normal(np.zeros(3), truth, size=4000)
    realised = (draws.T @ draws / 4000)[None]
    good = qlike_loss(truth[None], realised)[0]
    assert good < qlike_loss((1.5 * truth)[None], realised)[0]
    assert good < qlike_loss((0.6 * truth)[None], realised)[0]
    assert good < qlike_loss(np.diag(np.diag(truth))[None], realised)[0]
    assert gmv_realised_variance(truth[None], realised)[0] <= gmv_realised_variance(np.eye(3)[None], realised)[0]


# ------------------------------------------------- the whole chain: forecast, then score
def _mean_losses(frame: pd.DataFrame) -> dict:
    origins = _origins(frame.index, 750)
    dynamic = walk_forward_dynamic_covariance(frame, origins, min_train=750, refit_every=252, horizon=21,
                                              n_factors=2, models=("dcc",))
    static = static_forecasts(frame, dynamic.origins,
                              {"sample": {"lookback": 252}, "ewma": {"halflife": 60}})
    realised = realised_second_moment(frame, dynamic.origins, 21)
    return {name: float(np.nanmean(qlike_loss(f, realised)))
            for name, f in {**dynamic.forecasts, **static}.items()}


def test_dcc_wins_on_the_loss_when_the_truth_is_dynamic():
    r = simulate_dcc(3200, seed=11, omega=0.02, alpha=0.12, beta=0.86, a=0.06, b=0.92)
    frame = pd.DataFrame(r, index=pd.bdate_range("2008-01-02", periods=len(r)), columns=["a", "b", "c"])
    loss = _mean_losses(frame)
    assert loss["dcc"] < loss["sample"] - 0.3 and loss["dcc"] < loss["ewma"] - 0.1


def test_dcc_does_not_win_by_construction_when_the_truth_is_constant():
    rng = np.random.default_rng(11)
    r = rng.multivariate_normal(np.zeros(3), RBAR * 1e-4, size=3200)
    frame = pd.DataFrame(r, index=pd.bdate_range("2008-01-02", periods=len(r)), columns=["a", "b", "c"])
    loss = _mean_losses(frame)
    assert abs(loss["dcc"] - loss["sample"]) < 0.15            # a tie: the model has nothing to find
