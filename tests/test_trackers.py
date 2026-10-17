"""The two scalar Kalman trackers: the reversion speed of a spread whose behaviour changes, and the volatility of a return series, each checked against the process that generated the data, for
causality (a value through ``t`` must not depend on anything after it) and for the honesty of their own uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from src.econometrics import trackers


def _ar1(phi_path, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(len(phi_path) + 1)
    for t, phi in enumerate(phi_path):
        x[t + 1] = phi * x[t] + sd * rng.standard_normal()
    return x[1:]


# ------------------------------------------------------------------------------------------------------------------------------ AR(1) tracker
def test_the_ar1_tracker_follows_a_change_in_the_speed_of_reversion_and_its_uncertainty_shrinks_with_data():
    x = _ar1(np.r_[np.full(1500, 0.90), np.full(1500, 0.98)], seed=3)
    track = trackers.kalman_ar1(x)
    assert track.phi[700:1400].mean() == pytest.approx(0.90, abs=0.03)
    assert track.phi[2400:].mean() == pytest.approx(0.98, abs=0.025), "after the change it settles on the new coefficient"
    assert track.phi_sd[100] > track.phi_sd[1400] > 0 and np.isnan(track.phi[0])
    half = track.half_life(1.0)                                       # in steps (dt = 1)
    assert 4.0 < np.nanmedian(half[700:1400]) < 9.5 and np.nanmedian(half[2400:]) > 20, "the half-life lengthens from about 6.6 steps to about 34"
    z = track.innovation[np.isfinite(track.innovation)]
    assert abs(z.mean()) < 0.1 and z.std() == pytest.approx(1.0, rel=0.1), "the one-step innovations are standardised noise when the model holds"
    assert track.q > 0 and track.r > 0 and np.isfinite(track.loglik)


def test_with_a_fixed_coefficient_the_tracker_settles_near_it_and_a_random_walk_has_no_reversion():
    x = _ar1(np.full(3000, 0.95), seed=5)
    track = trackers.kalman_ar1(x)
    assert track.phi[-1] == pytest.approx(0.95, abs=0.03) and abs(track.phi[-1] - 0.95) < 4 * track.phi_sd[-1] + 0.02
    walk = np.cumsum(np.random.default_rng(1).standard_normal(3000))
    t = trackers.kalman_ar1(walk)
    assert np.nanmean(t.phi[1000:]) > 0.97, "a random walk shows a coefficient at one: no half-life"
    theta = trackers.kalman_ar1(x).theta(1 / 252)
    assert np.nanmedian(theta[1000:]) == pytest.approx(-np.log(0.95) * 252, rel=0.25)
    explicit = trackers.kalman_ar1(x, q=1e-6, r=1.0)
    assert explicit.q == 1e-6 and explicit.r == 1.0 and abs(explicit.phi[-1] - 0.95) < 0.04, "given noise levels are used as they are, in the units of the series"


def test_the_ar1_tracker_is_causal_and_handles_gaps():
    x = _ar1(np.full(600, 0.9), seed=7)
    full = trackers.kalman_ar1(x, q=1e-5, r=1.0)
    altered = x.copy()
    altered[400:] += 50.0
    other = trackers.kalman_ar1(altered, q=1e-5, r=1.0)
    assert np.allclose(full.phi[:399], other.phi[:399], equal_nan=True) and np.allclose(full.prediction[:400], other.prediction[:400], equal_nan=True), "nothing after t changes the value at t"
    gap = x.copy()
    gap[300:305] = np.nan
    with_gap = trackers.kalman_ar1(gap, q=1e-5, r=1.0)
    assert np.isfinite(with_gap.phi[310]) and with_gap.phi_sd[304] > full.phi_sd[304], "a gap widens the uncertainty and the filter carries on"
    with pytest.raises(ValueError):
        trackers.kalman_ar1(x[:10])


# ------------------------------------------------------------------------------------------------------------------------------- SV tracker
def _simulate_sv(n, mu=-9.0, phi=0.97, s_eta=0.15, seed=0):
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = mu
    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + s_eta * rng.standard_normal()
    return np.exp(h / 2.0) * rng.standard_normal(n), h


def test_the_sv_tracker_recovers_the_volatility_process_and_its_parameters():
    r, h = _simulate_sv(4000, seed=3)
    fit = trackers.sv_tracker(r)
    assert fit.converged and fit.n == 4000
    assert fit.mu == pytest.approx(-9.0, abs=0.35) and fit.phi == pytest.approx(0.97, abs=0.04) and fit.sigma_eta == pytest.approx(0.15, rel=0.4)
    true_vol = np.exp(h / 2.0)
    stationary_var = fit.sigma_eta ** 2 / (1 - fit.phi ** 2)
    achieved = np.corrcoef(fit.h[200:], h[200:])[0, 1]
    promised = np.sqrt(1.0 - fit.p[200:].mean() / stationary_var)               # the accuracy the filter's own variance claims: R^2 = 1 - P / Var(h)
    assert achieved > 0.5 and achieved == pytest.approx(promised, abs=0.08), "the filter is as accurate as it says it is (with noise this large, about 0.6 correlation with the true log variance)"
    assert fit.vol.shape == (4000,) and np.all(fit.vol > 0) and np.all(fit.p > 0) and np.all(fit.p < stationary_var + 1e-9), "the filter always knows more than the prior"
    ewma = np.sqrt(np.convolve(r ** 2, np.ones(21) / 21, mode="full")[:4000])
    err_sv = np.mean((np.log(fit.vol[300:]) - np.log(true_vol[300:])) ** 2)
    err_ewma = np.mean((np.log(ewma[300:] + 1e-12) - np.log(true_vol[300:])) ** 2)
    assert err_sv < err_ewma, "a little better than a 21-day rolling volatility at following the true level"


def test_the_sv_filter_is_causal_and_the_variance_forecast_reverts_to_the_unconditional_level():
    r, _ = _simulate_sv(1500, seed=4)
    fit = trackers.sv_tracker(r)
    again = trackers.sv_tracker(np.r_[r[:1000], np.random.default_rng(0).standard_normal(500) * 0.05])
    h_full, p_full, _ = trackers._sv_filter(np.log(r ** 2 + fit.offset), fit.mu, fit.phi, fit.sigma_eta)
    h_short, p_short, _ = trackers._sv_filter(np.log(r[:1000] ** 2 + fit.offset), fit.mu, fit.phi, fit.sigma_eta)
    assert np.allclose(h_full[:1000], h_short) and np.allclose(p_full[:1000], p_short), "the state through t does not depend on later returns"
    assert again.n == 1500
    f = fit.forecast_variance(2000)
    assert f.shape == (2000,) and f[-1] == pytest.approx(np.exp(fit.mu + 0.5 * fit.sigma_eta ** 2 / (1 - fit.phi ** 2)), rel=1e-6)
    far_high = fit.h[-1] > fit.mu
    assert (f[0] > f[-1]) == far_high or abs(fit.h[-1] - fit.mu) < 0.05, "from above the unconditional level the forecast falls towards it, from below it rises"
    draws = fit.simulate_log_variance(5, 400_000, seed=1)
    assert draws.shape == (400_000, 5)
    mc = np.exp(draws).mean(axis=0)
    assert np.allclose(mc, fit.forecast_variance(5), rtol=0.02), "simulated log variances reproduce the closed-form expected variance"


def test_the_sv_tracker_copes_with_zero_returns_and_short_samples():
    r, _ = _simulate_sv(600, seed=8)
    r[::50] = 0.0                                                      # closed-market days
    fit = trackers.sv_tracker(r)
    assert np.all(np.isfinite(fit.vol)) and fit.offset > 0
    with pytest.raises(ValueError):
        trackers.sv_tracker(r[:50])
    assert set(fit.params()) == {"mu", "phi", "sigma_eta"}
