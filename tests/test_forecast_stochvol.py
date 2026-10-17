"""The forecasters built on the option-model machinery, each on a world whose truth is known: heavy-tailed skewed shocks, a hidden log variance, a Heston market."""

from __future__ import annotations

import numpy as np
import pytest

from src.derivatives import levy
from src.forecast import stochvol
from src.forecast.registry import METHODS
from tests import forecast_worlds as W

HS = (1, 5, 21)


def forecast(method, history, n=2000, seed=0, horizons=HS):
    state = method.fit(history.tail(method.window), horizons)
    return method.predict(state, history.tail(method.lookback), horizons, n, np.random.default_rng(seed))


def test_the_new_methods_are_registered_with_their_family_but_are_not_in_the_default_set():
    from src.forecast.registry import DEFAULT, FAMILIES

    for name in ("garch_vg", "garch_nig", "heston", "sv_kalman"):
        assert name in METHODS and METHODS[name].family in FAMILIES and name not in DEFAULT
    assert METHODS["garch_vg"].heavy and METHODS["garch_nig"].heavy and METHODS["sv_kalman"].family == "State space"


# ----------------------------------------------------------------------------------------------------------------------- Levy shocks
@pytest.mark.parametrize("cls,law", [(stochvol.GarchVG, "vg"), (stochvol.GarchNIG, "nig")])
def test_the_shock_law_reproduces_the_skew_and_the_kurtosis_of_the_residuals_it_was_fitted_to_and_puts_them_in_the_forecast(cls, law):
    """The world's shocks are variance-gamma. The law is fitted to the GARCH residuals, so it should match *their* skew and kurtosis (a NIG fitted to variance-gamma shocks matches the moments and
    not the family) and both should show what the world has: a long left tail and fat tails."""
    truth = levy.VarianceGamma(sigma=0.9, theta=-0.4, nu=0.8).moments()
    h = W.to_history(W.levy_garch(3000, seed=2))
    m = cls()
    state = m.fit(h.tail(m.window), HS)
    z = state.std_resid.to_numpy()
    z = (z - z.mean()) / z.std()
    sample_skew, sample_kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)
    f = m.predict(state, h.tail(m.lookback), HS, 40_000, np.random.default_rng(0))
    assert f.info["shock_law"] == law and set(f.info["shock_law_parameters"]) >= {"sigma", "theta"}
    skew = f.info["shock_skewness"]
    assert skew < -0.4 and skew == pytest.approx(sample_skew, abs=0.6) and skew == pytest.approx(truth["skewness"], abs=0.45), (skew, sample_skew, truth["skewness"])    # the fit weighs the whole density, not the third moment, whose sampling error with tails this fat is a few tenths
    assert 1.5 < f.info["shock_excess_kurtosis"] < 3.5 * sample_kurt, (f.info["shock_excess_kurtosis"], sample_kurt, truth["excess_kurtosis"])     # a likelihood fit need not match the fourth sample moment, least of all in the wrong family
    if law == "vg":
        assert f.info["shock_excess_kurtosis"] == pytest.approx(sample_kurt, rel=0.7)
    lo, hi = np.quantile(f.at(1), [0.005, 0.995])
    assert abs(lo) > 1.1 * abs(hi), "the left tail is longer than the right, as in the world that was simulated"
    symmetric = forecast(stochvol.GarchMC(), h, n=40_000).at(1)
    slo, shi = np.quantile(symmetric, [0.005, 0.995])
    assert abs(abs(slo) - abs(shi)) < 0.12 * abs(slo), "the Student-t shocks of the plain GARCH have no skew"
    assert f.at(1).std() == pytest.approx(symmetric.std(), rel=0.2) and abs(f.at(21).mean()) < 4.5 * f.at(21).std() / np.sqrt(40_000)


def test_a_gaussian_world_gets_shocks_without_skew_or_excess_kurtosis():
    r = W.iid(3000, 0.01, seed=5)
    f = forecast(stochvol.GarchVG(), W.to_history(r), n=20_000)
    assert abs(f.info["shock_skewness"]) < 0.35 and f.info["shock_excess_kurtosis"] < 1.2


# ------------------------------------------------------------------------------------------------------------------ stochastic volatility
def test_the_stochastic_volatility_forecast_finds_the_planted_persistence_and_follows_the_volatility_regime():
    r, h = W.stochastic_volatility(3000, seed=3)
    hist = W.to_history(r)
    m = METHODS["sv_kalman"]
    f = forecast(m, hist)
    assert f.info["persistence"] == pytest.approx(0.97, abs=0.04) and f.info["volatility_of_log_variance"] == pytest.approx(0.15, rel=0.5)
    assert f.info["long_run_volatility_annual"] == pytest.approx(np.exp((-9.0 + 0.5 * 0.15 ** 2 / (1 - 0.97 ** 2)) / 2) * np.sqrt(252), rel=0.25)
    quiet = W.to_history(np.concatenate([r, np.full(30, 1e-4)]))
    wild = W.to_history(np.concatenate([r, np.random.default_rng(1).normal(0, 0.04, 30)]))
    state = m.fit(quiet.tail(m.window), HS)
    sd_quiet = m.predict(state, quiet.tail(m.lookback), HS, 4000, np.random.default_rng(0)).at(1).std()
    sd_wild = m.predict(state, wild.tail(m.lookback), HS, 4000, np.random.default_rng(0)).at(1).std()
    assert sd_wild > 2.0 * sd_quiet, "after a stormy month the one-day spread is much wider"
    assert f.info["half_life_days"] == pytest.approx(np.log(0.5) / np.log(0.97), rel=0.5)


def test_the_stochastic_volatility_forecast_is_calibrated_over_many_origins():
    r, _ = W.stochastic_volatility(2600, seed=4)
    m = METHODS["sv_kalman"]
    hist = W.to_history(r)
    state = m.fit(hist.slice(0, 1300).tail(m.window), HS)
    inside, n = 0, 0
    rng = np.random.default_rng(2)
    for t in range(1300, 2500, 4):
        h = hist.slice(0, t + 1)
        f = m.predict(state, h.tail(m.lookback), HS, 1500, rng)
        lo, hi = np.quantile(f.at(1), [0.05, 0.95])
        nxt = r[t]                                                                          # r[t] is the return that follows the price at position t
        inside += lo <= nxt <= hi
        n += 1
    assert 0.84 < inside / n < 0.96, inside / n


# ------------------------------------------------------------------------------------------------------------------------------ Heston
def test_the_method_of_moments_recovers_the_heston_parameters_on_average_and_roughly_in_each_sample():
    estimates = [stochvol.heston_moments(W.heston_returns(3500, seed=seed)[0]) for seed in range(8)]
    mean = {k: float(np.mean([e[k] for e in estimates])) for k in ("kappa", "theta", "xi", "rho")}
    assert mean["kappa"] == pytest.approx(4.0, rel=0.35) and mean["theta"] == pytest.approx(0.04, rel=0.3) and mean["xi"] == pytest.approx(0.5, rel=0.3) and mean["rho"] == pytest.approx(-0.7, abs=0.25), mean
    assert all(1.0 < e["kappa"] < 12.0 and 0.2 < e["xi"] < 0.9 and e["rho"] < -0.2 for e in estimates), estimates
    iid = stochvol.heston_moments(W.iid(3500, 0.0126, seed=1))
    assert iid["xi"] < 0.2, "returns with no volatility clustering: a volatility of variance that is small next to the planted 0.5"


def test_the_heston_forecast_follows_the_variance_and_is_centred():
    r, v = W.heston_returns(3500, seed=5)
    hist = W.to_history(r)
    m = METHODS["heston"]
    f = forecast(m, hist)
    assert f.info["long_run_volatility_annual"] == pytest.approx(np.sqrt(0.04), rel=0.3), f.info
    assert f.info["leverage_correlation"] < -0.3 and 1.0 < f.info["kappa"] < 12.0 and 0.2 < f.info["volatility_of_variance"] < 0.9
    assert f.info["feller_ratio"] > 0 and f.info["half_life_days"] > 0
    quiet = W.to_history(np.concatenate([r, np.full(30, 1e-4)]))
    wild = W.to_history(np.concatenate([r, np.random.default_rng(1).normal(0, 0.05, 30)]))
    state = m.fit(quiet.tail(m.window), HS)
    a = m.predict(state, quiet.tail(m.lookback), HS, 6000, np.random.default_rng(0))
    b = m.predict(state, wild.tail(m.lookback), HS, 6000, np.random.default_rng(0))
    assert b.at(1).std() > 2.0 * a.at(1).std() and b.at(21).std() > a.at(21).std()
    assert abs(a.at(21).mean()) < 4.5 * a.at(21).std() / np.sqrt(6000), "the draws are centred: no view on direction"
    assert b.info["volatility_today_annual"] > a.info["volatility_today_annual"]
    skew = float(np.mean(((b.at(21) - b.at(21).mean()) / b.at(21).std()) ** 3))
    assert skew < 0.0, "a negative return-variance correlation skews the longer horizons to the left"
