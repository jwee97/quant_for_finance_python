"""Stochastic differential equations: the order of the discretisation schemes, exact sampling of the square-root process, the quadratic-exponential Heston scheme against the properties it
must keep (a martingale price, the exact mean of the variance, the Black-Scholes limit), and the maximum-likelihood fits against the parameters they were generated from."""

from __future__ import annotations

import numpy as np
import pytest

from src.probability import sde


# ---------------------------------------------------------------------------------------------------------------------------- discretisation
def test_the_strong_order_of_euler_is_one_half_and_milstein_is_close_to_one():
    euler = sde.strong_error("euler", 0.1, 0.5, steps_list=(4, 8, 16, 32, 64, 128), n_paths=20_000)
    mil = sde.strong_error("milstein", 0.1, 0.5, steps_list=(4, 8, 16, 32, 64, 128), n_paths=20_000)
    assert 0.40 < euler["order"] < 0.60 and 0.80 < mil["order"] < 1.15
    assert all(a > b for a, b in zip(euler["errors"], euler["errors"][1:])) and mil["errors"][-1] < euler["errors"][-1] / 10


def test_the_weak_order_of_both_schemes_is_about_one_when_the_noise_is_shared_with_the_exact_solution():
    for scheme in ("euler", "milstein"):
        w = sde.weak_error(scheme, 0.1, 0.5, steps_list=(2, 4, 8, 16, 32), n_paths=200_000)
        assert 0.7 < w["order"] < 1.4, (scheme, w)


def test_the_schemes_share_noise_when_it_is_handed_over_and_check_their_arguments():
    a, b = (lambda x: 0.1 * x), (lambda x: 0.4 * x)
    p1, dw = sde.euler_maruyama(a, b, 1.0, 1.0, 10, 5, seed=3, return_noise=True)
    p2 = sde.euler_maruyama(a, b, 1.0, 1.0, 10, 5, increments=dw)
    assert np.array_equal(p1, p2) and p1.shape == (5, 11)
    with pytest.raises(ValueError):
        sde.strong_error("rk4", 0.1, 0.5)
    with pytest.raises(ValueError):
        sde.weak_error("rk4", 0.1, 0.5)


# ----------------------------------------------------------------------------------------------------------------------------- square root
def test_the_exact_cir_sampler_has_the_exact_conditional_mean_and_variance_and_never_goes_negative():
    kappa, theta, xi, v0, T = 1.5, 0.04, 0.4, 0.09, 0.5            # the Feller condition fails: 2 kappa theta = 0.12 < xi^2 = 0.16
    v = sde.simulate_cir(v0, kappa, theta, xi, T, 10, 400_000, exact=True, seed=1)
    mean, var = sde.cir_moments(v0, kappa, theta, xi, T)
    assert v.min() >= 0.0
    assert v[:, -1].mean() == pytest.approx(float(mean), rel=0.005) and v[:, -1].var() == pytest.approx(float(var), rel=0.02)
    one = sde.simulate_cir(v0, kappa, theta, xi, T / 10, 1, 400_000, exact=True, seed=2)
    m1, v1 = sde.cir_moments(v0, kappa, theta, xi, T / 10)
    assert one[:, -1].mean() == pytest.approx(float(m1), rel=0.003) and one[:, -1].var() == pytest.approx(float(v1), rel=0.02)


def test_full_truncation_euler_approaches_the_exact_distribution_as_the_step_shrinks():
    args = (0.05, 2.0, 0.04, 0.3)
    exact_mean = float(sde.cir_moments(*args, 1.0)[0])
    errs = [abs(sde.simulate_cir(*args, 1.0, steps, 400_000, exact=False, seed=6)[:, -1].mean() - exact_mean) for steps in (2, 10, 50)]
    assert errs[0] > 3e-3 and errs[0] > 5 * errs[1] and errs[1] > 3 * errs[2] and errs[2] < 1e-4, errs
    with pytest.raises(ValueError):
        sde.simulate_cir(0.04, -1.0, 0.04, 0.3, 1.0, 5, 10)


# ----------------------------------------------------------------------------------------------------------------------------------- Heston
HESTON = dict(S0=100.0, v0=0.04, r=0.03, q=0.01, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7)


@pytest.mark.parametrize("steps", [1, 4, 50])
def test_the_qe_scheme_keeps_the_discounted_price_a_martingale_and_the_variance_mean_for_any_step(steps):
    S, V = sde.simulate_heston(**HESTON, horizon=1.0, steps=steps, n_paths=400_000, scheme="qe", seed=steps, antithetic=True)
    fwd = HESTON["S0"] * np.exp((HESTON["r"] - HESTON["q"]) * 1.0)
    se = S[:, -1].std() / np.sqrt(len(S))
    assert abs(S[:, -1].mean() - fwd) < 4 * se + 1e-9
    exact_v = HESTON["theta"] + (HESTON["v0"] - HESTON["theta"]) * np.exp(-HESTON["kappa"])
    assert V[:, -1].mean() == pytest.approx(exact_v, rel=0.01) and V.min() >= 0.0 and S.shape == (400_000, steps + 1)


def test_the_qe_scheme_agrees_with_a_fine_full_truncation_euler_on_an_option_price():
    K, T = 105.0, 0.5
    qe = sde.simulate_heston(**HESTON, horizon=T, steps=10, n_paths=300_000, scheme="qe", seed=1)[0][:, -1]
    eu = sde.simulate_heston(**HESTON, horizon=T, steps=400, n_paths=60_000, scheme="euler", seed=2)[0][:, -1]
    df = np.exp(-HESTON["r"] * T)
    c_qe, c_eu = df * np.maximum(qe - K, 0).mean(), df * np.maximum(eu - K, 0).mean()
    se = df * np.hypot(np.maximum(qe - K, 0).std() / np.sqrt(len(qe)), np.maximum(eu - K, 0).std() / np.sqrt(len(eu)))
    assert abs(c_qe - c_eu) < 4 * se


def test_with_a_vanishing_volatility_of_variance_the_log_return_variance_is_the_integrated_variance():
    p = dict(HESTON, xi=0.02, rho=0.0)
    S, _ = sde.simulate_heston(**p, horizon=1.0, steps=20, n_paths=200_000, scheme="qe", seed=4)
    integrated = p["theta"] + (p["v0"] - p["theta"]) * (1.0 - np.exp(-p["kappa"])) / p["kappa"]          # the integral of the deterministic variance over a year
    assert np.log(S[:, -1]).var() == pytest.approx(integrated, rel=0.02)


def test_the_heston_simulator_checks_its_arguments_and_refuses_a_step_that_breaks_the_correction():
    with pytest.raises(ValueError):
        sde.simulate_heston(**HESTON, horizon=1.0, steps=10, n_paths=10, scheme="rk4")
    with pytest.raises(ValueError):
        sde.simulate_heston(**dict(HESTON, rho=1.0), horizon=1.0, steps=10, n_paths=10)
    with pytest.raises(ValueError):
        sde.simulate_heston(**dict(HESTON, v0=1.0, theta=1.0, kappa=5.0, xi=1.0, rho=0.9), horizon=10.0, steps=1, n_paths=100)       # one ten-year step: the exponential moment does not exist


# ----------------------------------------------------------------------------------------------------------------------------- estimation
def test_the_gbm_fit_recovers_the_parameters_and_its_standard_errors_describe_the_spread_of_the_estimates():
    mu, sigma, dt, n = 0.08, 0.25, 1 / 252, 252 * 4
    estimates, ses = [], []
    for seed in range(300):
        r = np.random.default_rng(seed).normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), n)
        f = sde.fit_gbm(log_returns=r, dt=dt)
        estimates.append((f.estimates["sigma"], f.estimates["mu"]))
        ses.append((f.std_errors["sigma"], f.std_errors["mu"]))
    est, se = np.array(estimates), np.array(ses)
    assert est[:, 0].mean() == pytest.approx(sigma, rel=0.01)
    assert est[:, 0].std() == pytest.approx(se[:, 0].mean(), rel=0.15) and est[:, 1].std() == pytest.approx(se[:, 1].mean(), rel=0.15)
    cover = np.mean((est[:, 1] - 1.96 * se[:, 1] <= mu) & (mu <= est[:, 1] + 1.96 * se[:, 1]))
    assert 0.91 < cover < 0.99, "the confidence interval of the drift holds about 95% of the time"
    assert se[:, 1].mean() > 0.3 * mu, "four years of daily data cannot pin the drift down"


def test_the_gbm_fit_takes_prices_and_refuses_ambiguous_or_short_input():
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 500)))
    assert sde.fit_gbm(prices=prices).n == 499
    with pytest.raises(ValueError):
        sde.fit_gbm(prices=prices, log_returns=np.zeros(5))
    with pytest.raises(ValueError):
        sde.fit_gbm(log_returns=np.zeros(5))
    with pytest.raises(ValueError):
        sde.fit_gbm()


def test_the_cir_fit_recovers_the_parameters_of_a_long_exact_sample_and_its_errors_are_honest():
    kappa, theta, xi, dt = 3.0, 0.04, 0.35, 1 / 52
    x = sde.simulate_cir(0.04, kappa, theta, xi, 2000 * dt, 2000, 1, exact=True, seed=11)[0, 1:]
    fit = sde.fit_cir(x, dt)
    est, se = fit.estimates, fit.std_errors
    assert fit.converged and est["theta"] == pytest.approx(theta, rel=0.25) and est["xi"] == pytest.approx(xi, rel=0.1) and est["kappa"] == pytest.approx(kappa, rel=0.4)
    assert abs(est["kappa"] - kappa) < 3.5 * se["kappa"] and abs(est["theta"] - theta) < 3.5 * se["theta"] and abs(est["xi"] - xi) < 3.5 * se["xi"]
    assert fit.ci["xi"][0] < xi < fit.ci["xi"][1] and fit.n == 1999
    assert fit.loglik >= sde.cir_loglik((kappa, theta, xi), x, dt) - 1e-6, "the fit is at least as likely as the truth"
    with pytest.raises(ValueError):
        sde.fit_cir(x[:10], dt)
    with pytest.raises(ValueError):
        sde.fit_cir(np.where(np.arange(100) == 5, -1.0, 1.0), dt)
    assert sde.cir_loglik((-1.0, 0.04, 0.3), x, dt) == -np.inf
