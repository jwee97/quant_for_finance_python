"""Pricing from a characteristic function: the Lewis integral and the Carr-Madan FFT against the closed forms that exist (Black-Scholes, the Merton series, the Heston quadrature), against
the density of the model (for the models whose density is known), against Monte Carlo with exact samplers, and against the properties a price must have (bounds, convexity in the strike,
parity, a smile of the right shape, a risk-neutral density that integrates to one and has the forward for mean)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from src.derivatives import fourier, levy
from src.derivatives import pricing as pr

S, R, Q = 100.0, 0.03, 0.01
STRIKES = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0])

MERTON_SETS = [(0.15, 2.0, -0.08, 0.12), (0.2, 0.5, -0.05, 0.1), (0.12, 12.0, -0.01, 0.03)]
HESTON_SETS = [dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7), dict(v0=0.09, kappa=0.8, theta=0.04, xi=0.9, rho=-0.5), dict(v0=0.02, kappa=4.0, theta=0.03, xi=0.2, rho=0.3)]
MODELS = {
    "bs": levy.BlackScholes(sigma=0.22),
    "merton": levy.Merton(sigma=0.15, lam=1.5, mu_j=-0.07, sigma_j=0.1),
    "kou": levy.Kou(sigma=0.15, lam=2.0, p=0.35, eta1=12.0, eta2=9.0),
    "vg": levy.VarianceGamma(sigma=0.18, theta=-0.12, nu=0.25),
    "nig": levy.NIG(sigma=0.18, theta=-0.12, kappa=0.25),
    "heston": levy.Heston(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7),
}


# ------------------------------------------------------------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0, 8.0])
def test_lewis_quadrature_and_fft_reproduce_the_black_scholes_price(T):
    m = levy.BlackScholes(sigma=0.25)
    ref = pr.bsm_price(S, STRIKES, T, R, Q, 0.25, True)
    assert np.abs(fourier.lewis_call(m, S, STRIKES, T, R, Q) - ref).max() < 1e-10
    assert np.abs(fourier.lewis_call(m, S, STRIKES, T, R, Q, method="quad") - ref).max() < 1e-10
    assert np.abs(fourier.price(m, S, STRIKES, T, R, Q, True, "fft") - ref).max() < 2e-7
    put = pr.bsm_price(S, STRIKES, T, R, Q, 0.25, False)
    assert np.abs(fourier.price(m, S, STRIKES, T, R, Q, False) - put).max() < 1e-10


@pytest.mark.parametrize("params", MERTON_SETS)
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
def test_the_fourier_price_of_a_jump_diffusion_is_the_corrected_merton_series(params, T):
    sigma, lam, mu_j, sigma_j = params
    m = levy.Merton(sigma=sigma, lam=lam, mu_j=mu_j, sigma_j=sigma_j)
    series_call = pr.merton_price(S, STRIKES, T, R, Q, sigma, lam, mu_j, sigma_j, True)
    series_put = pr.merton_price(S, STRIKES, T, R, Q, sigma, lam, mu_j, sigma_j, False)
    assert np.abs(fourier.price(m, S, STRIKES, T, R, Q, True) - series_call).max() < 1e-9
    assert np.abs(fourier.price(m, S, STRIKES, T, R, Q, False) - series_put).max() < 1e-9
    assert np.abs(fourier.price(m, S, STRIKES, T, R, Q, True, "fft") - series_call).max() < 5e-6


@pytest.mark.parametrize("params", HESTON_SETS)
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
def test_the_heston_characteristic_function_prices_like_the_gil_pelaez_quadrature(params, T):
    m = levy.Heston(**params)
    ref = pr.heston_price(S, STRIKES, T, R, Q, **params, upper=2000.0)          # the older, slower route: two integrals per strike
    assert np.abs(fourier.lewis_call(m, S, STRIKES, T, R, Q) - ref).max() < 2e-8
    assert np.abs(fourier.price(m, S, STRIKES, T, R, Q, True, "fft") - ref).max() < 2e-6


def test_the_heston_price_without_volatility_of_variance_is_black_scholes_at_the_integrated_variance():
    v0, kappa, theta, T = 0.09, 2.0, 0.04, 1.5
    integrated = theta + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T)               # the variance is then deterministic: its average over the life
    ref = pr.bsm_price(S, STRIKES, T, R, Q, np.sqrt(integrated), True)

    def gap(xi, rho):
        m = levy.Heston(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        return np.abs(fourier.price(m, S, STRIKES, T, R, Q, True) - ref).max()

    assert gap(1e-3, 0.0) < 1e-5, "uncorrelated: the volatility of variance enters at the second order"
    assert gap(1e-2, 0.0) == pytest.approx(100 * gap(1e-3, 0.0), rel=0.1)
    assert gap(1e-3, -0.5) == pytest.approx(10 * gap(1e-4, -0.5), rel=0.1), "correlated: it enters at the first order, as the skew rho * xi"


# --------------------------------------------------------------------------------------------------------------------------- the convergence
def test_the_trapezoid_rule_converges_geometrically_with_the_step_and_is_exact_at_the_default():
    m = levy.Heston(**HESTON_SETS[0])
    ref = fourier.lewis_call(m, S, STRIKES, 0.1, R, Q, method="quad")
    errs = [np.abs(fourier.lewis_call(m, S, STRIKES, 0.1, R, Q, step=h) - ref).max() for h in (0.4, 0.2, 0.1)]
    assert errs[0] > 1e-2 and 1e-7 < errs[1] < 1e-3 and errs[2] < 1e-8, errs
    assert errs[0] > 100 * errs[1] > 1e4 * errs[2], "the error falls faster than any power of the step (about exp(-pi / step))"


def test_the_integral_is_cut_where_the_integrand_is_negligible_and_longer_for_slowly_decaying_models():
    nig = levy.NIG(sigma=0.2, theta=-0.1, kappa=0.2)
    short, long_ = (fourier.default_upper(nig, T) for T in (0.02, 2.0))
    assert short > 10 * long_ > 100
    for T, U in ((0.02, short), (2.0, long_)):
        assert abs(nig.cf_centered(np.array([U - 0.5j]), T)[0]) / (U ** 2 + 0.25) < 1e-13
    vg = levy.VarianceGamma(sigma=0.2, theta=-0.1, nu=0.5)
    assert fourier.default_upper(vg, 0.02, limit=500.0) == 500.0, "the cap applies when the characteristic function decays like a power of u"


# ------------------------------------------------------------------------------------------------------- against the density of the model
def _price_from_the_density(model, K, T):
    """The call price as an integral of the payoff against the closed-form density of the increment (adaptive quadrature, the cusp at zero marked)."""
    drift = (R - Q - model.omega()) * T
    k1, k2, _, _ = model.cumulants()
    mean, sd = k1 * T, np.sqrt(k2 * T)
    lo, hi = max(np.log(K / S) - drift, mean - 40 * sd), mean + 40 * sd
    f = lambda x: (S * np.exp(x + drift) - K) * float(model.pdf(np.array([x]), T)[0])           # noqa: E731
    pts = [0.0] if lo < 0.0 < hi else None
    return np.exp(-R * T) * integrate.quad(f, lo, hi, points=pts, limit=400, epsabs=1e-12, epsrel=1e-11)[0]


@pytest.mark.parametrize("name", ["merton", "vg", "nig"])
@pytest.mark.parametrize("T", [0.25, 1.5])
def test_the_fourier_price_equals_the_payoff_integrated_against_the_closed_form_density(name, T):
    m = MODELS[name]
    K = np.array([85.0, 100.0, 115.0])
    ref = np.array([_price_from_the_density(m, k, T) for k in K])
    assert np.abs(fourier.lewis_call(m, S, K, T, R, Q) - ref).max() < 1e-8


# ------------------------------------------------------------------------------------------------------------------------------ Monte Carlo
@pytest.mark.parametrize("name", ["kou", "vg", "nig", "merton"])
def test_the_prices_of_the_jump_models_agree_with_exact_monte_carlo_and_the_discounted_price_is_a_martingale(name):
    m, T, n = MODELS[name], 0.75, 600_000
    ST = levy.simulate_paths(m, S, R, Q, T, 1, n, seed=11)[:, -1]
    fwd = S * np.exp((R - Q) * T)
    assert abs(ST.mean() - fwd) < 4.5 * ST.std() / np.sqrt(n)
    for K in (85.0, 100.0, 120.0):
        pay = np.exp(-R * T) * np.maximum(ST - K, 0.0)
        assert abs(fourier.price(m, S, K, T, R, Q) - pay.mean()) < 4.5 * pay.std() / np.sqrt(n), (name, K)


def test_the_heston_fourier_price_agrees_with_the_quadratic_exponential_simulation():
    p, T = HESTON_SETS[0], 1.0
    m = levy.Heston(**p)
    ST = levy.simulate_paths(m, S, R, Q, T, 12, 250_000, seed=5)[:, -1]
    for K in (90.0, 100.0, 115.0):
        pay = np.exp(-R * T) * np.maximum(ST - K, 0.0)
        assert abs(fourier.price(m, S, K, T, R, Q) - pay.mean()) < 4.5 * pay.std() / np.sqrt(len(pay)) + 0.02, K


# --------------------------------------------------------------------------------------------------------------------------------- properties
@pytest.mark.parametrize("name", list(MODELS))
def test_prices_respect_the_arbitrage_bounds_and_are_convex_and_decreasing_in_the_strike(name):
    m, T = MODELS[name], 0.5
    K = np.linspace(40.0, 200.0, 161)
    c = fourier.price(m, S, K, T, R, Q, True)
    p = fourier.price(m, S, K, T, R, Q, False)
    low = np.maximum(S * np.exp(-Q * T) - K * np.exp(-R * T), 0.0)
    assert np.all(c >= low - 1e-9) and np.all(c <= S * np.exp(-Q * T) + 1e-9)
    assert np.all(p >= np.maximum(K * np.exp(-R * T) - S * np.exp(-Q * T), 0.0) - 1e-9) and np.all(p <= K * np.exp(-R * T) + 1e-9)
    assert np.all(np.diff(c) < 1e-9) and np.all(np.diff(c, 2) > -1e-8), "a butterfly with a negative price would be an arbitrage"
    assert np.allclose(c - p, S * np.exp(-Q * T) - K * np.exp(-R * T), atol=1e-12)


@pytest.mark.parametrize("name", ["merton", "kou", "vg", "nig", "heston"])
def test_a_model_with_negative_jumps_or_correlation_produces_the_equity_skew_and_it_flattens_with_the_maturity(name):
    m = MODELS[name]
    F = lambda T: S * np.exp((R - Q) * T)                                            # noqa: E731
    skews = []
    for T in (0.25, 4.0):
        lo, hi = (float(fourier.implied_vols(m, S, np.array([k * F(T)]), T, R, Q)[0]) for k in (0.85, 1.15))
        assert lo > hi, (name, T, "puts are dearer than calls")
        skews.append(lo - hi)
    assert skews[0] > skews[1] > 0, (name, skews)


def test_symmetric_jumps_give_a_smile_with_both_wings_above_the_money_and_black_scholes_gives_a_flat_line():
    flat = fourier.implied_vols(levy.BlackScholes(sigma=0.3), S, STRIKES, 0.5, R, Q)
    assert np.abs(flat - 0.3).max() < 1e-7
    vg = levy.VarianceGamma(sigma=0.2, theta=0.0, nu=0.3)
    F = S * np.exp((R - Q) * 0.5)
    iv = fourier.implied_vols(vg, S, np.array([0.8, 1.0, 1.2]) * F, 0.5, R, Q)
    assert iv[0] > iv[1] and iv[2] > iv[1]
    assert abs(iv[0] - iv[2]) < 0.15 * (iv[0] - iv[1]), "symmetric: the two wings are about level"


def test_implied_vols_accept_a_scalar_strike_and_return_a_scalar():
    assert isinstance(fourier.implied_vols(MODELS["vg"], S, 100.0, 0.5, R, Q), float)
    assert isinstance(fourier.price(MODELS["vg"], S, 100.0, 0.5, R, Q), float)


# ---------------------------------------------------------------------------------------------------------------------------------- density
@pytest.mark.parametrize("name", ["bs", "merton", "kou", "vg", "nig", "heston"])
def test_the_risk_neutral_density_integrates_to_one_has_the_forward_for_mean_and_reprices_the_call(name):
    m, T = MODELS[name], 1.0
    d = fourier.risk_neutral_density(m, S, T, R, Q, points=401)
    s, dens = d["s"], d["density"]
    fwd = S * np.exp((R - Q) * T)
    assert d["mass"] == pytest.approx(1.0, abs=2e-3)
    assert integrate.trapezoid(s * dens, s) == pytest.approx(fwd, rel=4e-3)
    assert float(np.exp(-R * T) * integrate.trapezoid(np.maximum(s - 105.0, 0.0) * dens, s)) == pytest.approx(float(fourier.price(m, S, 105.0, T, R, Q)), abs=0.03)


def test_the_density_inversion_reproduces_the_lognormal_and_the_closed_form_densities_of_the_levy_models():
    T = 1.0
    d = fourier.risk_neutral_density(MODELS["bs"], S, T, R, Q, points=201)
    sigma = 0.22
    mu = np.log(S) + (R - Q - 0.5 * sigma ** 2) * T
    lognormal = np.exp(-0.5 * ((np.log(d["s"]) - mu) / (sigma * np.sqrt(T))) ** 2) / (d["s"] * sigma * np.sqrt(2 * np.pi * T))
    assert np.abs(d["density"] - lognormal).max() < 1e-7
    for name in ("merton", "vg", "nig"):
        m = MODELS[name]
        d = fourier.risk_neutral_density(m, S, T, R, Q, points=201)
        exact = m.pdf(d["x"] - (R - Q - m.omega()) * T, T) / d["s"]
        assert np.abs(d["density"] - exact).max() < 2e-4 * exact.max(), name


# ------------------------------------------------------------------------------------------------------------------------- arguments, edges
def test_the_fft_damping_must_sit_inside_the_strip_where_the_model_has_exponential_moments():
    kou = levy.Kou(sigma=0.15, lam=2.0, p=0.4, eta1=1.6, eta2=9.0)           # E[S^s] is finite only for s < 1.6: the damping alpha + 1 must stay below it
    assert kou.exponent_limit() == 1.6
    with pytest.raises(ValueError, match="exponent limit"):
        fourier.carr_madan(kou, S, 0.5, R, Q, alpha=0.7)
    with pytest.raises(ValueError, match="positive"):
        fourier.carr_madan(kou, S, 0.5, R, Q, alpha=-0.5)
    strikes, calls = fourier.carr_madan(kou, S, 0.5, R, Q)
    assert len(strikes) == len(calls) == 2 ** 12
    near = (strikes > 70) & (strikes < 140)
    assert np.abs(calls[near] - fourier.lewis_call(kou, S, strikes[near], 0.5, R, Q)).max() < 1e-8, "a weakly damped model gets a finer spacing, which keeps the transform exact"


def test_expired_options_are_worth_their_intrinsic_value_and_bad_arguments_are_refused():
    m = MODELS["vg"]
    assert np.array_equal(fourier.lewis_call(m, S, STRIKES, 0.0, R, Q), np.maximum(S - STRIKES, 0.0))
    with pytest.raises(ValueError):
        fourier.lewis_call(m, S, STRIKES, 0.5, step=0.0)
    with pytest.raises(ValueError):
        fourier.lewis_call(m, S, STRIKES, 0.5, step=0.9)
    with pytest.raises(ValueError):
        fourier.lewis_call(m, S, STRIKES, 0.5, method="simpson")
    with pytest.raises(ValueError):
        fourier.price(m, S, STRIKES, 0.5, method="cos")
