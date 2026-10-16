"""Exponential Lévy models and Heston: the martingale condition, the cumulants against the characteristic exponent, the samplers against the characteristic function, the closed-form
densities against Fourier inversion, and the maximum-likelihood fits against the laws that generated the data."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from src.derivatives import levy

MODELS = {
    "bs": levy.BlackScholes(sigma=0.2),
    "merton": levy.Merton(sigma=0.15, lam=2.0, mu_j=-0.08, sigma_j=0.12),
    "kou": levy.Kou(sigma=0.15, lam=3.0, p=0.35, eta1=14.0, eta2=9.0),
    "vg": levy.VarianceGamma(sigma=0.18, theta=-0.12, nu=0.25),
    "nig": levy.NIG(sigma=0.18, theta=-0.12, kappa=0.25),
}
HESTON = levy.Heston(v0=0.04, kappa=1.8, theta=0.05, xi=0.5, rho=-0.7)


# ------------------------------------------------------------------------------------------------------------------------- the model contract
@pytest.mark.parametrize("name", list(MODELS) + ["heston"])
def test_the_discounted_price_is_a_martingale_by_construction(name):
    m = MODELS.get(name, HESTON)
    for T, r, q in ((0.25, 0.03, 0.01), (2.0, -0.01, 0.0), (0.02, 0.05, 0.03)):
        assert abs(m.cf(-1j, T, r, q) - np.exp((r - q) * T)) < 1e-9
        assert abs(m.cf(0.0, T, r, q) - 1.0) < 1e-12


@pytest.mark.parametrize("name", list(MODELS))
def test_the_characteristic_function_is_a_valid_one_on_the_real_line(name):
    m = MODELS[name]
    u = np.linspace(-60, 60, 1201)
    phi = m.cf(u, 0.5, 0.02, 0.0)
    assert (np.abs(phi) <= 1.0 + 1e-12).all() and np.allclose(phi, np.conj(m.cf(-u, 0.5, 0.02, 0.0)))          # modulus at most 1 and Hermitian symmetry
    assert np.allclose(m.cf_centered(u, 0.5), phi * np.exp(-1j * u * 0.02 * 0.5))                              # the centred function does not depend on the rate


@pytest.mark.parametrize("name", list(MODELS))
def test_the_cumulants_are_the_taylor_coefficients_of_the_characteristic_exponent(name):
    m = MODELS[name]
    u = np.linspace(-0.35, 0.35, 61)
    coef = np.polynomial.polynomial.polyfit(u, m.psi(u), 8)                       # psi(u) = sum_n kappa_n (i u)^n / n!
    fact = [1, 1, 2, 6, 24]
    taylor = [coef[n] * fact[n] / (1j ** n) for n in range(1, 5)]
    assert np.allclose(np.real(taylor), m.cumulants(), rtol=2e-3, atol=2e-5), (np.real(taylor), m.cumulants())
    mom = m.moments()
    k1, k2, k3, k4 = m.cumulants()
    assert mom["variance"] == pytest.approx(k2) and mom["skewness"] == pytest.approx(k3 / k2 ** 1.5) and mom["excess_kurtosis"] == pytest.approx(k4 / k2 ** 2)


def test_the_jump_models_with_negative_mean_jumps_are_left_skewed_and_all_are_fat_tailed_except_black_scholes():
    assert MODELS["bs"].moments()["skewness"] == 0.0 and MODELS["bs"].moments()["excess_kurtosis"] == 0.0
    for name in ("merton", "kou", "vg", "nig"):
        assert MODELS[name].moments()["excess_kurtosis"] > 0.3, name
    for name in ("merton", "vg", "nig"):
        assert MODELS[name].moments()["skewness"] < 0, name


def test_the_nig_classical_parameters_reproduce_the_exponent():
    m = MODELS["nig"]
    c = m.classical_parameters()
    u = np.linspace(-20, 20, 81)
    classical = c["delta"] * (np.sqrt(c["alpha"] ** 2 - c["beta"] ** 2) - np.sqrt(c["alpha"] ** 2 - (c["beta"] + 1j * u) ** 2))
    assert np.allclose(classical, m.psi(u))


def test_parameters_are_validated_and_models_can_be_made_by_name():
    with pytest.raises(ValueError):
        levy.BlackScholes(sigma=-0.1)
    with pytest.raises(ValueError):
        levy.Kou(sigma=0.2, lam=1.0, p=0.5, eta1=0.9, eta2=5.0)
    with pytest.raises(ValueError):
        levy.Heston(v0=0.04, kappa=1.0, theta=0.04, xi=0.3, rho=1.0)
    with pytest.raises(TypeError):
        levy.Merton(sigma=0.2)
    with pytest.raises(TypeError):
        levy.VarianceGamma(sigma=0.2, theta=0.0, nu=0.2, extra=1.0)
    with pytest.raises(KeyError):
        levy.create("garch")
    vg = levy.create("vg", sigma=0.2, theta=-0.1, nu=0.3)
    assert vg == levy.VarianceGamma(sigma=0.2, theta=-0.1, nu=0.3) and vg.replace(nu=0.4).params["nu"] == 0.4 and "nu=0.3" in repr(vg)
    assert not levy.VarianceGamma(sigma=0.5, theta=1.0, nu=2.0).valid() and levy.VarianceGamma(sigma=0.2, theta=-0.1, nu=0.3).valid()
    assert not levy.NIG(sigma=0.5, theta=1.0, kappa=2.0).valid() and levy.NIG(sigma=0.2, theta=-0.1, kappa=0.3).valid()
    assert HESTON.feller is False and levy.Heston(v0=0.04, kappa=3.0, theta=0.05, xi=0.3, rho=-0.5).feller is True
    with pytest.raises(NotImplementedError):
        HESTON.psi(0.0)
    with pytest.raises(NotImplementedError):
        HESTON.sample(0.1, 5, np.random.default_rng(0))


# ------------------------------------------------------------------------------------------------------------------------------- the samplers
@pytest.mark.parametrize("name", list(MODELS))
def test_the_sampler_has_the_characteristic_function_and_the_moments_of_the_model(name):
    m = MODELS[name]
    dt = 0.08
    x = m.sample(dt, 600_000, np.random.default_rng(7))
    for u in (0.5, 2.0, 6.0, 15.0):
        emp = np.mean(np.exp(1j * u * x))
        assert abs(emp - np.exp(dt * m.psi(u))) < 3.5 / np.sqrt(len(x)), (name, u)
    k1, k2, _, _ = m.cumulants()
    se_mean = np.sqrt(k2 * dt / len(x))
    assert abs(x.mean() - k1 * dt) < 4 * se_mean and x.var() == pytest.approx(k2 * dt, rel=0.02)
    shape = m.sample(dt, (30, 4), np.random.default_rng(1))
    assert shape.shape == (30, 4)


# ------------------------------------------------------------------------------------------------------------------------------- the densities
@pytest.mark.parametrize("name", ["bs", "merton", "vg", "nig"])
def test_the_closed_form_density_integrates_to_one_and_agrees_with_fourier_inversion(name):
    m = MODELS[name]
    dt = 0.25                                                  # a quarter: the gamma and inverse-Gaussian clocks have shape dt / nu >= 1, so the density is finite at zero
    sd = np.sqrt(m.cumulants()[1] * dt)
    x = np.linspace(-9 * sd, 9 * sd, 4001)
    exact = m.pdf(x, dt)
    assert float(integrate.trapezoid(exact, x)) == pytest.approx(1.0, abs=2e-3)
    fft = levy.pdf_fft(m, dt, x)
    assert np.abs(exact - fft).max() < 0.01 * exact.max(), name


def test_the_density_of_the_kou_model_comes_from_inversion_and_matches_a_histogram_of_its_own_draws():
    m = MODELS["kou"]
    dt = 1.0 / 12.0
    x = m.sample(dt, 800_000, np.random.default_rng(3))
    sd = np.sqrt(m.cumulants()[1] * dt)
    grid = np.linspace(-8 * sd, 8 * sd, 4001)
    dens = m.pdf(grid, dt)
    assert float(integrate.trapezoid(dens, grid)) == pytest.approx(1.0, abs=2e-3)
    counts, edges = np.histogram(x, bins=60, range=(-5 * sd, 5 * sd))
    centres = 0.5 * (edges[1:] + edges[:-1])
    expected = np.interp(centres, grid, dens) * (edges[1] - edges[0]) * len(x)
    keep = expected > 50
    assert np.max(np.abs(counts[keep] - expected[keep]) / np.sqrt(expected[keep])) < 5.0


def test_the_variance_gamma_density_is_finite_at_zero_and_symmetric_without_skew():
    m = levy.VarianceGamma(sigma=0.2, theta=0.0, nu=0.3)
    d = m.pdf(np.array([-0.05, 0.0, 0.05]), 1.0 / 12.0)
    assert np.isfinite(d).all() and d[0] == pytest.approx(d[2]) and d[1] > d[0]


# ------------------------------------------------------------------------------------------------------------------------------------ fitting
def _daily(model, n, seed, drift=0.08):
    dt = 1.0 / 252.0
    return model.sample(dt, n, np.random.default_rng(seed)) + drift * dt


@pytest.mark.parametrize("name,truth", [
    ("vg", levy.VarianceGamma(sigma=0.16, theta=-0.1, nu=0.0045)),
    ("nig", levy.NIG(sigma=0.16, theta=-0.1, kappa=0.0045)),
    ("merton", levy.Merton(sigma=0.12, lam=15.0, mu_j=-0.01, sigma_j=0.025)),
])
def test_the_fit_recovers_the_variance_the_skew_and_the_tails_of_the_law_that_made_the_data(name, truth):
    r = _daily(truth, 30_000, 5)
    fit = levy.fit_returns(r, name)
    assert fit.converged and fit.k == len(truth.param_names) + 1 and fit.n == 30_000
    got, want = fit.model.moments(), truth.moments()
    assert got["variance"] == pytest.approx(want["variance"], rel=0.05)
    assert got["excess_kurtosis"] == pytest.approx(want["excess_kurtosis"], rel=0.30)
    assert got["skewness"] == pytest.approx(want["skewness"], abs=0.35)
    true_ll = levy._loglik(truth, r, 1 / 252, 0.08)
    assert fit.loglik >= true_ll - 0.5, "the fitted law is at least as likely as the truth"
    assert fit.summary()["model"] == name and fit.aic == pytest.approx(2 * fit.k - 2 * fit.loglik) and fit.bic > fit.aic


def test_normal_data_are_best_described_by_the_normal_and_fat_tailed_data_by_a_fat_tailed_law():
    normal = np.random.default_rng(2).normal(0.0003, 0.01, 6000)
    assert levy.compare_fits(normal, models=("bs", "vg", "nig"))[0].model.name == "bs"
    fat = _daily(levy.NIG(sigma=0.16, theta=-0.1, kappa=0.0045), 6000, 9)
    ranking = levy.compare_fits(fat, models=("bs", "vg", "nig"))
    bs = next(f for f in ranking if f.model.name == "bs")
    assert ranking[0].model.name in ("vg", "nig") and bs.aic - ranking[0].aic > 50


def test_the_fit_checks_its_input():
    with pytest.raises(ValueError):
        levy.fit_returns(np.zeros(50), "vg")
    with pytest.raises(ValueError):
        levy.fit_returns(np.random.default_rng(0).normal(size=500), "heston")
    assert [f.model.name for f in levy.compare_fits(np.random.default_rng(0).normal(0, 0.01, 400), models=("bs",))] == ["bs"]


# ------------------------------------------------------------------------------------------------------------------------------ risk-neutral paths
@pytest.mark.parametrize("name", list(MODELS) + ["heston"])
def test_simulated_risk_neutral_paths_have_the_forward_as_their_mean(name):
    m = MODELS.get(name, HESTON)
    S0, r, q, T = 100.0, 0.04, 0.01, 1.0
    paths = levy.simulate_paths(m, S0, r, q, T, 12, 300_000, seed=3, antithetic=True)
    assert paths.shape == (300_000, 13) and (paths[:, 0] == S0).all() and (paths > 0).all()
    se = paths[:, -1].std() / np.sqrt(len(paths))
    assert abs(paths[:, -1].mean() - S0 * np.exp((r - q) * T)) < 4.5 * se, name
    mid = paths[:, 6]
    assert abs(mid.mean() - S0 * np.exp((r - q) * T / 2)) < 4.5 * mid.std() / np.sqrt(len(paths))
