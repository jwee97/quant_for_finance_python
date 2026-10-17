"""Short-rate models: the closed-form bond prices satisfy their own partial differential equation, agree with simulation, have the right limits, and the fits recover the parameters that
generated a curve and a history."""

from __future__ import annotations

import numpy as np
import pytest

from src.assets import short_rate as sr
from src.probability import ou, sde

VAS = dict(kappa=0.8, theta=0.04, sigma=0.012)
CIR = dict(kappa=0.8, theta=0.04, sigma=0.09)                  # 2 kappa theta = 0.064 > sigma^2 = 0.0081: the rate stays positive


# ----------------------------------------------------------------------------------------------------------------------- the formulas
@pytest.mark.parametrize("model,params", [("vasicek", VAS), ("cir", CIR), ("cir", dict(kappa=0.3, theta=0.03, sigma=0.25))])
def test_the_bond_price_solves_the_bond_equation(model, params):
    """``P_tau = kappa (theta - r) P_r + (sigma^2 / 2) r^{2 gamma} P_rr - r P`` with ``gamma = 0`` (Vasicek) or ``1/2`` (CIR), checked by differencing the formula in maturity."""
    k, th, sg = params["kappa"], params["theta"], params["sigma"]
    for r in (0.01, 0.04, 0.09):
        for tau in (0.5, 2.0, 10.0):
            h = 1e-5
            p = float(sr.bond_price(model, r, tau, **params))
            p_tau = (float(sr.bond_price(model, r, tau + h, **params)) - float(sr.bond_price(model, r, tau - h, **params))) / (2 * h)
            b = float(sr.duration_to_short_rate(model, tau, **params))
            rhs = k * (th - r) * (-b * p) + 0.5 * sg ** 2 * (r if model == "cir" else 1.0) * b ** 2 * p - r * p
            assert p_tau == pytest.approx(rhs, rel=1e-6, abs=1e-9), (model, r, tau)


@pytest.mark.parametrize("model,params", [("vasicek", VAS), ("cir", CIR)])
def test_the_curve_starts_at_the_short_rate_and_ends_at_the_long_yield_and_the_duration_is_the_sensitivity(model, params):
    r = 0.02
    assert float(sr.bond_price(model, r, 0.0, **params)) == pytest.approx(1.0) and float(sr.duration_to_short_rate(model, 0.0, **params)) == 0.0
    y = sr.yield_curve(model, r, [1e-6, 1.0, 5.0, 30.0, 500.0], **params)
    assert y[0] == pytest.approx(r, abs=1e-6) and y[-1] == pytest.approx(sr.long_yield(model, **params), abs=2e-4)
    assert y[-1] < params["theta"], "the convexity of the bond price pulls the long yield below the long-run mean of the rate"
    assert y[0] < y[1] < y[2] < y[3], "the rate starts below its mean: the curve slopes up"
    inverted = sr.yield_curve(model, 0.09, [0.25, 1.0, 5.0, 30.0], **params)
    assert inverted[0] > inverted[-1], "and from above it slopes down"
    tau, h = 3.0, 1e-6
    d = (np.log(float(sr.bond_price(model, r - h, tau, **params))) - np.log(float(sr.bond_price(model, r + h, tau, **params)))) / (2 * h)
    assert d == pytest.approx(float(sr.duration_to_short_rate(model, tau, **params)), rel=1e-6)
    p = sr.bond_price(model, np.array([0.01, 0.03]), np.array([[1.0], [7.0]]), **params)
    assert p.shape == (2, 2) and np.all(p[1] < p[0])


def test_the_gaussian_model_prices_a_bond_above_par_at_a_negative_rate_and_the_square_root_model_never_does():
    assert float(sr.bond_price("vasicek", -0.02, 0.5, **VAS)) > 1.0 and float(sr.bond_price("cir", 1e-9, 0.5, **CIR)) < 1.0
    with pytest.raises(ValueError):
        sr.bond_price("hull-white", 0.02, 1.0, **VAS)
    with pytest.raises(ValueError):
        sr.bond_price("vasicek", 0.02, 1.0, kappa=-1.0, theta=0.04, sigma=0.01)
    with pytest.raises(ValueError):
        sr.bond_price("cir", 0.02, 1.0, kappa=1.0, theta=-0.04, sigma=0.01)


@pytest.mark.parametrize("model,params", [("vasicek", VAS), ("cir", CIR), ("cir", dict(kappa=0.5, theta=0.03, sigma=0.3))])         # the last violates the Feller condition
def test_the_closed_form_agrees_with_the_expected_discount_factor_over_simulated_paths(model, params):
    for tau in (1.0, 5.0):
        mc = sr.bond_price_mc(model, 0.025, tau, steps=40 if tau == 1.0 else 100, n_paths=40_000, seed=2, **params)
        assert abs(mc["price"] - float(sr.bond_price(model, 0.025, tau, **params))) < 4.5 * mc["se"] + 2e-5, (model, params, tau)


# ------------------------------------------------------------------------------------------------------------------------------- fits
@pytest.mark.parametrize("model,params", [("vasicek", VAS), ("cir", CIR)])
def test_a_curve_fit_recovers_the_level_and_the_starting_rate_when_the_speed_and_volatility_are_known(model, params):
    tau = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    truth = sr.yield_curve(model, 0.015, tau, **params)
    fit = sr.fit_curve(model, tau, truth, kappa=params["kappa"], sigma=params["sigma"])
    assert fit["rmse"] < 1e-8 and fit["params"]["theta"] == pytest.approx(params["theta"], abs=1e-5) and fit["params"]["r0"] == pytest.approx(0.015, abs=1e-6) and fit["success"]
    noisy = truth + np.random.default_rng(3).normal(0, 1e-4, len(tau))                       # a basis point of noise
    n_fit = sr.fit_curve(model, tau, noisy, kappa=params["kappa"], sigma=params["sigma"])
    assert 3e-5 < n_fit["rmse"] < 2e-4 and abs(n_fit["params"]["r0"] - 0.015) < 4 * n_fit["std_errors"]["r0"] + 1e-4 and n_fit["std_errors"]["theta"] > 0


def test_a_cross_section_alone_cannot_pin_down_all_four_parameters_but_still_reproduces_the_curve():
    tau = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    truth = sr.yield_curve("vasicek", 0.015, tau, **VAS)
    fit = sr.fit_curve("vasicek", tau, truth)
    assert fit["rmse"] < 1e-5, "the curve is matched"
    assert fit["std_errors"]["kappa"] > 5 * fit["std_errors"]["r0"] or np.isnan(fit["std_errors"]["kappa"]), "but the speed is the poorly determined one"
    assert fit["feller"] is True
    with pytest.raises(ValueError):
        sr.fit_curve("vasicek", tau[:3], truth[:3])


def test_a_history_fit_recovers_the_physical_parameters_of_each_model():
    dt, n = 1 / 252, 252 * 25
    v = ou.simulate_ou(0.04, 0.8, 0.04, 0.012, n * dt, n, 1, seed=5)[0]
    fit = sr.fit_history("vasicek", v, dt)
    for k, truth in (("kappa", 0.8), ("theta", 0.04), ("sigma", 0.012)):
        assert abs(fit["params"][k] - truth) < 4 * fit["std_errors"][k], k
    c = sde.simulate_cir(0.04, 0.8, 0.04, 0.09, n * dt, n, 1, exact=True, seed=6)[0]
    cfit = sr.fit_history("cir", c, dt)
    for k, truth in (("kappa", 0.8), ("theta", 0.04), ("xi", 0.09)):
        key = "sigma" if k == "xi" else k
        est = cfit["params"][k] if k in cfit["params"] else cfit["params"][key]
        assert abs(est - truth) < 4 * cfit["std_errors"][k if k in cfit["std_errors"] else key] + 1e-9, k
    with pytest.raises(ValueError):
        sr.fit_history("hull-white", v, dt)
