"""Release 8, P3-04: distributionally robust allocation with a documented uncertainty set.

A single asset has a closed-form robust position, which the solver must reproduce; the limiting cases (no uncertainty, huge uncertainty) are checked; the robust book must be unbeatable on the worst-case
measure it optimises and must pay for that in nominal utility; and the radii are calibrated against an analytic reference and against the dependence in the data."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.ml import allocation as alloc, dro

NAMES = list("ABCDEF")


def problem(seed=0, n=6):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = pd.DataFrame(A @ A.T / n * 0.02 + np.eye(n) * 0.01, index=NAMES[:n], columns=NAMES[:n])
    mu = pd.Series(rng.normal(0.02, 0.02, n), index=NAMES[:n])
    se = pd.Series(np.sqrt(np.diag(cov)) / 6.0, index=NAMES[:n])
    return mu, cov, se


FREE = alloc.AllocationSpec(risk_aversion=5.0, cost=0.0, turnover_penalty=0.0, position_max=0.35, gross_max=1.0)


# ------------------------------------------------------------------------------------------------------------------ closed form and limiting cases
def test_one_asset_matches_the_closed_form_robust_position():
    mu, cov, se = pd.Series({"A": 0.02}), pd.DataFrame([[0.04]], index=["A"], columns=["A"]), pd.Series({"A": 0.004})
    u = dro.UncertaintySet(kappa=2.0, rho=0.25)
    r = dro.solve(mu, cov, se, u, FREE)
    assert r.allocation.status == "optimal"
    assert r.weights["A"] == pytest.approx((0.02 - 2.0 * 0.004) / (5.0 * 1.25 * 0.04), abs=2e-4), "(mu - kappa se) / (lambda (1 + rho) sigma^2)"
    inside = dro.solve(pd.Series({"A": 0.006}), cov, se, u, FREE)
    assert inside.weights["A"] == pytest.approx(0.0, abs=1e-6), "a forecast inside its own box carries no position"
    short = dro.solve(pd.Series({"A": -0.02}), cov, se, u, FREE)
    assert short.weights["A"] == pytest.approx(-0.048, abs=2e-4)
    big = dro.solve(pd.Series({"A": 0.5}), cov, se, u, FREE)
    assert big.weights["A"] == pytest.approx(0.35, abs=1e-6), "the position limit still binds"
    assert r.worst_case["worst_case_return"] == pytest.approx(0.048 * (0.02 - 2.0 * 0.004), abs=1e-5) and r.nominal["nominal_return"] == pytest.approx(0.048 * 0.02, abs=1e-5)


def test_no_uncertainty_is_the_nominal_problem_and_a_vanishing_set_is_close_to_it():
    mu, cov, se = problem()
    nominal = alloc.solve(mu, cov, None, FREE)
    zero = dro.solve(mu, cov, se, dro.UncertaintySet(), FREE)
    assert zero.weights.to_numpy() == pytest.approx(nominal.weights.to_numpy(), abs=1e-12), "kappa = 0 and rho = 0 is the nominal problem, by the same path"
    tiny = dro.solve(mu, cov, se, dro.UncertaintySet(kappa=1e-7, rho=1e-9), FREE)
    assert tiny.weights.to_numpy() == pytest.approx(nominal.weights.to_numpy(), abs=1e-4)


def test_more_uncertainty_means_a_smaller_book_and_a_flat_one_when_every_forecast_is_inside_its_box():
    mu, cov, se = problem(1)
    gross = [float(dro.solve(mu, cov, se, dro.UncertaintySet(kappa=k), FREE).weights.abs().sum()) for k in (0.0, 0.5, 1.0, 2.0, 5.0)]
    assert all(a >= b - 1e-8 for a, b in zip(gross, gross[1:])) and gross[0] > gross[-1], gross
    flat = dro.solve(mu, cov, se, dro.UncertaintySet(kappa=1e3), FREE)
    assert float(flat.weights.abs().sum()) < 1e-6 and flat.allocation.status == "optimal", "a box wider than every forecast leaves nothing to bet on"
    risk = [float(r.weights @ cov @ r.weights) for r in (dro.solve(mu, cov, se, dro.UncertaintySet(rho=x), FREE) for x in (0.0, 0.5, 2.0, 10.0))]
    assert all(a >= b - 1e-10 for a, b in zip(risk, risk[1:])), "inflating the covariance lowers the variance taken"


# ------------------------------------------------------------------------------------------------------------------ worst case against simpler baselines
def test_the_robust_book_cannot_be_beaten_on_its_worst_case_and_gives_up_nominal_utility_to_get_it():
    mu, cov, se = problem(2)
    u = dro.UncertaintySet(kappa=2.0, rho=0.5)
    robust = dro.solve(mu, cov, se, u, FREE)
    assert robust.allocation.status == "optimal" and robust.allocation.residuals["max"] < 1e-7
    wc = lambda w: dro.worst_case(w, mu, cov, se, u, FREE.risk_aversion)                                  # noqa: E731
    others = dro.baselines(mu, cov, FREE)
    assert {"nominal", "shrunk_forecast", "min_variance", "equal_weight"} <= set(others)
    for name, w in others.items():
        assert wc(robust.weights)["worst_case_utility"] >= wc(w)["worst_case_utility"] - 1e-7, f"the robust book is optimal for the worst case: {name} cannot beat it"
    assert wc(others["nominal"])["nominal_utility"] > wc(robust.weights)["nominal_utility"] + 1e-6, "robustness is paid for in nominal utility"
    assert wc(robust.weights)["worst_case_utility"] > wc(others["nominal"])["worst_case_utility"], "and buys a better worst case than the nominal book"
    assert robust.worst_case["worst_case_return"] <= robust.nominal["nominal_return"] and robust.worst_case["worst_case_variance"] == pytest.approx(1.5 * robust.nominal["nominal_variance"])


def test_the_limits_of_the_book_are_respected_and_an_impossible_set_of_limits_is_reported_not_forced():
    mu, cov, se = problem(3)
    spec = alloc.AllocationSpec(risk_aversion=3.0, cost=0.0, long_only=True, gross_max=1.0, net_min=1.0, net_max=1.0, position_max=0.3)
    r = dro.solve(mu, cov, se, dro.UncertaintySet(kappa=1.0, rho=0.2), spec)
    w = r.weights
    assert r.allocation.status == "optimal" and w.sum() == pytest.approx(1.0, abs=1e-7) and (w >= -1e-9).all() and (w <= 0.3 + 1e-7).all() and r.allocation.utilization["gross_utilisation"] == pytest.approx(1.0, abs=1e-6)
    assert dro.baselines(mu, cov, spec)["min_variance"].sum() == pytest.approx(1.0, abs=1e-6), "the minimum-variance baseline of a long-only book is fully invested"
    impossible = alloc.AllocationSpec(long_only=True, gross_max=1.0, net_min=1.0, net_max=1.0, position_max=0.1)          # six names cannot reach 100% with a 10% limit
    bad = dro.solve(mu, cov, se, dro.UncertaintySet(kappa=1.0), impossible)
    assert bad.allocation.status == "infeasible" and bad.weights is None and "no book satisfies the constraints" in bad.allocation.reason and bad.worst_case == {}
    holes = mu.copy()
    holes["B"] = np.nan
    assert dro.solve(holes, cov, se, dro.UncertaintySet(kappa=1.0), FREE).allocation.status == "failed"
    with pytest.raises(ValueError, match="not negative"):
        dro.UncertaintySet(kappa=-1.0)
    with pytest.raises(ValueError, match="holding penalty is not negative"):
        alloc.solve(mu, cov, None, FREE, holding_penalty=-se)


# ------------------------------------------------------------------------------------------------------------------ calibration of the set
def test_the_mean_radius_is_calibrated_to_a_simultaneous_box_and_follows_the_dependence_in_the_data():
    rng = np.random.default_rng(5)
    iid = pd.DataFrame(rng.normal(0.0005, 0.01, (500, 6)), columns=NAMES)
    cal = dro.calibrate_mean_radius(iid, level=0.9, n_boot=1200, block=1.0, seed=1)
    analytic = stats.norm.ppf((1 + 0.9 ** (1 / 6)) / 2)
    assert cal["kappa"] == pytest.approx(analytic, abs=0.3) and cal["level"] == 0.9 and "estimation error only" in cal["note"]
    assert cal["se"].to_numpy() == pytest.approx(iid.std(ddof=1).to_numpy() / np.sqrt(500), rel=0.15)
    wide = dro.calibrate_mean_radius(iid, level=0.99, n_boot=600, block=1.0, seed=1)
    assert wide["kappa"] > cal["kappa"]
    ar = np.zeros((500, 6))
    e = rng.normal(0, 0.01, (500, 6))
    for t in range(1, 500):
        ar[t] = 0.6 * ar[t - 1] + e[t]
    se_dep = dro.standard_errors(pd.DataFrame(ar, columns=NAMES), block=21.0, n_boot=300)
    assert (se_dep > 1.4 * pd.DataFrame(ar).std(ddof=1).to_numpy() / np.sqrt(500)).all(), "the block bootstrap sees the serial dependence the naive standard error ignores"
    with pytest.raises(ValueError, match="at least 60"):
        dro.calibrate_mean_radius(iid.iloc[:30])
    with pytest.raises(ValueError, match="level is in"):
        dro.calibrate_mean_radius(iid, level=1.5)


def test_the_covariance_inflation_is_zero_when_forecasts_were_pessimistic_and_positive_when_risk_rose():
    rng = np.random.default_rng(6)
    n = 900
    scale = lambda path: pd.DataFrame(rng.normal(0, 1, (len(path), 5)) * np.asarray(path)[:, None], columns=list("ABCDE"))        # noqa: E731
    flat = dro.calibrate_inflation(scale(np.full(n, 0.01)), level=0.9)
    rising = dro.calibrate_inflation(scale(np.linspace(0.005, 0.03, n)), level=0.9)
    falling = dro.calibrate_inflation(scale(np.linspace(0.03, 0.005, n)), level=0.9)
    assert rising["rho"] > flat["rho"] > 0.0 and falling["rho"] < flat["rho"] and falling["rho"] >= 0.0
    assert flat["quantile_ratio"] > 1.0, "even with constant risk the trailing forecast is noisy, so the quantile of the ratio is above one"
    with pytest.raises(ValueError, match="rows; at least"):
        dro.calibrate_inflation(scale(np.full(100, 0.01)))
