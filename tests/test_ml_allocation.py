"""Release 6, P2-04, P2-14 and P2-16: the cost-aware allocation against independent solutions.

The references are closed forms and brute force, not the optimiser's own output: a single asset with a proportional cost has the soft-threshold solution; two assets with a budget constraint have the Lagrange
solution; three assets are searched on a grid; the no-trade optimum is the KKT condition ``|mu_i - lambda (Sigma w0)_i| <= c_i``. Status semantics are the contract: an infeasible problem says so before any objective is
optimised, and a fallback is labelled."""

import itertools

import numpy as np
import pandas as pd
import pytest

from src.ml import allocation as al, realism as rz

A = ["x"]


def one(mu, var, w0, spec, **kw):
    return al.solve(pd.Series({"x": mu}), pd.DataFrame([[var]], index=A, columns=A), pd.Series({"x": w0}), spec, **kw)


def soft_threshold_reference(mu, var, w0, lam, c, cap, gross):
    """argmax mu w - lam var w^2 / 2 - c |w - w0| on [-cap, cap] and |w| <= gross: the unconstrained optimum moves towards w0 by c / (lam var), and stops at w0 if that crosses it."""
    wu = mu / (lam * var)
    if abs(mu - lam * var * w0) <= c:
        w = w0
    elif mu - lam * var * w0 > c:
        w = wu - c / (lam * var)
    else:
        w = wu + c / (lam * var)
    return float(np.clip(w, -min(cap, gross), min(cap, gross)))


@pytest.mark.parametrize("mu,var,w0,c", [(0.02, 0.04, 0.0, 0.001), (0.02, 0.04, 0.0, 0.05), (0.02, 0.04, 0.15, 0.001), (-0.03, 0.04, 0.1, 0.002), (0.5, 0.04, 0.0, 0.001), (0.0, 0.04, 0.2, 0.002),
                                      (0.0, 0.04, -0.2, 0.002), (0.01, 0.09, 0.05, 0.0), (0.001, 0.01, -0.1, 0.003)])
def test_a_single_asset_with_a_proportional_cost_has_the_soft_threshold_solution(mu, var, w0, c):
    spec = al.AllocationSpec(risk_aversion=5.0, cost=c, position_max=0.35, gross_max=1.0, ridge=0.0)
    a = one(mu, var, w0, spec)
    assert a.status == "optimal" and a.weights is not None
    assert a.weights["x"] == pytest.approx(soft_threshold_reference(mu, var, w0, 5.0, c, 0.35, 1.0), abs=2e-6), (mu, var, w0, c)


def test_a_zero_forecast_does_nothing_from_flat_and_only_cuts_risk_from_a_book_when_the_cost_allows():
    spec = al.AllocationSpec(risk_aversion=5.0, cost=0.002, ridge=0.0)
    flat = one(0.0, 0.04, 0.0, spec)
    assert flat.no_trade and flat.weights["x"] == 0.0
    held = one(0.0, 0.04, 0.2, spec)
    assert held.weights["x"] == pytest.approx(0.002 / (5.0 * 0.04), abs=2e-6) and not held.no_trade, "risk reduction stops where the marginal risk (lambda var w) has fallen to the cost: 0.01, not 0.19"


def test_the_no_trade_optimum_is_the_kkt_condition_and_a_larger_edge_breaks_it():
    cov = pd.DataFrame([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.05]], index=list("abc"), columns=list("abc"))
    w0 = pd.Series([0.2, -0.1, 0.1], index=list("abc"))
    lam, c = 4.0, 0.004
    spec = al.AllocationSpec(risk_aversion=lam, cost=c, ridge=0.0, position_max=0.5, gross_max=2.0)
    marginal = lam * (cov.to_numpy() @ w0.to_numpy())
    mu_inside = pd.Series(marginal + np.array([0.9, -0.5, 0.2]) * c, index=list("abc"))
    a = al.solve(mu_inside, cov, w0, spec)
    assert a.status == "optimal" and a.no_trade and np.allclose(a.weights, w0, atol=1e-7), "every edge is inside the cost band: hold"
    mu_out = mu_inside.copy()
    mu_out["b"] += 3 * c
    b = al.solve(mu_out, cov, w0, spec)
    assert not b.no_trade and b.trades["b"] > 1e-4 and abs(b.trades["a"]) < 1e-6 and abs(b.trades["c"]) < 1e-3 * 5, "only the name whose edge left the band is traded (the covariance spreads a little)"


def test_two_assets_under_a_budget_constraint_have_the_lagrange_solution():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=list("ab"), columns=list("ab"))
    mu = pd.Series([0.03, 0.02], index=list("ab"))
    lam = 6.0
    spec = al.AllocationSpec(risk_aversion=lam, cost=0.0, ridge=0.0, gross_max=5.0, position_max=5.0, net_min=1.0, net_max=1.0)
    a = al.solve(mu, cov, pd.Series(0.0, index=list("ab")), spec)
    inv = np.linalg.inv(cov.to_numpy())
    nu = (np.ones(2) @ inv @ mu.to_numpy() - lam) / (np.ones(2) @ inv @ np.ones(2))
    ref = inv @ (mu.to_numpy() - nu) / lam
    assert a.status == "optimal" and np.allclose(a.weights.to_numpy(), ref, atol=1e-6) and a.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_three_assets_with_costs_and_caps_agree_with_a_search_of_the_feasible_grid():
    rng = np.random.default_rng(4)
    for trial in range(4):
        a = rng.normal(0, 1, (3, 3))
        cov = pd.DataFrame(a @ a.T * 0.01 + np.eye(3) * 0.02, index=list("abc"), columns=list("abc"))
        mu = pd.Series(rng.normal(0.01, 0.02, 3), index=list("abc"))
        w0 = pd.Series(rng.uniform(-0.2, 0.2, 3), index=list("abc"))
        spec = al.AllocationSpec(risk_aversion=3.0, cost=0.002, turnover_penalty=0.01, gross_max=0.9, position_max=0.45, ridge=0.0)
        sol = al.solve(mu, cov, w0, spec)
        assert sol.status == "optimal"
        axis = np.arange(-0.45, 0.4501, 0.0225)
        grid = np.array(list(itertools.product(axis, axis, axis)))
        grid = grid[np.abs(grid).sum(axis=1) <= 0.9 + 1e-12]
        d = grid - w0.to_numpy()
        util = grid @ mu.to_numpy() - 0.5 * 3.0 * np.einsum("ij,jk,ik->i", grid, cov.to_numpy(), grid) - 0.002 * np.abs(d).sum(axis=1) - 0.5 * 0.01 * (d ** 2).sum(axis=1)
        assert sol.components["utility"] >= util.max() - 1e-9, "no feasible grid point does better"
        assert sol.components["utility"] <= util.max() + 5e-4, "and the grid is dense enough to show the optimiser did not stop far from the best"
        assert sol.residuals["max"] < 1e-8 and sol.solver["method"] == "SLSQP"


def test_caps_bind_and_are_reported_and_long_only_never_goes_short():
    cov = pd.DataFrame(np.diag([0.04, 0.04, 0.04]), index=list("abc"), columns=list("abc"))
    mu = pd.Series([0.2, 0.15, -0.3], index=list("abc"))
    spec = al.AllocationSpec(risk_aversion=2.0, cost=0.0, position_max=0.3, gross_max=0.5, ridge=0.0)
    a = al.solve(mu, cov, None, spec)
    # KKT with a common multiplier theta on the gross budget: |w_i| = (|mu_i| - theta) / (lambda var). c (|mu| .3) is capped at .3, a gets the remaining .2 (theta = .184 > .15), b gets nothing
    assert a.utilization["gross_utilisation"] == pytest.approx(1.0, abs=1e-6) and a.weights["c"] == pytest.approx(-0.3, abs=1e-6) and a.weights["a"] == pytest.approx(0.2, abs=1e-6) and abs(a.weights["b"]) < 1e-6
    assert a.utilization["positions_at_cap"] == ["c"], "a negative forecast is sold short when shorting is allowed, up to the cap"
    lo = al.solve(mu, cov, None, al.AllocationSpec(risk_aversion=2.0, cost=0.0, position_max=0.3, gross_max=1.0, long_only=True, ridge=0.0))
    assert (lo.weights >= -1e-9).all() and abs(lo.weights["c"]) < 1e-8


def test_a_singular_covariance_still_has_a_solution_and_the_ridge_is_reported():
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=list("ab"), columns=list("ab"))
    a = al.solve(pd.Series([0.02, 0.01], index=list("ab")), cov, None, al.AllocationSpec(cost=0.0001))
    assert a.status == "optimal" and a.ridge_added == pytest.approx(1e-8 * 0.04) and np.isfinite(a.weights).all() and a.weights.abs().sum() <= 1.0 + 1e-8


def test_inconsistent_constraints_are_infeasible_before_any_optimisation_and_a_fallback_is_labelled():
    cov = pd.DataFrame(np.eye(3) * 0.04, index=list("abc"), columns=list("abc"))
    mu = pd.Series([0.01, 0.02, 0.03], index=list("abc"))
    w0 = pd.Series([0.1, 0.1, 0.1], index=list("abc"))
    bad = al.AllocationSpec(gross_max=1.0, net_min=2.0)
    none = al.solve(mu, cov, w0, bad)
    assert none.status == "infeasible" and none.weights is None and none.fallback_used is None and "no book satisfies" in none.reason and none.trades is None
    hold = al.solve(mu, cov, w0, al.AllocationSpec(gross_max=1.0, net_min=2.0, fallback="hold"))
    assert hold.status == "infeasible" and hold.fallback_used == "hold" and hold.weights.equals(w0) and hold.no_trade
    ew = al.solve(mu, cov, w0, al.AllocationSpec(gross_max=0.6, net_min=2.0, position_max=0.35, fallback="equal_weight"))
    assert ew.fallback_used == "equal_weight" and ew.status == "infeasible", "an equal-weight book is only ever returned with its label and the reason"
    nan = al.solve(mu.where(mu > 0.015), cov, w0, al.AllocationSpec())
    assert nan.status == "failed" and "missing values" in nan.reason and nan.weights is None
    with pytest.raises(ValueError):
        al.AllocationSpec(fallback="magic")
    with pytest.raises(ValueError):
        al.AllocationSpec(gross_max=-1.0)


def test_the_impact_term_is_the_realistic_modes_square_root_law_and_capital_changes_what_is_worth_trading():
    imp = rz.ImpactSpec(coefficient=0.5)
    sigma, adv, capital, trade = 0.02, 5e6, 2e6, 0.1
    bps = rz.impact_bps(imp, trade * capital, adv, sigma)["bps"]
    spec = al.AllocationSpec(impact=0.5, cost=0.0)
    k = al.impact_coefficients(spec, capital, pd.Series({"x": adv}), pd.Series({"x": sigma}))["x"]
    assert k * trade ** 1.5 == pytest.approx(bps / 1e4 * trade, rel=1e-12), "cost of capital = (impact in price) x (weight traded), with the price move of the engine's law"
    cov = pd.DataFrame([[0.04]], index=A, columns=A)
    results = []
    for cap in (1e5, 1e7, 1e9):
        a = al.solve(pd.Series({"x": 0.06}), cov, pd.Series({"x": 0.0}), al.AllocationSpec(risk_aversion=5.0, cost=0.0005, impact=0.5, ridge=0.0), capital=cap, adv=pd.Series({"x": 5e6}), sigma=pd.Series({"x": 0.02}))
        results.append((a.weights["x"], a.components["impact_cost"], a.components["utility"]))
    assert results[0][0] > results[1][0] > results[2][0] > 0, "more capital in the same name: a smaller book is worth building"
    assert results[0][2] > results[1][2] > results[2][2], "and the utility per unit of capital falls: capacity is limited"


def test_a_participation_limit_caps_every_trade_and_a_name_without_volume_cannot_be_traded():
    cov = pd.DataFrame(np.eye(3) * 0.04, index=list("abc"), columns=list("abc"))
    mu = pd.Series([0.2, 0.2, 0.2], index=list("abc"))
    adv = pd.Series({"a": 1e6, "b": 4e6, "c": 0.0})
    spec = al.AllocationSpec(risk_aversion=2.0, cost=0.0, position_max=0.4, gross_max=1.0, max_participation=0.1, ridge=0.0)
    a = al.solve(mu, cov, pd.Series(0.0, index=list("abc")), spec, capital=2e6, adv=adv)
    limit = 0.1 * adv / 2e6
    assert a.status == "optimal" and (a.trades.abs() <= limit + 1e-8).all() and a.weights["c"] == pytest.approx(0.0, abs=1e-9)
    assert set(a.utilization["trades_at_participation_limit"]) == {"a", "b"} and a.weights["a"] == pytest.approx(0.05, abs=1e-6) and a.weights["b"] == pytest.approx(0.2, abs=1e-6)
    with pytest.raises(ValueError, match="needs the capital"):
        al.solve(mu, cov, None, spec)
    stuck = al.solve(mu, cov, pd.Series([0.9, 0.0, 0.0], index=list("abc")), spec, capital=2e6, adv=adv)
    assert stuck.status == "infeasible", "a book already outside the caps that the participation limit stops from being repaired is infeasible, not silently clipped"


def test_partial_fills_are_carried_into_the_next_decision_and_the_book_drifts_with_returns():
    cov = pd.DataFrame([[0.04]], index=A, columns=A)
    dates = pd.bdate_range("2024-01-02", periods=4)
    mu = pd.DataFrame(0.1, index=dates, columns=A)
    spec = al.AllocationSpec(risk_aversion=5.0, cost=0.0005, position_max=0.35, gross_max=1.0, max_participation=0.1, ridge=0.0)
    flat = pd.DataFrame(0.0, index=dates, columns=A)
    out = al.simulate(mu, cov, flat, spec, capital=1e6, adv=pd.Series({"x": 1e6}))
    expected, held, turnover = [], 0.0, []
    for _ in range(4):                                                                         # a step of the limit (0.1 of capital) each time, until the cap; the cost shrinks equity, so the same shares are a slightly larger share
        ex = min(held + 0.1, 0.35)
        cost = 0.0005 * abs(ex - held)
        expected.append(ex)
        turnover.append(abs(ex - held))
        held = ex / (1.0 - cost)
    assert np.allclose(out["books"]["x"].to_numpy(), expected, atol=1e-8) and expected[3] == 0.35, "a trade the participation limit stopped is completed on the following decisions"
    assert out["table"]["limited_by_participation"].apply(len).tolist() == [1, 1, 1, 0] and np.allclose(out["table"]["turnover"].to_numpy(), turnover, atol=1e-8)
    r = pd.DataFrame([[0.1], [0.0], [0.0], [0.0]], index=dates, columns=A)
    drift = al.simulate(mu, cov, r, spec, capital=1e6, adv=pd.Series({"x": 1e6}))
    cost1 = 0.0005 * 0.1
    growth = 1.0 + 0.1 * 0.1 - cost1
    held = 0.1 * 1.1 / growth
    assert drift["table"]["gross_return"].iloc[0] == pytest.approx(0.01) and drift["books"]["x"].iloc[1] == pytest.approx(min(held + 0.1, 0.35), abs=1e-6), "the second decision starts from the drifted partial fill"
    assert drift["net_return"] == pytest.approx(drift["table"]["gross_return"].sum() - drift["table"]["cost"].sum())


def test_the_perturbation_grid_is_reported_whole_failures_are_counted_and_the_grid_is_hashed():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=list("ab"), columns=list("ab"))
    mu = pd.Series([0.03, 0.02], index=list("ab"))
    spec = al.AllocationSpec(risk_aversion=5.0, cost=0.001, gross_max=1.0, position_max=0.6, ridge=0.0)
    grid = {"mu_scale": [0.0, 0.5, 1.0, 1.5], "cost_scale": [1.0, 5.0], "gross_max": [1.0, 0.0001]}
    out = al.sensitivity(mu, cov, pd.Series([0.1, 0.1], index=list("ab")), spec, grid)
    assert len(out["points"]) == 16 and out["grid_hash"] == al.grid_hash({"gross_max": [1.0, 0.0001], "mu_scale": [0.0, 0.5, 1.0, 1.5], "cost_scale": [1.0, 5.0]}), "the hash does not depend on the order the grid was written in"
    assert out["failed"] == 0 and out["base_status"] == "optimal" and out["summary"]["max_weight_change"] > 0
    zero = [p for p in out["points"] if p["point"]["mu_scale"] == 0.0 and p["point"]["gross_max"] == 1.0 and p["point"]["cost_scale"] == 5.0][0]
    assert zero["status"] == "optimal" and zero["risk"] < 0.2
    base_w = out["base"]["weights"]
    assert base_w is not None
    tight = al.sensitivity(mu, cov, pd.Series([0.1, 0.1], index=list("ab")), al.AllocationSpec(gross_max=1.0, net_min=0.9, position_max=0.6, ridge=0.0), {"gross_max": [1.0, 0.5, 0.2]})
    assert tight["failed"] == 2 and [p["status"] for p in tight["points"]].count("infeasible") == 2, "an infeasible perturbation is a failure that is counted and listed"
    with pytest.raises(ValueError, match="maps"):
        al.sensitivity(mu, cov, None, spec, {"mystery": [1]})
    with pytest.raises(ValueError, match="more than"):
        al.sensitivity(mu, cov, None, spec, {"mu_scale": list(range(1, 9)), "cost_scale": list(range(1, 10))})


def test_window_inputs_scale_the_covariance_by_the_number_of_days():
    mu, cov = al.window_inputs(pd.Series({"x": 0.01}), pd.DataFrame([[0.0004]], index=A, columns=A), 5)
    assert cov.iloc[0, 0] == pytest.approx(0.002) and mu["x"] == 0.01
    with pytest.raises(ValueError):
        al.window_inputs(mu, cov, 0)
