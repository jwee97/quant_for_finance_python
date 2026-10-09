"""Mean-variance with a CVaR constraint: the CVaR formula, the Rockafellar-Uryasev programme against a general solver, and the effect of tightening the limit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog, minimize

from src.equity.cvar_portfolio import mean_variance_cvar, scenario_cvar, scenario_var


def scenarios(S=600, n=6, seed=0, fat=True):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (S, n)) * np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.05])[:n]
    if fat:
        crash = rng.random((S, n)) < 0.03
        base = base - crash * np.array([0.0, 0.0, 0.02, 0.05, 0.10, 0.15])[:n]                    # the last assets have fat left tails
    mu = np.array([0.004, 0.005, 0.006, 0.0075, 0.0095, 0.012])[:n]
    return base + mu, mu


def test_scenario_cvar_is_the_average_of_the_worst_losses_and_matches_the_ru_minimisation():
    rng = np.random.default_rng(1)
    R = rng.normal(0.0, 0.02, (1000, 3))
    w = np.array([0.5, 0.3, 0.2])
    loss = np.sort(-R @ w)
    assert scenario_cvar(w, R, 0.95) == pytest.approx(loss[-50:].mean(), rel=1e-12)               # 5% of 1000 scenarios is exactly the worst fifty
    assert scenario_var(w, R, 0.95) == pytest.approx(np.quantile(loss, 0.95))
    assert scenario_cvar(w, R, 0.95) >= scenario_var(w, R, 0.95)
    # a share that does not divide the scenarios evenly: the formula still equals the minimum over thresholds
    s = -R[:997] @ w
    grid = np.concatenate([s, np.linspace(s.min(), s.max(), 2000)])
    ru = min(z + np.maximum(s - z, 0).mean() / 0.05 for z in grid)
    assert scenario_cvar(w, R[:997], 0.95) == pytest.approx(ru, rel=1e-9)
    assert scenario_cvar(np.array([1.0, 0, 0]), np.array([[0.01, 0, 0], [-0.02, 0, 0]]) , 0.5) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        scenario_cvar(w, R, 1.0)


def test_with_no_limit_the_programme_is_plain_mean_variance():
    R, mu = scenarios(fat=False)
    cov = np.cov(R, rowvar=False)
    r = mean_variance_cvar(mu, R, None, risk_aversion=20.0)
    ref = minimize(lambda w: -(w @ mu) + 10.0 * w @ cov @ w, np.full(6, 1 / 6), jac=lambda w: -mu + 20.0 * cov @ w, bounds=[(0, 1)] * 6,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP", options={"ftol": 1e-15})
    assert r.ok and np.abs(r.weights.to_numpy() - ref.x).max() < 1e-5 and r.weights.sum() == pytest.approx(1.0)


def test_the_cvar_programme_is_the_linear_programme_when_risk_aversion_is_zero():
    R, mu = scenarios(400, 5, 2)
    limit = 0.03
    r = mean_variance_cvar(mu, R, limit, alpha=0.95, risk_aversion=0.0, fully_invested=False)
    # the same problem as an LP over (w, zeta, u)
    S, n = R.shape
    c = np.concatenate([-mu, [0.0], np.zeros(S)])
    A_ub = np.zeros((S + 2, n + 1 + S)); b_ub = np.zeros(S + 2)
    A_ub[:S, :n], A_ub[:S, n], A_ub[np.arange(S), n + 1 + np.arange(S)] = -R, -1.0, -1.0
    A_ub[S, n], A_ub[S, n + 1:] = 1.0, 1.0 / (0.05 * S); b_ub[S] = limit
    A_ub[S + 1, :n] = 1.0; b_ub[S + 1] = 1.0
    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * n + [(None, None)] + [(0, None)] * S, method="highs")
    assert r.ok and lp.success and r.expected_return == pytest.approx(-lp.fun, abs=1e-6) and r.cvar <= limit + 1e-6


def test_tightening_the_limit_lowers_the_tail_loss_and_the_return_and_the_limit_binds():
    R, mu = scenarios(800, 6, 3)
    free = mean_variance_cvar(mu, R, None, risk_aversion=2.0, fully_invested=False)
    results = {}
    for limit in (0.12, 0.08, 0.05, 0.03):
        r = mean_variance_cvar(mu, R, limit, risk_aversion=2.0, fully_invested=False)
        assert r.ok and r.cvar <= limit + 1e-6
        results[limit] = r
    assert results[0.12].cvar <= free.cvar + 1e-9 or results[0.12].binding is False
    tight = results[0.03]
    assert tight.binding and tight.cvar == pytest.approx(0.03, abs=1e-5) and tight.cash >= -1e-9                  # the limit binds; this portfolio gets there by diversifying away from the tails, not by holding cash
    assert results[0.03].expected_return < results[0.05].expected_return < results[0.08].expected_return + 1e-9
    assert results[0.03].cvar < results[0.05].cvar < results[0.08].cvar + 1e-9
    assert tight.weights.iloc[-1] < free.weights.iloc[-1] + 1e-9                                                  # the fat-tailed asset is the first to go
    assert (tight.weights >= -1e-9).all() and tight.weights.sum() <= 1 + 1e-9


def test_a_limit_that_cannot_be_met_fully_invested_is_reported_and_met_with_cash():
    R, mu = scenarios(500, 6, 4)
    S, n = R.shape
    # the smallest CVaR any fully invested long-only portfolio can have: a linear programme over (w, zeta, u)
    c = np.concatenate([np.zeros(n), [1.0], np.full(S, 1.0 / (0.05 * S))])
    A_ub = np.hstack([-R, -np.ones((S, 1)), -np.eye(S)])
    lp = linprog(c, A_ub=A_ub, b_ub=np.zeros(S), A_eq=np.concatenate([np.ones(n), np.zeros(S + 1)])[None, :], b_eq=[1.0], bounds=[(0, 1)] * n + [(None, None)] + [(0, None)] * S, method="highs")
    floor = lp.fun
    r = mean_variance_cvar(mu, R, floor * 0.5, fully_invested=True)
    assert not r.ok                                                                                              # no fully invested long-only portfolio is this safe
    r2 = mean_variance_cvar(mu, R, floor * 0.5, fully_invested=False)
    assert r2.ok and r2.cash >= 0.5 - 1e-6 and r2.cvar <= floor * 0.5 + 1e-6 and r2.binding                          # half the safest portfolio is half the risk, so at least half is cash (more if variance prefers a different mix)
    exact = mean_variance_cvar(mu, R, floor * 1.0000001, fully_invested=True)
    assert exact.ok and exact.cvar == pytest.approx(floor, abs=1e-6)


def test_the_cutting_plane_answer_equals_the_scenario_programme_and_the_fallback_finds_the_same_thing():
    from src.equity.cvar_portfolio import _solve_ru

    R, mu = scenarios(300, 6, 6)
    cov = np.cov(R, rowvar=False)
    free = mean_variance_cvar(mu, R, None, risk_aversion=3.0, fully_invested=False)
    limit = 0.5 * free.cvar
    cuts = mean_variance_cvar(mu, R, limit, risk_aversion=3.0, fully_invested=False)
    full = _solve_ru(mu, R, cov, limit, 0.95, 3.0, 0.0, 1.0, False)
    assert cuts.ok and full.ok and np.abs(cuts.weights.to_numpy() - full.x[:6]).max() < 1e-4
    forced = mean_variance_cvar(mu, R, limit, risk_aversion=3.0, fully_invested=False, max_cuts=2)                 # too few cuts to finish: the scenario programme takes over
    assert forced.ok and forced.status == "solved (scenario programme)" and np.abs(forced.weights.to_numpy() - cuts.weights.to_numpy()).max() < 1e-4


def test_inputs_are_validated_and_names_are_kept():
    R, mu = scenarios(100, 3, 5)
    df = pd.DataFrame(R, columns=list("XYZ"))
    r = mean_variance_cvar(mu, df, 0.1, fully_invested=False)
    assert list(r.weights.index) == list("XYZ")
    for bad in (dict(alpha=1.0), dict(alpha=0.2), dict(risk_aversion=-1.0)):
        with pytest.raises(ValueError):
            mean_variance_cvar(mu, df, 0.1, **bad)
    with pytest.raises(ValueError):
        mean_variance_cvar(mu[:2], df, 0.1)
    with pytest.raises(ValueError):
        mean_variance_cvar(mu, df.iloc[:10], 0.1)
