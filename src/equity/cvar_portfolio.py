"""Mean-variance optimisation with a CVaR constraint (Rockafellar and Uryasev).

Variance treats a gain and a loss of the same size alike and says nothing about the tail. Conditional value-at-risk at level ``a`` is the average loss on the worst ``1 - a`` of the scenarios, and
Rockafellar and Uryasev (2000) showed it is the minimum over a threshold ``zeta`` of ``zeta + E[(loss - zeta)+] / (1 - a)``, which makes a CVaR limit a set of *linear* constraints once each scenario's
shortfall ``u_s >= loss_s - zeta`` is a variable::

    maximise    mu'w - (risk_aversion / 2) w'Sigma w
    subject to  zeta + sum_s u_s / ((1 - a) S) <= cvar_limit,    u_s >= -R_s w - zeta,   u_s >= 0
                sum(w) = 1 (or <= 1, the rest in cash), lb <= w <= ub

With a loose limit it is plain mean-variance; as the limit tightens the portfolio moves away from the assets with fat left tails and toward cash. The scenarios are historical (or simulated) returns of
the holding period; the limit is in the same units.

**How it is solved.** The Rockafellar-Uryasev programme above has one variable per scenario and a dense, degenerate structure that a first-order method takes thousands of iterations on. The dual form of
CVaR is cheaper: ``CVaR(w) = max over q of -q'Rw``, where ``q`` puts weight ``1 / ((1 - a) S)`` on each of the worst ``(1 - a) S`` scenarios (and the remainder on the next). Every ``q`` therefore gives
a *valid* linear inequality ``-(R'q)'w <= cvar_limit``, and Kelley's cutting-plane method adds the cut that is most violated by the current answer until none is (CVaR is piecewise linear, so this
finishes in a finite, usually small, number of rounds). Each round is a quadratic programme in the ``n`` weights alone, solved by the interior-point method of :mod:`src.equity.qp`, which does not mind how nearly parallel the cuts get. Cutting planes can creep when there are many assets and a nearly linear objective; after ``max_cuts`` rounds the programme of the paper itself (one variable per scenario) is solved with the same interior-point method, which is slower (it grows with the scenarios) and does not creep. It is also the reference in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .qp import solve_qp_ipm


def scenario_cvar(weights, scenarios, alpha: float = 0.95) -> float:
    """The CVaR of a portfolio on ``scenarios`` (rows are scenarios of asset returns): the exact minimum over the threshold of ``zeta + mean((loss - zeta)+) / (1 - alpha)``, positive for a loss."""
    loss = -np.asarray(scenarios, dtype=float) @ np.asarray(weights, dtype=float)
    S = len(loss)
    if S < 2 or not 0.0 < alpha < 1.0:
        raise ValueError("at least two scenarios and 0 < alpha < 1")
    order = np.sort(loss)
    tail = np.concatenate([np.cumsum(order[::-1])[::-1][1:], [0.0]])                      # sum of the losses above each candidate threshold
    count_above = np.arange(S - 1, -1, -1)
    value = order + (tail - count_above * order) / ((1.0 - alpha) * S)
    return float(value.min())


def scenario_var(weights, scenarios, alpha: float = 0.95) -> float:
    """The value-at-risk: the loss exceeded in a ``1 - alpha`` share of the scenarios."""
    return float(np.quantile(-np.asarray(scenarios, dtype=float) @ np.asarray(weights, dtype=float), alpha))


@dataclass
class CVaRResult:
    weights: pd.Series
    cvar: float
    var: float
    expected_return: float
    volatility: float
    status: str
    ok: bool
    binding: bool = False
    cash: float = 0.0
    cuts: int = 0


def worst_tail_weights(weights, scenarios, alpha: float = 0.95) -> np.ndarray:
    """The distribution ``q`` over scenarios that attains the CVaR: ``1 / ((1 - alpha) S)`` on each of the worst ``(1 - alpha) S`` scenarios and the remainder on the next one."""
    loss = -np.asarray(scenarios, dtype=float) @ np.asarray(weights, dtype=float)
    S = len(loss)
    k = (1.0 - alpha) * S
    full = int(np.floor(k + 1e-12))
    order = np.argsort(-loss)
    q = np.zeros(S)
    q[order[:full]] = 1.0 / k
    if full < S and k - full > 1e-12:
        q[order[full]] = 1.0 - full / k
    return q


def _solve_ru(m, R, Sigma, cvar_limit, alpha, risk_aversion, lb, ub, fully_invested):
    """The Rockafellar-Uryasev programme itself: variables ``(w, zeta, u_1 .. u_S)``."""
    S, n = R.shape
    N = n + 1 + S
    P = np.zeros((N, N))
    P[:n, :n] = risk_aversion * Sigma
    q = np.concatenate([-m, [0.0], np.zeros(S)])
    shortfall = np.zeros((S, N))
    shortfall[:, :n], shortfall[:, n], shortfall[np.arange(S), n + 1 + np.arange(S)] = R, 1.0, 1.0       # u_s + R_s w + zeta >= 0
    tail = np.zeros((1, N))
    tail[0, n], tail[0, n + 1:] = 1.0, 1.0 / ((1.0 - alpha) * S)
    budget = np.zeros((1, N))
    budget[0, :n] = 1.0
    A = np.vstack([shortfall, tail, budget])
    lo = np.concatenate([np.zeros(S), [-np.inf], [1.0 if fully_invested else 0.0]])
    hi = np.concatenate([np.full(S, np.inf), [float(cvar_limit)], [1.0]])
    lbv = np.concatenate([np.broadcast_to(np.asarray(lb, dtype=float), (n,)), [-np.inf], np.zeros(S)])
    ubv = np.concatenate([np.broadcast_to(np.asarray(ub, dtype=float), (n,)), [np.inf], np.full(S, np.inf)])
    return solve_qp_ipm(P, q, A=A, l=lo, u=hi, lb=lbv, ub=ubv)


def mean_variance_cvar(mu, scenarios, cvar_limit: float | None, alpha: float = 0.95, risk_aversion: float = 5.0, cov=None, lb: float | np.ndarray = 0.0, ub: float | np.ndarray = 1.0,
                       fully_invested: bool = True, names=None, max_cuts: int = 60, tolerance: float = 1e-9, gap_tolerance: float = 1e-7) -> CVaRResult:
    """Maximise mean-variance utility subject to ``CVaR_alpha(w) <= cvar_limit`` on ``scenarios`` (a scenarios-by-assets table of returns). ``cov`` defaults to the covariance of the scenarios.
    ``fully_invested=False`` lets the weights sum to less than one (the rest is cash that earns and loses nothing), which is what a tight limit needs. ``cvar_limit=None`` drops the constraint."""
    R = np.asarray(scenarios, dtype=float)
    S, n = R.shape
    names = list(names) if names is not None else (list(scenarios.columns) if isinstance(scenarios, pd.DataFrame) else list(range(n)))
    m = np.asarray(mu, dtype=float)
    if m.shape != (n,) or not (np.isfinite(R).all() and np.isfinite(m).all()):
        raise ValueError("mu must match the assets and everything must be finite")
    if S < 20 or not 0.5 <= alpha < 1.0 or risk_aversion < 0:
        raise ValueError("at least 20 scenarios, 0.5 <= alpha < 1, risk_aversion >= 0")
    Sigma = np.cov(R, rowvar=False) if cov is None else np.asarray(cov, dtype=float)
    Sigma = 0.5 * (Sigma + Sigma.T).reshape(n, n)
    budget = np.ones((1, n))
    lo_b, hi_b = [1.0 if fully_invested else 0.0], [1.0]
    cuts: list[np.ndarray] = []
    w, status = np.full(n, np.nan), "no solution"
    for _ in range(max_cuts + 1):
        A = np.vstack([budget] + ([np.array(cuts)] if cuts else []))
        lo = np.concatenate([lo_b, np.full(len(cuts), -np.inf)])
        hi = np.concatenate([hi_b, np.full(len(cuts), float(cvar_limit) if cvar_limit is not None else np.inf)])
        r = solve_qp_ipm(risk_aversion * Sigma, -m, A=A, l=lo, u=hi, lb=lb, ub=ub)
        if not r.ok:
            return CVaRResult(pd.Series(np.nan, index=names), np.nan, np.nan, np.nan, np.nan, r.status, False, cuts=len(cuts))
        w, status = np.where(np.abs(r.x) < 1e-9, 0.0, r.x), r.status
        cvar_now = scenario_cvar(w, R, alpha) if cvar_limit is not None else 0.0
        if cvar_limit is None or cvar_now <= cvar_limit + tolerance:
            break
        if not fully_invested and cvar_now > 0:                                           # CVaR scales with the position: shrinking toward cash gives a feasible point and a bound on how far from the optimum it is
            feasible = w * (float(cvar_limit) / cvar_now)
            utility = lambda x: float(0.5 * risk_aversion * x @ Sigma @ x - m @ x)
            if utility(feasible) - utility(w) <= gap_tolerance * (1.0 + abs(utility(w))):
                w, status = feasible, "solved (within the duality gap)"
                break
        cuts.append(-(R.T @ worst_tail_weights(w, R, alpha)))                              # CVaR(w') >= q'(-R w') for every w', so this cut is valid and the current w violates it
    else:                                                                                    # the cuts are creeping: solve the programme with one variable per scenario instead
        full = _solve_ru(m, R, Sigma, cvar_limit, alpha, risk_aversion, lb, ub, fully_invested)
        if not full.ok:
            return CVaRResult(pd.Series(np.nan, index=names), np.nan, np.nan, np.nan, np.nan, full.status, False, cuts=len(cuts))
        w, status = np.where(np.abs(full.x[:n]) < 1e-8, 0.0, full.x[:n]), "solved (scenario programme)"
    cvar = scenario_cvar(w, R, alpha)
    return CVaRResult(pd.Series(w, index=names), cvar, scenario_var(w, R, alpha), float(w @ m), float(np.sqrt(max(w @ Sigma @ w, 0.0))), status, True,
                      bool(cvar_limit is not None and cvar >= cvar_limit - 1e-6), float(1.0 - w.sum()), len(cuts))
