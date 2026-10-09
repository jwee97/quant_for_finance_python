"""Multi-period portfolio trading with transaction costs: the quadratic-cost dynamic programme in closed form, model-predictive control with constraints and proportional costs, and the no-trade region.

A portfolio is not rebuilt from scratch each month. Trading costs money, and the forecast that justified a position fades, so the right position now depends on where you are, on the costs of getting
anywhere else, and on what you expect to want later. Three classic answers are implemented here, all for the same objective, the expected utility of a sequence of holdings ``x_0 ... x_(T-1)``::

    J  =  sum_t  beta^t [ alpha_t'x_t  -  (gamma/2) x_t'Sigma x_t  -  (1/2) (x_t - x_(t-1))'Lambda(x_t - x_(t-1))  -  kappa'|x_t - x_(t-1)| ]

``alpha_t`` is the expected return of period ``t`` known when the trade is made (a path: a forecast that decays), ``Sigma`` the covariance of the period's returns, ``gamma`` the risk aversion, ``Lambda`` the
quadratic (market-impact) cost, ``kappa`` the proportional cost per unit traded (spread and commission), ``beta`` a discount factor.

**Quadratic costs, solved exactly** (Garleanu and Pedersen 2013; Mei, DeMiguel and Nogales 2016 for many risky assets and general costs): with ``kappa = 0`` the value function stays quadratic,
``V_t(x) = -1/2 x'A_t x + b_t'x + c_t``, so a backward recursion (a Riccati equation) gives the optimal holdings in closed form, ``x_t = M_t^-1 (alpha_t + beta b_(t+1) + Lambda x_(t-1))`` with
``M_t = gamma Sigma + Lambda + beta A_(t+1)``. The policy is a *partial adjustment*: ``x_t - x_(t-1) = Q_t (a_t - x_(t-1))`` with an *aim portfolio* ``a_t`` and a *trade-rate matrix* ``Q_t``; the cost
of trading makes you move only part of the way to what you would hold if trading were free, and the aim already looks ahead, because it weighs the positions you expect to want in the following
periods. :func:`lq_solve`.

**Constraints and proportional costs, by model-predictive control** (Skaf and Boyd 2009): add leverage, position and exposure limits, or a proportional cost, and no closed form exists. The programme over
the next ``H`` periods is still a quadratic programme (proportional costs and gross limits add one auxiliary variable per name and period), so each period solve it, trade the first step, and re-solve
when the next forecast arrives. A quadratic value function from the unconstrained solution can be added as the terminal reward (approximate dynamic programming), which stands in for everything beyond the
horizon. :func:`mpc_plan`. With a proportional cost the answer has a *no-trade region* around the aim: while the holdings are inside it, do nothing; outside it, trade to its boundary, not to the aim
(:func:`one_period_trade`, :func:`no_trade_region`).

**A bound.** Constraints can only lower the best achievable utility, so the unconstrained optimum is an upper bound on what a constrained policy can reach (Skaf and Boyd tighten it with a Lagrangian
relaxation; this module uses the plain relaxation). :func:`utility` evaluates ``J`` along any path, which is how policies are compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qp import solve_qp, solve_qp_ipm


def cost_matrix(cov, cost, shape: str = "risk") -> np.ndarray:
    """The quadratic trading-cost matrix ``Lambda``: ``cost`` times ``Sigma`` (shape ``risk``: a volatile asset is costlier to trade, Garleanu and Pedersen's assumption), times the identity (``identity``) or
    times ``diag(Sigma)`` (``variance``). An array is taken as given: a vector is a diagonal, a matrix is ``Lambda``."""
    S = np.asarray(cov, dtype=float)
    n = S.shape[0]
    if np.ndim(cost) == 2:
        return 0.5 * (np.asarray(cost, dtype=float) + np.asarray(cost, dtype=float).T)
    if np.ndim(cost) == 1:
        return np.diag(np.asarray(cost, dtype=float))
    if cost < 0:
        raise ValueError("cost must not be negative")
    if shape == "risk":
        return float(cost) * 0.5 * (S + S.T)
    if shape == "identity":
        return float(cost) * np.eye(n)
    if shape == "variance":
        return float(cost) * np.diag(np.diag(S))
    raise ValueError("shape must be risk, identity or variance")


# ------------------------------------------------------------------------------------------------------------------ the exact solution with quadratic costs
@dataclass
class LQSolution:
    holdings: np.ndarray               # (T, n) the optimal holdings after trading in each period, starting from x0
    trades: np.ndarray                 # (T, n)
    aim: np.ndarray                    # (T, n) the holdings at which the policy would not trade at t (what you would hold if you could re-trade freely, looking ahead)
    trade_rate: list                   # Q_t = I - M_t^-1 Lambda: the share of the way from the previous holdings to the aim that is traded
    value: float                       # J of the optimal path
    A: list                            # V_t(x) = -1/2 x'A_t x + b_t'x + c_t for t = 0 ... T
    b: list
    c: list
    M: list
    Lam: np.ndarray
    discount: float

    def next_holdings(self, t: int, x_prev) -> np.ndarray:
        """The optimal holdings at ``t`` for *any* holdings before the trade: the closed-loop policy."""
        g = self._g(t)
        return np.linalg.solve(self.M[t], g + self.Lam @ np.asarray(x_prev, dtype=float))

    def _g(self, t: int) -> np.ndarray:
        return self._alpha[t] + self.discount * self.b[t + 1]

    def value_from(self, x_prev) -> float:
        """``V_0(x)``: the best utility from holdings ``x`` before the first trade."""
        x = np.asarray(x_prev, dtype=float)
        return float(-0.5 * x @ self.A[0] @ x + self.b[0] @ x + self.c[0])


def lq_solve(alpha, cov, risk_aversion: float, cost_quadratic, x0=None, discount: float = 1.0, liquidate: bool = False) -> LQSolution:
    """Backward recursion for the unconstrained problem with quadratic costs. ``alpha`` is the (T, n) path of expected returns; ``cost_quadratic`` is ``Lambda`` (see :func:`cost_matrix`); ``liquidate``
    makes the portfolio be sold at the end at the same quadratic cost."""
    alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
    T, n = alpha.shape
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    Lam = np.asarray(cost_quadratic, dtype=float)
    if S.shape != (n, n) or Lam.shape != (n, n):
        raise ValueError("cov and cost_quadratic must be n by n for n assets")
    if risk_aversion <= 0 or not 0 < discount <= 1:
        raise ValueError("risk_aversion > 0 and 0 < discount <= 1")
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
    A = [None] * (T + 1)
    b = [None] * (T + 1)
    c = [0.0] * (T + 1)
    M = [None] * T
    A[T] = Lam.copy() if liquidate else np.zeros((n, n))
    b[T] = np.zeros(n)
    for t in range(T - 1, -1, -1):
        M[t] = risk_aversion * S + Lam + discount * A[t + 1]
        g = alpha[t] + discount * b[t + 1]
        Minv_g, Minv_L = np.linalg.solve(M[t], g), np.linalg.solve(M[t], Lam)
        A[t] = Lam - Lam @ Minv_L
        A[t] = 0.5 * (A[t] + A[t].T)
        b[t] = Lam @ Minv_g
        c[t] = 0.5 * g @ Minv_g + discount * c[t + 1]
    sol = LQSolution(np.zeros((T, n)), np.zeros((T, n)), np.zeros((T, n)), [], 0.0, A, b, c, M, Lam, float(discount))
    sol._alpha = alpha
    prev = x0.copy()
    for t in range(T):
        g = sol._g(t)
        sol.holdings[t] = np.linalg.solve(M[t], g + Lam @ prev)
        sol.trades[t] = sol.holdings[t] - prev
        sol.aim[t] = np.linalg.solve(M[t] - Lam, g)                                  # M - Lambda = gamma Sigma + beta A: positive definite when Sigma is
        sol.trade_rate.append(np.eye(n) - np.linalg.solve(M[t], Lam))
        prev = sol.holdings[t]
    sol.value = sol.value_from(x0)
    return sol


def stationary_trade_rate(cov, risk_aversion: float, cost_quadratic, discount: float = 0.99, tol: float = 1e-12, max_iter: int = 100000):
    """The long-run (infinite-horizon) trade-rate matrix ``Q = I - M^-1 Lambda`` and ``M``, found by iterating the Riccati recursion to its fixed point. With a single asset ``Q`` is the share of the gap to the
    aim that is closed each period: it falls as trading gets costlier and rises with risk aversion."""
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    Lam = np.asarray(cost_quadratic, dtype=float)
    n = S.shape[0]
    A = np.zeros((n, n))
    for _ in range(max_iter):
        M = risk_aversion * S + Lam + discount * A
        A_new = Lam - Lam @ np.linalg.solve(M, Lam)
        A_new = 0.5 * (A_new + A_new.T)
        if np.abs(A_new - A).max() < tol:
            A = A_new
            break
        A = A_new
    M = risk_aversion * S + Lam + discount * A
    return np.eye(n) - np.linalg.solve(M, Lam), M, A


def utility(holdings, alpha, cov, risk_aversion: float, x0=None, cost_quadratic=None, cost_linear=None, discount: float = 1.0) -> float:
    """``J`` along a path of holdings (the objective of this module), for comparing policies. ``holdings`` and ``alpha`` are (T, n)."""
    X = np.atleast_2d(np.asarray(holdings, dtype=float))
    alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
    T, n = X.shape
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    Lam = np.zeros((n, n)) if cost_quadratic is None else np.asarray(cost_quadratic, dtype=float)
    kappa = np.zeros(n) if cost_linear is None else np.broadcast_to(np.asarray(cost_linear, dtype=float), (n,))
    prev = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
    total = 0.0
    for t in range(T):
        d = X[t] - prev
        total += discount ** t * (alpha[t] @ X[t] - 0.5 * risk_aversion * X[t] @ S @ X[t] - 0.5 * d @ Lam @ d - kappa @ np.abs(d))
        prev = X[t]
    return float(total)


# ------------------------------------------------------------------------------------------------------------------ constraints and proportional costs: a quadratic programme per period
@dataclass
class MPCResult:
    holdings: np.ndarray               # (H, n) the plan; the first row is the trade to make now
    trades: np.ndarray
    objective: float                   # J of the plan (maximised), including the terminal reward if one was given
    status: str
    ok: bool
    solver: str


def mpc_plan(alpha, cov, risk_aversion: float, x_prev, cost_quadratic=None, cost_linear=None, discount: float = 1.0, lb=None, ub=None, net=None, max_gross=None, exposures=None,
             exposure_bounds=None, terminal=None, solver: str = "auto") -> MPCResult:
    """The best holdings for the next ``H = len(alpha)`` periods from ``x_prev``, under the constraints, as one quadratic programme.

    ``lb, ub``: position bounds (a number or one per asset) in every period; ``net``: bounds ``(low, high)`` on the sum of the holdings; ``max_gross``: a limit on the sum of absolute holdings;
    ``exposures`` (k by n) with ``exposure_bounds`` (k by 2): any other linear exposures, such as beta or a sector; ``cost_linear``: the proportional cost per unit traded (a number or one per asset);
    ``terminal``: ``(A, b)`` of a value function ``-1/2 x'Ax + b'x`` for the holdings after the last period (from :func:`lq_solve`), the approximate-dynamic-programming terminal reward.
    ``solver`` is ``ipm`` (a dense interior-point method, accurate and fast for a few hundred variables), ``admm`` or ``auto``."""
    alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
    H, n = alpha.shape
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    x_prev = np.asarray(x_prev, dtype=float)
    Lam = np.zeros((n, n)) if cost_quadratic is None else np.asarray(cost_quadratic, dtype=float)
    kappa = np.zeros(n) if cost_linear is None else np.broadcast_to(np.asarray(cost_linear, dtype=float), (n,)).copy()
    use_l1, use_gross = bool(np.any(kappa > 0)), max_gross is not None
    if risk_aversion <= 0 or not 0 < discount <= 1 or S.shape != (n, n) or Lam.shape != (n, n) or x_prev.shape != (n,) or (kappa < 0).any():
        raise ValueError("risk_aversion > 0, 0 < discount <= 1, cov and cost matrices n by n, x_prev of length n, costs non-negative")
    nx = H * n
    N = nx * (1 + int(use_l1) + int(use_gross))
    it, iu = nx, nx + nx * int(use_l1)                                                     # where the |trade| and |holding| variables start
    X = lambda s: slice(s * n, (s + 1) * n)
    P, q = np.zeros((N, N)), np.zeros(N)
    w = discount ** np.arange(H)
    for s in range(H):
        P[X(s), X(s)] += w[s] * (risk_aversion * S + Lam)
        q[X(s)] -= w[s] * alpha[s]
        if s == 0:
            q[X(0)] -= Lam @ x_prev
        else:
            P[X(s), X(s - 1)] -= w[s] * Lam
            P[X(s - 1), X(s)] -= w[s] * Lam
            P[X(s - 1), X(s - 1)] += w[s] * Lam
        if use_l1:
            q[it + s * n:it + (s + 1) * n] += w[s] * kappa
    if terminal is not None:
        At, bt = terminal
        P[X(H - 1), X(H - 1)] += discount ** H * np.asarray(At, dtype=float)
        q[X(H - 1)] -= discount ** H * np.asarray(bt, dtype=float)
    rows, lo, hi = [], [], []

    def add(row, low, high):
        rows.append(row)
        lo.append(low)
        hi.append(high)

    for s in range(H):
        if use_l1:
            for i in range(n):                                                             # t >= x_s - x_(s-1)  and  t >= x_(s-1) - x_s
                up, dn = np.zeros(N), np.zeros(N)
                up[it + s * n + i], up[s * n + i] = 1.0, -1.0
                dn[it + s * n + i], dn[s * n + i] = 1.0, 1.0
                if s == 0:
                    add(up, -x_prev[i], np.inf)
                    add(dn, x_prev[i], np.inf)
                else:
                    up[(s - 1) * n + i], dn[(s - 1) * n + i] = 1.0, -1.0
                    add(up, 0.0, np.inf)
                    add(dn, 0.0, np.inf)
        if net is not None:
            row = np.zeros(N)
            row[X(s)] = 1.0
            add(row, -np.inf if net[0] is None else float(net[0]), np.inf if net[1] is None else float(net[1]))
        if use_gross:
            for i in range(n):                                                             # u >= x and u >= -x, with u >= 0 through the variable bounds
                a, b2 = np.zeros(N), np.zeros(N)
                a[iu + s * n + i], a[s * n + i] = 1.0, -1.0
                b2[iu + s * n + i], b2[s * n + i] = 1.0, 1.0
                add(a, 0.0, np.inf)
                add(b2, 0.0, np.inf)
            row = np.zeros(N)
            row[iu + s * n:iu + (s + 1) * n] = 1.0
            add(row, -np.inf, float(max_gross))
        if exposures is not None:
            E, EB = np.atleast_2d(np.asarray(exposures, dtype=float)), np.atleast_2d(np.asarray(exposure_bounds, dtype=float))
            for k in range(E.shape[0]):
                row = np.zeros(N)
                row[X(s)] = E[k]
                add(row, EB[k, 0], EB[k, 1])
    lbv, ubv = np.full(N, -np.inf), np.full(N, np.inf)
    if lb is not None:
        lbv[:nx] = np.tile(np.broadcast_to(np.asarray(lb, dtype=float), (n,)), H)
    if ub is not None:
        ubv[:nx] = np.tile(np.broadcast_to(np.asarray(ub, dtype=float), (n,)), H)
    if use_l1:
        lbv[it:it + nx] = 0.0
    if use_gross:
        lbv[iu:iu + nx] = 0.0
    A_mat, l_vec, u_vec = (np.array(rows), np.array(lo), np.array(hi)) if rows else (None, None, None)
    chosen = solver if solver != "auto" else ("ipm" if N <= 900 else "admm")
    if chosen == "ipm":
        r = solve_qp_ipm(P, q, A=A_mat, l=l_vec, u=u_vec, lb=lbv, ub=ubv)
    elif chosen == "admm":
        r = solve_qp(P, q, A=A_mat, l=l_vec, u=u_vec, lb=lbv, ub=ubv, max_iter=60000)
    else:
        raise ValueError("solver must be auto, ipm or admm")
    if not r.ok:
        return MPCResult(np.full((H, n), np.nan), np.full((H, n), np.nan), float("nan"), r.status, False, chosen)
    hold = r.x[:nx].reshape(H, n)
    trades = np.vstack([hold[:1] - x_prev, np.diff(hold, axis=0)])
    const = -0.5 * float(w[0] * x_prev @ Lam @ x_prev)
    return MPCResult(hold, trades, float(-r.objective + const), r.status, True, chosen)


def decaying_path(alpha0, persistence: float, horizon: int) -> np.ndarray:
    """The expected-return path of a forecast that fades: ``alpha0 * persistence^k`` for ``k = 0 ... horizon - 1``."""
    if not 0.0 <= persistence <= 1.0 or horizon < 1:
        raise ValueError("0 <= persistence <= 1 and horizon >= 1")
    a = np.asarray(alpha0, dtype=float)
    return persistence ** np.arange(horizon)[:, None] * a[None, :]


def mpc_step(alpha_path, cov, risk_aversion: float, x_prev, cost_quadratic=None, cost_linear=None, discount: float = 1.0, plan_horizon: int | None = None, terminal: bool = True,
             **constraints) -> tuple[np.ndarray, MPCResult]:
    """One step of model-predictive control. ``alpha_path`` is the whole forecast path (the expected return of each coming period); the plan covers its first ``plan_horizon`` periods (all of it by default)
    and returns the holdings to take now and the plan. With ``terminal`` the periods beyond the plan are summarised by the value function of the *unconstrained* problem over the rest of the path
    (approximate dynamic programming, Skaf and Boyd): with no constraints at all this reproduces the exact closed-form policy whatever the plan horizon; with constraints it is an approximation,
    and a good one when the constraints bind rarely."""
    alpha_path = np.atleast_2d(np.asarray(alpha_path, dtype=float))
    n = alpha_path.shape[1]
    H = len(alpha_path) if plan_horizon is None else min(int(plan_horizon), len(alpha_path))
    if H < 1:
        raise ValueError("plan_horizon must be at least 1")
    Lam = np.zeros((n, n)) if cost_quadratic is None else np.asarray(cost_quadratic, dtype=float)
    term = None
    if terminal and H < len(alpha_path):
        rest = lq_solve(alpha_path[H:], cov, risk_aversion, Lam, discount=discount)
        term = (rest.A[0], rest.b[0])
    plan = mpc_plan(alpha_path[:H], cov, risk_aversion, x_prev, Lam, cost_linear, discount, terminal=term, **constraints)
    return (plan.holdings[0] if plan.ok else np.asarray(x_prev, dtype=float)), plan


# ------------------------------------------------------------------------------------------------------------------ proportional costs: the no-trade region
def no_trade_region(alpha, cov, risk_aversion: float, cost_linear) -> tuple[np.ndarray, np.ndarray]:
    """For one period with a proportional cost ``kappa`` and no other cost, the holdings ``x`` at which it is optimal not to trade satisfy ``|alpha - gamma Sigma x| <= kappa`` in every asset: a
    parallelotope around the aim ``(gamma Sigma)^-1 alpha``. Returns the aim and the half-widths of the region along the axes of the *gradient*: ``(aim, kappa)``; see :func:`inside_no_trade_region`."""
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    a = np.asarray(alpha, dtype=float)
    return np.linalg.solve(risk_aversion * S, a), np.broadcast_to(np.asarray(cost_linear, dtype=float), a.shape).copy()


def inside_no_trade_region(x, alpha, cov, risk_aversion: float, cost_linear, tol: float = 1e-9) -> bool:
    """Whether holding ``x`` already satisfies the one-period no-trade condition ``|alpha - gamma Sigma x| <= kappa``."""
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    grad = np.asarray(alpha, dtype=float) - risk_aversion * S @ np.asarray(x, dtype=float)
    return bool((np.abs(grad) <= np.asarray(cost_linear, dtype=float) + tol).all())


def one_period_trade(x_prev, alpha, cov, risk_aversion: float, cost_linear, **constraints) -> np.ndarray:
    """The optimal holdings after one period's trade with a proportional cost (a one-period :func:`mpc_plan`)."""
    plan = mpc_plan(np.atleast_2d(alpha), cov, risk_aversion, x_prev, None, cost_linear, 1.0, **constraints)
    if not plan.ok:
        raise ValueError(f"the one-period problem did not solve ({plan.status})")
    return plan.holdings[0]
