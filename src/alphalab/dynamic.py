"""Alpha that fades, and what it costs to trade it.

Grinold and Kahn, *Advances in Active Portfolio Management*: a forecast has a half-life; trading it costs; the best policy trades part of the way to the portfolio you would hold if trading were free and
sometimes not at all. The optimiser itself (the Garleanu-Pedersen quadratic-cost solution, the model-predictive control with constraints, the no-trade region) is :mod:`src.equity.multiperiod`; this module is
the harness around it: policies on a panel of forecasts and realised returns, and the waterfall of what is lost between the paper portfolio and the traded one.

Conventions: ``alpha`` is a panel (dates x assets) of the expected return per period, known at the close of the date; holdings chosen at the close of ``t`` earn the return of ``t + 1 + lag``; costs are a
proportional ``cost_bps`` per unit traded plus ``impact`` times the squared trade (both in units of capital). Holdings are fractions of capital.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..equity import multiperiod as mp
from ..equity.qp import solve_qp

ANN = 252.0


def decay_path(alpha0, tau: float, steps: int) -> np.ndarray:
    """The forecast ``alpha0`` as it fades under exponential decay with time constant ``tau``: ``alpha0 exp(-k/tau)`` for ``k = 0 ... steps-1``, shape (steps, n). The path to give :func:`src.equity.multiperiod.lq_solve`."""
    a = np.asarray(alpha0, dtype=float)
    k = np.arange(int(steps))[:, None]
    return a[None, :] * np.exp(-k / float(tau)) if np.isfinite(tau) and tau > 0 else np.repeat(a[None, :], int(steps), axis=0)


def aim(alpha, cov, risk_aversion: float = 1.0, lb=None, ub=None) -> np.ndarray:
    """The holdings that maximise ``alpha'x - (gamma/2) x'Sigma x`` with no cost of trading: the closed form ``(gamma Sigma)^-1 alpha`` when unbounded, a QP with box limits ``lb`` and ``ub`` (scalars or one
    per asset; either may be ``None``) otherwise."""
    a = np.asarray(alpha, dtype=float)
    S = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    if lb is None and ub is None:
        return np.linalg.solve(risk_aversion * S, a)
    r = solve_qp(risk_aversion * S, -a, lb=-1e6 if lb is None else lb, ub=1e6 if ub is None else ub)
    return np.asarray(r.x, dtype=float)


def policy_full(ctx):
    """Hold the aim every period: what you would hold if trading were free."""
    return lambda t, x, a: ctx["aim"](a)


def policy_partial(ctx):
    """Garleanu-Pedersen partial adjustment: trade the stationary share ``Q`` of the way from where you are to the aim."""
    Q = ctx["Q"]
    return lambda t, x, a: x + Q @ (ctx["aim"](a) - x)


def policy_band(ctx):
    """No-trade region (proportional cost ``kappa``): hold while ``|alpha - gamma Sigma x| <= kappa`` in every asset, else trade to the edge of the region (the one-period constrained optimum)."""
    S, g, kappa = ctx["cov"], ctx["gamma"], ctx["kappa"]
    k = 1.0 / max(float(np.mean(np.diag(S))), 1e-300)                    # the problem is homogeneous in (alpha, Sigma, kappa): scale to order one so the QP solver is well conditioned

    def f(t, x, a):
        if mp.inside_no_trade_region(x, a, S, g, kappa):
            return x
        for solver in ("ipm", "admm"):                                   # the interior-point solver is faster; the splitting method is the fallback when it stalls
            plan = mp.mpc_plan(np.atleast_2d(a * k), S * k, g, x, None, kappa * k, 1.0, solver=solver)
            if plan.ok:
                return plan.holdings[0]
        raise ValueError(f"the one-period trade did not solve ({plan.status})")

    return f


def policy_threshold(ctx, share: float = 0.25):
    """A nonlinear rule: trade the whole way to the aim only when the distance to it, as a share of the aim's gross size, exceeds ``share``; otherwise hold."""
    def f(t, x, a):
        target = ctx["aim"](a)
        return target if np.abs(target - x).sum() > share * max(np.abs(target).sum(), 1e-12) else x

    return f


POLICIES = {"full": policy_full, "partial": policy_partial, "band": policy_band, "threshold": policy_threshold}


def simulate(alpha: pd.DataFrame, returns: pd.DataFrame, cov, policy: str = "partial", risk_aversion: float = 1.0, cost_bps: float = 5.0, impact: float = 0.0, lag: int = 0, lb=None, ub=None,
             alpha_scale: float = 1.0) -> dict:
    """Trade ``alpha`` with a policy and return the realised net and gross returns, the holdings and the cost. ``cov`` is the per-period covariance (fixed: estimate it on data before the first date)."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
    a, r = alpha.align(returns, join="inner")
    cols = list(a.columns)
    A, Rm = a.to_numpy(dtype=float) * alpha_scale, r.to_numpy(dtype=float)
    n = len(cols)
    if isinstance(cov, pd.DataFrame):
        missing = [c for c in cols if c not in cov.columns]
        if missing:
            raise ValueError(f"the covariance has no row for: {', '.join(map(str, missing[:5]))}")
        S = cov.reindex(index=cols, columns=cols).to_numpy(dtype=float)
    else:
        S = np.asarray(cov, dtype=float)
        if S.shape != (n, n):
            raise ValueError(f"the covariance must be {n} by {n} (one row per column of the alpha panel)")
    if not np.isfinite(S).all():
        raise ValueError("the covariance has non-finite entries")
    kappa = cost_bps * 1e-4
    ctx = {"cov": S, "gamma": risk_aversion, "kappa": np.full(n, kappa), "aim": lambda al: aim(al, S, risk_aversion, lb, ub)}
    if policy == "partial":
        Lam = 2.0 * max(impact, 1e-12) * np.eye(n)                      # the simulation charges impact * d'd = (1/2) d' Lambda d with Lambda = 2 impact I
        ctx["Q"] = mp.stationary_trade_rate(S, risk_aversion, Lam, discount=0.999)[0]
    f = POLICIES[policy](ctx)
    x, held = np.zeros(n), []
    gross_r, net_r, traded = np.zeros(len(a)), np.zeros(len(a)), np.zeros(len(a))
    lag = int(lag)
    for t in range(len(a) - 1 - lag):
        al = np.nan_to_num(A[t])
        x_new = f(t, x, al)
        d = x_new - x
        c = kappa * np.abs(d).sum() + impact * float(d @ d)
        g = float(x_new @ np.nan_to_num(Rm[t + 1 + lag]))
        gross_r[t + 1 + lag] += g
        net_r[t + 1 + lag] += g - c
        traded[t + 1 + lag] += np.abs(d).sum()
        held.append(x_new)
        x = x_new
    H = pd.DataFrame(held, index=a.index[:len(held)], columns=cols)
    s = slice(1 + lag, None)
    return {"net": pd.Series(net_r[s], index=a.index[s]), "gross": pd.Series(gross_r[s], index=a.index[s]), "holdings": H, "traded": pd.Series(traded[s], index=a.index[s]),
            "stats": stats(pd.Series(net_r[s], index=a.index[s]), pd.Series(gross_r[s], index=a.index[s]), pd.Series(traded[s], index=a.index[s]))}


def stats(net: pd.Series, gross: pd.Series | None = None, traded: pd.Series | None = None) -> dict:
    sd = float(net.std(ddof=1))
    out = {"ann_return": float(net.mean() * ANN), "ann_vol": sd * np.sqrt(ANN), "sharpe": float(net.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")}
    if gross is not None:
        out["gross_sharpe"] = float(gross.mean() / gross.std(ddof=1) * np.sqrt(ANN)) if gross.std(ddof=1) > 0 else float("nan")
        out["cost_per_year"] = float((gross - net).sum() / (len(net) / ANN))
    if traded is not None:
        out["turnover_per_year"] = float(traded.sum() / (len(net) / ANN))
    return out


def compare_policies(alpha, returns, cov, policies=("full", "partial", "band", "threshold"), **kw) -> pd.DataFrame:
    """The same forecasts, covariance and costs under each policy: one row per policy."""
    return pd.DataFrame({p: simulate(alpha, returns, cov, p, **kw)["stats"] for p in policies}).T


def implementation_waterfall(alpha, returns, cov, risk_aversion: float = 1.0, cost_bps: float = 5.0, lag: int = 1, lb=0.0, ub=None, trading_policy: str = "partial", **kw) -> pd.DataFrame:
    """Where the paper alpha goes. Five realised-return backtests of the same forecasts: the unconstrained portfolio at no cost (**paper**), with the position limits (**constraints**), with the trading cost paid
    by a policy that ignores it (**costs**), with a policy that respects it (**cost-aware trading**), and one period late (**delay**). ``sharpe`` and ``ann_return`` are the realised net figures at each step and
    ``lost`` the change from the step before."""
    steps = [("paper", dict(policy="full", cost_bps=0.0, lag=0, lb=None, ub=None)),
             ("constraints", dict(policy="full", cost_bps=0.0, lag=0, lb=lb, ub=ub)),
             ("costs", dict(policy="full", cost_bps=cost_bps, lag=0, lb=lb, ub=ub)),
             ("cost-aware trading", dict(policy=trading_policy, cost_bps=cost_bps, lag=0, lb=lb, ub=ub)),
             ("delay", dict(policy=trading_policy, cost_bps=cost_bps, lag=lag, lb=lb, ub=ub))]
    rows, prev = {}, None
    for name, spec in steps:
        s = simulate(alpha, returns, cov, risk_aversion=risk_aversion, **{**kw, **spec})["stats"]
        rows[name] = {"ann_return": s["ann_return"], "sharpe": s["sharpe"], "turnover_per_year": s.get("turnover_per_year", np.nan), "lost_return": (s["ann_return"] - prev["ann_return"]) if prev else 0.0,
                      "lost_sharpe": (s["sharpe"] - prev["sharpe"]) if prev else 0.0}
        prev = s
    return pd.DataFrame(rows).T
