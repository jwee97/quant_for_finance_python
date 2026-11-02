"""Cost-aware portfolio allocation (spec v3, Release 6, P2-04, P2-14, P2-16 and section 9.2).

The objective, for a return forecast that is calibrated in the units below, is

    maximise   mu'w - (lambda/2) w'Sigma w - c'|w - w0| - (kappa/2) |w - w0|^2 - sum_i k_i |w_i - w0_i|^1.5 + r_cash (1 - sum w)

with ``w0`` the book actually held. **Units.** ``mu`` and ``Sigma`` are for the same window (the holding period between rebalances: use ``window_inputs`` to scale a per-day covariance by the number of days, which
assumes the daily returns are uncorrelated through time and says so), and every cost is a fraction of capital per rebalance, so that all four terms are returns over that window. ``c`` is the one-way
proportional cost per unit of traded weight (commission, half-spread, slippage in decimal), ``kappa`` a quadratic turnover penalty, and the last cost term is the square-root market-impact law of the event-driven
stack, ``price move = Y sigma sqrt(participation)``, written in weight units: a trade of ``d`` of capital ``K`` in a name with average dollar volume ``V`` and daily volatility ``sigma`` costs
``Y sigma sqrt(K/V) |d|^1.5`` of capital (the same arithmetic as ``realism.impact_bps``, which the tests compare). Cash earns ``r_cash``; borrowing is not financed separately here.

**Constraints**: ``|w_i| <= position_max`` (zero on the short side when ``long_only``), gross ``sum |w| <= gross_max``, optional net bounds, optional group caps on gross, and a participation limit
``|d_i| <= max_participation * V_i / K`` when capital and average dollar volume are given. Integer lots and borrow/margin limits are not modelled here (the realistic execution adapter applies lots and borrow when
the target is executed).

**What comes back** (``Allocation``): the weights (``None`` when no solution exists and no fallback was configured), a status (``optimal``, ``infeasible``, ``failed``), the reason, whether a configured fallback was
used (and which), the objective split into expected return, risk, each cost term and the utility, the constraint utilisation, the feasibility residuals of the returned point and the solver's message. An
infeasible problem is detected by a linear programme over the same constraints before any objective is optimised. A fallback is never silent: ``fallback="hold"`` returns the current book, ``"equal_weight"``
an equal-weight book inside the constraints, and both are labelled in the result.

``sensitivity`` re-solves on a predeclared grid (scales of the forecast, the covariance and the costs, and values of the constraints) and reports how far the weights, the risk and the turnover move and how many
points failed. ``simulate`` runs the allocation through time with the book that was actually held at each step (drifted by the realised returns), so a participation limit that stops a trade short is carried into
the next decision instead of being forgotten.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

VERSION = "allocation-v1"
FALLBACKS = ("none", "hold", "equal_weight")
MAX_GRID = 64


@dataclass(frozen=True)
class AllocationSpec:
    risk_aversion: float = 5.0
    cost: float = 0.0005                    # one-way proportional cost per unit of traded weight (decimal: 5 bps)
    turnover_penalty: float = 0.0           # kappa
    impact: float | None = None             # Y of the square-root law; needs capital, average dollar volume and sigma
    cash_return: float = 0.0
    long_only: bool = False
    gross_max: float = 1.0
    net_min: float | None = None
    net_max: float | None = None
    position_max: float = 0.35
    group_max: float | None = None
    max_participation: float | None = None
    ridge: float = 1e-8                     # added to the covariance diagonal as a fraction of its mean, so a singular matrix still has a solution; reported
    tol: float = 1e-9
    max_iter: int = 400
    fallback: str = "none"

    def __post_init__(self):
        if self.risk_aversion < 0 or self.cost < 0 or self.turnover_penalty < 0 or (self.impact is not None and self.impact < 0):
            raise ValueError("risk aversion, cost, turnover penalty and impact are not negative")
        if self.gross_max <= 0 or self.position_max <= 0:
            raise ValueError("gross_max and position_max are positive")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}")
        if self.max_participation is not None and not 0 < self.max_participation <= 1:
            raise ValueError("max_participation is a share of average volume in (0, 1]")

    def content(self) -> dict:
        return {"schema": VERSION, **asdict(self)}

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class Allocation:
    weights: pd.Series | None
    status: str
    reason: str = ""
    fallback_used: str | None = None
    current: pd.Series | None = None
    components: dict = field(default_factory=dict)
    utilization: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)
    ridge_added: float = 0.0
    units: str = "returns over one holding window, as fractions of capital"

    @property
    def trades(self) -> pd.Series | None:
        return None if self.weights is None or self.current is None else self.weights - self.current

    @property
    def no_trade(self) -> bool:
        t = self.trades
        return t is not None and bool(t.abs().max() <= 1e-8)

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "fallback_used": self.fallback_used, "weights": None if self.weights is None else {str(k): float(v) for k, v in self.weights.items()},
                "trades": None if self.trades is None else {str(k): float(v) for k, v in self.trades.items()}, "no_trade": self.no_trade, "components": self.components, "utilization": self.utilization,
                "residuals": self.residuals, "solver": self.solver, "ridge_added": self.ridge_added, "units": self.units}


def window_inputs(mu_window: pd.Series, cov_daily: pd.DataFrame, days: int) -> tuple[pd.Series, pd.DataFrame]:
    """``(mu, Sigma)`` for a ``days``-day holding window: the forecast is already for the window, the daily covariance is multiplied by ``days`` (daily returns assumed uncorrelated through time)."""
    if days < 1:
        raise ValueError("the window is at least one day")
    return mu_window, cov_daily * float(days)


def impact_coefficients(spec: AllocationSpec, capital: float, adv: pd.Series, sigma: pd.Series) -> pd.Series:
    """``k_i = Y sigma_i sqrt(K / V_i)``: the cost of a trade ``d`` in name ``i`` is ``k_i |d|^1.5`` of capital. Names with no usable volume or volatility get ``inf`` (they cannot be traded)."""
    k = spec.impact * sigma * np.sqrt(capital / adv.where(adv > 0))
    return k.where(np.isfinite(k), np.inf)


def trade_limits(spec: AllocationSpec, capital: float | None, adv: pd.Series | None, index) -> pd.Series | None:
    """The largest trade per name, in weight units: ``max_participation * V_i / K``; zero for a name with no volume."""
    if spec.max_participation is None:
        return None
    if capital is None or adv is None:
        raise ValueError("a participation limit needs the capital and the average dollar volume of each name")
    return (spec.max_participation * adv.reindex(index).fillna(0.0).clip(lower=0.0) / float(capital)).clip(upper=10.0)


def _objective_parts(w: np.ndarray, w0: np.ndarray, mu: np.ndarray, cov: np.ndarray, spec: AllocationSpec, kimp: np.ndarray | None) -> dict:
    d = w - w0
    var = float(w @ cov @ w)
    lin = float(spec.cost * np.abs(d).sum())
    quad = float(0.5 * spec.turnover_penalty * d @ d)
    imp = float((kimp * np.abs(d) ** 1.5).sum()) if kimp is not None else 0.0
    er = float(mu @ w)
    cash = float(spec.cash_return * (1.0 - w.sum()))
    util = er + cash - 0.5 * spec.risk_aversion * var - lin - quad - imp
    return {"expected_return": er, "cash_return": cash, "risk": float(np.sqrt(max(var, 0.0))), "variance_penalty": float(0.5 * spec.risk_aversion * var), "linear_cost": lin, "turnover_penalty": quad,
            "impact_cost": imp, "total_cost": lin + quad + imp, "utility": util}


def _structure(n: int, spec: AllocationSpec, groups: np.ndarray | None, limit: np.ndarray | None):
    """Linear constraints on x = [p, q, b, s]: w = p - q, trade d = b - s."""
    eye, zero = np.eye(n), np.zeros((n, n))
    a_eq = np.hstack([eye, -eye, -eye, eye])
    rows, rhs = [], []
    ones = np.ones(n)
    rows.append(np.concatenate([-ones, -ones, np.zeros(n), np.zeros(n)]))
    rhs.append(-spec.gross_max)
    if spec.net_min is not None:
        rows.append(np.concatenate([ones, -ones, np.zeros(n), np.zeros(n)]))
        rhs.append(spec.net_min)
    if spec.net_max is not None:
        rows.append(np.concatenate([-ones, ones, np.zeros(n), np.zeros(n)]))
        rhs.append(-spec.net_max)
    if spec.group_max is not None and groups is not None:
        for g in sorted(set(groups.tolist())):
            m = (groups == g).astype(float)
            rows.append(np.concatenate([-m, -m, np.zeros(n), np.zeros(n)]))
            rhs.append(-spec.group_max)
    if limit is not None:
        for i in range(n):
            r = np.zeros(4 * n)
            r[2 * n + i] = -1.0
            r[3 * n + i] = -1.0
            rows.append(r)
            rhs.append(-float(limit[i]))
    ub = [spec.position_max] * n + ([0.0] * n if spec.long_only else [spec.position_max] * n)
    big = [None] * n
    bounds = [(0.0, u) for u in ub] + [(0.0, None) for _ in big] + [(0.0, None) for _ in big]
    return a_eq, np.array(rows).reshape(len(rows), 4 * n), np.array(rhs, dtype=float), bounds, zero


def _fallback(spec: AllocationSpec, w0: pd.Series, why: str, status: str, base: dict) -> Allocation:
    if spec.fallback == "hold":
        return Allocation(w0.copy(), status, why, "hold", w0, **base)
    if spec.fallback == "equal_weight":
        n = len(w0)
        each = min(spec.gross_max / n, spec.position_max)
        w = pd.Series(0.0 if n == 0 else each, index=w0.index)
        if spec.net_min is not None and w.sum() < spec.net_min:
            w = w * 0.0
        return Allocation(w, status, why, "equal_weight", w0, **base)
    return Allocation(None, status, why, None, w0, **base)


def solve(mu: pd.Series, cov: pd.DataFrame, current: pd.Series | None = None, spec: AllocationSpec | None = None, *, groups: dict | None = None, capital: float | None = None,
          adv: pd.Series | None = None, sigma: pd.Series | None = None) -> Allocation:
    spec = spec or AllocationSpec()
    assets = list(mu.index)
    n = len(assets)
    w0s = (current if current is not None else pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    w0 = w0s.to_numpy(dtype=float)
    m = mu.to_numpy(dtype=float)
    s = cov.reindex(index=assets, columns=assets).to_numpy(dtype=float)
    base = {}
    if not (np.isfinite(m).all() and np.isfinite(s).all()):
        return _fallback(spec, w0s, "the forecast or the covariance has missing values: nothing is optimised on them", "failed", base)
    s = (s + s.T) / 2.0
    ridge = float(spec.ridge * np.trace(s) / max(n, 1))
    s = s + ridge * np.eye(n)
    kimp = None
    if spec.impact:
        if capital is None or adv is None or sigma is None:
            raise ValueError("market impact needs the capital, the average dollar volume and the daily volatility of each name")
        kimp = impact_coefficients(spec, float(capital), adv.reindex(assets), sigma.reindex(assets)).to_numpy(dtype=float)
    limit_s = trade_limits(spec, capital, adv, assets)
    limit = None if limit_s is None else limit_s.to_numpy(dtype=float)
    garr = None if groups is None else np.array([groups.get(a, "_none") for a in assets])
    a_eq, a_in, r_in, bounds, _ = _structure(n, spec, garr, limit)
    if kimp is not None:                                                                       # a name that cannot be traded (no volume) is held at its current weight
        for i in np.where(~np.isfinite(kimp))[0]:
            bounds[2 * n + i] = (0.0, 0.0)
            bounds[3 * n + i] = (0.0, 0.0)
        kimp = np.where(np.isfinite(kimp), kimp, 0.0)
    # a linear programme over the same constraints decides feasibility before any objective is optimised
    lp = linprog(np.zeros(4 * n), A_ub=-a_in if len(r_in) else None, b_ub=-r_in if len(r_in) else None, A_eq=a_eq, b_eq=w0, bounds=bounds, method="highs")
    base = {"ridge_added": ridge}
    if lp.status == 2 or not lp.success:
        why = "no book satisfies the constraints from the current holdings" + (" with the participation limits" if limit is not None else "") + f" ({lp.message.strip()})"
        return _fallback(spec, w0s, why, "infeasible", {**base, "solver": {"method": "highs (feasibility)", "message": lp.message}})
    x0 = lp.x

    def f(x):
        p, q, b, t = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:]
        w, d, tt = p - q, b - t, b + t
        val = -m @ w - spec.cash_return * (1.0 - w.sum()) + 0.5 * spec.risk_aversion * w @ s @ w + spec.cost * tt.sum() + 0.5 * spec.turnover_penalty * d @ d
        if kimp is not None:
            val += float((kimp * tt ** 1.5).sum())
        return float(val)

    def grad(x):
        p, q, b, t = x[:n], x[n:2 * n], x[2 * n:3 * n], x[3 * n:]
        w, d, tt = p - q, b - t, np.clip(b + t, 0.0, None)
        gw = -m + spec.cash_return + spec.risk_aversion * (s @ w)
        gt = spec.cost + (1.5 * kimp * np.sqrt(tt) if kimp is not None else 0.0)
        gd = spec.turnover_penalty * d
        return np.concatenate([gw, -gw, gd + gt, -gd + gt])

    cons = [{"type": "eq", "fun": lambda x: a_eq @ x - w0, "jac": lambda x: a_eq}]
    if len(r_in):
        cons.append({"type": "ineq", "fun": lambda x: a_in @ x - r_in, "jac": lambda x: a_in})
    res = minimize(f, x0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": spec.max_iter, "ftol": 1e-14})
    x = np.clip(res.x, [b[0] for b in bounds], [np.inf if b[1] is None else b[1] for b in bounds])
    eq_res = float(np.abs(a_eq @ x - w0).max())
    in_res = float(np.maximum(0.0, r_in - a_in @ x).max()) if len(r_in) else 0.0
    ub = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    b_res = float(np.maximum(0.0, x - ub).max())
    residuals = {"equality": eq_res, "constraints": in_res, "bounds": b_res, "max": max(eq_res, in_res, b_res)}
    solver = {"method": "SLSQP", "iterations": int(res.nit), "message": str(res.message), "success": bool(res.success)}
    if residuals["max"] > max(1e-7, 100 * spec.tol):
        return _fallback(spec, w0s, f"the solver returned a point that violates the constraints by {residuals['max']:.2e}", "failed", {**base, "residuals": residuals, "solver": solver})
    if not res.success:
        # SLSQP can stop on its line search at a point that is already optimal; accept only if a restart from the point does not improve the objective and the point is feasible
        again = minimize(f, x, jac=grad, bounds=bounds, constraints=cons, method="SLSQP", options={"maxiter": spec.max_iter, "ftol": 1e-14})
        if f(again.x) < f(x) - 1e-9:
            return _fallback(spec, w0s, f"the solver did not converge: {res.message}", "failed", {**base, "residuals": residuals, "solver": solver})
    w = (x[:n] - x[n:2 * n])
    w = np.where(np.abs(w) < 1e-12, 0.0, w)
    wser = pd.Series(w, index=assets)
    parts = _objective_parts(w, w0, m, s, spec, kimp)
    gross, net = float(np.abs(w).sum()), float(w.sum())
    util = {"gross": gross, "gross_limit": spec.gross_max, "gross_utilisation": gross / spec.gross_max, "net": net, "positions_at_cap": [a for a, v in zip(assets, w) if abs(v) >= spec.position_max - 1e-7],
            "trades_at_participation_limit": [] if limit is None else [a for a, dd, lim in zip(assets, w - w0, limit) if lim > 0 and abs(dd) >= lim - 1e-7]}
    if garr is not None and spec.group_max is not None:
        util["groups"] = {str(g): float(np.abs(w[garr == g]).sum()) for g in sorted(set(garr.tolist()))}
    return Allocation(wser, "optimal", "", None, w0s, parts, util, residuals, solver, ridge)


# ------------------------------------------------------------------------------------------------------------------ P2-14: perturbation grid
GRID_KEYS = ("mu_scale", "cov_scale", "cost_scale", "gross_max", "position_max", "risk_aversion")


def grid_hash(grid: dict) -> str:
    return hashlib.sha256(json.dumps({k: list(v) for k, v in sorted(grid.items())}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sensitivity(mu: pd.Series, cov: pd.DataFrame, current: pd.Series | None, spec: AllocationSpec, grid: dict, **kw) -> dict:
    """Re-solve on every point of a predeclared grid (``mu_scale``, ``cov_scale``, ``cost_scale``, ``gross_max``, ``position_max``, ``risk_aversion``; a missing key stays at its base value) and report how the book
    moves. The base is the unperturbed problem. Failures are counted and listed, never dropped. The grid and its hash are fixed before the first solve; this analysis is development-data work and the caller records it as a look."""
    unknown = set(grid) - set(GRID_KEYS)
    if unknown or not grid or any(not isinstance(v, (list, tuple)) or not v for v in grid.values()):
        raise ValueError(f"the grid maps {list(GRID_KEYS)} to non-empty lists of values; got {sorted(grid)}")
    keys = sorted(grid)
    points = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    if len(points) > MAX_GRID:
        raise ValueError(f"{len(points)} points: more than {MAX_GRID}")
    base = solve(mu, cov, current, spec, **kw)
    rows = []
    for pt in points:
        sp = spec
        rep = {k: pt[k] for k in ("gross_max", "position_max", "risk_aversion") if k in pt}
        if "cost_scale" in pt:
            rep["cost"] = spec.cost * pt["cost_scale"]
            rep["turnover_penalty"] = spec.turnover_penalty * pt["cost_scale"]
            rep["impact"] = None if spec.impact is None else spec.impact * pt["cost_scale"]
        if rep:
            sp = AllocationSpec(**{**asdict(spec), **rep})
        a = solve(mu * pt.get("mu_scale", 1.0), cov * pt.get("cov_scale", 1.0), current, sp, **kw)
        row = {"point": pt, "status": a.status, "reason": a.reason}
        if a.weights is not None and base.weights is not None:
            row.update(weight_change=float(0.5 * (a.weights - base.weights).abs().sum()), risk=a.components["risk"], turnover=float(a.trades.abs().sum()), cost=a.components["total_cost"],
                       gross=a.utilization["gross"], no_trade=a.no_trade)
        rows.append(row)
    ok = [r for r in rows if "weight_change" in r]
    ch = np.array([r["weight_change"] for r in ok])
    return {"version": VERSION, "grid": {k: list(v) for k, v in grid.items()}, "grid_hash": grid_hash(grid), "base_status": base.status, "base": base.as_dict(), "points": rows,
            "failed": int(len(rows) - len(ok)), "summary": None if not len(ok) else {"max_weight_change": float(ch.max()), "median_weight_change": float(np.median(ch)), "share_no_trade": float(np.mean([r["no_trade"] for r in ok])),
                                                                                  "risk_range": [float(min(r["risk"] for r in ok)), float(max(r["risk"] for r in ok))],
                                                                                  "turnover_range": [float(min(r["turnover"] for r in ok)), float(max(r["turnover"] for r in ok))]},
            "notes": ["the whole grid is reported; no point is chosen", "weight_change is half the L1 distance to the base book"]}


# ------------------------------------------------------------------------------------------------------------------ P2-16: through time, with the holdings that were actually held
def simulate(mu_path: pd.DataFrame, cov: pd.DataFrame, returns: pd.DataFrame, spec: AllocationSpec, *, capital: float | None = None, adv: pd.DataFrame | pd.Series | None = None,
             sigma: pd.Series | None = None, start: pd.Series | None = None, groups: dict | None = None) -> dict:
    """Allocate at each row of ``mu_path`` (the forecast for the window that follows, indexed by decision date) from the book held after the previous window's returns, and earn ``returns`` (the same rows: the
    per-asset return of each window) on the executed book less its costs. A participation limit that stops a trade short is carried: the next decision starts from the partial fill."""
    assets = list(mu_path.columns)
    held = (start if start is not None else pd.Series(0.0, index=assets)).reindex(assets).fillna(0.0)
    rows, books = [], {}
    for date, mu in mu_path.iterrows():
        v = adv.loc[date] if isinstance(adv, pd.DataFrame) else adv
        a = solve(mu, cov, held, spec, groups=groups, capital=capital, adv=v, sigma=sigma)
        if a.weights is None:
            rows.append({"date": date, "status": a.status, "reason": a.reason, "net_return": float("nan")})
            continue
        executed = a.weights
        r = returns.loc[date].reindex(assets).fillna(0.0)
        gross_ret = float((executed * r).sum()) + spec.cash_return * (1.0 - float(executed.sum()))
        cost = a.components["total_cost"]
        rows.append({"date": date, "status": a.status, "fallback": a.fallback_used, "turnover": float(a.trades.abs().sum()), "gross_return": gross_ret, "cost": cost, "net_return": gross_ret - cost,
                     "gross": a.utilization["gross"], "limited_by_participation": a.utilization.get("trades_at_participation_limit", []), "no_trade": a.no_trade})
        books[date] = executed
        growth = 1.0 + gross_ret - cost                                                        # equity after the window; the book drifts with the returns and this is what is held at the next decision
        held = executed * (1.0 + r) / growth
    table = pd.DataFrame(rows).set_index("date")
    return {"version": VERSION, "table": table, "books": pd.DataFrame(books).T, "net_return": float(table["net_return"].sum()), "total_cost": float(table["cost"].sum()) if "cost" in table else 0.0,
            "note": "the book held at each decision is the previous executed book after the window's returns, as a share of the equity that is left after costs"}
