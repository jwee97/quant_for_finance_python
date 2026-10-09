"""Choosing a trade schedule: the quadratic program, the best-execution goals and the one-parameter schedule families.

**The problem.** Split ``X`` shares over ``n`` intervals into slices ``q_i >= 0`` that add up to ``X``. Trading faster costs more impact; trading slower leaves shares exposed to price moves. With ``x_i`` the shares
still unexecuted after slice ``i``::

    J(q) = E[cost](q) + lambda * Var[shortfall](q)         Var = ref^2 sigma^2 sum_i w_i x_i^2     (timing risk)

**Quadratic program (QP).** With impact linear in participation (``beta = 1``) or linearised about the order's average participation (``beta != 1``), ``E[cost] = sum_i k_i q_i^2`` plus a linear term for an
expected price drift, so ``J`` is a convex quadratic in ``q``: ``J = 1/2 q'Hq + g'q`` with ``H = 2 (K + lambda' L'WL)`` and ``g = ref side a - 2 lambda' X L'w`` (``L`` the cumulative-sum matrix,
``lambda' = lambda ref^2 sigma^2``). ``solve_qp`` solves it exactly under ``sum q = X, q >= 0`` with an active-set method. At ``lambda = 0`` the answer is the VWAP schedule (cost-minimising slices are
proportional to the volume), and as ``lambda`` grows the schedule moves toward the front: the Almgren-Chriss family.

**Power-law impact.** With the exponent ``beta`` of the market (about 0.6) the cost is ``sum_i c_i q_i^(1+beta)``: still convex, solved by SLSQP started from the QP answer.

**Risk aversion** is in bps units: ``J_bps = cost_bps + risk_aversion * variance_bps^2``, so ``1e-3`` already weighs risk like cost for an order of ten percent of a day's volume.

**Goals** pick ``lambda`` for you: ``min_cost`` (``lambda = 0``), ``min_cost_given_risk`` and ``min_risk_given_cost`` (bisection on ``lambda`` along the frontier, along which cost rises and risk falls),
``balanced`` (a given ``risk_aversion``) and ``price_improvement`` (maximise the probability that the shortfall beats a target, ``Phi((target - E) / std)``, over the frontier).

**Families.** ``exponential_trade`` trades at a rate that decays like ``exp(-kappa t)`` (``kappa = 0`` is TWAP, ``kappa > 0`` front-loads, ``kappa < 0`` back-loads) and finishes exactly;
``exponential_residual`` lets the SHARES REMAINING decay like ``X exp(-kappa t)`` (each interval trades a fixed fraction of what is left and the final interval sweeps the remainder);
``trade_rate`` trades a fixed share of the market's volume until done. ``fit_*`` finds the parameter that minimises ``J``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .impact import expected_cost, timing_variance
from .market import Market, Order


# ------------------------------------------------------------------------------------------------------------------ quadratic program
def _setup(order: Order, market: Market):
    window = order.window(market)
    volume = market.expected_volume()[window]
    weights = market.variance_weights()[window]
    return volume, weights, order.reference(market), order.shares


def _lambda_dollar(risk_aversion: float, order: Order, market: Market) -> float:
    """``risk_aversion`` (bps units) as the dollar-variance penalty ``lambda`` of the dollar objective."""
    return float(risk_aversion) * 1e4 / (order.reference(market) * order.shares)


def quadratic_form(order: Order, market: Market, risk_aversion: float, alpha_bps: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(H, g)`` of ``J(q) = 1/2 q'Hq + g'q`` (dollars) for the quadratic-cost model described in the module docstring."""
    volume, w, ref, X = _setup(order, market)
    n = len(volume)
    rho = X / volume.sum()
    k = ref * market.eta * market.sigma * rho ** (market.beta - 1.0) / volume                  # cost of slice i is k_i q_i^2, exact at the average participation
    lam = _lambda_dollar(risk_aversion, order, market) * ref ** 2 * market.sigma ** 2
    L = np.tril(np.ones((n, n)))                                                                 # x_i = X - (L q)_i
    H = 2.0 * (np.diag(k) + lam * (L.T * w) @ L)
    drift = np.linspace(0.0, 1.0, n + 1)[:-1] * alpha_bps * 1e-4
    g = order.side * ref * drift - 2.0 * lam * X * (L.T @ w)
    return H, g


def solve_qp_eq(H: np.ndarray, g: np.ndarray, A: np.ndarray, b: np.ndarray, max_iter: int = 500, tol: float = 1e-9) -> np.ndarray:
    """``min 1/2 q'Hq + g'q`` subject to ``A q = b`` and ``q >= 0`` for positive-definite ``H`` by an active-set method (``A`` has one row per equality, for instance one per stock of a basket).
    Raises ``RuntimeError`` if it does not settle."""
    n = len(g)
    A = np.atleast_2d(np.asarray(A, float))
    b = np.atleast_1d(np.asarray(b, float))
    free = np.ones(n, bool)
    scale = max(np.abs(H).max(), 1e-12)
    size = max(float(np.abs(b).max()), 1e-12)
    for _ in range(max_iter):
        idx = np.flatnonzero(free)
        Hf, Af = H[np.ix_(idx, idx)], A[:, idx]
        sol = np.linalg.solve(Hf, np.column_stack([Af.T, g[idx]]))                                # H^-1 A' and H^-1 g
        HiAt, Hig = sol[:, :-1], sol[:, -1]
        nu = np.linalg.solve(Af @ HiAt, b + Af @ Hig)                                            # H_FF q_F + g_F = A_F' nu  and  A_F q_F = b
        qf = HiAt @ nu - Hig
        if (qf < -tol * size).any():
            free[idx[np.argmin(qf)]] = False                                                     # the most negative slice leaves the active set
            continue
        q = np.zeros(n)
        q[idx] = np.maximum(qf, 0.0)
        fixed = np.flatnonzero(~free)
        if len(fixed) == 0:
            return q
        mu = H[fixed] @ q + g[fixed] - A[:, fixed].T @ nu                                        # the marginal cost of opening a fixed slice
        if (mu >= -tol * scale).all():
            return q
        free[fixed[np.argmin(mu)]] = True                                                        # the slice that would pay most to be opened joins
    raise RuntimeError("the active-set method did not settle")


def solve_qp(H: np.ndarray, g: np.ndarray, total: float, max_iter: int = 500, tol: float = 1e-9) -> np.ndarray:
    """``min 1/2 q'Hq + g'q`` subject to ``sum q = total`` and ``q >= 0`` (one stock): :func:`solve_qp_eq` with a single equality."""
    return solve_qp_eq(H, g, np.ones((1, len(g))), np.array([float(total)]), max_iter, tol)


def optimal_schedule(order: Order, market: Market, risk_aversion: float = 1e-3, alpha_bps: float = 0.0, exact: bool = True) -> np.ndarray:
    """The slices (shares per interval of the order's window) minimising ``E[cost] + risk_aversion * variance``. ``exact=False`` stays with the quadratic model (exact when ``beta = 1``); ``exact=True``
    polishes it with the market's own power-law impact."""
    H, g = quadratic_form(order, market, risk_aversion, alpha_bps)
    q = solve_qp(H, g, order.shares)
    if not exact or abs(market.beta - 1.0) < 1e-9:
        return q
    return _power_law(order, market, risk_aversion, alpha_bps, q)


def _power_law(order: Order, market: Market, risk_aversion: float, alpha_bps: float, start: np.ndarray) -> np.ndarray:
    volume, w, ref, X = _setup(order, market)
    n = len(volume)
    c = ref * market.eta * market.sigma * volume ** (-market.beta)
    lam = _lambda_dollar(risk_aversion, order, market) * ref ** 2 * market.sigma ** 2
    drift = np.linspace(0.0, 1.0, n + 1)[:-1] * alpha_bps * 1e-4
    L = np.tril(np.ones((n, n)))
    LtW = L.T * w
    b = market.beta
    s = X                                                                                        # work in units of the order so the optimiser sees numbers near one

    def f(z):
        q = z * s
        x = X - L @ q
        return (c @ np.maximum(q, 0.0) ** (1.0 + b) + order.side * ref * drift @ q + lam * (w @ x ** 2)) / (ref * s)

    def grad(z):
        q = z * s
        x = X - L @ q
        return ((1.0 + b) * c * np.maximum(q, 0.0) ** b + order.side * ref * drift - 2.0 * lam * (LtW @ x)) * s / (ref * s)

    res = minimize(f, start / s, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1.0, "jac": lambda z: np.ones(n)}],
                   options={"maxiter": 300, "ftol": 1e-14})
    if not res.success and f(res.x) > f(start / s):
        return start
    q = np.maximum(res.x, 0.0) * s
    return q * (X / q.sum())


def cost_and_risk(schedule, order: Order, market: Market, alpha_bps: float = 0.0, marketable: float = 1.0) -> dict:
    """Expected cost (bps), risk (bps, one standard deviation) and the dollar figures of a schedule."""
    c = expected_cost(schedule, order, market, alpha_bps, marketable)
    r = timing_variance(schedule, order, market)
    return {"cost_bps": c["total_bps"], "risk_bps": r["std_bps"], "cost": c["total"], "std": r["std"], "detail": c}


def frontier(order: Order, market: Market, risk_aversions=None, alpha_bps: float = 0.0, exact: bool = True) -> list[dict]:
    """The efficient frontier: for each risk aversion the cheapest schedule for its risk, with its expected cost and risk. Cost rises and risk falls as the risk aversion grows."""
    risk_aversions = np.logspace(-6, 0, 25) if risk_aversions is None else np.asarray(risk_aversions, float)
    rows = []
    for lam in risk_aversions:
        q = optimal_schedule(order, market, float(lam), alpha_bps, exact)
        rows.append({"risk_aversion": float(lam), "schedule": q, **cost_and_risk(q, order, market, alpha_bps)})
    return rows


# ------------------------------------------------------------------------------------------------------------------ best-execution goals
def min_cost(order: Order, market: Market, alpha_bps: float = 0.0, exact: bool = True) -> np.ndarray:
    """The cheapest schedule whatever the risk: the VWAP schedule when there is no expected drift."""
    return optimal_schedule(order, market, 0.0, alpha_bps, exact)


def _bisect(target: float, measure, lo: float = 1e-9, hi: float = 1e4, increasing: bool = True, iters: int = 60) -> float:
    """The log-spaced risk aversion where the monotone ``measure`` crosses ``target``."""
    a, b = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        above = measure(np.exp(mid)) > target
        if above == increasing:
            b = mid
        else:
            a = mid
    return float(np.exp(0.5 * (a + b)))


def min_cost_given_risk(order: Order, market: Market, max_risk_bps: float, alpha_bps: float = 0.0, exact: bool = True) -> dict:
    """The cheapest schedule whose timing risk (one standard deviation, bps) is at most ``max_risk_bps``: the point on the frontier where risk reaches the limit. ``feasible`` is always true here (a faster
    schedule always has less risk); ``binding`` says whether the limit actually constrained the choice."""
    q0 = min_cost(order, market, alpha_bps, exact)
    r0 = cost_and_risk(q0, order, market, alpha_bps)
    if r0["risk_bps"] <= max_risk_bps:
        return {"schedule": q0, "risk_aversion": 0.0, "binding": False, "feasible": True, **r0}
    lam = _bisect(max_risk_bps, lambda a: cost_and_risk(optimal_schedule(order, market, a, alpha_bps, exact), order, market, alpha_bps)["risk_bps"], increasing=False)
    q = optimal_schedule(order, market, lam, alpha_bps, exact)
    return {"schedule": q, "risk_aversion": lam, "binding": True, "feasible": True, **cost_and_risk(q, order, market, alpha_bps)}


def min_risk_given_cost(order: Order, market: Market, max_cost_bps: float, alpha_bps: float = 0.0, exact: bool = True) -> dict:
    """The least risky schedule whose expected cost is at most ``max_cost_bps``. ``feasible`` is false when even the cheapest schedule costs more (it is then returned)."""
    q0 = min_cost(order, market, alpha_bps, exact)
    r0 = cost_and_risk(q0, order, market, alpha_bps)
    if r0["cost_bps"] > max_cost_bps:
        return {"schedule": q0, "risk_aversion": 0.0, "binding": True, "feasible": False, **r0}
    top = optimal_schedule(order, market, 1e4, alpha_bps, exact)
    if cost_and_risk(top, order, market, alpha_bps)["cost_bps"] <= max_cost_bps:
        return {"schedule": top, "risk_aversion": 1e4, "binding": False, "feasible": True, **cost_and_risk(top, order, market, alpha_bps)}
    lam = _bisect(max_cost_bps, lambda a: cost_and_risk(optimal_schedule(order, market, a, alpha_bps, exact), order, market, alpha_bps)["cost_bps"], increasing=True)
    q = optimal_schedule(order, market, lam, alpha_bps, exact)
    return {"schedule": q, "risk_aversion": lam, "binding": True, "feasible": True, **cost_and_risk(q, order, market, alpha_bps)}


def balanced(order: Order, market: Market, risk_aversion: float = 1e-3, alpha_bps: float = 0.0, exact: bool = True) -> dict:
    """The standard cost-risk trade-off for a given risk aversion."""
    q = optimal_schedule(order, market, risk_aversion, alpha_bps, exact)
    return {"schedule": q, "risk_aversion": risk_aversion, **cost_and_risk(q, order, market, alpha_bps)}


def price_improvement(order: Order, market: Market, target_bps: float, alpha_bps: float = 0.0, exact: bool = True) -> dict:
    """The schedule that maximises the probability that the shortfall comes in below ``target_bps``, with the shortfall treated as normal: ``Phi((target - E) / std)``. Searches the frontier for the best
    ratio; returns the probability with it. If no schedule can reach the target on average, the cheapest one is returned (it also has the widest spread of outcomes, which is what gives it the best chance)."""
    from scipy.stats import norm

    best = None
    for lam in np.logspace(-8, 2, 41):
        q = optimal_schedule(order, market, float(lam), alpha_bps, exact)
        r = cost_and_risk(q, order, market, alpha_bps)
        z = (target_bps - r["cost_bps"]) / max(r["risk_bps"], 1e-9)
        if best is None or z > best[0]:
            best = (z, float(lam), q, r)
    z, lam, q, r = best
    return {"schedule": q, "risk_aversion": lam, "probability": float(norm.cdf(z)), "target_bps": target_bps, **r}


# ------------------------------------------------------------------------------------------------------------------ one-parameter families
def exponential_trade(order: Order, market: Market, kappa: float) -> np.ndarray:
    """Trade rate proportional to ``exp(-kappa t)`` over the horizon (``t`` in [0, 1]), scaled to finish exactly. ``kappa = 0`` is uniform in time (TWAP); positive front-loads; negative back-loads."""
    n = order.length(market)
    t = (np.arange(n) + 0.5) / n
    r = np.exp(-kappa * t)
    return order.shares * r / r.sum()


def exponential_residual(order: Order, market: Market, kappa: float) -> np.ndarray:
    """The shares still to trade decay like ``X exp(-kappa t)``: every interval trades the same fraction ``1 - exp(-kappa / n)`` of what is left, and the final interval sweeps the remainder
    ``X exp(-kappa (n - 1) / n)``. ``kappa >= 0``; ``kappa = 0`` waits and does everything at the end."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative for the residual schedule")
    n = order.length(market)
    x = order.shares * np.exp(-kappa * np.arange(1, n) / n)
    x = np.r_[order.shares, x, 0.0]
    return -np.diff(x)


def trade_rate(order: Order, market: Market, rate: float) -> np.ndarray:
    """Trade a fixed share ``rate`` of the market's TOTAL volume (``q = rate / (1 - rate) V`` against the volume ``V`` others are expected to trade) until the order is done. A rate below the order's own
    share of the volume leaves shares over, which the last interval takes."""
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1")
    volume = market.expected_volume()[order.window(market)]
    want = rate / (1.0 - rate) * volume
    done = np.minimum(np.cumsum(want), order.shares)
    q = np.diff(np.r_[0.0, done])
    q[-1] += order.shares - q.sum()
    return q


def _objective(schedule, order: Order, market: Market, risk_aversion: float, alpha_bps: float) -> float:
    r = cost_and_risk(schedule, order, market, alpha_bps)
    return r["cost_bps"] + risk_aversion * r["risk_bps"] ** 2


def fit_exponential_trade(order: Order, market: Market, risk_aversion: float = 1e-3, alpha_bps: float = 0.0, bounds=(-6.0, 12.0)) -> dict:
    """The ``kappa`` of :func:`exponential_trade` that minimises cost plus risk, and the schedule it gives."""
    res = minimize_scalar(lambda k: _objective(exponential_trade(order, market, k), order, market, risk_aversion, alpha_bps), bounds=bounds, method="bounded", options={"xatol": 1e-6})
    q = exponential_trade(order, market, res.x)
    return {"kappa": float(res.x), "schedule": q, **cost_and_risk(q, order, market, alpha_bps)}


def fit_exponential_residual(order: Order, market: Market, risk_aversion: float = 1e-3, alpha_bps: float = 0.0, bounds=(1e-3, 40.0)) -> dict:
    """The ``kappa`` of :func:`exponential_residual` that minimises cost plus risk, and the schedule it gives."""
    res = minimize_scalar(lambda k: _objective(exponential_residual(order, market, k), order, market, risk_aversion, alpha_bps), bounds=bounds, method="bounded", options={"xatol": 1e-6})
    q = exponential_residual(order, market, res.x)
    return {"kappa": float(res.x), "schedule": q, **cost_and_risk(q, order, market, alpha_bps)}


def fit_trade_rate(order: Order, market: Market, risk_aversion: float = 1e-3, alpha_bps: float = 0.0, bounds=(0.01, 0.6)) -> dict:
    """The participation ``rate`` of :func:`trade_rate` that minimises cost plus risk, and the schedule it gives."""
    res = minimize_scalar(lambda p: _objective(trade_rate(order, market, p), order, market, risk_aversion, alpha_bps), bounds=bounds, method="bounded", options={"xatol": 1e-7})
    q = trade_rate(order, market, res.x)
    return {"rate": float(res.x), "schedule": q, **cost_and_risk(q, order, market, alpha_bps)}
