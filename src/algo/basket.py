"""Basket (portfolio) trading: one schedule for many stocks that weighs their trading costs against the risk of the whole unexecuted list, and three tactics for what to trade first or where.

**A basket** is a trade list: signed shares per stock (positive buys, negative sells), prices, average daily volumes, daily volatilities and a correlation matrix. While the list is being worked, the part that is
not yet executed is a *portfolio* with its own price risk: a buy of one bank against a sale of another is nearly hedged, and executing only one leg leaves a bet on banks. That is the whole reason to schedule a
basket jointly rather than stock by stock.

**The basket schedule** chooses shares ``q[k, i]`` for stock ``k`` in interval ``i`` to minimise ``sum_k cost_k + lambda * Var`` with the unexecuted exposure ``e_i = s * p * x_i`` (sign, price, shares left)
and ``Var = sum_i w_i e_i' Sigma e_i``. The cost is each stock's own impact (the quadratic model of :mod:`src.algo.optimize`, linearised at that stock's average participation); ``risk_aversion`` is in bps
units of the basket's gross value, as for one stock. At zero risk aversion every stock follows its own volume (a VWAP basket); as it grows the schedule hedges first: it executes the legs that
reduce the remaining risk most, and keeps offsetting legs moving together. ``independent_schedule`` solves each stock on its own at the same dollar risk price, which is what the joint answer is measured against.

**Three tactics on a trade list** (definitions used here; the common idea is to decide *what* to execute, not only *when*):

* ``minimum_trading_risk_quantity``: executing a given share of the list's value, which shares to execute so that what is left has the least price risk (or: the smallest share of the list that brings the
  remaining risk down to a target). Executed fractions ``theta_k`` in [0, 1] minimise ``risk(x - theta x)`` subject to the executed value.
* ``maximum_trading_opportunity``: given how much of each stock can be executed right now (``available`` shares: a dark pool's offer, a block, the participation limit), the most value that can be executed
  without the REMAINING list being riskier than a limit (default: the original list's risk). Executing only the available long legs of a hedged list can leave a naked short; this is the bound on that.
* ``program_block``: split the list into *block* names (large against their volume: they would take too much of the market on the lit exchange) and *program* names, and choose which block names can be entered
  into dark pools without raising risk: every subset of them that might fill, and nothing else, must leave a remaining list no riskier than the original (the worst case over subsets, exact up to twelve
  names). Dark fills are uncertain, so the condition is on all of them.

The simulator for a basket is :func:`simulate_basket`: correlated price moves, noisy volumes, each stock's own impact; it reproduces the closed forms the schedule is built from (``tests/test_basket.py``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .market import Market, u_shape
from .optimize import optimal_schedule, solve_qp_eq


@dataclass
class Basket:
    """A trade list. ``shares`` are signed (positive buys, negative sells); ``corr`` is the correlation of daily returns (default 0.3 between every pair)."""

    names: list
    shares: np.ndarray
    prices: np.ndarray
    adv: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray | None = None
    spread: float = 0.0004
    eta: float = 0.142
    beta: float = 0.6
    gamma: float = 0.30
    n: int = 26
    u_strength: float = 2.0
    volume_noise: float = 0.25
    day_noise: float = 0.15

    def __post_init__(self):
        self.shares, self.prices, self.adv, self.sigma = (np.asarray(a, float) for a in (self.shares, self.prices, self.adv, self.sigma))
        k = len(self.names)
        if not (len(self.shares) == len(self.prices) == len(self.adv) == len(self.sigma) == k) or k < 1:
            raise ValueError("names, shares, prices, adv and sigma must have the same length")
        if (self.shares == 0).any() or (self.prices <= 0).any() or (self.adv <= 0).any() or (self.sigma <= 0).any():
            raise ValueError("shares must be non-zero and prices, adv and sigma positive")
        self.corr = np.full((k, k), 0.3) + 0.7 * np.eye(k) if self.corr is None else np.asarray(self.corr, float)
        if self.corr.shape != (k, k) or not np.allclose(self.corr, self.corr.T) or np.linalg.eigvalsh(self.corr).min() < -1e-9:
            raise ValueError("corr must be a symmetric positive semi-definite matrix")

    # ------------------------------------------------------------------------------------------------------------------------ the list
    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def side(self) -> np.ndarray:
        return np.sign(self.shares)

    @property
    def quantity(self) -> np.ndarray:
        return np.abs(self.shares)

    @property
    def value(self) -> np.ndarray:
        """The dollar value of each trade (unsigned)."""
        return self.prices * self.quantity

    @property
    def gross(self) -> float:
        return float(self.value.sum())

    @property
    def exposure(self) -> np.ndarray:
        """The signed dollar exposure of the whole list: buys positive, sells negative (what the unexecuted part is exposed to)."""
        return self.side * self.value

    def covariance(self) -> np.ndarray:
        """The covariance of daily returns."""
        return self.corr * np.outer(self.sigma, self.sigma)

    def risk(self, fraction=None) -> float:
        """One daily standard deviation, in dollars, of the unexecuted part when ``fraction`` of each stock (default none) has been executed."""
        left = self.exposure * (1.0 - (0.0 if fraction is None else np.asarray(fraction, float)))
        return float(np.sqrt(max(left @ self.covariance() @ left, 0.0)))

    def volume(self) -> np.ndarray:
        """Expected volume of each stock in each interval: ``size x n``."""
        return np.outer(self.adv, u_shape(self.n, self.u_strength))

    def variance_weights(self) -> np.ndarray:
        return u_shape(self.n, 0.5 * self.u_strength)

    def market(self, k: int) -> Market:
        """Stock ``k`` as a single-stock market, with this basket's impact settings."""
        return Market(price=float(self.prices[k]), adv=float(self.adv[k]), sigma=float(self.sigma[k]), spread=self.spread, n=self.n, eta=self.eta, beta=self.beta, gamma=self.gamma,
                      volume_noise=self.volume_noise, day_noise=self.day_noise, u_strength=self.u_strength)


def random_basket(size: int = 8, seed: int = 0, long_short: bool = True, market_corr: float = 0.5, notional: float = 20e6) -> Basket:
    """A plausible basket for demonstrations and tests: random sizes, volumes, volatilities and a one-factor correlation. With ``long_short`` roughly half are sells, so the list is nearly market neutral."""
    rng = np.random.default_rng(seed)
    adv = rng.uniform(0.6e6, 6e6, size)
    prices = rng.uniform(20.0, 150.0, size)
    sigma = rng.uniform(0.012, 0.03, size)
    value = rng.uniform(0.5, 1.5, size)
    value = notional * value / value.sum()
    shares = np.round(value / prices)
    if long_short:
        order = rng.permutation(size)
        shares[order[: size // 2]] *= -1.0
    beta = np.sqrt(market_corr) * np.ones(size) * rng.uniform(0.8, 1.2, size)
    corr = np.outer(beta, beta)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)
    return Basket([f"S{k + 1}" for k in range(size)], shares, prices, adv, sigma, corr)


# ------------------------------------------------------------------------------------------------------------------ the joint schedule
def _basket_form(b: Basket, risk_aversion: float):
    A, n = b.size, b.n
    m = A * n
    V = b.volume()
    X = b.quantity
    rho = X / V.sum(axis=1)
    k = b.prices[:, None] * b.eta * b.sigma[:, None] * rho[:, None] ** (b.beta - 1.0) / V                  # the quadratic model: slice q of stock k in interval i costs k[k, i] q^2
    lam = float(risk_aversion) * 1e4 / b.gross
    cov, w = b.covariance(), b.variance_weights()
    sp = b.side * b.prices
    e0 = sp * X
    H_risk = np.zeros((m, m))
    g = np.zeros(m)
    for i in range(n):
        M = np.zeros((A, m))
        for a in range(A):
            M[a, a * n: a * n + i + 1] = sp[a]                                                              # the exposure given up by executing the first i + 1 slices of stock a
        H_risk += 2.0 * lam * w[i] * M.T @ cov @ M
        g -= 2.0 * lam * w[i] * M.T @ cov @ e0
    Aeq = np.zeros((A, m))
    for a in range(A):
        Aeq[a, a * n: (a + 1) * n] = 1.0
    c_exact = (b.prices[:, None] * b.eta * b.sigma[:, None] * V ** (-b.beta)).reshape(m)                      # the market's own power law: slice q of stock k in interval i costs c q^(1 + beta)
    return 2.0 * np.diag(k.reshape(m)) + H_risk, g, Aeq, X, H_risk, c_exact


def basket_schedule(b: Basket, risk_aversion: float = 1e-3, exact: bool = False) -> np.ndarray:
    """The joint schedule: shares to trade in each interval for each stock, as a ``size x n`` array of non-negative numbers (the side is the list's). It solves the quadratic model exactly; ``exact=True``
    then polishes it with each market's own power-law impact (slower, and the cost grows quickly with the size: about 2 seconds for 16 stocks, 13 for 20 and 40 for 30, on one thread)."""
    H, g, A, X, H_risk, c = _basket_form(b, risk_aversion)
    q = solve_qp_eq(H, g, A, X)
    if exact and abs(b.beta - 1.0) > 1e-9:
        m, e = len(q), 1.0 + b.beta
        scale = np.repeat(X, b.n)                                                                          # work in fractions of each stock's order so the optimiser sees numbers near one
        cost0 = max(float(c @ np.maximum(q, 0.0) ** e), 1.0)
        f = lambda z: (c @ np.maximum(z * scale, 0.0) ** e + 0.5 * (z * scale) @ H_risk @ (z * scale) + g @ (z * scale)) / cost0
        grad = lambda z: (e * c * np.maximum(z * scale, 0.0) ** b.beta + H_risk @ (z * scale) + g) * scale / cost0
        res = minimize(f, q / scale, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * m, constraints=[{"type": "eq", "fun": lambda z: (A * scale) @ z - X, "jac": lambda z: A * scale}],
                       options={"maxiter": 200, "ftol": 1e-13})
        z = np.clip(res.x, 0.0, 1.0)
        polished = (z * scale).reshape(b.size, b.n)
        if f(res.x) <= f(q / scale) + 1e-12:
            q = (polished * (X / polished.sum(axis=1))[:, None]).reshape(m)
    return q.reshape(b.size, b.n)


def independent_schedule(b: Basket, risk_aversion: float = 1e-3, exact: bool = False) -> np.ndarray:
    """Each stock scheduled on its own, ignoring the other stocks, at the same dollar price of variance as the joint schedule (so the two answer the same question)."""
    from .market import Order

    lam = float(risk_aversion) * 1e4 / b.gross
    out = np.zeros((b.size, b.n))
    for k in range(b.size):
        order = Order(int(b.side[k]), float(b.quantity[k]))
        ra_k = lam * order.reference(b.market(k)) * order.shares / 1e4
        out[k] = optimal_schedule(order, b.market(k), ra_k, exact=exact)
    return out


def basket_cost_risk(b: Basket, schedule: np.ndarray) -> dict:
    """Expected cost (power-law temporary impact, permanent impact and half the spread, summed over stocks) and the standard deviation of the shortfall that price moves add, in dollars and in bps of the
    basket's gross value."""
    Q = np.asarray(schedule, float)
    V = b.volume()
    temp = float((b.prices[:, None] * Q * b.eta * b.sigma[:, None] * (np.maximum(Q, 0.0) / V) ** b.beta).sum())
    perm = float((0.5 * b.prices * b.gamma * b.sigma * Q.sum(axis=1) ** 2 / b.adv).sum())
    spread = float(0.5 * b.spread * (b.prices * Q.sum(axis=1)).sum())
    x = b.quantity[:, None] - np.cumsum(Q, axis=1)                                                            # shares left after each interval
    e = (b.side * b.prices)[:, None] * x
    cov, w = b.covariance(), b.variance_weights()
    var = float(sum(w[i] * e[:, i] @ cov @ e[:, i] for i in range(b.n)))
    total = temp + perm + spread
    G = b.gross
    return {"temporary": temp, "permanent": perm, "spread": spread, "cost": total, "cost_bps": 1e4 * total / G, "std": float(np.sqrt(var)), "risk_bps": 1e4 * float(np.sqrt(var)) / G}


def simulate_basket(b: Basket, schedule: np.ndarray, paths: int = 500, seed: int = 0, noise: bool = True, drift_bps: np.ndarray | None = None) -> dict:
    """Work the basket through ``paths`` simulated days (every slice marketable): correlated price moves, noisy volumes, each stock's own impact. Returns the shortfall per path in bps of gross value and its
    parts. With ``noise=False`` the volumes are exactly their profile, and the mean and variance match :func:`basket_cost_risk`."""
    rng = np.random.default_rng(seed)
    Q = np.asarray(schedule, float)
    A, n = b.size, b.n
    L = np.linalg.cholesky(b.corr + 1e-12 * np.eye(A))
    z = rng.standard_normal((paths, n, A)) @ L.T
    vol_noise = (np.exp(b.volume_noise * rng.standard_normal((paths, n, A)) - 0.5 * b.volume_noise ** 2) * np.exp(b.day_noise * rng.standard_normal((paths, 1, A)) - 0.5 * b.day_noise ** 2)
                 if noise else np.ones((paths, n, A)))
    w = b.variance_weights()
    V = b.volume().T                                                                                          # n x A
    side = b.side
    exo = np.zeros((paths, A))                                                                                # the price move without us, as a fraction of the arrival price
    own = np.zeros((paths, A))
    parts = {k: np.zeros(paths) for k in ("timing", "spread", "temporary", "permanent")}
    drift = np.zeros(A) if drift_bps is None else np.asarray(drift_bps, float) * 1e-4 / n
    for i in range(n):
        q = Q[:, i][None, :] * np.ones((paths, 1))
        Vi = V[i][None, :] * vol_noise[:, i, :]
        temp = b.eta * b.sigma * (q / Vi) ** b.beta
        perm = b.gamma * b.sigma * q / b.adv
        weight = (b.prices * q)
        parts["timing"] += (weight * side * exo).sum(axis=1)
        parts["spread"] += (weight * 0.5 * b.spread).sum(axis=1)
        parts["temporary"] += (weight * temp).sum(axis=1)
        parts["permanent"] += (weight * (own + 0.5 * perm)).sum(axis=1)
        exo = exo + b.sigma * np.sqrt(w[i]) * z[:, i, :] + drift
        own = own + perm
    G = b.gross
    out = {k: v / G * 1e4 for k, v in parts.items()}
    shortfall = sum(out.values())
    return {"shortfall_bps": shortfall, "parts_bps": out, "mean_bps": float(shortfall.mean()), "std_bps": float(shortfall.std(ddof=1))}


# ------------------------------------------------------------------------------------------------------------------ what to trade first
def _min_residual_risk(b: Basket, share: float):
    """Minimise the residual risk when ``share`` of the list's value has been executed, over executed fractions in [0, 1]. Returns ``(theta, risk)``."""
    cov = b.covariance()
    e0, value = b.exposure, b.value
    G = b.gross
    A = b.size
    f = lambda phi: float((e0 * phi) @ cov @ (e0 * phi)) / G ** 2                                             # phi = what is left of each stock
    grad = lambda phi: 2.0 * e0 * (cov @ (e0 * phi)) / G ** 2
    res = minimize(f, np.full(A, 1.0 - share), jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * A,
                   constraints=[{"type": "eq", "fun": lambda phi: value @ phi - (1.0 - share) * G, "jac": lambda phi: value}], options={"maxiter": 500, "ftol": 1e-15})
    phi = np.clip(res.x, 0.0, 1.0)
    return 1.0 - phi, float(np.sqrt(max(f(phi), 0.0)) * G)


def minimum_trading_risk_quantity(b: Basket, share: float | None = None, risk_target: float | None = None) -> dict:
    """Which shares to execute so that what is left carries the least price risk.

    Give ``share`` (the fraction of the list's value to execute, in [0, 1]): returns the executed fractions per stock that minimise the residual risk, and that risk. Give ``risk_target`` (dollars of daily
    standard deviation) instead: returns the smallest share whose best residual risk is at most the target. ``naive_risk`` is the residual risk of executing every stock in proportion."""
    if (share is None) == (risk_target is None):
        raise ValueError("give exactly one of share and risk_target")
    if share is None:
        if risk_target < 0:
            raise ValueError("risk_target must be non-negative")
        if b.risk() <= risk_target:
            share = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _min_residual_risk(b, mid)[1] > risk_target:
                    lo = mid
                else:
                    hi = mid
            share = hi
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be between 0 and 1")
    theta, risk = _min_residual_risk(b, share)
    return {"share": float(share), "executed_fraction": theta, "executed_shares": theta * b.quantity * b.side, "residual_risk": risk, "naive_risk": b.risk(np.full(b.size, share)),
            "original_risk": b.risk(), "risk_reduction": 1.0 - risk / b.risk() if b.risk() > 0 else 0.0}


def maximum_trading_opportunity(b: Basket, available=None, risk_limit: float | None = None) -> dict:
    """The most value that can be executed now without the remaining list being riskier than ``risk_limit`` (default: the original list's risk).

    ``available`` is how many shares of each stock can be executed now (default: all of them). Without the risk limit that would be everything available; the limit stops a hedged list from being worked
    leg by leg into a naked position. If no execution of what is available can bring the remaining risk down to a limit below the original's (``feasible`` is false), the lowest-risk execution is returned."""
    avail = b.quantity if available is None else np.minimum(np.asarray(available, float), b.quantity)
    if (avail < 0).any():
        raise ValueError("available must be non-negative")
    limit = b.risk() if risk_limit is None else float(risk_limit)
    cov, e0, value, G = b.covariance(), b.exposure, b.value, b.gross
    u = avail / b.quantity
    f = lambda th: -float(value @ th) / G
    grad = lambda th: -value / G
    cons = [{"type": "ineq", "fun": lambda th: (limit ** 2 - ((e0 * (1 - th)) @ cov @ (e0 * (1 - th)))) / G ** 2,
             "jac": lambda th: 2.0 * e0 * (cov @ (e0 * (1 - th))) / G ** 2}]
    best = None
    for start in (np.zeros(b.size), u * 0.5, np.minimum(u, 1.0)):
        res = minimize(f, start, jac=grad, method="SLSQP", bounds=[(0.0, float(x)) for x in u], constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
        th = np.clip(res.x, 0.0, u)
        risk = float(np.sqrt(max((e0 * (1 - th)) @ cov @ (e0 * (1 - th)), 0.0)))
        if risk <= limit * (1 + 1e-6) + 1e-9 and (best is None or value @ th > value @ best[0]):
            best = (th, risk)
    feasible = best is not None
    if best is None:                                                                                           # the limit cannot be met: the least risky execution of what is on offer
        res = minimize(lambda th: float((e0 * (1 - th)) @ cov @ (e0 * (1 - th))) / G ** 2, 0.5 * u, jac=lambda th: -2.0 * e0 * (cov @ (e0 * (1 - th))) / G ** 2, method="SLSQP",
                       bounds=[(0.0, float(x)) for x in u], options={"maxiter": 500, "ftol": 1e-15})
        th = np.clip(res.x, 0.0, u)
        best = (th, float(np.sqrt(max((e0 * (1 - th)) @ cov @ (e0 * (1 - th)), 0.0))))
    th, risk = best
    return {"executed_fraction": th, "executed_shares": th * b.quantity * b.side, "executed_value": float(value @ th), "share_of_list": float(value @ th) / G, "available_value": float(value @ u) / 1.0,
            "residual_risk": risk, "risk_limit": limit, "original_risk": b.risk(), "feasible": bool(feasible), "binding": bool(value @ th < value @ u - 1e-6 * G)}


def _worst_residual_risk(b: Basket, subset: tuple, rng=None, limit: int = 12) -> tuple[float, bool]:
    """The largest residual risk over every way the names in ``subset`` might fill (each fully or not at all): exact up to ``limit`` names, else the best of many random fill patterns."""
    cov, e0 = b.covariance(), b.exposure
    idx = list(subset)
    if not idx:
        return b.risk(), True
    def risk_of(fill):
        theta = np.zeros(b.size)
        theta[idx] = fill
        left = e0 * (1.0 - theta)
        return float(np.sqrt(max(left @ cov @ left, 0.0)))
    if len(idx) <= limit:
        return max(risk_of(np.array(p, float)) for p in itertools.product((0.0, 1.0), repeat=len(idx))), True
    rng = rng or np.random.default_rng(0)
    return max(risk_of(rng.integers(0, 2, len(idx)).astype(float)) for _ in range(4000)), False


def program_block(b: Basket, block_threshold: float = 0.05, risk_cap: float = 1.0, eligible=None, max_dark: int = 12) -> dict:
    """Split a basket into BLOCK names (trade at least ``block_threshold`` of a day's volume: too big for the lit market) and PROGRAM names (the rest), and pick the block names (or the ``eligible``
    names you give) that may be entered into dark pools without raising risk.

    A dark order may fill fully or not at all, name by name, so a set ``D`` is safe when EVERY subset ``S`` of it that fills leaves a remaining list with risk at most ``risk_cap`` times the original
    (the case that fills nothing is the original; the case that fills everything is among the subsets). Names are added greedily, biggest first, while the set stays safe; the worst case is exact up to
    twelve names in the set and sampled beyond that (``exact`` says which)."""
    size = b.quantity / b.adv
    block = [b.names[k] for k in range(b.size) if size[k] >= block_threshold]
    program = [b.names[k] for k in range(b.size) if size[k] < block_threshold]
    candidates = [k for k in range(b.size) if (b.names[k] in set(eligible) if eligible is not None else size[k] >= block_threshold)]
    candidates.sort(key=lambda k: -b.value[k])
    base = b.risk()
    limit = risk_cap * base
    chosen: list[int] = []
    exact = True
    for k in candidates:
        if len(chosen) >= max_dark:
            break
        worst, ok = _worst_residual_risk(b, tuple(chosen + [k]))
        if worst <= limit + 1e-9 * max(base, 1.0):
            chosen.append(k)
            exact = exact and ok
    worst, ok = _worst_residual_risk(b, tuple(chosen))
    dark_value = float(b.value[chosen].sum()) if chosen else 0.0
    return {"block": block, "program": program, "dark": [b.names[k] for k in chosen], "lit": [b.names[k] for k in range(b.size) if k not in chosen], "dark_value": dark_value,
            "dark_share": dark_value / b.gross, "original_risk": base, "worst_case_risk": worst, "risk_if_all_fill": float(b.risk(np.isin(np.arange(b.size), chosen).astype(float))),
            "exact": bool(exact and ok), "candidates": [b.names[k] for k in candidates]}
