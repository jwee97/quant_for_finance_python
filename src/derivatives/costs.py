"""Hedging an option when every trade costs money.

Black-Scholes prices an option as the cost of a continuously rebalanced hedge, which needs infinitely many trades and so, with a proportional cost on each, infinite cost. Three answers, from
the simplest to the most faithful (Leland 1985; Whalley and Wilmott 1997; Hodges and Neuberger 1989 with Davis, Panas and Zariphopoulou 1993). ``cost`` is always the proportional cost of
*one trade*, as a fraction of the value traded (half the relative bid-ask spread; Leland's round-trip ``k`` is twice it).

``leland_price``, ``leland_pde``   rebalance every ``dt``: a move of the price of size ``sigma S sqrt(dt)`` calls for a trade of ``Gamma`` times that many shares, costing ``cost S |Gamma|
                                   sigma S sqrt(2 dt / pi)`` on average, ``cost sigma S^2 |Gamma| sqrt(2 / (pi dt))`` a unit of time. That is a charge proportional to ``Gamma``, and
                                   ``Gamma`` is what the volatility term ``sigma^2 S^2 Gamma / 2`` multiplies: Black-Scholes with the volatility changed to ``sigma^2 (1 +- Le)``, ``Le = 2
                                   sqrt(2 / pi) cost / (sigma sqrt(dt))``, plus for the seller of a convex payoff and minus for the buyer. For a payoff whose gamma changes sign (a spread)
                                   the sign switches with it and the equation is nonlinear: ``leland_pde`` solves it on a grid.
``whalley_wilmott_band``           with trading at every instant and exponential utility with risk aversion ``gamma``, the best hedge keeps the stock position within ``H = (3 e^{-r(T-t)} cost
                                   S Gamma^2 / (2 gamma))^{1/3}`` of the Black-Scholes delta and trades only when it leaves that band, to its edge. The cost of hedging grows like ``cost^{2/3}``
                                   (a much slower rise than rebalancing on a clock, whose cost grows like ``cost / sqrt(dt)``).
``simulate_hedging``               the honest comparison: simulate paths, hedge a sold option by each rule (delta every step, Leland's adjusted delta, the band) and report the profit and loss,
                                   the costs paid and the number of trades.
``dpz_prices``                     the exact answer in a binomial market: the price at which an investor with exponential utility is indifferent to selling (or buying) the option, found by
                                   dynamic programming over the share holding. Without cost it is the Cox-Ross-Rubinstein price; with cost there is a bid price and an ask price, and between
                                   them the option cannot be hedged at a profit. The optimal holding has a no-trade region, which is where the Whalley-Wilmott band comes from.
"""

from __future__ import annotations

import time

import numpy as np

from ..numerics import linsolve
from . import levy
from .greeks import bsm_greeks
from .pricing import bsm_price

_C = np.sqrt(2.0 / np.pi)


# ----------------------------------------------------------------------------------------------------------------------------------- Leland
def leland_number(sigma: float, dt: float, cost: float) -> float:
    """``Le = 2 sqrt(2/pi) cost / (sigma sqrt(dt))``: the expected cost of hedging per unit of time as a fraction of the option's volatility term. Above 1 a buyer's adjusted variance is
    negative: the option cannot be replicated at a profit at that cost and frequency."""
    if sigma <= 0 or dt <= 0 or cost < 0:
        raise ValueError("sigma and dt must be positive and the cost not negative")
    return float(2.0 * _C * cost / (sigma * np.sqrt(dt)))


def leland_sigma(sigma: float, dt: float, cost: float, side: str = "sell") -> float:
    """Leland's adjusted volatility ``sigma sqrt(1 + Le)`` for the seller of a convex payoff and ``sigma sqrt(1 - Le)`` for the buyer."""
    if side not in ("sell", "buy"):
        raise ValueError("side must be 'sell' or 'buy'")
    le = leland_number(sigma, dt, cost)
    if side == "buy" and le >= 1.0:
        raise ValueError(f"the Leland number {le:.3g} is at least 1: a buyer cannot replicate the option at this cost and hedging frequency")
    return float(sigma * np.sqrt(1.0 + le if side == "sell" else 1.0 - le))


def leland_price(S: float, K: float, T: float, r: float, q: float, sigma: float, cost: float, dt: float, call: bool = True, side: str = "sell") -> float:
    """The price of a vanilla option when the hedge is rebalanced every ``dt`` and each trade costs ``cost``: Black-Scholes at Leland's adjusted volatility (a vanilla has a gamma of one
    sign, so the adjustment is constant)."""
    return float(bsm_price(S, K, T, r, q, leland_sigma(sigma, dt, cost, side), call))


def leland_pde(S: float, strikes, weights, T: float, r: float, q: float, sigma: float, cost: float, dt: float, side: str = "sell", n_space: int = 800, n_time: int = 400,
               s_max: float | None = None, nonlinear: bool = True) -> dict:
    """The value of the portfolio of calls ``Sum_i weights[i] max(S - strikes[i], 0)`` (a call spread, a butterfly, a single call) with Leland's cost, on a grid in ``S``.

    The volatility at each node is ``sigma sqrt(1 + s Le)`` with ``s`` the sign that makes it larger where the *seller* is long gamma (``s = +1`` where the portfolio's gamma is positive for
    ``side='sell'``, ``-1`` where negative; reversed for the buyer): the sign is taken from the previous step's gamma, so each step is one tridiagonal solve. With
    ``nonlinear=False`` the sign is held at ``+1`` (or ``-1`` for a buyer) everywhere: the constant-volatility price, right for a single vanilla. The scheme is Crank-Nicolson with two implicit half-steps to start. Returns the price, delta and gamma at ``S``
    and the grid."""
    strikes, weights = np.atleast_1d(np.asarray(strikes, dtype=float)), np.atleast_1d(np.asarray(weights, dtype=float))
    if len(strikes) != len(weights):
        raise ValueError("one weight per strike")
    if side not in ("sell", "buy"):
        raise ValueError("side must be 'sell' or 'buy'")
    le = leland_number(sigma, dt, cost)
    if le >= 1.0 and side == "buy":
        raise ValueError("the Leland number is at least 1: a buyer cannot replicate the option at this cost and hedging frequency")
    smax = float(s_max if s_max is not None else 4.0 * max(S, strikes.max()))
    s = np.linspace(0.0, smax, n_space + 1)
    h = s[1] - s[0]
    base = 1.0 if side == "sell" else -1.0
    dt_step = T / n_time
    i = np.arange(1, n_space)
    far_slope, far_const = float(weights.sum()), float((weights * strikes).sum())
    t0 = time.perf_counter()

    def advance(v, tau0, step, theta):
        """One theta step: the volatility at each node follows the sign of the gamma at the start of the step, so the matrix changes from step to step (one tridiagonal solve each)."""
        gamma = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / h ** 2
        sign = base * np.where(gamma >= 0.0, 1.0, -1.0) if nonlinear else np.full(n_space - 1, base)
        var = sigma ** 2 * np.maximum(1.0 + sign * le, 0.0)
        a = 0.5 * var * s[i] ** 2 / h ** 2
        b = 0.5 * (r - q) * s[i] / h
        c_lo, c_di, c_up = a - b, -2.0 * a - r, a + b                                  # the operator L: V_tau = L V
        tau1 = tau0 + step
        v_hi = far_slope * smax * np.exp(-q * tau1) - far_const * np.exp(-r * tau1)         # the far end is worth the linear asymptote, the near end (S = 0) nothing
        rhs = v[1:-1] + (1.0 - theta) * step * (c_lo * v[:-2] + c_di * v[1:-1] + c_up * v[2:])
        rhs[-1] += theta * step * c_up[-1] * v_hi
        new = np.empty_like(v)
        new[0], new[-1] = 0.0, v_hi
        new[1:-1] = linsolve.tridiagonal_solve(-theta * step * c_lo[1:], 1.0 - theta * step * c_di, -theta * step * c_up[:-1], rhs)
        return new

    v = np.maximum(s[:, None] - strikes[None, :], 0.0) @ weights
    tau = 0.0
    for k in range(n_time):
        if k < 2:                                                                       # two implicit half-steps in place of each of the first Crank-Nicolson steps (Rannacher)
            v = advance(v, tau, dt_step / 2.0, 1.0)
            v = advance(v, tau + dt_step / 2.0, dt_step / 2.0, 1.0)
        else:
            v = advance(v, tau, dt_step, 0.5)
        tau += dt_step
    d1 = np.gradient(v, h)
    d2 = np.zeros_like(v)
    d2[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / h ** 2
    return {"price": float(np.interp(S, s, v)), "delta": float(np.interp(S, s, d1)), "gamma": float(np.interp(S, s, d2)), "leland_number": le, "S": s, "values": v, "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------------------------------------------------------- Whalley-Wilmott
def whalley_wilmott_band(S, K: float, tau: float, r: float, q: float, sigma: float, cost: float, risk_aversion: float, call: bool = True):
    """The Black-Scholes delta and the half-width ``H = (3 e^{-r tau} cost S Gamma^2 / (2 risk_aversion))^{1/3}`` of the no-trade band around it, with ``tau`` the time to expiry."""
    if risk_aversion <= 0 or cost < 0:
        raise ValueError("the risk aversion must be positive and the cost not negative")
    g = bsm_greeks(S, K, tau, r, q, sigma, call)
    half = np.cbrt(1.5 * np.exp(-r * tau) * cost * np.asarray(S, dtype=float) * np.asarray(g["gamma"]) ** 2 / risk_aversion)
    return np.asarray(g["delta"]), half


# ------------------------------------------------------------------------------------------------------------------------------------ hedging
def simulate_hedging(S: float, K: float, T: float, r: float, q: float, sigma: float, cost: float, steps: int, n_paths: int = 20_000, strategy: str = "delta", call: bool = True,
                     risk_aversion: float = 1.0, mu: float | None = None, hedge_sigma: float | None = None, seed: int = 0, model: levy.Model | None = None) -> dict:
    """Sell one option at its Black-Scholes price (at ``hedge_sigma``, default ``sigma``) and hedge it at ``steps`` equally spaced dates, paying ``cost`` on every trade (including the first
    purchase and the final liquidation). The strategies:

    ``'delta'``      trade to the Black-Scholes delta at every date;
    ``'leland'``     the same with Leland's adjusted volatility in the delta (the option is also sold at Leland's price, which is how the adjustment pays for the hedge);
    ``'band'``       trade only when the holding leaves the Whalley-Wilmott band, to its nearest edge, with ``risk_aversion``;
    ``'none'``       hold nothing (the naked option, for scale).

    Prices follow ``model`` (default: geometric Brownian motion with volatility ``sigma`` and drift ``mu``, default ``r - q``) with the exact simulator, so the discrete-hedging error is the only
    approximation made. Returns the mean and spread of the discounted profit and loss, the mean cost and number of trades, and the paths' P&L."""
    if strategy not in ("delta", "leland", "band", "none"):
        raise ValueError("strategy must be 'delta', 'leland', 'band' or 'none'")
    hs = sigma if hedge_sigma is None else hedge_sigma
    dt = T / steps
    growth = (r - q) if mu is None else mu                                         # the expected growth of the price, ex dividends
    rng = np.random.default_rng(seed)
    if model is None:
        z = rng.standard_normal((n_paths, steps))
        paths = np.concatenate([np.full((n_paths, 1), S), S * np.exp(np.cumsum((growth - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z, axis=1))], axis=1)
    else:
        paths = levy.simulate_paths(model, S, growth + q, q, T, steps, n_paths, seed)
    sig_used = leland_sigma(hs, dt, cost, "sell") if strategy == "leland" else hs
    premium = float(bsm_price(S, K, T, r, q, sig_used, call))
    y = np.zeros(n_paths)
    cash = np.full(n_paths, premium)
    paid = np.zeros(n_paths)
    trades = np.zeros(n_paths)
    for i in range(steps + 1):
        tau = T - i * dt
        Si = paths[:, i]
        if i == steps:
            move = -y                                                              # liquidate the hedge at expiry
        elif strategy == "none":
            move = np.zeros(n_paths)
        else:
            g = bsm_greeks(Si, K, tau, r, q, sig_used, call)
            delta = np.asarray(g["delta"])
            if strategy == "band":
                half = np.cbrt(1.5 * np.exp(-r * tau) * cost * Si * np.asarray(g["gamma"]) ** 2 / risk_aversion)
                move = np.clip(y, delta - half, delta + half) - y                  # to the nearest edge of the band, or not at all
            else:
                move = delta - y
        fee = cost * Si * np.abs(move)
        cash = cash - Si * move - fee
        paid += fee
        trades += np.abs(move) > 1e-12
        y = y + move
        if i < steps:                                                              # one step on: the cash earns the rate and the shares pay their dividend
            cash = cash * np.exp(r * dt) + y * Si * (np.exp(q * dt) - 1.0)
    payoff = np.maximum((paths[:, -1] - K) if call else (K - paths[:, -1]), 0.0)
    pnl = (cash - payoff) * np.exp(-r * T)
    return {"strategy": strategy, "premium": premium, "mean": float(pnl.mean()), "std": float(pnl.std(ddof=1)), "se": float(pnl.std(ddof=1) / np.sqrt(n_paths)),
            "cost": float((paid * np.exp(-r * T)).mean()), "trades": float(trades.mean()), "pnl": pnl, "steps": int(steps), "n_paths": int(n_paths)}


def compare_strategies(S: float, K: float, T: float, r: float, q: float, sigma: float, cost: float, steps: int, n_paths: int = 20_000, call: bool = True, risk_aversion: float = 1.0,
                       seed: int = 0, strategies=("delta", "leland", "band")) -> list[dict]:
    """``simulate_hedging`` for each strategy on the same random numbers; the paths' P&L are dropped from the rows (the standard deviation and mean are kept)."""
    rows = []
    for st in strategies:
        res = simulate_hedging(S, K, T, r, q, sigma, cost, steps, n_paths, st, call, risk_aversion, seed=seed)
        res.pop("pnl")
        rows.append(res)
    return rows


# ----------------------------------------------------------------------------------------------------------------------------------- DPZ tree
def dpz_prices(S: float, K: float, T: float, r: float, sigma: float, cost: float, risk_aversion: float, n_steps: int = 50, call: bool = True, mu: float | None = None, n_hold: int = 201,
               hold_range: tuple[float, float] = (-1.25, 1.25), bands: bool = False) -> dict:
    """The writer's and the buyer's utility-indifference prices of a European option in a binomial market with proportional cost (Hodges and Neuberger 1989; Davis, Panas and
    Zariphopoulou 1993), no dividends.

    The investor has exponential utility ``-exp(-risk_aversion W)``, holds ``y`` shares and trades at each of ``n_steps`` dates. Because the utility is exponential the value factors as
    ``-exp(-g x e^{r (T - t)}) phi(t, y, S)`` with ``x`` the cash, and the recursion for ``ln phi`` is a minimisation over the new holding ``y'`` of the cost of the trade plus the
    log-expectation of next period's ``ln phi``. It is run three times: with no option, short one, and long one. Writer's price ``p_w = (ln phi_short - ln phi_none) / (g e^{rT})`` (at ``y = 0``)
    and buyer's ``p_b = (ln phi_none - ln phi_long) / (g e^{rT})``. ``mu`` is the drift of the stock under the investor's belief (default ``r``, which is the risk-neutral probability).
    Holdings are restricted to ``hold_range`` (wide enough for the buyer, who is short the stock, and the writer, who is long it) on ``n_hold`` points. ``bands=True`` also returns the
    no-trade band of the writer at the first date: the lowest and highest holding at which no trade is optimal, next to the frictionless (Cox-Ross-Rubinstein) delta."""
    if cost < 0 or risk_aversion <= 0 or sigma <= 0 or T <= 0 or n_steps < 1:
        raise ValueError("cost >= 0, risk aversion, sigma, T > 0 and at least one step are needed")
    t0 = time.perf_counter()
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q_rn = (np.exp(r * dt) - d) / (u - d)
    drift = r if mu is None else mu
    p = float(np.clip((np.exp(drift * dt) - d) / (u - d), 1e-9, 1.0 - 1e-9))
    lo_h, hi_h = hold_range
    y = np.linspace(lo_h, hi_h, n_hold)
    g = risk_aversion
    sign = 1.0 if call else -1.0

    def payoff(s):
        return np.maximum(sign * (s - K), 0.0)

    def terminal(s_nodes, kind):
        liq = y[None, :] * s_nodes[:, None] - cost * s_nodes[:, None] * np.abs(y)[None, :]
        pay = payoff(s_nodes)[:, None]
        if kind == "none":
            return -g * liq
        return -g * (liq - pay) if kind == "short" else -g * (liq + pay)

    def run(kind: str, want_band: bool = False):
        s_n = S * u ** np.arange(n_steps, -n_steps - 1, -2)                     # price nodes at the last date, highest first
        L = terminal(s_n, kind)
        band = None
        delta_y = y[None, :] - y[:, None]                                       # delta_y[i, j] = y_j - y_i: the trade from holding i to holding j
        for n in range(n_steps - 1, -1, -1):
            s_now = S * u ** np.arange(n, -n - 1, -2)
            up_vals, dn_vals = L[:-1], L[1:]                                    # node k at date n leads to k (up) and k + 1 (down) at date n + 1
            expect = np.logaddexp(np.log(p) + up_vals, np.log(1.0 - p) + dn_vals)       # ln E[phi(y', S')] for every node and holding y'
            scale = g * np.exp(r * (T - n * dt))
            new = np.empty_like(expect)
            for k, sk in enumerate(s_now):
                trade_cost = scale * (delta_y * sk + cost * sk * np.abs(delta_y))      # (y' - y) S + cost S |y' - y|, as a matrix over (y, y')
                total = trade_cost + expect[k][None, :]
                new[k] = total.min(axis=1)
                if want_band and n == 0 and k == 0:
                    best = total.argmin(axis=1)
                    stay = best == np.arange(len(y))
                    band = (float(y[stay].min()), float(y[stay].max())) if stay.any() else (float("nan"), float("nan"))
            L = new
        return L, band

    zero = int(np.argmin(np.abs(y)))
    ln_none, _ = run("none")
    ln_short, band = run("short", bands)
    ln_long, _ = run("long")
    scale0 = g * np.exp(r * T)
    writer = float((ln_short[0, zero] - ln_none[0, zero]) / scale0)
    buyer = float((ln_none[0, zero] - ln_long[0, zero]) / scale0)
    # the frictionless benchmark: Cox-Ross-Rubinstein on the same tree
    values = payoff(S * u ** np.arange(n_steps, -n_steps - 1, -2))
    for n in range(n_steps - 1, 0, -1):
        values = np.exp(-r * dt) * (q_rn * values[:-1] + (1.0 - q_rn) * values[1:])
    v_up, v_dn = (values[0], values[1]) if n_steps > 1 else (float(payoff(S * u)), float(payoff(S * d)))
    crr = float(np.exp(-r * dt) * (q_rn * v_up + (1.0 - q_rn) * v_dn))
    out = {"writer": writer, "buyer": buyer, "crr": crr, "spread": writer - buyer, "n_steps": n_steps, "n_hold": n_hold, "seconds": time.perf_counter() - t0}
    if bands:
        out["band"] = band
        out["crr_delta"] = float((v_up - v_dn) / (S * u - S * d))
    return out
