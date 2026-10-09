"""Execution with a mix of limit and market orders: how much of an order to work passively and how much to take now, found by dynamic programming.

A market order is certain and costs the spread plus impact. A limit order posted at the bid earns the half-spread if it fills and costs nothing if it does not, but it may not fill, and while the order waits
the price may run away. The trade-off is a stopping problem, and it is solved exactly here for a buyer of ``shares`` over ``intervals`` intervals. In each interval, with ``q`` of the ``units`` left, choose

* ``m`` units to buy now by market order, at ``half_spread + impact * m / q_total`` basis points a share, and
* ``s`` of the rest to post as a limit order at the bid, which fills ``f`` units with ``P(f >= j) = fill_first * fill_decay^(j - 1)`` for ``j <= s`` (a bigger order is less likely to fill all the way; the
  first unit is filled with probability ``fill_first``, which reflects the queue ahead of you), each filled share then costing ``adverse_selection - half_spread`` basis points (it earned the spread, and
  fills tend to come just before the price moves against you).

What is left after the interval costs ``drift`` basis points a share more for every interval it waits (the price is expected to run away: the urgency) and a risk charge of ``risk`` basis points a share
per unit of the order outstanding; at the end the remainder is bought by market order. ``V_t(q) = min over (m, s) of  m cost(m) + E[ f (adverse - half_spread) + V_(t+1)(q - m - f) + holding(q - m - f) ]``
is computed backwards for every ``(t, q)``. The result is a table of what to do in each state, which :func:`simulate` plays out against random fills, and which is compared with the two simple
rules: buy it all now, or post limit orders every interval and take what is left at the end. The model is stylised (one price level, no queue dynamics, fills independent across intervals); it shows
which way each force pushes, not what an order on a particular stock will cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LimitOrderModel:
    shares: float = 100_000.0
    intervals: int = 10
    half_spread_bps: float = 3.0
    impact_bps: float = 8.0                 # extra cost per share of buying the *whole* order by market order in one interval; proportional to the share bought that way
    drift_bps: float = 0.0                  # the adverse price move per interval that an unexecuted share costs
    adverse_selection_bps: float = 1.0      # expected loss per share filled by a limit order
    fill_first: float = 0.5                 # probability that a limit order of one unit fills at all in an interval
    fill_decay: float = 0.97                # P(fill >= j units) = fill_first * decay^(j-1)
    risk_bps: float = 0.0                   # risk charge per interval for the whole order outstanding (scales with the square of the share outstanding)
    units: int = 50                         # resolution of the order

    def __post_init__(self):
        if self.shares <= 0 or self.intervals < 1 or self.units < 2:
            raise ValueError("shares > 0, intervals >= 1, units >= 2")
        if min(self.half_spread_bps, self.impact_bps, self.adverse_selection_bps, self.risk_bps) < 0 or not 0.0 <= self.fill_first <= 1.0 or not 0.0 < self.fill_decay <= 1.0:
            raise ValueError("costs >= 0, 0 <= fill_first <= 1, 0 < fill_decay <= 1")

    @property
    def unit_shares(self) -> float:
        return self.shares / self.units

    def market_cost(self, m):
        """Cost in basis-point-shares of buying ``m`` units by market order in one interval."""
        u = np.asarray(m, dtype=float) * self.unit_shares
        return u * (self.half_spread_bps + self.impact_bps * u / self.shares)

    def holding_cost(self, r):
        """Cost in basis-point-shares of leaving ``r`` units outstanding for one interval (drift and the risk charge)."""
        u = np.asarray(r, dtype=float) * self.unit_shares
        return self.drift_bps * u + self.risk_bps * u * u / self.shares

    def fill_pmf(self, s: int) -> np.ndarray:
        """The probability that a limit order of ``s`` units fills 0, 1, ... s units."""
        pmf = np.zeros(s + 1)
        if s == 0:
            pmf[0] = 1.0
            return pmf
        tail = self.fill_first * self.fill_decay ** np.arange(s)                      # P(f >= j) for j = 1 ... s
        pmf[0] = 1.0 - tail[0]
        pmf[1:s] = tail[:-1] - tail[1:]
        pmf[s] = tail[-1]
        return pmf


@dataclass
class Policy:
    market: np.ndarray                      # (intervals, units + 1): units to buy by market order in interval t with q outstanding
    limit: np.ndarray                       # (intervals, units + 1): units to post as a limit order


@dataclass
class LimitOrderSolution:
    policy: Policy
    value: np.ndarray                       # (intervals + 1, units + 1) the expected cost in basis-point-shares from (t, q)
    cost_bps: float                         # the expected cost per share of the whole order, in basis points, objective included (drift and risk)
    model: LimitOrderModel


def _continuation(model: LimitOrderModel, V_next: np.ndarray) -> np.ndarray:
    """``W[r, s]``: the expected cost after the interval of posting ``s`` units when ``r`` units are left after any market order (``r >= s``)."""
    N = model.units
    fill_cost = model.unit_shares * (model.adverse_selection_bps - model.half_spread_bps)
    after = V_next + model.holding_cost(np.arange(N + 1))
    W = np.full((N + 1, N + 1), np.inf)
    for s in range(N + 1):
        pmf = model.fill_pmf(s)
        f = np.arange(s + 1)
        for r in range(s, N + 1):
            W[r, s] = pmf @ (f * fill_cost + after[r - f])
    return W


def solve(model: LimitOrderModel) -> LimitOrderSolution:
    """Backward induction for the best mix of market and limit orders in every state."""
    N, T = model.units, model.intervals
    q = np.arange(N + 1)
    V = np.zeros((T + 1, N + 1))
    V[T] = model.market_cost(q)                                                         # whatever is left at the end is bought at the market
    market = np.zeros((T, N + 1), dtype=int)
    limit = np.zeros((T, N + 1), dtype=int)
    for t in range(T - 1, -1, -1):
        W = _continuation(model, V[t + 1])
        best_s = W.argmin(axis=1)                                                        # for r units left after the market order, the best limit size (inf entries lose)
        best_w = W[np.arange(N + 1), best_s]
        for qq in range(N + 1):
            m = np.arange(qq + 1)
            total = model.market_cost(m) + best_w[qq - m]
            k = int(np.argmin(total))
            V[t, qq] = total[k]
            market[t, qq] = k
            limit[t, qq] = best_s[qq - k]
    return LimitOrderSolution(Policy(market, limit), V, float(V[0, N] / model.shares), model)


def evaluate(model: LimitOrderModel, policy: Policy) -> float:
    """The exact expected cost per share (basis points, drift and risk included) of a policy, by backward evaluation, with no simulation."""
    N, T = model.units, model.intervals
    V = model.market_cost(np.arange(N + 1))
    fill_cost = model.unit_shares * (model.adverse_selection_bps - model.half_spread_bps)
    for t in range(T - 1, -1, -1):
        V_new = np.zeros(N + 1)
        for q in range(N + 1):
            m = min(int(policy.market[t, q]), q)
            s = min(int(policy.limit[t, q]), q - m)
            pmf = model.fill_pmf(s)
            r = q - m - np.arange(s + 1)
            V_new[q] = model.market_cost(m) + pmf @ (np.arange(s + 1) * fill_cost + V[r] + model.holding_cost(r))
        V = V_new
    return float(V[N] / model.shares)


def all_market_policy(model: LimitOrderModel) -> Policy:
    """Buy everything at the market in the first interval."""
    N, T = model.units, model.intervals
    market = np.zeros((T, N + 1), dtype=int)
    market[0] = np.arange(N + 1)
    return Policy(market, np.zeros((T, N + 1), dtype=int))


def twap_market_policy(model: LimitOrderModel) -> Policy:
    """Equal market slices in every interval."""
    N, T = model.units, model.intervals
    market = np.zeros((T, N + 1), dtype=int)
    for t in range(T):
        market[t] = np.ceil(np.arange(N + 1) / (T - t)).astype(int)
    return Policy(market, np.zeros((T, N + 1), dtype=int))


def limit_then_market_policy(model: LimitOrderModel) -> Policy:
    """Post everything outstanding as a limit order in every interval and take what is left at the market at the end."""
    N, T = model.units, model.intervals
    return Policy(np.zeros((T, N + 1), dtype=int), np.tile(np.arange(N + 1), (T, 1)))


def simulate(model: LimitOrderModel, policy: Policy, paths: int = 2000, seed: int = 0) -> dict:
    """Play a policy out against random fills. Returns the mean cost per share in basis points (objective: drift and risk charge included), its standard error and spread across paths, and the share of the
    order bought by limit order."""
    rng = np.random.default_rng(seed)
    N, T = model.units, model.intervals
    fill_cost = model.unit_shares * (model.adverse_selection_bps - model.half_spread_bps)
    q = np.full(paths, N)
    cost = np.zeros(paths)
    limit_units = np.zeros(paths)
    for t in range(T):
        m = np.minimum(policy.market[t, q], q)
        s = np.minimum(policy.limit[t, q], q - m)
        cost += model.market_cost(m)
        U = rng.random(paths)
        with np.errstate(divide="ignore", invalid="ignore"):
            first = np.where(U < model.fill_first, 1 + np.floor(np.log(np.clip(U / max(model.fill_first, 1e-300), 1e-300, 1.0)) / np.log(model.fill_decay)) if model.fill_decay < 1.0 else np.inf, 0)
        f = np.minimum(s, np.nan_to_num(first, posinf=1e9))                              # P(f >= j) = fill_first * decay^(j-1): invert the tail with one uniform
        f = np.where(s > 0, f, 0).astype(int)
        cost += f * fill_cost
        limit_units += f
        q = q - m - f
        cost += model.holding_cost(q)
    cost += model.market_cost(q)                                                         # the cleanup at the end (its holding cost was charged above and is not repeated)
    per_share = cost / model.shares
    return {"cost_bps": float(per_share.mean()), "se_bps": float(per_share.std(ddof=1) / np.sqrt(paths)), "std_bps": float(per_share.std(ddof=1)), "limit_share": float(limit_units.mean() / N)}
