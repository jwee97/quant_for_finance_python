"""Constrained long-short portfolio construction: 130/30, market neutral, sector and beta neutral, with turnover limits and trading costs, as one quadratic programme.

The book is ``w = w_long - w_short`` with both parts non-negative, which turns the things a portfolio manager actually specifies into linear constraints on a quadratic objective::

    maximise   alpha'w  -  (risk_aversion / 2) w'Sigma w  -  cost'|w - w0|  -  tiny * (sum w_long + sum w_short)
    subject to sum(w_long) = gross_long          sum(w_short) = gross_short            (130/30: 1.3 and 0.3; long-only: 1 and 0; dollar neutral: 1 and 1)
               each name at most max_long long and max_short short                       (position limits)
               |net weight of each group - its target| <= group_tolerance                (sector neutral)
               |beta'w - target| <= beta_tolerance                                       (beta neutral)
               |loadings'w| inside the bounds                                            (any other exposure: size, momentum, ...)
               sum|w - w0| <= max_turnover                                               (turnover)

Splitting the book is exact because holding a name long and short at once only wastes gross and cost; the tiny linear penalty makes the optimum unique, so no name is on both sides. The turnover and
trading-cost terms use one extra variable per name, ``t >= |w - w0|``. The programme is solved with :func:`src.equity.qp.solve_qp`; the answer reports its status and the exposures it ended with, and
an infeasible specification is reported, never returned as a portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .qp import solve_qp


@dataclass
class BookSpec:
    """What the book must satisfy. ``gross_long`` and ``gross_short`` are exact totals when ``exact`` and upper limits otherwise."""

    gross_long: float = 1.0
    gross_short: float = 0.0
    exact: bool = True                          # the gross exposures are targets (True) or limits with the net exposure fixed (False)
    net_exposure: float | None = None           # the net exposure of a book with limits (default: gross_long - gross_short)
    max_long: float | np.ndarray = 0.10
    max_short: float | np.ndarray = 0.10
    groups: dict | None = None                  # asset -> group, for group neutrality
    group_targets: dict | None = None           # group -> target net weight (default: the group's share of the names, times the net exposure)
    group_tolerance: float | None = None        # None: groups are not constrained
    beta: np.ndarray | None = None              # betas of the assets
    beta_target: float = 0.0
    beta_tolerance: float | None = None
    loadings: np.ndarray | None = None          # k x n exposures to anything else
    loading_bounds: np.ndarray | None = None    # k x 2
    max_turnover: float | None = None
    trade_cost: float | np.ndarray = 0.0        # proportional cost of trading one unit of weight
    risk_aversion: float = 5.0
    penalty: float = 1e-3                       # a tiny cost of gross exposure, as a fraction of the average |alpha|: makes the optimum unique, so nothing is long and short at once

    @property
    def net(self) -> float:
        return self.gross_long - self.gross_short if self.net_exposure is None else self.net_exposure


@dataclass
class BookResult:
    weights: pd.Series
    status: str
    ok: bool
    objective: float = np.nan
    exposures: dict = field(default_factory=dict)
    iterations: int = 0


def _broadcast(value, n: int, name: str) -> np.ndarray:
    v = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    if (v < 0).any():
        raise ValueError(f"{name} must not be negative")
    return v


def optimise_book(alpha, cov, spec: BookSpec, held=None, names=None, max_iter: int = 40000) -> BookResult:
    """The best book under ``spec`` for expected returns ``alpha`` and covariance ``cov`` (both for the same period), starting from ``held`` (needed for turnover and trading costs)."""
    alpha = np.asarray(alpha, dtype=float)
    n = len(alpha)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (n, n) or not np.isfinite(cov).all() or not np.isfinite(alpha).all():
        raise ValueError("alpha and cov must be finite and conformable")
    names = list(names) if names is not None else [f"x{i}" for i in range(n)]
    if spec.risk_aversion <= 0 or spec.gross_long < 0 or spec.gross_short < 0:
        raise ValueError("risk_aversion > 0 and gross exposures >= 0")
    max_long, max_short = _broadcast(spec.max_long, n, "max_long"), _broadcast(spec.max_short, n, "max_short")
    if spec.gross_short == 0:
        max_short = np.zeros(n)
    held_w = np.zeros(n) if held is None else np.asarray(held, dtype=float)
    cost = _broadcast(spec.trade_cost, n, "trade_cost")
    use_trade = spec.max_turnover is not None or bool(np.any(cost > 0))
    m = 3 * n if use_trade else 2 * n

    S = 0.5 * (cov + cov.T)
    P = np.zeros((m, m))
    P[:n, :n], P[:n, n:2 * n], P[n:2 * n, :n], P[n:2 * n, n:2 * n] = S, -S, -S, S
    P *= spec.risk_aversion
    penalty = spec.penalty * max(float(np.abs(alpha).mean()), 1e-12)
    q = np.concatenate([-alpha + penalty, alpha + penalty, cost if use_trade else np.zeros(0)])

    rows, lo, hi = [], [], []

    def add(row, low, high):
        rows.append(row)
        lo.append(low)
        hi.append(high)

    def split(vec):
        """The row of A that applies a vector to the net weight w = w_long - w_short."""
        out = np.zeros(m)
        out[:n], out[n:2 * n] = vec, -vec
        return out

    long_row, short_row = np.zeros(m), np.zeros(m)
    long_row[:n], short_row[n:2 * n] = 1.0, 1.0
    if spec.exact:
        add(long_row, spec.gross_long, spec.gross_long)
        if spec.gross_short > 0:
            add(short_row, spec.gross_short, spec.gross_short)
    else:                                                                                  # limits on each side and the net exposure fixed
        add(long_row, 0.0, spec.gross_long)
        if spec.gross_short > 0:
            add(short_row, 0.0, spec.gross_short)
        add(split(np.ones(n)), spec.net, spec.net)
    if spec.groups and spec.group_tolerance is not None:
        members = pd.Series(spec.groups).reindex(names)
        for g in sorted({v for v in members.dropna().unique()}):
            idx = (members == g).to_numpy()
            target = (spec.group_targets or {}).get(g, spec.net * idx.mean())
            add(split(idx.astype(float)), target - spec.group_tolerance, target + spec.group_tolerance)
    if spec.beta is not None and spec.beta_tolerance is not None:
        add(split(np.asarray(spec.beta, dtype=float)), spec.beta_target - spec.beta_tolerance, spec.beta_target + spec.beta_tolerance)
    if spec.loadings is not None:
        L = np.atleast_2d(np.asarray(spec.loadings, dtype=float))
        B = np.atleast_2d(np.asarray(spec.loading_bounds, dtype=float))
        for k in range(L.shape[0]):
            add(split(L[k]), B[k, 0], B[k, 1])
    if use_trade:
        for i in range(n):                                                                    # t_i >= w_i - h_i  and  t_i >= h_i - w_i
            up, down = np.zeros(m), np.zeros(m)
            up[i], up[n + i], up[2 * n + i] = -1.0, 1.0, 1.0                                    # t_i - w_i >= -h_i
            down[i], down[n + i], down[2 * n + i] = 1.0, -1.0, 1.0                              # t_i + w_i >= h_i
            add(up, -held_w[i], np.inf)
            add(down, held_w[i], np.inf)
        if spec.max_turnover is not None:
            turnover = np.zeros(m)
            turnover[2 * n:] = 1.0
            add(turnover, 0.0, float(spec.max_turnover))
    A, l, u = np.array(rows), np.array(lo), np.array(hi)
    lb = np.zeros(m)
    if (max_long.sum() < (spec.gross_long if spec.exact else spec.net) - 1e-9) or (spec.exact and spec.gross_short > 0 and max_short.sum() < spec.gross_short - 1e-9):
        return BookResult(pd.Series(0.0, index=names), "infeasible: position limits cannot reach the gross exposure", False)

    def solve(cap_long, cap_short):
        ub = np.concatenate([cap_long, cap_short, np.full(n, np.inf) if use_trade else np.zeros(0)])
        return solve_qp(P, q, A=A, l=l, u=u, lb=lb, ub=ub, max_iter=max_iter)

    r = solve(max_long, max_short)
    if not r.ok:
        return BookResult(pd.Series(0.0, index=names), r.status, False, iterations=r.iterations)
    status, iterations = r.status, r.iterations
    overlap = np.minimum(r.x[:n], r.x[n:2 * n])
    if spec.exact and spec.gross_short > 0 and overlap.sum() > 1e-6:
        # holding a name long and short at once satisfies a gross target on paper and not in fact: give every name one side (the side it was net on, or its alpha's side) and solve again
        net_side = r.x[:n] - r.x[n:2 * n]
        long_side = np.where(np.abs(net_side) > 1e-7, net_side > 0, alpha >= np.median(alpha))
        second = solve(np.where(long_side, max_long, 0.0), np.where(long_side, 0.0, max_short))
        iterations += second.iterations
        if second.ok:
            r, status = second, "solved (one side per name)"
        else:
            status = "solved (gross exposures only on paper: no one-sided book reaches them)"
    w = r.x[:n] - r.x[n:2 * n]
    w = np.where(np.abs(w) < 1e-7, 0.0, w)
    exposures = {"gross_long": float(w[w > 0].sum()), "gross_short": float(-w[w < 0].sum()), "net": float(w.sum()), "turnover": float(np.abs(w - held_w).sum())}
    if spec.beta is not None:
        exposures["beta"] = float(np.asarray(spec.beta, dtype=float) @ w)
    if spec.groups:
        members = pd.Series(spec.groups).reindex(names)
        exposures["group_net"] = {g: float(w[(members == g).to_numpy()].sum()) for g in sorted({v for v in members.dropna().unique()})}
    if spec.loadings is not None:
        exposures["loadings"] = (np.atleast_2d(spec.loadings) @ w).tolist()
    return BookResult(pd.Series(w, index=names), status, True, float(r.objective), exposures, iterations)


# ------------------------------------------------------------------------------------------------------------------ the named books
PRESETS = {
    "long_only": dict(gross_long=1.0, gross_short=0.0),
    "130_30": dict(gross_long=1.3, gross_short=0.3),
    "120_20": dict(gross_long=1.2, gross_short=0.2),
    "market_neutral": dict(gross_long=1.0, gross_short=1.0),
    "dollar_neutral": dict(gross_long=0.5, gross_short=0.5),
}


def preset(name: str, **overrides) -> BookSpec:
    """A named book: ``long_only`` (fully invested), ``130_30``, ``120_20``, ``market_neutral`` (100% long, 100% short) or ``dollar_neutral`` (50% each side), with any field overridden."""
    if name not in PRESETS:
        raise KeyError(f"unknown book '{name}'; known: {', '.join(PRESETS)}")
    return BookSpec(**{**PRESETS[name], **overrides})
