"""Discount and projection curves: construction, interpolation, forwards, bumps and bootstrapping from par swap rates.

A :class:`DiscountCurve` is a set of zero rates (continuously compounded, ACT/365F) at tenor nodes, valid on a valuation date. Between nodes the discount factor is
log-linear (``log_linear_df``, the default: flat forward rates between nodes) or the zero rate is linear (``linear_zero``); beyond the last node the last forward is extended. Time
is measured ``(date - valuation).days / 365``, so ``df_date`` and ``df`` agree.

* ``forward_rate(t1, t2)``: the simple forward rate over ``[t1, t2]`` for an accrual fraction ``tau`` (``(DF1 / DF2 - 1) / tau``), which is how a floating coupon is projected.
* ``bumped(bp, tenor=None, width=None)``: a parallel shift or a localised "key rate" bump (a triangular bump centred on a node) for risk;
  ``shifted(fn)`` applies any function of tenor (steepeners, twists).
* ``bootstrap_par_curve``: the zero curve from par swap rates (fixed leg paying ``freq`` times a year; a single-curve bootstrap where the floating leg is worth par).
* ``tenor_basis_curve``: a projection curve for a floating tenor built from the discount curve plus a spread term structure (the multi-curve world after 2008, where 3M and
  6M rates are different).
* ``from_values`` / ``to_values``: the ``{tenor: rate}`` dictionaries carried by curve market-data events.
* ``CurveSet``: the discount and projection curves a swap needs, selected by the curve ids in its contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscountCurve:
    valuation: pd.Timestamp
    tenors: tuple
    zeros: tuple
    interpolation: str = "log_linear_df"
    name: str = ""

    def __post_init__(self):
        t = np.asarray(self.tenors, dtype=float)
        if len(t) < 1 or np.any(np.diff(t) <= 0):
            raise ValueError("tenors must be increasing")
        if self.interpolation not in ("log_linear_df", "linear_zero"):
            raise ValueError("interpolation must be log_linear_df or linear_zero")

    # -------------------------------------------------------------------------------------------------------------------------- discounting
    def _zero_at(self, t):
        t = np.asarray(t, dtype=float)
        nodes, z = np.asarray(self.tenors, float), np.asarray(self.zeros, float)
        if self.interpolation == "linear_zero":
            out = np.interp(t, nodes, z)
            return np.where(t > nodes[-1], z[-1], out)
        # log-linear DF: ln DF(t) = -z t is linear in t between nodes (with a node at t = 0, DF = 1)
        x = np.concatenate([[0.0], nodes])
        y = np.concatenate([[0.0], -z * nodes])
        ln_df = np.interp(t, x, y)
        beyond = t > nodes[-1]
        if len(nodes) >= 2:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            slope = -z[-1]
        ln_df = np.where(beyond, y[-1] + slope * (t - nodes[-1]), ln_df)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(t > 0, -ln_df / t, z[0])

    def zero(self, t):
        return self._zero_at(t)

    def df(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self._zero_at(np.maximum(t, 0.0)) * np.maximum(t, 0.0))

    def t(self, date) -> float:
        return (pd.Timestamp(date) - self.valuation).days / 365.0

    def df_date(self, date) -> float:
        return float(self.df(self.t(date)))

    def forward_rate(self, t1: float, t2: float, tau: float | None = None) -> float:
        """Simple forward rate over ``[t1, t2]``; ``tau`` is the accrual fraction under the coupon's day count (default ``t2 - t1``)."""
        tau = (t2 - t1) if tau is None else tau
        return float((self.df(t1) / self.df(t2) - 1.0) / tau)

    def forward_rate_dates(self, d1, d2, tau: float) -> float:
        return self.forward_rate(self.t(d1), self.t(d2), tau)

    # --------------------------------------------------------------------------------------------------------------------------- transforms
    def shifted(self, fn: Callable[[np.ndarray], np.ndarray], name: str | None = None) -> "DiscountCurve":
        """A curve whose zero rates are increased by ``fn(tenor)`` (in decimal rate units)."""
        z = np.asarray(self.zeros, float) + np.asarray(fn(np.asarray(self.tenors, float)), float)
        return DiscountCurve(self.valuation, self.tenors, tuple(z), self.interpolation, name or self.name)

    def bumped(self, bp: float = 1.0, tenor: float | None = None, width: float | None = None) -> "DiscountCurve":
        """Parallel bump of ``bp`` basis points, or a triangular bump centred on ``tenor`` that falls to zero at its neighbouring nodes (a key-rate bump)."""
        if tenor is None:
            return self.shifted(lambda t: np.full_like(t, bp * 1e-4))
        nodes = np.asarray(self.tenors, float)
        i = int(np.argmin(np.abs(nodes - tenor)))
        lo = nodes[i - 1] if i > 0 else nodes[i] - 1.0
        hi = nodes[i + 1] if i < len(nodes) - 1 else nodes[i] + 1.0
        c = nodes[i]

        def bump(t):
            out = np.zeros_like(t)
            left = (t >= lo) & (t <= c)
            right = (t > c) & (t <= hi)
            out[left] = (t[left] - lo) / (c - lo) if c > lo else 1.0
            out[right] = (hi - t[right]) / (hi - c) if hi > c else 1.0
            return out * bp * 1e-4
        return self.shifted(bump)

    def rolled(self, years: float) -> "DiscountCurve":
        """The curve as it will look after ``years`` if FORWARD rates are realised (the "forward curve"): ``DF'(t) = DF(t + years) / DF(years)``."""
        nodes = np.asarray(self.tenors, float)
        t_new = np.unique(np.concatenate([nodes[nodes > 1e-9] - years, [max(nodes[0], 1e-3)]]))
        t_new = t_new[t_new > 1e-9]
        df_new = self.df(t_new + years) / self.df(years)
        return DiscountCurve(self.valuation + pd.Timedelta(days=round(years * 365)), tuple(t_new), tuple(-np.log(df_new) / t_new), self.interpolation, self.name)

    def to_values(self) -> dict:
        return {float(t): float(z) for t, z in zip(self.tenors, self.zeros)}

    @staticmethod
    def from_values(valuation, values: dict, interpolation: str = "log_linear_df", name: str = "") -> "DiscountCurve":
        items = sorted((float(t), float(r)) for t, r in values.items())
        return DiscountCurve(pd.Timestamp(valuation), tuple(t for t, _ in items), tuple(r for _, r in items), interpolation, name)

    @staticmethod
    def flat(valuation, rate: float, name: str = "") -> "DiscountCurve":
        return DiscountCurve(pd.Timestamp(valuation), (1.0, 30.0), (rate, rate), "log_linear_df", name)

    @staticmethod
    def from_discount_factors(valuation, tenors, dfs, name: str = "") -> "DiscountCurve":
        t = np.asarray(tenors, float)
        return DiscountCurve(pd.Timestamp(valuation), tuple(t), tuple(-np.log(np.asarray(dfs, float)) / t), "log_linear_df", name)

    @staticmethod
    def from_nelson_siegel(valuation, beta0: float, beta1: float, beta2: float, lam: float, tenors=(0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30), name: str = "") -> "DiscountCurve":
        from ..assets.rates import nelson_siegel

        t = np.asarray(tenors, float)
        z = np.asarray(nelson_siegel(t, beta0, beta1, beta2, lam), float)
        return DiscountCurve(pd.Timestamp(valuation), tuple(t), tuple(z), "log_linear_df", name)


def bootstrap_par_curve(valuation, par_tenors, par_rates, freq: int = 2, name: str = "") -> DiscountCurve:
    """Zero curve from par swap rates with fixed payments ``freq`` times a year, a single curve for discounting and projection. At each par tenor ``T`` the discount factor satisfies
    ``par = (1 - DF(T)) / sum(tau DF(t_i))`` over the fixed payment dates; intermediate par rates are linearly interpolated."""
    tenors = np.asarray(par_tenors, float)
    rates = np.asarray(par_rates, float)
    grid = np.arange(1.0 / freq, tenors.max() + 1e-9, 1.0 / freq)
    par = np.interp(grid, tenors, rates)
    dfs = np.zeros(len(grid))
    tau = 1.0 / freq
    for i, (t, p) in enumerate(zip(grid, par)):
        annuity_prior = tau * dfs[:i].sum()
        dfs[i] = (1.0 - p * annuity_prior) / (1.0 + p * tau)
    keep = np.isin(np.round(grid, 8), np.round(tenors, 8)) | (np.arange(len(grid)) % max(freq // 2, 1) == 0)
    return DiscountCurve.from_discount_factors(valuation, grid[keep], dfs[keep], name)


def tenor_basis_curve(discount: DiscountCurve, spread_bp: dict, name: str = "") -> DiscountCurve:
    """A projection curve for a floating tenor: the discount curve's zero rates plus a term structure of tenor-basis spreads in basis points (``{tenor: bp}``)."""
    t = np.asarray(discount.tenors, float)
    s = np.interp(t, sorted(spread_bp), [spread_bp[k] for k in sorted(spread_bp)]) * 1e-4
    return DiscountCurve(discount.valuation, discount.tenors, tuple(np.asarray(discount.zeros, float) + s), discount.interpolation, name or f"{discount.name}+basis")


@dataclass
class CurveSet:
    """The curves a contract refers to by id. ``projection`` falls back to ``discount`` when a projection curve id is not present (single-curve world)."""

    discount: dict = field(default_factory=dict)
    projection: dict = field(default_factory=dict)

    def get_discount(self, curve_id: str) -> DiscountCurve:
        try:
            return self.discount[curve_id]
        except KeyError:
            raise KeyError(f"no discount curve '{curve_id}' (have {sorted(self.discount)})") from None

    def get_projection(self, curve_id: str, fallback: str | None = None) -> DiscountCurve:
        if curve_id in self.projection:
            return self.projection[curve_id]
        if curve_id in self.discount:
            return self.discount[curve_id]
        if fallback is not None:
            return self.get_discount(fallback)
        raise KeyError(f"no projection curve '{curve_id}'")

    def bumped(self, bp: float = 1.0, discount: bool = True, projection: bool = True, tenor: float | None = None) -> "CurveSet":
        return CurveSet({k: (v.bumped(bp, tenor) if discount else v) for k, v in self.discount.items()}, {k: (v.bumped(bp, tenor) if projection else v) for k, v in self.projection.items()})

    def shifted(self, fn) -> "CurveSet":
        return CurveSet({k: v.shifted(fn) for k, v in self.discount.items()}, {k: v.shifted(fn) for k, v in self.projection.items()})
