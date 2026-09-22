"""Portfolio constraints (Ch. 19 §19.2.3-§19.2.6, spec §29).

Constraints are expressed once, here, as a ``Constraints`` object that can be
handed to any optimiser. Two reasons this is worth the abstraction:

1. Every model in the ladder is then subject to *identical* limits, so when
   M6 beats M2 the difference is the objective, not a quietly looser cap.
2. A constraint set can be checked for feasibility before an optimiser is
   asked to solve an impossible problem, which otherwise produces a silent
   corner solution that looks like a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    """Position, leverage, group and turnover limits."""

    min_weight: float = 0.0
    max_weight: float = 0.25
    gross_leverage: float | None = 1.0      # sum |w|
    net_exposure: float | None = 1.0        # sum w  (None => unconstrained)
    group_limits: dict[str, float] = field(default_factory=dict)
    group_map: dict[str, str] = field(default_factory=dict)
    max_turnover: float | None = None
    long_only: bool = True

    @classmethod
    def from_config(cls, config, group_map: dict[str, str] | None = None) -> "Constraints":
        node = config.get("portfolio.constraints", {}) or {}
        min_w = float(node.get("min_weight", 0.0))
        return cls(
            min_weight=min_w,
            max_weight=float(node.get("max_weight", 0.25)),
            gross_leverage=node.get("gross_leverage", 1.0),
            net_exposure=1.0,
            group_limits=dict(node.get("group_limits", {}) or {}),
            group_map=dict(group_map or config.group_map),
            max_turnover=node.get("max_turnover"),
            long_only=min_w >= 0.0,
        )

    # -- inspection -------------------------------------------------------
    def bounds(self, assets) -> list[tuple[float, float]]:
        return [(self.min_weight, self.max_weight) for _ in assets]

    def groups(self, assets) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, asset in enumerate(assets):
            key = self.group_map.get(asset)
            if key in self.group_limits:
                out.setdefault(key, []).append(i)
        return out

    def is_feasible(self, assets) -> tuple[bool, str]:
        """Can any weight vector satisfy these constraints at once?

        Checked before optimising, because an infeasible problem does not
        announce itself: SLSQP simply returns the least-bad corner and the
        backtest runs on it.
        """
        n = len(assets)
        if n == 0:
            return False, "no assets"
        target = self.net_exposure if self.net_exposure is not None else 1.0
        if n * self.max_weight < target - 1e-9:
            return False, (f"{n} assets capped at {self.max_weight:.0%} cannot reach a net "
                           f"exposure of {target:.0%}")
        if n * self.min_weight > target + 1e-9:
            return False, "minimum weights already exceed the net exposure target"
        groups = self.groups(assets)
        if groups:
            reachable = 0.0
            for key, members in groups.items():
                reachable += min(len(members) * self.max_weight, self.group_limits[key])
            ungrouped = n - sum(len(v) for v in groups.values())
            reachable += ungrouped * self.max_weight
            if reachable < target - 1e-9:
                return False, (f"group limits cap total investable exposure at {reachable:.0%}, "
                               f"below the {target:.0%} target")
        return True, "feasible"

    # -- application ------------------------------------------------------
    def scipy_constraints(self, assets, previous: np.ndarray | None = None) -> list[dict]:
        """Equality/inequality constraints in scipy's ``minimize`` format."""
        out: list[dict] = []
        if self.net_exposure is not None:
            out.append({"type": "eq", "fun": lambda w, t=self.net_exposure: float(np.sum(w) - t)})
        if self.gross_leverage is not None and not self.long_only:
            out.append({"type": "ineq",
                        "fun": lambda w, g=self.gross_leverage: float(g - np.sum(np.abs(w)))})
        for key, members in self.groups(assets).items():
            limit = self.group_limits[key]
            idx = np.array(members)
            out.append({"type": "ineq",
                        "fun": lambda w, i=idx, l=limit: float(l - np.sum(w[i]))})
        if self.max_turnover is not None and previous is not None:
            out.append({"type": "ineq",
                        "fun": lambda w, p=previous, t=self.max_turnover:
                        float(t - np.sum(np.abs(w - p)))})
        return out

    def violations(self, weights: pd.Series) -> dict[str, float]:
        """Report by how much a weight vector breaches each constraint."""
        out: dict[str, float] = {}
        over = float((weights - self.max_weight).clip(lower=0.0).sum())
        under = float((self.min_weight - weights).clip(lower=0.0).sum())
        if over > 1e-9:
            out["max_weight"] = over
        if under > 1e-9:
            out["min_weight"] = under
        if self.net_exposure is not None and abs(float(weights.sum()) - self.net_exposure) > 1e-6:
            out["net_exposure"] = float(weights.sum()) - self.net_exposure
        if self.gross_leverage is not None and float(weights.abs().sum()) > self.gross_leverage + 1e-6:
            out["gross_leverage"] = float(weights.abs().sum()) - self.gross_leverage
        for key, limit in self.group_limits.items():
            members = [a for a in weights.index if self.group_map.get(a) == key]
            exposure = float(weights[members].sum()) if members else 0.0
            if exposure > limit + 1e-6:
                out[f"group_{key}"] = exposure - limit
        return out

    def project(self, weights: pd.Series, iterations: int = 200, tolerance: float = 1e-10) -> pd.Series:
        """Project a weight vector onto the constraint set.

        Used to make a heuristic portfolio (equal weight, inverse volatility,
        risk parity) obey the same limits as the optimised ones without
        re-solving. Alternating projections: box, then groups, then the
        exposure equality, repeated to convergence.
        """
        w = weights.astype(float).copy()
        target = self.net_exposure if self.net_exposure is not None else float(w.sum())

        for _ in range(iterations):
            before = w.copy()
            w = w.clip(lower=self.min_weight, upper=self.max_weight)

            for key, limit in self.group_limits.items():
                members = [a for a in w.index if self.group_map.get(a) == key]
                if not members:
                    continue
                exposure = float(w[members].sum())
                if exposure > limit + 1e-12:
                    w[members] = w[members] * (limit / exposure)

            total = float(w.sum())
            if abs(total) > 1e-12 and self.net_exposure is not None:
                headroom = (self.max_weight - w).clip(lower=0.0)
                deficit = target - total
                if deficit > 1e-12 and float(headroom.sum()) > 1e-12:
                    w = w + headroom * (deficit / float(headroom.sum()))
                elif deficit < -1e-12:
                    room = (w - self.min_weight).clip(lower=0.0)
                    if float(room.sum()) > 1e-12:
                        w = w + room * (deficit / float(room.sum()))

            if float((w - before).abs().max()) < tolerance:
                break
        return w


def apply_turnover_limit(target: pd.Series, previous: pd.Series, max_turnover: float) -> pd.Series:
    """Move only part of the way towards the target if the trade is too large.

    A partial move along the straight line from the current book to the
    target is the simplest turnover-constrained solution and, unlike dropping
    the rebalance entirely, it keeps the portfolio moving in the right
    direction.
    """
    previous = previous.reindex(target.index).fillna(0.0)
    required = float((target - previous).abs().sum())
    if max_turnover is None or required <= max_turnover + 1e-12 or required < 1e-12:
        return target
    scale = max_turnover / required
    return previous + (target - previous) * scale
