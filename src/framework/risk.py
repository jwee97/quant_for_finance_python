"""Adaptive risk: a volatility target that depends on the regime.

``vol_target_series`` is the engine's volatility targeting (``vol_target_scaling``, one implementation) with the
target allowed to change through time. With a constant target it reproduces the engine's scalar exactly (a tested identity).
``RegimeRiskPolicy`` sets the target as the probability-weighted average of per-regime targets, so risk is cut smoothly as a
crisis becomes more probable rather than flipped by a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..backtest.execution import vol_target_scaling


def vol_target_series(weights: pd.DataFrame, returns: pd.DataFrame, target: float | pd.Series, lookback: int = 63,
                      max_leverage: float = 1.5, periods_per_year: int = 252) -> tuple[pd.DataFrame, pd.Series]:
    return vol_target_scaling(weights, returns, target, lookback, max_leverage, periods_per_year)


@dataclass
class RegimeRiskPolicy:
    targets: dict = field(default_factory=lambda: {"LowVol": 0.10, "HighVol": 0.08, "Crisis": 0.05})
    default_target: float = 0.10
    lookback: int = 63
    max_leverage: float = 1.5

    def target_series(self, regimes, index: pd.DatetimeIndex) -> pd.Series:
        if regimes is None:
            return pd.Series(self.default_target, index=index)
        probs = regimes.probabilities.reindex(index)
        total = pd.Series(0.0, index=index)
        covered = pd.Series(0.0, index=index)
        for name in probs.columns:
            p = probs[name].fillna(0.0)
            total = total + p * float(self.targets.get(name, self.default_target))
            covered = covered + p
        return total + (1.0 - covered).clip(0.0, 1.0) * self.default_target

    def apply(self, weights: pd.DataFrame, returns: pd.DataFrame, regimes=None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        target = self.target_series(regimes, weights.index)
        scaled, scalar = vol_target_series(weights, returns, target, self.lookback, self.max_leverage)
        return scaled, scalar, target
