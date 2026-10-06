"""Adaptive risk: a volatility target that depends on the regime.

``vol_target_series`` is the same arithmetic as the engine's volatility targeting (``vol_target_scaling``) with the
target allowed to change through time. With a constant target it reproduces the engine's scalar exactly (a tested identity).
``RegimeRiskPolicy`` sets the target as the probability-weighted average of per-regime targets, so risk is cut smoothly as a
crisis becomes more probable rather than flipped by a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def vol_target_series(weights: pd.DataFrame, returns: pd.DataFrame, target: float | pd.Series, lookback: int = 63,
                      max_leverage: float = 1.5, periods_per_year: int = 252) -> tuple[pd.DataFrame, pd.Series]:
    aligned = returns.reindex(weights.index).reindex(columns=weights.columns)
    portfolio = (weights.shift(1) * aligned).sum(axis=1)
    realised = portfolio.rolling(lookback, min_periods=max(lookback // 2, 20)).std(ddof=1) * np.sqrt(periods_per_year)
    target_series = target if isinstance(target, pd.Series) else pd.Series(float(target), index=weights.index)
    scalar = (target_series.reindex(weights.index) / realised.replace(0.0, np.nan)).clip(upper=max_leverage).shift(1)
    scalar = scalar.fillna(1.0)
    return weights.mul(scalar, axis=0), scalar


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
