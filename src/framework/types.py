"""The vocabulary of the research pipeline: what a regime, a forecast and a panel of them are.

    Market -> Regime -> Forecast(mean, std, confidence) -> Portfolio -> Risk

A **forecast** is not a number but a distribution summary: ``mean`` and ``std`` of the next ``horizon`` days' return,
and a ``confidence`` that says how far the forecast sits from a coin flip. Portfolio sizing and forecast combination
then react to confidence, not only to the mean.

A **regime** is a named state of the market with a probability. Everything downstream (the allocator, the risk
budget) can react to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


def confidence_from(mean, std):
    """``|2 * P(r > 0) - 1|`` with ``P(r > 0) = Phi(mean / std)``: 0 for a coin flip, 1 for certainty.

    Works on scalars, arrays, Series and DataFrames, returning the same kind. A zero or missing ``std`` gives NaN
    rather than a spurious 1.
    """
    if isinstance(mean, pd.DataFrame):
        z = mean / std.where(std > 0)
        return (2.0 * pd.DataFrame(norm.cdf(z.to_numpy()), index=z.index, columns=z.columns) - 1.0).abs()
    if isinstance(mean, pd.Series):
        z = mean / std.where(std > 0)
        return (2.0 * pd.Series(norm.cdf(z.to_numpy()), index=z.index) - 1.0).abs()
    std = np.asarray(std, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.asarray(mean, dtype=float) / np.where(std > 0, std, np.nan)
    return np.abs(2.0 * norm.cdf(z) - 1.0)


@dataclass(frozen=True)
class Forecast:
    """One asset, one date: the expected return over the horizon, its uncertainty and the confidence."""

    mean: float
    std: float
    confidence: float

    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> "Forecast":
        return cls(float(mean), float(std), float(confidence_from(np.float64(mean), np.float64(std))))

    @property
    def probability_up(self) -> float:
        return float(norm.cdf(self.mean / self.std)) if self.std > 0 else float("nan")


@dataclass
class ForecastPanel:
    """Forecasts for many assets on many dates. Row ``t`` may use only information available at the close of ``t``."""

    mean: pd.DataFrame
    std: pd.DataFrame
    confidence: pd.DataFrame
    horizon: int = 21
    name: str = ""
    p_up: pd.DataFrame | None = None      # P(return > 0) when it has been calibrated; ``None`` means the Gaussian value Phi(mean / std)

    @classmethod
    def from_mean_std(cls, mean: pd.DataFrame, std: pd.DataFrame, horizon: int = 21, name: str = "") -> "ForecastPanel":
        std = std.reindex_like(mean)
        return cls(mean, std, confidence_from(mean, std), horizon, name)

    def probability_up(self) -> pd.DataFrame:
        """The calibrated probability of an up move if there is one, else the Gaussian ``Phi(mean / std)``."""
        if self.p_up is not None:
            return self.p_up
        z = self.mean / self.std.where(self.std > 0)
        return pd.DataFrame(norm.cdf(z.to_numpy()), index=z.index, columns=z.columns)

    def at(self, date, asset) -> Forecast:
        return Forecast(float(self.mean.at[date, asset]), float(self.std.at[date, asset]), float(self.confidence.at[date, asset]))

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.mean.index

    def restrict(self, index) -> "ForecastPanel":
        return ForecastPanel(self.mean.reindex(index), self.std.reindex(index), self.confidence.reindex(index), self.horizon, self.name,
                             None if self.p_up is None else self.p_up.reindex(index))

    def first_valid(self) -> pd.Timestamp | None:
        valid = self.mean.dropna(how="all")
        return valid.index[0] if len(valid) else None


@dataclass(frozen=True)
class Regime:
    """The market's state on one date: its most probable name, that probability and the full distribution."""

    name: str
    probability: float
    probabilities: dict = field(default_factory=dict)
    label: str = ""

    def __str__(self) -> str:
        return f"Regime(name={self.name!r}, probability={self.probability:.2f})"


@dataclass
class RegimeSeries:
    """Regime probabilities through time: one column per regime name, rows summing to one where defined."""

    probabilities: pd.DataFrame
    detector: str = ""
    annotations: pd.DataFrame = field(default_factory=pd.DataFrame)     # e.g. a Bull/Bear trend flag

    @property
    def names(self) -> list[str]:
        return list(self.probabilities.columns)

    def at(self, date) -> Regime:
        row = self.probabilities.loc[date].dropna()
        if row.empty:
            return Regime("Unknown", 0.0, {})
        name = str(row.idxmax())
        label = name
        if not self.annotations.empty and date in self.annotations.index and isinstance(self.annotations.at[date, "trend"], str):
            label = f"{name}{self.annotations.at[date, 'trend']}"
        return Regime(name, float(row.max()), {k: float(v) for k, v in row.items()}, label)

    def hard_labels(self) -> pd.Series:
        valid = self.probabilities.dropna(how="all")
        labels = valid.idxmax(axis=1)
        return labels.reindex(self.probabilities.index)

    def share(self) -> pd.Series:
        return self.hard_labels().value_counts(normalize=True)

    def validate(self) -> None:
        sums = self.probabilities.dropna(how="all").sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError("regime probabilities must sum to one on every date where they are defined")
        if (self.probabilities.dropna(how="all") < -1e-12).any().any():
            raise ValueError("negative regime probability")
