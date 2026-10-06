"""Forecast models, the calibration that turns a score into a forecast, and forecast combination.

A **score** is any causal signal (momentum, a z-score, a yield-curve slope). A **forecast** is
``Forecast(mean, std, confidence)``. ``score_to_forecast`` is the bridge: it learns, from labels that have already
matured, how many units of return a unit of score has been worth, and multiplies. That is the only calibration, it is
pooled over assets, and it uses only pairs ``(score_s, return over s+1..s+h)`` with ``s <= t - h`` at date ``t``.

Combination happens at the FORECAST level: a mixture of the models' Gaussians. Where models disagree, the combined
``std`` widens and the confidence falls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.probabilistic import ewma_sigma
from ..signals.alpha_engine import forward_returns, matured_ic, trailing_ir, trust_weights
from ..utils.dates import rebalance_dates
from .types import ForecastPanel, confidence_from


class ForecastModel:
    """Base class. Subclasses implement ``score`` (or override ``forecast`` to produce means directly)."""

    name = "model"
    family = ""
    description = ""
    horizon = 21

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def score(self, data) -> pd.DataFrame:
        raise NotImplementedError

    def forecast(self, data, min_observations: int = 504, allow_negative: bool = False, halflife: float = 40.0) -> ForecastPanel:
        score = self.score(data)
        score = score.where(data.investable.reindex_like(score).fillna(False))
        return score_to_forecast(score, data.returns, self.horizon, min_observations, allow_negative, halflife, name=self.name)

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "description": self.description, "params": self.params()}


def calibration_slope(score: pd.DataFrame, returns: pd.DataFrame, horizon: int, min_observations: int = 504,
                      allow_negative: bool = False) -> pd.Series:
    """Pooled no-intercept slope of the forward ``horizon``-day return on the score, using only MATURED pairs.

    The value on date ``t`` uses pairs from dates ``s <= t - horizon``, whose forward return was complete by ``t``.
    NaN until ``min_observations`` such DATES (each with at least five assets) exist.
    """
    fwd = forward_returns(returns, horizon).reindex_like(score)
    valid = score.notna() & fwd.notna()
    x, y = score.where(valid), fwd.where(valid)
    sxy = (x * y).sum(axis=1).cumsum().shift(horizon)
    sxx = (x * x).sum(axis=1).cumsum().shift(horizon)
    days = (valid.sum(axis=1) >= 5).cumsum().shift(horizon)            # dates with a usable cross-section whose label has matured
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (sxy / sxx.where(sxx > 0)).where(days >= min_observations)
    return slope if allow_negative else slope.clip(lower=0.0)


def score_to_forecast(score: pd.DataFrame, returns: pd.DataFrame, horizon: int = 21, min_observations: int = 504,
                      allow_negative: bool = False, halflife: float = 40.0, name: str = "") -> ForecastPanel:
    slope = calibration_slope(score, returns, horizon, min_observations, allow_negative)
    std = ewma_sigma(returns, halflife, horizon).reindex_like(score)
    mean = score.mul(slope, axis=0)
    return ForecastPanel.from_mean_std(mean, std.where(mean.notna()), horizon, name)


# --------------------------------------------------------------------------------------------- combination
def _stack(panels: dict[str, ForecastPanel]):
    names = list(panels)
    first = panels[names[0]]
    index, columns = first.mean.index, first.mean.columns
    mean = np.stack([panels[n].mean.reindex(index=index, columns=columns).to_numpy() for n in names])
    std = np.stack([panels[n].std.reindex(index=index, columns=columns).to_numpy() for n in names])
    conf = np.stack([panels[n].confidence.reindex(index=index, columns=columns).to_numpy() for n in names])
    return names, index, columns, mean, std, conf


def combine_forecasts(panels: dict[str, ForecastPanel], rule: str = "equal", weights: pd.DataFrame | None = None) -> ForecastPanel:
    """Gaussian-mixture combination. ``rule``: ``equal``, ``confidence`` (w ~ confidence), ``precision`` (w ~ 1/std^2) or ``given``
    (``weights`` is a dates x models frame, e.g. from a trust rule). Weights are renormalised over the models present."""
    names, index, columns, mean, std, conf = _stack(panels)
    present = np.isfinite(mean) & np.isfinite(std)
    if rule == "equal":
        w = np.where(present, 1.0, 0.0)
    elif rule == "confidence":
        w = np.where(present, np.nan_to_num(conf, nan=0.0) + 1e-3, 0.0)
    elif rule == "precision":
        w = np.where(present, 1.0 / np.where(std > 0, std, np.nan) ** 2, 0.0)
        w = np.nan_to_num(w)
    elif rule == "given":
        if weights is None:
            raise ValueError("rule='given' needs a weights frame")
        wm = weights.reindex(index=index, columns=names).to_numpy()
        w = np.where(present, np.nan_to_num(wm.T[:, :, None] * np.ones((1, 1, len(columns)))), 0.0)
    else:
        raise ValueError(f"unknown combination rule '{rule}'")
    total = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / np.where(total > 0, total, np.nan)
        m = np.nansum(w * np.where(present, mean, 0.0), axis=0)
        var = np.nansum(w * np.where(present, std ** 2 + (mean - m) ** 2, 0.0), axis=0)
    empty = ~(total > 0)
    m[empty], var[empty] = np.nan, np.nan
    out_mean = pd.DataFrame(m, index=index, columns=columns)
    out_std = pd.DataFrame(np.sqrt(var), index=index, columns=columns)
    return ForecastPanel(out_mean, out_std, confidence_from(out_mean, out_std), next(iter(panels.values())).horizon, rule)


def ic_trust_weights(panels: dict[str, ForecastPanel], returns: pd.DataFrame, window: int = 504, min_obs: int = 126) -> pd.DataFrame:
    """Monthly trust weights proportional to the positive part of each model's trailing information ratio of its MATURED rank IC."""
    horizon = next(iter(panels.values())).horizon
    fwd = forward_returns(returns, horizon)
    index = pd.DatetimeIndex(returns.index)
    scores = pd.DataFrame({n: trailing_ir(matured_ic(p.mean.reindex(index), fwd, horizon), window, min_obs) for n, p in panels.items()})
    return trust_weights(scores, rebalance_dates(index, "monthly"), shrink=0.0)
