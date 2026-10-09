"""Combination rules that weigh the models by the covariance of their information coefficients (the optimal alpha model of Qian, Hua and Sorensen).

``ic_weighted`` trusts each model by its own IC and ignores that two momentum models make the same bets. Here the weights are ``w = Sigma^-1 mu`` with ``mu`` the mean and ``Sigma`` the covariance
of the models' matured ICs over a trailing window, which maximises the information ratio of the combined alpha: a model that moves against the others is worth more than its own IC says and a copy of
one already in the mix is worth less. With the forecasts made orthogonal first (``orthogonal_ic``) the formula is exact; without it (``optimal_ic``) it is the same rule applied to the forecasts
as they are.

Both are causal: the weights on date ``t`` use the ICs of dates ``s <= t - horizon`` (the realised return after ``s`` had ended by ``t``) and the forecasts of ``t``. Weights are non-negative here
because the platform mixes the models' forecast distributions: a model that has not earned is left out rather than traded the other way (``src.equity.alpha_model.optimal_alpha`` allows negative weights
on a table of factors).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..equity.alpha_model import (best_possible_ir, gram_schmidt, ic_moments, information_coefficients, max_ir_weights, standardize_all, symmetric_orthogonalize)
from ..signals.alpha_engine import forward_returns
from ..utils.dates import rebalance_dates
from .types import ForecastPanel

ORTHOGONALIZATIONS = ("gram_schmidt", "symmetric")


def _standardised(panels: dict, index: pd.DatetimeIndex, orthogonalize: str | None) -> dict:
    z = standardize_all({n: p.mean.reindex(index) for n, p in panels.items()})
    if orthogonalize == "gram_schmidt":
        return gram_schmidt(z)
    if orthogonalize == "symmetric":
        return symmetric_orthogonalize(z)
    if orthogonalize is not None:
        raise ValueError(f"orthogonalize must be one of {ORTHOGONALIZATIONS} or None")
    return z


def _weights(z: dict, horizon: int, returns: pd.DataFrame, window: int, min_obs: int, shrink: float) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly non-negative max-IR weights (rows sum to one) on the standardised forecasts ``z`` and the information ratio they were expected to reach."""
    names, index = list(z), pd.DatetimeIndex(returns.index)
    ic = information_coefficients(z, forward_returns(returns, horizon), "spearman", min_assets=5).reindex(index).shift(horizon)       # the IC that is KNOWN on each date
    equal = np.full(len(names), 1.0 / len(names))
    rows, expected = {}, {}
    for date in rebalance_dates(index, "monthly"):
        if date not in ic.index:
            continue
        known = ic.loc[:date].dropna().iloc[-window:]
        w, best = equal, np.nan
        if len(known) >= min_obs:
            mean, cov = ic_moments(known, shrink)
            fitted = max_ir_weights(mean, cov, nonnegative=True).to_numpy()
            if fitted.sum() > 0:                                       # when no model has earned anything the weights stay equal: a combination rule is not a timing switch
                w, best = fitted, best_possible_ir(mean, cov)
        rows[date], expected[date] = w, best
    weights = pd.DataFrame.from_dict(rows, orient="index", columns=names).reindex(index).ffill().fillna(1.0 / len(names))
    return weights, pd.Series(expected).reindex(index).ffill()


def optimal_ic_weights(panels: dict, returns: pd.DataFrame, window: int = 504, min_obs: int = 252, shrink: float = 0.3) -> tuple[pd.DataFrame, pd.Series]:
    """Weights (dates x models, to pass to ``combine_forecasts(..., 'given', w)``) that maximise the information ratio of the combined forecast, and the ratio expected at each date.

    The IC weights apply to forecasts standardised across assets on each date, so they are divided by each model's dispersion before they are applied to the forecasts in return units;
    ``combine_forecasts`` then normalises them to sum to one over the models present."""
    index = pd.DatetimeIndex(returns.index)
    horizon = next(iter(panels.values())).horizon
    z = _standardised(panels, index, None)
    w, expected = _weights(z, horizon, returns, window, min_obs, shrink)
    dispersion = pd.DataFrame({n: p.mean.reindex(index).std(axis=1) for n, p in panels.items()})
    raw = (w / dispersion.where(dispersion > 1e-12)).fillna(0.0)
    return raw, expected


def orthogonal_ic_forecast(panels: dict, returns: pd.DataFrame, orthogonalize: str = "gram_schmidt", window: int = 504, min_obs: int = 252,
                           shrink: float = 0.3) -> tuple[ForecastPanel, pd.DataFrame, pd.Series]:
    """The same rule on forecasts made orthogonal across assets on every date first (so the formula is exact), returned as one combined forecast.

    The composite is a weighted sum of the orthogonalised forecasts, standardised and scaled back to return units by the weighted average dispersion of the models' own forecasts; its spread is the
    average of the models'. Returns ``(forecast, weights on the orthogonalised forecasts, expected information ratio)``; the weights are in the order the models were listed (the first keeps what it shares)."""
    index = pd.DatetimeIndex(returns.index)
    horizon = next(iter(panels.values())).horizon
    z = _standardised(panels, index, orthogonalize)
    w, expected = _weights(z, horizon, returns, window, min_obs, shrink)
    names = list(panels)
    total = sum(z[n].fillna(0.0) * w[n].to_numpy()[:, None] for n in names)
    seen = np.logical_or.reduce([z[n].notna().to_numpy() for n in names])
    sd = total.where(seen).std(axis=1).replace(0.0, np.nan)
    composite = total.where(seen).div(sd, axis=0)
    dispersion = pd.DataFrame({n: panels[n].mean.reindex(index).std(axis=1) for n in names})
    known = dispersion.notna().astype(float)
    scale = (dispersion.fillna(0.0) * w).sum(axis=1) / (known * w).sum(axis=1).replace(0.0, np.nan)
    mean = composite.mul(scale, axis=0)
    stds = np.stack([panels[n].std.reindex(index=index, columns=mean.columns).to_numpy() for n in names])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # an asset with no model on a date has no spread
        std = pd.DataFrame(np.nanmean(stds, axis=0), index=index, columns=mean.columns)
    first = next(iter(panels.values()))
    return ForecastPanel.from_mean_std(mean, std.where(mean.notna()), first.horizon, "orthogonal_ic"), w, expected
