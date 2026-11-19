"""Was the risk forecast right, and is the model drifting?

**Bias statistic** (Menchero and others; Paleologo ch. 4): if ``sigma_(t-1)`` is the volatility forecast for period ``t``, the standardised returns ``r_t / sigma_(t-1)`` have unit standard deviation when the
forecast is right. The bias statistic is that standard deviation over a window; with ``w`` observations its standard error is about ``1/sqrt(2w)``, so a value outside ``1 +- 2/sqrt(2w)`` says the model
under- (above) or over-states (below) risk. :func:`bias_statistic`.

**Drift** (:func:`exposure_drift`): how far the factor exposures and the factor volatilities have moved since the model was last fitted: a model whose inputs have moved a long way is a model to refit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bias_statistic(returns: pd.Series, forecast_vol: pd.Series, window: int = 252, min_periods: int | None = None) -> dict:
    """Rolling standard deviation of ``returns / forecast_vol`` (the forecast must already be the one made BEFORE each return: pass it lagged). ``verdict`` is for the latest window."""
    z = (returns / forecast_vol.reindex(returns.index)).replace([np.inf, -np.inf], np.nan)
    roll = z.rolling(window, min_periods=min_periods or max(window // 2, 20)).std(ddof=1)
    n = z.rolling(window, min_periods=min_periods or max(window // 2, 20)).count()
    band = 2.0 / np.sqrt(2.0 * n)
    last = roll.dropna()
    if last.empty:
        return {"series": roll, "low": 1 - band, "high": 1 + band, "latest": float("nan"), "full": float("nan"), "verdict": "not enough data"}
    cur, b = float(last.iloc[-1]), float(band.loc[last.index[-1]])
    full = float(z.dropna().std(ddof=1)) if z.notna().sum() > 2 else float("nan")
    verdict = "ok" if abs(cur - 1.0) <= b else ("risk understated: realised moves exceed the forecast" if cur > 1 else "risk overstated: realised moves are smaller than the forecast")
    return {"series": roll, "low": 1 - band, "high": 1 + band, "latest": cur, "full": full, "verdict": verdict}


def portfolio_forecast_check(weights: pd.Series, returns: pd.DataFrame, estimator, lookback: int = 252, step: int = 21, horizon: int | None = None) -> pd.DataFrame:
    """Forecast against realised volatility of a fixed-weight portfolio: every ``step`` periods, estimate the covariance on the previous ``lookback`` rows with ``estimator(window) -> covariance``
    (per-period units) and compare ``sqrt(h' Sigma h)`` with the realised standard deviation of the next ``horizon`` (default ``step``) portfolio returns. Columns: forecast, realised, ratio."""
    h = weights.reindex(returns.columns).fillna(0.0).to_numpy(dtype=float)
    pr = returns.fillna(0.0).to_numpy(dtype=float) @ h
    horizon = int(horizon or step)
    rows = {}
    for t in range(lookback, len(returns) - horizon, step):
        cov = np.asarray(estimator(returns.iloc[t - lookback:t]), dtype=float)
        fc = float(np.sqrt(max(h @ cov @ h, 0.0)))
        real = float(pr[t:t + horizon].std(ddof=1))
        rows[returns.index[t]] = {"forecast": fc, "realised": real, "ratio": real / fc if fc > 0 else np.nan}
    return pd.DataFrame(rows).T


def exposure_drift(before, after) -> pd.DataFrame:
    """Per factor, how the portfolio-level exposure (or the asset-by-factor mean exposure) and the factor volatility changed between two fitted models. ``before`` and ``after`` are
    :class:`~src.riskengine.factor_model.FactorModel`; only the factors in both are compared."""
    common = [f for f in before.factors if f in after.factors]
    ba = before.exposures[common].mean()
    aa = after.exposures[common].reindex(before.exposures.index.intersection(after.exposures.index))[common].mean()
    vb = pd.Series(np.sqrt(np.diag(before.factor_cov.loc[common, common])), index=common)
    va = pd.Series(np.sqrt(np.diag(after.factor_cov.loc[common, common])), index=common)
    shared = before.exposures.index.intersection(after.exposures.index)
    shift = (after.exposures.loc[shared, common] - before.exposures.loc[shared, common]).abs().mean()
    return pd.DataFrame({"mean_exposure_before": ba, "mean_exposure_after": aa, "mean_abs_exposure_change": shift, "factor_vol_before": vb, "factor_vol_after": va,
                         "factor_vol_change": va / vb.where(vb > 0) - 1.0})
