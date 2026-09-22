"""Volatility estimation (Ch. 20 §20.2).

Five estimators, from the crudest to the most structured:

``rolling``       equally weighted standard deviation over k days
``ewma``          exponentially weighted, parameterised by half-life
``parkinson``     high-low range estimator
``garman_klass``  OHLC range estimator
``garch``         GARCH(1,1), sigma^2_t = omega + alpha e^2_{t-1} + beta sigma^2_{t-1}

The reason to have all five is not completeness. It is that the choice of
volatility estimator *is* the position-sizing decision in a vol-targeted
strategy, so it has to be tested rather than assumed. ``evaluate_forecasts``
scores them out of sample against realised volatility, which is the only test
that matters for this project.

Every estimator is causal: the value stamped at ``t`` uses data up to and
including ``t`` only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.dates import TRADING_DAYS_PER_YEAR
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)
ANN = TRADING_DAYS_PER_YEAR


def rolling_volatility(returns: pd.DataFrame, window: int = 63, annualise: bool = True,
                       min_periods: int | None = None) -> pd.DataFrame:
    """Equally weighted rolling standard deviation."""
    min_periods = min_periods or max(int(window * 0.6), 10)
    vol = returns.rolling(window, min_periods=min_periods).std(ddof=1)
    return vol * np.sqrt(ANN) if annualise else vol


def expanding_volatility(returns: pd.DataFrame, min_periods: int = 63, annualise: bool = True) -> pd.DataFrame:
    vol = returns.expanding(min_periods=min_periods).std(ddof=1)
    return vol * np.sqrt(ANN) if annualise else vol


def ewma_volatility(returns: pd.DataFrame, halflife: float = 40.0, annualise: bool = True,
                    min_periods: int = 30) -> pd.DataFrame:
    """Exponentially weighted volatility.

    ``halflife`` is used rather than RiskMetrics' lambda because it is the
    interpretable parameter: "how many days until an observation counts half
    as much". lambda = 0.94 corresponds to a half-life of ~11 days.
    """
    variance = (returns ** 2).ewm(halflife=halflife, min_periods=min_periods).mean()
    vol = np.sqrt(variance)
    return vol * np.sqrt(ANN) if annualise else vol


def lambda_to_halflife(lam: float) -> float:
    return float(np.log(0.5) / np.log(lam))


def parkinson_volatility(high: pd.DataFrame, low: pd.DataFrame, window: int = 21,
                         annualise: bool = True) -> pd.DataFrame:
    """Parkinson (1980) high-low range estimator.

    Uses the intraday range, so it extracts roughly five times more
    information per observation than a close-to-close estimator -- at the cost
    of ignoring overnight gaps, which for ETFs is a real omission.
    """
    log_range = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * log_range.rolling(window, min_periods=max(window // 2, 5)).mean()
    vol = np.sqrt(variance)
    return vol * np.sqrt(ANN) if annualise else vol


def garman_klass_volatility(open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                            close: pd.DataFrame, window: int = 21, annualise: bool = True) -> pd.DataFrame:
    """Garman-Klass (1980) OHLC estimator."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    daily = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    variance = daily.rolling(window, min_periods=max(window // 2, 5)).mean().clip(lower=0.0)
    vol = np.sqrt(variance)
    return vol * np.sqrt(ANN) if annualise else vol


def realised_volatility(returns: pd.DataFrame, window: int = 21, annualise: bool = True) -> pd.DataFrame:
    """Backward-looking realised volatility -- the *target* in forecast tests."""
    variance = (returns ** 2).rolling(window, min_periods=max(window // 2, 5)).mean()
    vol = np.sqrt(variance)
    return vol * np.sqrt(ANN) if annualise else vol


def forward_realised_volatility(returns: pd.DataFrame, horizon: int = 21, annualise: bool = True) -> pd.DataFrame:
    """Realised volatility over ``(t, t+h]``, stamped at ``t``."""
    return realised_volatility(returns, horizon, annualise).shift(-horizon)


# ---------------------------------------------------------------------------
# GARCH (Ch. 20 §20.2.5-6)
# ---------------------------------------------------------------------------
@dataclass
class GarchFit:
    """A fitted GARCH model plus the scale its parameters are expressed in.

    ``arch`` is numerically unhappy when the data are daily decimals, and the
    universe here spans SHY (0.9% annualised... 0.09% daily) to SLV, so a
    single fixed rescaling cannot work for every series. Each series is scaled
    to unit standard deviation before fitting and the factor is carried here
    so that every quantity can be converted back to return units exactly once.
    """

    result: object
    scale: float

    @property
    def params(self) -> pd.Series:
        return self.result.params

    def unpack(self) -> tuple[float, float, float, float]:
        """(omega, alpha, beta, mu) in *scaled* units."""
        params = self.result.params
        omega = float(params.get("omega", np.nan))
        alpha = float(sum(v for k, v in params.items() if str(k).startswith("alpha")))
        beta = float(sum(v for k, v in params.items() if str(k).startswith("beta")))
        mu = float(params.get("mu", 0.0))
        return omega, alpha, beta, mu

    def conditional_volatility(self) -> np.ndarray:
        """Conditional volatility in return units."""
        return np.asarray(self.result.conditional_volatility, dtype=float) / self.scale

    def last_state(self) -> tuple[float, float]:
        """(sigma^2, residual) at the end of the fit, in scaled units."""
        sigma = float(np.asarray(self.result.conditional_volatility, dtype=float)[-1])
        resid = float(np.asarray(self.result.resid, dtype=float)[-1])
        return sigma ** 2, resid


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t",
              rescale: float | None = None) -> GarchFit:
    """Fit GARCH(p, q) to one return series.

    ``rescale=None`` scales the series to unit standard deviation, which is
    where the optimiser is best behaved. Pass a number to fix the scale.
    ``arch`` is an optional dependency; the caller gets a clear error if it is
    missing rather than a silent fallback.
    """
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("GARCH requires the 'arch' package: pip install arch") from exc

    series = returns.dropna()
    if len(series) < 100:
        raise ValueError(f"need at least 100 observations to fit GARCH, got {len(series)}")
    scale = float(rescale) if rescale is not None else float(1.0 / max(series.std(ddof=1), 1e-12))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # data-scale warnings: handled by `scale`
        model = arch_model(series * scale, vol="GARCH", p=p, q=q, dist=dist, mean="constant",
                           rescale=False)
        result = model.fit(disp="off", show_warning=False)
    return GarchFit(result=result, scale=scale)


def garch_conditional_volatility(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t",
                                 annualise: bool = True) -> pd.Series:
    """In-sample conditional volatility from a GARCH(p, q) fit.

    In sample, so this is a *description* of the volatility process, not a
    forecast. Use ``garch_rolling_forecast`` for anything the backtest sees.
    """
    fit = fit_garch(returns, p, q, dist)
    vol = pd.Series(fit.conditional_volatility(), index=returns.dropna().index)
    return vol * np.sqrt(ANN) if annualise else vol


def garch_parameters(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t") -> dict:
    """Parameter table for the GARCH write-up, including persistence.

    ``persistence = alpha + beta`` is the number to look at: at 0.99 a
    volatility shock has a half-life of ~69 days; at >= 1 the model is
    integrated (IGARCH) and has no long-run variance to revert to, which is
    exactly the case the forecaster below must handle without exploding.
    """
    fit = fit_garch(returns, p, q, dist)
    omega, alpha, beta, _ = fit.unpack()
    persistence = alpha + beta
    # omega is in scaled units: variance scales by scale^2.
    omega_return_units = omega / (fit.scale ** 2)
    long_run = (
        float(np.sqrt(omega_return_units / (1.0 - persistence)) * np.sqrt(ANN))
        if persistence < 1.0 - 1e-6 else float("nan")
    )
    return {
        "omega": omega_return_units,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "half_life_days": float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else float("inf"),
        "long_run_vol_ann": long_run,
        "integrated": bool(persistence >= 1.0 - 1e-6),
        "loglikelihood": float(fit.result.loglikelihood),
        "aic": float(fit.result.aic),
        "bic": float(fit.result.bic),
        "nu": float(fit.params.get("nu", np.nan)),
    }


def _horizon_variance(sigma2_next: float, omega: float, persistence: float, horizon: int) -> float:
    """Average one-step variance over ``(t, t+h]`` from the GARCH recursion.

    For ``persistence < 1`` the forecast decays geometrically towards the
    long-run variance. For an integrated fit (``persistence >= 1``) there is
    no long-run variance; the correct forecast is the IGARCH one, which grows
    linearly in omega rather than diverging.
    """
    if persistence < 1.0 - 1e-8:
        long_run = omega / (1.0 - persistence)
        steps = [long_run + (persistence ** step) * (sigma2_next - long_run) for step in range(horizon)]
    else:
        steps = [sigma2_next + step * omega for step in range(horizon)]
    return float(np.mean(steps))


def garch_rolling_forecast(returns: pd.Series, horizon: int = 21, refit_every: int = 63,
                           min_train: int = 750, annualise: bool = True,
                           max_vol_multiple: float = 10.0) -> pd.Series:
    """Walk-forward GARCH(1,1) volatility forecast.

    The model is refit every ``refit_every`` days on data up to that point and
    then rolled forward with realised shocks. Nothing uses data after the
    stamp date, which is what makes the comparison against the simpler
    estimators in ``evaluate_forecasts`` fair.

    ``max_vol_multiple`` caps the forecast at that multiple of the training
    sample's unconditional volatility. A cap is not cosmetic: an unconstrained
    near-integrated recursion can print a volatility of 10^6, and a position
    sizer dividing by it would silently go to zero.
    """
    series = returns.dropna()
    if len(series) < min_train + horizon:
        return pd.Series(dtype=float, index=series.index)

    forecasts = pd.Series(index=series.index, dtype=float)
    failures = 0
    anchor = min_train
    while anchor < len(series):
        train = series.iloc[:anchor]
        try:
            fit = fit_garch(train)
            omega, alpha, beta, mu = fit.unpack()
            persistence = alpha + beta
            sigma2, resid = fit.last_state()
        except Exception as exc:  # optimiser failure: skip this window, but say so
            failures += 1
            LOGGER.warning("GARCH refit at position %d failed (%s); skipping window", anchor, exc)
            anchor += refit_every
            continue

        cap = max_vol_multiple * float(train.std(ddof=1)) * fit.scale
        block = series.iloc[anchor:anchor + refit_every]
        for stamp, value in block.items():
            sigma2 = omega + alpha * resid ** 2 + beta * sigma2
            sigma2 = min(sigma2, cap ** 2)
            horizon_var = _horizon_variance(sigma2, omega, persistence, horizon)
            horizon_var = min(horizon_var, cap ** 2)
            forecasts.loc[stamp] = np.sqrt(max(horizon_var, 1e-18)) / fit.scale
            resid = float(value * fit.scale - mu)
        anchor += refit_every

    forecasts = forecasts.dropna()
    if forecasts.empty:
        raise RuntimeError(
            f"GARCH walk-forward produced no forecasts ({failures} refits failed). "
            "Check the 'arch' installation rather than treating this as a missing series."
        )
    return forecasts * np.sqrt(ANN) if annualise else forecasts


# ---------------------------------------------------------------------------
# Forecast evaluation (spec §33)
# ---------------------------------------------------------------------------
def evaluate_forecasts(forecasts: dict[str, pd.Series], realised: pd.Series) -> pd.DataFrame:
    """Score volatility forecasts against subsequently realised volatility.

    Reported: correlation, RMSE, mean bias, and the QLIKE loss

        QLIKE = realised^2 / forecast^2 - log(realised^2 / forecast^2) - 1

    QLIKE is used because it is robust to the fact that realised volatility is
    itself a noisy proxy for the latent quantity, and it penalises
    *under*-forecasting asymmetrically -- which is the error that matters when
    the forecast is sizing a position.
    """
    rows = {}
    for name, forecast in forecasts.items():
        aligned = pd.concat([forecast.rename("f"), realised.rename("r")], axis=1).dropna()
        aligned = aligned[(aligned["f"] > 0) & (aligned["r"] > 0)]
        if len(aligned) < 30:
            continue
        f, r = aligned["f"].to_numpy(), aligned["r"].to_numpy()
        ratio = (r ** 2) / (f ** 2)
        rows[name] = {
            "n_obs": int(len(aligned)),
            "corr": float(np.corrcoef(f, r)[0, 1]),
            "rmse": float(np.sqrt(np.mean((f - r) ** 2))),
            "mae": float(np.mean(np.abs(f - r))),
            "mean_bias": float(np.mean(f - r)),
            "mean_ratio": float(np.mean(f / r)),
            "qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
            "r2": float(1.0 - np.sum((r - f) ** 2) / np.sum((r - r.mean()) ** 2)),
        }
    return pd.DataFrame(rows).T.sort_values("qlike") if rows else pd.DataFrame()


def volatility_panel(market, method: str = "ewma", window: int = 63, halflife: float = 40.0,
                     annualise: bool = True) -> pd.DataFrame:
    """Dispatch to an estimator using the ``MarketData`` fields it needs."""
    returns = market.returns()
    if method == "rolling":
        return rolling_volatility(returns, window, annualise)
    if method == "expanding":
        return expanding_volatility(returns, window, annualise)
    if method == "ewma":
        return ewma_volatility(returns, halflife, annualise)
    if method == "parkinson":
        return parkinson_volatility(market.high, market.low, window, annualise)
    if method == "garman_klass":
        return garman_klass_volatility(market.open_, market.high, market.low, market.close, window, annualise)
    if method == "realised":
        return realised_volatility(returns, window, annualise)
    raise ValueError(f"unknown volatility method '{method}'")
