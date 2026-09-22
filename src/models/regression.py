"""Expected-return models, information coefficient and regression
(Ch. 20 §20.1, especially §20.1.6-20.1.8).

Three things this module is careful about, because each is a documented way to
fool yourself:

**Overlapping observations.** A 21-day forward return sampled daily is 21-times
overlapping. The point estimate is fine; the naive standard error is wrong by
roughly sqrt(21). Every regression here reports Newey-West (HAC) standard
errors with a lag chosen from the horizon, and ``compare_standard_errors``
shows the size of the lie.

**Cross-sectional vs time-series IC.** They answer different questions ("did
the signal rank assets correctly today?" vs "did the signal time this asset
correctly?"). Both are computed and never averaged together.

**IC is not a Sharpe ratio.** The Fundamental Law
``IR ~ IC * sqrt(breadth)`` converts one into the other, and the conversion
is where an IC of 0.03 stops sounding negligible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------
def _rowwise_correlation(x: np.ndarray, y: np.ndarray, min_valid: int) -> np.ndarray:
    """Pearson correlation of each row pair, ignoring NaNs, fully vectorised.

    Applied to ranks this is Spearman's rho. The loop-free form matters: the
    signal families in Stages 3-5 need ~10^5 cross-sectional correlations, and
    calling scipy once per date turns a 20-second study into a 20-minute one.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    xv = np.where(valid, x, np.nan)
    yv = np.where(valid, y, np.nan)
    counts = valid.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # Dates with no investable asset are all-NaN rows; they are dropped by
        # the min_valid filter below, so their empty-slice warning is noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_x = np.nanmean(xv, axis=1, keepdims=True)
        mean_y = np.nanmean(yv, axis=1, keepdims=True)
        dx = np.where(valid, xv - mean_x, 0.0)
        dy = np.where(valid, yv - mean_y, 0.0)
        numerator = (dx * dy).sum(axis=1)
        denominator = np.sqrt((dx ** 2).sum(axis=1) * (dy ** 2).sum(axis=1))
        rho = np.where(denominator > 1e-14, numerator / denominator, np.nan)
    return np.where(counts >= min_valid, rho, np.nan)


def cross_sectional_ic(signal: pd.DataFrame, forward_returns: pd.DataFrame,
                       method: str = "spearman", min_assets: int = 5) -> pd.Series:
    """IC per date: ``corr(S_{i,t}, r_{i,t+h})`` across assets.

    Spearman by default. With the excess kurtosis documented in Stage 2, a
    Pearson IC is largely a report on which asset had the biggest move, not on
    whether the ranking was right.
    """
    signal, forward = signal.align(forward_returns, join="inner")
    if signal.empty:
        return pd.Series(dtype=float, name="ic")

    both_valid = signal.notna() & forward.notna()
    left = signal.where(both_valid)
    right = forward.where(both_valid)
    if method == "spearman":
        left = left.rank(axis=1, method="average")
        right = right.rank(axis=1, method="average")
    elif method != "pearson":
        raise ValueError("method must be 'spearman' or 'pearson'")

    rho = _rowwise_correlation(
        left.to_numpy(dtype=float), right.to_numpy(dtype=float), min_assets
    )
    return pd.Series(rho, index=signal.index, name="ic").dropna()


def time_series_ic(signal: pd.DataFrame, forward_returns: pd.DataFrame,
                   method: str = "spearman") -> pd.Series:
    """IC per asset through time: does the signal time *this* asset?"""
    signal, forward = signal.align(forward_returns, join="inner")
    out = {}
    for column in signal.columns:
        pair = pd.concat([signal[column].rename("s"), forward[column].rename("r")], axis=1).dropna()
        if len(pair) < 60:
            continue
        x, y = pair["s"].to_numpy(), pair["r"].to_numpy()
        if np.std(x) < 1e-14 or np.std(y) < 1e-14:
            continue
        rho = stats.spearmanr(x, y).statistic if method == "spearman" else np.corrcoef(x, y)[0, 1]
        out[column] = float(rho)
    return pd.Series(out, name="time_series_ic")


def ic_summary(ic: pd.Series, horizon: int = 1, breadth: int | None = None,
               periods_per_year: int = 252) -> dict:
    """Mean IC, its volatility, the information ratio of the IC, and FLAM.

    The t-statistic corrects for the overlap induced by a multi-day horizon:
    daily observations of an ``h``-day forward return contain roughly
    ``n / h`` independent observations, so the naive t-statistic is inflated by
    about ``sqrt(h)``.
    """
    clean = ic.dropna()
    if clean.empty:
        return {}
    n = len(clean)
    mean, std = float(clean.mean()), float(clean.std(ddof=1))
    independent = max(n / max(horizon, 1), 1.0)
    t_naive = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    t_adjusted = mean / (std / np.sqrt(independent)) if std > 0 else np.nan
    out = {
        "n_obs": int(n),
        "n_independent": float(independent),
        "mean_ic": mean,
        "median_ic": float(clean.median()),
        "std_ic": std,
        "ic_ir": mean / std if std > 0 else np.nan,
        "t_stat_naive": float(t_naive),
        "t_stat_overlap_adjusted": float(t_adjusted),
        "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(t_adjusted)))) if np.isfinite(t_adjusted) else np.nan,
        "hit_rate": float((clean > 0).mean()),
        "ic_skew": float(stats.skew(clean, bias=False)),
    }
    if breadth:
        # Fundamental Law of Active Management: IR = IC * sqrt(breadth).
        # Breadth is the number of independent bets per year: one rebalance
        # per horizon, times the number of assets ranked.
        rebalances = periods_per_year / max(horizon, 1)
        out["breadth_per_year"] = float(breadth * rebalances)
        out["implied_information_ratio"] = float(mean * np.sqrt(breadth * rebalances))
    return out


def rolling_ic(signal: pd.DataFrame, forward_returns: pd.DataFrame, window: int = 252,
               method: str = "spearman") -> pd.Series:
    """Rolling mean of the daily cross-sectional IC.

    Stability matters more than level. A signal whose IC is positive on
    average but negative for five years at a stretch is not tradable.
    """
    ic = cross_sectional_ic(signal, forward_returns, method)
    return ic.rolling(window, min_periods=max(window // 4, 30)).mean()


def ic_decay(signal: pd.DataFrame, returns: pd.DataFrame, horizons=(1, 5, 10, 21, 42, 63),
             method: str = "spearman", breadth: int | None = None) -> pd.DataFrame:
    """IC as a function of forecast horizon (spec §18).

    The shape of this curve, not its peak, is the useful output: it says how
    quickly the information decays and therefore what holding period is
    economically sensible. A signal that only works at one horizon is usually
    a signal that does not work.
    """
    from ..features.returns import forward_returns as make_forward

    rows = {}
    for horizon in horizons:
        forward = make_forward(returns, horizon)
        ic = cross_sectional_ic(signal, forward, method)
        summary = ic_summary(ic, horizon=horizon, breadth=breadth)
        if summary:
            rows[horizon] = summary
    out = pd.DataFrame(rows).T
    out.index.name = "horizon"
    return out


# ---------------------------------------------------------------------------
# Regression with HAC standard errors
# ---------------------------------------------------------------------------
@dataclass
class RegressionResult:
    params: pd.Series
    std_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    hac_lags: int
    note: str = ""

    def to_row(self, name: str = "beta") -> dict:
        return {
            "alpha": float(self.params.get("const", np.nan)),
            "beta": float(self.params.get(name, np.nan)),
            "se_beta": float(self.std_errors.get(name, np.nan)),
            "t_beta": float(self.t_values.get(name, np.nan)),
            "p_beta": float(self.p_values.get(name, np.nan)),
            "r_squared": self.r_squared,
            "n_obs": self.n_obs,
            "hac_lags": self.hac_lags,
        }

    def summary_line(self, name: str = "beta") -> str:
        return (
            f"beta={self.params.get(name, np.nan):+.5f} "
            f"(SE {self.std_errors.get(name, np.nan):.5f}, "
            f"t={self.t_values.get(name, np.nan):+.2f}, "
            f"p={self.p_values.get(name, np.nan):.4f}, "
            f"R2={self.r_squared:.4f}, n={self.n_obs}, HAC lags={self.hac_lags})"
        )


def newey_west_lags(horizon: int, n_obs: int | None = None) -> int:
    """Lag truncation for the HAC estimator.

    ``horizon - 1`` is the mechanical overlap. Newey-West's rule of thumb
    ``4 (n/100)^(2/9)`` is used as a floor so that genuine serial correlation
    beyond the overlap is also covered.
    """
    mechanical = max(int(horizon) - 1, 0)
    if n_obs:
        rule = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
        return max(mechanical, rule)
    return mechanical


def ols_hac(y: pd.Series, X: pd.DataFrame, hac_lags: int | None = None,
            horizon: int = 1, add_constant: bool = True) -> RegressionResult:
    """OLS with Newey-West heteroskedasticity- and autocorrelation-consistent
    standard errors (Ch. 20 §20.1.4, correcting for serial correlation)."""
    import statsmodels.api as sm

    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(frame) < 30:
        raise ValueError(f"not enough observations for regression: {len(frame)}")
    endog = frame["__y__"]
    exog = frame.drop(columns="__y__")
    if add_constant:
        exog = sm.add_constant(exog, has_constant="add")

    lags = hac_lags if hac_lags is not None else newey_west_lags(horizon, len(frame))
    model = sm.OLS(endog.to_numpy(), exog.to_numpy())
    fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    names = list(exog.columns)
    return RegressionResult(
        params=pd.Series(fitted.params, index=names),
        std_errors=pd.Series(fitted.bse, index=names),
        t_values=pd.Series(fitted.tvalues, index=names),
        p_values=pd.Series(fitted.pvalues, index=names),
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        n_obs=int(len(frame)),
        hac_lags=int(lags),
    )


def pooled_panel_regression(signal: pd.DataFrame, forward_returns: pd.DataFrame, horizon: int = 21,
                            standardise: bool = True, demean_by_date: bool = True) -> RegressionResult:
    """Pooled regression ``r_{i,t+h} = alpha + beta S_{i,t} + eps``.

    Cross-sectional demeaning of both sides removes the common market factor,
    so ``beta`` measures relative predictive power rather than the market's
    average return. Standard errors are clustered in time via the HAC
    estimator, which is what handles the cross-sectional correlation that
    makes a naive pooled t-statistic absurd.
    """
    signal, forward = signal.align(forward_returns, join="inner")
    if demean_by_date:
        signal = signal.sub(signal.mean(axis=1), axis=0)
        forward = forward.sub(forward.mean(axis=1), axis=0)
    if standardise:
        signal = signal.div(signal.std(axis=1, ddof=1).replace(0.0, np.nan), axis=0)

    stacked = pd.concat(
        [signal.stack().rename("signal"), forward.stack().rename("forward")], axis=1
    ).dropna()
    if stacked.empty:
        raise ValueError("no overlapping signal/return observations")

    # The HAC lag is chosen from the number of *dates*, not the number of
    # stacked (asset, date) rows: the dependence being corrected for runs
    # along the time axis, and inflating the sample by the cross-section
    # would make the correction far too short.
    n_dates = int(stacked.index.get_level_values(0).nunique())
    n_assets = int(stacked.index.get_level_values(1).nunique())
    result = ols_hac(
        stacked["forward"], stacked[["signal"]],
        hac_lags=newey_west_lags(horizon, n_dates), horizon=horizon,
    )
    result.note = (
        f"pooled over {n_assets} assets and {n_dates} dates; "
        f"HAC lags {result.hac_lags} for a {horizon}-day horizon"
    )
    return result


def per_asset_regression(signal: pd.DataFrame, forward_returns: pd.DataFrame,
                         horizon: int = 21) -> pd.DataFrame:
    """Time-series regression per asset, with HAC standard errors.

    Reports the fraction of assets with the hypothesised sign, which is a far
    more robust statement than one pooled coefficient: an effect present in
    two assets out of fifteen is not a multi-asset effect.
    """
    rows = {}
    for column in signal.columns:
        pair = pd.concat(
            [forward_returns[column].rename("y"), signal[column].rename("signal")], axis=1
        ).dropna()
        if len(pair) < 120:
            continue
        try:
            result = ols_hac(pair["y"], pair[["signal"]], horizon=horizon)
        except ValueError:
            continue
        rows[column] = result.to_row("signal")
    return pd.DataFrame(rows).T


def compare_standard_errors(y: pd.Series, X: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Naive OLS vs HAC standard errors on the same regression.

    This table is the argument for the HAC estimator: with overlapping
    forward returns the naive t-statistic can be inflated several-fold, which
    is exactly how an insignificant signal gets published as significant.
    """
    import statsmodels.api as sm

    frame = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    exog = sm.add_constant(frame.drop(columns="__y__"), has_constant="add")
    plain = sm.OLS(frame["__y__"].to_numpy(), exog.to_numpy()).fit()
    lags = newey_west_lags(horizon, len(frame))
    robust = plain.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)
    names = list(exog.columns)
    return pd.DataFrame(
        {
            "coefficient": pd.Series(plain.params, index=names),
            "se_naive": pd.Series(plain.bse, index=names),
            "se_hac": pd.Series(robust.bse, index=names),
            "t_naive": pd.Series(plain.tvalues, index=names),
            "t_hac": pd.Series(robust.tvalues, index=names),
            "se_inflation": pd.Series(robust.bse / plain.bse, index=names),
        }
    )


# ---------------------------------------------------------------------------
# Expected returns for the optimiser (Ch. 20 §20.1.1-20.1.3)
# ---------------------------------------------------------------------------
def signal_to_expected_returns(signal: pd.DataFrame, target_spread: float = 0.05,
                               shrinkage: float = 0.5, periods_per_year: int = 252) -> pd.DataFrame:
    """Map a standardised signal onto an annualised expected-return vector.

    Two deliberate choices, both aimed at the estimation-error problem the
    book raises in Ch. 19 §19.3:

    ``target_spread`` fixes the cross-sectional dispersion of mu rather than
    inheriting whatever scale the signal happens to have. The optimiser is
    exquisitely sensitive to this number, so it should be an explicit research
    assumption, not an accident of signal construction.

    ``shrinkage`` pulls mu towards the cross-sectional mean before it ever
    reaches the optimiser -- James-Stein in spirit: the cross-sectional mean
    is estimated far more precisely than any individual asset's mean.
    """
    centred = signal.sub(signal.mean(axis=1), axis=0)
    scaled = centred.div(signal.std(axis=1, ddof=1).replace(0.0, np.nan), axis=0)
    mu = scaled * target_spread
    if shrinkage > 0:
        mu = (1.0 - shrinkage) * mu + shrinkage * mu.mean(axis=1).to_frame().to_numpy()
    return mu / periods_per_year if periods_per_year != 1 else mu


def historical_expected_returns(returns: pd.DataFrame, lookback: int = 252,
                                shrinkage: float = 0.5) -> pd.DataFrame:
    """Trailing mean returns, shrunk towards the cross-sectional mean.

    Included as the benchmark expected-return model. Its poor performance is
    the point: Ch. 19 §19.3's estimation-error experiment is far more
    convincing when the naive alternative is in the same table.
    """
    trailing = returns.rolling(lookback, min_periods=max(lookback // 2, 60)).mean()
    grand = trailing.mean(axis=1)
    return (1.0 - shrinkage) * trailing + shrinkage * grand.to_frame().to_numpy()
