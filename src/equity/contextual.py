"""Contextual and nonlinear alpha models: when one set of weights and one straight line are not enough.

**Contextual.** A factor does not work the same way everywhere. Value may pay among stocks with low growth and not among stocks with high growth; revisions may matter most where earnings are volatile.
:func:`contextual_alpha` splits the stocks on each date by a variable known on that date (a tercile of value, growth or earnings variability), measures every factor's IC *inside each bucket*, and
weighs the factors in each bucket by their own IC covariance (the optimal alpha model of :mod:`src.equity.alpha_model`, run once per bucket). A stock's alpha is the composite its bucket's weights give it,
standardised inside the bucket so that no bucket is louder than another. If the factors really work the same way everywhere, the buckets agree and nothing is lost but some noise.

**Nonlinear.** Returns are not always a straight line in a factor. The relation of returns to capital expenditure is the textbook case: both starved and gorged firms do worse than the middle, so the
linear IC is near zero while the factor matters a great deal. :func:`quadratic`, :func:`interaction` and :func:`conditional` build the terms (a square, a product of two factors, a factor that only
counts where another is high or low), and :func:`fama_macbeth_forecast` turns any set of terms into a forecast: on every date it regresses the following returns on the terms across stocks, and the
forecast for today is the trailing average of those slopes (known by today) times today's terms.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..stats.panel import _nw_mean
from ..stats.regression import newey_west_lag
from ..signals.transform import cross_sectional_zscore
from .alpha_model import combine_factors, ic_moments, information_coefficients, max_ir_weights, standardize, standardize_all
from .fundamentals import YEAR, FactorInputs, ratio


# ------------------------------------------------------------------------------------------------------------------ contexts
def buckets(variable: pd.DataFrame, n: int = 3, investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-sectional quantile buckets of ``variable`` on every date: 0 holds the lowest ``1/n`` of the stocks, ``n-1`` the highest, missing where the variable is missing. Ties go to the lower bucket's
    neighbour by rank, so the buckets are as even as the data allow."""
    if n < 2:
        raise ValueError("n must be at least 2")
    work = variable.where(investable.reindex_like(variable).fillna(False)) if investable is not None else variable
    pct = work.rank(axis=1, pct=True, method="first")
    out = np.ceil(pct * n) - 1.0
    return out.clip(lower=0, upper=n - 1).where(work.notna())


def growth_variable(x: FactorInputs) -> pd.DataFrame:
    """Sales growth over a year."""
    sales = x.field("sales")
    return ratio(sales, sales.shift(YEAR).where(sales.shift(YEAR) > 0)) - 1.0


def earnings_variability(x: FactorInputs, years: int = 4) -> pd.DataFrame:
    """The standard deviation of earnings over assets across the last ``years`` annual observations (each as known a year apart)."""
    roa = ratio(x.field("net_income"), x.field("total_assets"))
    stack = np.stack([roa.shift(k * YEAR).to_numpy() for k in range(years)])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                     # too few observations for a standard deviation is a missing value
        sd = np.where(np.isfinite(stack).sum(axis=0) >= max(years - 1, 2), np.nanstd(stack, axis=0, ddof=1), np.nan)
    return pd.DataFrame(sd, index=roa.index, columns=roa.columns)


def context_variable(name: str, x: FactorInputs) -> pd.DataFrame:
    """A conditioning variable by name: ``value`` (book to price), ``growth`` (sales growth), ``earnings_variability`` or ``size`` (market value)."""
    if name == "value":
        return ratio(x.field("book_equity"), x.market_cap)
    if name == "growth":
        return growth_variable(x)
    if name == "earnings_variability":
        return earnings_variability(x)
    if name == "size":
        return x.market_cap
    raise KeyError(f"unknown context '{name}'; known: value, growth, earnings_variability, size")


# ------------------------------------------------------------------------------------------------------------------ contextual alpha
@dataclass
class ContextualAlpha:
    alpha: pd.DataFrame          # the composite alpha, standardised inside each bucket (or across all stocks)
    weights: dict                # bucket -> dates x factors weights used
    ic: dict                     # bucket -> dates x factors IC of the period after each date
    context: pd.DataFrame        # the bucket of every stock on every date
    factors: dict                # the standardised factors the weights apply to


def contextual_alpha(factors: dict[str, pd.DataFrame], forward: pd.DataFrame, context: pd.DataFrame, horizon: int = 1, window: int = 36, min_obs: int = 12, shrink: float = 0.3,
                     nonnegative: bool = False, investable: pd.DataFrame | None = None, min_assets: int = 8, method: str = "spearman", scale: str = "context") -> ContextualAlpha:
    """The optimal alpha model run separately in every bucket of ``context`` (a table of bucket labels, dates by assets), walk-forward.

    The weights used in a bucket on date ``t`` come from the ICs measured inside that bucket on dates up to ``t - horizon`` (their forward windows had ended); until ``min_obs`` of them exist the
    bucket uses equal weights. ``scale='context'`` standardises the composite inside each bucket on each date, ``'global'`` across all stocks."""
    if scale not in ("context", "global"):
        raise ValueError("scale must be 'context' or 'global'")
    std = standardize_all(factors, investable)
    names = list(std)
    index = std[names[0]].index
    labels = sorted(np.unique(context.to_numpy()[np.isfinite(context.to_numpy())]).tolist())
    if not labels:
        raise ValueError("the context has no buckets")
    ic_by, w_by = {}, {}
    pieces = []
    for c in labels:
        inside = (context.reindex(index) == c)
        masked = {k: v.where(inside) for k, v in std.items()}
        ic = information_coefficients(masked, forward.where(inside).reindex(index), method, min_assets).reindex(index)
        equal = pd.Series(1.0 / len(names), index=names)
        rows = {}
        for i, t in enumerate(index):
            known = ic.iloc[: max(i - horizon + 1, 0)].dropna().iloc[-window:]
            if len(known) < min_obs:
                rows[t] = equal
                continue
            mean, cov = ic_moments(known, shrink)
            rows[t] = max_ir_weights(mean, cov, nonnegative)
        weights = pd.DataFrame(rows).T.reindex(columns=names)
        composite = combine_factors({k: v.where(inside) for k, v in std.items()}, weights, renormalize=scale == "context")
        pieces.append(composite.where(inside))
        ic_by[c], w_by[c] = ic, weights
    alpha = pieces[0]
    for piece in pieces[1:]:
        alpha = alpha.combine_first(piece)
    if scale == "global":
        alpha = cross_sectional_zscore(alpha)
    return ContextualAlpha(alpha, w_by, ic_by, context, std)


# ------------------------------------------------------------------------------------------------------------------ nonlinear terms
def quadratic(factor: pd.DataFrame, investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """The square of the standardised factor, itself standardised: large where the factor is extreme in either direction, which is what an inverted U or a U in returns responds to."""
    return standardize(standardize(factor, investable) ** 2, None, 0.0)


def interaction(a: pd.DataFrame, b: pd.DataFrame, investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """The product of two standardised factors, standardised: large where both are high or both low, negative where they disagree."""
    return standardize(standardize(a, investable) * standardize(b, investable), None, 0.0)


def conditional(a: pd.DataFrame, b: pd.DataFrame, where: str = "high", n: int = 3, investable: pd.DataFrame | None = None) -> pd.DataFrame:
    """The standardised factor ``a`` where the stock is in the ``'high'`` or ``'low'`` bucket of ``b`` (out of ``n``) and zero elsewhere: an effect that only exists in part of the universe."""
    if where not in ("high", "low"):
        raise ValueError("where must be 'high' or 'low'")
    bucket = buckets(b, n, investable)
    inside = bucket == (n - 1 if where == "high" else 0)
    za = standardize(a, investable)
    return za.where(inside, 0.0).where(za.notna())


def nonlinear_features(factors: dict[str, pd.DataFrame], terms: list[tuple], investable: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """The linear factors plus the terms asked for: ``('quadratic', 'f')``, ``('interaction', 'f', 'g')`` or ``('conditional', 'f', 'g', 'high' | 'low')``. Terms are named like ``f^2``, ``f*g``, ``f|g=high``."""
    out = {k: standardize(v, investable) for k, v in factors.items()}
    for term in terms:
        kind = term[0]
        if kind == "quadratic":
            out[f"{term[1]}^2"] = quadratic(factors[term[1]], investable)
        elif kind == "interaction":
            out[f"{term[1]}*{term[2]}"] = interaction(factors[term[1]], factors[term[2]], investable)
        elif kind == "conditional":
            where = term[3] if len(term) > 3 else "high"
            out[f"{term[1]}|{term[2]}={where}"] = conditional(factors[term[1]], factors[term[2]], where, investable=investable)
        else:
            raise ValueError(f"unknown term '{kind}'; use quadratic, interaction or conditional")
    return out


@dataclass
class FamaMacBethForecast:
    forecast: pd.DataFrame      # the expected return of every stock on every date, in the units of ``forward``
    slopes: pd.DataFrame        # the cross-sectional slope of each term on each date (the return to one standard deviation of it)
    average: pd.DataFrame       # the average slope, its Newey-West t-statistic and its information ratio over all dates


def fama_macbeth_forecast(features: dict[str, pd.DataFrame], forward: pd.DataFrame, horizon: int = 1, window: int = 60, min_obs: int = 24, investable: pd.DataFrame | None = None,
                          min_assets: int | None = None) -> FamaMacBethForecast:
    """Forecast returns with the average Fama-MacBeth slopes of the past: on each date ``s`` regress ``forward[s]`` on the terms across stocks, then the forecast on date ``t`` is the mean of the slopes
    of the last ``window`` dates ``s <= t - horizon`` (their windows had ended) times the terms of ``t``. Before ``min_obs`` slopes exist the forecast is missing."""
    names = list(features)
    index, columns = forward.index, forward.columns
    X = np.stack([features[k].reindex(index=index, columns=columns).to_numpy(dtype=float) for k in names], axis=-1)
    if investable is not None:
        X = np.where(investable.reindex(index=index, columns=columns).fillna(False).to_numpy()[..., None], X, np.nan)
    Y = forward.to_numpy(dtype=float)
    min_assets = min_assets or (len(names) + 6)
    gammas = np.full((len(index), len(names) + 1), np.nan)
    for t in range(len(index)):
        ok = np.isfinite(Y[t]) & np.isfinite(X[t]).all(axis=1)
        if ok.sum() >= min_assets:
            Z = np.column_stack([np.ones(ok.sum()), X[t][ok]])
            gammas[t], *_ = np.linalg.lstsq(Z, Y[t][ok], rcond=None)
    slopes = pd.DataFrame(gammas[:, 1:], index=index, columns=names)
    rows = np.full((len(index), len(names)), np.nan)
    for t in range(len(index)):
        known = slopes.iloc[: max(t - horizon + 1, 0)].dropna().iloc[-window:]
        if len(known) >= min_obs:
            rows[t] = known.mean().to_numpy()
    mean_slope = pd.DataFrame(rows, index=index, columns=names)
    forecast = sum(features[k].reindex(index=index, columns=columns).fillna(0.0).mul(mean_slope[k], axis=0) for k in names)
    forecast = forecast.where(mean_slope.notna().all(axis=1).to_numpy()[:, None] & np.isfinite(X).any(axis=-1))
    clean = slopes.dropna()
    summary = {}
    for k in names:
        mu, se = _nw_mean(clean[k].to_numpy(), newey_west_lag(len(clean))) if len(clean) > 3 else (np.nan, np.nan)
        summary[k] = {"slope": mu, "t": mu / se if se and se > 0 else np.nan, "ir": clean[k].mean() / clean[k].std(ddof=1) if len(clean) > 3 else np.nan}
    return FamaMacBethForecast(forecast, slopes, pd.DataFrame(summary).T)
