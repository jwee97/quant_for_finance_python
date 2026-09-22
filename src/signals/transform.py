"""Signal-to-position translation (Ch. 22 §22.3.10, spec §39).

The single most important architectural rule in this project::

    forecast  !=  position

A forecast is a statement about the world: "this asset is likely to outperform".
A position is a statement about capital: "hold 7% of the book here". Conflating
them is how a strategy ends up with 60% of its risk in one asset because that
asset happened to have the largest signal value, which is a property of the
signal's scale rather than of its conviction.

The transform stack below is applied identically in research and in the
backtest, in this order::

    winsorise -> cross-sectional demean -> standardise -> clip
              -> risk scale -> constrain -> normalise leverage

Every step is causal: it uses only that date's cross-section, or trailing
data, never the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cross-sectional transforms (one date at a time)
# ---------------------------------------------------------------------------
def winsorize(signal: pd.DataFrame, quantile: float = 0.02) -> pd.DataFrame:
    """Clip each date's cross-section to its own quantiles.

    Trims the influence of an extreme signal value without deleting the
    observation -- the same principle the data stage applies to prices.
    """
    if quantile <= 0:
        return signal
    lower = signal.quantile(quantile, axis=1)
    upper = signal.quantile(1.0 - quantile, axis=1)
    return signal.clip(lower=lower, upper=upper, axis=0)


def cross_sectional_demean(signal: pd.DataFrame, weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Subtract the cross-sectional mean, making the signal relative.

    This is what converts "everything looks good" into "these look better than
    those", and it is what makes the resulting book roughly market neutral in
    signal space.
    """
    if weights is None:
        return signal.sub(signal.mean(axis=1), axis=0)
    weighted = (signal * weights).sum(axis=1) / weights.sum(axis=1)
    return signal.sub(weighted, axis=0)


def cross_sectional_zscore(signal: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Standardise each date's cross-section to mean 0, standard deviation 1."""
    centred = cross_sectional_demean(signal)
    std = signal.std(axis=1, ddof=ddof).replace(0.0, np.nan)
    return centred.div(std, axis=0)


def cross_sectional_rank(signal: pd.DataFrame, centred: bool = True) -> pd.DataFrame:
    """Percentile rank across assets, optionally centred on zero.

    Discards magnitude and keeps ordering. Given the fat tails documented in
    Stage 2, this is often the more honest representation of a signal.
    """
    ranks = signal.rank(axis=1, pct=True, method="average").where(signal.notna())
    return (ranks - 0.5) * 2.0 if centred else ranks


def tanh_squash(signal: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Bounded, monotone squash. Keeps ordering but caps conviction smoothly."""
    return np.tanh(signal / scale)


def clip_signal(signal: pd.DataFrame, limit: float = 3.0) -> pd.DataFrame:
    return signal.clip(lower=-limit, upper=limit) if limit else signal


def neutralise(signal: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Remove group means so the signal takes no net asset-class bet.

    Without this, a cross-sectional momentum signal on a multi-asset universe
    is mostly a bet on "equities over bonds", which is a risk premium, not
    alpha. Neutralising isolates the within-group timing decision.
    """
    out = signal.copy()
    labels = pd.Series({t: groups.get(t, "other") for t in signal.columns})
    for group in labels.unique():
        members = labels.index[labels == group].tolist()
        block = signal[members]
        out[members] = block.sub(block.mean(axis=1), axis=0)
    return out


# ---------------------------------------------------------------------------
# Signal -> position
# ---------------------------------------------------------------------------
def risk_scale(signal: pd.DataFrame, volatility: pd.DataFrame, target_vol: float = 0.10,
               floor: float = 1e-4) -> pd.DataFrame:
    """Convert a standardised forecast into a risk-equalised position.

    ``w_i ∝ S_i / sigma_i``: two assets with the same conviction get the same
    *risk*, not the same notional. On a universe spanning SHY (1.5% vol) and
    SLV (33%), equal notional would mean the commodity sleeve is the entire
    strategy.
    """
    vol = volatility.reindex_like(signal).clip(lower=floor)
    return signal * (target_vol / vol)


def normalise_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Rescale each date so ``sum |w| = gross``."""
    total = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(total, axis=0) * gross


def normalise_net(weights: pd.DataFrame, net: float = 1.0) -> pd.DataFrame:
    """Rescale each date so ``sum w = net`` (fully invested, long only)."""
    total = weights.sum(axis=1).replace(0.0, np.nan)
    return weights.div(total, axis=0) * net


def apply_weight_cap(weights: pd.DataFrame, max_abs: float = 0.25, gross: float | None = 1.0,
                     iterations: int = 50, tolerance: float = 1e-10) -> pd.DataFrame:
    """Cap positions at ``max_abs`` while holding gross exposure at ``gross``.

    Capping and renormalising fight each other: rescaling every weight to
    restore gross exposure pushes the capped names straight back over the cap.
    The fix is not more iterations of the same move, it is to redistribute the
    freed exposure only among the names that are *not* at the cap::

        clip -> compute the shortfall in gross exposure
             -> rescale only the uncapped names to absorb it
             -> repeat until no name breaches

    If ``n * max_abs < gross`` the request is infeasible (you cannot hold 100%
    gross in 3 names capped at 25%); the book is then capped and the gross
    exposure comes in below target rather than silently breaching the cap.
    """
    capped = weights.clip(lower=-max_abs, upper=max_abs)
    if gross is None:
        return capped

    values = capped.to_numpy(dtype=float).copy()
    finite = np.isfinite(values)
    values = np.where(finite, values, 0.0)

    for _ in range(iterations):
        magnitude = np.abs(values)
        total = magnitude.sum(axis=1)
        gap = gross - total
        if np.nanmax(np.abs(gap)) < tolerance:
            break
        at_cap = magnitude >= max_abs - 1e-12
        free_sum = np.where(at_cap, 0.0, magnitude).sum(axis=1)
        capped_sum = np.where(at_cap, magnitude, 0.0).sum(axis=1)
        target_free = gross - capped_sum
        # Rows with no room left (everything at the cap, or nothing to scale)
        # are infeasible: leave them capped rather than breaching.
        scalable = (free_sum > 1e-12) & (target_free > 0.0)
        factor = np.ones_like(free_sum)
        factor[scalable] = target_free[scalable] / free_sum[scalable]
        values = np.where(at_cap, values, values * factor[:, None])
        values = np.clip(values, -max_abs, max_abs)

    out = pd.DataFrame(values, index=weights.index, columns=weights.columns)
    return out.where(finite, np.nan)


def long_only(weights: pd.DataFrame) -> pd.DataFrame:
    """Drop short positions and renormalise the remaining longs."""
    positive = weights.clip(lower=0.0)
    return normalise_net(positive, 1.0)


def signal_to_positions(
    signal: pd.DataFrame,
    volatility: pd.DataFrame | None = None,
    *,
    winsorize_quantile: float = 0.02,
    cross_sectional: bool = True,
    scale: str = "zscore",
    clip: float | None = 3.0,
    groups: dict[str, str] | None = None,
    target_vol: float = 0.10,
    max_abs_weight: float = 0.25,
    gross_leverage: float = 1.0,
    long_only_book: bool = False,
    investable: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full forecast -> position stack.

    Returns weights indexed exactly like ``signal``. Dates on which no asset
    is investable produce a flat (all-zero) book rather than NaN, so the
    backtest can distinguish "no position" from "missing data".
    """
    work = signal.copy()
    if investable is not None:
        work = work.where(investable.reindex_like(work).fillna(False))

    work = winsorize(work, winsorize_quantile)
    if groups:
        work = neutralise(work, groups)
    if cross_sectional:
        work = cross_sectional_demean(work)

    if scale == "zscore":
        work = cross_sectional_zscore(work)
    elif scale == "rank":
        work = cross_sectional_rank(work)
    elif scale == "tanh":
        work = tanh_squash(work)
    elif scale not in {"none", None}:
        raise ValueError(f"unknown scale '{scale}'")

    work = clip_signal(work, clip)

    if volatility is not None:
        work = risk_scale(work, volatility, target_vol)

    if long_only_book:
        work = long_only(work)
        work = apply_weight_cap(work, max_abs_weight, gross=None)
        work = normalise_net(work, 1.0)
    else:
        work = normalise_gross(work, gross_leverage)
        work = apply_weight_cap(work, max_abs_weight, gross_leverage)

    return work.fillna(0.0)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """``TO_t = sum_i |w_{i,t} - w_{i,t-1}|`` (spec §22)."""
    return weights.diff().abs().sum(axis=1)


def effective_number_of_positions(weights: pd.DataFrame) -> pd.Series:
    """Inverse Herfindahl of the weight vector: how concentrated is the book?"""
    normalised = weights.abs().div(weights.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    return 1.0 / (normalised ** 2).sum(axis=1)
