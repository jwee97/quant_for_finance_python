"""Market-impact execution costs, capacity and no-trade bands (Generation 3, Priority 7).

Generation 1 charged a per-asset LINEAR rate (spread and commission) on every
dollar traded and said so: with no impact term, the high-turnover books are
flattered. This module adds the missing term. The cost per dollar of capital
traded in asset ``i`` on day ``t`` is::

    linear_i + Y * sigma_{i,t} * sqrt( min(|dw_{i,t}| * AUM / ADV_{i,t}, cap) )

``sigma`` is the 21-day daily volatility and ``ADV`` the 63-day average dollar
volume, both measured through the PREVIOUS close, so nothing in the cost of
trading on day ``t`` uses day ``t`` data. ``|dw|`` is the day's weight change.
As ``AUM -> 0`` the second term vanishes and the model IS the Generation 1
linear model, which ``tests/test_impact.py`` pins to machine precision.

The square-root law (impact per dollar proportional to volatility times the
square root of participation) is the standard empirical regularity for
metaorders in listed markets; ``Y`` is its order-one coefficient and is a
sensitivity here, not an estimate. Nothing in this module was fitted to the
ETFs in the universe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANN = 252.0


@dataclass(frozen=True)
class ImpactSettings:
    coefficient: float = 1.0
    max_participation: float = 1.0


def market_state(close: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame,
                 vol_window: int = 21, adv_window: int = 63) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily volatility and average daily dollar volume, each through the previous close."""
    columns = returns.columns
    dollar = (close.reindex(returns.index).reindex(columns=columns)
              * volume.reindex(returns.index).reindex(columns=columns))
    adv = dollar.rolling(adv_window, min_periods=max(adv_window // 2, 10)).mean().shift(1)
    sigma = returns.rolling(vol_window, min_periods=max(vol_window // 2, 10)).std(ddof=1).shift(1)
    return sigma, adv


def _fill_cross_section(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace missing entries by that day's cross-sectional median; report how many were replaced."""
    median = frame.median(axis=1)
    missing = frame.isna()
    filled = frame.where(~missing, pd.DataFrame(np.repeat(median.to_numpy()[:, None], frame.shape[1], axis=1),
                                                index=frame.index, columns=frame.columns))
    return filled, int(missing.to_numpy().sum())


def impact_cost_frame(trades: pd.DataFrame, aum: float, sigma: pd.DataFrame, adv: pd.DataFrame,
                      linear_rates: pd.Series, settings: ImpactSettings = ImpactSettings()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-asset cost as a fraction of capital, and the participation rate behind it.

    Returns ``(cost, participation)`` where participation is the UNCAPPED ``|dw| * AUM / ADV``.
    Missing volatility or volume is filled by the day's cross-sectional median (counted by
    ``fill_counts``), so a gap never makes a trade free.
    """
    abs_trades = trades.abs()
    linear = abs_trades * linear_rates.reindex(trades.columns).to_numpy()[None, :]
    if aum <= 0.0:
        return linear, abs_trades * 0.0
    sig = sigma.reindex(index=trades.index, columns=trades.columns)
    dollar_volume = adv.reindex(index=trades.index, columns=trades.columns)
    sig, _ = _fill_cross_section(sig)
    dollar_volume, _ = _fill_cross_section(dollar_volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        participation = abs_trades * aum / dollar_volume.where(dollar_volume > 0)
    participation = participation.fillna(0.0)
    capped = participation.clip(upper=settings.max_participation)
    impact = settings.coefficient * sig.fillna(0.0) * np.sqrt(capped) * abs_trades
    return linear + impact, participation


def fill_counts(trades: pd.DataFrame, sigma: pd.DataFrame, adv: pd.DataFrame) -> int:
    """Number of traded asset-days whose volatility or volume had to be filled from the cross-section."""
    traded = trades.abs() > 1e-12
    missing = (sigma.reindex_like(trades).isna() | adv.reindex_like(trades).isna()) & traded
    return int(missing.to_numpy().sum())


def impact_net_returns(gross: pd.Series, trades: pd.DataFrame, aum: float, sigma: pd.DataFrame, adv: pd.DataFrame,
                       linear_rates: pd.Series, settings: ImpactSettings = ImpactSettings()) -> tuple[pd.Series, pd.Series, dict]:
    """Net returns under the impact model, the daily cost series and participation statistics."""
    cost, participation = impact_cost_frame(trades, aum, sigma, adv, linear_rates, settings)
    daily = cost.sum(axis=1).reindex(gross.index).fillna(0.0)
    traded = trades.abs() > 1e-12
    part = participation.where(traded).stack()
    stats = {
        "mean_participation": float(part.mean()) if len(part) else 0.0,
        "p95_participation": float(part.quantile(0.95)) if len(part) else 0.0,
        "max_participation": float(part.max()) if len(part) else 0.0,
        "share_capped": float((part > settings.max_participation).mean()) if len(part) else 0.0,
        "traded_asset_days": int(traded.to_numpy().sum()),
    }
    return gross - daily, daily, stats


def sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    sd = clean.std(ddof=1)
    return float(ANN ** 0.5 * clean.mean() / sd) if len(clean) > 2 and sd > 0 else float("nan")


def capacity_curve(gross: pd.Series, trades: pd.DataFrame, grid, sigma: pd.DataFrame, adv: pd.DataFrame,
                   linear_rates: pd.Series, settings: ImpactSettings = ImpactSettings()) -> pd.Series:
    """Net Sharpe at each AUM in ``grid``."""
    return pd.Series({float(a): sharpe(impact_net_returns(gross, trades, float(a), sigma, adv, linear_rates, settings)[0]) for a in grid})


def capacity_from_curve(curve: pd.Series, linear_sharpe: float) -> float | str:
    """AUM at which the curve falls to half the linear-cost Sharpe (log-linear interpolation).

    Returns ``"n/a"`` when the linear-cost Sharpe is not positive, ``"below grid"`` when the book is already under
    half at the smallest AUM, and ``"above grid"`` when it never gets there.
    """
    if not np.isfinite(linear_sharpe) or linear_sharpe <= 0:
        return "n/a"
    target = 0.5 * linear_sharpe
    aum = curve.index.to_numpy(dtype=float)
    values = curve.to_numpy(dtype=float)
    if values[0] <= target:
        return "below grid"
    for k in range(1, len(values)):
        if values[k] <= target:
            frac = (values[k - 1] - target) / (values[k - 1] - values[k])
            return float(np.exp(np.log(aum[k - 1]) + frac * (np.log(aum[k]) - np.log(aum[k - 1]))))
    return "above grid"


def banded_execution(book_weights: pd.DataFrame, trades: pd.DataFrame, returns: pd.DataFrame,
                     band: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-execute an engine book with a per-asset no-trade band.

    ``book_weights`` and ``trades`` are the engine's ``held`` weights and executed trades. On a rebalance
    date (a row with any trade, or the first row the engine holds anything, even if that is an all-cash book)
    the engine's held row IS the target. The banded book holds each asset's
    executed (drifted) weight unless the target differs from it by MORE than ``band``, in which case it
    trades to the target. Between rebalances it drifts with returns exactly as the engine does, preserving
    gross exposure. ``band = 0`` reproduces the engine's trades.

    Returns ``(held, trades)`` with the same timing as the engine: ``held[t]`` is in force at the close of
    ``t`` and earns the return of ``t+1``.
    """
    index, columns = book_weights.index, book_weights.columns
    targets = book_weights.to_numpy(dtype=float)
    traded_in = trades.reindex(index).fillna(0.0).abs().sum(axis=1).to_numpy() > 1e-12
    ret = returns.reindex(index).reindex(columns=columns).fillna(0.0).to_numpy(dtype=float)
    held = np.full_like(targets, np.nan)
    traded = np.zeros_like(targets)
    current = None
    for i in range(len(index)):
        if current is not None:
            grown = current * (1.0 + ret[i])
            total = np.abs(grown).sum()
            gross = np.abs(current).sum()
            current = grown * (gross / total) if total > 1e-12 else grown
        if (traded_in[i] or current is None) and np.isfinite(targets[i]).any():
            target = np.nan_to_num(targets[i], nan=0.0)
            if current is None:
                new = target
            else:
                new = np.where(np.abs(target - current) > band, target, current)
            traded[i] = new - (current if current is not None else 0.0)
            current = new
        if current is not None:
            held[i] = current
    held_frame = pd.DataFrame(held, index=index, columns=columns)
    return held_frame, pd.DataFrame(traded, index=index, columns=columns)


def banded_gross_returns(held: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """``R_t = w_{t-1}' r_t`` for a held book (the engine's own convention)."""
    aligned = returns.reindex(held.index).reindex(columns=held.columns)
    gross = (held.shift(1) * aligned).sum(axis=1, skipna=True)
    return gross.where(held.shift(1).notna().any(axis=1) & aligned.notna().any(axis=1))


def scheduling_table(trade_weight: float, aum: float, sigma: float, adv: float, days=(1, 2, 5, 10),
                     coefficient: float = 1.0, linear_rate: float = 0.0) -> pd.DataFrame:
    """Almgren-Chriss style comparison for one trade of ``trade_weight`` (fraction of capital) in one asset.

    Spreading the order evenly over ``k`` days cuts the per-day participation by ``k``, so temporary impact
    per dollar falls by ``sqrt(k)``; the cost is timing risk, ``sigma * sqrt(k) * |trade| / sqrt(3)`` as the
    one-standard-deviation price move on the unexecuted part under even slicing (a standard small-trade
    approximation). Descriptive only.
    """
    rows = []
    for k in days:
        participation = abs(trade_weight) * aum / adv / k
        impact = coefficient * sigma * np.sqrt(min(participation, 1.0))
        rows.append({
            "days": k, "participation_per_day": participation,
            "cost_bps_of_trade": 1e4 * (linear_rate + impact),
            "timing_risk_bps_of_trade": 1e4 * sigma * np.sqrt((k - 1) * (2 * k - 1) / (6.0 * k)) if k > 1 else 0.0,
        })
    return pd.DataFrame(rows).set_index("days")
