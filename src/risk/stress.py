"""Stress testing and scenario analysis (Ch. 21 §21.2.5, spec §37).

Two complementary exercises, and the difference matters:

**Historical regimes** replay what actually happened. The strength is that
every correlation, every gap and every liquidity effect is real; the weakness
is that the next crisis will not be a repeat of the last one.

**Hypothetical scenarios** apply a shock vector to today's book. The strength
is that you can ask about a move that has never happened; the weakness is that
you have to invent the correlation structure, and inventing it optimistically
is how stress tests become theatre.

Regime windows are defined ex ante in ``config/risk.yaml`` from published
macro chronology -- never from the strategy's own drawdowns, which would
guarantee a flattering answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.returns import cumulative_returns, max_drawdown
from ..utils.dates import DateWindow, windows_from_config
from .cvar import historical_cvar
from .var import historical_var


def regime_statistics(returns: pd.Series, window: DateWindow, alpha: float = 0.95,
                      periods_per_year: int = 252) -> dict:
    """Performance and risk inside one named window."""
    block = window.apply(returns).dropna()
    if len(block) < 5:
        return {}
    curve = cumulative_returns(block)
    vol = float(block.std(ddof=1) * np.sqrt(periods_per_year))
    return {
        "regime": window.label,
        "start": window.start.date().isoformat() if window.start is not None else "",
        "end": window.end.date().isoformat() if window.end is not None else "",
        "n_days": int(len(block)),
        "total_return": float(curve.iloc[-1] - 1.0),
        "ann_return": float(block.mean() * periods_per_year),
        "ann_vol": vol,
        "sharpe": float(block.mean() * periods_per_year / vol) if vol > 0 else np.nan,
        "max_drawdown": max_drawdown(curve),
        "worst_day": float(block.min()),
        "best_day": float(block.max()),
        "var_95": historical_var(block, alpha),
        "cvar_95": historical_cvar(block, alpha),
        "hit_rate": float((block > 0).mean()),
    }


def regime_table(strategies: dict[str, pd.Series], regimes: dict[str, DateWindow],
                 alpha: float = 0.95) -> pd.DataFrame:
    """Every strategy through every regime."""
    rows = []
    for name, series in strategies.items():
        for key, window in regimes.items():
            stats = regime_statistics(series, window, alpha)
            if stats:
                stats.update({"strategy": name, "regime_key": key})
                rows.append(stats)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    front = ["strategy", "regime", "start", "end", "n_days", "total_return", "ann_vol",
             "max_drawdown", "var_95", "cvar_95"]
    return frame.loc[:, front + [c for c in frame.columns if c not in front]]


def regime_correlations(returns: pd.DataFrame, regimes: dict[str, DateWindow]) -> pd.DataFrame:
    """Average pairwise correlation inside each regime.

    The number that decides whether a diversified book stays diversified.
    """
    rows = {}
    n = returns.shape[1]
    for key, window in regimes.items():
        block = window.apply(returns).dropna(how="any")
        if len(block) < 10:
            continue
        corr = block.corr().to_numpy()
        off_diagonal = corr[~np.eye(n, dtype=bool)]
        rows[window.label] = {
            "n_days": int(len(block)),
            "mean_correlation": float(np.nanmean(off_diagonal)),
            "max_correlation": float(np.nanmax(off_diagonal)),
            "min_correlation": float(np.nanmin(off_diagonal)),
        }
    full = returns.dropna(how="any").corr().to_numpy()
    rows["Full sample"] = {
        "n_days": int(len(returns.dropna(how="any"))),
        "mean_correlation": float(np.nanmean(full[~np.eye(n, dtype=bool)])),
        "max_correlation": float(np.nanmax(full[~np.eye(n, dtype=bool)])),
        "min_correlation": float(np.nanmin(full[~np.eye(n, dtype=bool)])),
    }
    return pd.DataFrame(rows).T


def apply_scenario(weights: pd.Series, shocks: dict[str, float],
                   asset_class: dict[str, str]) -> dict:
    """Instantaneous P&L of a book under a shock vector by asset class."""
    per_asset = {}
    for asset, weight in weights.items():
        key = asset_class.get(asset, "")
        shock = shocks.get(key, shocks.get(asset, 0.0))
        per_asset[asset] = float(weight) * float(shock)
    total = float(sum(per_asset.values()))
    contributions = pd.Series(per_asset)
    return {
        "total_pnl": total,
        "worst_contributor": str(contributions.idxmin()) if len(contributions) else "",
        "worst_contribution": float(contributions.min()) if len(contributions) else np.nan,
        "contributions": contributions,
    }


def scenario_table(books: dict[str, pd.Series], scenarios: dict, asset_class: dict[str, str]) -> pd.DataFrame:
    """Every book under every hypothetical scenario."""
    rows = []
    for book_name, weights in books.items():
        for key, spec in (scenarios or {}).items():
            result = apply_scenario(weights, spec.get("shocks", {}), asset_class)
            rows.append(
                {
                    "book": book_name,
                    "scenario": spec.get("label", key),
                    "pnl": result["total_pnl"],
                    "worst_contributor": result["worst_contributor"],
                    "worst_contribution": result["worst_contribution"],
                }
            )
    return pd.DataFrame(rows)


def worst_windows(returns: pd.Series, window: int = 21, top: int = 5) -> pd.DataFrame:
    """The worst rolling windows in the sample, found rather than assumed.

    A useful complement to the named regimes: if the strategy's worst month is
    not one of the regimes we defined, the regime list has a blind spot.
    """
    rolling = returns.rolling(window).sum()
    worst = rolling.nsmallest(top * 10).sort_index()
    # Keep only non-overlapping episodes.
    selected, last = [], None
    for stamp, value in worst.sort_values().items():
        if last is not None and any(abs((stamp - s).days) < window for s, _ in selected):
            continue
        selected.append((stamp, value))
        last = stamp
        if len(selected) >= top:
            break
    return pd.DataFrame(
        [{"window_end": s.date().isoformat(), "cumulative_return": float(v)} for s, v in selected]
    )


def load_regimes(config) -> dict[str, DateWindow]:
    return windows_from_config(config.get("risk.stress.regimes", {}) or {})
