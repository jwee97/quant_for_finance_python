"""Helpers shared by the strategies: point-in-time volatility, continuous futures histories and risk-based sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ann_vol(ctx, instrument_id: str, window: int = 60, floor: float = 0.01) -> float:
    """Annualised volatility of log returns over the last ``window`` observations known now (``nan`` if too few)."""
    r = ctx.data.returns(instrument_id, window) if not ctx.registry.is_chain(instrument_id) else continuous_history(ctx, instrument_id, window + 1).pipe(lambda s: np.log(s).diff().dropna())
    if len(r) < max(10, window // 3):
        return float("nan")
    return float(max(r.std(ddof=1) * np.sqrt(252.0), floor))


def continuous_history(ctx, chain_id: str, n: int = 260) -> pd.Series:
    """A back-adjusted continuous price series for a future chain, from the contracts' own point-in-time histories.

    On each date the series follows the contract the chain's roll rule designates (the first contract whose roll date has not passed); at a switch the earlier prices are scaled by
    ``new / old`` on the switch date, so percentage returns are right and no roll gap appears. Only data known now are used; the result is memoised for the current engine time."""
    cache = ctx.data.__dict__.setdefault("_chain_cache", {})
    key = (chain_id, ctx.ts)
    if cache.get("key") != ctx.ts:
        cache.clear()
        cache["key"] = ctx.ts
    if key not in cache:
        cache[key] = _continuous(ctx, chain_id)
    return cache[key].tail(n)


def _continuous(ctx, chain_id: str) -> pd.Series:
    chain = ctx.registry.chain(chain_id)
    daily = {}
    for c in chain.contracts:
        s = ctx.data.history(c.instrument_id, None)
        if len(s):
            daily[c.instrument_id] = s.groupby(s.index.normalize()).last()
    if not daily:
        return pd.Series(dtype=float)
    df = pd.DataFrame(daily).sort_index()
    ids = [c.instrument_id for c in chain.contracts if c.instrument_id in df.columns]
    df = df[ids]
    roll = np.array([chain.roll_date(ctx.registry.get(i)).value for i in ids])
    dates = df.index.to_numpy().astype("datetime64[ns]").astype("int64")
    px = df.to_numpy(dtype=float)
    valid = (roll[None, :] >= dates[:, None]) & np.isfinite(px)
    has = valid.any(axis=1)
    pick = valid.argmax(axis=1)
    rows = np.arange(len(df))
    level = np.where(has, px[rows, pick], np.nan)
    out = pd.Series(level, index=df.index).dropna()
    pick = pd.Series(pick, index=df.index)[out.index].to_numpy()
    adj = out.to_numpy().copy()
    pos = {d: k for k, d in enumerate(df.index)}
    for k in np.nonzero(pick[1:] != pick[:-1])[0] + 1:
        old = df.iloc[pos[out.index[k]], pick[k - 1]]
        if np.isfinite(old) and old > 0:
            adj[:k] *= out.iloc[k] / old
    return pd.Series(adj, index=out.index)


def risk_weights(signals: dict[str, float], vols: dict[str, float], target_vol: float, max_leverage: float = 3.0, max_weight: float = 1.0) -> dict[str, float]:
    """Weights (fractions of capital) proportional to ``signal / vol`` scaled so the sum of ``|w| * vol`` equals ``target_vol`` shared equally, with caps on each weight and on
    leverage. Instruments with no volatility estimate are skipped."""
    ids = [i for i, s in signals.items() if np.isfinite(vols.get(i, np.nan)) and vols[i] > 0]
    if not ids:
        return {}
    n = len(ids)
    w = {i: float(np.clip(signals[i] * (target_vol / n) / vols[i], -max_weight, max_weight)) for i in ids}
    gross = sum(abs(v) for v in w.values())
    if gross > max_leverage:
        w = {i: v * max_leverage / gross for i, v in w.items()}
    return w
