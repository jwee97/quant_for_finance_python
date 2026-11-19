"""Attribution into selection, sizing and timing, and the simulation of stops and leverage rules.

Paleologo, *Advanced Portfolio Management*, ch. 8. With holdings ``w`` (dates x assets) and returns ``r``, the profit of date ``t`` is ``sum_i w_(t-1,i) r_(t,i) = G_t sum_i u_(t,i) r_(t,i)`` with ``G_t`` the gross
exposure ``sum |w|`` and ``u`` the position as a share of it. Compare it with a portfolio that holds the same names and sides in EQUAL size (``v = sign(w)/n_t``) at the AVERAGE gross ``G``:

    selection  =  G sum_i v_(t,i) r_(t,i)                  the names and the sides, with nothing else
    sizing     =  G sum_i (u_(t,i) - v_(t,i)) r_(t,i)      the effect of making some bets bigger than others
    timing     =  (G_t - G) sum_i u_(t,i) r_(t,i)          the effect of being more or less invested at different times

The three add to the profit exactly. Timing is not market timing in the usual sense (that needs the exposure to a factor): it is the contribution of the variation of the book's size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def selection_sizing_timing(weights: pd.DataFrame, returns: pd.DataFrame, lag: int = 1) -> dict:
    """The decomposition above. ``weights`` at date ``t`` earn the return of ``t + lag`` (``lag = 1``: positions set at the close of ``t`` earn the next day's return). Returns the per-date components and
    a summary (total, share of the total, annualised information ratio of each, the hit rate of the sides picked)."""
    w, r = weights.align(returns, join="inner")
    w = w.shift(int(lag))
    ok = w.notna().all(axis=1) & r.notna().all(axis=1)
    w, r = w[ok].fillna(0.0), r[ok]
    if len(w) < 20:
        raise ValueError("need at least 20 dates with holdings and returns")
    G_t = w.abs().sum(axis=1)
    live = G_t > 0
    u = w.div(G_t.where(live), axis=0).fillna(0.0)
    n_t = (w != 0).sum(axis=1)
    v = np.sign(w).div(n_t.where(n_t > 0), axis=0).fillna(0.0)
    Gbar = float(G_t[live].mean())
    sel = Gbar * (v * r).sum(axis=1)
    siz = Gbar * ((u - v) * r).sum(axis=1)
    tim = (G_t - Gbar).where(live, 0.0) * (u * r).sum(axis=1)
    total = (w * r).sum(axis=1)
    comp = pd.DataFrame({"selection": sel, "sizing": siz, "timing": tim, "total": total})
    comp["residual"] = comp["total"] - comp[["selection", "sizing", "timing"]].sum(axis=1)
    side_hit = float(((np.sign(w) * np.sign(r)) > 0).where(w != 0).stack().mean()) if (w != 0).any().any() else float("nan")
    summary = {}
    for c in ("selection", "sizing", "timing", "total"):
        s = comp[c]
        summary[c] = {"sum": float(s.sum()), "share_of_total": float(s.sum() / comp["total"].sum()) if comp["total"].sum() != 0 else float("nan"), "ir": float(s.mean() / s.std(ddof=1) * np.sqrt(ANN)) if s.std(ddof=1) > 0 else float("nan"),
                      "mean_per_year": float(s.mean() * ANN)}
    return {"components": comp, "summary": pd.DataFrame(summary).T, "hit_rate": side_hit, "mean_gross": Gbar, "max_residual": float(comp["residual"].abs().max())}


def stop_loss(returns: pd.Series, limit: float, scale: float = 0.0, cooloff: int = 21) -> dict:
    """A strategy-level stop. When the cumulative loss from the running peak of the *managed* equity reaches ``limit`` (a fraction of capital), cut exposure to ``scale`` (0 = flat) for ``cooloff`` periods,
    then return to full size with the peak reset. Exposure for date ``t`` is decided from the equity at ``t - 1``, so no rule uses the return it is applied to. Returns the managed returns, the
    exposure and the number of stops. A stop does not create edge: on a random walk it trades drawdown against time out of the market."""
    r = returns.fillna(0.0).to_numpy(dtype=float)
    expo = np.ones(len(r))
    eq, peak, off, stops = 1.0, 1.0, 0, 0
    out = np.zeros(len(r))
    for t in range(len(r)):
        if off > 0:
            expo[t] = scale
            off -= 1
            if off == 0:
                peak = eq
        out[t] = expo[t] * r[t]
        eq *= 1.0 + out[t]
        peak = max(peak, eq)
        if off == 0 and eq / peak - 1.0 <= -limit:
            off, stops = int(cooloff), stops + 1
    s = pd.Series(out, index=returns.index)
    return {"returns": s, "exposure": pd.Series(expo, index=returns.index), "stops": stops, "max_drawdown": _mdd(s), "unmanaged_max_drawdown": _mdd(returns.fillna(0.0)), "time_out": float((expo < 1).mean())}


def volatility_target(returns: pd.Series, target: float = 0.10, lookback: int = 63, max_leverage: float = 3.0, periods_per_year: float = ANN) -> dict:
    """Scale each day's exposure to ``target`` annualised volatility on the volatility measured over the previous ``lookback`` days (never including the day it is applied to), capped at ``max_leverage``."""
    vol = returns.rolling(lookback, min_periods=max(lookback // 2, 10)).std(ddof=1).shift(1) * np.sqrt(periods_per_year)
    lev = (target / vol).clip(upper=max_leverage).fillna(0.0)
    out = lev * returns
    live = lev > 0
    return {"returns": out, "leverage": lev, "sharpe": _sharpe(out[live]), "unmanaged_sharpe": _sharpe(returns[live]), "max_drawdown": _mdd(out[live]), "unmanaged_max_drawdown": _mdd(returns[live]),
            "realised_vol": float(out[live].std(ddof=1) * np.sqrt(periods_per_year)) if live.sum() > 2 else float("nan")}


def _sharpe(r: pd.Series) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(ANN)) if len(r) > 2 and s > 0 else float("nan")


def _mdd(r: pd.Series) -> float:
    eq = (1.0 + r).cumprod()
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else float("nan")
