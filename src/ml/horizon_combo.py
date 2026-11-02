"""Combining short-, medium- and long-horizon signals with horizon-aware units, dependence and turnover controls (spec v3, Release 8, P3-05). Optional: a single-horizon workflow does not use it.

* **Units.** A forecast of the return over ``h`` sessions is ``h`` times a per-session forecast. Signals of different horizons are put on one scale, the per-session expected return ``score / h``,
  before anything is combined; adding a 20-session forecast to a 1-session one as if they were the same unit would let the long horizon dominate by a factor of twenty.
* **Weights across horizons.** ``equal``, or ``ic``: proportional to each horizon's mean rank IC over a trailing window of decisions whose labels have **settled** (a decision of horizon ``h`` is known
  to have settled ``lag + h`` sessions later), floored at zero; when no horizon has shown a positive settled IC the weights are zero and the book is flat, not equal-weighted by default.
* **Dependence.** Signals of nearby horizons are correlated; the report gives their correlation and the effective number of independent signals (the participation ratio of the correlation matrix's
  eigenvalues) so that three horizons are not counted as three independent views.
* **Two ways to build the book, neither of which can count an exposure twice.** ``score``: blend the per-session scores into one and build one book with one holding period. ``sleeves``: build each
  horizon's own book with its own holding period (the platform's tranche averaging: ``h`` sleeves of ``1/h`` of the horizon's capital) and add them with capital shares that sum to one, so the combined
  gross is at most the shares-weighted gross of the sleeves and never their sum.
* **Turnover control.** A cap on the one-way turnover of each step partially adjusts the book toward its target (``w_t = w_{t-1} + theta (w*_t - w_{t-1})``), so a fast signal cannot trade the long
  horizon's book away.
* **The comparison** is with each single-horizon book and with the best single horizon, through the same engine and costs, and by paired Sharpe differences with intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import alpha as al
from . import inference as inf
from .backtest import run_backtest
from .signals import build_weights
from .targets import forward_return

VERSION = "horizon-combo-v1"
BOOKS = ("score", "sleeves")
WEIGHTS = ("equal", "ic")


def per_session(scores_by_h: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Each horizon's return forecast divided by its horizon: the per-session expected return."""
    if not scores_by_h or any(int(h) < 1 for h in scores_by_h):
        raise ValueError("give at least one signal, each with a horizon of at least one session")
    return {int(h): s / float(h) for h, s in scores_by_h.items()}


def dependence(scores_by_h: dict[int, pd.DataFrame], min_assets: int = 5) -> dict:
    """The mean cross-sectional correlation between the per-session signals of every pair of horizons, and the effective number of independent signals (participation ratio of the eigenvalues)."""
    ps = per_session(scores_by_h)
    hs = sorted(ps)
    m = len(hs)
    corr = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = ps[hs[i]], ps[hs[j]]
            both = a.notna() & b.notna()
            ar, br = a.where(both).rank(axis=1), b.where(both).rank(axis=1)
            ac, bc = ar.sub(ar.mean(axis=1), axis=0), br.sub(br.mean(axis=1), axis=0)
            den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
            c = ((ac * bc).sum(axis=1) / den.where(den > 1e-12))[both.sum(axis=1) >= min_assets]
            corr[i, j] = corr[j, i] = float(c.mean()) if c.notna().any() else 0.0
    ev = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    return {"horizons": hs, "correlation": pd.DataFrame(corr, index=hs, columns=hs), "effective_independent_signals": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "note": "the participation ratio of the correlation matrix: three identical signals count as one, three independent ones as three"}


def ic_by_horizon(scores_by_h: dict[int, pd.DataFrame], prices: pd.DataFrame, lag: int, min_assets: int = 5) -> dict[int, pd.Series]:
    """The daily rank IC of each horizon's signal against the return over its own horizon, starting ``lag`` sessions after the decision."""
    out = {}
    for h, s in scores_by_h.items():
        fwd = forward_return(prices, int(h), int(lag)).reindex(index=s.index, columns=s.columns)
        out[int(h)] = al.ic_table(s, fwd, min_assets)["ic"]
    return out


def horizon_weights(ic: dict[int, pd.Series], lag: int, method: str = "equal", window: int = 252, min_dates: int = 40) -> pd.DataFrame:
    """The weight of each horizon on each decision date. ``ic`` uses only decisions settled by the date (``position + lag + h <= date``); the result is a date x horizon frame whose rows sum to one, or to
    zero where no horizon has a positive settled mean IC."""
    hs = sorted(ic)
    index = ic[hs[0]].index
    if method not in WEIGHTS:
        raise ValueError(f"the weighting is one of {list(WEIGHTS)}")
    if method == "equal":
        return pd.DataFrame(1.0 / len(hs), index=index, columns=hs)
    stats = {}
    for h in hs:
        settled = ic[h].shift(int(lag) + int(h))                                                    # the IC of decision s is known from session s + lag + h on
        stats[h] = settled.rolling(int(window), min_periods=int(min_dates)).mean()
    m = pd.DataFrame(stats).clip(lower=0.0)
    total = m.sum(axis=1)
    return m.div(total.where(total > 0), axis=0).fillna(0.0)


def cap_turnover(target: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    """Move toward each target by at most ``cap`` of one-way turnover per step: ``w_t = w_{t-1} + theta (w*_t - w_{t-1})`` with ``theta = min(1, cap / one-way distance)``."""
    if cap is None:
        return target
    if cap <= 0:
        raise ValueError("the turnover cap is positive")
    out = np.zeros(target.shape)
    prev = np.zeros(target.shape[1])
    for i, row in enumerate(target.to_numpy(dtype=float)):
        step = row - prev
        dist = 0.5 * np.abs(step).sum()
        theta = 1.0 if dist <= cap else cap / dist
        prev = prev + theta * step
        out[i] = prev
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def build(scores_by_h: dict[int, pd.DataFrame], signal_cfg: dict, *, prices: pd.DataFrame, lag: int = 1, book: str = "sleeves", weighting: str = "equal", window: int = 252,
          turnover_cap: float | None = None, investable: pd.DataFrame | None = None, returns: pd.DataFrame | None = None) -> dict:
    """The combined book and the evidence of how it was built. ``signal_cfg`` is the ``signal`` section of an experiment (the position rule and its limits); its ``holding_period`` is replaced by each
    horizon's own in ``sleeves`` mode and by the weighted average horizon in ``score`` mode."""
    if book not in BOOKS:
        raise ValueError(f"the book is one of {list(BOOKS)}")
    ps = per_session(scores_by_h)
    hs = sorted(ps)
    ic = ic_by_horizon(scores_by_h, prices, lag)
    w = horizon_weights(ic, lag, weighting, window).reindex(ps[hs[0]].index).fillna(0.0)
    if book == "score":
        blended = sum(ps[h].mul(w[h], axis=0) for h in hs)
        hold = max(1, int(round(float((w * np.array(hs)).sum(axis=1).replace(0.0, np.nan).mean())))) if (w.sum(axis=1) > 0).any() else 1
        target = build_weights(blended, {**signal_cfg, "holding_period": hold}, investable=investable, returns=returns)
        sleeves = None
    else:
        sleeves = {h: build_weights(ps[h], {**signal_cfg, "holding_period": h}, investable=investable, returns=returns) for h in hs}
        target = sum(sleeves[h].mul(w[h], axis=0) for h in hs)                                         # capital shares: the weights already sum to one (or to zero when no horizon has an edge)
        hold = None
    final = cap_turnover(target, turnover_cap)
    return {"version": VERSION, "book": book, "weighting": weighting, "weights": w, "target": target, "final": final, "sleeves": sleeves, "holding_period": hold, "turnover_cap": turnover_cap,
            "dependence": dependence(scores_by_h), "flat_dates": int(((w.sum(axis=1) <= 1e-12)).sum()), "ic": ic}


def evaluate(scores_by_h: dict[int, pd.DataFrame], signal_cfg: dict, returns: pd.DataFrame, prices: pd.DataFrame, bt_cfg: dict, *, lag: int = 1, investable: pd.DataFrame | None = None, n_boot: int = 500,
             seed: int = 0, **kw) -> dict:
    """The combination and each single-horizon book through the same engine and costs, with the paired Sharpe difference of the combination against each."""
    built = build(scores_by_h, signal_cfg, prices=prices, lag=lag, investable=investable, returns=returns, **kw)
    streams = {"combination": run_backtest(built["final"], returns, investable, bt_cfg, lag).net_returns}
    singles = {}
    for h, s in per_session(scores_by_h).items():
        singles[f"horizon_{h}"] = build_weights(s, {**signal_cfg, "holding_period": h}, investable=investable, returns=returns)
        streams[f"horizon_{h}"] = run_backtest(singles[f"horizon_{h}"], returns, investable, bt_cfg, lag).net_returns
    first = max(int(np.argmax(built["final"].abs().sum(axis=1) > 0)), 0)
    start = built["final"].index[first]
    rows = []
    for name, r in streams.items():
        r = r.loc[start:]
        sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252.0)) if r.std(ddof=1) > 0 else None
        rows.append({"name": name, "sharpe": sh, "ann_return": float(r.mean() * 252.0)})
    best = max((x for x in rows if x["name"] != "combination" and x["sharpe"] is not None), key=lambda x: x["sharpe"], default=None)
    paired = {}
    for name in streams:
        if name == "combination":
            continue
        paired[name] = inf.paired_difference(streams["combination"].loc[start:], streams[name].loc[start:], statistic="sharpe", n_boot=n_boot, seed=seed, horizon=1, label=f"combination minus {name}",
                                             scope=f"{len(streams) - 1} single-horizon baselines").as_dict()
    return {"version": VERSION, "rows": rows, "best_single": None if best is None else best["name"], "paired_vs_single": paired, "dependence": built["dependence"]["effective_independent_signals"],
            "flat_dates": built["flat_dates"], "note": "the same engine, costs and start date for every row; a paired difference whose interval includes zero is not a win"}
