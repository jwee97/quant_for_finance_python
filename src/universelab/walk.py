"""A walk-forward portfolio on a universe that changes over time.

At each rebalance date ``p`` the decision uses only rows strictly before ``p - 1`` (the one-day information lag the platform uses everywhere: data through day ``d`` can be traded at the close of ``d + 1`` and
earns from ``d + 2``). The steps, in order, each with data known at that date:

1. who is eligible (point-in-time membership, and a complete window of returns: a name without a full trailing window is not tradeable yet);
2. the screen narrows them;
3. a construction method from ``src.portfolio.lab`` turns the window into weights (a method that fails falls back to equal weight and is counted);
4. weights are capped, held with drift until the next rebalance, and every change is charged at ``cost_bps`` one way.

A name that stops having returns while held (it was delisted) earns zero from then and is counted in ``missing_return_days``: no delisting return is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..portfolio import lab
from ..portfolio.covariance import estimate_covariance
from . import screens as sc

ANN = 252.0


@dataclass
class Run:
    net: pd.Series
    gross: pd.Series
    turnover: pd.Series                       # one-way, on rebalance days
    weights: pd.DataFrame                     # the target weights set at each rebalance, stamped at the date they take effect
    held: pd.DataFrame                        # the drifting book at each close (what earns the next day)
    n_names: pd.Series
    fallbacks: int = 0
    skipped: int = 0                          # rebalance dates with too few names to build a portfolio (held cash)
    missing_return_days: int = 0
    spec: dict = field(default_factory=dict)


def run(returns: pd.DataFrame, screen: dict, method: str, *, dollar_volume: pd.DataFrame | None = None, eligible: pd.DataFrame | None = None, groups: dict | None = None, lookback: int = 252,
        step: int = 21, cost_bps: float = 10.0, max_weight: float = 1.0, cov_method: str = "shrinkage", mu_shrink: float = 0.8, risk_aversion: float = 3.0, exclude: tuple = ()) -> Run:
    """Walk the universe forward. ``returns`` is date x ticker with NaN where a name has no price; ``eligible`` the point-in-time eligibility (all true if None); ``exclude`` removes names for the whole run (used by
    the stock-removal sensitivity)."""
    if method not in lab.METHODS:
        raise ValueError(f"unknown method '{method}'")
    spec = sc.validate(screen)
    R = returns.drop(columns=[c for c in exclude if c in returns.columns])
    elig = None if eligible is None else eligible.reindex(index=R.index, columns=R.columns).fillna(False)
    n, k = R.shape
    if k < sc.MIN_NAMES:
        raise ValueError(f"need at least {sc.MIN_NAMES} names, have {k}")
    if n < lookback + 3 * step:
        raise ValueError(f"need at least {lookback + 3 * step} days of returns, have {n}")
    cost = cost_bps * 1e-4
    cols = list(R.columns)
    vals = np.nan_to_num(R.to_numpy(dtype=float), nan=0.0)
    present = R.notna().to_numpy()
    held = np.zeros(k)
    net, gross, turn = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    target_rows: dict = {}
    held_rows = np.zeros((n, k))
    names = np.zeros(n)
    fallbacks = skipped = missing = 0
    rebalance = set(range(lookback + 2, n - 1, step))
    started = False
    for t in range(n):
        if t in rebalance:
            p = t
            win = R.iloc[p - 1 - lookback:p - 1]                                  # rows strictly before p - 1
            ok = [c for c in cols if win[c].notna().all() and (elig is None or bool(elig.iloc[p - 2][c]))]
            w_full = np.zeros(k)
            if len(ok) >= sc.MIN_NAMES:
                dv = None if dollar_volume is None else dollar_volume.reindex(index=win.index, columns=ok)
                try:
                    keep = sc.select(win[ok], dv, spec, groups)
                except sc.ScreenError:
                    raise
                if len(keep) >= sc.MIN_NAMES:
                    sub = win[keep]
                    cov = estimate_covariance(sub, cov_method, lookback, annualise=True)
                    mu = sub.mean() * ANN
                    mu = mu_shrink * mu.mean() + (1 - mu_shrink) * mu
                    try:
                        w = np.asarray(pd.Series(lab.METHODS[method].fn(sub, {"cov": cov.to_numpy(dtype=float), "mu": mu.to_numpy(dtype=float), "lam": risk_aversion}), index=keep).to_numpy(dtype=float))
                        if not np.isfinite(w).all() or w.sum() <= 0:
                            raise RuntimeError("non-finite weights")
                        w = np.clip(w, 0.0, None) / np.clip(w, 0.0, None).sum()
                    except Exception:
                        w, fallbacks = np.full(len(keep), 1.0 / len(keep)), fallbacks + 1
                    if max_weight * len(keep) >= 1.0 - 1e-12:
                        w = lab.cap_weights(w, max_weight)
                    for name, x in zip(keep, w):
                        w_full[cols.index(name)] = x
                else:
                    skipped += 1
            else:
                skipped += 1
            turn[t] = float(np.abs(w_full - held).sum()) / 2.0                     # one-way turnover of this rebalance, against the book as it had drifted
            held = w_full
            target_rows[R.index[t]] = w_full
            started = True
        held_rows[t] = held
        names[t] = float((held > 1e-12).sum())
        if started and t + 1 < n:                                                   # the book held at the close of t earns the return of t + 1, then drifts
            r = vals[t + 1]
            missing += int(((~present[t + 1]) & (held > 1e-12)).sum())
            gross[t + 1] = float(held @ r)
            drifted = held * (1.0 + r)
            if drifted.sum() > 1e-12 and held.sum() > 0:
                held = drifted / drifted.sum() * held.sum()
    # costs are charged on the day the trade is made (the close of t), reducing that day's return stream entry
    idx = R.index
    net_s, gross_s = pd.Series(net, idx), pd.Series(gross, idx)
    turn_s = pd.Series(turn, idx)
    cost_s = (turn_s.fillna(0.0) * 2.0 * cost)
    net_s = (gross_s.fillna(0.0) - cost_s).where(gross_s.notna() | turn_s.notna())                 # the day the first book is built is a day of the run, free or not
    first = net_s.first_valid_index()
    wdf = pd.DataFrame(target_rows, index=cols).T
    return Run(net=net_s.loc[first:] if first is not None else net_s.iloc[0:0], gross=gross_s.loc[first:] if first is not None else gross_s.iloc[0:0], turnover=turn_s.dropna(), weights=wdf,
               held=pd.DataFrame(held_rows, index=idx, columns=cols), n_names=pd.Series(names, idx), fallbacks=fallbacks, skipped=skipped, missing_return_days=missing,
               spec={"screen": spec, "method": method, "lookback": lookback, "step": step, "cost_bps": cost_bps, "max_weight": max_weight})
