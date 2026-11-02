"""Filtering low-edge trades by their net edge in units of forecast uncertainty (spec v3, Release 7, P2-09).

A trade is the change from the book held to the book proposed, asset by asset. Its expected edge is what the forecast says the change earns over the holding window, ``mu_i * d_i`` for a change ``d_i``; its
price is ``c * |d_i|`` with a safety margin ``m`` on the cost; and the **defined uncertainty** of that edge is ``sigma_e * |d_i|``, where ``sigma_e`` is the root-mean-square error of the model's own forecasts
against the returns that followed them, measured only on decisions whose outcome had already settled (``s + label_offset <= t``) over the last ``window`` such decisions. A trade is made if

    (mu_i * sign(d_i) - (1 + m) * c) / sigma_e  >=  k,

and otherwise the asset stays at the weight it had. Trades that reduce a position the forecast no longer supports have positive edge in this sense and pass. The threshold ``k`` is fitted on **development data
only**: it is one setting of the ``signal`` section, compared over a declared grid on the validation segment in a frozen comparison card (each value a counted look), never chosen on the segments a frozen
plan protects.

Lower coverage is not better performance. ``apply`` returns the books and a log of what was proposed, made and rejected (by count and by turnover), the mean number of positions and the concentration of the
book with and without the filter, so the diversification the filter takes away is as visible as the cost it saves; the comparison card judges the result on the net return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .notrade import decision_dates

VERSION = "tradefilter-v1"
EPS = 1e-9


def forecast_error_scale(scores: pd.DataFrame, returns: pd.DataFrame, horizon: int, lag: int, window: int = 252, min_dates: int = 40) -> pd.Series:
    """``sigma_e`` by decision date: the root-mean-square of ``score - realised forward return`` over the last ``window`` decisions whose outcome had settled by that date. NaN until ``min_dates`` have settled."""
    from .targets import forward_return

    prices = (1.0 + returns.fillna(0.0)).cumprod()
    fwd = forward_return(prices, horizon, lag).reindex(index=scores.index, columns=scores.columns)
    err2 = ((scores - fwd) ** 2).mean(axis=1)                                                    # per decision date, over the assets that have both
    offset = int(lag) + int(horizon)
    settled = err2.shift(offset)                                                                  # the value of decision s becomes known `offset` sessions later
    mse = settled.rolling(int(window), min_periods=int(min_dates)).mean()
    return np.sqrt(mse)


def accept(mu: pd.Series, current: pd.Series, target: pd.Series, *, cost: float, cost_margin: float, scale: float, k: float) -> pd.DataFrame:
    """The verdict on every proposed trade of one date: its change, the forecast edge per unit traded, the net edge per unit after the margined cost, the net edge in uncertainty units, and whether it passes."""
    assets = list(target.index)
    d = target.reindex(assets).fillna(0.0) - current.reindex(assets).fillna(0.0)
    sign = np.sign(d)
    edge = mu.reindex(assets).fillna(0.0) * sign
    net = edge - (1.0 + cost_margin) * cost
    z = net / scale if scale > 0 and np.isfinite(scale) else pd.Series(np.nan, index=assets)
    ok = (d.abs() <= EPS) | (z >= k)
    return pd.DataFrame({"change": d, "edge_per_unit": edge, "net_edge_per_unit": net, "z": z, "accepted": ok}, index=assets)


def apply(targets: pd.DataFrame, scores: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    k = float(cfg["trade_filter_k"])
    cost, margin = float(cfg["cost_bps"]) / 1e4, float(cfg.get("trade_filter_cost_margin", 1.0))
    lag_h, h = int(cfg["label_offset"]), int(cfg["window_days"])
    sigma = forecast_error_scale(scores, returns, h, lag_h - h, int(cfg.get("trade_filter_window", 252)))
    dates = set(decision_dates(targets.index, str(cfg.get("rebalance", "daily")), int(cfg.get("lag_total", 1))))
    out = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    last = pd.Series(0.0, index=targets.columns)
    log = {"version": VERSION, "k": k, "cost_margin": margin, "cost_per_unit": cost, "uncertainty": "root-mean-square forecast error over settled decisions, window " + str(int(cfg.get("trade_filter_window", 252))),
           "dates": 0, "proposed": 0, "accepted": 0, "rejected": 0, "no_uncertainty_estimate": 0, "turnover_proposed": 0.0, "turnover_made": 0.0, "positions_unfiltered": [], "positions_filtered": [],
           "hhi_unfiltered": [], "hhi_filtered": []}

    def conc(w: pd.Series):
        g = float(w.abs().sum())
        return (int((w.abs() > EPS).sum()), float(((w.abs() / g) ** 2).sum()) if g > EPS else None)

    for i, date in enumerate(targets.index):
        if date in dates:
            tgt = targets.loc[date]
            s = sigma.get(date, np.nan)
            mu = scores.loc[date] if date in scores.index else pd.Series(0.0, index=targets.columns)
            if not np.isfinite(s) or s <= 0:
                log["no_uncertainty_estimate"] += 1                                              # without an uncertainty estimate nothing is filtered, and that is counted
                new = tgt.reindex(targets.columns).fillna(0.0)
                log["dates"] += 1
            else:
                v = accept(mu, last, tgt, cost=cost, cost_margin=margin, scale=float(s), k=k)
                moving = v[v["change"].abs() > EPS]
                log["dates"] += 1
                log["proposed"] += len(moving)
                log["accepted"] += int(moving["accepted"].sum())
                log["rejected"] += int((~moving["accepted"]).sum())
                log["turnover_proposed"] += float(moving["change"].abs().sum())
                log["turnover_made"] += float(moving.loc[moving["accepted"], "change"].abs().sum())
                keep = last.reindex(targets.columns).fillna(0.0).copy()
                for a in v.index[v["accepted"]]:
                    keep[a] = tgt.get(a, 0.0)
                new = keep
            n0, h0 = conc(tgt.fillna(0.0))
            n1, h1 = conc(new)
            log["positions_unfiltered"].append(n0)
            log["positions_filtered"].append(n1)
            if h0 is not None and h1 is not None:
                log["hhi_unfiltered"].append(h0)
                log["hhi_filtered"].append(h1)
            last = new
        out.iloc[i] = last.to_numpy()
    log["coverage_by_count"] = float(log["accepted"] / log["proposed"]) if log["proposed"] else None
    log["coverage_by_turnover"] = float(log["turnover_made"] / log["turnover_proposed"]) if log["turnover_proposed"] > EPS else None
    for key in ("positions_unfiltered", "positions_filtered", "hhi_unfiltered", "hhi_filtered"):
        v = log.pop(key)
        log["mean_" + key] = float(np.mean(v)) if v else None
    log["note"] = "lower coverage is not better performance: the net return of the filtered book against the unfiltered one is what a comparison card judges"
    return out, log
