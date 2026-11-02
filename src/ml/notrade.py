"""No-trade regions: do not trade unless the move earns its cost (spec v3, Release 7, P2-05).

A rebalance that improves the book by less than it costs destroys value, and a book that is moved every time the target twitches pays for noise. The rule here compares two alternatives **under the same
forecast and the same risk model**, from the book actually held:

* hold the current book ``w0`` (after whatever is forced, below), or
* move to the proposed target ``w*``,

by their utility ``U(w) = mu'w - (lambda / 2) w'Sigma w``, with ``mu`` the forecast return over the holding window and ``Sigma`` the covariance over the same window. The move is made only if its gain
exceeds its incremental cost with a safety margin: ``U(w*) - U(w0) > (1 + m) * c * |w* - w0|_1``. **Hysteresis**: ``m`` is ``margin`` when the book is at rest and the smaller ``hold_margin`` while a trading
episode is under way (the previous decision traded), so a gain that hovers around the threshold does not switch the book in and out.

**Hard controls override the discretionary rule**, and say so. Before the comparison the current book is brought inside the limits: a name that can no longer be held (delisted, not investable) is closed, a
position above the position limit is cut to it, a gross above the limit is scaled to it; each is a recorded forced trade with its reason, made whether or not it pays. If the *target* itself is infeasible (a
position above the limit, a gross above the limit, a name that cannot be held), it is not an alternative: the book is held (after the forced trades) and the decision says ``target infeasible`` with the reasons.

``apply`` runs the rule through time on a panel of target books, from the book it held the day before, at the dates on which the backtest engine would trade, and returns the books and a log of every decision:
how often it traded, how often a trade was forced or a target was infeasible, the cost avoided and the utility given up by holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.execution import rebalance_dates

VERSION = "notrade-v1"
EPS = 1e-9


def utility(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, risk_aversion: float) -> float:
    a = w.reindex(cov.index).fillna(0.0).to_numpy()
    m = mu.reindex(cov.index).fillna(0.0).to_numpy()
    return float(m @ a - 0.5 * risk_aversion * a @ cov.to_numpy() @ a)


def decide(current: pd.Series, target: pd.Series, mu: pd.Series, cov: pd.DataFrame, *, cost: float, margin: float, hold_margin: float | None = None, risk_aversion: float = 5.0, active: bool = False,
           position_max: float | None = None, gross_max: float | None = None, net_min: float | None = None, net_max: float | None = None, tradable: pd.Series | None = None) -> dict:
    """One decision. Returns ``weights`` (the book after the decision), ``traded``, ``reason``, the two utilities, the incremental ``cost``, the ``threshold`` it had to beat, the ``forced`` trades with their
    reasons, whether the target was feasible (and why not), and ``active`` for the next decision."""
    if margin < 0 or cost < 0:
        raise ValueError("margin and cost are not negative")
    hold_m = margin if hold_margin is None else float(hold_margin)
    assets = list(cov.index)
    w0 = current.reindex(assets).fillna(0.0).astype(float)
    wt = target.reindex(assets).fillna(0.0).astype(float)
    can = (tradable.reindex(assets).fillna(False).astype(bool) if tradable is not None else pd.Series(True, index=assets))
    forced = []
    for a in assets:                                                                              # hard controls first: they are not up for discussion
        if not can[a] and abs(w0[a]) > EPS:
            forced.append({"asset": a, "from": float(w0[a]), "to": 0.0, "reason": "the name can no longer be held"})
            w0[a] = 0.0
    if position_max is not None:
        for a in assets:
            if abs(w0[a]) > position_max + EPS:
                cut = float(np.sign(w0[a]) * position_max)
                forced.append({"asset": a, "from": float(w0[a]), "to": cut, "reason": "above the position limit"})
                w0[a] = cut
    if gross_max is not None and w0.abs().sum() > gross_max + EPS:
        before = float(w0.abs().sum())
        w0 = w0 * (gross_max / before)
        forced.append({"asset": "*", "from": before, "to": float(gross_max), "reason": "gross exposure above the limit (the whole book scaled)"})
    problems = []
    if (wt[~can].abs() > EPS).any():
        problems.append("the target holds a name that cannot be held")
    if position_max is not None and (wt.abs() > position_max + EPS).any():
        problems.append("the target is above the position limit")
    if gross_max is not None and wt.abs().sum() > gross_max + EPS:
        problems.append("the target's gross exposure is above the limit")
    if net_min is not None and wt.sum() < net_min - EPS or net_max is not None and wt.sum() > net_max + EPS:
        problems.append("the target's net exposure is outside its bounds")
    u0, ut = utility(w0, mu, cov, risk_aversion), utility(wt, mu, cov, risk_aversion)
    gain = ut - u0
    turnover = float((wt - w0).abs().sum())
    inc_cost = float(cost * turnover)
    threshold = (1.0 + (hold_m if active else margin)) * inc_cost
    out = {"utility_hold": u0, "utility_target": ut, "gain": gain, "cost": inc_cost, "threshold": threshold, "forced": forced, "target_feasible": not problems, "infeasible_reasons": problems,
           "turnover_if_traded": turnover}
    if problems:
        return {**out, "weights": w0, "traded": False, "reason": "target infeasible: " + "; ".join(problems) + (" (forced trades were still made)" if forced else ""), "active": False}
    if turnover <= EPS:
        return {**out, "weights": w0, "traded": False, "reason": "already at the target", "active": False}
    if gain > threshold:
        return {**out, "weights": wt, "traded": True, "reason": f"the gain {gain:.3g} exceeds the cost with its margin {threshold:.3g}", "active": True}
    return {**out, "weights": w0, "traded": False, "reason": (f"the gain {gain:.3g} does not exceed the cost with its margin {threshold:.3g}" + (" (hysteresis: the lower hold margin applied)" if active else "")), "active": False}


def decision_dates(index: pd.DatetimeIndex, frequency: str, lag: int) -> pd.DatetimeIndex:
    """The decision dates whose book the engine trades: the rebalance dates, ``lag`` sessions earlier (the engine lags the targets before it restricts them to the schedule)."""
    marks = rebalance_dates(index, frequency)
    pos = index.get_indexer(marks)
    pos = pos[pos >= 0] - int(lag)
    return index[pos[pos >= 0]]


def apply(targets: pd.DataFrame, scores: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Run ``decide`` through time. ``targets`` are the books the rule proposes (stamped at their decision dates), ``scores`` the forecast returns over ``window_days``; the covariance is fitted as of the date
    from the returns up to it (refitted every ``cov_every`` decisions) and scaled to the window. Between decision dates the last book is carried."""
    from . import risk_models as rm

    cov_spec = rm.CovSpec(str(cfg.get("cov_method", "lw_constant_correlation")), window=int(cfg.get("cov_window", 252)))
    days = int(cfg["window_days"])
    cost = float(cfg["cost_bps"]) / 1e4
    margin, hold_margin = float(cfg["no_trade_margin"]), cfg.get("no_trade_hold_margin")
    lam = float(cfg.get("risk_aversion", 5.0))
    pos_max, gross_max = cfg.get("max_weight"), float(cfg.get("gross", 1.0)) + 1e-9
    dates = set(decision_dates(targets.index, str(cfg.get("rebalance", "daily")), int(cfg.get("lag_total", 1))))
    out = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    last = pd.Series(0.0, index=targets.columns)
    active = False
    fit, fit_at = None, -10 ** 9
    log = {"version": VERSION, "decisions": 0, "traded": 0, "held_below_threshold": 0, "target_infeasible": 0, "forced_trades": 0, "forced_reasons": {}, "no_covariance": 0, "cost_avoided": 0.0,
           "utility_given_up": 0.0, "margin": margin, "hold_margin": margin if hold_margin is None else float(hold_margin), "cost_per_unit": cost, "risk_aversion": lam, "window_days": days, "events": []}
    for i, date in enumerate(targets.index):
        if date in dates and targets.loc[date].abs().sum() + last.abs().sum() > 0:
            mu = scores.loc[date] if date in scores.index else pd.Series(np.nan, index=targets.columns)
            if fit is None or i - fit_at >= int(cfg.get("cov_every", 5)):
                try:
                    fit, fit_at = rm.fit_asof(returns, date, cov_spec, investable), i
                except ValueError:
                    fit = None
            if fit is None or mu.dropna().empty:
                log["no_covariance"] += 1
                out.iloc[i] = last.to_numpy()
                continue
            names = [a for a in fit.matrix.index if a in targets.columns]
            tradable = None if investable is None else investable.loc[date].reindex(targets.columns).fillna(False)
            cov = fit.matrix.loc[names, names] * days
            d = decide(last, targets.loc[date], mu.reindex(names).fillna(0.0), cov, cost=cost, margin=margin, hold_margin=None if hold_margin is None else float(hold_margin), risk_aversion=lam, active=active,
                       position_max=None if pos_max is None else float(pos_max), gross_max=gross_max, tradable=tradable)
            log["decisions"] += 1
            log["traded"] += int(d["traded"])
            log["held_below_threshold"] += int((not d["traded"]) and d["target_feasible"] and d["reason"] != "already at the target")
            log["target_infeasible"] += int(not d["target_feasible"])
            log["forced_trades"] += len(d["forced"])
            for f in d["forced"]:
                log["forced_reasons"][f["reason"]] = log["forced_reasons"].get(f["reason"], 0) + 1
            if (not d["traded"]) and d["target_feasible"] and d["turnover_if_traded"] > EPS:
                log["cost_avoided"] += d["cost"]
                log["utility_given_up"] += max(d["gain"], 0.0)
            if len(log["events"]) < 25 and (d["forced"] or not d["target_feasible"]):
                log["events"].append({"date": str(date.date()), "reason": d["reason"], "forced": d["forced"][:3]})
            active = d["active"]
            last = d["weights"].reindex(targets.columns).fillna(0.0)
        out.iloc[i] = last.to_numpy()
    return out, log
