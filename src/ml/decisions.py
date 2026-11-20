"""Explicit buy / sell / hold decisions (ML Trading Strategy Builder, phase 3).

A model produces a *score* for each asset on each day. A *decision policy* turns scores into actions, and the actions into target weights. Keeping the two apart is the point: the model is trained once, and
the policy (thresholds, sizing, stops, holding limits, rebalance frequency, shorting and cash rules) is a separate, inspectable set of choices that can be changed, compared and validated on its own.

The policy is a small state machine per asset. The state is flat, long or short; each day the machine looks only at what is known at that day's close (the score, the confidence, the trailing volatility and the
price path up to and including that close) and emits one of

``BUY``       open a long          ``SELL``     close a long (a rule exit)     ``SHORT``   open a short       ``COVER``   close a short
``INCREASE``  add to a position    ``REDUCE``   trim a position                ``HOLD``    keep it
``STOP_LOSS`` / ``TAKE_PROFIT`` / ``TIME_EXIT``  forced exits, each with its own reason (and a cooldown before the asset may be re-entered)
``REBALANCE`` a change of size with no change of conviction

Selling a long is therefore a different action from opening a short, and ``allow_short`` decides whether the second exists at all. The score is converted to a cross-sectional *percentile rank* each day, so a
return forecast, a ranking score and a class probability are all read the same way and the thresholds mean the same thing for every model (an entry at the top 20% is the top 20% whatever the model's units).
Hysteresis (``exit_quantile`` wider than ``entry_quantile``) stops a name that has just entered from being sold the next day for a trivial change of rank.

Timing follows the platform's convention exactly: a decision made from the close of ``t`` is traded ``lag`` days later by the backtest engine and earns from then on. The stop-loss and take-profit levels are
measured from the *executed* entry price (the close ``lag`` days after the decision), and a stop is tested on the close of each day after that, so a stop that fires at ``t`` is itself executed at ``t + lag``.
The policy never sees a price after the day it decides on.

Output: the target weights (date x asset), and a decision log, one row per decision with the information that was available: the prediction, its rank and confidence, the rule that fired and the book before and
after. ``summarise`` and ``trade_log.py`` build the trade ledger and the analytics from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "decisions-v1"

ACTIONS = ("BUY", "SELL", "SHORT", "COVER", "INCREASE", "REDUCE", "HOLD", "STOP_LOSS", "TAKE_PROFIT", "TIME_EXIT", "REBALANCE")
OPENING = {"BUY", "SHORT"}
CLOSING = {"SELL", "COVER", "STOP_LOSS", "TAKE_PROFIT", "TIME_EXIT"}

DEFAULTS = {
    "entry_quantile": 0.2,          # open a position on an asset whose score is in the best (worst, for a short) 20% of the day
    "exit_quantile": 0.4,           # close it when it falls out of the best 40%: wider than the entry, so a position is not churned on a rank change of one place
    "min_confidence": 0.0,          # the model's confidence (0..1) needed to open; a position already held is not closed for low confidence alone
    "max_positions": 5,             # long positions at most (and short positions at most, separately)
    "max_position": 0.25,           # the largest weight of one asset
    "sizing": "equal",              # equal | inverse_vol | score
    "allow_short": False,
    "long_exposure": 1.0,           # the gross long exposure when fully invested: what is left is cash
    "short_exposure": 0.0,          # the gross short exposure (used only with allow_short)
    "cash_floor": 0.0,              # a minimum share of the book kept in cash
    "stop_loss_vol": None,          # the loss, in multiples of the entry-day volatility scaled to the holding horizon, at which a position is closed (None: no stop)
    "take_profit_vol": None,        # the same for a profit
    "max_holding": None,            # the most trading days a position is kept (None: no limit)
    "cooldown": 5,                  # days after a stop, take-profit or time exit before the asset may be bought again
    "rebalance_days": 5,            # entries, rule exits and size changes happen only every this many days; stops, take-profits and time exits are checked daily
    "resize_band": 0.25,            # a held position is resized only when its target moves by more than this share of the target (stops churn from small sizing changes)
    "horizon": 5,                   # the model's horizon in days, used to scale the stop and take-profit levels
    "min_edge_bps": None,           # if set, the score is read as a forecast return and a position is opened only when the forecast exceeds this many bps plus the round-trip cost
    "cost_bps": 0.0,                # the one-way cost, used only for the min_edge test (the backtest charges the real cost)
}

POLICY_KEYS = tuple(DEFAULTS)


def policy_from(cfg: dict) -> dict:
    """The decision policy: the defaults overlaid with any ``decision_*`` key of the signal configuration (``decision_entry_quantile`` and so on), validated."""
    p = dict(DEFAULTS)
    for k in POLICY_KEYS:
        if f"decision_{k}" in cfg and cfg[f"decision_{k}"] is not None:
            p[k] = cfg[f"decision_{k}"]
        elif f"decision_{k}" in cfg and k in ("stop_loss_vol", "take_profit_vol", "max_holding", "min_edge_bps"):
            p[k] = None
    validate(p)
    return p


def validate(p: dict) -> None:
    def need(ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(f"decision policy: {msg}")

    need(0.0 < float(p["entry_quantile"]) <= 0.5, "entry_quantile is between 0 and 0.5")
    need(float(p["entry_quantile"]) <= float(p["exit_quantile"]) <= 1.0, "exit_quantile is between entry_quantile and 1 (a position is kept at least as long as it would have been opened)")
    need(0.0 <= float(p["min_confidence"]) <= 1.0, "min_confidence is between 0 and 1")
    need(int(p["max_positions"]) >= 1, "max_positions is at least 1")
    need(0.0 < float(p["max_position"]) <= 1.0, "max_position is in (0, 1]")
    need(p["sizing"] in ("equal", "inverse_vol", "score"), "sizing is equal, inverse_vol or score")
    need(0.0 <= float(p["long_exposure"]) <= 3.0 and 0.0 <= float(p["short_exposure"]) <= 3.0, "exposures are between 0 and 3")
    need(not float(p["short_exposure"]) > 0 or bool(p["allow_short"]), "short_exposure needs allow_short")
    need(0.0 <= float(p["cash_floor"]) < 1.0 and float(p["long_exposure"]) <= 1.0 - float(p["cash_floor"]) + 1e-9 + (float(p["short_exposure"]) if p["allow_short"] else 0.0),
         "long_exposure leaves no room for the cash floor")
    for k in ("stop_loss_vol", "take_profit_vol"):
        need(p[k] is None or float(p[k]) > 0, f"{k} is positive (or unset)")
    need(p["max_holding"] is None or int(p["max_holding"]) >= 1, "max_holding is at least 1 day (or unset)")
    need(int(p["cooldown"]) >= 0 and int(p["rebalance_days"]) >= 1 and int(p["horizon"]) >= 1, "cooldown >= 0, rebalance_days >= 1 and horizon >= 1")
    need(0.0 <= float(p["resize_band"]) <= 5.0, "resize_band is between 0 and 5")
    need(p["min_edge_bps"] is None or float(p["min_edge_bps"]) >= 0, "min_edge_bps is not negative (or unset)")


def _cross_rank(score: pd.DataFrame) -> pd.DataFrame:
    """The percentile rank (1 = best) of each asset's score among the day's valid scores, 0..1 with the top at 1 (ties share the average rank)."""
    r = score.rank(axis=1, method="average", pct=True)
    return r


def _sizes(names: list[int], score_row: np.ndarray, vol_row: np.ndarray, p: dict, exposure: float) -> dict[int, float]:
    """Weights of the selected names (as positive numbers) summing to at most ``exposure`` with none above ``max_position``."""
    if not names or exposure <= 0:
        return {}
    if p["sizing"] == "inverse_vol":
        raw = np.array([1.0 / max(vol_row[i], 1e-6) if np.isfinite(vol_row[i]) else 0.0 for i in names])
    elif p["sizing"] == "score":
        raw = np.array([max(abs(score_row[i] - 0.5), 1e-6) for i in names])
    else:
        raw = np.ones(len(names))
    if raw.sum() <= 0:
        raw = np.ones(len(names))
    w = raw / raw.sum() * exposure
    cap = float(p["max_position"])
    for _ in range(5):                                                         # cap and redistribute what was cut to the names still under the cap
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        if not under.any() or w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return {i: float(min(x, cap)) for i, x in zip(names, w)}


def schedule_from(cfg: dict) -> list[tuple[pd.Timestamp, dict]]:
    """The dated policy changes of a signal configuration (``decision_schedule``: ``[{"from": "2019-01-02", "policy": {"entry_quantile": 0.3}}, ...]``) as ``(date, full policy)`` pairs in date order, each
    entry the base policy overlaid with that entry's settings. From its date on, an entry replaces the base; the state (positions, entry prices, cooldowns) carries over unchanged."""
    base = policy_from(cfg)
    out = []
    for e in cfg.get("decision_schedule") or []:
        extra = {k: v for k, v in (e.get("policy") or {}).items() if k in POLICY_KEYS}
        bad = set(e.get("policy") or {}) - set(POLICY_KEYS)
        if bad:
            raise ValueError(f"decision schedule: unknown settings {sorted(bad)}")
        merged = {**base, **extra}
        validate(merged)
        out.append((pd.Timestamp(e["from"]), merged))
    out.sort(key=lambda x: x[0])
    return out


def run(score: pd.DataFrame, returns: pd.DataFrame, policy: dict, *, vol: pd.DataFrame | None = None, confidence: pd.DataFrame | None = None, investable: pd.DataFrame | None = None,
        lag: int = 1, forecast_is_return: bool = False, schedule: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(weights, decisions)``: the target weights stamped at the decision date, and the decision log.

    ``score`` is date x asset (higher is better); ``returns`` are the daily simple returns the price path is rebuilt from; ``vol`` the trailing daily volatilities (known at each close); ``confidence`` in 0..1.
    """
    p = dict(policy)
    validate(p)
    idx, cols = score.index, list(score.columns)
    n, k = len(idx), len(cols)
    ret = returns.reindex(index=idx, columns=cols)
    px = (1.0 + ret.fillna(0.0)).cumprod().to_numpy()
    inv = np.ones((n, k), bool) if investable is None else investable.reindex(index=idx, columns=cols).fillna(False).to_numpy(bool)
    sc = score.where(pd.DataFrame(inv, index=idx, columns=cols))
    rank = _cross_rank(sc).to_numpy()
    scv = sc.to_numpy(dtype=float)
    vl = np.full((n, k), np.nan) if vol is None else vol.reindex(index=idx, columns=cols).to_numpy(dtype=float)
    cf = np.ones((n, k)) if confidence is None else confidence.reindex(index=idx, columns=cols).fillna(0.0).to_numpy(dtype=float)
    sched = list(schedule or [])
    sched_i = -1                                                               # the policy in force changes on the dates of the schedule; the book and its state do not
    base_policy = p

    def settings(q: dict):
        return float(q["entry_quantile"]), float(q["exit_quantile"]), int(q["max_positions"]), int(q["horizon"]), int(q["rebalance_days"]), 2.0 * float(q["cost_bps"]) / 1e4

    ent, ext, maxpos, h, reb, rt_cost = settings(p)

    state = np.zeros(k)                                                        # +1 long, -1 short, 0 flat
    weight = np.zeros(k)                                                       # the current target weight
    entry_px = np.full(k, np.nan)                                              # executed entry price (the close `lag` days after the decision); NaN until executed
    entry_idx = np.full(k, -1)                                                 # the decision day that opened the position
    entry_vol = np.full(k, np.nan)
    cool = np.zeros(k, int)
    W = np.zeros((n, k))
    rows: list[dict] = []

    def log(t, i, action, reason, new_w, **extra):
        rows.append({"date": idx[t], "asset": cols[i], "action": action, "reason": reason, "score": float(scv[t, i]) if np.isfinite(scv[t, i]) else np.nan, "rank_pct": float(rank[t, i]) if np.isfinite(rank[t, i]) else np.nan,
                     "confidence": float(cf[t, i]), "prev_weight": float(weight[i]), "target_weight": float(new_w), "price": float(px[t, i]) if np.isfinite(px[t, i]) else np.nan,
                     "entry_price": float(entry_px[i]) if np.isfinite(entry_px[i]) else np.nan, "holding_days": int(t - entry_idx[i]) if entry_idx[i] >= 0 else 0, **extra})

    for t in range(n):
        while sched_i + 1 < len(sched) and sched[sched_i + 1][0] <= idx[t]:
            sched_i += 1
            p = sched[sched_i][1]
            ent, ext, maxpos, h, reb, rt_cost = settings(p)
        if sched_i < 0:
            p = base_policy
        valid = inv[t] & np.isfinite(rank[t])
        cool = np.maximum(cool - 1, 0)
        for i in range(k):                                                     # an executed entry price exists once `lag` days have passed since the decision
            if state[i] != 0 and not np.isfinite(entry_px[i]) and t >= entry_idx[i] + lag and np.isfinite(px[t, i]):
                entry_px[i] = px[entry_idx[i] + lag, i] if np.isfinite(px[entry_idx[i] + lag, i]) else px[t, i]
        forced: dict[int, tuple[str, str]] = {}
        for i in range(k):                                                     # daily risk exits on what the close shows
            if state[i] == 0:
                continue
            if not valid[i] and not inv[t, i]:
                forced[i] = ("SELL" if state[i] > 0 else "COVER", "asset no longer investable")
                continue
            held = t - entry_idx[i]
            if p["max_holding"] is not None and held >= int(p["max_holding"]):
                forced[i] = ("TIME_EXIT", f"held {held} days (limit {int(p['max_holding'])})")
                continue
            if np.isfinite(entry_px[i]) and np.isfinite(px[t, i]) and np.isfinite(entry_vol[i]):
                pnl = state[i] * (px[t, i] / entry_px[i] - 1.0)
                unit = entry_vol[i] * np.sqrt(h)
                if p["stop_loss_vol"] is not None and pnl <= -float(p["stop_loss_vol"]) * unit:
                    forced[i] = ("STOP_LOSS", f"loss {pnl:+.1%} reached the stop at -{float(p['stop_loss_vol']):g} x {unit:.1%}")
                elif p["take_profit_vol"] is not None and pnl >= float(p["take_profit_vol"]) * unit:
                    forced[i] = ("TAKE_PROFIT", f"gain {pnl:+.1%} reached the target at +{float(p['take_profit_vol']):g} x {unit:.1%}")
        is_reb = (t % reb == 0)
        new_state, new_entry = state.copy(), []
        if is_reb:
            for i in range(k):                                                 # rule exits
                if state[i] == 0 or i in forced:
                    continue
                if not valid[i]:
                    continue
                if state[i] > 0 and rank[t, i] < 1.0 - ext:
                    forced[i] = ("SELL", f"rank {rank[t, i]:.0%} fell below the exit line of {1.0 - ext:.0%}")
                elif state[i] < 0 and rank[t, i] > ext:
                    forced[i] = ("COVER", f"rank {rank[t, i]:.0%} rose above the cover line of {ext:.0%}")
        for i, (act, why) in forced.items():
            log(t, i, act, why, 0.0, unrealised_ret=float(state[i] * (px[t, i] / entry_px[i] - 1.0)) if np.isfinite(entry_px[i]) else np.nan)
            new_state[i] = 0
            if act in ("STOP_LOSS", "TAKE_PROFIT", "TIME_EXIT"):
                cool[i] = int(p["cooldown"]) + 1
        if is_reb:                                                             # entries: the best-ranked names not held, within the position limits
            def edge_ok(i, sign):
                if p["min_edge_bps"] is None or not forecast_is_return:
                    return True
                return sign * scv[t, i] * 1e4 >= float(p["min_edge_bps"]) + rt_cost * 1e4
            held_long = int((new_state > 0).sum())
            order = np.argsort(-np.where(valid, rank[t], -1.0))
            for i in order:
                if held_long >= maxpos:
                    break
                if not valid[i] or new_state[i] != 0 or cool[i] > 0:
                    continue
                if rank[t, i] >= 1.0 - ent and cf[t, i] >= float(p["min_confidence"]) and edge_ok(i, 1):
                    new_state[i] = 1
                    new_entry.append((i, "BUY", f"rank {rank[t, i]:.0%} is in the top {ent:.0%}" + (f", confidence {cf[t, i]:.0%}" if confidence is not None else "")))
                    held_long += 1
            if p["allow_short"]:
                held_short = int((new_state < 0).sum())
                for i in order[::-1]:
                    if held_short >= maxpos:
                        break
                    if not valid[i] or new_state[i] != 0 or cool[i] > 0:
                        continue
                    if rank[t, i] <= ent and cf[t, i] >= float(p["min_confidence"]) and edge_ok(i, -1):
                        new_state[i] = -1
                        new_entry.append((i, "SHORT", f"rank {rank[t, i]:.0%} is in the bottom {ent:.0%}" + (f", confidence {cf[t, i]:.0%}" if confidence is not None else "")))
                        held_short += 1
        # sizing: the whole book is re-sized on a rebalance day; between rebalances, only exits change it
        target = np.zeros(k)
        if is_reb:
            longs = [i for i in range(k) if new_state[i] > 0]
            shorts = [i for i in range(k) if new_state[i] < 0]
            exp_l = min(float(p["long_exposure"]), 1.0 - float(p["cash_floor"]) + (float(p["short_exposure"]) if p["allow_short"] else 0.0))
            for i, wv in _sizes(longs, rank[t], vl[t], p, exp_l).items():
                target[i] = wv
            for i, wv in _sizes(shorts, 1.0 - rank[t], vl[t], p, float(p["short_exposure"]) if p["allow_short"] else 0.0).items():
                target[i] = -wv
        else:
            target = np.where(new_state != 0, weight, 0.0)
        for i, act, why in new_entry:
            log(t, i, act, why, target[i])
            entry_idx[i], entry_px[i], entry_vol[i] = t, np.nan, vl[t, i]
        for i in range(k):
            if new_state[i] != 0 and state[i] != 0 and i not in forced:       # a position carried through the day
                if is_reb and abs(target[i] - weight[i]) > float(p["resize_band"]) * max(abs(target[i]), 1e-9):
                    log(t, i, "INCREASE" if abs(target[i]) > abs(weight[i]) else "REDUCE", "target size moved by more than the resize band", target[i],
                        unrealised_ret=float(state[i] * (px[t, i] / entry_px[i] - 1.0)) if np.isfinite(entry_px[i]) else np.nan)
                else:
                    target[i] = weight[i]                                      # inside the band: keep the size (no trade)
                    if is_reb:
                        log(t, i, "HOLD", f"rank {rank[t, i]:.0%} still inside the exit line", target[i],
                            unrealised_ret=float(state[i] * (px[t, i] / entry_px[i] - 1.0)) if np.isfinite(entry_px[i]) else np.nan)
        state = new_state
        weight = np.where(state != 0, target, 0.0)
        W[t] = weight
        for i in range(k):
            if state[i] == 0:
                entry_px[i], entry_idx[i], entry_vol[i] = np.nan, -1, np.nan
    weights = pd.DataFrame(W, index=idx, columns=cols)
    decisions = pd.DataFrame(rows)
    if len(decisions):
        decisions["kind"] = decisions["action"].map(lambda a: "open" if a in OPENING else ("close" if a in CLOSING else ("resize" if a in ("INCREASE", "REDUCE", "REBALANCE") else "hold")))
    return weights, decisions


def summarise(decisions: pd.DataFrame) -> dict:
    """Counts of each action and the reasons exits happened for (a quick read of what the policy actually did)."""
    if decisions is None or not len(decisions):
        return {"n": 0, "by_action": {}}
    return {"n": int(len(decisions)), "by_action": {k: int(v) for k, v in decisions["action"].value_counts().items()},
            "n_opens": int((decisions["kind"] == "open").sum()), "n_closes": int((decisions["kind"] == "close").sum()), "version": VERSION}
