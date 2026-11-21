"""The trade ledger, round trips and trading analytics (ML Strategy Explorer, phase 4).

Three layers are kept apart, because a signal is not an order and an order is not a fill:

1. the model's *prediction* (a score and rank on a decision date),
2. the policy's *decision* (``decisions.py``: BUY / SELL / HOLD ... with its reason and target weight),
3. the *execution* (the backtest engine's trades: the book changes ``lag`` days after the decision, at that day's close, at the configured cost).

``executions`` reads layer 3 off the engine's result and joins layer 2 onto it by asset and decision date, so each trade says which decision caused it, and a decision that never became a trade (a target the
engine did not need to trade, or one that was cancelled by a later decision before it was executed) shows up in ``unexecuted``. ``round_trips`` pairs each opening with its close and reports the position's own
return and its contribution to the portfolio; ``analytics`` summarises both. Costs here are the engine's linear cost on traded notional (the realistic order-level model is ``execution.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "trade-log-v1"


def _side(prev: float, new: float) -> str:
    if abs(prev) < 1e-12 and new > 0:
        return "OPEN_LONG"
    if abs(prev) < 1e-12 and new < 0:
        return "OPEN_SHORT"
    if prev > 0 and abs(new) < 1e-12:
        return "CLOSE_LONG"
    if prev < 0 and abs(new) < 1e-12:
        return "CLOSE_SHORT"
    if prev > 0 and new < 0:
        return "FLIP_TO_SHORT"
    if prev < 0 and new > 0:
        return "FLIP_TO_LONG"
    if prev > 0:
        return "ADD_LONG" if new > prev else "TRIM_LONG"
    return "ADD_SHORT" if new < prev else "TRIM_SHORT"


def executions(held: pd.DataFrame, trades: pd.DataFrame, returns: pd.DataFrame, cost_bps: float, decisions: pd.DataFrame | None = None, lag: int = 1, min_trade: float = 1e-9) -> pd.DataFrame:
    """One row per executed trade: the day, asset, kind of trade, the weight traded, the book before and after, the price index at the fill, the cost, and the decision behind it (when there is one).

    ``held`` is the book in force at each close (the engine's ``weights``); ``trades`` its signed changes by execution day (the engine's ``trades``); ``returns`` the asset returns the price path is rebuilt from.
    A trade is the change that the engine made at that day's close; weight drift (a position growing with its price) is not a trade.
    """
    t = trades.fillna(0.0)
    px = (1.0 + returns.reindex(columns=t.columns).fillna(0.0)).cumprod()
    rows = []
    for asset in t.columns:
        col = t[asset]
        for d in col.index[col.abs() > min_trade]:
            new = float(held[asset].get(d, np.nan)) if asset in held.columns else np.nan
            delta = float(col.loc[d])
            prev = new - delta if np.isfinite(new) else np.nan
            rows.append({"date": d, "asset": asset, "kind": _side(prev, new) if np.isfinite(prev) else "UNKNOWN", "side": "BUY" if delta > 0 else "SELL", "trade_weight": delta,
                         "weight_before": prev, "weight_after": new, "price_index": float(px[asset].loc[d]), "cost": abs(delta) * float(cost_bps) / 1e4})
    out = pd.DataFrame(rows)
    if not len(out):
        return pd.DataFrame(columns=["date", "asset", "kind", "side", "trade_weight", "weight_before", "weight_after", "price_index", "cost", "decision_date", "action", "reason", "rank_pct", "score"])
    cal = returns.index                                                         # the full calendar, so the first executions still find the day they were decided
    pos = {d: i for i, d in enumerate(cal)}
    out["decision_date"] = [cal[pos[d] - lag] if pos[d] - lag >= 0 else pd.NaT for d in out["date"]]
    if decisions is not None and len(decisions):
        dd = decisions[decisions["kind"].isin(["open", "close", "resize"])].copy()
        dd = dd.rename(columns={"date": "decision_date"})[["decision_date", "asset", "action", "reason", "rank_pct", "score", "confidence"]]
        dd = dd.groupby(["decision_date", "asset"], sort=False).agg(action=("action", " + ".join), reason=("reason", " -> ".join), rank_pct=("rank_pct", "last"), score=("score", "last"),
                                                                  confidence=("confidence", "last")).reset_index()        # a flip is a close and an open decided the same day: one trade, both reasons
        out = out.merge(dd, on=["decision_date", "asset"], how="left")
    else:
        for c in ("action", "reason", "rank_pct", "score", "confidence"):
            out[c] = np.nan
    if decisions is not None and len(decisions):                                      # a trade with no decision behind it is the engine putting a drifted position back to its target on a day it traded the book
        none = out["action"].isna()
        out.loc[none, "action"] = "DRIFT"
        out.loc[none, "reason"] = "size restored to target while the book was being traded (weights drift with prices)"
    return out.sort_values(["date", "asset"]).reset_index(drop=True)


def unexecuted(decisions: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """Decisions that changed the target but did not become a trade (the engine held the book, or a later decision replaced them before they were executed)."""
    if decisions is None or not len(decisions):
        return pd.DataFrame()
    acts = decisions[decisions["kind"].isin(["open", "close", "resize"])]
    if not len(fills):
        return acts.copy()
    done = set(zip(pd.to_datetime(fills["decision_date"]), fills["asset"]))
    mask = [(d, a) not in done for d, a in zip(acts["date"], acts["asset"])]
    return acts[mask].copy()


def round_trips(held: pd.DataFrame, returns: pd.DataFrame, fills: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Each position from the day it was opened to the day it was closed (a position still open at the end is reported as open, marked at the last close).

    ``position_return`` is the return on the capital put into that position (the compounded asset return, signed by the side); ``contribution`` the sum of the daily weight x return, the share of the
    portfolio's return the position produced; ``cost`` the costs of its trades; ``net_contribution`` the contribution less that cost.
    """
    h = held.fillna(0.0)
    r = returns.reindex(index=h.index, columns=h.columns).fillna(0.0)
    prev = h.shift(1).fillna(0.0)                                              # the book that earned the day's return
    daily = prev * r
    rows = []
    for asset in h.columns:
        w = h[asset].to_numpy()
        pw = prev[asset].to_numpy()
        i, n = 0, len(w)
        while i < n:
            if abs(w[i]) > 1e-12 and (abs(pw[i]) <= 1e-12 or np.sign(pw[i]) != np.sign(w[i])):                  # a flip from long to short closes one position and opens the next on the same day
                side = 1.0 if w[i] > 0 else -1.0
                j = i + 1
                while j < n and abs(w[j]) > 1e-12 and np.sign(w[j]) == side:
                    j += 1
                closed = j < n
                end = j if closed else n - 1                                   # the day the book went flat (or the last day)
                span = slice(i + 1, end + 1)                                   # the days the position earned: after the open day, up to and including the close day
                ret_path = r[asset].to_numpy()[span]
                pos_ret = float(np.prod(1.0 + side * ret_path) - 1.0) if len(ret_path) else 0.0
                contrib = float(daily[asset].to_numpy()[span].sum())
                f = fills[(fills["asset"] == asset) & (fills["date"] >= h.index[i]) & (fills["date"] <= h.index[end])] if len(fills) else fills
                cost = 0.0
                for _, fl in (f.iterrows() if len(f) else []):
                    share = 1.0
                    if str(fl["kind"]).startswith("FLIP"):                      # one fill closes the old position and opens the next: each trip pays for its own part of it
                        tot = abs(fl["weight_before"]) + abs(fl["weight_after"])
                        share = abs(fl["weight_before"]) / tot if closed and fl["date"] == h.index[end] else abs(fl["weight_after"]) / tot
                    cost += float(fl["cost"]) * share
                opening = f.iloc[0] if len(f) else None
                closing = f.iloc[-1] if (len(f) and closed) else None
                if closing is not None and " + " in str(closing.get("action")):                         # a flip: the close is the first leg, the open of the next trip the second
                    closing = closing.copy(); closing["action"] = str(closing["action"]).split(" + ")[0]; closing["reason"] = str(closing["reason"]).split(" -> ")[0]
                if opening is not None and " + " in str(opening.get("action")) and str(opening.get("kind", "")).startswith("FLIP"):
                    opening = opening.copy(); opening["reason"] = str(opening["reason"]).split(" -> ")[-1]
                rows.append({"asset": asset, "side": "LONG" if side > 0 else "SHORT", "open_date": h.index[i], "close_date": h.index[end] if closed else pd.NaT, "status": "closed" if closed else "open",
                             "holding_days": int(end - i), "peak_weight": float(np.abs(w[i:end + 1]).max()), "position_return": pos_ret, "contribution": contrib, "cost": cost,
                             "net_contribution": contrib - cost, "open_reason": None if opening is None else opening.get("reason"), "close_reason": None if closing is None else closing.get("reason"),
                             "close_action": None if closing is None else closing.get("action")})
                i = end if closed else n
            else:
                i += 1
    out = pd.DataFrame(rows)
    return out.sort_values(["open_date", "asset"]).reset_index(drop=True) if len(out) else out


def analytics(trips: pd.DataFrame, held: pd.DataFrame, net: pd.Series, gross: pd.Series, turnover: pd.Series, periods: int = 252) -> dict:
    """The trading statistics: hit rate and payoff, holding times, exposure and cash, turnover, and the gross-to-net gap."""
    closed = trips[trips["status"] == "closed"] if len(trips) else trips
    out: dict = {"n_round_trips": int(len(closed)), "n_open": int((trips["status"] == "open").sum()) if len(trips) else 0}
    if len(closed):
        win = closed["position_return"] > 0
        gains, losses = closed.loc[win, "position_return"], closed.loc[~win, "position_return"]
        gp, gl = float(closed.loc[closed["net_contribution"] > 0, "net_contribution"].sum()), float(-closed.loc[closed["net_contribution"] < 0, "net_contribution"].sum())
        out.update({"win_rate": float(win.mean()), "avg_win": float(gains.mean()) if len(gains) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                    "payoff_ratio": float(gains.mean() / -losses.mean()) if len(gains) and len(losses) and losses.mean() < 0 else None,
                    "profit_factor": float(gp / gl) if gl > 0 else None, "expectancy": float(closed["position_return"].mean()),
                    "avg_holding_days": float(closed["holding_days"].mean()), "median_holding_days": float(closed["holding_days"].median()),
                    "holding_histogram": _hist(closed["holding_days"]), "long_trips": int((closed["side"] == "LONG").sum()), "short_trips": int((closed["side"] == "SHORT").sum()),
                    "best": _trip(closed.loc[closed["position_return"].idxmax()]), "worst": _trip(closed.loc[closed["position_return"].idxmin()])})
        by = closed.groupby("asset").agg(trips=("position_return", "size"), win_rate=("position_return", lambda s: float((s > 0).mean())), contribution=("net_contribution", "sum"),
                                         avg_holding=("holding_days", "mean"))
        out["by_asset"] = [{"asset": a, **{k: float(v) for k, v in row.items()}} for a, row in by.sort_values("contribution", ascending=False).iterrows()]
    h = held.fillna(0.0)
    gross_exp, net_exp = h.abs().sum(axis=1), h.sum(axis=1)
    out.update({"avg_gross_exposure": float(gross_exp.mean()), "avg_net_exposure": float(net_exp.mean()), "avg_cash": float((1.0 - gross_exp).clip(lower=0.0).mean()),
                "avg_positions": float((h.abs() > 1e-9).sum(axis=1).mean()), "pct_days_invested": float((gross_exp > 1e-9).mean()), "ann_turnover": float(turnover.mean() * periods),
                "gross_ann_return": float(gross.dropna().mean() * periods), "net_ann_return": float(net.dropna().mean() * periods)})
    out["cost_drag_ann"] = out["gross_ann_return"] - out["net_ann_return"]
    return out


def _hist(days: pd.Series, edges=(1, 2, 5, 10, 20, 40, 80, 160)) -> list[dict]:
    d = days.to_numpy()
    lo, rows = 0, []
    for e in edges:
        rows.append({"label": f"{lo + 1}-{e}", "count": int(((d > lo) & (d <= e)).sum())})
        lo = e
    rows.append({"label": f">{lo}", "count": int((d > lo).sum())})
    return rows


def _trip(row) -> dict:
    return {"asset": row["asset"], "side": row["side"], "open_date": str(pd.Timestamp(row["open_date"]).date()), "position_return": float(row["position_return"]), "holding_days": int(row["holding_days"])}


def _label_rule_trades(fills: pd.DataFrame, scores: pd.DataFrame | None, method: str) -> pd.DataFrame:
    """Label the trades of a run whose position rule keeps no decision log (every rule except ``decision_policy``): BUY or SELL, the model's score and its rank on the day the book was decided, and a reason
    that says what the rule did. The rule turned the scores into a target book; each trade is the change in that book, so its reason is the change and the rank behind it."""
    if not len(fills):
        return fills
    out = fills.copy()
    out["action"] = np.where(out["side"] == "BUY", "BUY", "SELL")
    if scores is not None and len(scores):
        rank = scores.rank(axis=1, method="average", pct=True)                                 # 1 = the best score of the day
        n_valid = scores.notna().sum(axis=1)
        sc, rk, nv = [], [], []
        for d, a in zip(pd.to_datetime(out["decision_date"]), out["asset"]):
            ok = d in scores.index and a in scores.columns
            sc.append(float(scores.at[d, a]) if ok else np.nan)
            rk.append(float(rank.at[d, a]) if ok else np.nan)
            nv.append(int(n_valid.get(d, 0)) if ok else 0)
        out["score"], out["rank_pct"] = sc, rk
    else:
        out["score"], out["rank_pct"], nv = np.nan, np.nan, [0] * len(out)
    reasons = []
    for r, n in zip(out.itertuples(), nv):
        where = f"; its score ranked it in the top {max(1, round((1 - r.rank_pct) * n + 1))} of {n}" if np.isfinite(r.rank_pct) and n else ""
        reasons.append(f"{method} moved the target weight from {r.weight_before:.1%} to {r.weight_after:.1%} ({r.kind.replace('_', ' ').lower()}){where}")
    out["reason"], out["confidence"] = reasons, np.nan
    return out


def build(decisions: pd.DataFrame | None, parts: dict, returns: pd.DataFrame, cfg: dict, lag: int | None = None, scores: pd.DataFrame | None = None) -> dict:
    """Everything the Strategy Explorer needs for one run: the decision log, the executed trades, the round trips and a compact summary (the analytics, counts and the policy in force).

    ``parts`` is ``trim_to_signal``'s result (the book, the trades and the return streams from the first day a position could earn); ``lag`` the total delay between decision and execution.
    ``decisions`` is None for a position rule that keeps no decision log: the ledger is then built from the book alone and each trade is labelled from ``scores`` (``_label_rule_trades``).
    """
    from . import decisions as dc

    bt, sig = cfg["backtest"], cfg["signal"]
    lag = int(sig.get("lag_total", 1)) if lag is None else int(lag)
    cost_bps = float(bt["cost_bps"]) + float(bt["slippage_bps"])
    held, trades = parts["weights"], parts["trades"]
    rule_based = decisions is None
    fills = executions(held, trades, returns, cost_bps, decisions, lag)
    if rule_based:
        fills = _label_rule_trades(fills, scores, str(sig.get("method")))
    trips = round_trips(held, returns, fills, cost_bps)
    gone = unexecuted(decisions, fills)
    summary = {"version": VERSION, "policy": dc.policy_from(sig) if not rule_based else {"method": str(sig.get("method"))}, "decisions": dc.summarise(decisions), "n_executions": int(len(fills)),
               "n_unexecuted": int(len(gone)), "lag_days": lag, "cost_bps": cost_bps, "rebalance": str(bt["rebalance"]), "rule_based": rule_based,
               "analytics": analytics(trips, held, parts["net"], parts["gross"], parts["turnover"])}
    if rule_based:
        summary["note"] = (f"The position rule '{sig.get('method')}' turns the scores into a target book and keeps no decision log: each buy and sell below is a change in that book, labelled with the model's score and "
                           "rank on the day it was decided. Use the decision_policy rule for explicit entries, exits, stops and a decision log.")
    elif str(bt["rebalance"]) != "on_change":
        summary["note"] = (f"The book trades on a {bt['rebalance']} schedule, so a decision made between rebalance dates waits for the next one: an event-driven policy (stops, time exits) is exact only with "
                           "backtest.rebalance = on_change.")
    return {"decisions": decisions, "executions": fills, "round_trips": trips, "unexecuted": gone, "summary": summary}
