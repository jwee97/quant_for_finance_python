"""The signal book: what a set of rules says today, what changed since the last time it was asked, and how the signals it issued have done since.

A *follow* is a market (stocks and ETFs, futures, crypto or options), the assets and the rules to watch. Running it reads the rules on the last day of the data (the same rule boards the Futures and
Crypto tabs show, through ``src.assets.strategy_runs.signals_board``) and makes a *snapshot*: one row per rule and asset with the rule's score, its strength (the score over the rule's own typical
size, so 1 is a typical position and the sign is the side) and the side. Snapshots are kept in an append-only ledger (:mod:`src.signalbook.store`), and that ledger is what makes the record honest:

* ``changes`` compares a snapshot with the one before it and lists every side that flipped;
* ``track_record`` scores every past snapshot against the prices that came *after* it was issued. A signal read at the close of day ``d`` is entered at the close of ``d + 1`` and held ``h`` days, the
  same one-day lag the backtests use, so nothing in the record could not have been traded. A signal whose horizon has not passed yet is not counted.

A backtest says what a rule would have done on data it has already seen. The record here only ever contains signals that were written down before their outcome existed, which is why it starts empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRONG = 0.25                                             # a strength this far from zero counts as a view (the boards' own limit)
HORIZONS = (1, 5, 21)
STALE_AFTER = {"equity": 3, "futures": 3, "options": 3, "crypto": 2}       # days without a new bar before the data is called stale (business days, calendar days for crypto)
MARKETS = {"equity": "Stocks and ETFs", "futures": "Futures", "crypto": "Crypto", "options": "Options"}

EQUITY_BOARD = (
    ("Trend, one year", "tsmom", {}), ("Trend, six speeds (Carver)", "carver_ewmac", {}), ("Trend, t-statistic", "tstat_momentum", {}), ("Trend, three horizons", "tsmom_multi", {}),
    ("Trend, moving averages", "ma_ensemble", {}), ("Trend, adaptive", "kama_trend", {}), ("Momentum against the others", "xs_momentum", {}), ("Momentum net of the market", "residual_momentum", {}),
    ("Reversal, one month", "short_term_reversal", {}), ("Mean reversion", "mean_reversion", {}), ("Low volatility", "low_volatility", {}), ("Volatility-managed hold", "vol_managed_long", {}),
)
OPTION_RULES = (
    ("vol_rank", "Implied volatility in its one-year range", "Above 80% of the range implied volatility is dear (short volatility); below 20% it is cheap (long volatility). A real chain is one snapshot and has no range."),
    ("vol_premium", "Implied against realised volatility", "Implied below 0.9 times the last month's realised volatility: options are cheap (long volatility). Above 1.3 times: dear (short volatility)."),
    ("term_stress", "Term structure", "The front of the volatility curve above the back, as in stress: long volatility. Otherwise no view."),
)


def side(strength) -> str:
    if strength is None or not np.isfinite(strength) or abs(strength) <= STRONG:
        return "flat"
    return "long" if strength > 0 else "short"


def board_rows(board: dict, price_key=None) -> tuple[list[dict], list[dict]]:
    """``(rows, skipped)`` from a ``signals_board`` result: one row per rule and asset, and the rules that could not be computed with why. ``price_key(rule_id, asset)`` names the price column the
    signal is scored against later (the asset itself by default)."""
    rows, skipped = [], []
    for rule in board["signals"]:
        if "strength" not in rule:
            skipped.append({"rule": rule["id"], "label": rule["label"], "why": rule.get("error", "not computed")})
            continue
        for asset in board["assets"]:
            s = rule["strength"].get(asset)
            rows.append({"rule": rule["id"], "label": rule["label"], "asset": asset, "score": rule["score"].get(asset), "strength": s, "side": side(s),
                         "price_key": price_key(rule["id"], asset) if price_key else asset})
    return rows, skipped


def option_rows(overview: dict) -> tuple[list[dict], list[dict]]:
    """The volatility signals of one underlying from the Options tab's overview. The side is the side of *volatility* (long: own options; short: sell them), and nothing here is scored later: a
    volatility position's result needs a history of option prices, which one day's chain does not have."""
    asset, rows, skipped = overview["symbol"], [], []

    def add(rule, strength, score):
        label = next(r[1] for r in OPTION_RULES if r[0] == rule)
        rows.append({"rule": rule, "label": label, "asset": asset, "score": None if score is None or not np.isfinite(score) else float(score), "strength": float(strength), "side": side(strength), "price_key": None})

    rank = overview.get("iv_rank")
    if rank is None or not np.isfinite(rank):
        skipped.append({"rule": "vol_rank", "label": OPTION_RULES[0][1], "why": overview.get("iv_rank_basis") or "no history of implied volatility"})
    else:
        add("vol_rank", -1.0 if rank > 0.8 else 1.0 if rank < 0.2 else 0.0, rank)
    iv, rv = overview.get("atm_iv_30d"), overview.get("realized_vol_21d")
    if iv is None or rv is None or not np.isfinite(iv) or not np.isfinite(rv) or rv <= 0:
        skipped.append({"rule": "vol_premium", "label": OPTION_RULES[1][1], "why": "no 30-day implied volatility or no realised volatility for this underlying"})
    else:
        add("vol_premium", 1.0 if iv < 0.9 * rv else -1.0 if iv > 1.3 * rv else 0.0, iv / rv)
    add("term_stress", 1.0 if overview.get("term_inverted") else 0.0, overview.get("term_slope"))
    return rows, skipped


def freshness(market: str, last_day, today, simulated: bool = False) -> dict:
    """How old the data behind a snapshot is. ``age_days`` counts the days after ``last_day`` up to ``today`` on which a new bar was due (business days; every day for crypto)."""
    last, now = pd.Timestamp(last_day).normalize(), pd.Timestamp(today).normalize()
    if now <= last:
        age = 0
    elif market == "crypto":
        age = int((now - last).days)
    else:
        age = int(np.busday_count(last.date(), now.date()))
    limit = STALE_AFTER.get(market, 3)
    problems = []
    if simulated:
        problems.append("These signals are read on simulated data, not a real market: use them to see how the board reads, not to trade.")
    if age > limit:
        problems.append(f"The data ends on {last.date()}, {age} {'days' if market == 'crypto' else 'trading days'} ago: these are that day's signals, not today's. Refresh the data before acting on them.")
    return {"last_day": str(last.date()), "age_days": age, "stale": age > limit, "simulated": bool(simulated), "problems": problems}


def consensus(rows: list[dict]) -> list[dict]:
    """For each asset, how many rules are long, short and without a view, and the average strength of the rules that were computed."""
    out = {}
    for r in rows:
        c = out.setdefault(r["asset"], {"asset": r["asset"], "long": 0, "short": 0, "flat": 0, "_s": []})
        c[r["side"]] += 1
        if r["strength"] is not None and np.isfinite(r["strength"]):
            c["_s"].append(r["strength"])
    result = []
    for c in out.values():
        s = c.pop("_s")
        n = c["long"] + c["short"] + c["flat"]
        c["net"] = (c["long"] - c["short"]) / n if n else 0.0
        c["mean_strength"] = float(np.mean(s)) if s else None
        c["view"] = "long" if c["net"] >= 0.5 else "short" if c["net"] <= -0.5 else "leaning long" if c["net"] >= 0.2 else "leaning short" if c["net"] <= -0.2 else "mixed"
        result.append(c)
    return result


def changes(previous: dict | None, current: dict) -> list[dict]:
    """Every rule and asset whose side differs from the snapshot before. The first snapshot of a follow has no changes: there is nothing to compare it with."""
    if previous is None:
        return []
    before = {(r["rule"], r["asset"]): r for r in previous["rows"]}
    out = []
    for r in current["rows"]:
        old = before.get((r["rule"], r["asset"]))
        if old is not None and old["side"] != r["side"]:
            out.append({"rule": r["rule"], "label": r["label"], "asset": r["asset"], "from": old["side"], "to": r["side"], "strength": r["strength"], "was": old["strength"]})
    return out


def track_record(snapshots: list[dict], prices: pd.DataFrame, horizons=HORIZONS) -> dict:
    """How the signals of past ``snapshots`` (of one follow, oldest first) did on ``prices`` (a frame of closes that reaches past them). One snapshot per data day counts (the first issued for that
    day). A long or short signal read at the close of day ``d`` is entered at the next close and held ``h`` days; its result is the asset's return over those days, signed by the side. ``baseline``
    is the same entries held long whatever the signal said: a rule earns its keep only by beating that. A snapshot issued after the close it would have been entered at (a run on data that was
    already days old) is left out and counted in ``late``: the record holds only signals that were written down before their outcome existed."""
    cols = {c: prices[c].to_numpy(dtype=float) for c in prices.columns}
    index = prices.index
    seen, per = set(), {}
    counted = pending = late = 0
    first = last = None
    for snap in snapshots:
        day = pd.Timestamp(snap["as_of"])
        if snap["as_of"] in seen:
            continue
        seen.add(snap["as_of"])
        i = int(index.searchsorted(day, side="right")) - 1
        if i < 0:
            continue
        if i + 1 < len(index) and index[i + 1] < pd.Timestamp(str(snap["issued"])[:10]):
            late += 1                                                                               # written down after its entry close had already happened: not a signal anyone could have traded
            continue
        counted += 1
        first, last = first or snap["as_of"], snap["as_of"]
        for r in snap["rows"]:
            key = r.get("price_key")
            if r["side"] == "flat" or key is None or key not in cols:
                continue
            sign, p = (1.0 if r["side"] == "long" else -1.0), cols[key]
            for h in horizons:
                if i + 1 + h >= len(p):
                    pending += 1
                    continue
                a, b = p[i + 1], p[i + 1 + h]
                if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
                    continue
                ret = b / a - 1.0
                for name in (r["rule"], "__all__"):
                    slot = per.setdefault(name, {"label": r["label"] if name != "__all__" else "Every rule", "by": {}})["by"].setdefault(h, {"signed": [], "long": []})
                    slot["signed"].append(sign * ret)
                    slot["long"].append(ret)

    def summary(entry):
        out = {}
        for h, v in entry["by"].items():
            s, base = np.asarray(v["signed"]), np.asarray(v["long"])
            out[str(h)] = {"n": int(len(s)), "hit_rate": float((s > 0).mean()), "mean": float(s.mean()), "baseline": float(base.mean()), "edge": float(s.mean() - base.mean())}
        return out

    rules = [{"rule": k, "label": v["label"], "horizons": summary(v)} for k, v in per.items() if k != "__all__"]
    overall = summary(per["__all__"]) if "__all__" in per else {}
    return {"snapshots": counted, "first": first, "last": last, "horizons": list(horizons), "overall": overall, "rules": sorted(rules, key=lambda r: r["label"]), "pending": pending, "late": late,
            "note": "Signals a day apart overlap when they are held for more than a day, so the count overstates how many independent bets there were."}


def snapshot(follow: dict, rows: list[dict], skipped: list[dict], fresh: dict, issued: str, source: str) -> dict:
    return {"follow": follow["id"], "market": follow["market"], "issued": issued, "as_of": fresh["last_day"], "source": source, "data": fresh, "rows": rows, "skipped": skipped}


# ------------------------------------------------------------------------------------------------------------------ the paper portfolio
PAPER_DEFAULTS = {"cost_bps": 5.0, "long_only": False, "capital": 100_000.0}
MIN_DAYS_FOR_SHARPE = 60


def target_weights(rows: list[dict], long_only: bool = False) -> dict:
    """What a snapshot says to hold: for each instrument the share of its rules that are long less the share that are short, scaled so the sizes add up to 1 (fully invested, no leverage).
    ``long_only`` drops the shorts before scaling. Signals with nothing to score them against (options) are left out; a snapshot with no view holds cash."""
    tally = {}
    for r in rows:
        key = r.get("price_key")
        if key is None:
            continue
        t = tally.setdefault(key, [0, 0])
        t[0] += (r["side"] == "long") - (r["side"] == "short")
        t[1] += 1
    net = {k: v[0] / v[1] for k, v in tally.items() if v[1]}
    if long_only:
        net = {k: max(v, 0.0) for k, v in net.items()}
    gross = sum(abs(v) for v in net.values())
    return {k: (v / gross if gross > 0 else 0.0) for k, v in net.items()}


def paper_portfolio(snapshots: list[dict], prices: pd.DataFrame, cost_bps: float = 5.0, long_only: bool = False, capital: float = 100_000.0) -> dict:
    """The follow traded on paper, from the ledger alone. Each snapshot that could have been traded (the first of its data day, issued before its entry close, as in ``track_record``) sets the
    target weights; they are bought at the close after the snapshot's data day and held until the next snapshot's entry. Every change of weight pays ``cost_bps`` of what is traded. ``baseline`` is
    the same instruments held in equal weights over the same days. ``target`` is what the latest snapshot says to hold and ``trades`` what that changes, whether or not it could be scored yet."""
    index = prices.index
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seen, plan = set(), []
    for snap in snapshots:
        if snap["as_of"] in seen:
            continue
        seen.add(snap["as_of"])
        i = int(index.searchsorted(pd.Timestamp(snap["as_of"]), side="right")) - 1
        if i < 0 or i + 1 >= len(index) or index[i + 1] < pd.Timestamp(str(snap["issued"])[:10]):
            continue                                                                                # not yet enterable, or written down after its entry close
        plan.append((i + 1, target_weights(snap["rows"], long_only)))
    latest = snapshots[-1] if snapshots else None
    target = target_weights(latest["rows"], long_only) if latest else {}
    held = plan[-1][1] if plan else {}
    keys = sorted(set(target) | set(held))
    trades = [{"asset": k, "from": held.get(k, 0.0), "to": target.get(k, 0.0), "change": target.get(k, 0.0) - held.get(k, 0.0), "amount": (target.get(k, 0.0) - held.get(k, 0.0)) * capital} for k in keys]
    out = {"cost_bps": float(cost_bps), "long_only": bool(long_only), "capital": float(capital), "target": [{"asset": k, "weight": target[k], "amount": target[k] * capital} for k in sorted(target)],
           "trades": [t for t in trades if abs(t["change"]) > 1e-9], "target_as_of": latest["as_of"] if latest else None, "days": 0, "curve": None, "stats": None,
           "note": "Sizes are the net share of rules long or short, scaled to be fully invested without leverage. Fills are at the close, with no slippage beyond the cost per trade, no borrowing cost for shorts and no margin."}
    if not plan or plan[0][0] + 1 >= len(index):
        return out
    start = plan[0][0]
    cols = [c for c in prices.columns if any(c in w for _, w in plan)]
    days = range(start + 1, len(index))
    value, base, turnover, costs = [1.0], [1.0], 0.0, 0.0
    k, current, previous = 0, plan[0][1], {}
    pending_cost = sum(abs(current.get(c, 0.0)) for c in cols) * cost_bps / 1e4
    turnover += sum(abs(current.get(c, 0.0)) for c in cols)
    for t in days:
        r = sum(current.get(c, 0.0) * float(rets[c].iloc[t]) for c in cols) - pending_cost
        costs += pending_cost
        pending_cost = 0.0
        value.append(value[-1] * (1.0 + r))
        base.append(base[-1] * (1.0 + float(np.mean([rets[c].iloc[t] for c in cols])) if cols else base[-1]))
        while k + 1 < len(plan) and plan[k + 1][0] <= t:                                              # the next snapshot's weights are bought at this close
            k += 1
            previous, current = current, plan[k][1]
            traded = sum(abs(current.get(c, 0.0) - previous.get(c, 0.0)) for c in cols)
            turnover += traded
            pending_cost += traded * cost_bps / 1e4
    daily = np.diff(value) / np.asarray(value[:-1])
    n = len(daily)
    peak = np.maximum.accumulate(value)
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if n >= MIN_DAYS_FOR_SHARPE and daily.std(ddof=1) > 0 else None
    out.update(days=n, curve={"dates": [str(index[start].date())] + [str(index[t].date()) for t in days], "portfolio": [float(v) for v in value], "baseline": [float(v) for v in base]},
               stats={"total_return": float(value[-1] - 1.0), "baseline_return": float(base[-1] - 1.0), "volatility": float(daily.std(ddof=1) * np.sqrt(252)) if n > 1 else None, "sharpe": sharpe,
                      "max_drawdown": float(np.min(np.asarray(value) / peak - 1.0)), "turnover": float(turnover), "costs": float(costs), "rebalances": len(plan), "value": float(value[-1] * capital),
                      "sharpe_needs_days": MIN_DAYS_FOR_SHARPE})
    return out
