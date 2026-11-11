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
