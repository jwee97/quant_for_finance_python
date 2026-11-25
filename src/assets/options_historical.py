"""An EMPIRICAL options backtest on the dated chains the platform has stored (``assets.option_history``), replayed at their own bid and ask.

Nothing here is modelled: a position is opened at the touch of the chain of a decision date (buy at the ask, sell at the bid), marked on later dates at the exit touch of the SAME contract in that date's
chain, and settled at intrinsic value from the underlying price of the first chain on or after expiry. A contract that is absent or has no two-sided market on a later date is counted as unmarkable and is
never priced with a model; a trade whose expiry has no later chain stays open and is reported as such. Snapshots whose source is simulated are refused. It needs at least ``MIN_DATES`` dated chains.

Not modelled (listed in the result): American early exercise and assignment, corporate-action contract adjustments, margin financing, intraday fills, volume limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import options_analytics as oa

MIN_DATES = 20
MULT = oa.MULTIPLIER
NOT_MODELLED = ["American early exercise and assignment", "contract adjustments for corporate actions", "interest on margin and cash", "intraday fills and volume limits", "dividend assignment risk of short calls"]


def _unavailable(reason: str, needs: list[str] | None = None) -> dict:
    return {"state": "unavailable", "reason": reason, "needs": needs or []}


def _quote(chain: pd.DataFrame, expiry, strike: float, right: str):
    sel = chain[(chain["expiry"] == expiry) & (chain["right"] == right) & ((chain["strike"] - strike).abs() < 1e-9)]
    if sel.empty:
        return None
    r = sel.iloc[0]
    return (float(r["bid"]), float(r["ask"])) if r["bid"] > 0 and r["ask"] > 0 else None


def run(history, underlying: str, structure: str = "cash_secured_put", dte: int = 30, every: int = 7, quantity: int = 1, commission: float = 0.65, train_share: float = 0.6, min_dates: int = MIN_DATES) -> dict:
    dates = history.dates(underlying)
    meta = {m["date"]: m for m in history.meta(underlying)}
    sims = sorted(d for d in dates if str(meta.get(d, {}).get("source")) in ("simulated", "model"))
    if sims:
        return _unavailable(f"{len(sims)} of the stored chains are simulated; an empirical backtest uses observed chains only", ["observed daily snapshots (Snapshot button on a downloaded or Massive chain)"])
    if len(dates) < min_dates:
        return _unavailable(f"{len(dates)} dated chains of {underlying} are stored; an empirical backtest needs at least {min_dates}", ["a daily snapshot of the chain on each trading day, or a vendor history with historical bid and ask quotes"])
    if structure not in oa.STRATEGY_INFO:
        raise ValueError(f"unknown structure '{structure}'")
    chains = {pd.Timestamp(d): history.load(underlying, d) for d in dates}
    days = sorted(chains)
    spot = {d: float(c["underlying"].iloc[0]) for d, c in chains.items()}
    trades, last_open = [], None
    for d in days:
        if last_open is not None and (d - last_open).days < every:
            continue
        chain = oa.prepare_chain(chains[d])
        try:
            legs, meta_c = oa.named_structure(chain, spot[d], structure, dte=dte, quantity=quantity)
        except (ValueError, KeyError, IndexError):
            continue
        entry, resolved, ok = 0.0, [], True
        for l in legs:
            if l["type"] != "option":
                ok = False
                break
            e = oa.nearest_expiry_to(chain, l["dte"])
            q = _quote(chain, e, float(l["strike"]), l["right"])
            if q is None:
                ok = False
                break
            px = q[1] if l["quantity"] > 0 else q[0]
            entry += l["quantity"] * MULT * px
            resolved.append({"expiry": pd.Timestamp(e), "strike": float(l["strike"]), "right": l["right"], "qty": float(l["quantity"]), "entry": px})
        if not ok or not resolved:
            continue
        fees = sum(abs(r["qty"]) for r in resolved) * commission
        last_open = d
        end = max(r["expiry"] for r in resolved)
        later = [x for x in days if x > d]
        marks, unmarkable = [], 0
        for x in later:
            if x > end:
                break
            val, good = 0.0, True
            for r in resolved:
                if x >= r["expiry"]:
                    s = spot[x]
                    val += r["qty"] * MULT * (max(s - r["strike"], 0.0) if r["right"] == "C" else max(r["strike"] - s, 0.0))
                    continue
                q = _quote(chains[x], r["expiry"], r["strike"], r["right"])
                if q is None:
                    good = False
                    break
                val += r["qty"] * MULT * (q[0] if r["qty"] > 0 else q[1])
            if good:
                marks.append((x, val))
            else:
                unmarkable += 1
        settle = next((x for x in later if x >= end), None)
        settled = settle is not None
        final = None
        if settled:
            final = sum(r["qty"] * MULT * (max(spot[settle] - r["strike"], 0.0) if r["right"] == "C" else max(r["strike"] - spot[settle], 0.0)) for r in resolved)
        trades.append({"open": str(d.date()), "expiry": str(end.date()), "structure": structure, "entry_debit": float(entry), "fees": float(fees), "settled": bool(settled), "settle_date": None if not settled else str(settle.date()),
                       "pnl": None if not settled else float(final - entry - 2 * fees), "marks": len(marks), "unmarkable_days": unmarkable,
                       "last_mark_pnl": None if not marks else float(marks[-1][1] - entry - fees), "legs": [{**r, "expiry": str(r["expiry"].date())} for r in resolved]})
    done = [t for t in trades if t["settled"]]
    if len(done) < 3:
        return _unavailable(f"only {len(done)} trades opened and settled within the stored chains; the stored period is too short for this structure and holding period", ["a longer run of daily snapshots"])
    p = np.array([t["pnl"] for t in done])
    cut = max(int(len(done) * train_share), 1)
    part = lambda x: {"trades": int(len(x)), "total": float(np.sum(x)), "mean": float(np.mean(x)), "win_rate": float(np.mean(np.array(x) > 0)), "worst": float(np.min(x))} if len(x) else None          # noqa: E731
    return {"state": "ok", "underlying": underlying, "structure": structure, "dte": dte, "snapshots": len(days), "first": str(days[0].date()), "last": str(days[-1].date()), "mode": {"id": "empirical", "label": "Empirical backtest on stored observed chains"},
            "trades": done, "open_trades": [t for t in trades if not t["settled"]], "summary": part(p), "in_sample": part(p[:cut]), "out_of_sample": part(p[cut:]), "unmarkable_days": int(sum(t["unmarkable_days"] for t in trades)),
            "split_note": f"the first {cut} settled trades are the in-sample period and the rest out of sample; no setting is fitted here, so the split shows stability, not selection",
            "not_modelled": NOT_MODELLED, "fills": "buy at the ask, sell at the bid of the stored chain, plus commissions on entry and exit; marks use the opposite touch"}
