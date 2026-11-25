"""Historical REPLAY of recorded crypto observations: funding carry, cross-exchange spreads, dated-futures basis and liquidity-aware execution.

A replay uses ONLY what ``assets.crypto_store`` recorded. At every decision time it sees the latest valid book received at or before that time and no older than the staleness limit; nothing received later is visible.
It never manufactures a missing order book, funding settlement or futures observation: a decision time without the data it needs is skipped and counted by reason, and the coverage block says how much of the
period was usable. This is not the hypothetical execution simulator (``crypto_research.simulate_execution``), which walks one stored book with ASSUMED replenishment: a replay is labelled ``historical_replay`` and its
fills are against books that existed.

Every result carries gross P&L, fees, funding, financing, slippage, net P&L, maximum drawdown, turnover, exposure, the data coverage and a transaction-level audit trail.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .crypto_research import PERIODS, walk_book

MODE = {"id": "historical_replay", "label": "Replay of recorded observations", "hypothetical": False}


class BookSeries:
    """The valid book rows of one instrument, for as-of reads without look-ahead."""

    def __init__(self, df: pd.DataFrame):
        d = df[df["quality"] == "ok"].sort_values("recv_ts_ms").reset_index(drop=True) if len(df) else df
        self.df, self.t = d, (d["recv_ts_ms"].to_numpy() if len(d) else np.array([], dtype=np.int64))

    def asof(self, t_ms: int, max_age_s: float) -> dict | None:
        i = int(np.searchsorted(self.t, t_ms, side="right")) - 1
        if i < 0 or t_ms - int(self.t[i]) > max_age_s * 1000:
            return None
        r = self.df.iloc[i]
        return {"recv_ts_ms": int(r["recv_ts_ms"]), "exchange_ts_ms": None if pd.isna(r["exchange_ts_ms"]) else int(r["exchange_ts_ms"]), "sequence": None if pd.isna(r["sequence"]) else int(r["sequence"]),
                "bids": list(zip(r["bids_px"], r["bids_qty"])), "asks": list(zip(r["asks_px"], r["asks_qty"])), "age_s": (t_ms - int(r["recv_ts_ms"])) / 1000.0}

    def after(self, t_ms: int, max_wait_s: float) -> dict | None:
        """The first valid book received after ``t_ms`` within ``max_wait_s``: what an order sent at ``t_ms`` would meet when it arrives (used for the second leg)."""
        i = int(np.searchsorted(self.t, t_ms, side="right"))
        if i >= len(self.t) or int(self.t[i]) - t_ms > max_wait_s * 1000:
            return None
        return self.asof(int(self.t[i]), 0.001)


def _mid(b: dict) -> float:
    return 0.5 * (b["bids"][0][0] + b["asks"][0][0])


def _drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    e = np.array(equity, float)
    return float((e - np.maximum.accumulate(e)).min())


def _summary(trades: list[dict], equity: list[float], capital: float, gross_exposure: list[float], span_s: float) -> dict:
    keys = ("gross", "fees", "funding", "financing", "slippage")
    tot = {k: float(sum(t.get(k, 0.0) for t in trades)) for k in keys}
    net = tot["gross"] - tot["fees"] + tot["funding"] - tot["financing"]
    traded = float(sum(t.get("notional", 0.0) for t in trades))
    return {"gross_pnl": tot["gross"], "fees": tot["fees"], "funding": tot["funding"], "financing": tot["financing"], "slippage_included_in_gross": tot["slippage"], "net_pnl": float(net), "max_drawdown": _drawdown(equity),
            "turnover": traded / capital if capital else None, "mean_exposure": float(np.mean(gross_exposure)) / capital if gross_exposure and capital else 0.0, "trades": len(trades), "span_days": span_s / 86400.0,
            "net_return_annualised": float(net / capital * PERIODS * 86400.0 / span_s) if span_s > 0 and capital else None}


def _unavailable(reason: str, needs: list[str]) -> dict:
    return {"state": "unavailable", "reason": reason, "needs": needs, "mode": MODE}


# ------------------------------------------------------------------------------------------------------------------ cross-exchange
def cross_exchange(store, a: tuple, b: tuple, notional: float = 10_000.0, fee_bps: tuple = (10.0, 10.0), max_age_s: float = 5.0, max_skew_s: float = 2.0, min_edge_bps: float = 0.0, leg_delay_s: float = 1.0,
                   inventory_usd: float = 100_000.0, start=None, end=None) -> dict:
    """Spot-versus-spot (or perpetual-versus-perpetual) spread replay between recorded books ``a`` and ``b`` (``(venue, instrument)``). At each recorded book of ``a`` the decision sees both books at that moment
    (each no older than ``max_age_s`` and received within ``max_skew_s`` of the other); the better direction is traded if its net edge after both fees at the decision-time depth is at least ``min_edge_bps``. The first
    leg fills at the decision-time book; the second leg meets the first book of its venue received after ``leg_delay_s``: if there is none the position is left unhedged and counted, never assumed closed. Each venue
    holds ``inventory_usd`` of cash and the same of coin at the first price: a trade is limited by what the selling venue holds and the buying venue can pay, and balances are not rebalanced (no transfers)."""
    A, B = BookSeries(store.books(*a, start=start, end=end)), BookSeries(store.books(*b, start=start, end=end))
    if A.df is None or not len(A.df) or not len(B.df):
        return _unavailable("one of the two instruments has no valid recorded book in the period", ["run the recorder (`python -m src.cli crypto-record`) on both venues for the period"])
    skipped = {"no_book_b": 0, "stale_or_skewed": 0, "no_edge": 0, "no_inventory": 0, "leg2_unavailable": 0}
    cash = {"a": inventory_usd, "b": inventory_usd}
    first_mid = _mid(A.asof(int(A.t[0]), 1e9))
    coin = {"a": inventory_usd / first_mid, "b": inventory_usd / first_mid}
    trades, equity, expo = [], [], []
    for t in A.t:
        t = int(t)
        ba, bb = A.asof(t, max_age_s), B.asof(t, max_age_s)
        if ba is None or bb is None:
            other, series = (bb, B) if bb is None else (ba, A)
            skipped["stale_or_skewed" if series.asof(t, 1e12) is not None else "no_book_b"] += 1                 # an older book exists (stale) or none had been recorded yet
            continue
        if abs(ba["recv_ts_ms"] - bb["recv_ts_ms"]) > max_skew_s * 1000:
            skipped["stale_or_skewed"] += 1
            continue
        best = None
        for buy_on, sell_on, bbuy, bsell, fb, fs in (("a", "b", ba, bb, fee_bps[0], fee_bps[1]), ("b", "a", bb, ba, fee_bps[1], fee_bps[0])):
            fill = walk_book(bbuy["asks"], notional=min(notional, cash[buy_on]))
            if fill["coins"] <= 0:
                continue
            qty = min(fill["coins"], coin[sell_on])
            if qty <= 0:
                continue
            buy_q, sell_q = walk_book(bbuy["asks"], qty=qty), walk_book(bsell["bids"], qty=qty)
            if sell_q["unfilled_coins"] and sell_q["unfilled_coins"] > 1e-12:
                qty = sell_q["coins"]
                buy_q, sell_q = walk_book(bbuy["asks"], qty=qty), walk_book(bsell["bids"], qty=qty)
            edge = (sell_q["dollars"] * (1 - fs / 1e4) - buy_q["dollars"] * (1 + fb / 1e4)) / buy_q["dollars"] * 1e4 if buy_q["dollars"] > 0 else -1e9
            if best is None or edge > best["edge"]:
                best = {"edge": edge, "buy_on": buy_on, "sell_on": sell_on, "qty": qty, "bbuy": bbuy, "bsell": bsell, "fb": fb, "fs": fs}
        if best is None:
            skipped["no_inventory"] += 1
            continue
        if best["edge"] < min_edge_bps:
            skipped["no_edge"] += 1
            continue
        sb, ss = (A, B) if best["buy_on"] == "a" else (B, A)
        buy = walk_book(best["bbuy"]["asks"], qty=best["qty"])                                         # leg 1 at the decision-time book
        leg2 = ss.after(t + int(leg_delay_s * 1000), 5.0 * max(leg_delay_s, 1.0))
        if leg2 is None:
            skipped["leg2_unavailable"] += 1
            trades.append({"time_ms": t, "status": "open_unhedged", "buy_on": best["buy_on"], "sell_on": best["sell_on"], "qty": best["qty"], "note": "the second leg met no book within the wait: left unhedged and EXCLUDED from P&L", "gross": 0.0, "fees": 0.0})
            continue
        sell = walk_book(leg2["bids"], qty=best["qty"])
        if sell["coins"] < best["qty"] * 0.999:
            qty = sell["coins"]
        else:
            qty = best["qty"]
        if qty <= 0:
            skipped["leg2_unavailable"] += 1
            continue
        buy, sell = walk_book(best["bbuy"]["asks"], qty=qty), walk_book(leg2["bids"], qty=qty)
        fees = buy["dollars"] * best["fb"] / 1e4 + sell["dollars"] * best["fs"] / 1e4
        mid_a, mid_b = _mid(best["bbuy"]), _mid(best["bsell"])
        slip = (buy["dollars"] - qty * mid_a) + (qty * mid_b - sell["dollars"])
        cash[best["buy_on"]] -= buy["dollars"] * (1 + best["fb"] / 1e4)
        coin[best["buy_on"]] += qty
        cash[best["sell_on"]] += sell["dollars"] * (1 - best["fs"] / 1e4)
        coin[best["sell_on"]] -= qty
        gross = sell["dollars"] - buy["dollars"]
        trades.append({"time_ms": t, "status": "filled", "buy_on": best["buy_on"], "sell_on": best["sell_on"], "qty": float(qty), "buy_vwap": buy["vwap"], "sell_vwap": sell["vwap"], "decision_edge_bps": float(best["edge"]), "notional": float(buy["dollars"]),
                       "gross": float(gross), "fees": float(fees), "slippage": float(slip), "decision_books": {"a_recv": ba["recv_ts_ms"], "b_recv": bb["recv_ts_ms"], "a_seq": ba["sequence"], "b_seq": bb["sequence"]},
                       "leg2_book_recv_ms": leg2["recv_ts_ms"], "leg2_delay_s": (leg2["recv_ts_ms"] - t) / 1000.0})
        equity.append(float(sum(x["gross"] - x["fees"] for x in trades if x["status"] == "filled")))                     # cumulative net P&L after each filled trade
        expo.append(float(buy["dollars"]))
    filled = [x for x in trades if x["status"] == "filled"]
    span = (int(A.t[-1]) - int(A.t[0])) / 1000.0
    out = _summary(filled, equity, 2 * inventory_usd, expo, span)
    out.update({"state": "ok", "mode": MODE, "strategy": "cross_exchange", "venues": {"a": list(a), "b": list(b)}, "trades_detail": filled, "unhedged": [x for x in trades if x["status"] == "open_unhedged"],
                "coverage": {"decision_times": int(len(A.t)), "skipped": skipped, "usable_share": float((len(filled) + skipped["no_edge"]) / max(len(A.t), 1)), "first_ms": int(A.t[0]), "last_ms": int(A.t[-1])},
                "assumptions": ["Takers on both legs; fees as given. The first leg fills at the decision-time book, the second at the first book received after the delay.", "Balances per venue are fixed: no transfers or withdrawals are modelled.",
                                "The books are REST snapshots at the recorded times; fills between snapshots are not observed."]})
    return out


# ------------------------------------------------------------------------------------------------------------------ funding carry
def funding_carry(store, spot: tuple, perp: tuple, funding: tuple, notional: float = 10_000.0, entry_annualised: float = 0.10, exit_annualised: float = 0.0, window_settlements: int = 9, fee_bps: tuple = (10.0, 5.0),
                  max_age_s: float = 10.0, financing_apr: float = 0.0, leverage: float = 3.0, mmr: float = 0.005, start=None, end=None) -> dict:
    """Spot-perpetual funding carry replayed on recorded books and recorded funding settlements. Entry: at a recorded perpetual book, the mean of the last ``window_settlements`` settlements STRICTLY BEFORE that
    time, annualised, exceeds ``entry_annualised`` (spot bought at the asks, perpetual sold at the bids, taker fees). Each recorded settlement while open pays ``rate x perpetual notional at the mark`` to the short
    (the notional is the last recorded mid at or before the settlement); a missing settlement is counted, never filled in. Exit when the trailing rate falls below ``exit_annualised`` or the data end. The perpetual
    margin (``notional / leverage`` plus funding and mark-to-market) is checked at every recorded book: below ``mmr`` of the notional the position is closed as a margin call."""
    S, P = BookSeries(store.books(*spot, start=start, end=end)), BookSeries(store.books(*perp, start=start, end=end))
    F = store.funding(*funding)
    if not len(S.df) or not len(P.df):
        return _unavailable("the spot or the perpetual has no valid recorded book in the period", ["run the recorder on both instruments for the period"])
    if F.empty:
        return _unavailable("no funding settlement is recorded for this instrument", ["run the recorder (it stores OKX funding settlements) or download the funding history"])
    F = F.sort_values("funding_time_ms").reset_index(drop=True)
    ft, fr, fh = F["funding_time_ms"].to_numpy(), F["rate"].to_numpy(), F["interval_hours"].to_numpy()
    skipped = {"no_spot_book": 0, "no_funding_history": 0, "below_entry": 0}
    trades, equity, expo = [], [], []
    pos = None
    last_settle = None
    for t in P.t:
        t = int(t)
        bp = P.asof(t, max_age_s)
        bs = S.asof(t, max_age_s)
        if bp is None:
            continue
        k = int(np.searchsorted(ft, t, side="left"))                                                # settlements strictly before t
        trailing = None
        if k >= window_settlements:
            trailing = float(np.mean(fr[k - window_settlements:k] * (24.0 / fh[k - window_settlements:k])) * PERIODS)
        if pos is None:
            if bs is None:
                skipped["no_spot_book"] += 1
                continue
            if trailing is None:
                skipped["no_funding_history"] += 1
                continue
            if trailing < entry_annualised:
                skipped["below_entry"] += 1
                continue
            buy = walk_book(bs["asks"], notional=notional)
            qty = buy["coins"]
            sell = walk_book(bp["bids"], qty=qty)
            if qty <= 0 or sell["coins"] < qty * 0.999:
                skipped["no_spot_book"] += 1
                continue
            fees = buy["dollars"] * fee_bps[0] / 1e4 + sell["dollars"] * fee_bps[1] / 1e4
            pos = {"open_ms": t, "qty": qty, "spot_cost": buy["dollars"], "perp_px": sell["vwap"], "perp_notional0": sell["dollars"], "fees": fees, "funding": 0.0, "settlements": 0, "missed": 0,
                   "margin0": sell["dollars"] / leverage, "slip": (buy["dollars"] - qty * _mid(bs)) + (qty * _mid(bp) - sell["dollars"]), "entry_trailing": trailing}
            last_settle = t
            continue
        mark = _mid(bp)
        due = [(int(ft[i]), float(fr[i])) for i in range(len(ft)) if last_settle < ft[i] <= t]
        for st, rate in due:
            pm = P.asof(st, 3600.0)                                                                  # the mark at the settlement: the last recorded perpetual book at or before it
            if pm is None:
                pos["missed"] += 1
                continue
            pos["funding"] += rate * pos["qty"] * _mid(pm)
            pos["settlements"] += 1
        last_settle = t
        perp_pnl = pos["qty"] * (pos["perp_px"] - mark)
        margin = pos["margin0"] + perp_pnl + pos["funding"]
        call = margin < mmr * pos["qty"] * mark
        financing = (pos["spot_cost"] + pos["margin0"]) * financing_apr * (t - pos["open_ms"]) / 1000.0 / (PERIODS * 86400.0)
        mtm = (bs and pos["qty"] * bs["bids"][0][0] - pos["spot_cost"]) or 0.0
        equity.append(float(mtm + perp_pnl + pos["funding"] - pos["fees"] - financing))
        expo.append(float(pos["spot_cost"] + pos["perp_notional0"]))
        exit_now = (trailing is not None and trailing < exit_annualised) or call or t == int(P.t[-1])
        if exit_now and bs is not None:
            sellspot = walk_book(bs["bids"], qty=pos["qty"])
            cover = walk_book(bp["asks"], qty=pos["qty"])
            if sellspot["coins"] < pos["qty"] * 0.999 or cover["coins"] < pos["qty"] * 0.999:
                continue
            fees = pos["fees"] + sellspot["dollars"] * fee_bps[0] / 1e4 + cover["dollars"] * fee_bps[1] / 1e4
            gross = (sellspot["dollars"] - pos["spot_cost"]) + (pos["perp_notional0"] - cover["dollars"])
            slip = pos["slip"] + (pos["qty"] * _mid(bs) - sellspot["dollars"]) + (cover["dollars"] - pos["qty"] * _mid(bp))
            trades.append({"open_ms": pos["open_ms"], "close_ms": t, "status": "margin_call" if call else "closed", "qty": float(pos["qty"]), "notional": float(pos["spot_cost"]), "gross": float(gross), "fees": float(fees), "funding": float(pos["funding"]),
                           "financing": float(financing), "slippage": float(slip), "settlements": pos["settlements"], "settlements_missed": pos["missed"], "entry_trailing_annualised": pos["entry_trailing"], "exit_trailing_annualised": trailing})
            pos = None
    span = (int(P.t[-1]) - int(P.t[0])) / 1000.0
    out = _summary(trades, equity, notional * (1 + 1 / leverage), expo, span)
    out.update({"state": "ok", "mode": MODE, "strategy": "funding_carry", "trades_detail": trades, "open_position": None if pos is None else {"open_ms": pos["open_ms"], "qty": pos["qty"], "note": "open at the end of the data: not closed, not in the P&L"},
                "coverage": {"decision_times": int(len(P.t)), "skipped": skipped, "funding_settlements": int(len(ft)), "first_ms": int(P.t[0]), "last_ms": int(P.t[-1])},
                "assumptions": ["Spot bought and perpetual sold with taker fees at recorded book depth; funding is the recorded settlements times the perpetual notional at the recorded mark.", "Margin is the initial margin plus mark-to-market and funding; the margin-call rate is the one given (hypothetical unless taken from a verified schedule).",
                                "Borrowing, withdrawal and custody costs are not modelled; financing is the stated annual rate on the capital."]})
    return out


# ------------------------------------------------------------------------------------------------------------------ dated-futures basis
def futures_basis(store, venue: str, coin: str, entry_annualised: float = 0.05, fee_bps: float = 3.5, notional: float = 10_000.0, tolerance_days: float = 1.5) -> dict:
    """Cash-and-carry on a recorded dated-futures curve: at each observation, for each contract whose annualised basis (mark over index, x 365 / days) is above ``entry_annualised`` and which has not been traded, buy
    the index and sell the future; hold to the contract's expiry. Closed at expiry only when an observation within ``tolerance_days`` after expiry exists (the future then equals the index, so the P&L is the entry
    basis less the move of the index against the fixed dollar notional, here measured on the index observation); otherwise the trade stays OPEN and is reported, never settled with an assumed price. Fees apply to both legs
    on entry; the inverse-contract convexity of Deribit futures is not modelled."""
    F = store.futures(venue, coin)
    if F.empty:
        return _unavailable("no dated-futures observation is recorded", ["run the recorder: it stores Deribit's dated futures at each pass"])
    F = F.sort_values("obs_ts_ms").reset_index(drop=True)
    F["expiry_ms"] = F["expiry"].map(lambda s: int(pd.Timestamp(s, tz="UTC").timestamp() * 1000) + 8 * 3600 * 1000)            # 08:00 UTC expiry
    traded, trades, openp = set(), [], []
    for t, grp in F.groupby("obs_ts_ms"):                                                                 # observations in time order: a decision sees only the curve observed at its own time
        for r in grp.itertuples():
            days = (r.expiry_ms - t) / 86_400_000.0
            if days < 1 or r.instrument in traded:
                continue
            basis = r.mark_price / r.index_price - 1.0
            ann = basis * 365.0 / days
            if ann >= entry_annualised:
                traded.add(r.instrument)
                qty = notional / r.index_price
                fees = notional * (fee_bps / 1e4) * 2
                end = F[(F["instrument"] == r.instrument) & (F["obs_ts_ms"] >= r.expiry_ms) & (F["obs_ts_ms"] <= r.expiry_ms + tolerance_days * 86_400_000)]
                rec = {"instrument": r.instrument, "open_ms": int(t), "expiry": r.expiry, "entry_basis": float(basis), "entry_annualised": float(ann), "qty": float(qty), "notional": float(notional), "fees": float(fees), "funding": 0.0, "financing": 0.0}
                if len(end):
                    e = end.iloc[0]
                    rec.update(status="settled", close_ms=int(e["obs_ts_ms"]), slippage=0.0, gross=float(qty * (r.mark_price - r.index_price)))          # a linear future sold at F0 and the index bought at I0 lock F0 - I0 per coin at convergence
                    trades.append(rec)
                else:
                    openp.append({**rec, "status": "open", "note": "no observation within the tolerance after expiry: not settled, not in the P&L"})
    if not trades and not openp:
        return {"state": "ok", "mode": MODE, "strategy": "futures_basis", "trades": 0, "trades_detail": [], "open_positions": [], "gross_pnl": 0.0, "net_pnl": 0.0, "coverage": {"observations": int(F["obs_ts_ms"].nunique()), "reason": "no contract met the entry threshold"},
                "assumptions": []}
    capital = notional
    out = _summary(trades, [], capital, [], max((F["obs_ts_ms"].max() - F["obs_ts_ms"].min()) / 1000.0, 1.0))
    out.update({"state": "ok", "mode": MODE, "strategy": "futures_basis", "trades_detail": trades, "open_positions": openp, "coverage": {"observations": int(F["obs_ts_ms"].nunique()), "first_ms": int(F["obs_ts_ms"].min()), "last_ms": int(F["obs_ts_ms"].max()), "contracts": int(F["instrument"].nunique())},
                "assumptions": ["Entry at the recorded mark and index of the observation; settlement at the index of the first observation within the tolerance after expiry, never an assumed price.", "Fees on both legs at entry; settlement at expiry has no fee here.", "Inverse-contract convexity and margin are not modelled."]})
    return out


# ------------------------------------------------------------------------------------------------------------------ liquidity
def execution_history(store, venue: str, instrument: str, side: str, notional: float, start=None, end=None, max_age_s: float = 10.0) -> dict:
    """What a market order of ``notional`` dollars WOULD have cost against every recorded book of an instrument: the distribution over time of slippage against the mid, the unfilled share and the depth. These are real books at
    the recorded times; the order is not placed in them one after another, so the result is a distribution, not a path."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    S = BookSeries(store.books(venue, instrument, start=start, end=end))
    if not len(S.df):
        return _unavailable("no valid recorded book for this instrument in the period", ["run the recorder"])
    rows = []
    for i in range(len(S.df)):
        b = S.asof(int(S.t[i]), max_age_s)
        fill = walk_book(b["asks"] if side == "buy" else b["bids"], notional=notional)
        if fill["coins"] <= 0:
            continue
        m = _mid(b)
        slip = ((fill["vwap"] / m - 1.0) if side == "buy" else (1.0 - fill["vwap"] / m)) * 1e4
        rows.append({"time_ms": int(S.t[i]), "slippage_bps": float(slip), "unfilled_share": float(fill["unfilled_dollars"] / notional), "spread_bps": float((b["asks"][0][0] / b["bids"][0][0] - 1) * 1e4)})
    s = pd.DataFrame(rows)
    q = lambda c, p: float(np.nanpercentile(s[c], p))                                                  # noqa: E731
    return {"state": "ok", "mode": MODE, "observations": int(len(s)), "slippage_bps": {"median": q("slippage_bps", 50), "p90": q("slippage_bps", 90), "max": float(s["slippage_bps"].max())}, "unfilled_share_mean": float(s["unfilled_share"].mean()),
            "spread_bps_median": q("spread_bps", 50), "fully_filled_share": float((s["unfilled_share"] < 1e-9).mean()), "series": rows[-300:], "first_ms": int(s["time_ms"].min()), "last_ms": int(s["time_ms"].max()),
            "note": "each recorded book is a separate experiment against a real book; the replenishment between orders is not part of it"}
