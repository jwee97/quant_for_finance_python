"""Crypto research that refuses to invent data: calendar audit, observed-only gating, cash-and-carry economics, depth-aware cross-venue execution, funding intelligence, an execution simulator and
portfolio liquidation risk.

Three rules run through the module.

* **Observed or unavailable.** A number is either computed from data the platform holds (``platform``, ``downloaded``, ``user_file``) or the result is ``{"state": "unavailable", ...}`` with what is
  needed to make it available. Nothing here substitutes a simulated series for a missing one; ``crypto_analytics`` keeps its simulations for the illustrative views, and ``observed_only`` is how a
  caller switches them off.
* **A quoted spread is not an arbitrage.** ``executable_cross`` walks the real order-book levels of the two venues, charges fees, and then lists what is still not modelled (transfers, inventory, funding
  while the legs are open, latency). Its verdict says what the numbers show at the moment of the snapshot, never that a profit is confirmed.
* **Residuals are shown.** The cash-and-carry decomposition reports funding, basis change, the delta left by unequal leg sizes, costs and financing, and the residual against an independent total.

Calendar: crypto trades every day, so returns here are daily on every calendar day and annualised by 365 (``PERIODS``).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..instruments.crypto import VENUES
from .common import last_zscore, percentile_rank

PERIODS = 365.0
OBSERVED = ("platform", "downloaded", "user_file")
FUNDING_UTC_HOURS = {8: (0, 8, 16), 4: (0, 4, 8, 12, 16, 20), 1: tuple(range(24))}


def is_observed(source: str | None) -> bool:
    return str(source) in OBSERVED


def unavailable(reason: str, needs: list[str] | None = None) -> dict:
    """The shape of a result the platform cannot produce from observed data."""
    return {"state": "unavailable", "reason": reason, "needs": list(needs or [])}


# ------------------------------------------------------------------------------------------------------------------ calendar
def calendar_audit(index) -> dict:
    """Is a daily index a 24/7 calendar? Counts missing calendar days and weekend marks, states the annualisation that fits and what the legacy weekday pipeline (252) would drop."""
    idx = pd.DatetimeIndex(index)
    idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize().unique().sort_values()
    if len(idx) < 2:
        return unavailable("fewer than two daily marks")
    full = pd.date_range(idx[0], idx[-1], freq="D")
    missing = full.difference(idx)
    weekend = int((idx.dayofweek >= 5).sum())
    coverage = len(idx) / len(full)
    return {"state": "ok", "first": str(idx[0].date()), "last": str(idx[-1].date()), "marks": int(len(idx)), "calendar_days": int(len(full)), "missing_days": int(len(missing)), "coverage": float(coverage),
            "weekend_marks": weekend, "is_24_7": bool(coverage >= 0.98 and weekend > 0), "periods_per_year": int(PERIODS),
            "weekday_pipeline": {"periods_per_year": 252, "marks_it_drops": weekend,
                                 "note": "the platform's equity-style pipeline keeps Monday to Friday marks and annualises by 252; a Friday-to-Monday return then spans three days. Use the 24/7 engine for crypto numbers."}}


def annualise(daily: pd.Series) -> dict:
    """Mean, volatility and Sharpe of a daily 24/7 return series, annualised by 365 (never 252)."""
    r = pd.Series(daily).dropna().astype(float)
    if len(r) < 10:
        return {"n": int(len(r)), "ann_return": None, "ann_vol": None, "sharpe": None}
    sd = float(r.std(ddof=1))
    return {"n": int(len(r)), "ann_return": float(r.mean() * PERIODS), "ann_vol": float(sd * math.sqrt(PERIODS)), "sharpe": float(r.mean() / sd * math.sqrt(PERIODS)) if sd > 0 else None}


# ------------------------------------------------------------------------------------------------------------------ cash and carry
def carry_pnl(s, notional: float = 1e6, leverage: float = 3.0, fee_bps: float = 5.0, slippage_bps: float = 2.0, financing_apr: float = 0.0, maintenance: float | None = None,
              days: int = 365) -> dict:
    """What a cash-and-carry (long spot, short perpetual, equal USD notional at entry) earned, split into funding received, basis change, the delta left by unequal leg sizes, trading costs and financing.

    ``leverage`` is the perpetual leg's leverage: its initial margin is ``notional / leverage`` held on the exchange, and the spot leg is paid in full, so the return is on ``notional x (1 + 1/leverage)``.
    ``financing_apr`` charges that capital's opportunity cost. Margin is tracked daily on the perpetual account with NO top-ups, so a day where the balance falls under the maintenance requirement is a
    margin call and the first one is where the position would be liquidated. Funding reversals (days the short pays) are counted. For an inverse perpetual (Deribit) the total is recomputed in coin and the
    gap to the linear decomposition is the explicit residual."""
    if s.perp is None or s.funding is None:
        return unavailable(f"{s.symbol} needs a perpetual price and a funding history", ["perp", "funding"])
    if not (is_observed(s.source)):
        return unavailable(f"{s.symbol} is {s.source}: the carry economics are computed on observed data only", ["observed spot, perpetual and funding"])
    frame = pd.DataFrame({"spot": s.spot, "perp": s.perp, "funding": s.funding}).dropna().tail(int(days) + 1)
    if len(frame) < 10:
        return unavailable("fewer than ten days with spot, perpetual and funding together")
    venue = VENUES.get(s.funding_venue, VENUES["DERIBIT"])
    perp_type = venue["perp_type"]
    mm = float(maintenance if maintenance is not None else venue["mmr"])
    S, P, F = frame["spot"].to_numpy(float), frame["perp"].to_numpy(float), frame["funding"].to_numpy(float)
    N, L = float(notional), max(float(leverage), 1.0)
    qs, qp, margin0 = N / S[0], N / P[0], N / L
    capital = N + margin0
    dS, dP = np.diff(S), np.diff(P)
    funding = qp * P[:-1] * F[1:]                                                                         # earned by the short on the perpetual's notional
    basis = qp * (dS - dP)                                                                                # the legs' price moves net of each other
    delta = (qs - qp) * dS                                                                                # what unequal coin sizes leave unhedged
    perp_leg = -qp * dP
    margin = margin0 + np.cumsum(perp_leg + funding)
    required = mm * qp * P[1:]
    ratio = margin / required
    breach = np.flatnonzero(margin < required)
    cost_in = (fee_bps + slippage_bps) / 1e4 * (qs * S[0] + qp * P[0])
    cost_out = (fee_bps + slippage_bps) / 1e4 * (qs * S[-1] + qp * P[-1])
    n_days = len(frame) - 1
    financing = capital * float(financing_apr) * n_days / PERIODS
    comp = {"funding_received": float(funding.sum()), "basis_change": float(basis.sum()), "unhedged_delta": float(delta.sum()), "trading_costs": float(-(cost_in + cost_out)), "financing": float(-financing)}
    comp["collateral_revaluation"] = 0.0
    if perp_type == "inverse":                                                                             # coin-margined: pnl and margin are in coin, valued at the spot of each day
        coin = N * (1.0 / P[1:] - 1.0 / P[:-1])                                                              # a short of N dollars, in coin
        fund_coin = F[1:] * N / P[:-1]
        margin_coin = margin0 / S[0] + np.cumsum(coin + fund_coin)
        total = qs * S[-1] - N + margin_coin[-1] * S[-1] - margin0 + comp["trading_costs"] + comp["financing"]
        comp["collateral_revaluation"] = float(margin0 / S[0] * (S[-1] - S[0]))                              # the margin is held in coin, so it moves with the spot
        note_res = "inverse perpetual: the margin is held in coin (shown as collateral revaluation); the residual is the convexity of coin-denominated pnl the linear split cannot express"
    else:
        total = qs * S[-1] - N + (margin[-1] - margin0) + comp["trading_costs"] + comp["financing"]
        note_res = "linear perpetual: the components add to the total up to rounding"
    parts_sum = sum(comp.values())
    comp["residual"] = float(total - parts_sum)
    neg = F[1:] < 0
    run = best = 0
    for v in neg:
        run = run + 1 if v else 0
        best = max(best, run)
    cumf = np.cumsum(funding)
    worst_funding_dd = float((cumf - np.maximum.accumulate(np.maximum(cumf, 0.0))).min())
    p_liq = (margin0 / qp + P[0]) / (1.0 + mm)
    shocks = {}
    for up in (0.1, 0.2, 0.3, 0.5, 1.0):
        bal = margin0 - qp * P[-1] * up + (margin[-1] - margin0)
        req = mm * qp * P[-1] * (1.0 + up)
        shocks[f"+{int(up * 100)}%"] = {"margin_ratio": float(bal / req), "topup_needed": float(max(req - bal, 0.0))}
    avg_f = float(funding.mean())
    out = {"state": "ok", "symbol": s.symbol, "source": s.source, "perp_venue": s.funding_venue, "perp_type": perp_type, "days": int(n_days), "notional": N, "leverage": L, "capital": float(capital),
           "pnl": {**{k: float(v) for k, v in comp.items()}, "total": float(total), "return_on_capital": float(total / capital), "annualised_on_capital": float(total / capital * PERIODS / n_days)},
           "margin": {"initial": float(margin0), "maintenance_rate": mm, "min_ratio": float(ratio.min()), "margin_calls": int(len(breach)), "first_breach_day": None if not len(breach) else str(frame.index[1 + breach[0]].date()),
                      "liquidation_price_at_entry": float(p_liq), "liquidation_move_from_entry": float(p_liq / P[0] - 1.0), "shock_table": shocks},
           "funding": {"mean_per_day_usd": avg_f, "negative_days": int(neg.sum()), "longest_negative_run": int(best), "worst_cumulative_drawdown_usd": worst_funding_dd,
                       "breakeven_days": float((cost_in + cost_out) / avg_f) if avg_f > 0 else None},
           "residual_note": note_res,
           "series": {"dates": [str(d.date()) for d in frame.index[1:]], "funding": [round(float(x), 2) for x in np.cumsum(funding)], "basis": [round(float(x), 2) for x in np.cumsum(basis)],
                      "margin_ratio": [round(float(x), 4) for x in ratio]},
           "assumptions": ["Equal USD notional on both legs at entry, held to the last day shown, no rebalancing and no margin top-ups.",
                           f"{fee_bps:g} bps fee and {slippage_bps:g} bps slippage per leg on entry and exit (four fills).",
                           f"Funding as stored for {s.funding_venue} (daily, earned by the short); it is paid on the perpetual's notional, which drifts with its price.",
                           f"Capital is the spot leg plus the perpetual's initial margin; financing at {financing_apr:.1%} a year is its opportunity cost.",
                           "Borrowing costs, withdrawal fees, tax and exchange outages are not modelled."]}
    return out


# ------------------------------------------------------------------------------------------------------------------ order books
def walk_book(levels: list, notional: float | None = None, qty: float | None = None) -> dict:
    """Consume ``levels`` (``[price, size_in_coin]`` best first) for ``notional`` dollars or ``qty`` coins. Returns the coins and dollars filled, the volume-weighted price and what stayed unfilled."""
    if (notional is None) == (qty is None):
        raise ValueError("give either a notional or a quantity")
    coins = dollars = 0.0
    used = 0
    want_d, want_q = (float(notional), None) if notional is not None else (None, float(qty))
    for p, size in levels:
        p, size = float(p), float(size)
        if want_d is not None:
            take = min(want_d - dollars, p * size) / p
        else:
            take = min(want_q - coins, size)
        if take <= 0:
            break
        coins += take
        dollars += take * p
        used += 1
        if (want_d is not None and dollars >= want_d - 1e-9) or (want_q is not None and coins >= want_q - 1e-12):
            break
    vwap = dollars / coins if coins > 0 else None
    return {"coins": float(coins), "dollars": float(dollars), "vwap": None if vwap is None else float(vwap), "levels_used": used,
            "unfilled_dollars": float(max((want_d or 0.0) - dollars, 0.0)) if want_d is not None else None, "unfilled_coins": float(max((want_q or 0.0) - coins, 0.0)) if want_q is not None else None}


def _age_seconds(stamp: str | None, now: datetime | None = None) -> float | None:
    if not stamp:
        return None
    try:
        t = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    now = now or datetime.now(timezone.utc)
    return float((now - (t if t.tzinfo else t.replace(tzinfo=timezone.utc))).total_seconds())


def executable_cross(snapshot: dict | None, notional: float = 1e6, fee_mode: str = "taker_taker", fees: dict | None = None, max_age_s: float = 5.0, now: datetime | None = None,
                     kinds: tuple = ("spot", "perp")) -> dict:
    """Cross-venue pairs priced on the real order-book levels of a snapshot: buy ``notional`` dollars on one venue by walking its asks, sell the same coins into the other's bids. Reports the quoted (top of
    book) edge beside the executable one, the price impact, the fill fraction (the smaller of what both books can absorb), fees, and per pair the inventory or collateral it would need.

    ``fee_mode`` ``taker_taker`` (both legs cross) or ``maker_taker`` (the buy leg rests at the best bid at the maker fee: its fill is NOT guaranteed and is flagged). ``fees`` overrides
    ``{venue: {"taker": bps, "maker": bps}}``; without it the snapshot's entry-level taker fees apply and the maker fee is taken as 0 (flagged as an assumption). ``max_age_s`` is the age after which a
    quote is stale. The verdict is never "arbitrage confirmed": the best it says is that the edge survives costs at the moment of the snapshot."""
    if not snapshot or not snapshot.get("venues"):
        return unavailable("no order-book snapshot: quotes and books are not stored by the platform", ["download the venue snapshot (Free data, crypto venues) just before looking"])
    if fee_mode not in ("taker_taker", "maker_taker"):
        raise ValueError("fee_mode must be taker_taker or maker_taker")
    age = _age_seconds(snapshot.get("fetched_at"), now)
    stale = age is None or age > max_age_s
    venues = [v for v in snapshot["venues"] if v.get("kind") in kinds and v.get("bids") and v.get("asks")]
    rows = []
    for a in venues:
        for b in venues:
            if a["venue"] == b["venue"] or a["kind"] != b["kind"]:
                continue
            fa = (fees or {}).get(a["venue"], {})
            fb = (fees or {}).get(b["venue"], {})
            buy_fee = float(fa.get("maker", 0.0)) if fee_mode == "maker_taker" else float(fa.get("taker", a.get("taker_bps", 0.0)))
            sell_fee = float(fb.get("taker", b.get("taker_bps", 0.0)))
            top_ask, top_bid = float(a["asks"][0][0]), float(b["bids"][0][0])
            if fee_mode == "maker_taker":
                top_buy = float(a["bids"][0][0])
                buy = walk_book([[p, sz] for p, sz in a["bids"]], notional=notional)
            else:
                top_buy = top_ask
                buy = walk_book(a["asks"], notional=notional)
            if buy["coins"] <= 0:
                continue
            sell = walk_book(b["bids"], qty=buy["coins"])
            filled = min(buy["coins"], sell["coins"])
            if filled <= 0:
                continue
            buy_part = walk_book(a["bids"] if fee_mode == "maker_taker" else a["asks"], qty=filled)
            sell_part = walk_book(b["bids"], qty=filled)
            cost, proceeds = buy_part["dollars"], sell_part["dollars"]
            fees_usd = cost * buy_fee / 1e4 + proceeds * sell_fee / 1e4
            net = proceeds - cost - fees_usd
            quoted = (top_bid / top_buy - 1.0) * 1e4
            executable = (proceeds / cost - 1.0) * 1e4
            reqs = ([f"{filled:,.4f} {snapshot.get('coin', '')} already held at {b['label']} to sell without waiting for a transfer", f"${cost:,.0f} cash at {a['label']}"] if a["kind"] == "spot"
                    else [f"collateral on both venues (about ${notional / 10:,.0f} each at 10x)", "margin and funding exposure until both legs are closed"])
            notes = []
            if stale:
                notes.append("the quotes are older than the stale limit: the edge is probably gone" if age is not None else "the snapshot has no timestamp, so its age is unknown")
            if filled < buy["coins"] * 0.999 or buy["unfilled_dollars"]:
                notes.append("partial fill: the books cannot absorb the whole size")
            if fee_mode == "maker_taker":
                notes.append("the resting buy order may never fill; the result assumes it does")
            if {a.get("quote_asset"), b.get("quote_asset")} == {"USDT", "USD"}:
                notes.append("one venue quotes in USDT: part of the gap is the USDT/USD rate")
            verdict = ("stale" if stale else "no_edge_after_costs" if net <= 0 else "edge_at_snapshot_not_confirmed")
            rows.append({"buy": a["venue"], "sell": b["venue"], "kind": a["kind"], "quoted_bps": float(quoted), "executable_bps": float(executable), "price_impact_bps": float(quoted - executable),
                         "fees_bps": float(buy_fee + sell_fee), "net_usd": float(net), "net_bps": float(net / cost * 1e4), "filled_usd": float(cost), "fill_fraction": float(cost / notional),
                         "requirements": reqs, "notes": notes, "verdict": verdict})
    rows.sort(key=lambda r: r["net_usd"], reverse=True)
    return {"state": "ok", "coin": snapshot.get("coin"), "fetched_at": snapshot.get("fetched_at"), "age_seconds": age, "stale": bool(stale), "max_age_s": max_age_s, "notional": float(notional),
            "fee_mode": fee_mode, "pairs": rows,
            "not_modelled": ["transfer time and fees between venues", "inventory and collateral limits at each venue", "funding paid or received while perpetual legs are open", "latency between the two venues' books (the snapshot is read venue by venue, not simultaneously)", "price movement after the snapshot"],
            "reading": "A quoted spread that survives these costs is a lead to check against live books, not a confirmed arbitrage."}


# ------------------------------------------------------------------------------------------------------------------ execution simulator
def volume_profile(n: int, shape: str = "flat") -> np.ndarray:
    """Slice weights of a VWAP schedule. ``flat`` is TWAP; ``u_shape`` is an ASSUMED intraday profile with more volume at the ends of the UTC day (not measured)."""
    if shape == "flat":
        w = np.ones(n)
    elif shape == "u_shape":
        x = np.linspace(-1.0, 1.0, n)
        w = 1.0 + 0.8 * x ** 2
    else:
        raise ValueError("shape must be flat or u_shape")
    return w / w.sum()


def simulate_execution(venue: dict, side: str, notional: float, order_type: str = "market", limit_bps: float = 0.0, slices: int = 1, replenish: float = 1.0, profile: str = "flat",
                       fee_bps: float | None = None, vol_annual: float | None = None, horizon_hours: float = 0.0) -> dict:
    """Execute ``notional`` dollars on one venue's saved book. ``market`` walks the book at once; ``limit`` takes only what is at or inside ``limit_bps`` of the mid (the rest is left unfilled);
    ``twap``/``vwap`` split the order into ``slices`` child market orders. HYPOTHETICAL: the book is a snapshot, so the child orders see it with ``replenish`` (0 to 1) of the depth the previous child
    consumed back in place, an assumption and not a measurement. ``vol_annual`` and ``horizon_hours`` add the one-standard-deviation timing risk of waiting. Compared with the immediate market order."""
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    if order_type not in ("market", "limit", "twap", "vwap"):
        raise ValueError("order_type must be market, limit, twap or vwap")
    if not (notional > 0):
        raise ValueError("notional must be positive")
    if not 0.0 <= replenish <= 1.0:
        raise ValueError("replenish must be between 0 and 1")
    bids, asks = [list(map(float, x)) for x in venue["bids"]], [list(map(float, x)) for x in venue["asks"]]
    levels = asks if side == "buy" else bids
    mid = (bids[0][0] + asks[0][0]) / 2.0
    fee = float(venue.get("taker_bps", 0.0) if fee_bps is None else fee_bps)
    sign = 1.0 if side == "buy" else -1.0

    def slip(vwap):
        return None if vwap is None else float(sign * (vwap / mid - 1.0) * 1e4)

    market = walk_book(levels, notional=notional)
    out = {"state": "ok", "venue": venue.get("venue"), "side": side, "notional": float(notional), "order_type": order_type, "mid": float(mid), "fee_bps": fee, "market_reference": {"vwap": market["vwap"], "slippage_bps": slip(market["vwap"]), "unfilled_usd": market["unfilled_dollars"]}}
    if order_type == "market":
        res = market
    elif order_type == "limit":
        edge = mid * (1 + sign * float(limit_bps) / 1e4)
        inside = [[p, s] for p, s in levels if (p <= edge if side == "buy" else p >= edge)]
        res = walk_book(inside, notional=notional) if inside else {"coins": 0.0, "dollars": 0.0, "vwap": None, "unfilled_dollars": float(notional), "levels_used": 0}
        out["limit_price"] = float(edge)
    else:
        n = max(int(slices), 1)
        w = volume_profile(n, "flat" if order_type == "twap" else profile)
        book = [list(x) for x in levels]
        coins = dollars = 0.0
        children = []
        for i in range(n):
            child = walk_book(book, notional=notional * w[i])
            coins += child["coins"]
            dollars += child["dollars"]
            children.append({"slice": i + 1, "notional": float(notional * w[i]), "vwap": child["vwap"], "slippage_bps": slip(child["vwap"]), "unfilled_usd": child["unfilled_dollars"]})
            taken = child["coins"]                                                                          # remove what the child consumed, then let the book recover a share of it
            gone = taken
            for lv in book:
                use = min(lv[1], gone)
                lv[1] -= use
                gone -= use
                if gone <= 1e-12:
                    break
            consumed = [(l0[1] - l1[1]) for l0, l1 in zip(levels, book)]
            for lv, orig, c in zip(book, levels, consumed):
                lv[1] = min(orig[1], lv[1] + replenish * c)
        res = {"coins": coins, "dollars": dollars, "vwap": dollars / coins if coins > 0 else None, "unfilled_dollars": float(max(notional - dollars, 0.0)), "levels_used": None}
        out["children"] = children
        out["assumptions"] = [f"each child order sees the saved book with {replenish:.0%} of the depth the earlier ones consumed restored (an assumption, not a measurement)",
                              "the mid is held at its snapshot value: no drift between slices" + ("" if not vol_annual else ", priced as timing risk below"),
                              "the volume profile is " + ("flat" if order_type == "twap" or profile == "flat" else "an assumed U shape, not measured volume")]
    filled = res["dollars"]
    out["result"] = {"filled_usd": float(filled), "fill_fraction": float(filled / notional), "unfilled_usd": float(max(notional - filled, 0.0)), "vwap": res["vwap"], "slippage_bps": slip(res["vwap"]),
                     "fee_usd": float(filled * fee / 1e4), "all_in_bps": None if res["vwap"] is None else float(slip(res["vwap"]) + fee),
                     "saving_vs_market_bps": None if res["vwap"] is None or market["vwap"] is None else float(slip(market["vwap"]) - slip(res["vwap"]))}
    if vol_annual and horizon_hours > 0:
        out["result"]["timing_risk_bps_1sd"] = float(vol_annual * math.sqrt(horizon_hours / (24 * PERIODS)) * 1e4)
    out["reading"] = ("An immediate market order pays the book's depth; slicing lowers that only if the book recovers between slices and the price does not move, neither of which is observed here."
                      if order_type in ("twap", "vwap") else "Walks the saved levels of one moment; a live book changes within seconds.")
    return out


# ------------------------------------------------------------------------------------------------------------------ funding intelligence
def funding_calendar(venue: str, now: datetime | None = None, count: int = 6) -> dict:
    hours = VENUES[venue]["funding_hours"]
    slots = FUNDING_UTC_HOURS.get(hours, (0, 8, 16))
    now = now or datetime.now(timezone.utc)
    upcoming = []
    day = now.replace(minute=0, second=0, microsecond=0)
    for d in range(3):
        for h in slots:
            t = (day + pd.Timedelta(days=d)).replace(hour=h)
            if t > now:
                upcoming.append(t.strftime("%Y-%m-%dT%H:%MZ"))
    return {"venue": venue, "interval_hours": hours, "settlement_hours_utc": list(slots), "next": upcoming[:count],
            "note": "Deribit accrues funding continuously; the platform stores its daily sum, so only the daily total is observed." if venue == "DERIBIT" else "Rate stored daily; the payment is made at each settlement hour."}


def funding_regimes(f: pd.Series, window: int = 30, high: float = 0.10, low: float = 0.0) -> dict:
    """Regime of the trailing ``window``-day annualised funding: ``negative`` (below ``low``), ``normal``, ``crowded`` (above ``high``) and the dates it changed."""
    ann = (f.rolling(window, min_periods=window).mean() * PERIODS).dropna()
    if ann.empty:
        return {"current": None, "changes": []}
    label = pd.Series(np.where(ann < low, "negative", np.where(ann > high, "crowded", "normal")), index=ann.index)
    chg = label[label != label.shift()]
    changes = [{"date": str(d.date()), "to": str(v), "annualised": float(ann.loc[d])} for d, v in chg.items()][1:]
    return {"current": str(label.iloc[-1]), "since": str(chg.index[-1].date()), "thresholds": {"negative_below": low, "crowded_above": high}, "changes": changes[-12:], "n_changes": len(changes)}


def funding_intelligence(series: dict, venues: tuple = ("DERIBIT", "BINANCE", "BYBIT", "OKX"), observed_only: bool = True, days: int = 365, now: datetime | None = None) -> dict:
    """Funding by coin and venue, normalised to a per-interval rate and an annualised rate (365), with its percentile and z-score, regime changes, the cross-venue dispersion, the spot-perpetual basis
    percentile and the payment calendar. With ``observed_only`` a venue whose funding is simulated is listed as unavailable instead of shown. Open interest, liquidations and dated futures are not
    stored by the platform: they are reported as unavailable with what would be needed."""
    from . import crypto_analytics as ca

    rows, skipped, by_coin = [], [], {}
    for sym, s in series.items():
        if s.funding is None:
            skipped.append({"symbol": sym, "venue": None, "reason": "no funding history"})
            continue
        per = {}
        for v in venues:
            f, src = ca.venue_funding(s, v)
            if observed_only and not is_observed(src):
                skipped.append({"symbol": sym, "venue": v, "reason": f"{src} data: not shown in observed-only mode"})
                continue
            f = f.dropna()
            hours = VENUES[v]["funding_hours"]
            tail = f.tail(days)
            last = float(f.iloc[-1])
            row = {"symbol": sym, "venue": v, "source": src, "interval_hours": hours, "rate_per_interval": float(last * hours / 24.0), "annualised": float(last * PERIODS),
                   "avg_7d": float(f.tail(7).mean() * PERIODS), "avg_30d": float(f.tail(30).mean() * PERIODS), "zscore_90d": last_zscore(f, 90), "percentile_1y": percentile_rank(tail, last),
                   "percentile_full": percentile_rank(f, last), "history_days": int(len(f)), "positive_share_1y": float((tail > 0).mean()), "regime": funding_regimes(f), "calendar": funding_calendar(v, now)}
            rows.append(row)
            per[v] = f
        if len(per) >= 2:
            latest = {v: float(f.tail(7).mean() * PERIODS) for v, f in per.items()}
            hi, lo = max(latest, key=latest.get), min(latest, key=latest.get)
            by_coin[sym] = {"venues": sorted(latest), "dispersion_annualised": float(latest[hi] - latest[lo]), "highest": hi, "lowest": lo, "latest_7d": latest}
        elif sym in series:
            by_coin[sym] = {"venues": sorted(per), "dispersion_annualised": None, "reason": "fewer than two observed venues: no dispersion"}
    basis = {}
    for sym, s in series.items():
        if s.perp is not None and is_observed(s.source):
            b = (s.perp / s.spot - 1.0).dropna()
            if len(b) > 30:
                basis[sym] = {"basis": float(b.iloc[-1]), "annualised_if_held_8h": float(b.iloc[-1]), "percentile_1y": percentile_rank(b.tail(365), float(b.iloc[-1])), "zscore_90d": last_zscore(b, 90)}
    return {"state": "ok", "observed_only": bool(observed_only), "rows": rows, "unavailable_rows": skipped, "dispersion": by_coin, "basis": basis,
            "open_interest": unavailable("open interest by exchange is not stored by the platform", ["an exchange integration (OKX, Binance, Bybit public open-interest endpoints)", "a stored daily history of it"]),
            "liquidations": unavailable("liquidation observations are not stored by the platform", ["an exchange liquidation feed or a vendor history"]),
            "dated_futures": unavailable("the dated-futures term structure is not stored for crypto", ["Deribit or CME dated futures prices by expiry"]),
            "normalisation": "rate per interval = daily rate x interval hours / 24; annualised = daily rate x 365; funding is the rate EARNED BY THE SHORT (positive when longs pay)."}


# ------------------------------------------------------------------------------------------------------------------ portfolio leverage and liquidation
def portfolio_liquidation(positions: list[dict], collateral: float, betas: dict | None = None, liquidation_fee_bps: float = 50.0, shocks: tuple = (-0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5)) -> dict:
    """Cross-margined liquidation risk of perpetual positions. ``positions`` are ``{symbol, venue, side (long/short), notional}``; ``collateral`` is the account's equity in dollars. A market shock ``x``
    moves each coin by ``beta x`` (``betas`` per symbol against the shock factor, 1 when missing; pass observed betas). Equity after the shock is collateral plus the positions' pnl; the requirement
    is each venue's maintenance rate on the shocked notional. Reports the leverage, a liquidation price per position as if isolated with its share of collateral, the portfolio shock at which equity
    meets the requirement, and the loss of the liquidation fee."""
    if not positions:
        return unavailable("no positions")
    if collateral <= 0:
        raise ValueError("collateral must be positive")
    pos = []
    for p in positions:
        venue = str(p.get("venue", "BINANCE")).upper()
        if venue not in VENUES:
            raise ValueError(f"unknown venue '{venue}'")
        side = str(p.get("side", "long")).lower()
        if side not in ("long", "short"):
            raise ValueError("side must be long or short")
        n = float(p["notional"])
        if not n > 0:
            raise ValueError("every notional must be positive")
        pos.append({"symbol": str(p["symbol"]).upper(), "venue": venue, "sign": 1.0 if side == "long" else -1.0, "notional": n, "mmr": VENUES[venue]["mmr"], "max_leverage": VENUES[venue]["max_leverage"]})
    beta = {k.upper(): float(v) for k, v in (betas or {}).items()}

    def state(x: float) -> dict:
        pnl = sum(q["sign"] * q["notional"] * beta.get(q["symbol"], 1.0) * x for q in pos)
        equity = collateral + pnl
        req = sum(q["mmr"] * q["notional"] * max(1.0 + beta.get(q["symbol"], 1.0) * x, 0.0) for q in pos)
        return {"equity": float(equity), "required": float(req), "margin_ratio": float(equity / req) if req > 0 else None, "liquidated": bool(equity < req)}

    gross = sum(q["notional"] for q in pos)
    net = sum(q["sign"] * q["notional"] for q in pos)
    table = {f"{x:+.0%}": state(x) for x in shocks}
    grid = np.linspace(-0.95, 0.95, 1901)
    down = next((float(x) for x in grid[grid <= 0][::-1] if state(float(x))["liquidated"]), None)
    up = next((float(x) for x in grid[grid >= 0] if state(float(x))["liquidated"]), None)
    for q in pos:
        share = collateral * q["notional"] / gross
        lev = q["notional"] / share
        move = (q["mmr"] - 1.0 / lev) / (1.0 - q["mmr"]) if q["sign"] > 0 else (1.0 / lev - q["mmr"]) / (1.0 + q["mmr"])
        q["isolated"] = {"leverage": float(lev), "liquidation_move": float(move), "over_venue_limit": bool(lev > q["max_leverage"])}
        q.pop("sign"), q.pop("mmr")
    first = down if down is not None and (up is None or abs(down) <= abs(up)) else up
    return {"state": "ok", "collateral": float(collateral), "gross_notional": float(gross), "net_notional": float(net), "gross_leverage": float(gross / collateral), "net_leverage": float(net / collateral),
            "positions": pos, "shock_table": table, "liquidation_shock_down": down, "liquidation_shock_up": up, "nearest_liquidation_shock": first,
            "liquidation_fee_loss_at_nearest": None if first is None else float(liquidation_fee_bps / 1e4 * gross),
            "assumptions": ["Cross margin across the listed positions; the shock moves every coin by its beta times the shock at once, with no gap, no funding and no auto-deleveraging.",
                            "Maintenance rates are the platform's per-venue defaults, not your account's tiered rates.",
                            "Isolated liquidation move: the fall (long) or rise (short) of that coin that wipes out the position's share of the collateral down to the maintenance requirement."]}
