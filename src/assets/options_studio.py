"""Options Strategy Studio: opportunity scanner, real-contract builder, scenario laboratory, trade validation and a hypothesis-based structure ranking.

Nothing here predicts. The scanner lists where a chain looks unusual against its own history and its own neighbouring expiries, with the quote freshness and liquidity that decide whether the
oddity is tradable; the builder prices a position from the bid and ask of real listed contracts and the limit prices you give; the scenario laboratory revalues it under stated shocks and splits the
P&L into the underlying, volatility, skew, time, costs and an explicit residual; the ranking orders the feasible structures by what they would earn if YOUR stated view were right, and what they lose
if it were wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import options_analytics as oa

MULT = oa.MULTIPLIER
NEAR = 0.05                                                            # |ln K/F| within which a quote counts as near the money


def _expiry_dte(g: pd.DataFrame) -> int:
    return int(round(float(g["T"].iloc[0]) * 365))


def quality(g: pd.DataFrame, stale: pd.Series | None = None) -> dict:
    """Freshness and liquidity of the quotes near the money of one expiry: median spread as a share of the mid, median volume and open interest, the share of stale quotes and the oldest quote age."""
    near = g[(g["k"].abs() <= NEAR) & (g["bid"] > 0)]
    if near.empty:
        return {"quotes": 0, "tradable": False, "reason": "no two-sided quotes near the money"}
    spread = ((near["ask"] - near["bid"]) / near["mid"].replace(0, np.nan)).median()
    stale_share = float(stale.loc[near.index].mean()) if stale is not None else 0.0
    age = float(g.loc[near.index, "quote_age_min"].max()) if "quote_age_min" in g else None
    oi, vol = float(near["open_interest"].median()), float(near["volume"].median())
    reasons = []
    if not np.isfinite(spread) or spread > 0.10:
        reasons.append(f"median spread {spread:.0%} of the mid" if np.isfinite(spread) else "no usable spread")
    if stale_share > 0.2:
        reasons.append(f"{stale_share:.0%} of the near-the-money quotes are stale")
    if oi <= 0 and vol <= 0:
        reasons.append("no volume or open interest")
    return {"quotes": int(len(near)), "median_spread_pct": float(spread) if np.isfinite(spread) else None, "median_open_interest": oi, "median_volume": vol, "stale_share": stale_share,
            "oldest_quote_min": age, "tradable": not reasons, "reason": "; ".join(reasons)}


def _robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med)) * 1.4826
    return (values - med) / mad if mad > 1e-12 else np.zeros_like(values)


def scan(chain: pd.DataFrame, spot: float, prices: pd.Series | None = None, iv_history: pd.Series | None = None, min_dte: int = 3, min_history: int = 20) -> dict:
    """Where this chain looks unusual: implied volatility against its own history and against realised volatility, skew dislocations across expiries, term-structure kinks, inversions and possible event
    volatility, and calendar-spread candidates priced on real quotes. Every item carries the quote freshness and liquidity of the expiries it rests on, and ``tradable`` says whether those quotes
    could be traded at all. A scanner lists leads; a lead is not a recommendation."""
    stale = oa.quote_staleness(chain)
    term = [t for t in oa.term_structure(chain) if t["dte"] >= min_dte]
    if len(term) < 2:
        return {"state": "unavailable", "reason": "fewer than two expiries with a usable at-the-money volatility", "needs": ["a chain with at least two expiries"]}
    by_expiry = {pd.Timestamp(t["expiry"]): chain[chain["expiry"] == pd.Timestamp(t["expiry"])] for t in term}
    q = {t["expiry"]: quality(by_expiry[pd.Timestamp(t["expiry"])], stale) for t in term}
    items: list[dict] = []
    iv30 = oa.interpolate_iv(term, 30.0)

    # implied against its own history, and against what the underlying realised
    hist = None
    if iv_history is not None and iv_history.dropna().size >= min_history:
        h = iv_history.dropna()
        pct = float((h < iv30).mean())
        z = float((iv30 - h.mean()) / h.std(ddof=1)) if h.std(ddof=1) > 0 else 0.0
        hist = {"state": "ok", "iv_30d": iv30, "percentile": pct, "zscore": z, "n": int(h.size), "source": "dated chains of this underlying kept by the platform"}
        if pct >= 0.9 or pct <= 0.1:
            items.append({"kind": "iv_history", "title": f"30-day implied volatility is at the {pct:.0%} percentile of its own history", "metric": iv30, "z": z, "expiries": [], "quality": None,
                          "detail": f"{iv30:.1%} against a mean of {h.mean():.1%} over {h.size} dated chains", "tradable": None})
    else:
        n = 0 if iv_history is None else int(iv_history.dropna().size)
        hist = {"state": "unavailable", "reason": f"{n} dated chains are kept; comparing implied volatility with its own history needs at least {min_history}",
                "needs": ["take a snapshot of the chain on each trading day (Snapshot button)"]}
    realised = None
    if prices is not None and len(prices.dropna()) > 80:
        logret = np.log(prices.dropna().astype(float)).diff().dropna()
        rv = logret.rolling(21).std(ddof=1).dropna() * math.sqrt(252.0)
        cur = float(rv.iloc[-1])
        rv_pct = float((rv.tail(504) < cur).mean())
        realised = {"state": "ok", "rv_21d": cur, "rv_percentile": rv_pct, "iv_30d": iv30, "ratio": float(iv30 / cur) if cur > 0 else None, "variance_premium": float(iv30 ** 2 - cur ** 2),
                    "reading": "implied variance above realised is the premium sellers collect on average; it is not a forecast of the next month"}
        if cur > 0 and (iv30 / cur > 1.5 or iv30 / cur < 0.8):
            items.append({"kind": "iv_vs_realised", "title": f"30-day implied is {iv30 / cur:.2f}x 21-day realised volatility", "metric": float(iv30 / cur), "z": None, "expiries": [], "quality": None,
                          "detail": f"implied {iv30:.1%}, realised {cur:.1%} (its own {rv_pct:.0%} percentile of two years)", "tradable": None})
    else:
        realised = {"state": "unavailable", "reason": "no price history for the underlying", "needs": ["a price history"]}

    # skew dislocations across expiries
    skews = []
    for t in term:
        try:
            m = oa.skew_metrics(chain, pd.Timestamp(t["expiry"]))
        except ValueError:
            continue
        if np.isfinite(m.get("rr_25d", np.nan)):
            skews.append(m)
    if len(skews) >= 4:
        rr = np.array([m["rr_25d"] for m in skews])
        z = _robust_z(rr)
        med = float(np.median(rr))
        for m, zi in zip(skews, z):
            if abs(zi) > 2.5 and abs(m["rr_25d"] - med) > 0.01:
                items.append({"kind": "skew_dislocation", "title": f"{m['expiry']}: 25-delta risk reversal {m['rr_25d']:+.1%} against a cross-expiry median of {med:+.1%}", "metric": float(m["rr_25d"]), "z": float(zi),
                              "expiries": [m["expiry"]], "quality": q[m["expiry"]], "detail": "the skew of this expiry is far from its neighbours' (robust z-score of the 25-delta risk reversal)", "tradable": q[m["expiry"]]["tradable"]})
    skew_rows = [{"expiry": m["expiry"], "dte": m["dte"], "rr_25d": m["rr_25d"], "bf_25d": m["bf_25d"], "put_25d_iv": m["put_25d_iv"], "call_25d_iv": m["call_25d_iv"]} for m in skews]

    # term structure: kinks, negative forward variance, event bumps
    d = np.array([t["dte"] for t in term], float)
    T = d / 365.0
    iv = np.array([t["atm_iv"] for t in term])
    w = iv ** 2 * T
    fwd = []
    for i in range(len(term) - 1):
        dv = (w[i + 1] - w[i]) / (T[i + 1] - T[i])
        fwd.append({"from": term[i]["expiry"], "to": term[i + 1]["expiry"], "forward_vol": float(math.sqrt(dv)) if dv > 0 else None, "forward_variance": float(dv)})
        if dv <= 0:
            items.append({"kind": "negative_forward_variance", "title": f"{term[i]['expiry']} to {term[i + 1]['expiry']}: total variance falls with maturity", "metric": float(dv), "z": None,
                          "expiries": [term[i]["expiry"], term[i + 1]["expiry"]], "quality": {"front": q[term[i]["expiry"]], "back": q[term[i + 1]["expiry"]]},
                          "detail": "a calendar arbitrage in the quotes, or stale or one-sided quotes in one of the two expiries", "tradable": q[term[i]["expiry"]]["tradable"] and q[term[i + 1]["expiry"]]["tradable"]})
    fv = np.array([f["forward_vol"] if f["forward_vol"] else np.nan for f in fwd])
    for i in range(len(fwd)):
        local = [fv[j] for j in range(max(0, i - 2), min(len(fv), i + 3)) if j != i and np.isfinite(fv[j])]
        if np.isfinite(fv[i]) and len(local) >= 2 and fv[i] > 1.2 * float(np.median(local)) and fv[i] - float(np.median(local)) > 0.02:
            items.append({"kind": "event_vol", "title": f"forward volatility {fv[i]:.1%} between {fwd[i]['from']} and {fwd[i]['to']}, above its neighbours' {float(np.median(local)):.1%}", "metric": float(fv[i]), "z": None,
                          "expiries": [fwd[i]["from"], fwd[i]["to"]], "quality": {"front": q[fwd[i]["from"]], "back": q[fwd[i]["to"]]},
                          "detail": "possible event volatility (earnings, a central-bank date, a data release) inside the window; the platform holds no event calendar, so the cause is yours to check",
                          "tradable": q[fwd[i]["from"]]["tradable"] and q[fwd[i]["to"]]["tradable"]})
    if len(term) >= 4:
        dev = []
        for i in range(1, len(term) - 1):
            w_interp = np.interp(T[i], [T[i - 1], T[i + 1]], [w[i - 1], w[i + 1]])
            dev.append(iv[i] - math.sqrt(max(w_interp, 1e-12) / T[i]))
        zs = _robust_z(np.array(dev))
        for i, (dv_, zi) in enumerate(zip(dev, zs), start=1):
            if abs(zi) > 2.5 and abs(dv_) > 0.01:
                e = term[i]["expiry"]
                items.append({"kind": "term_kink", "title": f"{e}: at-the-money volatility {dv_:+.1%} off the line between its neighbours", "metric": float(dv_), "z": float(zi), "expiries": [e],
                              "quality": q[e], "detail": "an expiry that sits off the interpolated term structure (robust z-score of its deviation in total variance)", "tradable": q[e]["tradable"]})
    inverted = bool(np.isfinite(iv[0]) and len(iv) > 2 and iv[0] > iv[-1] + 0.01)

    # calendar candidates priced on real quotes
    cal = []
    for i, a in enumerate(term):
        for b in term[i + 1:]:
            if a["dte"] < 7 or b["dte"] - a["dte"] < 14 or b["dte"] > 150:
                continue
            if a["atm_iv"] - b["atm_iv"] < 0.01:
                continue
            ga = by_expiry[pd.Timestamp(a["expiry"])]
            k = float(ga.iloc[(ga["strike"] - a["forward"]).abs().argsort()].iloc[0]["strike"])
            try:
                built = build_real(chain, [{"type": "option", "expiry": a["expiry"], "strike": k, "right": "C", "quantity": -1}, {"type": "option", "expiry": b["expiry"], "strike": k, "right": "C", "quantity": 1}],
                                   spot, analysis=False)
            except ValueError:
                continue
            ok = q[a["expiry"]]["tradable"] and q[b["expiry"]]["tradable"]
            cal.append({"kind": "calendar", "title": f"sell {a['expiry']} / buy {b['expiry']} call calendar at {k:g}", "metric": float(a["atm_iv"] - b["atm_iv"]), "z": None, "expiries": [a["expiry"], b["expiry"]],
                        "quality": {"front": q[a["expiry"]], "back": q[b["expiry"]]}, "tradable": ok,
                        "detail": f"front volatility {a['atm_iv']:.1%} over back {b['atm_iv']:.1%}; debit {built['totals']['at_mid']:,.0f} at the mid, {built['totals']['at_touch']:,.0f} crossing both spreads",
                        "legs": built["legs_in"], "debit_mid": built["totals"]["at_mid"], "debit_touch": built["totals"]["at_touch"]})
    cal.sort(key=lambda r: (-int(bool(r["tradable"])), -r["metric"]))
    items.extend(cal[:5])
    items.sort(key=lambda r: (-int(bool(r.get("tradable"))), -(abs(r["z"]) if r.get("z") is not None else 0.0)))
    return {"state": "ok", "spot": float(spot), "chain_date": str(pd.Timestamp(chain["date"].iloc[0]).date()) if "date" in chain else None, "items": items, "iv_history": hist, "realised": realised,
            "term": [{**t, "forward_vol_to_next": None} for t in term], "forward_vols": fwd, "skew": skew_rows, "term_inverted": inverted, "quality": q,
            "reading": "Each item is a lead from the quotes as they are. A flagged expiry with stale or wide quotes ('tradable': false) is more likely a data artefact than an opportunity."}


# ------------------------------------------------------------------------------------------------------------------ real contracts
def _resolve(chain: pd.DataFrame, leg: dict) -> pd.Series:
    exp = pd.Timestamp(leg["expiry"])
    sel = chain[(chain["expiry"] == exp) & (chain["right"] == str(leg["right"]).upper()[:1]) & ((chain["strike"] - float(leg["strike"])).abs() < 1e-9)]
    if sel.empty:
        listed = chain[(chain["expiry"] == exp) & (chain["right"] == str(leg["right"]).upper()[:1])]["strike"].sort_values()
        if listed.empty:
            raise ValueError(f"no {leg['right']} contracts are listed for {exp.date()}; listed expiries are {sorted({str(pd.Timestamp(e).date()) for e in chain['expiry'].unique()})[:12]}")
        near = listed.iloc[(listed - float(leg["strike"])).abs().argsort()].iloc[:4].tolist()
        raise ValueError(f"strike {leg['strike']:g} is not listed for {exp.date()}; the nearest listed strikes are {sorted(near)}")
    return sel.iloc[0]


def _limit_status(qty: float, limit: float | None, bid: float, ask: float) -> str:
    if limit is None:
        return "no limit: priced at the mid"
    if qty > 0:
        return "marketable: pays up to the ask" if limit >= ask else "inside the spread: may not fill" if limit > bid else "at or below the bid: unlikely to fill"
    return "marketable: receives down to the bid" if limit <= bid else "inside the spread: may not fill" if limit < ask else "at or above the ask: unlikely to fill"


def build_real(chain: pd.DataFrame, legs_in: list[dict], spot: float, rate: float | None = None, dividend: float | None = None, commission: float = 0.65, horizon_days: int | None = None,
               analysis: bool = True) -> dict:
    """A position of real listed contracts priced from the chain: every leg's bid, ask, mid, spread, volume, open interest, the limit price you gave and what it implies (marketable or resting), and
    the cost of the package at the touch (buying at the ask, selling at the bid), at the mid and at your limits. With ``analysis`` the payoff, break-evens, Greeks and odds of ``strategy_analysis`` are
    returned for the actual entry price. Quantities are contracts (shares for ``type: stock``), positive to buy."""
    if not legs_in:
        raise ValueError("add at least one leg")
    if len(legs_in) > 12:
        raise ValueError("at most 12 legs")
    stale = oa.quote_staleness(chain)
    rows, opt_legs, flags = [], [], []
    touch = mid = assumed = 0.0
    commissions = 0.0
    as_of = pd.Timestamp(chain["date"].iloc[0]) if "date" in chain else pd.Timestamp.today().normalize()
    for leg in legs_in:
        qty = float(leg.get("quantity", 0))
        if qty == 0:
            raise ValueError("a leg's quantity cannot be zero")
        limit = None if leg.get("limit") in (None, "") else float(leg["limit"])
        if str(leg.get("type", "option")) == "stock":
            px = float(leg.get("price") or spot)
            use = limit if limit is not None else px
            rows.append({"type": "stock", "label": "shares", "quantity": qty, "bid": None, "ask": None, "mid": px, "limit": limit, "assumed_price": use, "cash": qty * use, "status": "shares at the last price"})
            touch += qty * px
            mid += qty * px
            assumed += qty * use
            opt_legs.append({"type": "stock", "quantity": qty})
            continue
        r = _resolve(chain, leg)
        bid, ask, m = float(r["bid"]), float(r["ask"]), float(r["mid"])
        if bid <= 0 or ask <= 0:
            flags.append(f"{leg['right']} {float(leg['strike']):g} {pd.Timestamp(leg['expiry']).date()} has no two-sided market (bid {bid:g}, ask {ask:g})")
        use = limit if limit is not None else m
        px_touch = ask if qty > 0 else bid
        idx = r.name
        stale_leg = bool(stale.loc[idx]) if idx in stale.index else False
        spread_pct = (ask - bid) / m if m > 0 else None
        if spread_pct is not None and spread_pct > 0.10:
            flags.append(f"{leg['right']} {float(leg['strike']):g} {pd.Timestamp(leg['expiry']).date()}: the spread is {spread_pct:.0%} of the mid")
        if stale_leg:
            flags.append(f"{leg['right']} {float(leg['strike']):g} {pd.Timestamp(leg['expiry']).date()}: the quote is stale")
        status = _limit_status(qty, limit, bid, ask)
        rows.append({"type": "option", "label": f"{'Call' if str(leg['right']).upper().startswith('C') else 'Put'} {float(leg['strike']):g} {pd.Timestamp(leg['expiry']).date()}", "expiry": str(pd.Timestamp(leg["expiry"]).date()),
                     "strike": float(leg["strike"]), "right": str(leg["right"]).upper()[:1], "quantity": qty, "bid": bid, "ask": ask, "mid": m, "spread_pct": spread_pct, "volume": float(r.get("volume", 0.0)),
                     "open_interest": float(r.get("open_interest", 0.0)), "iv": float(r["iv"]), "stale": stale_leg, "limit": limit, "assumed_price": use, "touch_price": px_touch, "cash": qty * MULT * use, "status": status})
        touch += qty * MULT * px_touch
        mid += qty * MULT * m
        assumed += qty * MULT * use
        commissions += abs(qty) * commission
        opt_legs.append({"type": "option", "right": str(leg["right"]).upper()[:1], "strike": float(leg["strike"]), "quantity": qty, "dte": max(int((pd.Timestamp(leg["expiry"]) - as_of).days), 1), "iv": float(r["iv"])})
    unfilled = [r for r in rows if r["type"] == "option" and ("unlikely" in r["status"] or "may not fill" in r["status"])]
    totals = {"at_touch": float(touch + commissions), "at_mid": float(mid + commissions), "at_limits": float(assumed + commissions), "spread_cost": float(touch - mid), "commission": float(commissions),
              "convention": "positive is a debit paid, negative a credit received; dollars including the contract multiplier"}
    out = {"state": "ok", "legs": rows, "legs_in": [{k: v for k, v in l.items() if k in ("type", "expiry", "strike", "right", "quantity", "limit")} for l in legs_in], "totals": totals, "risk_flags": flags,
           "fill": {"legs_resting": len(unfilled), "all_marketable": not unfilled and all(l.get("limit") is not None or True for l in rows),
                    "reading": "every leg is marketable at the stated limits" if not unfilled else f"{len(unfilled)} leg(s) rest inside or outside the spread and may not fill; the package may be left partly filled"}}
    if not analysis:
        return out
    opts = [l for l in opt_legs if l["type"] == "option"]
    if not opts:
        raise ValueError("a strategy needs at least one option leg")
    first = chain[chain["expiry"] == pd.Timestamp(legs_in[[i for i, l in enumerate(legs_in) if str(l.get("type", "option")) == "option"][0]]["expiry"])]
    forward, atm = float(first["F"].iloc[0]), oa.atm_iv(first)
    rt = float(rate if rate is not None else chain["rate"].median())
    dv = float(dividend if dividend is not None else chain["dividend"].median()) if "dividend" in chain else 0.0
    a = oa.strategy_analysis(opt_legs, spot, forward, atm if np.isfinite(atm) else float(np.nanmedian([l["iv"] for l in opts])), rt, dv, oa.chain_skew_slope(chain), horizon_days=horizon_days, fill="mid", commission=commission)
    out["analysis"] = reprice(a, float(assumed + commissions))
    out["analysis"]["entry"] = {"price_basis": "your limits where given, the mid otherwise", "debit": float(assumed + commissions)}
    return out


def reprice(a: dict, actual_cost: float) -> dict:
    """``strategy_analysis`` at its model entry cost, moved to the actual entry cost: every P&L curve shifts by the difference, and the maximum profit and loss, the break-evens, the odds and the margin are
    recomputed from the shifted expiry curve on the same lognormal view."""
    delta = float(a["cost"] - actual_cost)
    out = dict(a)
    out["curve"] = {k: ([round(float(x) + delta, 4) for x in v] if k != "spot" else v) for k, v in a["curve"].items()}
    out["grid"] = {**a["grid"], "pnl": [[float(x) + delta for x in row] for row in a["grid"]["pnl"]]}
    s, e = np.asarray(a["curve"]["spot"], float), np.asarray(out["curve"]["expiry"], float)
    out["cost"], out["net_debit"], out["net_credit"] = float(actual_cost), float(max(actual_cost, 0.0)), float(max(-actual_cost, 0.0))
    if not a["unlimited_profit"]:
        out["max_profit"] = float(e.max())
    if not a["unlimited_loss"]:
        out["max_loss"] = float(e.min())
    out["breakevens"] = oa._breakevens(s, e)
    edges, mass = np.asarray(a["density"]["edges"], float), np.asarray(a["density"]["mass"], float)
    centres = 0.5 * (edges[1:] + edges[:-1])
    out["prob_profit"] = float(mass[np.interp(centres, s, e) > 0].sum())
    out["expected_value"] = float(a["expected_value"] + delta)
    mp, ml = out.get("max_profit"), out.get("max_loss")
    out["reward_risk"] = float(mp / -ml) if mp is not None and ml is not None and ml < 0 and not a["unlimited_profit"] and not a["unlimited_loss"] else None
    if a["margin"].get("defined_risk"):
        out["margin"] = {**a["margin"], "estimate": float(max(-out["max_loss"], 0.0))}
    out["reprice_shift"] = delta
    return out


# ------------------------------------------------------------------------------------------------------------------ scenarios
def _positions(legs: list[dict]) -> list[dict]:
    out = []
    for l in legs:
        if l.get("type") == "stock":
            out.append({"type": "stock", "quantity": float(l["quantity"])})
        else:
            out.append({"type": "option", "right": l["right"], "strike": float(l["strike"]), "quantity": float(l["quantity"]), "dte": float(l["dte"]), "iv": float(l["iv"])})
    return out


def _value(pos: list[dict], spot: float, rate: float, dividend: float, as_of, ds=0.0, dv=0.0, days=0.0, skew=0.0, forward: float | None = None) -> float:
    moved = []
    for p in pos:
        if p["type"] == "option" and skew:
            ref = forward if forward else spot
            moved.append({**p, "iv": max(p["iv"] + skew * math.log(p["strike"] / ref), 0.02)})
        else:
            moved.append(p)
    return float(oa.position_values(moved, spot, rate, dividend, as_of, ds, dv, days).get("value", pd.Series([0.0])).sum())


def scenario_lab(legs: list[dict], spot: float, rate: float, dividend: float, as_of=None, entry_cost: float = 0.0, costs: float = 0.0, spot_moves=(-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
                 vol_moves=(-0.10, -0.05, 0.0, 0.05, 0.10), skew_shifts=(-0.15, -0.05, 0.0, 0.05, 0.15), day_steps=(0, 7, 14, 21, 30), scenarios: list[dict] | None = None, forward: float | None = None) -> dict:
    """Revalue a position (legs as ``strategy_analysis`` takes them: option legs with ``dte`` and ``iv``) under stated shocks. Four heatmaps (spot x parallel volatility, spot x skew shift, spot x days
    passed, and the combined scenarios you list) give the P&L against today's value; ``skew shift`` adds ``shift x ln(K/F)`` to each strike's volatility, so a negative shift steepens the equity skew. Each
    scenario's P&L is attributed to the underlying (delta and gamma), volatility (vega, vanna, volga), skew, time (theta) and the entry costs, and what the Taylor terms miss is the explicit residual."""
    pos = _positions(legs)
    if not any(p["type"] == "option" for p in pos):
        raise ValueError("a scenario needs at least one option leg")
    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today().normalize())
    base_frame = oa.position_values(pos, spot, rate, dividend, as_of)
    v0 = float(base_frame["value"].sum())
    ref = forward or spot

    def pnl(ds=0.0, dv=0.0, days=0.0, skew=0.0) -> float:
        return _value(pos, spot, rate, dividend, as_of, ds, dv, days, skew, ref) - v0

    heat_vol = {"spot": [float(x) for x in spot_moves], "vol": [float(x) for x in vol_moves], "pnl": [[pnl(ds, dv) for ds in spot_moves] for dv in vol_moves]}
    heat_skew = {"spot": [float(x) for x in spot_moves], "skew": [float(x) for x in skew_shifts], "pnl": [[pnl(ds, 0.0, 0.0, sk) for ds in spot_moves] for sk in skew_shifts]}
    heat_time = {"spot": [float(x) for x in spot_moves], "days": [float(x) for x in day_steps], "pnl": [[pnl(ds, 0.0, d) for ds in spot_moves] for d in day_steps]}
    named = scenarios or [{"name": "spot -5%, vol +5 pts", "spot": -0.05, "vol": 0.05, "skew": 0.0, "days": 7}, {"name": "spot +5%, vol -5 pts", "spot": 0.05, "vol": -0.05, "skew": 0.0, "days": 7},
                          {"name": "quiet, 21 days pass", "spot": 0.0, "vol": 0.0, "skew": 0.0, "days": 21}, {"name": "skew steepens, spot flat", "spot": 0.0, "vol": 0.0, "skew": -0.10, "days": 0},
                          {"name": "crash: spot -15%, vol +15 pts, skew steepens", "spot": -0.15, "vol": 0.15, "skew": -0.10, "days": 3}]
    rows = []
    for sc in named[:12]:
        ds, dv, sk, days = float(sc.get("spot", 0)), float(sc.get("vol", 0)), float(sc.get("skew", 0)), float(sc.get("days", 0))
        total = pnl(ds, dv, days, sk)
        no_skew = pnl(ds, dv, days, 0.0)
        skew_effect = total - no_skew
        tay = oa._attribution(base_frame, ds, dv, days, no_skew, "sticky_strike", 0.0)
        parts = {"delta": tay["delta"], "gamma": tay["gamma"], "vega": tay["vega"], "cross_greeks": tay["vanna"] + tay["volga"], "theta": tay["theta"], "skew": skew_effect, "costs": -float(costs)}
        explained = parts["delta"] + parts["gamma"] + parts["vega"] + parts["cross_greeks"] + parts["theta"] + skew_effect
        parts["residual"] = float(total - explained)                                    # what the Taylor terms miss (third order and beyond, expiry kinks): reported, never forced to zero
        rows.append({"name": str(sc.get("name", "scenario")), "spot": ds, "vol": dv, "skew": sk, "days": days, "pnl_before_costs": float(total), "pnl": float(total - costs),
                     "return_on_entry": float((total - costs) / abs(entry_cost)) if abs(entry_cost) > 1e-9 else None, "attribution": {k: float(v) for k, v in parts.items()}})
    return {"state": "ok", "value_today": v0, "attribution_method": "The scenario P&L is an exact full revaluation. Delta, gamma, vega, theta and the cross-Greeks (vanna, volga) are second-order Taylor terms from today's Greeks; skew is an exact revaluation difference; costs are the entry spread and commissions; the residual is total minus these and is not forced to zero.", "heatmaps": {"spot_vol": heat_vol, "spot_skew": heat_skew, "spot_time": heat_time}, "scenarios": rows,
            "greeks_today": {"delta_dollars": float((base_frame["delta"] * base_frame["spot0"]).sum()), "gamma_pnl_1pct": float((0.5 * base_frame["gamma"] * (0.01 * base_frame["spot0"]) ** 2).sum()),
                             "vega_per_point": float(base_frame["vega"].sum()), "theta_per_day": float(base_frame["theta"].sum()), "vanna": float(base_frame["vanna"].sum()), "volga": float(base_frame["volga"].sum())},
            "assumptions": ["Black-Scholes revaluation with each leg's own implied volatility; strikes keep their volatility when the spot moves (sticky strike) unless a skew shift is stated.",
                            "Volatility shocks are parallel, in volatility points; the skew shift adds shift x ln(K / F) per strike.", "Time passes with the spot and volatility held as stated; expired legs settle at intrinsic value.",
                            "The residual is what the Greek Taylor terms miss (higher orders, expiry kinks); costs are the entry spread and commissions, shown apart from the market moves."]}


# ------------------------------------------------------------------------------------------------------------------ hypothesis ranking
VIEWS = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


def recommend(chain: pd.DataFrame, spot: float, direction: str = "neutral", move: float = 0.05, vol_view: str = "neutral", vol_move: float = 0.03, horizon_days: int = 30,
            max_loss: float | None = None, margin_budget: float | None = None, quantity: int = 1, top: int = 8, rate: float | None = None, dividend: float | None = None, stability: bool = True) -> dict:
    """Rank the platform's structures by what they would earn IF the stated hypothesis came true by the horizon: the underlying ends ``move`` up (bullish), down (bearish) or within ``move`` of where it is
    (neutral), and implied volatility changes by ``vol_move`` points (long volatility up, short volatility down, neutral none). Each structure is priced on this chain's own quotes and entered at the touch.
    The table also shows what it loses if the opposite happened, so a high hypothesis P&L bought with a large wrong-way loss is visible. Structures that break ``max_loss`` (dollars) or ``margin_budget``
    are listed as excluded with the reason. This ranks payoffs under an assumption; it does not predict which assumption is right."""
    if direction not in VIEWS:
        raise ValueError(f"direction must be one of {list(VIEWS)}")
    if vol_view not in ("long", "short", "neutral"):
        raise ValueError("vol_view must be long, short or neutral")
    if not 0.005 <= move <= 0.5 or not 0.0 <= vol_move <= 0.5 or horizon_days < 3:
        raise ValueError("move must be 0.5% to 50%, vol_move 0 to 50 points and the horizon at least 3 days")
    sign = VIEWS[direction]
    vsign = {"long": 1.0, "short": -1.0, "neutral": 0.0}[vol_view]
    ctx = oa.structure_context(chain, horizon_days)
    as_of = pd.Timestamp(chain["date"].iloc[0]) if "date" in chain else pd.Timestamp.today().normalize()
    rt = float(rate if rate is not None else chain["rate"].median())
    dv = float(dividend if dividend is not None else chain["dividend"].median()) if "dividend" in chain else 0.0
    hyp = {"spot": sign * move, "vol": vsign * vol_move}
    wrong = {"spot": -sign * move, "vol": -vsign * vol_move}
    rows, excluded = [], []
    for name in oa.STRATEGY_INFO:
        try:
            legs, meta = oa.named_structure(chain, spot, name, dte=horizon_days, quantity=quantity)
            if not any(l["type"] == "option" for l in legs):
                continue
            a = oa.strategy_analysis(legs, spot, meta["forward"], meta["atm_iv"], rt, dv, oa.chain_skew_slope(chain), horizon_days=min(horizon_days, min(l["dte"] for l in legs if l["type"] == "option")), cash_secured=(name == "cash_secured_put"))
            friction = _friction(chain, legs, as_of)
        except (ValueError, KeyError, IndexError) as exc:
            excluded.append({"strategy": name, "reason": f"cannot be built on this chain: {exc}"})
            continue
        horizon = float(min(horizon_days, min(l["dte"] for l in legs if l["type"] == "option")))
        pos = _positions(legs)
        v0 = float(oa.position_values(pos, spot, rt, dv, as_of)["value"].sum())

        def at(ds, dvol):
            return float(oa.position_values(pos, spot, rt, dv, as_of, ds, dvol, horizon)["value"].sum()) - v0

        if direction == "neutral":
            hyp_pnl = float(np.mean([at(0.0, hyp["vol"]), at(0.5 * move, hyp["vol"]), at(-0.5 * move, hyp["vol"])]))
            wrong_pnl = float(min(at(move, wrong["vol"]), at(-move, wrong["vol"])))
        else:
            hyp_pnl, wrong_pnl = at(hyp["spot"], hyp["vol"]), at(wrong["spot"], wrong["vol"])
        hyp_pnl -= friction
        wrong_pnl -= friction
        delta = a["greeks"]["delta_dollars"]
        vega = a["greeks"]["vega_per_point"]
        loss = -a["max_loss"] if not a["unlimited_loss"] else None
        margin = a["margin"]["estimate"] if not a["unlimited_loss"] else None
        reasons = []
        if max_loss is not None and (loss is None or loss > max_loss):
            reasons.append("loss is unlimited" if loss is None else f"maximum loss {loss:,.0f} exceeds {max_loss:,.0f}")
        if margin_budget is not None and (margin is None or margin > margin_budget):
            reasons.append("margin is not bounded" if margin is None else f"margin {margin:,.0f} exceeds {margin_budget:,.0f}")
        if sign != 0 and delta * sign < 0:
            reasons.append("its net delta points against the stated direction")
        if vsign != 0 and vega * vsign < 0:
            reasons.append("its net vega points against the stated volatility view")
        row = {"strategy": name, "group": oa.group_of(name), "net_debit": a["net_debit"], "net_credit": a["net_credit"], "max_loss": a["max_loss"], "max_profit": a["max_profit"], "unlimited_loss": a["unlimited_loss"],
               "unlimited_profit": a["unlimited_profit"], "margin": margin, "delta_dollars": float(delta), "vega_per_point": float(vega), "theta_per_day": a["greeks"]["theta_per_day"], "friction": float(friction),
               "pnl_if_right": hyp_pnl, "pnl_if_wrong": wrong_pnl, "right_per_wrong": float(hyp_pnl / -wrong_pnl) if wrong_pnl < 0 and hyp_pnl > 0 else None,
               "prob_profit_market_view": a["prob_profit"], "legs": [{**{k: l[k] for k in ("type", "right", "strike", "quantity", "dte") if k in l}, **({"expiry": str(pd.Timestamp(oa.nearest_expiry_to(chain, l["dte"])).date())} if l["type"] == "option" else {})} for l in legs]}
        (excluded.append({"strategy": name, "reason": "; ".join(reasons)}) if reasons else rows.append(row))
    rows.sort(key=lambda r: -(r["pnl_if_right"] + 0.0))
    stable = None
    if stability and rows:
        order = {r["strategy"]: i for i, r in enumerate(rows)}
        runs = []
        for km, kv in ((0.5, 1.0), (1.5, 1.0), (1.0, 0.5), (1.0, 1.5)):
            alt = recommend(chain, spot, direction, move * km, vol_view, vol_move * kv, horizon_days, max_loss, margin_budget, quantity, len(rows), rate, dividend, stability=False)
            pos = {r["strategy"]: i for i, r in enumerate(alt["ranked"])}
            common = [s for s in order if s in pos]
            rho = float(pd.Series([order[s] for s in common]).corr(pd.Series([pos[s] for s in common]), method="spearman")) if len(common) > 2 else None
            runs.append({"move_x": km, "vol_move_x": kv, "top": alt["ranked"][0]["strategy"] if alt["ranked"] else None, "rank_correlation": rho})
        same = sum(1 for r in runs if r["top"] == rows[0]["strategy"])
        stable = {"runs": runs, "top_unchanged_in": same, "of": len(runs), "stable": bool(same >= 3), "reading": ("the top structure stays first under +/-50% changes of the stated move and volatility change" if same >= 3
                  else "the top structure changes when the stated move or volatility change is perturbed by 50%: the ranking is fragile, treat it as one of several comparable candidates")}
    hold = {"hold_100_shares": float(100 * spot * sign * move), "cash": 0.0, "note": "the same move on 100 shares of the underlying, and cash earning nothing, for comparison with the structures' payoff if right"}
    return {"state": "ok", "stability": stable, "baselines": hold, "hypothesis": {"direction": direction, "move": move, "vol_view": vol_view, "vol_move": vol_move, "horizon_days": horizon_days, "max_loss": max_loss, "margin_budget": margin_budget},
            "ranked": rows[:top], "n_feasible": len(rows), "excluded": excluded, "context": {"forward": ctx["forward"], "atm_iv": ctx["atm_iv"], "expiry": str(pd.Timestamp(ctx["expiry"]).date())},
            "reading": "Ordered by profit if the stated hypothesis is right, after crossing the spread; the 'if wrong' column is the same structure under the opposite move. It is a ranking of payoffs under an assumption, not a forecast.",
            "assumptions": ["each structure is entered at the touch of this chain (buys at the ask, sells at the bid) plus commissions", "Black-Scholes revaluation at the horizon with each leg's own implied volatility",
                            "volatility moves in parallel by the stated points; the horizon is the stated days or the first expiry, whichever is earlier"]}


def _friction(chain: pd.DataFrame, legs: list[dict], as_of, commission: float = 0.65) -> float:
    """The dollar cost of crossing the bid-ask spread on every option leg of a structure (half the spread per contract) plus commissions, from the chain's own quotes."""
    cost = 0.0
    for l in legs:
        if l["type"] != "option":
            continue
        e = oa.nearest_expiry_to(chain, l["dte"])
        g = chain[(chain["expiry"] == e) & (chain["right"] == l["right"])]
        r = g.iloc[(g["strike"] - l["strike"]).abs().argsort()].iloc[0]
        cost += abs(l["quantity"]) * (MULT * (float(r["ask"]) - float(r["bid"])) / 2.0 + commission)
    return float(cost)


def portfolio_impact(book: list[dict], proposed: list[dict], spot: float, rate: float, dividend: float, as_of=None) -> dict:
    """The change a proposed position makes to a book (positions in ``position_values`` form: option ``right``, ``strike``, ``dte``, ``iv``, ``quantity``, shares as ``type: stock``): dollar delta, gamma P&L of
    a 1% move, vega per point, theta per day, vanna and volga, and the worst loss of the stress grid, before and after, with the contribution of each expiry. A hypothetical book is the caller's to label."""
    if not proposed:
        raise ValueError("a proposed position is needed")
    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today().normalize())
    before = oa.portfolio_risk(book, spot, rate, dividend, as_of) if book else None
    after = oa.portfolio_risk(list(book) + list(proposed), spot, rate, dividend, as_of)
    keys = ("delta_dollars", "gamma_pnl_1pct", "vega_per_point", "theta_per_day", "vanna_delta_per_vol_point", "volga_vega_per_vol_point", "rho_per_1pct")
    b = {k: (before["totals"][k] if before else 0.0) for k in keys}
    a = {k: after["totals"][k] for k in keys}
    worst_b = before["worst"]["pnl"] if before else 0.0
    return {"state": "ok", "hypothetical_book": not bool(book), "before": b, "after": a, "change": {k: a[k] - b[k] for k in keys}, "worst_stress": {"before": float(worst_b), "after": float(after["worst"]["pnl"]), "change": float(after["worst"]["pnl"] - worst_b),
            "scenario": {"spot": after["worst"]["spot_shock"], "vol": after["worst"]["vol_shock"]}}, "by_expiry_after": after["by_expiry"], "risk_flags": after["risk_flags"]}
