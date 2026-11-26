"""Futures data validation: contract specifications, and what a continuous series is and is not.

* **Specifications** (``validate_specs``): a multiplier, a tick size and a currency for every contract, an expiry that follows the first trading day, expiries that increase along a chain without repeats, margins that make
  sense (maintenance not above initial), and prices that sit on the tick grid. Each finding names the contract and the rule.
* **Continuous versus executable prices** (``executability``): a continuous series is a research device. Its ratio-adjusted history is a rescaling of old prices, its difference-adjusted history a shift by points, so an
  old level on it is a price nobody could have traded and, shifted far enough, it can be zero or negative. Only the last segment equals what the exchange showed. The report says which, and how far the adjusted level
  is from the raw one at each point in time.
* **Calendar spread versus roll loss** (``roll_table``): at a roll the new contract trades above (contango) or below (backwardation) the old one. That gap is a calendar spread, a difference between two prices on the same
  day; selling one and buying the other at those prices realises no profit or loss except the cost of crossing two markets. What a holder of the front contract earns or loses over time is the convergence of its price
  to spot (the roll yield), which shows in the contract's own price changes. Treating the gap as a one-off loss, or leaving it in a spliced price series as a return, double counts it. The table gives every roll's gap in
  points and percent, the annualised roll yield it implies, and the execution cost, side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..instruments.futures import Future


def validate_specs(registry, prices: dict[str, pd.Series] | None = None) -> list[dict]:
    """Findings (``contract``, ``rule``, ``severity``, ``detail``) for every future in ``registry``; an empty list means consistent. ``prices`` (contract id -> series) are checked against the tick grid."""
    out: list[dict] = []

    def add(contract, rule, severity, detail):
        out.append({"contract": contract, "rule": rule, "severity": severity, "detail": detail})
    futures = [i for i in registry if isinstance(i, Future)]
    for f in futures:
        cid = f.instrument_id
        if not (f.contract_multiplier and f.contract_multiplier > 0):
            add(cid, "multiplier", "error", f"the multiplier is {f.contract_multiplier!r}: it must be positive")
        if not (f.tick_size and f.tick_size > 0):
            add(cid, "tick_size", "error", f"the tick size is {f.tick_size!r}: it must be positive")
        if not f.currency:
            add(cid, "currency", "error", "no currency")
        if f.expiry is None or pd.isna(f.expiry):
            add(cid, "expiry", "error", "no expiry date")
        elif f.first_trade is not None and f.first_trade > f.expiry:
            add(cid, "listing_before_expiry", "error", f"first trading day {pd.Timestamp(f.first_trade).date()} is after the expiry {pd.Timestamp(f.expiry).date()}")
        im, mm = getattr(f, "initial_margin", 0.0) or 0.0, getattr(f, "maintenance_margin", 0.0) or 0.0
        if mm > im:
            add(cid, "margin_order", "error", f"maintenance margin {mm:g} is above initial margin {im:g}")
        if im == 0.0 and mm == 0.0:
            add(cid, "margin_missing", "info", "no margin recorded: a margin or liquidation study of this contract cannot be run")
        if prices and cid in prices and f.tick_size and f.tick_size > 0:
            p = pd.to_numeric(prices[cid], errors="coerce").dropna()
            q = p / f.tick_size
            off = int((np.abs(q - q.round()) > 1e-6 * np.maximum(1.0, np.abs(q))).sum())
            if off:
                add(cid, "price_off_tick", "warning", f"{off} prices are not a multiple of the tick size {f.tick_size:g}")
    by_root: dict[str, list[Future]] = {}
    for f in futures:
        by_root.setdefault(str(getattr(f, "root", None) or f.underlying_id or f.instrument_id.rstrip("0123456789")), []).append(f)
    for root, fs in by_root.items():
        fs = [f for f in fs if f.expiry is not None and not pd.isna(f.expiry)]
        exp = [pd.Timestamp(f.expiry) for f in fs]
        if len(set(exp)) != len(exp):
            add(root, "duplicate_expiry", "error", "two contracts of the chain have the same expiry")
        multipliers = {f.contract_multiplier for f in fs}
        if len(multipliers) > 1:
            add(root, "mixed_multiplier", "warning", f"the chain mixes multipliers {sorted(multipliers)}: returns and P&L across contracts need each contract's own")
    return out


def roll_table(long: pd.DataFrame, roll_days: int = 5, exec_cost_points: float = 0.0) -> pd.DataFrame:
    """One row per roll of the front-contract splice: the date, the contracts, both prices on that date, the calendar spread (new minus old, in points and as a fraction of the old price), the days between the two
    expiries, the annualised roll yield the spread implies for a long holder (``-spread / old x 365 / days``: negative in contango), the execution cost and whether the market was in contango or backwardation.
    The spread is NOT a profit or a loss of the roll: ``realised_by_roll`` is only the cost of crossing two markets."""
    from .futures import continuous_series

    cs = continuous_series(long, roll_days, "none")
    df = long.copy()
    df["date"], df["expiry"] = pd.to_datetime(df["date"]), pd.to_datetime(df["expiry"])
    px = {c: g.set_index("date")["price"] for c, g in df.groupby("contract")}
    exp = df.drop_duplicates("contract").set_index("contract")["expiry"]
    rows = []
    held = cs.contract
    for d in cs.roll_dates:
        i = list(held.index).index(d)
        old, new = held.iloc[i - 1], held.iloc[i]
        o, n = float(px[old][d]), float(px[new][d])
        days = max((exp[new] - exp[old]).days, 1)
        spread = n - o
        rows.append({"date": d, "from": old, "to": new, "old_price": o, "new_price": n, "spread_points": spread, "spread_pct": spread / o, "days_between_expiries": days,
                     "annualised_roll_yield": -(spread / o) * 365.0 / days, "structure": "contango" if spread > 0 else "backwardation" if spread < 0 else "flat",
                     "realised_by_roll": -float(exec_cost_points)})
    return pd.DataFrame(rows)


def executability(cs, long: pd.DataFrame | None = None) -> dict:
    """How far a continuous series' levels are from prices that could have traded: for each date the adjusted level and the raw (executable) price, the largest gap, whether any adjusted level is at or below zero, and
    which dates are executable (the adjusted and raw levels agree)."""
    adj, raw = cs.price, cs.raw
    gap = (adj - raw)
    exact = gap.abs() <= 1e-9 * np.maximum(1.0, raw.abs())
    return {"n": int(len(adj)), "executable_from": None if not exact.any() else str(exact[exact].index[0].date()) if exact.iloc[-1] else None, "share_executable": float(exact.mean()),
            "max_abs_gap": float(gap.abs().max()), "max_relative_gap": float((gap.abs() / raw.abs().replace(0, np.nan)).max()), "min_adjusted_level": float(adj.min()), "non_positive_levels": int((adj <= 0).sum()),
            "rolls": len(cs.roll_dates),
            "meaning": "Only the dates after the last roll are executable prices. Earlier levels of an adjusted series are research levels: a percentage of them is not a return, and a level at or below zero is not a price."}
