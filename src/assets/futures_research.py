"""Term-structure research for futures: spreads, regimes, curve changes, liquidity and calendars, from the same point-in-time contract tables the rest of the workspace uses.

Every figure is computed from contracts that were listed and priced on the date in question: a contract is a fixed identifier (``CLZ26``) with its own expiry, and the "rank" panels (``F1``, ``F2`` ...) are
rebuilt for each date from what was alive then, so no date sees a contract or a price that did not yet exist. Calendar-spread figures are price *differences*: a spread is not a cost or a profit of anything
until a position in it is opened and closed (see ``futures_analytics.roll_decomposition``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import last_zscore, percentile_rank
from .futures import contract_calendar, term_structure


def _alive(data, date) -> pd.DataFrame:
    d = pd.Timestamp(date)
    g = data.long[pd.to_datetime(data.long["date"]) == d].sort_values("expiry")
    if len(g) < 2:
        raise ValueError(f"fewer than two contracts on {d.date()}")
    return g


def spread_matrix(data, date=None, depth: int = 8) -> dict:
    """Every calendar spread of the nearest ``depth`` contracts on a date: the later contract's price less the earlier's, in points, in per cent of the earlier price, and per contract in money."""
    d = pd.Timestamp(date) if date is not None else data.as_of
    g = _alive(data, d).head(depth)
    px, names = g["price"].to_numpy(float), [str(c) for c in g["contract"]]
    n = len(g)
    pts = [[None if j <= i else float(px[j] - px[i]) for j in range(n)] for i in range(n)]
    pct = [[None if j <= i else float(px[j] / px[i] - 1.0) for j in range(n)] for i in range(n)]
    return {"date": str(d.date()), "contracts": names, "expiries": [str(pd.Timestamp(e).date()) for e in g["expiry"]], "price": [float(x) for x in px], "points": pts, "percent": pct,
            "dollars": [[None if v is None else v * data.spec.multiplier for v in row] for row in pts], "convention": "column contract less row contract (later less earlier); above zero is contango",
            "multiplier": data.spec.multiplier}


def spread_history(data, near: int = 1, far: int = 2, window: int = 252) -> dict:
    """The history of the spread between the ``near``-th and ``far``-th nearest contracts (by rank on each date) with its z-score and percentile against the trailing ``window``. The contract behind a rank
    changes at each expiry: a jump at a roll is the next pair of contracts, not a move of the market (the series marks those dates)."""
    ts = term_structure(data.long[["date", "contract", "expiry", "price"]], max(near, far))
    a, b = f"F{near}", f"F{far}"
    if a not in ts["price"] or b not in ts["price"]:
        raise ValueError("the curve does not reach that far")
    sp = (ts["price"][b] - ts["price"][a]).dropna()
    pair = (ts["contract"][a].astype(str) + "/" + ts["contract"][b].astype(str)).reindex(sp.index)
    changed = pair != pair.shift(1)
    z_now = last_zscore(sp, window)
    roll = sp[changed.fillna(False)]
    return {"pair": f"F{near}/F{far}", "current": float(sp.iloc[-1]), "contracts_now": str(pair.iloc[-1]), "zscore": None if not np.isfinite(z_now) else float(z_now), "percentile": float(percentile_rank(sp.tail(window), float(sp.iloc[-1]))),
            "window": int(window), "mean": float(sp.tail(window).mean()), "std": float(sp.tail(window).std(ddof=1)), "dates": [str(x.date()) for x in sp.index[-600:]], "spread": [round(float(v), 5) for v in sp.tail(600)],
            "contract_changes": [str(x.date()) for x in roll.index[-40:]], "note": "the pair of contracts behind F1/F2 changes at each expiry: the series is continuous in rank, not in contract"}


def regime_analysis(data, depth: int = 2, tol: float = 0.005) -> dict:
    """The curve's regime day by day from the front roll yield: ``contango`` below ``-tol`` annualised, ``backwardation`` above ``+tol``, ``flat`` between. Share of days, number and length of spells, the
    current spell, the transition matrix and the share by calendar year."""
    ts = term_structure(data.long[["date", "contract", "expiry", "price"]], max(depth, 2))
    ry = (np.log(ts["price"]["F1"] / ts["price"]["F2"]) / (ts["ttm"]["F2"] - ts["ttm"]["F1"]).where(lambda x: x > 1e-6)).dropna()
    reg = pd.Series(np.where(ry > tol, "backwardation", np.where(ry < -tol, "contango", "flat")), index=ry.index)
    spells = (reg != reg.shift(1)).cumsum()
    runs = reg.groupby(spells).agg(["first", "size"])
    trans = pd.crosstab(reg.shift(1).dropna(), reg.iloc[1:], normalize="index")
    by_year = reg.groupby(reg.index.year).value_counts(normalize=True).unstack(fill_value=0.0)
    return {"tolerance": tol, "days": int(len(reg)), "share": {k: float(v) for k, v in reg.value_counts(normalize=True).items()}, "current": str(reg.iloc[-1]), "current_spell_days": int(runs["size"].iloc[-1]),
            "spells": {k: {"count": int((runs["first"] == k).sum()), "mean_days": float(runs.loc[runs["first"] == k, "size"].mean())} for k in runs["first"].unique()},
            "transition": {str(a): {str(b): float(v) for b, v in row.items()} for a, row in trans.iterrows()},
            "by_year": {int(y): {k: float(v) for k, v in row.items()} for y, row in by_year.iterrows()},
            "dates": [str(x.date()) for x in reg.index[-600:]], "roll_yield": [round(float(v), 5) for v in ry.tail(600)],
            "caveat": "a regime label is a threshold on the front roll yield; with few years of history the spell statistics describe the sample, not a cycle"}


def change_decomposition(data, date_from, date_to) -> dict:
    """How the curve moved between two dates, for the contracts alive on both: the change in ln price of each, split into a **level** (the average change), a **slope** (change along time to expiry, from a
    straight-line fit of the changes on the centred time to expiry at the later date) and a **curvature/residual** (what is left). Contracts that exist on one date only are listed, not guessed."""
    a, b = _alive(data, date_from), _alive(data, date_to)
    both = sorted(set(a["contract"]) & set(b["contract"]), key=lambda c: float(b.loc[b["contract"] == c, "ttm"].iloc[0]))
    if len(both) < 3:
        raise ValueError("fewer than three contracts are alive on both dates: the move cannot be decomposed")
    pa = a.set_index("contract")["price"].reindex(both).to_numpy(float)
    pb = b.set_index("contract")["price"].reindex(both).to_numpy(float)
    tt = b.set_index("contract")["ttm"].reindex(both).to_numpy(float)
    ch = np.log(pb / pa)
    x = tt - tt.mean()
    slope = float(np.polyfit(x, ch, 1)[0]) if np.ptp(x) > 0 else 0.0
    level = float(ch.mean())
    resid = ch - level - slope * x
    total_ss = float(np.sum((ch - 0.0) ** 2)) or 1.0
    return {"from": str(pd.Timestamp(date_from).date()), "to": str(pd.Timestamp(date_to).date()), "contracts": [str(c) for c in both], "log_change": [float(v) for v in ch], "level": level, "slope_per_year": slope,
            "fitted_level": [level] * len(both), "fitted_slope": [float(slope * v) for v in x], "residual": [float(v) for v in resid],
            "share_of_variation": {"level": float(len(ch) * level ** 2 / total_ss), "slope": float(np.sum((slope * x) ** 2) / total_ss), "residual": float(np.sum(resid ** 2) / total_ss)},
            "only_on_first_date": sorted(set(a["contract"]) - set(b["contract"])), "only_on_second_date": sorted(set(b["contract"]) - set(a["contract"])),
            "note": "contracts that expired between the dates, and ones listed since, are left out of the decomposition"}


def liquidity(data, date=None, depth: int = 8) -> dict:
    """Open interest and volume by contract on a date, each contract's share of the total, the open-interest-weighted time to expiry and the contract with the most open interest (which is where a roll
    strategy should be). A curve without open interest says so instead of showing zeros."""
    g = _alive(data, date if date is not None else data.as_of).head(depth)
    oi, vol = g["open_interest"].to_numpy(float), g["volume"].to_numpy(float) if "volume" in g else np.full(len(g), np.nan)
    has_oi = bool(np.isfinite(oi).any())
    tot = float(np.nansum(oi)) if has_oi else None
    rows = [{"contract": str(c), "expiry": str(pd.Timestamp(e).date()), "dte": int(round(float(t) * 365)), "open_interest": None if not np.isfinite(o) else float(o), "volume": None if not np.isfinite(v) else float(v),
             "oi_share": None if not has_oi or not np.isfinite(o) else float(o / tot)} for c, e, t, o, v in zip(g["contract"], g["expiry"], g["ttm"], oi, vol)]
    weighted = float(np.nansum(oi * g["ttm"].to_numpy(float)) / tot * 365.0) if has_oi and tot else None
    return {"date": str(pd.Timestamp(g["date"].iloc[0]).date()), "rows": rows, "has_open_interest": has_oi, "oi_weighted_dte": weighted,
            "most_liquid": None if not has_oi else str(g["contract"].iloc[int(np.nanargmax(oi))]), "note": None if has_oi else "this curve has no open interest, so liquidity cannot be assessed"}


def expiry_calendar(data, months_ahead: int = 12, roll_days: int = 5) -> dict:
    """The coming expiries by the root's exchange rule, with the calendar roll date ``roll_days`` before each. Exchange holidays are not modelled: a listed date can be a day off, and the exchange's own
    calendar governs."""
    spec = data.spec
    start = data.as_of
    cal = contract_calendar(start, start + pd.DateOffset(months=months_ahead), spec.months, spec.rule)
    cal = cal[(cal["expiry"] >= start) & (cal["expiry"] <= start + pd.DateOffset(months=months_ahead))]
    return {"root": data.root, "rule": spec.rule, "rows": [{"contract": f"{data.root}{r.contract}", "expiry": str(r.expiry.date()), "weekday": r.expiry.day_name(), "roll_date": str((r.expiry - pd.Timedelta(days=roll_days)).date()),
                                                            "dte": int((r.expiry - start).days)} for r in cal.itertuples()],
            "caveat": "dates follow the exchange rule for the root; holidays are not modelled (a date may be one business day off) and a contract's actual last trading day is the exchange's"}


def compare_curves(curves: dict, date=None, depth: int = 8) -> dict:
    """Several roots' curves side by side, each normalised to its own front contract (price over the front), against days to expiry: the comparison is of shape, not of level."""
    out = []
    for root, data in curves.items():
        g = _alive(data, date if date is not None else data.as_of).head(depth)
        out.append({"root": root, "name": data.spec.name, "date": str(pd.Timestamp(g["date"].iloc[0]).date()), "dte": [int(round(float(t) * 365)) for t in g["ttm"]], "relative": [float(p / g["price"].iloc[0]) for p in g["price"]],
                    "contracts": [str(c) for c in g["contract"]]})
    return {"curves": out, "note": "each curve is divided by its own front-contract price: above 1 is contango"}
