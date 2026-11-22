"""Stock splits, found in the stored prices, and what they do to share counts and per-share figures.

A market capitalisation is the unadjusted close times a share count taken from a filing. If the company splits 4:1 *after* that filing, the close drops to a quarter and the filing's share count does not,
so the market capitalisation (and every yield built on it) is a quarter of the truth until the next filing. The same holds for per-share figures (book value, net current assets, Graham number) set
against a post-split price.

Splits are inferred, not looked up: on a split day the unadjusted close falls by the ratio while the adjusted close does not, so ``(close_t / close_{t-1}) / (adj_t / adj_{t-1})`` is about 1/ratio; on a
dividend day it is only slightly below 1. A move of the quotient outside [0.8, 1.25] that is close to a usual ratio (2:1, 3:2, 5:1, 1:10 …) is taken as a split; any other large move is reported as
*unexplained* and is **not** applied (a merger, a spin-off or a data error is not a split). Spin-offs and mergers are not detected here: they change the company, not the share count, and are listed as a
limitation. A source that stores only adjusted prices gives no splits at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOW, HIGH = 0.8, 1.25
USUAL = sorted({float(a) for a in range(2, 21)} | {1.5, 2.5, 4 / 3, 5 / 4} | {1.0 / a for a in range(2, 21)} | {2 / 3, 4 / 5, 2 / 5})
TOLERANCE = 0.03


def _snap(ratio: float) -> float | None:
    best = min(USUAL, key=lambda u: abs(np.log(ratio / u)))
    return best if abs(np.log(ratio / best)) <= TOLERANCE else None


def detect(close: pd.DataFrame, adj: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """``(splits, unexplained)``: each ``{security_id, date, ratio, quotient}``; ``ratio`` is shares after per share before (4.0 for a 4-for-1 split, 0.1 for a 1-for-10 reverse split)."""
    quotient = (close / close.shift(1)) / (adj / adj.shift(1))
    splits, other = [], []
    for sid in quotient.columns:
        q = quotient[sid].dropna()
        for date, v in q[(q < LOW) | (q > HIGH)].items():
            ratio = 1.0 / float(v)
            snapped = _snap(ratio)
            rec = {"security_id": int(sid), "date": date, "quotient": float(v)}
            (splits if snapped else other).append({**rec, "ratio": snapped if snapped else ratio})
    return splits, other


def cumulative_log(close: pd.DataFrame, adj: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Daily cumulative log split ratio per company (0 before the first split) and the events."""
    splits, other = detect(close, adj)
    cum = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for s in splits:
        cum.loc[cum.index >= s["date"], s["security_id"]] += np.log(s["ratio"])
    return cum, splits, other


def since_update(panel: pd.DataFrame, cum_at_dates: pd.DataFrame) -> pd.DataFrame:
    """For each company and date, the cumulative split ratio between the date the panel's value last changed (a new filing) and the date: the factor a share count must be multiplied by (and a per-share
    figure divided by) to be on the basis of the date's price. Where a company has no value the factor is 1."""
    out = pd.DataFrame(1.0, index=panel.index, columns=panel.columns)
    cum = cum_at_dates.reindex(index=panel.index, columns=panel.columns).fillna(0.0)
    for c in panel.columns:
        v = panel[c].to_numpy(float)
        changed = np.r_[True, ~np.isclose(v[1:], v[:-1], rtol=1e-12, atol=0, equal_nan=True)]
        last = np.maximum.accumulate(np.where(changed, np.arange(len(v)), 0))
        base = cum[c].to_numpy()[last]
        out[c] = np.exp(cum[c].to_numpy() - base)
    return out.where(panel.notna(), 1.0)


# ------------------------------------------------------------------------------------------------------------------------ spin-offs, mergers and the like
EVENT_KINDS = ("spin_off", "merger", "acquisition", "delisting", "reorganisation", "other")


def record_event(store, security_id: int, event_date: str, kind: str, counterparty: str = "", value_fraction: float | None = None, note: str = "") -> None:
    """Record something the prices cannot tell apart from a data error: a spin-off (``value_fraction`` is the share of the company's value that left), a merger or acquisition, a reorganisation. Splits are
    found automatically and need no record. The statements do not reflect the event until the next filing after it, so between the two dates market-based figures are not reliable."""
    from datetime import date as _d
    from .errors import FundamentalsError
    from .schema import now_iso
    if kind not in EVENT_KINDS:
        raise FundamentalsError(f"kind is one of {', '.join(EVENT_KINDS)}")
    try:
        _d.fromisoformat(event_date)
    except ValueError:
        raise FundamentalsError("event_date is a date such as 2021-06-30") from None
    if value_fraction is not None and not 0 < value_fraction < 1:
        raise FundamentalsError("value_fraction is the share of value that left, between 0 and 1 (0.3 for a spin-off worth 30% of the company)")
    with store.db.transaction() as c:
        c.execute("INSERT OR REPLACE INTO corporate_events (security_id, event_date, kind, counterparty, value_fraction, note, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (int(security_id), event_date, kind, counterparty[:120], value_fraction, note[:300], now_iso()))


def events(store, security_ids: list[int] | None = None) -> list[dict]:
    sql, args = "SELECT security_id, event_date, kind, counterparty, value_fraction, note FROM corporate_events", []
    if security_ids is not None:
        sql += f" WHERE security_id IN ({','.join('?' * len(security_ids))})"
        args = [int(i) for i in security_ids]
    with store.db.transaction() as c:
        return [dict(zip(("security_id", "event_date", "kind", "counterparty", "value_fraction", "note"), r)) for r in c.execute(sql + " ORDER BY event_date", args).fetchall()]


def unreliable_mask(ctx) -> pd.DataFrame:
    """Dates by companies, True where market-based figures cannot be trusted: from a recorded event until the first rebalance date on which a filing newer than the event is the latest the company has
    (a spin-off's business is still in the statements, an acquirer's new shares are not yet in its share count). A delisting ends the company's series and needs no window."""
    out = pd.DataFrame(False, index=ctx.dates, columns=ctx.ids)
    evs = [e for e in events(ctx.store, ctx.ids) if e["kind"] != "delisting"]
    if not evs:
        return out
    rev = ctx.panel("revenue").where(lambda x: x.notna())
    for e in evs:
        sid, day = e["security_id"], pd.Timestamp(e["event_date"])
        after = ctx.dates[ctx.dates >= day]
        if not len(after) or sid not in out.columns:
            continue
        col = ctx.tables.fy if "fy" in dir(ctx.tables) else None
        newest = None
        if col is not None and len(col):
            fyrows = col[(col["security_id"] == sid) & (col["available_date"] > day)]
            newest = fyrows["available_date"].min() if len(fyrows) else None
        stop = newest if newest is not None and pd.notna(newest) else None
        out.loc[(out.index >= day) & (True if stop is None else out.index < stop), sid] = True
    return out


def candidates(ctx) -> list[dict]:
    """Large price moves that are not splits and have no recorded event: what a person should look at and, if it is a spin-off or merger, record."""
    ctx.split_factor(ctx.panel("shares_outstanding")) if getattr(ctx, "_cache", None) is not None and "split_cum" not in ctx._cache else None
    have = {(e["security_id"], e["event_date"]) for e in events(ctx.store, ctx.ids)}
    return [{"security_id": o["security_id"], "date": str(o["date"])[:10], "ratio": o["ratio"], "note": "the unadjusted close moved by a factor that is not a usual split ratio: spin-off, merger or a data error?"}
            for o in getattr(ctx, "unexplained_price_moves", []) if (o["security_id"], str(o["date"])[:10]) not in have]
