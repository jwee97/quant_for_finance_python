"""Point-in-time eligibility.

A backtest that picks its stocks from *today's* list of index members has used the future: companies that failed, were acquired or dropped out are not on the list, so what is left is the survivors.
That hindsight can be large, and it cannot be corrected without a history of who was in the index (or listed) on each date, which this platform does not ship and does not invent.

So eligibility has two honest forms:

* ``from_table``: the person supplies dated listings (ticker, first date, last date, how it ended) and optionally dated index membership. Eligibility on a date is exactly what the table says. Coverage is
  ``complete`` only if the person says the table includes the securities that left; otherwise it is ``partial`` and the survivorship limitation is stated beside every result.
* ``current_constituents``: no table. Every ticker is eligible from its first price. The result carries ``hindsight = True`` and the sentence that the universe was chosen with knowledge of who survived.

Delistings: a security that leaves has no returns after its last day. The lab never invents a delisting return: a position still held on that day earns zero from then on and is counted in the result
(``missing_return_days``), so the limitation is visible and not silently optimistic.
"""

from __future__ import annotations

import io

import pandas as pd

from ..ml.realism import Listing, Membership

COLUMNS = ("ticker", "start", "end", "status")
HINDSIGHT = ("The universe is today's list of names, chosen with knowledge of who survived to today. Companies that failed, were acquired or left the index are absent, so returns are conditional on survival "
             "(survivorship bias) and are overstated by an amount this platform cannot measure without a point-in-time membership history. Supply one to remove it.")


class MembershipError(ValueError):
    pass


def from_table(rows: pd.DataFrame, coverage: str = "partial") -> Membership:
    """A ``Membership`` from a table with columns ticker, start and (optionally) end and status. ``end`` empty means still listed; status is one of active, delisted, failed, merged."""
    df = rows.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("ticker", "start") if c not in df.columns]
    if missing:
        raise MembershipError(f"the membership table needs the columns ticker and start (and optionally end and status); missing {missing}")
    listings = []
    for _, r in df.iterrows():
        end = r.get("end")
        end = None if end is None or (isinstance(end, float) and pd.isna(end)) or str(end).strip() == "" else pd.Timestamp(end)
        start = pd.Timestamp(r["start"])
        if end is not None and end <= start:
            raise MembershipError(f"{r['ticker']}: the end date {end.date()} is not after the start date {start.date()}")
        raw = r.get("status")
        raw = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw).strip().lower()       # an empty cell is NaN once read: it means "not said", not the word nan
        status = raw or ("active" if end is None else "delisted")
        try:
            listings.append(Listing(security_id=str(r["ticker"]).strip().upper(), ticker=str(r["ticker"]).strip().upper(), start=start, end=end, status=status))
        except ValueError as exc:
            raise MembershipError(f"{r['ticker']}: {exc}") from None
    if not listings:
        raise MembershipError("the membership table has no rows")
    return Membership(listings, coverage=coverage)


def from_csv(text: str, coverage: str = "partial") -> Membership:
    try:
        return from_table(pd.read_csv(io.StringIO(text)), coverage)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise MembershipError(f"the membership table could not be read: {exc}") from None


def eligibility(dates, tickers, membership: Membership | None) -> tuple[pd.DataFrame, dict]:
    """``(eligible, info)``: a date x ticker boolean frame and what it means. ``info['hindsight']`` is true when no table was given."""
    idx = pd.DatetimeIndex(dates)
    if membership is None:
        return pd.DataFrame(True, index=idx, columns=list(tickers)), {"basis": "current_constituents", "hindsight": True, "coverage": "none", "limitation": HINDSIGHT}
    el = membership.eligible_tickers(idx, list(tickers))
    surv = membership.survivorship()
    return el, {"basis": "point_in_time", "hindsight": False, "coverage": surv["coverage"], "securities": surv["securities"], "ended": surv["ended"], "limitation": surv["limitation"]}
