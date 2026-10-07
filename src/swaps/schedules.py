"""Swap schedules: day-count conventions, date generation, business-day adjustment, resets, payments and notional profiles.

**Day counts** (``year_fraction``): ``ACT/360``, ``ACT/365F``, ``30/360`` (US bond basis, with the end-of-February rules left out), ``30E/360`` (Eurobond basis) and ``ACT/ACT``
(ISDA: each calendar year's days over 365 or 366).

**Generation** (``generate_schedule``): regular periods of ``frequency`` months are built BACKWARD from maturity (the market standard: the stub falls at the front) or FORWARD from the
effective date (``short_back``/``long_back``), optionally on month ends (``end_of_month``). A short stub is a first or last period shorter than the regular one; a long stub merges it
into its neighbour. Every period carries its unadjusted and adjusted dates (business-day convention of the payment calendar), its accrual fraction (day count on the adjusted dates),
its payment date (adjusted end plus the payment lag in business days), and for floating legs its reset and fixing dates (reset in advance at the adjusted start or in arrears at the
adjusted end, fixing ``fixing_lag_days`` reset-calendar business days earlier).

**Notional profile** (``notional_profile``): the notional at the START of each period: ``bullet`` (constant), ``amortizing`` (each period repays the next amount in the schedule; the
last amount repeats), ``accreting`` (each period adds it) or ``custom`` (the schedule is the list of start-of-period notionals).
"""

from __future__ import annotations

import calendar as _cal

import pandas as pd

from ..instruments.calendars import get_calendar
from ..instruments.rates import months_in

DAY_COUNTS = ("ACT/360", "ACT/365F", "30/360", "30E/360", "ACT/ACT")


def _d360(d1, d2, euro: bool) -> float:
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if euro:
        dd1, dd2 = min(dd1, 30), min(dd2, 30)
    else:
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31 and dd1 >= 30:
            dd2 = 30
    return (360 * (y2 - y1) + 30 * (m2 - m1) + (dd2 - dd1)) / 360.0


def year_fraction(start, end, convention: str) -> float:
    """The accrual fraction of a year between two dates under a day-count convention."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e < s:
        return -year_fraction(e, s, convention)
    if convention == "ACT/360":
        return (e - s).days / 360.0
    if convention == "ACT/365F":
        return (e - s).days / 365.0
    if convention == "30/360":
        return _d360(s, e, False)
    if convention == "30E/360":
        return _d360(s, e, True)
    if convention == "ACT/ACT":
        total, cur = 0.0, s
        while cur < e:
            year_end = pd.Timestamp(cur.year + 1, 1, 1)
            nxt = min(e, year_end)
            total += (nxt - cur).days / (366.0 if _cal.isleap(cur.year) else 365.0)
            cur = nxt
        return total
    raise ValueError(f"unknown day count '{convention}'; use one of {DAY_COUNTS}")


def _add_months(d: pd.Timestamp, k: int, eom: bool) -> pd.Timestamp:
    out = d + pd.DateOffset(months=k)
    if eom and d == d + pd.offsets.MonthEnd(0):
        out = out + pd.offsets.MonthEnd(0)
    return out


def unadjusted_dates(effective, maturity, frequency: str, stub: str = "short_front", end_of_month: bool = False) -> list[pd.Timestamp]:
    """The period boundary dates (including effective and maturity), before business-day adjustment."""
    eff, mat = pd.Timestamp(effective), pd.Timestamp(maturity)
    m = months_in(frequency)
    if m == 0:
        return [eff, mat]
    backward = stub in ("short_front", "long_front")
    dates = [mat] if backward else [eff]
    k = 1
    while True:
        d = _add_months(mat, -m * k, end_of_month) if backward else _add_months(eff, m * k, end_of_month)
        if (backward and d <= eff) or (not backward and d >= mat):
            break
        dates.append(d)
        k += 1
    if backward:
        dates.append(eff)
        dates = dates[::-1]
        if stub == "long_front" and len(dates) > 2 and dates[1] - dates[0] < pd.Timedelta(days=15 * m):
            dates.pop(1)                                             # merge a short front stub into the next period
    else:
        dates.append(mat)
        if stub == "long_back" and len(dates) > 2 and dates[-1] - dates[-2] < pd.Timedelta(days=15 * m):
            dates.pop(-2)
    return dates


def generate_schedule(effective, maturity, frequency: str, daycount: str, calendar: str = "WEEKDAY", convention: str = "modified_following", stub: str = "short_front",
                      end_of_month: bool = False, payment_lag_days: int = 0, payment_calendar: str | None = None, fixing_lag_days: int = 2, reset_calendar: str | None = None,
                      in_arrears: bool = False, adjust_accrual: bool = True) -> pd.DataFrame:
    """One row per accrual period: ``start``, ``end`` (unadjusted), ``adj_start``, ``adj_end``, ``accrual``, ``pay_date``, ``reset_date``, ``fixing_date``."""
    cal = get_calendar(calendar)
    pay_cal = get_calendar(payment_calendar or calendar)
    reset_cal = get_calendar(reset_calendar or calendar)
    dates = unadjusted_dates(effective, maturity, frequency, stub, end_of_month)
    rows = []
    for a, b in zip(dates[:-1], dates[1:]):
        adj_a, adj_b = cal.adjust(a, convention), cal.adjust(b, convention)
        acc = year_fraction(adj_a, adj_b, daycount) if adjust_accrual else year_fraction(a, b, daycount)
        pay = pay_cal.add_business_days(pay_cal.adjust(adj_b, convention), payment_lag_days) if payment_lag_days else pay_cal.adjust(adj_b, convention)
        reset = adj_b if in_arrears else adj_a
        fixing = reset_cal.add_business_days(reset, -fixing_lag_days) if fixing_lag_days else reset
        rows.append({"start": a, "end": b, "adj_start": adj_a, "adj_end": adj_b, "accrual": acc, "pay_date": pay, "reset_date": reset, "fixing_date": fixing})
    return pd.DataFrame(rows)


def notional_profile(notional: float, n_periods: int, kind: str = "bullet", schedule: tuple = ()) -> list[float]:
    """The notional at the start of each of ``n_periods`` accrual periods."""
    if kind == "bullet":
        return [float(notional)] * n_periods
    if kind == "custom":
        sched = list(schedule)
        if len(sched) < n_periods:
            sched = sched + [sched[-1]] * (n_periods - len(sched))
        return [float(x) for x in sched[:n_periods]]
    steps = list(schedule)
    if not steps:
        raise ValueError(f"{kind} needs an amortization schedule")
    out, cur = [], float(notional)
    for i in range(n_periods):
        out.append(cur)
        step = steps[i] if i < len(steps) else steps[-1]
        cur = cur - step if kind == "amortizing" else cur + step
        cur = max(cur, 0.0)
    return out
