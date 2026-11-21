from __future__ import annotations

import calendar as _cal
from datetime import date


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(y, m + 1, min(d.day, _cal.monthrange(y, m + 1)[1]))


def month_end(year: int, month: int) -> date:
    return date(year, month, _cal.monthrange(year, month)[1])


def period_end(fiscal_year: int, fiscal_end_month: int, kind: str) -> date:
    """The end date of a fiscal ``kind`` (FY, Q1, H1, Q3) of the fiscal year numbered ``fiscal_year``, which ends in ``fiscal_end_month`` of that year."""
    start = month_end(fiscal_year - 1, fiscal_end_month)
    return month_end(*(lambda d: (d.year, d.month))(add_months(start, {"Q1": 3, "H1": 6, "Q2": 6, "Q3": 9, "FY": 12}[kind])))


def number(text) -> float | None:
    t = str(text or "").strip().replace(",", "").replace(" ", "")
    if not t or t in ("-", "−"):
        return None
    neg = t.startswith("(") and t.endswith(")") or t.startswith("△") or t.startswith("▲")
    t = t.strip("()△▲")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v
