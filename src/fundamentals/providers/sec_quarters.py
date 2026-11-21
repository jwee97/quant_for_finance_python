"""Quarterly statements from the SEC's company facts: the 10-Q reports, read the way the annual ones are (see :mod:`sec_edgar`), so that a US company has trailing-twelve-month figures from real filings.

A 10-Q is not a small 10-K. What it holds, and what this module does about it:

* **Flows come as the quarter and as the year to date.** The income statement gives the three months and the year to date; the cash flow statement gives only the year to date (Q1: three months, Q2:
  six, Q3: nine). A quarter's flow is the three-month fact when the filing tags one, else the year to date less the year to date of the quarter before (from the earlier filing, as first reported).
  A line with neither is left out: a quarter is never guessed.
* **The fourth quarter is never filed.** The 10-K holds the year, so Q4 is the year less the nine months (flows) and the year-end balance (stocks). It is public on the day of the 10-K and is marked
  as derived in the period's record id (``<accession>#Q4``): it was never a figure a company reported.
* **Per-share figures and weighted share counts do not add up**, so they are kept only when the quarter itself is tagged, and never derived for Q4.
* **A fact is public on the day its filing was made.** The quarter is the 10-Q in which it first appeared as the current period (not the comparative column of a later one), with that filing's date; a
  restatement is a later filing and the point-in-time engine treats it as a later vintage.
* **The fiscal quarter** is numbered from the company's own fiscal year ends (the annual statements): a period ending one quarter before a year end is Q3 and so on, and a quarter after the last year end
  continues the next fiscal year.

Nothing here talks to the SEC: it reads a payload the provider has already fetched (or the stored trimmed copy).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, timedelta

from .. import canonical
from ..store import PeriodStatement

QUARTER_FORMS = frozenset({"10-Q", "10-Q/A", "10-QT", "10-QT/A"})
QUARTER_DAYS = (80, 100)                                   # three months, with the 13-week quarters and the odd month
NINE_MONTH_DAYS = (255, 290)
TOL = 7
QUARTER_LENGTH = 91.31

NON_ADDITIVE = frozenset(source for code, options in canonical.SEC_MAP.items() if canonical.METRICS[code].kind in ("per_share", "shares") for source, _ in options)


def _near(a: date, b: date, tolerance: int = TOL) -> bool:
    return abs((a - b).days) <= tolerance


def _span(f) -> int | None:
    return None if f.start is None else (f.end - f.start).days


def locate(end: date, year_ends: list[tuple[date, int]]) -> tuple[int, int] | None:
    """``(fiscal_year, quarter)`` of a quarter ending ``end``, from the company's fiscal year ends ``[(date, fiscal_year)]``; ``None`` for a date that is a year end or does not sit on a quarter."""
    ends = sorted(year_ends)
    if not ends:
        return None
    ahead = next(((d, fy) for d, fy in ends if (d - end).days >= -TOL), None)
    if ahead is None:                                                                          # after the last year end: the next fiscal year(s)
        last, fy = ends[-1]
        steps = max(1, math.ceil(((end - last).days - TOL) / 365.25))
        year_end, label = last + timedelta(days=round(365.25 * steps)), fy + steps
    else:
        year_end, label = ahead
        while (year_end - end).days > 3 * QUARTER_LENGTH + 20:                                 # before the first year end we hold: step back by years
            year_end, label = year_end - timedelta(days=365), label - 1
    behind = round((year_end - end).days / QUARTER_LENGTH)
    if behind not in (1, 2, 3):
        return None
    return label, 4 - behind


def parse_quarters(payload: dict) -> tuple[list[PeriodStatement], dict]:
    """``(periods, info)``: one :class:`PeriodStatement` for each fiscal quarter the payload's 10-Q filings (and, for the fourth quarter, its 10-K) support. ``info["lines"]`` says what was read and how."""
    from . import sec_edgar as se
    annual, summary = se.parse_companyfacts(payload)
    if not annual:
        return [], {**summary, "lines": []}
    facts, currency = se._read_facts(payload, se.ANNUAL_FORMS | QUARTER_FORMS)
    year_ends = [(date.fromisoformat(p.period_end), p.fiscal_year) for p in annual if p.fiscal_year is not None]
    meta: dict[str, tuple[str, str]] = {}
    by_accn: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for field, items in facts.items():
        for f in items:
            by_accn[f.accn][field].append(f)
            meta[f.accn] = (f.filed, f.form)
    # each 10-Q's own quarter: the latest date that several different elements end on
    quarters: dict[tuple[int, int], dict] = {}
    for accn in sorted((a for a in by_accn if meta[a][1] in QUARTER_FORMS), key=lambda a: (meta[a][0], a)):
        ends: dict[date, set] = defaultdict(set)
        for field, items in by_accn[accn].items():
            if field not in canonical.SEC_FIELDS or canonical.SEC_FIELDS[field] == "cover":
                continue
            for f in items:
                if se._unit_ok(f.unit, currency):
                    ends[f.end].add(field)
        strong = [e for e, fields in ends.items() if len(fields) >= se.MIN_FIELDS_PER_PERIOD]
        if not strong:
            continue
        end = max(strong)
        where = locate(end, year_ends)
        if where is None or where in quarters:                                                 # the first filing to report the quarter keeps it
            continue
        quarters[where] = {"accn": accn, "end": end, "filed": meta[accn][0], "form": meta[accn][1]}
    ytd_start: dict[tuple[int, int], date | None] = {}
    for where, q in quarters.items():                                                          # the start of the fiscal year: where the year-to-date facts begin
        starts = Counter(f.start for items in by_accn[q["accn"]].values() for f in items if f.start is not None and f.end == q["end"] and (_span(f) or 0) >= QUARTER_DAYS[0] - 1 and se._unit_ok(f.unit, currency))
        long_starts = Counter({s: n for s, n in starts.items() if (q["end"] - s).days > QUARTER_DAYS[1] or where[1] == 1})
        ytd_start[where] = (long_starts or starts).most_common(1)[0][0] if (long_starts or starts) else None

    def pick(candidates, end: date, kind: str):
        best = None
        for f in candidates:
            if not se._unit_ok(f.unit, currency):
                continue
            if kind == "instant" and f.start is None and _near(f.end, end):
                best = f
            elif kind == "cover" and f.end >= end - timedelta(days=TOL):
                best = f if best is None or f.end >= best.end else best
        return best

    def flow_in(field: str, accn: str, start: date | None, end: date, low: int, high: int):
        """The fact in a filing that covers ``start``..``end`` (or a span in ``low``..``high`` days) and ends on ``end``."""
        found = None
        for f in by_accn[accn].get(field, []):
            if f.start is None or not se._unit_ok(f.unit, currency) or not _near(f.end, end):
                continue
            if start is not None and not _near(f.start, start):
                continue
            if start is None and not (low <= (f.end - f.start).days <= high):
                continue
            found = f
        return found

    def earlier(field: str, where_key: tuple[int, int], start: date, end: date):
        """A year-to-date fact (``start``..``end``) from a filing no later than the quarter's: how the quarter before was reported."""
        accns = [q["accn"] for k, q in quarters.items() if k[0] == where_key[0] and k[1] <= where_key[1]]
        for accn in accns:
            hit = flow_in(field, accn, start, end, 0, 0)
            if hit is not None:
                return hit
        return None

    periods: list[PeriodStatement] = []
    lines: list[dict] = []
    nine_months: dict[int, dict[str, float]] = {}
    for where in sorted(quarters):
        fy, qn = where
        q = quarters[where]
        accn, end = q["accn"], q["end"]
        start = ytd_start.get(where)
        values: dict[str, float] = {}
        derived = 0
        for field, kind in canonical.SEC_FIELDS.items():
            if kind in ("instant", "cover"):
                fact = pick([f for f in by_accn[accn].get(field, []) if field not in canonical.SEC_POSITIVE_ONLY or f.value > 0], end, kind)
                if fact is not None:
                    values[field] = fact.value
                continue
            three = flow_in(field, accn, None, end, *QUARTER_DAYS)
            if three is not None:
                values[field] = three.value
                continue
            if field in NON_ADDITIVE or start is None or qn == 1:
                continue
            ytd = flow_in(field, accn, start, end, 0, 0)
            prior_end = quarters.get((fy, qn - 1), {}).get("end")
            before = earlier(field, (fy, qn - 1), start, prior_end) if prior_end is not None else None
            if ytd is not None and before is not None:
                values[field] = ytd.value - before.value
                derived += 1
        if qn == 3 and start is not None:                                                       # the nine months, kept for the fourth quarter
            nine = {}
            for field, kind in canonical.SEC_FIELDS.items():
                if kind == "duration" and field not in NON_ADDITIVE:
                    fact = flow_in(field, accn, start, end, 0, 0)
                    if fact is not None:
                        nine[field] = fact.value
            se._build_synthetic(nine)
            nine_months[fy] = nine
        se._build_synthetic(values)
        values = {f: v for f, v in values.items() if f in se._MAPPED}
        if not values:
            continue
        taxonomy = "IFRS" if sum(k.startswith("ifrs-full:") for k in values) > sum(k.startswith("us-gaap:") for k in values) else "US GAAP"
        periods.append(PeriodStatement(f"Q{qn}", end.isoformat(), values, fiscal_year=fy, period_start=(end - timedelta(days=91)).isoformat(), filing_date=q["filed"], reporting_currency=currency,
                                       accounting_standard=taxonomy, provider_record_id=accn))
        lines.append({"fiscal_year": fy, "fiscal_period": f"Q{qn}", "period_end": end.isoformat(), "filing_date": q["filed"], "form": q["form"], "accession": accn, "derived_lines": derived})
    for p in annual:                                                                           # the fourth quarter: the year less the nine months, the year-end balance
        nine = nine_months.get(p.fiscal_year)
        if not nine:
            continue
        annual_values = dict(p.values)
        values = {}
        for field, kind in canonical.SEC_FIELDS.items():
            if kind in ("instant", "cover") and field in annual_values:
                values[field] = annual_values[field]
            elif kind == "duration" and field not in NON_ADDITIVE and field in annual_values and field in nine:
                values[field] = annual_values[field] - nine[field]
        for name in canonical.SEC_SUMS:                                                          # the made-up sums: a balance is the year-end one, a flow is the year less the nine months
            if name in (canonical.SEC_DEBT_CURRENT, canonical.SEC_CASH_AND_INVESTMENTS, canonical.SEC_TEMP_EQUITY):
                if name in annual_values:
                    values[name] = annual_values[name]
            elif name in annual_values and name in nine:
                values[name] = annual_values[name] - nine[name]
        se._build_synthetic(values)
        values = {f: v for f, v in values.items() if f in se._MAPPED}
        if not values:
            continue
        end = date.fromisoformat(p.period_end)
        periods.append(PeriodStatement("Q4", p.period_end, values, fiscal_year=p.fiscal_year, period_start=(end - timedelta(days=91)).isoformat(), filing_date=p.filing_date, reporting_currency=currency,
                                       accounting_standard=p.accounting_standard, provider_record_id=f"{p.provider_record_id}#Q4"))
        lines.append({"fiscal_year": p.fiscal_year, "fiscal_period": "Q4", "period_end": p.period_end, "filing_date": p.filing_date, "form": "10-K", "accession": f"{p.provider_record_id}#Q4", "derived_lines": len(values)})
    periods.sort(key=lambda x: (x.period_end, x.fiscal_period))
    return periods, {**summary, "lines": lines}
