"""Period resolution: what a column of figures covers.

A heading is read for the *end date*, the *duration* and whether it is a forecast or a restatement; the statement says whether the column is an instant (a balance sheet) or a span. Nothing here uses the
file name or an isolated year: a year in a heading becomes a period only if the heading belongs to a column of reported figures, and it borrows its day and month from the heading itself ("Year ended March 31"),
then from the fiscal year end the document states, then (flagged as a guess) from December 31. 52/53-week years (Apple's September 26, 2020) are recognised by the end date lying within a few days of a month end.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from . import pdf as pdf_module
from .model import FinancialPeriod

DURATION = (
    (re.compile(r"twelve\s+months|12\s+months|years?\s+ended|year\s+ended|fiscal\s+years?|full[\s-]year|annual|52\s+weeks|53\s+weeks|12개월|十二个月|十二個月|通年|年度|当連結会計年度|前連結会計年度|exercice|geschäftsjahr|ejercicio", re.I), 12, "ANNUAL"),
    (re.compile(r"six\s+months|6\s+months|half[\s-]year|interim|first\s+half|second\s+half|半期|上半期|六个月|六個月|상반기|반기|semestre|halbjahr", re.I), 6, "HALF_YEAR"),
    (re.compile(r"nine\s+months|9\s+months|year[\s-]to[\s-]date|\bytd\b|九个月|九個月|9개월|neuf\s+mois|neun\s+monate", re.I), 9, "YTD"),
    (re.compile(r"three\s+months|3\s+months|quarter|\bq[1-4]\b|四半期|分期|第[一二三四1-4]季|3개월|trimestre|quartal|三个月|三個月", re.I), 3, "QUARTER"),
)
FORECAST = re.compile(r"forecast|forecasts|estimate[sd]?|outlook|projection|projected|guidance|budget|plan\b|target\b|予想|見通し|予測|전망|예상|预测|預測|展望|pronostic|prévision|prognose|pronóstico", re.I)
RESTATED = re.compile(r"restated|re-?presented|as\s+restated|as\s+adjusted|revised|修正|訂正|재작성|重述|retraité|angepasst", re.I)
WEEKS = re.compile(r"\b(52|53)[-\s]weeks?\b", re.I)


def month_end(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def nominal_month(end: date) -> tuple[int, int]:
    """(year, month) a period end belongs to: within a week of a month end it is that month (a 52/53-week year ending September 26 is a September year)."""
    last = month_end(end.year, end.month)
    if last - end.day <= 7:
        return end.year, end.month
    if end.day <= 6:                                                                                  # the first days of the next month: the previous month
        prev = end - timedelta(days=end.day)
        return prev.year, prev.month
    return end.year, end.month


def months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month + (1 if end.day >= month_end(end.year, end.month) - 7 else 0) - (0 if start.day <= 3 else 1)


def fiscal_year_of(end: date, year_end_month: int | None) -> int:
    y, m = nominal_month(end)
    if year_end_month is None:
        return y
    return y if m <= year_end_month else y + 1


def fiscal_quarter_of(end: date, year_end_month: int | None) -> int:
    _, m = nominal_month(end)
    ye = year_end_month or 12
    return ((m - ye - 1) % 12) // 3 + 1


def duration_of(text: str) -> tuple[int | None, str | None]:
    for rx, months, kind in DURATION:
        if rx.search(text):
            return months, kind
    return None, None


def build(end: date, text: str, instant: bool, year_end_month: int | None, guessed_date: bool = False) -> FinancialPeriod:
    """The period of a column whose end date is known, from the words of its heading (``text``) and whether the statement is a balance sheet (``instant``)."""
    forecast = bool(FORECAST.search(text))
    restated = bool(RESTATED.search(text))
    weeks = int(WEEKS.search(text).group(1)) if WEEKS.search(text) else None
    fy, fq = fiscal_year_of(end, year_end_month), None
    if instant:
        return FinancialPeriod(end_date=end.isoformat(), period_type="INSTANT", fiscal_year=fy, is_forecast=forecast, is_restated=restated)
    months, kind = duration_of(text)
    if kind is None:
        ye = year_end_month
        if ye is not None and nominal_month(end)[1] == ye:
            months, kind = 12, "ANNUAL"
        else:
            return FinancialPeriod(end_date=end.isoformat(), period_type="UNKNOWN_FLOW", fiscal_year=fy, fiscal_quarter=fiscal_quarter_of(end, year_end_month), is_forecast=forecast, is_restated=restated)
    if kind == "QUARTER":
        fq = fiscal_quarter_of(end, year_end_month)
    start = None
    if months:
        y, m = nominal_month(end)
        total = y * 12 + (m - 1) - months + 1
        sy, sm = divmod(total, 12)
        start = date(sy, sm + 1, 1).isoformat()
    return FinancialPeriod(end_date=end.isoformat(), start_date=start, duration_months=months, period_type=kind, fiscal_year=fy, fiscal_quarter=fq, is_forecast=forecast, is_restated=restated, weeks=weeks)


def fiscal_period_label(p: FinancialPeriod, year_end_month: int | None) -> str:
    """The label the rest of the importer uses: FY, Q1-Q4, H1, H2 (an instant takes the label of its position in the fiscal year)."""
    if p.period_type in ("ANNUAL",):
        return "FY"
    if p.period_type == "QUARTER":
        return f"Q{p.fiscal_quarter or 1}"
    if p.period_type == "HALF_YEAR":
        end = date.fromisoformat(p.end_date)
        return "H2" if year_end_month and nominal_month(end)[1] == year_end_month else "H1"
    end = date.fromisoformat(p.end_date)
    ye = year_end_month
    if ye is not None:
        return "FY" if nominal_month(end)[1] == ye else f"Q{fiscal_quarter_of(end, ye)}"
    return "FY" if end.month == 12 else f"Q{fiscal_quarter_of(end, None)}"


YEAR_TOKEN = re.compile(r"(?<![\d.,])((?:19|20)\d{2})(?![\d,]|\s*%)")


def column_date(levels: list[str], context: str, year_end: tuple | None, ctx: dict | None) -> tuple[str, bool] | None:
    """The end date a column's own heading names, bottom level first: a full date; a numbered term; or a single year, which takes its day and month from the heading context ("Year ended March 31"),
    then from the fiscal year end the report states, then (flagged as a guess) from December 31. Returns ``(iso date, guessed)`` or None. Two or more dates in one level are not read here (the heading
    is merged or spans columns: the table-level reader orders them)."""
    for level in reversed(levels):
        found = pdf_module._headings(level, year_end, ctx or {})
        if len(found) == 1:
            return found[0]["period_end"], bool(found[0].get("guessed"))
        if len(found) > 1:
            return None
        terms = list(dict.fromkeys(int(t) for t in pdf_module.TERM.findall(level)))
        if len(terms) == 1 and ctx and ctx.get("terms") and terms[0] in ctx["terms"]:
            return ctx["terms"][terms[0]].isoformat(), False
        years = YEAR_TOKEN.findall(level)
        if len(set(years)) == 1:
            named = pdf_module._day_month(" ".join(levels) + " " + context)
            if named is None:                                                                         # "As at 31 March 2020" in the title: the day and month of the statement's own date
                full = pdf_module._headings(context, year_end, ctx or {})
                if len(full) >= 1 and not full[0].get("guessed"):
                    d0 = date.fromisoformat(full[0]["period_end"])
                    named = (d0.month, d0.day)
            month, day = named or year_end or (12, 31)
            y = int(years[0])
            return date(y, month, min(day, month_end(y, month))).isoformat(), named is None and year_end is None
    return None


def heading_dates(text: str, year_end: tuple | None, ctx: dict | None) -> list[dict]:
    """The period ends a piece of heading text names (reuses the multilingual date reader)."""
    return pdf_module._headings(text, year_end, ctx or {})


__all__ = ["build", "duration_of", "fiscal_period_label", "fiscal_year_of", "fiscal_quarter_of", "nominal_month", "heading_dates", "FORECAST", "RESTATED"]
