"""Financial reports a person types or pastes in themselves (the Fundamentals tab's *Enter reports*), stored with the figures of the providers and used by every study.

An entry is one company and a list of periods; each period is a column of a spreadsheet: a period end, the date the report was published, a currency and a value for any of the canonical lines
(:data:`canonical.METRICS`). The figures are stored as ``provider = "manual"`` exactly as a provider's are (periods, observations, a point-in-time availability date), so the ratios, scores,
factors and alpha tests that read the database read them with no special case. What differs, and is said on the page:

* **Units.** A person types in thousands or millions; the page says which and the figures are stored in the reporting currency's own units, like every other source.
* **Costs and outflows.** Cost of revenue, R&D, SG&A, interest, depreciation, capital spending, buybacks, dividends and acquisitions are kept as positive amounts, whichever way the person wrote
  them, ``(1,000)`` or ``1,000``.
* **Dates.** The date a report became public is what keeps a backtest honest. Without one the figures are treated as public a standard lag after the period end and flagged ``estimated``, as for
  a provider that gives no filing date; they are never treated as known on the period end.
* **Corrections and restatements.** Fixing a typing mistake overwrites the period (nothing was ever public that was wrong, so no history is kept). A real restatement is stored as a new vintage,
  public only from the date the user gives, like a provider's.
* **Rank.** Where a provider also reports a period, the provider leads and these figures fill only the lines it lacks, and only if they became public no later than the provider's report
  (:class:`pit.PitSettings` ranks an unlisted source last, and the engine never fills a gap with a figure that was published later than the figure it completes). The preview and the save say where
  the two disagree and whether the extra lines are used.

Nothing here reaches a provider and nothing is written until :func:`save`: :func:`preview` works on a temporary copy of the one company.
"""

from __future__ import annotations

import csv
import io
import math
import re
import tempfile
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import canonical, metrics, prices
from .errors import FundamentalsError
from .jsonutil import clean
from .pit import PeriodTables, PitSettings, availability
from .store import CompanyProfile, FundamentalsStore, PeriodStatement

PROVIDER = "manual"
_WRITING = threading.Lock()                                  # two requests that write the same company at once (a save and a removal) would meet halfway: one at a time
FISCAL_PERIODS = ("FY", "Q1", "Q2", "Q3", "Q4", "H1", "H2")
UNITS = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}
MAX_PERIODS = 60
MAX_TEXT = 150_000                                   # the server refuses a request body above 200,000 bytes
MAX_ROWS = 400
TICKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:\-^=]{0,23}$")
CURRENCY = re.compile(r"^[A-Za-z]{3}$")
MODES = ("correct", "restate")
STATEMENT_TITLES = {"income": "Income statement", "balance": "Balance sheet", "cashflow": "Cash flow statement", "shares": "Shares and per-share"}

# the lines most ratios and every score need: the page shows these first and says which are missing
CORE = frozenset({"revenue", "cost_of_revenue", "gross_profit", "operating_income", "income_before_tax", "income_tax", "interest_expense", "net_income", "eps_diluted",
                  "cash_and_equivalents", "accounts_receivable", "inventory", "current_assets", "total_assets", "accounts_payable", "current_liabilities", "short_term_debt", "long_term_debt",
                  "total_liabilities", "shareholders_equity", "retained_earnings", "operating_cash_flow", "capital_expenditure", "depreciation_amortization", "weighted_avg_shares_diluted",
                  "shares_outstanding"})
# lines filled in from others when left blank (canonical._derive), with the rule in words
AUTO = {"gross_profit": "revenue less cost of revenue", "ebit": "operating income", "ebitda": "EBIT plus depreciation and amortisation", "free_cash_flow": "operating cash flow less capital expenditure",
        "total_debt": "short-term plus long-term debt", "cash": "cash and equivalents", "shares_outstanding": "diluted weighted-average shares"}

# other names a line goes by in a spreadsheet or a report (the canonical label and code are always accepted); matched without case, punctuation or spacing
EXTRA_NAMES: dict[str, tuple[str, ...]] = {
    "revenue": ("total revenue", "total revenues", "revenues", "net sales", "sales", "net revenue", "net revenues", "turnover", "total net sales", "sales revenue"),
    "cost_of_revenue": ("cost of sales", "cost of goods sold", "cogs", "cost of goods and services sold", "total cost of revenue", "cost of products sold"),
    "gross_profit": ("gross income", "gross margin"),
    "research_and_development": ("r&d", "rnd", "research development", "research and development expense", "research and development expenses"),
    "selling_general_admin": ("sg&a", "sga", "selling general and administrative expenses", "selling general administrative", "selling and administrative expenses"),
    "operating_income": ("operating profit", "income from operations", "operating income loss", "ebit operating", "profit from operations"),
    "ebit": ("earnings before interest and tax", "earnings before interest and taxes"),
    "ebitda": ("earnings before interest tax depreciation and amortization", "earnings before interest tax depreciation and amortisation"),
    "interest_expense": ("interest expenses", "interest paid", "finance costs", "interest and debt expense"),
    "income_before_tax": ("pretax income", "pre tax income", "profit before tax", "earnings before tax", "ebt", "income before income taxes", "income before taxes"),
    "income_tax": ("income taxes", "tax expense", "provision for income taxes", "income tax expense", "taxes"),
    "net_income": ("net profit", "net earnings", "profit for the year", "profit for the period", "net income loss", "earnings", "net income attributable to shareholders",
                   "net income attributable to common shareholders"),
    "eps_basic": ("basic eps", "earnings per share basic", "basic earnings per share", "eps basic"),
    "eps_diluted": ("diluted eps", "earnings per share diluted", "diluted earnings per share", "eps diluted", "eps"),
    "cash": ("cash and short term investments", "cash cash equivalents and short term investments", "cash and marketable securities"),
    "cash_and_equivalents": ("cash", "cash and cash equivalents", "cash equivalents"),
    "accounts_receivable": ("receivables", "trade receivables", "accounts receivable net", "trade and other receivables", "net receivables"),
    "inventory": ("inventories", "inventory net"),
    "current_assets": ("total current assets",),
    "total_assets": ("assets", "total assets"),
    "ppe": ("property plant and equipment", "ppe net", "net ppe", "property plant equipment", "fixed assets", "property and equipment net"),
    "goodwill": ("goodwill net",),
    "intangible_assets": ("intangibles", "other intangible assets", "intangible assets net"),
    "accounts_payable": ("payables", "trade payables", "accounts payable and accrued", "trade and other payables"),
    "current_liabilities": ("total current liabilities",),
    "short_term_debt": ("short term borrowings", "current portion of long term debt", "current debt", "short term debt and current portion"),
    "long_term_debt": ("long term borrowings", "non current debt", "long term debt net", "noncurrent debt"),
    "total_debt": ("debt", "total borrowings", "total interest bearing debt", "gross debt"),
    "total_liabilities": ("liabilities", "total liabilities"),
    "shareholders_equity": ("stockholders equity", "total equity", "equity", "shareholders equity", "total shareholders equity", "total stockholders equity", "book value of equity"),
    "minority_interest": ("non controlling interest", "noncontrolling interest", "noncontrolling interests", "non controlling interests"),
    "preferred_equity": ("preferred stock", "preferred shares"),
    "retained_earnings": ("retained earnings accumulated deficit", "accumulated deficit", "retained profits"),
    "operating_cash_flow": ("cash from operations", "net cash from operating activities", "net cash provided by operating activities", "cash flow from operations", "cfo",
                            "cash flows from operating activities", "operating cash flows"),
    "capital_expenditure": ("capex", "capital expenditures", "purchase of property plant and equipment", "purchases of property and equipment", "payments for property plant and equipment",
                            "additions to property plant and equipment"),
    "free_cash_flow": ("fcf",),
    "depreciation_amortization": ("depreciation and amortization", "depreciation and amortisation", "d&a", "da", "depreciation", "depreciation amortization and depletion"),
    "stock_based_compensation": ("share based compensation", "stock based comp", "sbc", "share based payments", "stock compensation"),
    "acquisitions": ("acquisition of businesses", "purchase of businesses", "business acquisitions", "acquisitions net of cash acquired"),
    "share_repurchases": ("buybacks", "share buybacks", "repurchase of common stock", "repurchases of common stock", "stock repurchases", "purchase of treasury stock"),
    "dividends_paid": ("dividends", "cash dividends paid", "dividends paid to shareholders", "common dividends paid"),
    "investing_cash_flow": ("cash from investing activities", "net cash used in investing activities", "net cash from investing activities", "cash flows from investing activities"),
    "financing_cash_flow": ("cash from financing activities", "net cash used in financing activities", "net cash from financing activities", "cash flows from financing activities"),
    "shares_outstanding": ("shares", "common shares outstanding", "shares outstanding basic", "basic shares outstanding", "number of shares"),
    "weighted_avg_shares_diluted": ("diluted shares", "diluted weighted average shares", "weighted average diluted shares", "weighted average shares diluted", "diluted shares outstanding",
                                    "weighted average number of diluted shares"),
}
PRICE_NAMES = ("price", "share price", "stock price", "price per share", "closing price", "close", "price at filing", "share price at filing")
META_NAMES = {
    "filing_date": ("filing date", "date filed", "filed", "filed on", "published", "publication date", "report date", "date of filing", "date published", "reported", "announcement date", "release date"),
    "period_end": ("period end", "period ending", "period end date", "fiscal year end", "year end", "end date", "balance sheet date", "fiscal period end", "date"),
    "currency": ("currency", "reporting currency", "ccy"),
    "fiscal_year": ("fiscal year", "fy", "year"),
    "fiscal_period": ("fiscal period", "period", "period type"),
}
MONTHS = {name: i for i, name in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
QUARTER_END = {"Q1": (3, 31), "Q2": (6, 30), "Q3": (9, 30), "Q4": (12, 31), "H1": (6, 30), "H2": (12, 31), "FY": (12, 31)}


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9&]+", " ", str(text).lower().replace("&", " & ")).replace(" & ", "&").strip()


def _alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for code, m in canonical.METRICS.items():
        for name in (code.replace("_", " "), m.label, *EXTRA_NAMES.get(code, ())):
            table.setdefault(_norm(name), code)
    return table


ALIASES = _alias_table()
META_ALIASES = {_norm(name): key for key, names in META_NAMES.items() for name in names}
PRICE_ALIASES = {_norm(name) for name in PRICE_NAMES}


def line_for(label) -> str | None:
    """The canonical line a spreadsheet row label stands for (``"Net sales"`` is ``revenue``), or ``None``."""
    return ALIASES.get(_norm(label))


# ------------------------------------------------------------------------------------------------------------------ the template
def template() -> dict:
    """What the entry grid offers: the lines by statement (with the ones most ratios need marked, and the ones that fill themselves in), the units and the fiscal periods."""
    statements = []
    for key in canonical.STATEMENTS:
        items = []
        for m in canonical.METRICS.values():
            if m.statement != key:
                continue
            unit = "shares" if m.kind == "shares" else "per_share" if m.kind == "per_share" else "currency"
            items.append({"code": m.code, "label": m.label, "kind": m.kind, "unit": unit, "text": m.text, "nonnegative": m.nonnegative, "core": m.code in CORE,
                          "auto": AUTO.get(m.code), "magnitude": m.code in canonical.MANUAL_MAGNITUDES})
        statements.append({"key": key, "title": STATEMENT_TITLES[key], "items": items})
    return {"statements": statements, "units": list(UNITS), "fiscal_periods": list(FISCAL_PERIODS), "max_periods": MAX_PERIODS,
            "rules": ["Type each amount in the unit you choose (thousands, millions…); the page stores it in the reporting currency's own units.",
                      "Costs and amounts spent (cost of revenue, R&D, SG&A, interest, depreciation, capital spending, buybacks, dividends, acquisitions) are stored as positive amounts: type them either way, (1,000) or 1,000.",
                      "Leave a line empty when you do not have it: an empty line is missing, never zero. Type 0 only when the figure really is zero.",
                      "Give the date each report was published. Without it the figures are treated as public a standard time after the period end (120 days for a year) and flagged as estimated.",
                      "Where a provider also holds the same period, the provider's figures lead and yours fill only the lines it lacks."]}


# ------------------------------------------------------------------------------------------------------------------ reading what was pasted
_MISSING = {"", "-", "--", "—", "–", "n/a", "na", "nm", "n.m.", "none", "null", "nan", "n.a."}
_PLAIN = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def parse_number(cell) -> float | None:
    """A number from a spreadsheet cell: ``1,234.5``, ``(1,234)`` (negative), ``-1234``, ``$1,234``; an empty cell, a dash or ``n/a`` is missing (``None``). Anything else raises ``ValueError``."""
    if cell is None:
        return None
    if isinstance(cell, bool):
        raise ValueError("a yes/no is not a number")
    if isinstance(cell, (int, float, np.integer, np.floating)):
        value = float(cell)
        if not math.isfinite(value):
            raise ValueError("not a finite number")
        return value
    text = str(cell).strip().replace("−", "-").replace(" ", " ")
    if text.lower() in _MISSING:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[\s$€£¥*]", "", text)
    if "," in text:
        if not _THOUSANDS.match(text):
            raise ValueError(f"{cell!r} is not a number")
        text = text.replace(",", "")
    if not _PLAIN.match(text):
        raise ValueError(f"{cell!r} is not a number")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{cell!r} is not a finite number")
    return -value if negative else value


def _year(text: str) -> int:
    n = int(text)
    return n if n >= 100 else (2000 + n if n < 70 else 1900 + n)


def _plausible(info: dict | None) -> dict | None:
    """A header that is a year is a report's year, not any four digits: 1500 in a row of revenue is not the year 1500."""
    return info if info is None or 1900 <= info["fiscal_year"] <= 2100 else None


def _end_of_month(year: int, month: int) -> str:
    return (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")


def parse_date(cell) -> str | None:
    """``YYYY-MM-DD`` from a cell: an ISO date, ``12/31/2023`` (month first unless the first number cannot be a month), ``31 Dec 2023``, ``Dec 31, 2023``. Empty is ``None``; anything else raises."""
    if cell is None:
        return None
    if isinstance(cell, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(cell).strftime("%Y-%m-%d")
    s = str(cell).strip()
    if s.lower() in _MISSING:
        return None
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?$", s)
    parts = None
    if m:
        parts = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = re.match(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$", s)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            parts = (y, a, b) if a <= 12 else (y, b, a)
        else:
            m = re.match(r"^(\d{1,2})[ \-]([A-Za-z]{3,9})\.?,?[ \-](\d{4})$", s)
            if m and m.group(2)[:3].lower() in MONTHS:
                parts = (int(m.group(3)), MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
            else:
                m = re.match(r"^([A-Za-z]{3,9})\.? (\d{1,2}),? (\d{4})$", s)
                if m and m.group(1)[:3].lower() in MONTHS:
                    parts = (int(m.group(3)), MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
    if parts is None:
        raise ValueError(f"{cell!r} is not a date (use YYYY-MM-DD)")
    try:
        return date(*parts).isoformat()
    except ValueError:
        raise ValueError(f"{cell!r} is not a real date") from None


def parse_period_label(label) -> dict | None:
    """What a column header says (see :func:`_parse_period_label`): ``None`` when it is not a period or its year cannot be a report's."""
    return _plausible(_parse_period_label(label))


def _parse_period_label(label) -> dict | None:
    """What a column header says: ``{"fiscal_period", "fiscal_year", "period_end", "guessed"}``, or ``None`` when it is not a period. ``FY2023``, ``2023``, ``2023-12-31``, ``Dec 2023``,
    ``Q1 2024``, ``1Q24``, ``H1 2024`` are understood. A year or a quarter gives no exact date, so the calendar end is assumed (``guessed``) and the page shows it for correction.
    ``TTM`` and estimates (``2025E``) raise ``ValueError`` with the reason: a trailing figure is built by the application from the quarters, and an estimate is not a report."""
    s = str(label).strip()
    if not s:
        return None
    low = re.sub(r"\s+", " ", s.lower().replace("_", " "))
    if low in ("ttm", "ltm") or low.startswith(("ttm ", "ltm ")):
        raise ValueError("a trailing-twelve-month column is built by the application from the quarters: enter the quarters")
    if re.search(r"(?<![a-z])\d{2,4}\s*(e|est|estimate|f|forecast|p)$", low) or "estimate" in low or "forecast" in low:
        raise ValueError("an estimate or a forecast is not a report")
    try:
        iso = parse_date(s)
    except ValueError:
        iso = None
    if iso:
        return {"fiscal_period": "FY", "fiscal_year": int(iso[:4]), "period_end": iso, "guessed": False}
    m = re.match(r"^(?:fy|fiscal year|year)?\s*'?(\d{4}|\d{2})\s*a?$", low)
    if m and (len(m.group(1)) == 4 or low.startswith(("fy", "fiscal", "year", "'"))):
        y = _year(m.group(1))
        return {"fiscal_period": "FY", "fiscal_year": y, "period_end": f"{y}-12-31", "guessed": True}
    for pattern, order in ((r"^(?:fy)?\s*(q[1-4]|h[12])[\s\-'/]*(?:fy)?\s*(\d{4}|\d{2})$", "pq"), (r"^([1-4])q[\s\-'/]*(\d{4}|\d{2})$", "nq"), (r"^([12])h[\s\-'/]*(\d{4}|\d{2})$", "nh"),
                           (r"^(\d{4})[\s\-/]*(q[1-4]|h[12])$", "yp")):
        m = re.match(pattern, low)
        if not m:
            continue
        if order == "pq":
            period, y = m.group(1).upper(), _year(m.group(2))
        elif order == "nq":
            period, y = f"Q{m.group(1)}", _year(m.group(2))
        elif order == "nh":
            period, y = f"H{m.group(1)}", _year(m.group(2))
        else:
            period, y = m.group(2).upper(), _year(m.group(1))
        month, day = QUARTER_END[period]
        return {"fiscal_period": period, "fiscal_year": y, "period_end": f"{y}-{month:02d}-{day:02d}", "guessed": True}
    m = re.match(r"^([a-z]{3,9})\.?[\s\-'/,]*(\d{4}|\d{2})$", low)
    if m and m.group(1)[:3] in MONTHS:
        y = _year(m.group(2))
        return {"fiscal_period": "FY", "fiscal_year": y, "period_end": _end_of_month(y, MONTHS[m.group(1)[:3]]), "guessed": False}
    return None


def _cells(text: str) -> list[list[str]]:
    if "\t" in text:
        rows = [line.split("\t") for line in text.splitlines()]
    else:
        sep = ";" if text.count(";") > text.count(",") else ","
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=sep))
        except csv.Error as error:
            raise FundamentalsError(f"the table could not be read as comma-separated text: {error}") from None
    rows = [[c.strip() for c in row] for row in rows]
    rows = [row for row in rows if any(row)]
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def _transpose(rows: list[list[str]]) -> list[list[str]]:
    return [list(r) for r in zip(*rows)]


def _hits(cells: list[str]) -> int:
    return sum(1 for c in cells if c and (line_for(c) or _norm(c) in META_ALIASES or _norm(c) in PRICE_ALIASES))


def parse_table(text) -> dict:
    """Read a table pasted from Excel (tab-separated) or a CSV file: line items down the left, one period per column (or the other way round: the layout is detected).

    Returns the periods (``columns``), the typed values by canonical line (``values``, one entry per column, ``None`` where empty), a ``price`` row if there was one, and what could not be used:
    ``unrecognised`` row labels (with the figures they carried), ``duplicates`` (a line given twice keeps the first, so the net income that opens the cash flow statement does not replace the one
    of the income statement), ``problems`` (a cell that is not a number) and ``notes``. Amounts are as typed: the unit is chosen on the page and applied when saved."""
    if not isinstance(text, str) or not text.strip():
        raise FundamentalsError("paste the table first (from Excel, or a CSV file)")
    if len(text) > MAX_TEXT:
        raise FundamentalsError(f"the table is too large to read here (more than {MAX_TEXT:,} characters)")
    rows = _cells(text)
    if len(rows) < 2 or len(rows[0]) < 2:
        raise FundamentalsError("the table needs a header row of periods and at least one row of figures, with the line names in the first column")
    notes: list[str] = []
    transposed = False
    if _hits(rows[0][1:]) > _hits([r[0] for r in rows[1:]]) and _hits(rows[0][1:]) >= 2:
        rows, transposed = _transpose(rows), True
        notes.append("The periods ran down the rows, so the table was turned the other way round.")
    if len(rows) > MAX_ROWS:
        raise FundamentalsError(f"the table has {len(rows)} rows; at most {MAX_ROWS} are read")
    head, first = None, None
    for i, row in enumerate(rows[:12]):
        labels = []
        for c in row[1:]:
            try:
                labels.append(bool(c) and parse_period_label(c) is not None)
            except ValueError:
                labels.append(True)
        if any(labels) and sum(labels) >= max(1, math.ceil(len([c for c in row[1:] if c]) / 2)):
            head, first = i, row
            break
    if head is None:
        raise FundamentalsError("no header row of periods was found: put the periods (FY2023, 2023-12-31, Q1 2024…) across the top, one per column")
    columns: list[dict] = []
    keep: list[int] = []
    for j, c in enumerate(first):
        if j == 0 or not c:
            continue
        try:
            info = parse_period_label(c)
        except ValueError as error:
            notes.append(f"Column “{c}” was left out: {error}.")
            continue
        if info is None:
            notes.append(f"Column “{c}” was left out: it is not a period.")
            continue
        columns.append({"label": c, **info, "filing_date": None, "currency": None})
        keep.append(j)
    if not columns:
        raise FundamentalsError("no column of the header row is a period: use FY2023, 2023-12-31, Q1 2024…")
    values: dict[str, list] = {}
    price: list | None = None
    unrecognised: list[dict] = []
    duplicates: list[str] = []
    problems: list[dict] = []
    for row in rows[head + 1:]:
        label = re.sub(r"[:*]+$", "", row[0]).strip()
        cells = [row[j] for j in keep]
        if not label or not any(cells):
            continue
        key = _norm(label)
        meta = META_ALIASES.get(key)
        if meta:
            for col, cell in zip(columns, cells):
                if not cell:
                    continue
                try:
                    if meta in ("filing_date", "period_end"):
                        got = parse_date(cell)
                        if got:
                            col[meta], col["guessed"] = got, (col["guessed"] if meta != "period_end" else False)
                            if meta == "period_end":
                                col["fiscal_year"] = col["fiscal_year"] if col.get("fiscal_year") else int(got[:4])
                    elif meta == "currency":
                        if not CURRENCY.match(cell):
                            raise ValueError(f"{cell!r} is not a three-letter currency code")
                        col["currency"] = cell.upper()
                    elif meta == "fiscal_year":
                        col["fiscal_year"] = int(parse_number(cell))
                    elif meta == "fiscal_period":
                        if cell.upper() not in FISCAL_PERIODS:
                            raise ValueError(f"{cell!r} is not one of {', '.join(FISCAL_PERIODS)}")
                        col["fiscal_period"] = cell.upper()
                except (ValueError, TypeError) as error:
                    problems.append({"label": label, "column": col["label"], "cell": cell, "message": str(error)})
            continue
        code = ALIASES.get(key)
        is_price = key in PRICE_ALIASES
        if code is None and not is_price:
            unrecognised.append({"label": label, "figures": sum(1 for c in cells if c)})
            continue
        typed = []
        for col, cell in zip(columns, cells):
            try:
                typed.append(parse_number(cell))
            except ValueError as error:
                typed.append(None)
                problems.append({"label": label, "column": col["label"], "cell": cell, "message": str(error)})
        if is_price:
            price = typed
        elif code in values:
            duplicates.append(label)
        else:
            values[code] = typed
    if not values:
        raise FundamentalsError("no row of the table is a line this application knows (Revenue, Net income, Total assets…): check that the line names are in the first column")
    return {"columns": columns, "values": values, "price": price, "unrecognised": unrecognised, "duplicates": duplicates, "problems": problems, "notes": notes, "transposed": transposed,
            "rows_read": len(rows) - head - 1}


# ------------------------------------------------------------------------------------------------------------------ checking an entry
def _today() -> date:
    return datetime.now(timezone.utc).date()


def _issue(level: str, message: str, period: str | None = None, field: str | None = None, column: int | None = None) -> dict:
    return {"level": level, "message": message, "period": period, "field": field, "column": column}


def _iso(value, what: str, issues: list, label: str | None, column: int | None) -> str | None:
    try:
        return parse_date(value)
    except ValueError as error:
        issues.append(_issue("error", f"{what}: {error}", label, None, column))
        return None


def clean_company(raw) -> dict:
    """The identity of a company to create: a ticker and, if known, a name, exchange, country, currency, sector, industry and the symbol Yahoo knows it by (for prices)."""
    if not isinstance(raw, dict):
        raise FundamentalsError("company must be a dictionary with at least a ticker")
    out: dict = {}
    ticker = str(raw.get("ticker") or "").strip().upper()
    if not TICKER.match(ticker):
        raise FundamentalsError("a ticker is letters, digits and . : - ^ = only, up to 24 characters (for example NVDA or 7203.T)")
    out["ticker"] = ticker
    for key, limit in (("name", 120), ("sector", 80), ("industry", 80)):
        value = str(raw.get(key) or "").strip()
        if len(value) > limit:
            raise FundamentalsError(f"the company's {key} is longer than {limit} characters")
        out[key] = value or None
    exchange = str(raw.get("exchange") or "").strip().upper()
    if exchange and not re.match(r"^[A-Z0-9.\-]{1,12}$", exchange):
        raise FundamentalsError("an exchange code is letters and digits, up to 12 characters (for example XNAS)")
    out["exchange"] = exchange
    country = str(raw.get("country") or "").strip().upper()
    if country and not re.match(r"^[A-Z]{2}$", country):
        raise FundamentalsError("country is a two-letter ISO code such as US, JP or GB")
    out["country"] = country or None
    currency = str(raw.get("currency") or "").strip().upper()
    if currency and not CURRENCY.match(currency):
        raise FundamentalsError("currency is a three-letter code such as USD, EUR or JPY")
    out["currency"] = currency or None
    symbol = str(raw.get("price_symbol") or "").strip().upper()
    if symbol and (len(symbol) > 24 or not all(c.isalnum() or c in ".-^=" for c in symbol)):
        raise FundamentalsError("a price symbol is letters, digits and . - ^ = only, up to 24 characters")
    out["price_symbol"] = symbol or None
    return out


def check_entry(raw, security: dict | None = None, today: date | None = None) -> dict:
    """Validate an entry and say what is wrong or doubtful, without raising for the content of a cell.

    ``raw``: ``{"units": {"money", "shares"}, "currency"?, "periods": [{"fiscal_period", "period_end", "fiscal_year"?, "filing_date"?, "currency"?, "price"?, "values": {line: number}}]}``.
    Returns ``{"periods": [...], "issues": [...], "units": {...}, "ok": bool}`` where each issue has a ``level`` (``error`` stops a save; ``warning`` and ``info`` do not), a ``message``, and the
    ``period`` and ``field`` it is about. A request that is not shaped like an entry raises :class:`FundamentalsError`."""
    if not isinstance(raw, dict):
        raise FundamentalsError("the entry must be a dictionary with a list of periods")
    periods_in = raw.get("periods")
    if not isinstance(periods_in, list) or not periods_in:
        raise FundamentalsError("add at least one period (a column) with figures")
    if len(periods_in) > MAX_PERIODS:
        raise FundamentalsError(f"at most {MAX_PERIODS} periods can be entered at a time")
    units_in = raw.get("units") or {}
    if not isinstance(units_in, dict):
        raise FundamentalsError("units must say the unit of the amounts and of the share counts")
    units = {"money": str(units_in.get("money") or "units"), "shares": str(units_in.get("shares") or "units")}
    for key, value in units.items():
        if value not in UNITS:
            raise FundamentalsError(f"the unit of the {key} must be one of {', '.join(UNITS)}")
    money, shares_unit = UNITS[units["money"]], UNITS[units["shares"]]
    day = today or _today()
    default_currency = str(raw.get("currency") or (security or {}).get("currency") or "").strip().upper() or None
    issues: list[dict] = []
    out: list[dict] = []
    seen: dict[tuple, int] = {}
    for column, item in enumerate(periods_in):
        if not isinstance(item, dict):
            raise FundamentalsError("each period must be a dictionary")
        fiscal_period = str(item.get("fiscal_period") or "FY").strip().upper()
        end_raw = item.get("period_end")
        label = f"{fiscal_period} to {end_raw}" if end_raw else f"column {column + 1}"
        if fiscal_period not in FISCAL_PERIODS:
            issues.append(_issue("error", f"the period type must be one of {', '.join(FISCAL_PERIODS)}", label, None, column))
            continue
        period_end = _iso(end_raw, "the period end", issues, label, column)
        if period_end is None:
            if not any(i["column"] == column and i["level"] == "error" for i in issues):
                issues.append(_issue("error", "the period end date is missing", label, None, column))
            continue
        if period_end > day.isoformat():
            issues.append(_issue("error", f"the period end {period_end} is in the future: a report can only be entered once the period has ended", label, None, column))
            continue
        if period_end < "1900-01-01":
            issues.append(_issue("error", "the period end is before 1900", label, None, column))
            continue
        label = f"{fiscal_period} to {period_end}"
        key = (fiscal_period, period_end)
        if key in seen:
            issues.append(_issue("error", f"the {label} period appears twice (columns {seen[key] + 1} and {column + 1})", label, None, column))
            continue
        seen[key] = column
        try:
            fiscal_year = int(item["fiscal_year"]) if item.get("fiscal_year") not in (None, "") else int(period_end[:4])
        except (TypeError, ValueError):
            issues.append(_issue("error", "the fiscal year must be a whole number", label, None, column))
            continue
        if abs(fiscal_year - int(period_end[:4])) > 1:
            issues.append(_issue("error", f"fiscal year {fiscal_year} does not fit a period ending {period_end} (a fiscal year ends in that year or the next)", label, None, column))
            continue
        filing = _iso(item.get("filing_date"), "the publication date", issues, label, column)
        if filing:
            if filing < period_end:
                issues.append(_issue("error", f"the publication date {filing} is before the period end {period_end}: a report cannot be public before the period it covers has ended", label, None, column))
                continue
            if filing > day.isoformat():
                issues.append(_issue("error", f"the publication date {filing} is in the future", label, None, column))
                continue
            if (pd.Timestamp(filing) - pd.Timestamp(period_end)).days > 400:
                issues.append(_issue("warning", f"the report is dated {(pd.Timestamp(filing) - pd.Timestamp(period_end)).days} days after the period it covers: check the date", label, None, column))
        else:
            lag = PitSettings().lag_for(fiscal_period)
            issues.append(_issue("warning", f"no publication date: studies will treat these figures as public {lag} days after the period end and flag them as estimated. "
                                            "Enter the date the report was published to avoid that.", label, None, column))
        currency = str(item.get("currency") or default_currency or "").strip().upper()
        if not CURRENCY.match(currency):
            issues.append(_issue("error", "the reporting currency is missing (a three-letter code such as USD)", label, None, column))
            continue
        price = None
        if item.get("price") not in (None, ""):
            try:
                price = parse_number(item["price"])
            except ValueError as error:
                issues.append(_issue("error", f"the share price: {error}", label, "price", column))
                continue
            if price is not None and price <= 0:
                issues.append(_issue("error", "the share price must be positive", label, "price", column))
                continue
        typed_in = item.get("values") or {}
        if not isinstance(typed_in, dict):
            raise FundamentalsError("values must be a dictionary of line name to number")
        typed: dict[str, float] = {}
        scaled: dict[str, float] = {}
        bad = False
        for code, cell in typed_in.items():
            if code not in canonical.METRICS:
                issues.append(_issue("error", f"{code!r} is not a line this application stores", label, str(code), column))
                bad = True
                continue
            try:
                number = parse_number(cell)
            except ValueError as error:
                issues.append(_issue("error", f"{canonical.METRICS[code].label}: {error}", label, code, column))
                bad = True
                continue
            if number is None:
                continue
            kind = canonical.METRICS[code].kind
            typed[code] = number
            scaled[code] = number * (shares_unit if kind == "shares" else 1.0 if kind == "per_share" else money)
        if bad:
            continue
        if not scaled:
            issues.append(_issue("error", "this period has no figures", label, None, column))
            continue
        observations, _ = canonical.normalise_values(PROVIDER, scaled)
        for code, o in observations.items():
            if canonical.METRICS[code].nonnegative and o["value"] < 0:
                issues.append(_issue("error", f"{canonical.METRICS[code].label} cannot be negative", label, code, column))
                bad = True
        if bad:
            continue
        out.append({"column": column, "label": label, "fiscal_period": fiscal_period, "fiscal_year": fiscal_year, "period_end": period_end, "filing_date": filing, "currency": currency,
                    "price": price, "typed": typed, "values": scaled, "observations": observations, "derived": sorted(c for c, o in observations.items() if o["flag"] == "derived")})
        issues.extend(_checks(out[-1], money))
    years = sorted(p["period_end"] for p in out if p["fiscal_period"] == "FY")
    for a, b in zip(years, years[1:]):
        gap = (pd.Timestamp(b) - pd.Timestamp(a)).days
        if gap > 400 or gap < 330:
            issues.append(_issue("info", f"the annual periods ending {a} and {b} are {gap} days apart: a year-on-year change is measured against a year earlier, so a gap leaves growth missing", None))
    return {"periods": out, "issues": issues, "units": units, "ok": not any(i["level"] == "error" for i in issues) and bool(out)}


def _near(a: float, b: float, slack: float, tolerance: float = 0.01) -> bool:
    return abs(a - b) <= max(tolerance * max(abs(a), abs(b)), slack)


def _checks(p: dict, money: float) -> list[dict]:
    """The identities a set of statements has to satisfy, as warnings: a figure that breaks one is a typing mistake or a unit mix-up more often than a fact about the company."""
    label, col = p["label"], p["column"]
    obs = {c: o["value"] for c, o in p["observations"].items()}
    entered = set(p["values"])                                                    # a figure the person typed, not one filled in from others
    slack = 2.0 * money
    out: list[dict] = []

    def say(message: str, *fields: str) -> None:
        out.append(_issue("warning", message, label, fields[0] if fields else None, col))

    if all(c in obs for c in ("total_assets", "total_liabilities", "shareholders_equity")):
        claims = obs["total_liabilities"] + obs["shareholders_equity"] + obs.get("minority_interest", 0.0) + obs.get("temporary_equity", 0.0)
        if not _near(obs["total_assets"], claims, slack):
            say(f"total assets ({obs['total_assets']:,.0f}) differ from total liabilities plus equity ({claims:,.0f}) by {obs['total_assets'] - claims:,.0f}. "
                "Check the three figures, and add minority interest or redeemable equity if the company has them.", "total_assets")
    if {"revenue", "cost_of_revenue", "gross_profit"} <= entered and not _near(obs["revenue"] - obs["cost_of_revenue"], obs["gross_profit"], slack):
        say(f"revenue less cost of revenue is {obs['revenue'] - obs['cost_of_revenue']:,.0f} but gross profit is {obs['gross_profit']:,.0f}.", "gross_profit")
    if {"operating_cash_flow", "capital_expenditure", "free_cash_flow"} <= entered and not _near(obs["operating_cash_flow"] - obs["capital_expenditure"], obs["free_cash_flow"], slack):
        say(f"operating cash flow less capital expenditure is {obs['operating_cash_flow'] - obs['capital_expenditure']:,.0f} but free cash flow is {obs['free_cash_flow']:,.0f}.", "free_cash_flow")
    if {"short_term_debt", "long_term_debt", "total_debt"} <= entered and obs["total_debt"] + slack < obs["short_term_debt"] + obs["long_term_debt"] - slack:
        say("total debt is less than short-term plus long-term debt.", "total_debt")
    if "current_assets" in obs and "total_assets" in obs and obs["current_assets"] > obs["total_assets"] + slack:
        say("current assets exceed total assets.", "current_assets")
    if "current_liabilities" in obs and "total_liabilities" in obs and obs["current_liabilities"] > obs["total_liabilities"] + slack:
        say("current liabilities exceed total liabilities.", "current_liabilities")
    if "cash_and_equivalents" in obs and "current_assets" in obs and obs["cash_and_equivalents"] > obs["current_assets"] + slack:
        say("cash exceeds current assets.", "cash_and_equivalents")
    eps, shares, income = obs.get("eps_diluted", obs.get("eps_basic")), obs.get("weighted_avg_shares_diluted", obs.get("shares_outstanding")), obs.get("net_income")
    if eps and shares and income and eps * income > 0 and not 0.4 <= (eps * shares) / income <= 2.5:
        say(f"earnings per share times the share count is {eps * shares:,.0f} but net income is {income:,.0f}: the units of the amounts, the share count and the per-share figure may not match "
            "(a share count typed in millions while amounts are in thousands, for example).", "weighted_avg_shares_diluted")
    if obs.get("revenue") and obs.get("total_assets") and obs["revenue"] > 0 and obs["total_assets"] > 0 and not 0.001 <= obs["revenue"] / obs["total_assets"] <= 30:
        say(f"revenue is {obs['revenue'] / obs['total_assets']:,.3f} times total assets, which is unusual: check that both are in the same unit.", "revenue")
    for code in ("revenue", "total_assets"):
        if code in obs and obs[code] == 0:
            say(f"{canonical.METRICS[code].label} is zero, so the ratios that divide by it will be missing.", code)
    missing = [canonical.METRICS[c].label for c in sorted(CORE) if c not in obs and c not in AUTO]
    if missing:
        out.append(_issue("info", f"{len(CORE) - len(missing)} of the {len(CORE)} lines most ratios use are given; not given: {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''}. "
                                  "A ratio is computed only when every figure it uses is present.", label, None, col))
    return out


# ------------------------------------------------------------------------------------------------------------------ the company
def _company_row(store: FundamentalsStore, security_id) -> dict:
    try:
        sid = int(security_id)
    except (TypeError, ValueError):
        raise FundamentalsError("security_id must be a whole number") from None
    row = store.get_security(sid)
    if row is None:
        raise FundamentalsError(f"no company with id {sid}")
    return row


def _resolve(store: FundamentalsStore, raw: dict) -> tuple[dict | None, dict | None]:
    """``(existing company row, new company)``: exactly one is set. A ticker that is already in the database is refused, so a company is never entered twice."""
    if raw.get("security_id") not in (None, ""):
        return _company_row(store, raw["security_id"]), None
    company = clean_company(raw.get("company"))
    with store.db.transaction() as conn:
        row = conn.execute("SELECT security_id, company_name FROM securities WHERE ticker = ? AND exchange = ?", (company["ticker"], company["exchange"])).fetchone()
    if row is not None:
        raise FundamentalsError(f"{company['ticker']} is already in your database (company {row['security_id']}{', ' + row['company_name'] if row['company_name'] else ''}): "
                                "choose it from the list instead of creating it again")
    return None, company


def _profile(company: dict) -> CompanyProfile:
    return CompanyProfile(provider=PROVIDER, provider_symbol=company["ticker"], ticker=company["ticker"], name=company["name"] or company["ticker"], exchange=company["exchange"],
                          country=company["country"], currency=company["currency"], sector=company["sector"], industry=company["industry"])


def _statements(periods: list[dict]) -> list[PeriodStatement]:
    return [PeriodStatement(fiscal_period=p["fiscal_period"], period_end=p["period_end"], values=dict(p["values"]), fiscal_year=p["fiscal_year"], filing_date=p["filing_date"],
                            reporting_currency=p["currency"], provider_record_id=PROVIDER) for p in periods]


# ------------------------------------------------------------------------------------------------------------------ saving
def overlaps(store: FundamentalsStore, security_id: int, periods: list[dict]) -> list[dict]:
    """The periods a provider also holds, and how its figures compare with these.

    The provider's figures lead. Lines only this entry has fill the gaps only if they became public no later than the provider's report (a figure that was published later cannot be used as if it had
    been known then): ``used_in_studies`` says whether that holds, from the two availability dates the engine itself works out. ``differ`` lists the lines both have that disagree by more than 0.5%."""
    mine = {(p["fiscal_period"], p["period_end"]): p for p in periods}
    frame = store.periods([int(security_id)])
    frame = frame[frame["vintage"] == 0].copy()
    if not (frame["provider"] != PROVIDER).any():
        return []
    frame["available"], _ = availability(frame, PitSettings())
    obs = store.observations([int(security_id)])
    out = []
    for r in frame[frame["provider"] != PROVIDER].itertuples():
        key = (r.fiscal_period, r.period_end.strftime("%Y-%m-%d"))
        p = mine.get(key)
        if p is None:
            continue
        own = frame[(frame["provider"] == PROVIDER) & (frame["fiscal_period"] == r.fiscal_period) & (frame["period_end"] == r.period_end)]
        theirs = obs[obs["period_id"] == r.period_id].set_index("metric_code")["value"]
        compared, differ = 0, []
        for code, o in p["observations"].items():
            if code in theirs.index:
                compared += 1
                a, b = o["value"], float(theirs[code])
                if abs(a - b) > 0.005 * max(abs(a), abs(b), 1e-12):
                    differ.append({"line": code, "label": canonical.METRICS[code].label, "mine": a, "theirs": b})
        extra = sum(1 for code in p["observations"] if code not in theirs.index)
        used = None if own.empty else bool(own["available"].iloc[0] <= r.available)
        out.append({"label": p["label"], "provider": r.provider, "compared": compared, "differ": differ[:12], "differ_count": len(differ), "only_mine": extra, "used_in_studies": used,
                    "mine_public": None if own.empty else own["available"].iloc[0].strftime("%Y-%m-%d"), "theirs_public": r.available.strftime("%Y-%m-%d")})
    return out


def refresh_metrics(store: FundamentalsStore, security_id: int) -> int:
    """Write the derived metrics of a company to ``fundamental_metrics`` again (the old rows of the company are removed first, so a figure that was corrected or cleared leaves no stale ratio)."""
    with store.db.transaction() as conn:
        conn.execute("DELETE FROM fundamental_metrics WHERE security_id = ?", (int(security_id),))
    return metrics.persist(store, PeriodTables(store, [int(security_id)], PitSettings()), ("fy", "ttm"))


def save(store: FundamentalsStore, raw, mode: str = "correct", restated_on=None, today: date | None = None) -> dict:
    with _WRITING:
        return _save(store, raw, mode, restated_on, today)


def _save(store: FundamentalsStore, raw, mode: str, restated_on, today: date | None) -> dict:
    """Store an entry. ``mode`` ``correct`` (the default) writes the periods as they are, overwriting a period entered before (a typing mistake corrected); ``restate`` records a change to a
    period entered before as a new vintage, public only from ``restated_on``. Every check of :func:`check_entry` must pass. Returns what was done per period and a summary."""
    if mode not in MODES:
        raise FundamentalsError(f"mode must be one of {', '.join(MODES)}")
    if not isinstance(raw, dict):
        raise FundamentalsError("the entry must be a dictionary")
    existing, company = _resolve(store, raw)
    day = today or _today()
    checked = check_entry(raw, existing or company, day)
    errors = [i for i in checked["issues"] if i["level"] == "error"]
    if errors:
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        raise FundamentalsError(f"nothing was saved: {errors[0]['period'] + ': ' if errors[0]['period'] else ''}{errors[0]['message']}{more}")
    periods = checked["periods"]
    restated = None
    if mode == "restate":
        try:
            restated = parse_date(restated_on)
        except ValueError as error:
            raise FundamentalsError(f"the date of the restatement: {error}") from None
        if restated is None:
            raise FundamentalsError("a restatement needs the date it was published: it is public only from that day")
        if restated > day.isoformat():
            raise FundamentalsError("the date of the restatement is in the future")
        late = [p["label"] for p in periods if restated < p["period_end"]]
        if late:
            raise FundamentalsError(f"the restatement is dated before the period it restates ({late[0]})")
    created = False
    if existing is None:
        security_id = store.upsert_security(_profile(company))
        created = True
        if company["price_symbol"]:
            prices.set_price_symbol(store, security_id, company["price_symbol"])
    else:
        security_id = int(existing["security_id"])
    counts = {"added": 0, "corrected": 0, "restated": 0, "unchanged": 0}
    outcomes = []
    if mode == "restate":
        def newest() -> dict:
            out: dict = {}
            for r in store.held_periods(security_id, PROVIDER):
                out[(r["fiscal_period"], r["period_end"])] = max(r["vintage"], out.get((r["fiscal_period"], r["period_end"]), -1))
            return out

        before = newest()
        result = store.store_financials(security_id, PROVIDER, _statements(periods), retrieved=restated)
        counts["added"], counts["restated"], counts["unchanged"] = result["periods"], result["vintages"], result["unchanged"]
        after = newest()
        for p in periods:
            key = (p["fiscal_period"], p["period_end"])
            outcomes.append({"label": p["label"], "outcome": "added" if key not in before else "restated" if after.get(key, 0) > before[key] else "unchanged"})
    else:
        with store.db.transaction() as conn:
            for p, statement in zip(periods, _statements(periods)):
                row = conn.execute("SELECT id, vintage FROM financial_periods WHERE security_id = ? AND provider = ? AND fiscal_period = ? AND period_end = ? ORDER BY vintage DESC LIMIT 1",
                                   (security_id, PROVIDER, p["fiscal_period"], p["period_end"])).fetchone()
                dates = (p["filing_date"], None, None)
                if row is None:
                    period_id = FundamentalsStore._insert_period(conn, security_id, PROVIDER, statement, p["fiscal_year"], p["period_end"], dates, 0, "filed" if p["filing_date"] else "estimated")
                    FundamentalsStore._insert_observations(conn, period_id, PROVIDER, p["observations"], p["currency"], restated=False)
                    counts["added"] += 1
                    outcomes.append({"label": p["label"], "outcome": "added"})
                    continue
                period_id, vintage = int(row[0]), int(row[1])
                old = {r[0]: float(r[1]) for r in conn.execute("SELECT metric_code, value FROM fundamental_observations WHERE financial_period_id = ?", (period_id,))}
                new = {c: float(o["value"]) for c, o in p["observations"].items()}
                same = old.keys() == new.keys() and all(_near(old[c], new[c], 0.0, 1e-9) for c in new)
                conn.execute("DELETE FROM fundamental_observations WHERE financial_period_id = ?", (period_id,))
                FundamentalsStore._insert_observations(conn, period_id, PROVIDER, p["observations"], p["currency"], restated=vintage > 0)
                quality = "restated" if vintage > 0 else ("filed" if p["filing_date"] else "estimated")
                if vintage > 0:
                    conn.execute("UPDATE financial_periods SET fiscal_year = ?, reporting_currency = ?, pit_quality = ?, updated_at = ? WHERE id = ?",
                                 (p["fiscal_year"], p["currency"], quality, datetime.now(timezone.utc).isoformat(timespec="seconds"), period_id))
                else:
                    conn.execute("UPDATE financial_periods SET fiscal_year = ?, filing_date = ?, accepted_date = NULL, publication_date = NULL, reporting_currency = ?, pit_quality = ?, "
                                 "updated_at = ? WHERE id = ?", (p["fiscal_year"], p["filing_date"], p["currency"], quality, datetime.now(timezone.utc).isoformat(timespec="seconds"), period_id))
                counts["unchanged" if same else "corrected"] += 1
                outcomes.append({"label": p["label"], "outcome": "unchanged" if same else "corrected"})
    audit = {"units": checked["units"], "mode": mode, "restated_on": restated, "periods": [{"fiscal_period": p["fiscal_period"], "period_end": p["period_end"], "fiscal_year": p["fiscal_year"],
                                                                                         "filing_date": p["filing_date"], "currency": p["currency"], "typed": p["typed"]} for p in periods]}
    store.save_raw(PROVIDER, security_id, "manual_entry", {"mode": mode, "saved": day.isoformat()}, audit)
    written = refresh_metrics(store, security_id)
    info = store.get_security(security_id) or {}
    return clean({"security_id": security_id, "ticker": info.get("ticker"), "name": info.get("company_name"), "created_company": created, "mode": mode, **counts, "periods": outcomes,
                  "observations": sum(len(p["observations"]) for p in periods), "metrics_written": written, "overlaps": overlaps(store, security_id, periods),
                  "issues": [i for i in checked["issues"] if i["level"] != "error"]})


def entries(store: FundamentalsStore, security_id) -> dict:
    """The periods of a company that were entered by hand, as they are stored (the latest vintage of each), so they can be put back in the grid and corrected."""
    info = _company_row(store, security_id)
    sid = int(info["security_id"])
    out = []
    with store.db.transaction() as conn:
        rows = conn.execute("SELECT p.id, p.fiscal_year, p.fiscal_period, p.period_end, p.filing_date, p.publication_date, p.reporting_currency, p.vintage, p.pit_quality, "
                            "(SELECT MAX(q.vintage) FROM financial_periods q WHERE q.security_id = p.security_id AND q.provider = p.provider AND q.fiscal_period = p.fiscal_period "
                            "AND q.period_end = p.period_end) AS latest FROM financial_periods p WHERE p.security_id = ? AND p.provider = ? ORDER BY p.period_end, p.fiscal_period, p.vintage",
                            (sid, PROVIDER)).fetchall()
        for r in rows:
            if r["vintage"] != r["latest"]:
                continue
            values = {o["metric_code"]: float(o["value"]) for o in conn.execute("SELECT metric_code, value FROM fundamental_observations WHERE financial_period_id = ?", (r["id"],))}
            out.append({"fiscal_year": r["fiscal_year"], "fiscal_period": r["fiscal_period"], "period_end": r["period_end"], "filing_date": r["filing_date"], "published": r["publication_date"],
                        "currency": r["reporting_currency"], "vintage": r["vintage"], "quality": r["pit_quality"], "values": values})
    big = [abs(v) for p in out for c, v in p["values"].items() if canonical.METRICS[c].kind in ("flow", "stock") and v]
    median = float(np.median(big)) if big else 0.0
    suggested = "billions" if median >= 5e10 else "millions" if median >= 5e6 else "thousands" if median >= 5e3 else "units"
    return clean({"security_id": sid, "ticker": info.get("ticker"), "name": info.get("company_name"), "currency": info.get("currency"), "source": info.get("source"),
                  "periods": out, "suggested_unit": suggested})


def delete_period(store: FundamentalsStore, security_id, fiscal_period: str, period_end: str) -> dict:
    with _WRITING:
        return _delete_period(store, security_id, fiscal_period, period_end)


def _delete_period(store: FundamentalsStore, security_id, fiscal_period: str, period_end: str) -> dict:
    """Remove a period entered by hand (every vintage of it) and the ratios derived from it. Figures a provider holds for the same period are not touched."""
    info = _company_row(store, security_id)
    sid = int(info["security_id"])
    fiscal_period = str(fiscal_period or "").upper()
    if fiscal_period not in FISCAL_PERIODS:
        raise FundamentalsError(f"the period type must be one of {', '.join(FISCAL_PERIODS)}")
    try:
        end = parse_date(period_end)
    except ValueError as error:
        raise FundamentalsError(f"the period end: {error}") from None
    if end is None:
        raise FundamentalsError("the period end is needed")
    with store.db.transaction() as conn:
        found = conn.execute("SELECT COUNT(*) FROM financial_periods WHERE security_id = ? AND provider = ? AND fiscal_period = ? AND period_end = ?", (sid, PROVIDER, fiscal_period, end)).fetchone()[0]
        if not found:
            raise FundamentalsError(f"no period {fiscal_period} to {end} was entered by hand for this company")
        conn.execute("DELETE FROM financial_periods WHERE security_id = ? AND provider = ? AND fiscal_period = ? AND period_end = ?", (sid, PROVIDER, fiscal_period, end))
    written = refresh_metrics(store, sid)
    return {"security_id": sid, "deleted": int(found), "fiscal_period": fiscal_period, "period_end": end, "metrics_written": written}


# ------------------------------------------------------------------------------------------------------------------ the preview
def _copy_rows(src, dst, table: str, where: str, args: list, override: dict | None = None, skip_ids: set | None = None, id_col: str | None = None) -> None:
    cur = src.execute(f"SELECT * FROM {table} WHERE {where}", args)
    names = [d[0] for d in cur.description]
    for row in cur.fetchall():
        record = dict(zip(names, row))
        if skip_ids is not None and id_col and record[id_col] in skip_ids:
            continue
        record.update(override or {})
        dst.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(record)}) VALUES ({', '.join('?' * len(record))})", list(record.values()))


def _stage(store: FundamentalsStore, target: FundamentalsStore, existing: dict | None, company: dict | None, periods: list[dict]) -> int:
    """Put one company in a temporary database as the real one would hold it once the entry is saved: its other figures and prices as they are, the periods typed here in place of any typed before."""
    if existing is None:
        sid = target.upsert_security(_profile(company))
    else:
        sid = int(existing["security_id"])
        mine = {(p["fiscal_period"], p["period_end"]) for p in periods}
        with store.db.transaction() as src, target.db.transaction() as dst:
            _copy_rows(src, dst, "securities", "security_id = ?", [sid], {"market_id": None})
            drop = {r["id"] for r in src.execute("SELECT id, fiscal_period, period_end FROM financial_periods WHERE security_id = ? AND provider = ?", (sid, PROVIDER)).fetchall()
                    if (r["fiscal_period"], r["period_end"]) in mine}
            _copy_rows(src, dst, "financial_periods", "security_id = ?", [sid], skip_ids=drop, id_col="id")
            _copy_rows(src, dst, "fundamental_observations", "financial_period_id IN (SELECT id FROM financial_periods WHERE security_id = ?)", [sid], skip_ids=drop, id_col="financial_period_id")
            _copy_rows(src, dst, "security_identifiers", "security_id = ?", [sid])
            _copy_rows(src, dst, "prices_daily", "security_id = ?", [sid])
    target.store_financials(sid, PROVIDER, _statements(periods))
    return sid


def multiples(values: dict, price: float) -> dict:
    """What the share price the person typed implies for one annual period, in the forms the factor library uses (yields over market capitalisation or enterprise value, missing where the
    denominator is not positive) and as the familiar multiples. Not stored."""
    shares = values.get("shares_outstanding") or values.get("weighted_avg_shares_diluted")
    if not shares or shares <= 0:
        return {"error": "the number of shares is needed to turn a price into a market capitalisation"}
    cap = price * shares
    debt, cash = values.get("total_debt"), values.get("cash_and_equivalents", values.get("cash"))
    other = values.get("minority_interest", 0.0) + values.get("preferred_equity", 0.0) + values.get("temporary_equity", 0.0)
    ev = cap + (debt or 0.0) - (cash or 0.0) + other if (debt is not None or cash is not None) else None

    def over(num, den):
        return None if num is None or den is None or den <= 0 else num / den

    income, fcf, ebitda = values.get("net_income"), values.get("free_cash_flow"), values.get("ebitda")
    ebit = values.get("operating_income", values.get("ebit"))
    paid = values.get("dividends_paid")
    bought = values.get("share_repurchases")
    payout = None if paid is None and bought is None else (paid or 0.0) + (bought or 0.0)
    rows = [("market_cap", "Market capitalisation", cap, "currency"), ("enterprise_value", "Enterprise value", ev, "currency"),
            ("earnings_yield", "Earnings yield (net income / market cap)", over(income, cap), "yield"), ("fcf_yield", "Free cash flow yield", over(fcf, cap), "yield"),
            ("book_to_price", "Book to price (equity / market cap)", over(values.get("shareholders_equity"), cap), "yield"), ("sales_to_price", "Sales to price", over(values.get("revenue"), cap), "yield"),
            ("ebit_to_ev", "EBIT / enterprise value", over(ebit, ev), "yield"), ("ebitda_to_ev", "EBITDA / enterprise value", over(ebitda, ev), "yield"),
            ("gross_profit_to_ev", "Gross profit / enterprise value", over(values.get("gross_profit"), ev), "yield"), ("dividend_yield", "Dividend yield", over(paid, cap), "yield"),
            ("buyback_yield", "Buyback yield", over(bought, cap), "yield"), ("shareholder_yield", "Shareholder yield (dividends + buybacks)", over(payout, cap), "yield"),
            ("pe", "Price / earnings", over(cap, income), "multiple"), ("pb", "Price / book", over(cap, values.get("shareholders_equity")), "multiple"),
            ("ps", "Price / sales", over(cap, values.get("revenue")), "multiple"), ("p_fcf", "Price / free cash flow", over(cap, fcf), "multiple"),
            ("ev_ebitda", "EV / EBITDA", over(ev, ebitda), "multiple"), ("ev_ebit", "EV / EBIT", over(ev, ebit), "multiple"), ("ev_sales", "EV / sales", over(ev, values.get("revenue")), "multiple")]
    return {"price": price, "rows": [{"name": n, "label": label, "value": v, "unit": unit} for n, label, v, unit in rows]}


def _ratio_table(tables: PeriodTables, security_id: int, periods: list[dict]) -> dict:
    """The derived metrics (every definition of :mod:`metrics`, the scores included) for the periods entered: annual columns on the annual basis, and for quarters the trailing twelve months
    ending then, where four quarters are held."""
    columns: list[dict] = []
    series: dict[str, list] = {}
    wanted = {"fy": {p["period_end"] for p in periods if p["fiscal_period"] == "FY"}, "ttm": {p["period_end"] for p in periods if p["fiscal_period"] != "FY"}}
    for basis in ("fy", "ttm"):
        table = tables.table(basis)
        if table.empty or not wanted[basis]:
            continue
        values = metrics.compute(tables, basis)
        mask = (table["security_id"] == int(security_id)) & table["period_end"].dt.strftime("%Y-%m-%d").isin(wanted[basis])
        for idx in table.index[mask]:
            end = table.at[idx, "period_end"].strftime("%Y-%m-%d")
            columns.append({"label": (f"FY {table.at[idx, 'fiscal_year']}" if basis == "fy" else f"TTM to {end}"), "basis": basis, "period_end": end})
            for name in values.columns:
                series.setdefault(name, []).append(None if pd.isna(values.at[idx, name]) else float(values.at[idx, name]))
    order = sorted(range(len(columns)), key=lambda i: (columns[i]["basis"] != "fy", columns[i]["period_end"]))
    columns = [columns[i] for i in order]
    info = {d["name"]: d for d in metrics.catalogue()}
    rows = []
    for name, vals in series.items():
        vals = [vals[i] for i in order]
        if all(v is None for v in vals):
            continue
        rows.append({"name": name, "label": info[name]["label"], "family": info[name]["family"], "unit": info[name]["unit"], "text": info[name]["text"], "values": vals})
    families = []
    for r in rows:
        if r["family"] not in families:
            families.append(r["family"])
    rows.sort(key=lambda r: families.index(r["family"]))                                  # the definitions were registered in the order they were written: the table reads by family
    return {"columns": columns, "rows": rows, "families": families, "defined": len(info), "computed": len(rows)}


def preview(store: FundamentalsStore, raw, root=".", today: date | None = None, mode: str = "correct") -> dict:
    """Check an entry and, if it has no error, compute everything the Fundamentals tab would show for the company once the entry was saved: the ratios and scores of every period entered, the
    multiples at a share price typed in, and the whole company view. It works on a temporary copy of the one company; nothing is written to the database."""
    from .service import FundamentalsService
    from .workbench import Workbench
    if not isinstance(raw, dict):
        raise FundamentalsError("the entry must be a dictionary")
    existing, company = _resolve(store, raw)
    checked = check_entry(raw, existing or company, today)
    out: dict = {"ok": checked["ok"], "issues": checked["issues"], "units": checked["units"],
                 "mode_note": ("A restatement is stored as a new vintage, public only from the date given. Studies read the figures as they were first filed, so it changes the raw panels and not the "
                               "ratios below, which show the restated figures as typed.") if mode == "restate" else None}
    if not checked["ok"]:
        return clean(out)
    periods = checked["periods"]
    with tempfile.TemporaryDirectory(prefix="manual_preview_") as folder:
        path = Path(folder) / "preview.db"
        target = FundamentalsStore(path)
        sid = _stage(store, target, existing, company, periods)
        latest_price = next((p for p in sorted(periods, key=lambda p: p["period_end"], reverse=True) if p["fiscal_period"] == "FY" and p["price"]), None)
        if latest_price is not None:                                                       # only in the copy: it lets the valuation yardsticks stand next to the price that was typed
            with target.db.transaction() as conn:
                conn.execute("INSERT OR REPLACE INTO prices_daily (security_id, date, adj_close, close, volume) VALUES (?,?,?,?,NULL)",
                             (sid, (today or _today()).isoformat(), latest_price["price"], latest_price["price"]))
        wb = Workbench(root, service=FundamentalsService(root, db_path=path))
        tables = PeriodTables(target, [sid], PitSettings())
        out["ratios"] = _ratio_table(tables, sid, periods)
        out["company"] = wb.company(sid, years=max(12, len(periods) + 3))
        try:
            out["valuation"] = wb.valuation(sid) if latest_price is not None else None
        except FundamentalsError as error:
            out["valuation"] = None
            out["valuation_note"] = str(error)
    out["multiples"] = [{"label": p["label"], "period_end": p["period_end"], **multiples({c: o["value"] for c, o in p["observations"].items()}, p["price"])}
                        for p in periods if p["price"] and p["fiscal_period"] == "FY"]
    skipped = [p["label"] for p in periods if p["price"] and p["fiscal_period"] != "FY"]
    out["multiples_note"] = (f"Multiples are shown for annual periods only; {', '.join(skipped)} would need the four quarters of a year." if skipped else None)
    out["derived"] = {p["label"]: p["derived"] for p in periods}
    out["overlaps"] = overlaps(store, existing["security_id"], periods) if existing is not None else []
    closes = store.closes([int(existing["security_id"])]) if existing is not None else pd.DataFrame()
    out["last_close"] = None if closes.empty or pd.isna(closes.iloc[-1, 0]) else {"date": str(closes.index[-1].date()), "close": float(closes.iloc[-1, 0])}
    out["new_company"] = company
    out["existing"] = None if existing is None else {"security_id": int(existing["security_id"]), "ticker": existing.get("ticker"), "name": existing.get("company_name")}
    return clean(out)
