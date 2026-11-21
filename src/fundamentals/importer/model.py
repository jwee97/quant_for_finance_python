"""Typed records of the extraction pipeline (standard-library dataclasses: no new dependency; they validate in ``__post_init__`` where a wrong value would be silent and expensive).

``FinancialPeriod``      what a column or a value covers: an instant or a span, annual / quarterly / half-year / year-to-date, forecast, restated
``ColumnRole``           what a column *is*: a period of reported figures, or a change, a percentage, a note reference, a forecast, another currency
``TableColumn``          a column of a table with its header hierarchy, box, role, period, currency, unit, scope, confidence and the reasons for each
``FinancialCandidate``   a figure found in a table cell, with the evidence for and against it; several can compete for one (metric, period)
``FinancialObservation`` the candidate that was selected, in exact decimal arithmetic, with raw and normalised values and the whole source
``ValidationResult``     one named check with a four-valued outcome (PASS / FAIL / INSUFFICIENT_DATA / NOT_APPLICABLE): missing data is never a pass
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum


class ColumnRole(str, Enum):
    CURRENT_PERIOD = "CURRENT_PERIOD"
    PRIOR_PERIOD = "PRIOR_PERIOD"
    HISTORICAL_PERIOD = "HISTORICAL_PERIOD"
    ABSOLUTE_CHANGE = "ABSOLUTE_CHANGE"
    PERCENT_CHANGE = "PERCENT_CHANGE"
    FORECAST = "FORECAST"
    RESTATED = "RESTATED"
    NOTE_REFERENCE = "NOTE_REFERENCE"
    PER_SHARE = "PER_SHARE"
    ALTERNATIVE_CURRENCY = "ALTERNATIVE_CURRENCY"
    OTHER = "OTHER"


REPORTING_ROLES = (ColumnRole.CURRENT_PERIOD, ColumnRole.PRIOR_PERIOD, ColumnRole.HISTORICAL_PERIOD, ColumnRole.RESTATED)


class Check(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


PERIOD_TYPES = ("INSTANT", "ANNUAL", "QUARTER", "HALF_YEAR", "YTD", "OTHER", "UNKNOWN_FLOW")


@dataclass(frozen=True)
class FinancialPeriod:
    end_date: str
    start_date: str | None = None
    duration_months: int | None = None
    period_type: str = "UNKNOWN_FLOW"
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    is_forecast: bool = False
    is_restated: bool = False
    weeks: int | None = None                      # 52 / 53 for a fiscal year that ends on a weekday rule rather than a month end

    def __post_init__(self):
        if self.period_type not in PERIOD_TYPES:
            raise ValueError(f"unknown period type {self.period_type!r}")

    @property
    def is_instant(self) -> bool:
        return self.period_type == "INSTANT"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableColumn:
    col: int
    header_levels: list[str] = field(default_factory=list)       # the header hierarchy, top to bottom, as printed
    bbox: tuple | None = None
    role: ColumnRole = ColumnRole.OTHER
    period: FinancialPeriod | None = None
    currency: str | None = None
    unit: str | None = None                                       # units | thousands | millions | billions
    scope: str = "unspecified"                                    # consolidated | parent | unspecified
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    resolved_header: str = ""

    @property
    def original_header(self) -> str:
        return " / ".join(h for h in self.header_levels if h)

    def to_dict(self) -> dict:
        return {"col": self.col, "original_header": self.original_header, "header_levels": self.header_levels, "resolved_header": self.resolved_header, "bbox": list(self.bbox) if self.bbox else None,
                "role": self.role.value, "period": self.period.to_dict() if self.period else None, "currency": self.currency, "unit": self.unit, "scope": self.scope, "confidence": round(self.confidence, 3),
                "reasons": self.reasons}


@dataclass
class FinancialCandidate:
    metric: str
    label: str
    raw_value: Decimal
    page: int | None
    table_id: str
    row: int
    col: int
    bbox: tuple | None
    statement: str                                                # IS | BS | CF | SUMMARY | SEGMENT | ...
    scope: str
    column_role: str
    period_end: str
    currency: str | None
    unit: str | None
    authority: int                                                # 100 primary consolidated statement ... 10 note
    score: float = 0.0
    status: str = "candidate"                                     # candidate | selected | rejected
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_value"] = str(self.raw_value)
        return d


@dataclass
class ValidationResult:
    check_id: str
    label: str
    status: Check
    detail: str = ""
    lines: list[str] = field(default_factory=list)
    period: str | None = None

    def to_dict(self) -> dict:
        return {"id": self.check_id, "label": self.label, "status": self.status.value, "detail": self.detail, "lines": self.lines, "period": self.period}


def normalise(raw: Decimal | float | str, multiplier: int | float) -> Decimal:
    """Raw reported value times the unit multiplier, in exact decimal arithmetic (never through a float)."""
    return Decimal(str(raw)) * Decimal(str(multiplier))


__all__ = ["ColumnRole", "REPORTING_ROLES", "Check", "FinancialPeriod", "TableColumn", "FinancialCandidate", "ValidationResult", "normalise"]
