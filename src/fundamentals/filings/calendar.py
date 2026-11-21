"""What each country's listed companies are expected to file, and how a source's own names for a report map onto the application's fiscal periods.

The expected set matters for coverage: a Hong Kong company files a half-year and an annual report, so a missing Q1 is not a gap; a Japanese company filed quarterly reports until the regime changed, so
which quarters count depends on the date. Rules here are the usual obligations of the main market in each country, stated with their limits (``note``); a company may file more (voluntary quarterly
reports are common in Hong Kong and Singapore), and a smaller market may file less. A period held beyond the expected set is simply held.

``fiscal_period`` values are the application's own: ``FY`` (twelve months), ``Q1`` to ``Q4`` (three months), ``H1`` and ``H2`` (six months).
"""

from __future__ import annotations

from datetime import date

# country code (ISO 3166-1 alpha-2) -> what is filed. Each entry: periods, standard, currency, note.
CALENDAR: dict[str, dict] = {
    "US": {"periods": ("Q1", "Q2", "Q3", "Q4", "FY"), "standard": "US GAAP", "currency": "USD", "note": "10-Q for three quarters, 10-K for the year; Q4 is derived (never filed). Foreign private issuers file 20-F (annual) and 6-K."},
    "JP": {"periods": ("Q1", "Q2", "Q3", "FY"), "standard": "JGAAP / IFRS / US GAAP", "currency": "JPY", "note": "Quarterly securities reports for Q1 to Q3 and an annual securities report; for fiscal years starting from April 2024 the quarterly securities report was replaced by a half-year report, while exchange rules still require quarterly earnings summaries (tanshin). Check the filer's own documents."},
    "KR": {"periods": ("Q1", "H1", "Q3", "FY"), "standard": "K-IFRS", "currency": "KRW", "note": "Quarterly (Q1, Q3), half-year and annual business reports on OpenDART; the half-year and the annual report carry the full statements."},
    "SG": {"periods": ("H1", "FY"), "standard": "SFRS(I) / IFRS", "currency": "SGD", "note": "Quarterly reporting stopped being mandatory in 2020: half-year and full-year results are the minimum; many companies still report quarterly."},
    "CN": {"periods": ("Q1", "H1", "Q3", "FY"), "standard": "CAS (PRC GAAP)", "currency": "CNY", "note": "Q1, half-year, Q3 and annual reports for Shanghai and Shenzhen; the half-year and annual reports are the full statements, Q1 and Q3 are shorter."},
    "HK": {"periods": ("H1", "FY"), "standard": "HKFRS / IFRS", "currency": "HKD", "note": "Interim (half-year) and annual reports; quarterly reports are voluntary. Mainland issuers may report in CNY."},
    "FR": {"periods": ("H1", "FY"), "standard": "IFRS", "currency": "EUR", "note": "Annual financial reports (ESEF, inline XBRL) and half-year reports; quarterly information is voluntary."},
    "ES": {"periods": ("H1", "FY"), "standard": "IFRS", "currency": "EUR", "note": "Annual (ESEF) and half-year reports; quarterly information is voluntary."},
    "DE": {"periods": ("Q1", "H1", "Q3", "FY"), "standard": "IFRS", "currency": "EUR", "note": "Annual (ESEF) and half-year reports; quarterly statements are required for the Prime Standard segment of the Frankfurt exchange."},
    "CA": {"periods": ("Q1", "Q2", "Q3", "Q4", "FY"), "standard": "IFRS / US GAAP", "currency": "CAD", "note": "Interim reports for three quarters and an annual report on SEDAR+; Q4 is derived."},
    "BR": {"periods": ("Q1", "Q2", "Q3", "FY"), "standard": "BR GAAP (CPC) / IFRS", "currency": "BRL", "note": "ITR for three quarters and DFP for the year, on CVM's open data; the figures are consolidated where the company has subsidiaries."},
}
DEFAULT = {"periods": ("Q1", "Q2", "Q3", "FY"), "standard": "", "currency": None, "note": "No rule is known for this country: quarters and the year are expected."}

# a source's own name for a report -> (fiscal period, is_correction)
EDINET_DOC_TYPES = {"120": ("FY", False), "130": ("FY", True), "140": ("Q", False), "150": ("Q", True), "160": ("H1", False), "170": ("H1", True)}
DART_REPORT_CODES = {"11013": "Q1", "11012": "H1", "11014": "Q3", "11011": "FY"}
DART_NAME_WORDS = (("사업보고서", "FY"), ("반기보고서", "H1"), ("분기보고서", "Q"))
CHINA_WORDS = (("半年度报告", "H1"), ("半年报", "H1"), ("中期报告", "H1"), ("第一季度报告", "Q1"), ("一季报", "Q1"), ("第三季度报告", "Q3"), ("三季报", "Q3"), ("年度报告", "FY"), ("年报", "FY"))
HK_WORDS = (("annual report", "FY"), ("interim report", "H1"), ("interim results", "H1"), ("annual results", "FY"), ("年報", "FY"), ("年度報告", "FY"), ("中期報告", "H1"), ("quarterly", "Q"))
SG_WORDS = (("annual report", "FY"), ("full year financial statements", "FY"), ("full-year", "FY"), ("half year", "H1"), ("1h", "H1"), ("quarter", "Q"), ("financial statements", "FY"))
JP_WORDS = (("有価証券報告書", "FY"), ("四半期報告書", "Q"), ("半期報告書", "H1"), ("決算短信", "FY"))
GENERIC_WORDS = (("annual", "FY"), ("interim", "H1"), ("half-year", "H1"), ("half year", "H1"), ("quarter", "Q"), ("10-k", "FY"), ("10-q", "Q"), ("20-f", "FY"))


def expected(country: str | None) -> dict:
    return CALENDAR.get((country or "").upper(), DEFAULT)


def expected_periods(country: str | None, year: int | None = None) -> tuple[str, ...]:
    """The fiscal periods a company of ``country`` is expected to have for a fiscal year. For Japan the quarterly reports are expected for fiscal years before 2024 only."""
    rule = expected(country)["periods"]
    if (country or "").upper() == "JP" and year is not None and year >= 2024:
        return ("H1", "FY")
    return rule


def classify_title(title: str, words=GENERIC_WORDS) -> str | None:
    """The kind of report a title names (``FY``, ``H1``, ``Q1``, ``Q3`` or ``Q`` when the quarter is not in the words), or None."""
    low = (title or "").lower()
    for word, kind in words:
        if word.lower() in low:
            return kind
    return None


def quarter_of(period_end: date, fiscal_year_end_month: int) -> str:
    """Q1 to Q4 of a quarter that ends on ``period_end`` for a fiscal year that ends in ``fiscal_year_end_month``."""
    behind = (fiscal_year_end_month - period_end.month) % 12
    return {0: "Q4", 3: "Q3", 6: "Q2", 9: "Q1"}.get(behind, "Q")
