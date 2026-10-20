"""The application's own vocabulary for company fundamentals, and how each data provider's names map onto it.

Everything downstream (the database, the point-in-time engine, the factor engine) speaks the canonical metric codes defined here, never a provider's field names. A provider adapter hands over
whatever it was given (``source_metric``: value); :func:`normalise_values` turns that into canonical observations, records which source field each came from, fills the few values that are exact
arithmetic on others (gross profit, free cash flow, total debt) and says so, and reports every source field it did not recognise so a mapping gap is visible instead of silent.

Units. Money is in the reporting currency's own units (not thousands, not millions); per-share values in currency per share; share counts in shares. Cash *outflows* the statements report as
negative numbers (capital expenditure, dividends, buybacks, acquisitions) are stored as positive amounts: ``capital_expenditure = 12`` means twelve was spent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    code: str
    label: str
    statement: str            # income | balance | cashflow | shares
    kind: str                 # flow (over a period) | stock (at the period end) | per_share (a flow, per share) | shares (a count)
    nonnegative: bool = False  # True when a negative value is an error, not a loss
    text: str = ""


def _m(code, label, statement, kind, nonneg=False, text=""):
    return Metric(code, label, statement, kind, nonneg, text)


METRICS: dict[str, Metric] = {m.code: m for m in [
    # income statement (flows)
    _m("revenue", "Revenue", "income", "flow", True, "Sales to customers over the period."),
    _m("cost_of_revenue", "Cost of revenue", "income", "flow", True, "Direct cost of the goods and services sold."),
    _m("gross_profit", "Gross profit", "income", "flow", False, "Revenue less cost of revenue."),
    _m("research_and_development", "Research and development", "income", "flow", True, "Research and development expense."),
    _m("selling_general_admin", "Selling, general and administrative", "income", "flow", True, "Selling, general and administrative expense."),
    _m("operating_income", "Operating income", "income", "flow", False, "Profit from operations before interest and tax."),
    _m("ebit", "EBIT", "income", "flow", False, "Earnings before interest and tax (operating income when the provider gives no other figure)."),
    _m("ebitda", "EBITDA", "income", "flow", False, "EBIT plus depreciation and amortisation."),
    _m("interest_expense", "Interest expense", "income", "flow", True, "Interest paid or accrued on debt."),
    _m("income_before_tax", "Income before tax", "income", "flow", False, "Profit before income tax."),
    _m("income_tax", "Income tax", "income", "flow", False, "Income tax expense."),
    _m("net_income", "Net income", "income", "flow", False, "Profit for the period."),
    _m("eps_basic", "EPS (basic)", "income", "per_share", False, "Net income per basic share."),
    _m("eps_diluted", "EPS (diluted)", "income", "per_share", False, "Net income per diluted share."),
    # balance sheet (stocks)
    _m("cash", "Cash and short-term investments", "balance", "stock", True, "Cash plus short-term investments."),
    _m("cash_and_equivalents", "Cash and equivalents", "balance", "stock", True, "Cash and cash equivalents."),
    _m("accounts_receivable", "Receivables", "balance", "stock", True, "Trade and other receivables."),
    _m("inventory", "Inventory", "balance", "stock", True, "Inventories."),
    _m("current_assets", "Current assets", "balance", "stock", True, "Assets expected to turn into cash within a year."),
    _m("total_assets", "Total assets", "balance", "stock", True, "Everything the company owns."),
    _m("ppe", "Property, plant and equipment", "balance", "stock", True, "Net property, plant and equipment."),
    _m("goodwill", "Goodwill", "balance", "stock", True, "Goodwill."),
    _m("intangible_assets", "Intangible assets", "balance", "stock", True, "Intangible assets other than goodwill."),
    _m("accounts_payable", "Payables", "balance", "stock", True, "Trade payables."),
    _m("current_liabilities", "Current liabilities", "balance", "stock", True, "Obligations due within a year."),
    _m("short_term_debt", "Short-term debt", "balance", "stock", True, "Borrowings due within a year."),
    _m("long_term_debt", "Long-term debt", "balance", "stock", True, "Borrowings due after a year."),
    _m("total_debt", "Total debt", "balance", "stock", True, "Short-term plus long-term borrowings."),
    _m("total_liabilities", "Total liabilities", "balance", "stock", True, "Everything the company owes."),
    _m("shareholders_equity", "Shareholders' equity", "balance", "stock", False, "Assets less liabilities (the owners' book value)."),
    _m("minority_interest", "Minority interest", "balance", "stock", False, "Equity held by minority shareholders of subsidiaries."),
    _m("preferred_equity", "Preferred equity", "balance", "stock", True, "Preferred stock."),
    # cash flow (flows)
    _m("operating_cash_flow", "Operating cash flow", "cashflow", "flow", False, "Net cash from operating activities."),
    _m("capital_expenditure", "Capital expenditure", "cashflow", "flow", True, "Cash spent on property, plant and equipment (positive)."),
    _m("free_cash_flow", "Free cash flow", "cashflow", "flow", False, "Operating cash flow less capital expenditure."),
    _m("depreciation_amortization", "Depreciation and amortisation", "cashflow", "flow", True, "Non-cash charge for the wear of assets."),
    _m("stock_based_compensation", "Stock-based compensation", "cashflow", "flow", True, "Non-cash pay in shares or options."),
    _m("acquisitions", "Acquisitions", "cashflow", "flow", True, "Cash spent buying businesses (positive)."),
    _m("share_repurchases", "Share repurchases", "cashflow", "flow", True, "Cash spent buying back shares (positive)."),
    _m("dividends_paid", "Dividends paid", "cashflow", "flow", True, "Cash paid to shareholders (positive)."),
    _m("investing_cash_flow", "Investing cash flow", "cashflow", "flow", False, "Net cash from investing activities."),
    _m("financing_cash_flow", "Financing cash flow", "cashflow", "flow", False, "Net cash from financing activities."),
    # share counts
    _m("shares_outstanding", "Shares outstanding", "shares", "shares", True, "Shares in issue at the period end."),
    _m("weighted_avg_shares_diluted", "Diluted weighted-average shares", "shares", "shares", True, "Average diluted share count over the period."),
]}

STATEMENTS = ("income", "balance", "cashflow", "shares")
FLOW_KINDS = ("flow", "per_share")


# ------------------------------------------------------------------------------------------------------------------ source mappings
# canonical code -> list of (source field, transform). A transform is "as_is", "abs" (the statements show an outflow as a negative number) or "outflow" (an outflow only: a positive source value
# is something else, such as proceeds from a disposal, and is dropped).
FMP_MAP: dict[str, list[tuple[str, str]]] = {
    # income statement
    "revenue": [("income:revenue", "as_is")], "cost_of_revenue": [("income:costOfRevenue", "as_is")], "gross_profit": [("income:grossProfit", "as_is")],
    "research_and_development": [("income:researchAndDevelopmentExpenses", "as_is")], "selling_general_admin": [("income:sellingGeneralAndAdministrativeExpenses", "as_is")],
    "operating_income": [("income:operatingIncome", "as_is")], "ebit": [("income:ebit", "as_is")], "ebitda": [("income:ebitda", "as_is")], "interest_expense": [("income:interestExpense", "abs")],
    "income_before_tax": [("income:incomeBeforeTax", "as_is")], "income_tax": [("income:incomeTaxExpense", "as_is")], "net_income": [("income:netIncome", "as_is")],
    "eps_basic": [("income:eps", "as_is")], "eps_diluted": [("income:epsDiluted", "as_is"), ("income:epsdiluted", "as_is")],
    "weighted_avg_shares_diluted": [("income:weightedAverageShsOutDil", "as_is")],
    # balance sheet
    "cash": [("balance:cashAndShortTermInvestments", "as_is")], "cash_and_equivalents": [("balance:cashAndCashEquivalents", "as_is")],
    "accounts_receivable": [("balance:netReceivables", "as_is"), ("balance:accountsReceivables", "as_is")], "inventory": [("balance:inventory", "as_is")],
    "current_assets": [("balance:totalCurrentAssets", "as_is")], "total_assets": [("balance:totalAssets", "as_is")], "ppe": [("balance:propertyPlantEquipmentNet", "as_is")],
    "goodwill": [("balance:goodwill", "as_is")], "intangible_assets": [("balance:intangibleAssets", "as_is")],
    "accounts_payable": [("balance:accountPayables", "as_is"), ("balance:totalPayables", "as_is")], "current_liabilities": [("balance:totalCurrentLiabilities", "as_is")],
    "short_term_debt": [("balance:shortTermDebt", "as_is")], "long_term_debt": [("balance:longTermDebt", "as_is")], "total_debt": [("balance:totalDebt", "as_is")],
    "total_liabilities": [("balance:totalLiabilities", "as_is")], "shareholders_equity": [("balance:totalStockholdersEquity", "as_is"), ("balance:totalEquity", "as_is")],
    "minority_interest": [("balance:minorityInterest", "as_is")], "preferred_equity": [("balance:preferredStock", "as_is")],
    # cash flow statement
    "operating_cash_flow": [("cashflow:netCashProvidedByOperatingActivities", "as_is"), ("cashflow:operatingCashFlow", "as_is")], "capital_expenditure": [("cashflow:capitalExpenditure", "abs")],
    "free_cash_flow": [("cashflow:freeCashFlow", "as_is")], "depreciation_amortization": [("cashflow:depreciationAndAmortization", "as_is"), ("income:depreciationAndAmortization", "as_is")],
    "stock_based_compensation": [("cashflow:stockBasedCompensation", "as_is")], "acquisitions": [("cashflow:acquisitionsNet", "outflow")],
    "share_repurchases": [("cashflow:commonStockRepurchased", "outflow")], "dividends_paid": [("cashflow:commonDividendsPaid", "abs"), ("cashflow:netDividendsPaid", "abs")],
    "investing_cash_flow": [("cashflow:netCashProvidedByInvestingActivities", "as_is")], "financing_cash_flow": [("cashflow:netCashProvidedByFinancingActivities", "as_is")],
}

# OpenFilings normalises to a small set of line-item codes (income statement, balance sheet, cash flow). It gives no debt split, no depreciation, no SG&A or R&D and no share counts, so the
# factors that need those are simply missing for a company only OpenFilings covers; the coverage table in the Data Manager shows it.
OPENFILINGS_MAP: dict[str, list[tuple[str, str]]] = {
    "revenue": [("revenue", "as_is")], "cost_of_revenue": [("cost_of_revenue", "as_is")], "gross_profit": [("gross_profit", "as_is")],
    "operating_income": [("operating_income_loss", "as_is")], "interest_expense": [("finance_costs", "abs")], "income_before_tax": [("profit_before_tax", "as_is")],
    "income_tax": [("income_tax", "as_is")], "net_income": [("net_income_loss", "as_is")], "eps_basic": [("basic_eps", "as_is")], "eps_diluted": [("diluted_eps", "as_is")],
    "cash_and_equivalents": [("cash_and_cash_equivalents", "as_is")], "accounts_receivable": [("trade_receivables", "as_is")], "inventory": [("inventory", "as_is")],
    "ppe": [("property_plant_equipment", "as_is")], "goodwill": [("goodwill", "as_is")], "intangible_assets": [("intangible_assets", "as_is")],
    "current_assets": [("current_assets", "as_is")], "total_assets": [("total_assets", "as_is")], "current_liabilities": [("current_liabilities", "as_is")],
    "total_liabilities": [("total_liabilities", "as_is")], "shareholders_equity": [("total_equity", "as_is")],
    "operating_cash_flow": [("operating_cash_flow", "as_is")], "capital_expenditure": [("capital_expenditure", "abs")], "dividends_paid": [("dividends_paid", "abs")],
    "investing_cash_flow": [("investing_cash_flow", "as_is")], "financing_cash_flow": [("financing_cash_flow", "as_is")],
}
# an approximation worth saying so: OpenFilings' finance costs include more than interest on debt
OPENFILINGS_APPROXIMATE = {"interest_expense"}

# the simulated provider speaks canonical codes already
IDENTITY_MAP: dict[str, list[tuple[str, str]]] = {code: [(code, "as_is")] for code in METRICS}

SOURCE_MAPS = {"fmp": FMP_MAP, "openfilings": OPENFILINGS_MAP, "simulated": IDENTITY_MAP}
APPROXIMATE = {"openfilings": OPENFILINGS_APPROXIMATE}


def _finite(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _apply(value: float, transform: str) -> float | None:
    if transform == "as_is":
        return value
    if transform == "abs":
        return abs(value)
    if transform == "outflow":
        return -value if value < 0 else None
    raise ValueError(f"unknown transform {transform!r}")


def normalise_values(provider: str, values: dict) -> tuple[dict[str, dict], list[str]]:
    """Canonical observations from one period's source fields.

    Returns ``({code: {"value", "source_metric", "flag"}}, unmapped)`` where ``flag`` is ``ok``, ``derived`` (computed here from other observations) or ``estimated`` (the source field is only an
    approximation of the canonical metric) and ``unmapped`` lists the source fields that map to nothing (kept in the raw payload, reported in the Data Manager).
    """
    table = SOURCE_MAPS.get(provider)
    if table is None:
        raise ValueError(f"no field mapping for provider {provider!r}")
    out: dict[str, dict] = {}
    used: set[str] = set()
    for code, options in table.items():
        for source, transform in options:
            if source not in values:
                continue
            v = _finite(values[source])
            if v is None:
                continue
            used.add(source)
            v = _apply(v, transform)
            if v is not None and code not in out:
                out[code] = {"value": v, "source_metric": source, "flag": "estimated" if code in APPROXIMATE.get(provider, ()) else "ok"}
    known = {source for options in table.values() for source, _ in options}
    unmapped = sorted(str(k) for k in values if k not in known and k not in used)
    _derive(out)
    return out, unmapped


def _derive(obs: dict[str, dict]) -> None:
    """The values that are exact arithmetic on others, filled only when the provider gave no figure of its own, and marked ``derived``."""
    def has(code):
        return code in obs

    def put(code, value, sources):
        if code not in obs and value is not None and math.isfinite(value):
            obs[code] = {"value": float(value), "source_metric": "derived: " + sources, "flag": "derived"}

    if has("revenue") and has("cost_of_revenue"):
        put("gross_profit", obs["revenue"]["value"] - obs["cost_of_revenue"]["value"], "revenue - cost_of_revenue")
    if has("operating_income"):
        put("ebit", obs["operating_income"]["value"], "operating_income")
    if has("ebit") and has("depreciation_amortization"):
        put("ebitda", obs["ebit"]["value"] + obs["depreciation_amortization"]["value"], "ebit + depreciation_amortization")
    if has("operating_cash_flow") and has("capital_expenditure"):
        put("free_cash_flow", obs["operating_cash_flow"]["value"] - obs["capital_expenditure"]["value"], "operating_cash_flow - capital_expenditure")
    if has("short_term_debt") or has("long_term_debt"):
        put("total_debt", (obs.get("short_term_debt", {"value": 0.0})["value"]) + (obs.get("long_term_debt", {"value": 0.0})["value"]), "short_term_debt + long_term_debt")
    if has("cash_and_equivalents"):
        put("cash", obs["cash_and_equivalents"]["value"], "cash_and_equivalents")
    if has("weighted_avg_shares_diluted"):
        put("shares_outstanding", obs["weighted_avg_shares_diluted"]["value"], "weighted_avg_shares_diluted")


def metric(code: str) -> Metric:
    try:
        return METRICS[code]
    except KeyError:
        raise KeyError(f"unknown metric {code!r}") from None


def flows() -> list[str]:
    return [c for c, m in METRICS.items() if m.kind in FLOW_KINDS]


def stocks() -> list[str]:
    return [c for c, m in METRICS.items() if m.kind in ("stock", "shares")]
