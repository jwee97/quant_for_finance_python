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
    _m("temporary_equity", "Redeemable (temporary) equity", "balance", "stock", True, "Redeemable preferred stock and redeemable minority interests, shown between liabilities and equity because the holders can require the company to buy them back."),
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
# canonical code -> list of (source field, transform), first match wins. A transform is "as_is", "abs" (a cash outflow or a cost whose sign differs between companies and markets: the application
# stores the amount spent as a positive number) or "outflow" (an outflow only: a positive source value is something else, such as proceeds from a disposal, and is dropped).
#
# OpenFilings (https://openfilings.org) returns its own canonical line-item keys, as filed, in the reporting currency's units (the documentation says "reporting currency units"; it does not say
# whether cash outflows are negative, so the outflow lines are read as magnitudes). Its taxonomy has 191 keys; these are the ones the research uses. Keys not listed here are never requested.
OPENFILINGS_MAP: dict[str, list[tuple[str, str]]] = {
    # income statement
    "revenue": [("revenue", "as_is")], "cost_of_revenue": [("cost_of_revenue", "as_is"), ("cost_of_goods_sold", "as_is")], "gross_profit": [("gross_profit", "as_is")],
    "research_and_development": [("rnd", "as_is")], "selling_general_admin": [("sga_expense", "as_is")], "operating_income": [("operating_income", "as_is")], "ebit": [("ebit", "as_is")],
    "ebitda": [("ebitda", "as_is")], "interest_expense": [("interest_expense", "abs")], "income_before_tax": [("pretax_income", "as_is")], "income_tax": [("income_tax", "as_is")],
    "net_income": [("net_income", "as_is"), ("net_income_attributable", "as_is")], "eps_basic": [("eps_basic", "as_is")], "eps_diluted": [("eps_diluted", "as_is")],
    # balance sheet
    "cash": [("total_cash_st_investments", "as_is")], "cash_and_equivalents": [("cash", "as_is")],
    "accounts_receivable": [("accounts_receivable", "as_is"), ("trade_receivables", "as_is"), ("total_receivables", "as_is")], "inventory": [("inventory", "as_is")],
    "current_assets": [("current_assets", "as_is")], "total_assets": [("total_assets", "as_is")], "ppe": [("net_ppe", "as_is")], "goodwill": [("goodwill", "as_is")],
    "intangible_assets": [("other_intangibles", "as_is")], "accounts_payable": [("accounts_payable", "as_is"), ("trade_payables", "as_is")],
    "current_liabilities": [("current_liabilities", "as_is")], "short_term_debt": [("short_term_debt", "as_is"), ("ibd_current", "as_is")],
    "long_term_debt": [("long_term_debt", "as_is"), ("ibd_noncurrent", "as_is")], "total_debt": [("total_debt", "as_is")], "total_liabilities": [("total_liabilities", "as_is")],
    "shareholders_equity": [("equity", "as_is")], "minority_interest": [("noncontrolling_interest", "as_is")], "preferred_equity": [("preferred_stock", "as_is")],
    # cash flow statement
    "operating_cash_flow": [("operating_cash_flow", "as_is")], "capital_expenditure": [("capex", "abs")], "free_cash_flow": [("free_cash_flow", "as_is")],
    "depreciation_amortization": [("depreciation_amortization", "as_is"), ("da_supplemental", "as_is")], "stock_based_compensation": [("stock_based_comp", "as_is")],
    "acquisitions": [("cash_acquisitions", "abs")], "share_repurchases": [("repurchase_of_common_stock", "abs")],
    "dividends_paid": [("cash_dividends_paid", "abs"), ("common_dividends_paid", "abs")], "investing_cash_flow": [("investing_cash_flow", "as_is")],
    "financing_cash_flow": [("financing_cash_flow", "as_is")],
    # share counts
    "weighted_avg_shares_diluted": [("shares_diluted", "as_is")], "shares_outstanding": [("shares_basic", "as_is")],
}
# a source field that is only an approximation of the canonical metric is said to be: OpenFilings gives a basic share count for the year, not the count in issue at the period end
OPENFILINGS_APPROXIMATE = {"shares_outstanding"}
OPENFILINGS_KEYS: tuple[str, ...] = tuple(dict.fromkeys(source for options in OPENFILINGS_MAP.values() for source, _ in options))

# SEC EDGAR (https://www.sec.gov/edgar): the XBRL facts every US filer (and the foreign companies that file annual reports with the SEC) tags in its filings, as filed. A source field is written
# ``taxonomy:Element`` exactly as the filing tags it (``us-gaap:Revenues``, ``ifrs-full:Revenue``), so the source field stored with each figure says which element it came from. Companies change the
# element they use for the same line over the years (revenue is ``SalesRevenueNet`` before 2018 and ``RevenueFromContractWithCustomerExcludingAssessedTax`` after, for many), so each canonical metric
# lists the elements in the order they are tried, within one filing: the total comes before its parts, and the broad element before the narrow one. Money is in the filing's own currency units.
def _gaap(*names: str, transform: str = "as_is") -> list[tuple[str, str]]:
    return [(f"us-gaap:{name}", transform) for name in names]


def _ifrs(*names: str, transform: str = "as_is") -> list[tuple[str, str]]:
    return [(f"ifrs-full:{name}", transform) for name in names]


SEC_DEBT_CURRENT = "derived: borrowings due within a year (short-term borrowings + commercial paper + current long-term debt)"
SEC_CASH_AND_INVESTMENTS = "derived: cash and equivalents + short-term investments"
SEC_POSITIVE_ONLY = frozenset({"dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding", "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
                               "us-gaap:WeightedAverageNumberOfSharesOutstandingBasicAndDiluted", "ifrs-full:AdjustedWeightedAverageShares"})     # a share count of zero is a cover page left at 0, not a company without shares
SEC_LIABILITIES = "derived: total assets - total equity"
SEC_TEMP_EQUITY = "derived: redeemable equity of the parent + redeemable minority interests"      # the two parts of the mezzanine when the filing gives no total (Kraft Heinz: 8,320 + 23)
SEC_PARENT_EQUITY = "derived: total equity - minority interest"                       # for a filing that tags only the total (AT&T, Verizon, Kinder Morgan): the parent's share, not the total
SEC_SGA = "derived: selling and marketing + general and administrative"
SEC_PRETAX = "derived: domestic + foreign income before tax"
SEC_MAP: dict[str, list[tuple[str, str]]] = {
    # income statement
    "revenue": _gaap("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                     "RevenuesNetOfInterestExpense") + _ifrs("Revenue", "RevenueFromContractsWithCustomers"),
    "cost_of_revenue": _gaap("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices") + _ifrs("CostOfSales"),
    "gross_profit": _gaap("GrossProfit") + _ifrs("GrossProfit"),
    "research_and_development": _gaap("ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost") + _ifrs("ResearchAndDevelopmentExpense"),
    "selling_general_admin": _gaap("SellingGeneralAndAdministrativeExpense") + [(SEC_SGA, "as_is")] + _ifrs("SellingGeneralAndAdministrativeExpense"),
    "operating_income": _gaap("OperatingIncomeLoss") + _ifrs("ProfitLossFromOperatingActivities"),
    "interest_expense": _gaap("InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense", "InterestExpenseNonoperating", "InterestExpenseLongTermDebt", transform="abs")
                        + _ifrs("InterestExpense", "FinanceCosts", transform="abs"),
    "income_before_tax": _gaap("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments") + [(SEC_PRETAX, "as_is")] + _ifrs("ProfitLossBeforeTax"),
    "income_tax": _gaap("IncomeTaxExpenseBenefit", "IncomeTaxExpenseBenefitContinuingOperations") + _ifrs("IncomeTaxExpenseContinuingOperations"),
    "net_income": _gaap("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic") + _ifrs("ProfitLossAttributableToOwnersOfParent", "ProfitLoss"),
    "eps_basic": _gaap("EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted", "IncomeLossFromContinuingOperationsPerBasicShare", "IncomeLossFromContinuingOperationsPerBasicAndDilutedShare")
                 + _ifrs("BasicEarningsLossPerShare"),
    "eps_diluted": _gaap("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "IncomeLossFromContinuingOperationsPerDilutedShare", "IncomeLossFromContinuingOperationsPerBasicAndDilutedShare")
                   + _ifrs("DilutedEarningsLossPerShare"),
    # balance sheet
    "cash": _gaap("CashCashEquivalentsAndShortTermInvestments") + [(SEC_CASH_AND_INVESTMENTS, "as_is")],
    "cash_and_equivalents": _gaap("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash", "CashAndDueFromBanks") + _ifrs("CashAndCashEquivalents"),
    "accounts_receivable": _gaap("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsAndOtherReceivablesNetCurrent") + _ifrs("TradeAndOtherCurrentReceivables", "CurrentTradeReceivables"),
    "inventory": _gaap("InventoryNet") + _ifrs("Inventories"),
    "current_assets": _gaap("AssetsCurrent") + _ifrs("CurrentAssets"),
    "total_assets": _gaap("Assets") + _ifrs("Assets"),
    "ppe": _gaap("PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization") + _ifrs("PropertyPlantAndEquipment"),
    "goodwill": _gaap("Goodwill") + _ifrs("Goodwill"),
    "intangible_assets": _gaap("IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet") + _ifrs("IntangibleAssetsOtherThanGoodwill"),
    "accounts_payable": _gaap("AccountsPayableCurrent", "AccountsPayableTradeCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent") + _ifrs("TradeAndOtherCurrentPayables", "CurrentTradePayables"),
    "current_liabilities": _gaap("LiabilitiesCurrent") + _ifrs("CurrentLiabilities"),
    "short_term_debt": _gaap("DebtCurrent") + [(SEC_DEBT_CURRENT, "as_is")],
    "long_term_debt": _gaap("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermNotesPayable", "SeniorLongTermNotes", "ConvertibleNotesPayable") + _ifrs("NoncurrentPortionOfNoncurrentBorrowings"),
    "total_liabilities": _gaap("Liabilities") + _ifrs("Liabilities") + [(SEC_LIABILITIES, "as_is")],
    "temporary_equity": _gaap("TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests", "TemporaryEquityCarryingAmount") + [(SEC_TEMP_EQUITY, "as_is")],
    "shareholders_equity": _gaap("StockholdersEquity") + [(SEC_PARENT_EQUITY, "as_is")] + _gaap("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest") + _ifrs("EquityAttributableToOwnersOfParent", "Equity"),
    "minority_interest": _gaap("MinorityInterest") + _ifrs("NoncontrollingInterests"),
    "preferred_equity": _gaap("PreferredStockValue", "PreferredStockValueOutstanding"),
    # cash flow statement
    "operating_cash_flow": _gaap("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations") + _ifrs("CashFlowsFromUsedInOperatingActivities"),
    "capital_expenditure": _gaap("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements", transform="abs")
                           + _ifrs("PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", transform="abs"),
    "depreciation_amortization": _gaap("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation")
                                 + _ifrs("DepreciationAndAmortisationExpense"),
    "stock_based_compensation": _gaap("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense") + _ifrs("AdjustmentsForSharebasedPayments"),
    "acquisitions": _gaap("PaymentsToAcquireBusinessesNetOfCashAcquired", "PaymentsToAcquireBusinessesGross", transform="abs"),
    "share_repurchases": _gaap("PaymentsForRepurchaseOfCommonStock", transform="abs"),
    "dividends_paid": _gaap("PaymentsOfDividends", "PaymentsOfDividendsCommonStock", transform="abs") + _ifrs("DividendsPaidClassifiedAsFinancingActivities", transform="abs"),
    "investing_cash_flow": _gaap("NetCashProvidedByUsedInInvestingActivities") + _ifrs("CashFlowsFromUsedInInvestingActivities"),
    "financing_cash_flow": _gaap("NetCashProvidedByUsedInFinancingActivities") + _ifrs("CashFlowsFromUsedInFinancingActivities"),
    # share counts
    "weighted_avg_shares_diluted": _gaap("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasicAndDiluted") + _ifrs("AdjustedWeightedAverageShares"),
    "shares_outstanding": _gaap("CommonStockSharesOutstanding") + [("dei:EntityCommonStockSharesOutstanding", "as_is")],
}
# the cover page's share count is as of a date a few weeks after the period end, not at the period end itself; earnings per share from continuing operations leaves out what a discontinued business earned
SEC_APPROXIMATE_SOURCES = {"dei:EntityCommonStockSharesOutstanding", "us-gaap:IncomeLossFromContinuingOperationsPerBasicShare", "us-gaap:IncomeLossFromContinuingOperationsPerDilutedShare",
                           "us-gaap:IncomeLossFromContinuingOperationsPerBasicAndDilutedShare"}
# a synthetic source field the SEC adapter builds from lines of the same filing when it has no single element for the figure: ``(groups needed, [[alternatives], ...])``. Each group is one line of the
# statements (the first alternative the filing has is taken) and the figure is the sum of the groups; it is made only when enough groups are there.
SEC_SUMS: dict[str, tuple[int, list[list[str]]]] = {
    SEC_DEBT_CURRENT: (1, [["us-gaap:ShortTermBorrowings"], ["us-gaap:CommercialPaper"], ["us-gaap:LongTermDebtCurrent", "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent"]]),
    SEC_TEMP_EQUITY: (1, [["us-gaap:TemporaryEquityCarryingAmountAttributableToParent"],
                          ["us-gaap:RedeemableNoncontrollingInterestEquityCarryingAmount", "us-gaap:RedeemableNoncontrollingInterestEquityCommonCarryingAmount",
                           "us-gaap:TemporaryEquityCarryingAmountAttributableToNoncontrollingInterest"]]),
    SEC_SGA: (2, [["us-gaap:SellingAndMarketingExpense", "us-gaap:SalesAndMarketingExpense"], ["us-gaap:GeneralAndAdministrativeExpense"]]),
    SEC_PRETAX: (2, [["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"], ["us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"]]),
    SEC_CASH_AND_INVESTMENTS: (2, [["us-gaap:CashAndCashEquivalentsAtCarryingValue", "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
                                   ["us-gaap:ShortTermInvestments", "us-gaap:MarketableSecuritiesCurrent", "us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent", "us-gaap:AvailableForSaleSecuritiesCurrent"]]),
}


def _sec_field_types() -> dict[str, str]:
    """Which kind of XBRL fact each source field is: ``duration`` (a flow over the fiscal year), ``instant`` (a balance at the period end) or ``cover`` (the front page of the filing)."""
    out: dict[str, str] = {}
    for code, options in SEC_MAP.items():
        kind = "duration" if METRICS[code].kind in FLOW_KINDS or code == "weighted_avg_shares_diluted" else "instant"
        for source, _ in options:
            if source.startswith("dei:"):
                out.setdefault(source, "cover")
            elif not source.startswith("derived:"):
                out.setdefault(source, kind)
    for name, (_, groups) in SEC_SUMS.items():
        kind = "instant" if name in (SEC_DEBT_CURRENT, SEC_CASH_AND_INVESTMENTS, SEC_TEMP_EQUITY) else "duration"
        for group in groups:
            for part in group:
                out.setdefault(part, kind)
    return out


SEC_FIELDS: dict[str, str] = _sec_field_types()

# the simulated provider speaks canonical codes already
IDENTITY_MAP: dict[str, list[tuple[str, str]]] = {code: [(code, "as_is")] for code in METRICS}

SOURCE_MAPS = {"openfilings": OPENFILINGS_MAP, "sec_edgar": SEC_MAP, "simulated": IDENTITY_MAP}
APPROXIMATE = {"openfilings": OPENFILINGS_APPROXIMATE}
APPROXIMATE_SOURCES = {"sec_edgar": SEC_APPROXIMATE_SOURCES}


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


def _flag(provider: str, code: str, source: str) -> str:
    """``derived`` for a source field a provider adapter computed from other figures of the same filing (its name starts ``derived:``), ``estimated`` for one that only approximates the metric."""
    if source.startswith("derived:"):
        return "derived"
    return "estimated" if code in APPROXIMATE.get(provider, ()) or source in APPROXIMATE_SOURCES.get(provider, ()) else "ok"


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
                out[code] = {"value": v, "source_metric": source, "flag": _flag(provider, code, source)}
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
