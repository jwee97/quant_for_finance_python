"""Mapping the element names of other countries' taxonomies, and Brazil's account codes, onto the application's canonical lines.

* **Japan (EDINET).** Japanese GAAP filings tag with ``jppfs_cor`` (NetSales, OperatingIncome, Assets...), IFRS filings with ``jpigp_cor`` (the same names with an ``IFRS`` suffix: RevenueIFRS, AssetsIFRS), and the
  cover facts with ``jpdei_cor``. The names below are those of the published taxonomy's main statement elements; a company's own extension elements are not mapped and are listed as unmapped. A line
  that is only an approximation of the canonical one (net assets for the owners' equity, cash and deposits for cash) is flagged ``estimated``.
* **Korea (OpenDART)** and **Europe (ESEF)** tag with IFRS (``ifrs-full``), which the SEC mapping already covers.
* **Brazil (CVM)** publishes the statements as CSV with a numbered chart of accounts (``1`` total assets, ``3.01`` net revenue...), mapped here.
"""

from __future__ import annotations

# canonical line -> element local names (without the IFRS suffix), tried in order; True marks an approximation
JP_ELEMENTS: dict[str, list[tuple[str, bool]]] = {
    "revenue": [("NetSales", False), ("Revenue", False), ("OperatingRevenue1", False), ("OperatingRevenues", False)],
    "cost_of_revenue": [("CostOfSales", False)],
    "gross_profit": [("GrossProfit", False)],
    "selling_general_admin": [("SellingGeneralAndAdministrativeExpenses", False)],
    "operating_income": [("OperatingIncome", False), ("OperatingProfitLoss", False)],
    "income_before_tax": [("IncomeBeforeIncomeTaxes", False), ("ProfitLossBeforeTax", False)],
    "income_tax": [("IncomeTaxes", False), ("IncomeTaxExpense", False)],
    "net_income": [("ProfitLossAttributableToOwnersOfParent", False), ("ProfitLoss", False)],
    "total_assets": [("Assets", False)],
    "current_assets": [("CurrentAssets", False)],
    "current_liabilities": [("CurrentLiabilities", False)],
    "total_liabilities": [("Liabilities", False)],
    "shareholders_equity": [("EquityAttributableToOwnersOfParent", False), ("ShareholdersEquity", False), ("NetAssets", True), ("Equity", True)],
    "minority_interest": [("NonControllingInterests", False), ("MinorityInterests", False)],
    "cash_and_equivalents": [("CashAndCashEquivalents", False), ("CashAndDeposits", True)],
    "accounts_receivable": [("NotesAndAccountsReceivableTrade", False), ("TradeAndOtherReceivables", False)],
    "inventory": [("Inventories", False)],
    "retained_earnings": [("RetainedEarnings", False)],
    "operating_cash_flow": [("NetCashProvidedByUsedInOperatingActivities", False)],
    "investing_cash_flow": [("NetCashProvidedByUsedInInvestmentActivities", False), ("NetCashProvidedByUsedInInvestingActivities", False)],
    "financing_cash_flow": [("NetCashProvidedByUsedInFinancingActivities", False)],
    "depreciation_amortization": [("DepreciationAndAmortizationOpeCF", False), ("DepreciationAndAmortization", False)],
}
JP_ABS = {"capital_expenditure": [("PurchaseOfPropertyPlantAndEquipmentInvCF", False), ("PurchaseOfPropertyPlantAndEquipment", False)], "dividends_paid": [("CashDividendsPaidFinCF", False), ("DividendsPaid", False)]}
JP_PREFIXES = ("jppfs_cor:", "jpigp_cor:")


def jp_values(fields: dict[str, float]) -> tuple[dict[str, dict], list[str]]:
    """``({line: {value, source_metric, flag}}, unmapped)`` from ``{"jppfs_cor:NetSales": ...}``-style fields."""
    local = {}
    for f, v in fields.items():
        if f.startswith(JP_PREFIXES):
            name = f.split(":", 1)[1]
            local.setdefault(name[:-4] if name.endswith("IFRS") else name, (f, v))
    out: dict[str, dict] = {}
    used: set[str] = set()
    for table, take_abs in ((JP_ELEMENTS, False), (JP_ABS, True)):
        for code, options in table.items():
            for name, approx in options:
                if name in local:
                    f, v = local[name]
                    out[code] = {"value": abs(v) if take_abs else v, "source_metric": f, "flag": "estimated" if approx else "ok"}
                    used.add(f)
                    break
    return out, sorted(f for f in fields if f.startswith(JP_PREFIXES) and f not in used)[:80]


# Brazil: CVM's chart of accounts for consolidated or individual statements (CD_CONTA). Net revenue, costs and results are in the DRE; assets in the BPA; liabilities and equity in the BPP; cash flow (indirect) in DFC_MD.
CVM_ACCOUNTS: dict[str, dict[str, str]] = {
    "DRE": {"3.01": "revenue", "3.02": "cost_of_revenue", "3.03": "gross_profit", "3.05": "operating_income", "3.07": "income_before_tax", "3.11": "net_income"},
    "BPA": {"1": "total_assets", "1.01": "current_assets", "1.01.01": "cash_and_equivalents", "1.01.03": "accounts_receivable", "1.01.04": "inventory"},
    "BPP": {"2.01": "current_liabilities", "2.03": "shareholders_equity"},
    "DFC_MD": {"6.01": "operating_cash_flow", "6.02": "investing_cash_flow", "6.03": "financing_cash_flow"},
}
CVM_NET_INCOME_ALT = {"3.11": "net_income"}                                                                  # "Lucro/Prejuízo Consolidado do Período" (includes minorities)
