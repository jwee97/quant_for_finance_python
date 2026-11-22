"""Sector-aware applicability and sector figures.

A bank has no gross margin, no inventory and no current ratio; an insurer has no capital expenditure worth a yield; a REIT's net income is mostly depreciation and revaluation. A factor computed from a
statement line a company does not have, or one that means something else for its industry, must be **not applicable** (missing), not a low score: otherwise a bank ranks last on every margin and the
composite punishes it for its industry.

* :func:`family_of` assigns ``bank``, ``insurer``, ``reit`` or ``general`` from the company's sector and industry text (a person can override it with ``set_family``; a company with neither text is
  ``general`` and is reported as *unclassified*, never guessed).
* ``NOT_APPLICABLE`` lists, per family, the factors that do not apply; :func:`apply` blanks them for that family's companies and :func:`applicability` says which and why.
* Industry figures that are not lines of the canonical statements (net interest margin, CET1, non-performing loans, combined ratio, solvency, occupancy, gearing, FFO …) are read from the figures a
  person **confirmed** in the importer (``custom_observations``) as ``custom:<name>`` factors, point-in-time (public from the report's filing date), only for the families they apply to.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

BANK = ("bank", "banks", "banking", "thrift", "mortgage finance", "capital markets", "diversified financial", "consumer finance", "credit services")
INSURER = ("insurance", "insurer", "reinsurance", "assurance", "underwriting")
REIT = ("reit", "real estate investment trust", "real estate investment trusts", "equity real estate", "property trust", "mortgage reit")
FAMILIES = ("bank", "insurer", "reit", "general")

_FINANCIAL_NA = {"gross_margin", "gross_profitability", "gross_profit_to_ev", "ebitda_to_ev", "ebitda_margin", "ebit_to_ev", "fcf_yield", "fcf_margin", "fcf_conversion", "current_ratio", "quick_ratio", "cash_ratio",
                 "working_capital_to_assets", "inventory_turnover", "receivables_turnover", "cash_conversion_cycle", "capex_to_assets", "net_investment_to_assets", "net_debt_to_ebitda", "debt_to_equity",
                 "debt_to_assets", "interest_coverage", "greenblatt_roc", "altman_z_original", "altman_z_prime", "altman_z_double_prime", "roic", "owner_earnings_margin", "operating_profitability",
                 "naive_distance_to_default", "asset_turnover", "fixed_asset_turnover", "ncav_to_price", "operating_margin"}
NOT_APPLICABLE = {"bank": _FINANCIAL_NA, "insurer": _FINANCIAL_NA | {"cfo_yield"},
                  "reit": {"gross_margin", "gross_profitability", "gross_profit_to_ev", "inventory_turnover", "cash_conversion_cycle", "greenblatt_roc", "capex_to_assets", "net_investment_to_assets",
                           "fcf_conversion", "quick_ratio", "ncav_to_price", "fixed_asset_turnover"}, "general": set()}
REASON = {"bank": "banks fund themselves with deposits and debt and report interest as revenue: this line does not exist or means something else for them",
          "insurer": "insurers' revenue is premiums and investment income, their liabilities are insurance contracts: this line does not exist or means something else for them",
          "reit": "a REIT's earnings are dominated by depreciation and revaluation and it holds property, not inventory: use FFO and the sector figures"}

# industry figures read from confirmed custom observations: name -> (label, families, direction, what it is)
SECTOR_FIGURES = {
    "net_interest_margin": ("Net interest margin", ("bank",), 1, "Net interest income over average interest-earning assets."),
    "cost_to_income": ("Cost-to-income ratio", ("bank",), -1, "Operating expenses over operating income."),
    "npl_ratio": ("Non-performing loan ratio", ("bank",), -1, "Non-performing loans over gross loans."),
    "loan_loss_provisions_ratio": ("Loan-loss provisions to loans", ("bank",), -1, "Provisions for credit losses over average loans."),
    "cet1_ratio": ("CET1 capital ratio", ("bank",), 1, "Common equity tier 1 capital over risk-weighted assets."),
    "return_on_tangible_equity": ("Return on tangible equity", ("bank",), 1, "Net income over tangible equity."),
    "combined_ratio": ("Combined ratio", ("insurer",), -1, "Claims plus expenses over earned premiums (below 100% is an underwriting profit)."),
    "solvency_ratio": ("Solvency ratio", ("insurer",), 1, "Available capital over required capital."),
    "insurance_service_result_margin": ("Insurance service result margin", ("insurer",), 1, "Insurance service result over insurance revenue."),
    "ffo_per_unit": ("Funds from operations per unit", ("reit",), 1, "FFO per unit (a per-unit amount in the report's currency)."),
    "affo_per_unit": ("Adjusted FFO per unit", ("reit",), 1, "AFFO per unit."),
    "distribution_per_unit": ("Distribution per unit", ("reit",), 1, "Distribution per unit."),
    "occupancy": ("Occupancy", ("reit",), 1, "Occupied share of lettable area."),
    "gearing": ("Gearing", ("reit",), -1, "Borrowings over total assets or deposited property."),
    "net_asset_value_per_unit": ("Net asset value per unit", ("reit",), 1, "NAV per unit."),
    "research_intensity": ("R&D intensity", ("general",), 1, "Research and development over revenue."),
    "deferred_revenue_growth": ("Deferred revenue growth", ("general",), 1, "Year-on-year growth in deferred revenue."),
    "order_backlog": ("Order backlog", ("general",), 1, "Orders received and not yet delivered."),
}
STALE_DAYS = 540


def family_of(sector: str | None, industry: str | None) -> tuple[str, str]:
    """``(family, why)``: from the sector and industry text; ``general`` with why ``unclassified`` when both are empty."""
    text = f"{sector or ''} {industry or ''}".lower()
    if not text.strip():
        return "general", "unclassified"
    words = set(re.findall(r"[a-z]+", text))
    if any(k in text for k in REIT) or "reit" in words:
        return "reit", "sector/industry text names a REIT"
    if any(k in text for k in INSURER):
        return "insurer", "sector/industry text names insurance"
    if any(k in text for k in BANK):
        return "bank", "sector/industry text names banking"
    return "general", "sector/industry text names none of bank, insurer, REIT"


def families(ctx) -> pd.Series:
    if "families" not in ctx._cache:
        manual = getattr(ctx.store, "family_overrides", lambda: {})()
        ctx._cache["families"] = pd.Series({i: manual.get(i) or family_of(ctx.info.loc[i, "sector"], ctx.info.loc[i, "industry"])[0] for i in ctx.ids})
    return ctx._cache["families"]


def applicable_mask(ctx, factor: str) -> pd.Series:
    """Per company: is the factor meaningful for its family?"""
    fam = families(ctx)
    return ~fam.map(lambda f: factor in NOT_APPLICABLE.get(f, set())).astype(bool)


def apply(ctx, factor: str, raw: pd.DataFrame) -> pd.DataFrame:
    ok = applicable_mask(ctx, factor)
    keep = ok.reindex(raw.columns).fillna(True).to_numpy(bool)
    return raw.where(np.broadcast_to(keep[None, :], raw.shape))


def applicability(ctx, factor: str) -> dict:
    fam = families(ctx)
    bad = fam[~applicable_mask(ctx, factor)]
    return {"factor": factor, "not_applicable_companies": [int(i) for i in bad.index], "families": sorted(set(bad)), "reason": {f: REASON[f] for f in sorted(set(bad))},
            "unclassified_companies": int(sum(1 for i in ctx.ids if family_of(ctx.info.loc[i, "sector"], ctx.info.loc[i, "industry"])[1] == "unclassified")), "universe": int(len(fam))}


def catalogue() -> list[dict]:
    return [{"name": f"custom:{k}", "label": v[0], "families": list(v[1]), "direction": v[2], "text": v[3], "source": "figures you confirmed in the importer (custom observations)"} for k, v in SECTOR_FIGURES.items()]


def custom_panel(ctx, code: str) -> pd.DataFrame:
    """Dates by companies: the latest confirmed figure of ``code`` public on or before each date (the filing date; an estimate only when none was given), not older than 540 days. For a sector figure only the
    companies of its families have values."""
    name = code.split("custom:", 1)[-1]
    rows = ctx.store.custom_observations(ctx.ids, history=True)
    rows = [r for r in rows if r["metric_code"] in (f"custom:{name}", name)]
    out = pd.DataFrame(np.nan, index=ctx.dates, columns=ctx.ids)
    by: dict[int, list] = {}
    for r in rows:
        by.setdefault(r["security_id"], []).append(r)
    for sid, rs in by.items():
        rs.sort(key=lambda r: (r["available_from"], r["period_end"], r["vintage"]))
        avail = pd.to_datetime([r["available_from"] for r in rs]) + pd.Timedelta(days=ctx.settings.info_lag_days)
        vals = np.array([r["value"] for r in rs], float)
        ends = pd.to_datetime([r["period_end"] for r in rs])
        for k, d in enumerate(ctx.dates):
            seen = np.where(avail <= d)[0]
            if len(seen):
                j = seen[np.lexsort((np.array([rs[i]["vintage"] for i in seen]), ends[seen].asi8))[-1]]
                if (d - avail[j]).days <= STALE_DAYS:
                    out.iloc[k, out.columns.get_loc(sid)] = vals[j]
    if name in SECTOR_FIGURES:
        fam = families(ctx)
        keep = fam.isin(SECTOR_FIGURES[name][1]).reindex(out.columns).fillna(False).to_numpy(bool)
        out = out.where(np.broadcast_to(keep[None, :], out.shape))
    return out
