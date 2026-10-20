"""Derived company metrics: the ratios many studies reuse, computed once from the point-in-time tables on an annual (``fy``) or trailing-twelve-month (``ttm``) basis.

Every definition is written in terms of the canonical figures and nothing else, so it works on any provider's data and says in one place what it means. The rules that keep a ratio honest:

* **A denominator that cannot meaningfully divide gives a missing value, never a number.** A margin needs positive revenue, a return on equity needs positive equity (a company with negative equity
  has an undefined return on it, not a hugely negative one), a growth rate needs a positive base. Zero, negative and missing denominators all become NaN; the factor engine then treats the
  company as having no value, which is what it is.
* **A missing input is missing, never zero.** A ratio is computed only when every figure it uses is present.
* **Averages need both ends.** ``roe_avg`` and ``sloan_accruals`` use the average of this and last year's balance and are missing in a company's first year; ``roe`` and ``accruals_ratio`` use the closing
  balance and are not.
* **Growth is year on year on the same basis** (a year earlier by period end, found by date, not by row position), so a company with a gap in its history gets a missing growth rate, not a wrong one.

The result is persisted in ``fundamental_metrics`` with :data:`VERSION`; a change to a formula bumps the version, so old and new numbers are never mixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .pit import PeriodTables
from .schema import now_iso
from .store import FundamentalsStore

VERSION = "1"
DEFAULT_TAX_RATE = 0.25            # used for NOPAT when a company's own effective rate cannot be measured (no pre-tax profit, or a rate outside 0-45%)
MIN_INVESTED_CAPITAL = 0.05        # invested capital is floored at this share of total assets so that a company with large cash does not get an infinite return


@dataclass(frozen=True)
class Derived:
    name: str
    label: str
    family: str                    # profitability | dupont | leverage | liquidity | efficiency | cashflow | growth | investment | momentum | quality
    unit: str                      # ratio | days | currency | per_share | score
    text: str
    formula: Callable[["Ctx"], pd.Series]
    stage: int = 1                 # 1: from the statements; 2: from stage-1 metrics (changes and scores)


class Ctx:
    """The table being computed, with this year's figures and those of a year (or more) before."""

    def __init__(self, tables: PeriodTables, basis: str):
        self.tables, self.basis = tables, basis
        self.t = tables.table(basis)

    def __getitem__(self, name: str) -> pd.Series:
        if name in self.t.columns:
            return self.t[name].astype(float)
        return pd.Series(np.nan, index=self.t.index)

    def prior(self, name: str, years: int = 1) -> pd.Series:
        if name not in self.t.columns:
            return pd.Series(np.nan, index=self.t.index)
        return self.tables.prior_col(self.basis, name, years)

    def first(self, *names: str) -> pd.Series:
        """The first of several equivalent figures that is present, row by row (``cash_and_equivalents`` where ``cash`` is missing, for example)."""
        out = self[names[0]]
        for n in names[1:]:
            out = out.fillna(self[n])
        return out


# ------------------------------------------------------------------------------------------------------------------ safe arithmetic
def ratio(num: pd.Series, den: pd.Series, positive: bool = True) -> pd.Series:
    """``num / den``, missing where ``den`` is missing, zero or (with ``positive``) not positive; an infinite result is missing too."""
    den = den.where(den > 0) if positive else den.where(den != 0)
    out = num / den
    return out.where(np.isfinite(out))


def growth(cur: pd.Series, prev: pd.Series) -> pd.Series:
    """Year-on-year growth ``cur / prev - 1`` where ``prev`` is positive (a rate of change from zero or from a loss is not a growth rate)."""
    return ratio(cur, prev) - 1.0


def avg(a: pd.Series, b: pd.Series) -> pd.Series:
    """The average of a closing and an opening balance; missing unless both are positive (an average of a positive and a negative equity says nothing about either)."""
    return (a.where(a > 0) + b.where(b > 0)) / 2.0


# ------------------------------------------------------------------------------------------------------------------ the registry
DEFINITIONS: list[Derived] = []


def metric(name: str, label: str, family: str, unit: str, text: str, stage: int = 1):
    def register(fn):
        DEFINITIONS.append(Derived(name, label, family, unit, text, fn, stage))
        return fn
    return register


def _cash(c: Ctx) -> pd.Series:
    return c.first("cash_and_equivalents", "cash")


def _tax_rate(c: Ctx) -> pd.Series:
    rate = ratio(c["income_tax"], c["income_before_tax"])
    return rate.where((rate >= 0) & (rate <= 0.45)).fillna(DEFAULT_TAX_RATE)


def _invested_capital(c: Ctx) -> pd.Series:
    debt = c.first("total_debt")
    raw = debt + c["shareholders_equity"] + c["minority_interest"].fillna(0.0) - _cash(c)
    return np.maximum(raw, MIN_INVESTED_CAPITAL * c["total_assets"]).where(c["total_assets"] > 0)


def _nopat(c: Ctx) -> pd.Series:
    return c.first("operating_income", "ebit") * (1.0 - _tax_rate(c))


# --- profitability
@metric("gross_margin", "Gross margin", "profitability", "ratio", "Gross profit over revenue.")
def _gross_margin(c): return ratio(c["gross_profit"], c["revenue"])


@metric("operating_margin", "Operating margin", "profitability", "ratio", "Operating income over revenue.")
def _operating_margin(c): return ratio(c.first("operating_income", "ebit"), c["revenue"])


@metric("ebitda_margin", "EBITDA margin", "profitability", "ratio", "EBITDA over revenue.")
def _ebitda_margin(c): return ratio(c["ebitda"], c["revenue"])


@metric("net_margin", "Net margin", "profitability", "ratio", "Net income over revenue.")
def _net_margin(c): return ratio(c["net_income"], c["revenue"])


@metric("fcf_margin", "Free cash flow margin", "profitability", "ratio", "Free cash flow over revenue.")
def _fcf_margin(c): return ratio(c.first("free_cash_flow"), c["revenue"])


@metric("roe", "Return on equity", "profitability", "ratio", "Net income over closing shareholders' equity; missing when equity is not positive.")
def _roe(c): return ratio(c["net_income"], c["shareholders_equity"])


@metric("roe_avg", "Return on average equity", "profitability", "ratio", "Net income over the average of this and last year's equity (both positive).")
def _roe_avg(c): return ratio(c["net_income"], avg(c["shareholders_equity"], c.prior("shareholders_equity")))


@metric("roa", "Return on assets", "profitability", "ratio", "Net income over closing total assets.")
def _roa(c): return ratio(c["net_income"], c["total_assets"])


@metric("roa_avg", "Return on average assets", "profitability", "ratio", "Net income over the average of this and last year's assets.")
def _roa_avg(c): return ratio(c["net_income"], avg(c["total_assets"], c.prior("total_assets")))


@metric("roic", "Return on invested capital", "profitability", "ratio",
        "After-tax operating income (at the company's own tax rate, else 25%) over debt plus equity less cash, floored at 5% of assets.")
def _roic(c): return ratio(_nopat(c), _invested_capital(c))


@metric("gross_profitability", "Gross profitability", "profitability", "ratio", "Gross profit over total assets (Novy-Marx).")
def _gross_profitability(c): return ratio(c["gross_profit"], c["total_assets"])


@metric("operating_profitability", "Operating profitability", "profitability", "ratio",
        "Revenue less cost of revenue, SG&A and interest, over equity (Fama-French); missing when any of them is missing.")
def _operating_profitability(c): return ratio(c["revenue"] - c["cost_of_revenue"] - c["selling_general_admin"] - c["interest_expense"], c["shareholders_equity"])


@metric("cash_return_on_assets", "Operating cash flow to assets", "profitability", "ratio", "Operating cash flow over total assets.")
def _cash_return_on_assets(c): return ratio(c["operating_cash_flow"], c["total_assets"])


# --- DuPont
@metric("tax_burden", "Tax burden", "dupont", "ratio", "Net income over income before tax (the share of profit the company keeps after tax).")
def _tax_burden(c): return ratio(c["net_income"], c["income_before_tax"])


@metric("interest_burden", "Interest burden", "dupont", "ratio", "Income before tax over operating income (the share left after interest).")
def _interest_burden(c): return ratio(c["income_before_tax"], c.first("operating_income", "ebit"))


@metric("asset_turnover", "Asset turnover", "dupont", "ratio", "Revenue over closing total assets.")
def _asset_turnover(c): return ratio(c["revenue"], c["total_assets"])


@metric("equity_multiplier", "Equity multiplier", "dupont", "ratio", "Total assets over equity: how much of the assets the owners did not pay for.")
def _equity_multiplier(c): return ratio(c["total_assets"], c["shareholders_equity"])


# --- leverage
@metric("debt_to_equity", "Debt to equity", "leverage", "ratio", "Total debt over equity; missing when equity is not positive.")
def _debt_to_equity(c): return ratio(c["total_debt"], c["shareholders_equity"])


@metric("debt_to_assets", "Debt to assets", "leverage", "ratio", "Total debt over total assets.")
def _debt_to_assets(c): return ratio(c["total_debt"], c["total_assets"])


@metric("liabilities_to_assets", "Liabilities to assets", "leverage", "ratio", "Total liabilities over total assets.")
def _liabilities_to_assets(c): return ratio(c["total_liabilities"], c["total_assets"])


@metric("equity_ratio", "Equity ratio", "leverage", "ratio", "Equity over total assets.")
def _equity_ratio(c): return ratio(c["shareholders_equity"], c["total_assets"])


@metric("net_debt", "Net debt", "leverage", "currency", "Total debt less cash (negative: more cash than debt).")
def _net_debt(c): return c["total_debt"] - _cash(c)


@metric("net_debt_to_ebitda", "Net debt to EBITDA", "leverage", "ratio", "Net debt over EBITDA; missing when EBITDA is not positive.")
def _net_debt_to_ebitda(c): return ratio(c["total_debt"] - _cash(c), c["ebitda"])


@metric("interest_coverage", "Interest coverage", "leverage", "ratio", "Operating income over interest expense; missing when the company pays no interest.")
def _interest_coverage(c): return ratio(c.first("operating_income", "ebit"), c["interest_expense"])


# --- liquidity
@metric("current_ratio", "Current ratio", "liquidity", "ratio", "Current assets over current liabilities.")
def _current_ratio(c): return ratio(c["current_assets"], c["current_liabilities"])


@metric("quick_ratio", "Quick ratio", "liquidity", "ratio", "Current assets less inventory, over current liabilities.")
def _quick_ratio(c): return ratio(c["current_assets"] - c["inventory"], c["current_liabilities"])


@metric("cash_ratio", "Cash ratio", "liquidity", "ratio", "Cash over current liabilities.")
def _cash_ratio(c): return ratio(_cash(c), c["current_liabilities"])


@metric("working_capital", "Working capital", "liquidity", "currency", "Current assets less current liabilities.")
def _working_capital(c): return c["current_assets"] - c["current_liabilities"]


@metric("working_capital_to_assets", "Working capital to assets", "liquidity", "ratio", "Working capital over total assets.")
def _working_capital_to_assets(c): return ratio(c["current_assets"] - c["current_liabilities"], c["total_assets"])


# --- efficiency
@metric("fixed_asset_turnover", "Fixed-asset turnover", "efficiency", "ratio", "Revenue over net property, plant and equipment.")
def _fixed_asset_turnover(c): return ratio(c["revenue"], c["ppe"])


@metric("inventory_turnover", "Inventory turnover", "efficiency", "ratio", "Cost of revenue over inventory.")
def _inventory_turnover(c): return ratio(c["cost_of_revenue"], c["inventory"])


@metric("receivables_turnover", "Receivables turnover", "efficiency", "ratio", "Revenue over receivables.")
def _receivables_turnover(c): return ratio(c["revenue"], c["accounts_receivable"])


@metric("payables_turnover", "Payables turnover", "efficiency", "ratio", "Cost of revenue over payables.")
def _payables_turnover(c): return ratio(c["cost_of_revenue"], c["accounts_payable"])


@metric("days_sales_outstanding", "Days sales outstanding", "efficiency", "days", "365 days over receivables turnover.")
def _dso(c): return ratio(pd.Series(365.0, index=c.t.index), _receivables_turnover(c))


@metric("days_inventory", "Days inventory", "efficiency", "days", "365 days over inventory turnover.")
def _dio(c): return ratio(pd.Series(365.0, index=c.t.index), _inventory_turnover(c))


@metric("days_payables", "Days payables", "efficiency", "days", "365 days over payables turnover.")
def _dpo(c): return ratio(pd.Series(365.0, index=c.t.index), _payables_turnover(c))


@metric("cash_conversion_cycle", "Cash conversion cycle", "efficiency", "days", "Days sales outstanding plus days inventory less days payables.")
def _ccc(c): return _dso(c) + _dio(c) - _dpo(c)


# --- cash flow and earnings quality
@metric("fcf", "Free cash flow", "cashflow", "currency", "Operating cash flow less capital expenditure (the provider's figure when it gives one).")
def _fcf(c): return c.first("free_cash_flow").fillna(c["operating_cash_flow"] - c["capital_expenditure"])


@metric("fcf_conversion", "Free cash flow conversion", "cashflow", "ratio", "Free cash flow over net income; missing when net income is not positive.")
def _fcf_conversion(c): return ratio(_fcf(c), c["net_income"])


@metric("cfo_to_net_income", "Operating cash flow to net income", "cashflow", "ratio", "Operating cash flow over net income; missing when net income is not positive.")
def _cfo_to_ni(c): return ratio(c["operating_cash_flow"], c["net_income"])


@metric("accruals_ratio", "Accruals ratio", "cashflow", "ratio", "Net income less operating cash flow, over closing assets: the share of profit that is not cash (high is a warning).")
def _accruals(c): return ratio(c["net_income"] - c["operating_cash_flow"], c["total_assets"])


@metric("sloan_accruals", "Sloan accruals", "cashflow", "ratio", "Net income less operating cash flow, over average assets (Sloan 1996).")
def _sloan(c): return ratio(c["net_income"] - c["operating_cash_flow"], avg(c["total_assets"], c.prior("total_assets")))


@metric("capex_to_revenue", "Capex to revenue", "cashflow", "ratio", "Capital expenditure over revenue.")
def _capex_to_revenue(c): return ratio(c["capital_expenditure"], c["revenue"])


@metric("capex_to_depreciation", "Capex to depreciation", "cashflow", "ratio", "Capital expenditure over depreciation and amortisation: above 1 the company is growing its asset base.")
def _capex_to_da(c): return ratio(c["capital_expenditure"], c["depreciation_amortization"])


@metric("rd_intensity", "R&D intensity", "cashflow", "ratio", "Research and development over revenue (missing, not zero, where the company does not report it).")
def _rd_intensity(c): return ratio(c["research_and_development"], c["revenue"])


@metric("sga_intensity", "SG&A intensity", "cashflow", "ratio", "Selling, general and administrative expense over revenue.")
def _sga_intensity(c): return ratio(c["selling_general_admin"], c["revenue"])


@metric("sbc_to_revenue", "Stock compensation to revenue", "cashflow", "ratio", "Stock-based compensation over revenue.")
def _sbc_to_revenue(c): return ratio(c["stock_based_compensation"], c["revenue"])


@metric("dividend_payout", "Dividend payout", "cashflow", "ratio", "Dividends paid over net income; missing when net income is not positive.")
def _payout(c): return ratio(c["dividends_paid"], c["net_income"])


# --- growth
def _growth_of(name: str, label: str, source: str, text: str):
    metric(name, label, "growth", "ratio", text)(lambda c, s=source: growth(c[s], c.prior(s)))


_growth_of("revenue_growth", "Revenue growth", "revenue", "Year-on-year growth in revenue.")
_growth_of("gross_profit_growth", "Gross profit growth", "gross_profit", "Year-on-year growth in gross profit.")
_growth_of("operating_income_growth", "Operating income growth", "operating_income", "Year-on-year growth in operating income (missing from a loss).")
_growth_of("ebitda_growth", "EBITDA growth", "ebitda", "Year-on-year growth in EBITDA.")
_growth_of("net_income_growth", "Net income growth", "net_income", "Year-on-year growth in net income (missing from a loss).")
_growth_of("eps_growth", "EPS growth", "eps_diluted", "Year-on-year growth in diluted earnings per share (missing from a loss).")
_growth_of("ocf_growth", "Operating cash flow growth", "operating_cash_flow", "Year-on-year growth in operating cash flow.")
_growth_of("asset_growth", "Asset growth", "total_assets", "Year-on-year growth in total assets (the investment effect: high growth tends to precede low returns).")
_growth_of("equity_growth", "Equity growth", "shareholders_equity", "Year-on-year growth in equity.")
_growth_of("capex_growth", "Capex growth", "capital_expenditure", "Year-on-year growth in capital expenditure.")


@metric("share_growth", "Share count growth", "growth", "ratio", "Year-on-year change in shares outstanding: net issuance (positive) or net buyback (negative).")
def _share_growth(c): return growth(c.first("shares_outstanding", "weighted_avg_shares_diluted"), c.prior("shares_outstanding").fillna(c.prior("weighted_avg_shares_diluted")))


@metric("revenue_cagr_3y", "Revenue growth, 3-year", "growth", "ratio", "Compound annual revenue growth over three years.")
def _cagr3(c):
    out = ratio(c["revenue"], c.prior("revenue", 3)) ** (1.0 / 3.0) - 1.0
    return out.where(np.isfinite(out))


# --- investment
@metric("capex_to_assets", "Capex to assets", "investment", "ratio", "Capital expenditure over total assets.")
def _capex_to_assets(c): return ratio(c["capital_expenditure"], c["total_assets"])


@metric("net_investment_to_assets", "Net investment to assets", "investment", "ratio", "Capital expenditure less depreciation, over total assets.")
def _net_investment(c): return ratio(c["capital_expenditure"] - c["depreciation_amortization"], c["total_assets"])


# --- stage 2: changes in the stage-1 metrics (fundamental momentum) and their second differences
def _change_of(source: str, label: str, family: str = "momentum"):
    name = f"d_{source}"
    metric(name, f"Change in {label.lower()}", family, "ratio", f"{label} this year less {label.lower()} a year earlier.", stage=2)(lambda c, s=source: c[s] - c.prior(s))


for _source, _label in (("gross_margin", "Gross margin"), ("operating_margin", "Operating margin"), ("net_margin", "Net margin"), ("fcf_margin", "Free cash flow margin"),
                        ("roe", "Return on equity"), ("roa", "Return on assets"), ("roic", "Return on invested capital"), ("asset_turnover", "Asset turnover"),
                        ("current_ratio", "Current ratio"), ("debt_to_assets", "Debt to assets"), ("revenue_growth", "Revenue growth"), ("interest_coverage", "Interest coverage"),
                        ("gross_profitability", "Gross profitability"), ("accruals_ratio", "Accruals ratio")):
    _change_of(_source, _label)


@metric("d2_operating_margin", "Acceleration of operating margin", "momentum", "ratio", "Change in the change in operating margin.", stage=3)
def _d2_operating_margin(c): return c["d_operating_margin"] - c.prior("d_operating_margin")


@metric("d2_roic", "Acceleration of ROIC", "momentum", "ratio", "Change in the change in return on invested capital.", stage=3)
def _d2_roic(c): return c["d_roic"] - c.prior("d_roic")


@metric("revenue_growth_acceleration", "Revenue growth acceleration", "growth", "ratio", "This year's revenue growth less last year's.", stage=3)
def _growth_acceleration(c): return c["d_revenue_growth"]


# ------------------------------------------------------------------------------------------------------------------ computing and keeping
def register_scores(definitions: list[Derived]) -> None:
    """Add metrics defined elsewhere (the Piotroski and Beneish scores of :mod:`analytics`) to the registry."""
    known = {d.name for d in DEFINITIONS}
    DEFINITIONS.extend(d for d in definitions if d.name not in known)


def compute(tables: PeriodTables, basis: str = "fy") -> pd.DataFrame:
    """Every derived metric for every row of the ``basis`` table, index-aligned with it. The columns are also attached to the table, so later stages (and the factor engine) can look back a year."""
    from . import analytics  # noqa: F401  (registers the Piotroski and anomaly scores)
    table = tables.table(basis)
    if table.empty:
        return pd.DataFrame(index=table.index, columns=[d.name for d in DEFINITIONS], dtype=float)
    out: dict[str, pd.Series] = {}
    for stage in sorted({d.stage for d in DEFINITIONS}):
        ctx = Ctx(tables, basis)
        batch = {}
        for d in DEFINITIONS:
            if d.stage != stage:
                continue
            value = d.formula(ctx)
            batch[d.name] = pd.Series(np.asarray(value, dtype=float), index=table.index).replace([np.inf, -np.inf], np.nan)
        out.update(batch)
        tables.add_columns(basis, batch)
    return pd.DataFrame(out, index=table.index)


def catalogue() -> list[dict]:
    from . import analytics  # noqa: F401  (registers the Piotroski and anomaly scores)
    return [{"name": d.name, "label": d.label, "family": d.family, "unit": d.unit, "text": d.text} for d in DEFINITIONS]


def persist(store: FundamentalsStore, tables: PeriodTables, bases: tuple = ("fy", "ttm"), version: str = VERSION) -> int:
    """Write the derived metrics of every period to ``fundamental_metrics`` (missing values are not written). Returns the number of rows."""
    stamp, written = now_iso(), 0
    with store.db.transaction() as conn:
        for basis in bases:
            table = tables.table(basis)
            if table.empty:
                continue
            values = compute(tables, basis)
            long = values.assign(security_id=table["security_id"].to_numpy(), period_id=table["period_id"].to_numpy()).melt(id_vars=["security_id", "period_id"], var_name="metric_name",
                                                                                                                         value_name="metric_value").dropna(subset=["metric_value"])
            rows = [(int(s), int(p), basis, str(n), float(v), version, stamp) for s, p, n, v in zip(long["security_id"], long["period_id"], long["metric_name"], long["metric_value"])]
            conn.executemany("INSERT OR REPLACE INTO fundamental_metrics (security_id, financial_period_id, basis, metric_name, metric_value, calculation_version, calculated_at) "
                             "VALUES (?,?,?,?,?,?,?)", rows)
            written += len(rows)
    return written
