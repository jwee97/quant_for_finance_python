"""The factor library: named, documented definitions of the signals fundamental research tests, each a function of the research context that returns a panel (dates by securities) of raw values.

Families (the spec's list): value, quality, profitability, growth, investment, leverage, liquidity, efficiency, earnings quality, fundamental momentum, price momentum, risk, size, and composites.

Every factor carries a **direction**: +1 when theory says a higher value should earn more, -1 when a lower one should (asset growth, accruals, leverage, volatility). The direction is the *prior*; a
study may flip it, but picking the sign that happened to work is a choice made on the data and is counted as a second trial by the discovery engine.

Rules the definitions follow, so that a number means one thing:

* Value factors are **yields** (earnings, cash flow, sales, book or payout over price or enterprise value), never multiples, so that a loss or a tiny denominator does not blow up a ratio. A yield is
  missing where its denominator is not positive: negative book value, negative enterprise value.
* An amount set against a price (a yield) is converted from the reporting to the trading currency at the exchange rate of the date; a ratio of one company's own figures is never converted.
* A missing input is a missing factor; nothing is replaced by zero. Combining factors is :mod:`combine`'s job and says how gaps are handled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .errors import FundamentalsError
from .research import ResearchContext

FAMILIES = ("value", "quality", "profitability", "growth", "investment", "leverage", "liquidity", "efficiency", "earnings_quality", "fundamental_momentum", "price_momentum", "risk", "size")


@dataclass(frozen=True)
class FactorDef:
    name: str
    label: str
    family: str
    direction: int                   # +1: higher should earn more; -1: lower should
    needs: tuple                     # statement figures or derived metrics it reads (an empty tuple: prices only)
    fn: Callable[[ResearchContext], pd.DataFrame]
    text: str


FACTORS: dict[str, FactorDef] = {}


def factor(name: str, label: str, family: str, direction: int, needs: tuple, text: str):
    def register(fn):
        if name in FACTORS:
            raise ValueError(f"factor {name!r} is defined twice")
        FACTORS[name] = FactorDef(name, label, family, direction, tuple(needs), fn, text)
        return fn
    return register


def _positive(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.where(panel > 0)


def _per_price(ctx: ResearchContext, name: str, basis: str = "ttm") -> pd.DataFrame:
    """An amount over market capitalisation, the amount converted to the trading currency first."""
    return ctx.to_trading(ctx.panel(name, basis)) / _positive(ctx.market_cap())


def _per_ev(ctx: ResearchContext, name: str) -> pd.DataFrame:
    return ctx.to_trading(ctx.panel(name)) / _positive(ctx.enterprise_value())


def _metric(name: str, label: str, family: str, direction: int, text: str, source: str | None = None):
    """A factor that is a derived metric, as known on the date."""
    metric = source or name
    factor(name, label, family, direction, (metric,), text)(lambda ctx, m=metric: ctx.panel(m))


# ------------------------------------------------------------------------------------------------------------------ value
@factor("earnings_yield", "Earnings yield", "value", 1, ("net_income",), "Trailing net income over market capitalisation.")
def _earnings_yield(ctx): return _per_price(ctx, "net_income")


@factor("fcf_yield", "Free cash flow yield", "value", 1, ("fcf",), "Trailing free cash flow over market capitalisation.")
def _fcf_yield(ctx): return _per_price(ctx, "fcf")


@factor("cfo_yield", "Operating cash flow yield", "value", 1, ("operating_cash_flow",), "Trailing operating cash flow over market capitalisation.")
def _cfo_yield(ctx): return _per_price(ctx, "operating_cash_flow")


@factor("sales_to_price", "Sales to price", "value", 1, ("revenue",), "Trailing revenue over market capitalisation.")
def _sales_to_price(ctx): return _per_price(ctx, "revenue")


@factor("book_to_price", "Book to price", "value", 1, ("shareholders_equity",), "Shareholders' equity over market capitalisation; missing when equity is not positive.")
def _book_to_price(ctx): return _per_price(ctx, "shareholders_equity").where(ctx.panel("shareholders_equity") > 0)


@factor("ebit_to_ev", "EBIT to enterprise value", "value", 1, ("operating_income",), "Trailing operating income over enterprise value; missing when the enterprise value is not positive.")
def _ebit_to_ev(ctx): return _per_ev(ctx, "operating_income")


@factor("ebitda_to_ev", "EBITDA to enterprise value", "value", 1, ("ebitda",), "Trailing EBITDA over enterprise value.")
def _ebitda_to_ev(ctx): return _per_ev(ctx, "ebitda")


@factor("gross_profit_to_ev", "Gross profit to enterprise value", "value", 1, ("gross_profit",), "Trailing gross profit over enterprise value.")
def _gp_to_ev(ctx): return _per_ev(ctx, "gross_profit")


@factor("dividend_yield", "Dividend yield", "value", 1, ("dividends_paid",), "Trailing dividends paid over market capitalisation.")
def _dividend_yield(ctx): return _per_price(ctx, "dividends_paid")


@factor("buyback_yield", "Buyback yield", "value", 1, ("share_repurchases",), "Trailing share repurchases over market capitalisation.")
def _buyback_yield(ctx): return _per_price(ctx, "share_repurchases")


@factor("shareholder_yield", "Shareholder yield", "value", 1, ("dividends_paid", "share_repurchases"), "Dividends plus buybacks over market capitalisation; both must be known.")
def _shareholder_yield(ctx):
    paid = ctx.panel("dividends_paid") + ctx.panel("share_repurchases")
    return ctx.to_trading(paid) / _positive(ctx.market_cap())


# ------------------------------------------------------------------------------------------------------------------ quality and profitability
_metric("roe", "Return on equity", "quality", 1, "Net income over closing equity.")
_metric("roic", "Return on invested capital", "quality", 1, "After-tax operating income over invested capital.")
_metric("gross_profitability", "Gross profitability", "quality", 1, "Gross profit over total assets (Novy-Marx).")
_metric("operating_profitability", "Operating profitability", "quality", 1, "Revenue less costs, SG&A and interest over equity (Fama-French).")
_metric("piotroski_f", "Piotroski F-score", "quality", 1, "Nine signs of improving financial health.")
_metric("gross_margin", "Gross margin", "profitability", 1, "Gross profit over revenue.")
_metric("operating_margin", "Operating margin", "profitability", 1, "Operating income over revenue.")
_metric("net_margin", "Net margin", "profitability", 1, "Net income over revenue.")
_metric("ebitda_margin", "EBITDA margin", "profitability", 1, "EBITDA over revenue.")
_metric("fcf_margin", "Free cash flow margin", "profitability", 1, "Free cash flow over revenue.")
_metric("roa", "Return on assets", "profitability", 1, "Net income over closing assets.")
_metric("cash_return_on_assets", "Operating cash flow to assets", "profitability", 1, "Operating cash flow over closing assets.")


@factor("margin_stability", "Margin stability", "quality", 1, ("operating_margin",), "Minus the standard deviation of the operating margin over the last five years of monthly observations.")
def _margin_stability(ctx):
    return -ctx.panel("operating_margin").rolling(60, min_periods=36).std()


@factor("earnings_stability", "Earnings stability", "quality", 1, ("roa",), "Minus the standard deviation of the return on assets over the last five years of monthly observations.")
def _earnings_stability(ctx):
    return -ctx.panel("roa").rolling(60, min_periods=36).std()


# ------------------------------------------------------------------------------------------------------------------ growth
for _n, _l, _t in (("revenue_growth", "Revenue growth", "Year-on-year growth in trailing revenue."), ("gross_profit_growth", "Gross profit growth", "Year-on-year growth in gross profit."),
                   ("operating_income_growth", "Operating income growth", "Year-on-year growth in operating income."), ("net_income_growth", "Net income growth", "Year-on-year growth in net income."),
                   ("eps_growth", "EPS growth", "Year-on-year growth in earnings per share."), ("ocf_growth", "Operating cash flow growth", "Year-on-year growth in operating cash flow."),
                   ("revenue_cagr_3y", "Revenue growth, 3 years", "Three-year compound growth in revenue."),
                   ("revenue_growth_acceleration", "Revenue growth acceleration", "This year's revenue growth less last year's.")):
    _metric(_n, _l, "growth", 1, _t)

# ------------------------------------------------------------------------------------------------------------------ investment
_metric("asset_growth", "Asset growth", "investment", -1, "Year-on-year growth in total assets: firms that grow their balance sheet fastest tend to earn less afterwards.")
_metric("capex_to_assets", "Capex to assets", "investment", -1, "Capital expenditure over total assets.")
_metric("net_investment_to_assets", "Net investment to assets", "investment", -1, "Capital expenditure less depreciation over total assets.")
_metric("share_growth", "Net share issuance", "investment", -1, "Year-on-year growth in the share count: issuers tend to underperform, repurchasers to outperform.")

# ------------------------------------------------------------------------------------------------------------------ leverage and liquidity
_metric("debt_to_equity", "Debt to equity", "leverage", -1, "Total debt over equity (missing when equity is not positive).")
_metric("debt_to_assets", "Debt to assets", "leverage", -1, "Total debt over total assets.")
_metric("net_debt_to_ebitda", "Net debt to EBITDA", "leverage", -1, "Net debt over EBITDA (missing when EBITDA is not positive).")
_metric("interest_coverage", "Interest coverage", "leverage", 1, "Operating income over interest expense.")
_metric("equity_ratio", "Equity ratio", "leverage", 1, "Equity over total assets.")
_metric("current_ratio", "Current ratio", "liquidity", 1, "Current assets over current liabilities.")
_metric("quick_ratio", "Quick ratio", "liquidity", 1, "Current assets less inventory over current liabilities.")
_metric("cash_ratio", "Cash ratio", "liquidity", 1, "Cash over current liabilities.")
_metric("working_capital_to_assets", "Working capital to assets", "liquidity", 1, "Working capital over total assets.")

# ------------------------------------------------------------------------------------------------------------------ efficiency
_metric("asset_turnover", "Asset turnover", "efficiency", 1, "Revenue over total assets.")
_metric("fixed_asset_turnover", "Fixed-asset turnover", "efficiency", 1, "Revenue over net property, plant and equipment.")
_metric("inventory_turnover", "Inventory turnover", "efficiency", 1, "Cost of revenue over inventory.")
_metric("receivables_turnover", "Receivables turnover", "efficiency", 1, "Revenue over receivables.")
_metric("cash_conversion_cycle", "Cash conversion cycle", "efficiency", -1, "Days of sales outstanding plus days of inventory less days of payables.")

# ------------------------------------------------------------------------------------------------------------------ earnings quality
_metric("accruals_ratio", "Accruals ratio", "earnings_quality", -1, "Net income less operating cash flow over closing assets: profit that is not cash.")
_metric("sloan_accruals", "Sloan accruals", "earnings_quality", -1, "Net income less operating cash flow over average assets (Sloan 1996).")
_metric("cfo_to_net_income", "Operating cash flow to net income", "earnings_quality", 1, "Operating cash flow over net income.")
_metric("fcf_conversion", "Free cash flow conversion", "earnings_quality", 1, "Free cash flow over net income.")
_metric("accounting_anomaly_score", "Accounting anomaly score", "earnings_quality", -1, "Beneish-type score of how unusual the year's accounting changes look (a screen, not evidence of fraud).")

# ------------------------------------------------------------------------------------------------------------------ fundamental momentum
for _n, _l in (("d_gross_margin", "Change in gross margin"), ("d_operating_margin", "Change in operating margin"), ("d_net_margin", "Change in net margin"),
               ("d_fcf_margin", "Change in free cash flow margin"), ("d_roe", "Change in return on equity"), ("d_roa", "Change in return on assets"), ("d_roic", "Change in ROIC"),
               ("d_asset_turnover", "Change in asset turnover"), ("d_current_ratio", "Change in current ratio"), ("d_gross_profitability", "Change in gross profitability"),
               ("d2_operating_margin", "Acceleration of operating margin"), ("d2_roic", "Acceleration of ROIC")):
    _metric(_n, _l, "fundamental_momentum", 1, f"{_l}: this year's less last year's (second differences for acceleration).")
_metric("d_debt_to_assets", "Change in leverage", "fundamental_momentum", -1, "Debt over assets this year less last year's.")
_metric("d_accruals_ratio", "Change in accruals", "fundamental_momentum", -1, "Accruals ratio this year less last year's.")


# ------------------------------------------------------------------------------------------------------------------ price
@factor("mom_12_1", "Momentum, 12 minus 1 months", "price_momentum", 1, (), "Return from twelve months ago to one month ago (the most recent month is left out: it tends to reverse).")
def _mom_12_1(ctx): return ctx.past(252, 21)


@factor("mom_6_1", "Momentum, 6 minus 1 months", "price_momentum", 1, (), "Return from six months ago to one month ago.")
def _mom_6_1(ctx): return ctx.past(126, 21)


@factor("mom_3_1", "Momentum, 3 minus 1 months", "price_momentum", 1, (), "Return from three months ago to one month ago.")
def _mom_3_1(ctx): return ctx.past(63, 21)


@factor("reversal_1m", "Short-term reversal", "price_momentum", -1, (), "The last month's return: last month's winners tend to give some back.")
def _reversal(ctx): return ctx.past(21, 0)


@factor("high_52w", "Closeness to the 52-week high", "price_momentum", 1, (), "Price over its highest price of the last year.")
def _high_52w(ctx):
    high = ctx.adj.rolling(252, min_periods=126).max().iloc[ctx.pos].set_axis(ctx.dates)
    last = ctx.adj.iloc[ctx.pos].set_axis(ctx.dates)
    return last / high


# ------------------------------------------------------------------------------------------------------------------ risk and size
@factor("volatility_63", "Volatility, 3 months", "risk", -1, (), "Annualised standard deviation of daily returns over the last three months.")
def _vol63(ctx): return ctx.volatility(63)


@factor("volatility_252", "Volatility, 12 months", "risk", -1, (), "Annualised standard deviation of daily returns over the last year.")
def _vol252(ctx): return ctx.volatility(252)


@factor("beta", "Market beta", "risk", -1, (), "Sensitivity to the equal-weighted market over the last year.")
def _beta(ctx): return ctx.beta(252)


@factor("idiosyncratic_volatility", "Idiosyncratic volatility", "risk", -1, (), "Volatility left after the market: the square root of total variance less beta squared times market variance.")
def _idio(ctx):
    total, beta = ctx.volatility(252) ** 2, ctx.beta(252)
    market = ctx.market_return().rolling(252, min_periods=126).std().iloc[ctx.pos].set_axis(ctx.dates) * np.sqrt(252.0)
    return np.sqrt((total - beta ** 2 * (market ** 2).to_numpy()[:, None]).clip(lower=0.0))


@factor("log_market_cap", "Size", "size", -1, (), "Natural log of market capitalisation in US dollars: the small-cap premium says smaller should earn more.")
def _size(ctx): return np.log(_positive(ctx.market_cap_usd()))


# ------------------------------------------------------------------------------------------------------------------ the registry
def names(family: str | None = None) -> list[str]:
    return [n for n, f in FACTORS.items() if family in (None, f.family)]


def get(name: str) -> FactorDef:
    try:
        return FACTORS[name]
    except KeyError:
        raise FundamentalsError(f"no factor called {name!r}; see the factor catalogue") from None


def catalogue() -> list[dict]:
    return [{"name": f.name, "label": f.label, "family": f.family, "direction": f.direction, "needs": list(f.needs), "text": f.text} for f in FACTORS.values()]


def available(ctx: ResearchContext) -> dict[str, bool]:
    """Which factors the data can support: every figure the factor reads must be in the tables."""
    return {n: all(ctx.has(need) for need in f.needs) for n, f in FACTORS.items()}


def compute(ctx: ResearchContext, name: str) -> pd.DataFrame:
    """The raw panel of factor ``name`` (cached on the context); infinities become missing values."""
    key = ("factor", name)
    if key not in ctx._cache:
        f = get(name)
        for need in f.needs:
            if not ctx.has(need):
                raise FundamentalsError(f"factor {name!r} needs {need!r}, which the database does not have for this universe")
        raw = f.fn(ctx)
        ctx._cache[key] = raw.replace([np.inf, -np.inf], np.nan).reindex(index=ctx.dates, columns=ctx.ids)
    return ctx._cache[key]


def coverage(ctx: ResearchContext, name: str) -> float:
    """The share of investable (date, company) cells that have a value."""
    mask = ctx.investable()
    return float(compute(ctx, name).where(mask).notna().to_numpy().sum() / max(int(mask.to_numpy().sum()), 1))


def composite(ctx: ResearchContext, parts: list[tuple[str, float]], min_parts: int | None = None) -> pd.DataFrame:
    """A weighted sum of standardised factors, each signed by its direction (so that a high composite is "good" on every part). A company is scored only where at least ``min_parts`` of the parts
    exist (by default every part); the weights are renormalised over the parts it has, so a missing part is left out of its score, never counted as zero."""
    from .transforms import apply
    zs, ws = [], []
    for name, weight in parts:
        f = get(name)
        z = apply(compute(ctx, name), [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx) * (1.0 if f.direction >= 0 else -1.0)
        zs.append(z)
        ws.append(float(weight))
    stack = np.stack([z.to_numpy(float) for z in zs], axis=0)
    w = np.array(ws)[:, None, None]
    have = np.isfinite(stack)
    total = np.where(have, stack, 0.0) * w
    weight_sum = (have * np.abs(w)).sum(axis=0)
    enough = have.sum(axis=0) >= (len(parts) if min_parts is None else min_parts)
    value = np.where(enough & (weight_sum > 0), total.sum(axis=0) / np.where(weight_sum > 0, weight_sum, 1.0), np.nan)
    return pd.DataFrame(value, index=ctx.dates, columns=ctx.ids)


COMPOSITES = {
    "value_composite": ("Value composite", "value", [("earnings_yield", 1), ("fcf_yield", 1), ("book_to_price", 1), ("sales_to_price", 1), ("ebitda_to_ev", 1)], 3),
    "quality_composite": ("Quality composite", "quality", [("roic", 1), ("gross_profitability", 1), ("accruals_ratio", 1), ("debt_to_assets", 1), ("margin_stability", 1)], 3),
    "growth_composite": ("Growth composite", "growth", [("revenue_growth", 1), ("gross_profit_growth", 1), ("operating_income_growth", 1), ("revenue_cagr_3y", 1)], 2),
    "momentum_composite": ("Fundamental momentum composite", "fundamental_momentum", [("d_operating_margin", 1), ("d_roic", 1), ("d_asset_turnover", 1), ("d_gross_margin", 1)], 2),
}


def _register_composites():
    for name, (label, family, parts, minimum) in COMPOSITES.items():
        needs = tuple(dict.fromkeys(need for part, _ in parts for need in FACTORS[part].needs))             # what the parts read, not the parts themselves
        FACTORS[name] = FactorDef(name, label, family, 1, needs, (lambda ctx, p=parts, m=minimum: composite(ctx, p, m)),
                                  "Equal-weighted average of standardised, direction-signed factors: " + ", ".join(n for n, _ in parts) + ".")


_register_composites()
