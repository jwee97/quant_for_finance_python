"""The fundamental factor library: value, quality, momentum and estimate-revision factors, each as a table of dates by assets that uses only what was known on the date.

Every factor is defined by what it measures (``cfo2ev`` is cash flow from operations over enterprise value) and carries ``sign``, the direction the literature expects for it (+1: more is better, -1: more
is worse, so the model buys the low end). The sign is a prior, not a result: :mod:`src.equity.alpha_model` learns weights from the data and is free to disagree, and every number the library
produces is the raw quantity, not its negative.

Value (how much the company earns per dollar the market asks)::

    cfo2ev        cash flow from operations / enterprise value          ebitda2ev    EBITDA / enterprise value
    earnings_yield   trailing net income / market value                 forward_earnings_yield   consensus EPS for the year / price
    payout_yield  (dividends + buybacks - issuance) / market value      nxf2ev       net external financing / enterprise value (sign -1)
    b2p           book equity / market value                            s2ev         sales / enterprise value

Quality (how well the business is run and how it funds itself)::

    rnoa          after-tax operating income / average net operating assets     cfroi    cash flow return on investment (the internal rate of return of the asset base)
    operating_leverage   (COGS + SG&A) / total assets                          operating_leverage_change   its change over a year
    wc_inc        increase in working capital / average assets (sign -1)        nco_inc  increase in net non-current operating assets / average assets (sign -1)
    icapx         capex over its three-year average, less one (sign -1)         capxg    capex growth over a year (sign -1)
    xf            net external financing / average assets (sign -1)             share_inc   growth in shares outstanding over a year (sign -1)

Momentum (prices alone) and estimates (analysts)::

    ret1          the last month's return (sign -1: short-term reversal)        ret9     the 9-month return ending a month ago
    adj_ret9      ret9 per unit of its own volatility                           earn_rev9   9-month revision of consensus EPS / price
    earn_diff9    share of analysts revising up less down, averaged over 9 months     ltg_rev9   9-month revision of the long-term growth forecast

A month is 21 trading days and a year 252. Flows are trailing twelve months; changes compare with what was known a year ago, so a quarterly filing is compared with the figure from the same
quarter a year earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .fundamentals import DEBT, MONTH, YEAR, FactorInputs, ratio

WINDOW9 = 9 * MONTH
FINANCING = ("equity_issuance", "equity_repurchase", "dividends", "debt_issuance", "debt_repayment")             # a financing factor needs at least one of these (a missing one counts as zero)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    style: str
    needs: tuple
    sign: int
    text: str
    fn: Callable[..., pd.DataFrame]


FACTORS: dict[str, FactorSpec] = {}


def factor(name: str, style: str, needs: tuple, sign: int, text: str):
    def wrap(fn):
        FACTORS[name] = FactorSpec(name, style, tuple(needs), sign, text, fn)
        return fn
    return wrap


def _net_external_financing(x: FactorInputs) -> pd.DataFrame:
    """Equity issued less bought back less dividends, plus debt issued less repaid (Bradshaw, Richardson and Sloan 2006); a component the file lacks counts as zero."""
    total = 0.0
    for name, sign in (("equity_issuance", 1), ("equity_repurchase", -1), ("dividends", -1), ("debt_issuance", 1), ("debt_repayment", -1)):
        if x.fund.has(name):
            total = total + sign * x.field(name).fillna(0.0)
    return total if isinstance(total, pd.DataFrame) else pd.DataFrame(np.nan, index=x.index, columns=x.columns)


def _total_debt(x: FactorInputs) -> pd.DataFrame:
    return x.total_debt


# ------------------------------------------------------------------------------------------------------------------ value
@factor("cfo2ev", "value", ("cfo", "shares_outstanding", "cash", DEBT), +1, "cash flow from operations over enterprise value")
def cfo2ev(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("cfo"), x.enterprise_value)


@factor("ebitda2ev", "value", ("ebitda", "shares_outstanding", "cash", DEBT), +1, "EBITDA over enterprise value")
def ebitda2ev(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("ebitda"), x.enterprise_value)


@factor("earnings_yield", "value", ("net_income", "shares_outstanding"), +1, "trailing net income over market value")
def earnings_yield(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("net_income"), x.market_cap)


@factor("forward_earnings_yield", "value", ("eps_fy1",), +1, "consensus earnings per share for the current year over price")
def forward_earnings_yield(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("eps_fy1"), x.prices)


@factor("payout_yield", "value", ("dividends", "equity_repurchase", "shares_outstanding"), +1, "dividends plus buybacks (less any shares issued) over market value")
def payout_yield(x: FactorInputs) -> pd.DataFrame:
    issued = x.field("equity_issuance").fillna(0.0) if x.fund.has("equity_issuance") else 0.0
    return ratio(x.field("dividends").fillna(0.0) + x.field("equity_repurchase").fillna(0.0) - issued, x.market_cap)


@factor("nxf2ev", "value", ("shares_outstanding", "cash", DEBT, FINANCING), -1, "net external financing over enterprise value (a firm that raises money is worse than one that returns it)")
def nxf2ev(x: FactorInputs) -> pd.DataFrame:
    return ratio(_net_external_financing(x), x.enterprise_value)


@factor("b2p", "value", ("book_equity", "shares_outstanding"), +1, "book equity over market value")
def b2p(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("book_equity"), x.market_cap)


@factor("s2ev", "value", ("sales", "shares_outstanding", "cash", DEBT), +1, "sales over enterprise value")
def s2ev(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("sales"), x.enterprise_value)


# ------------------------------------------------------------------------------------------------------------------ quality
def net_operating_assets(x: FactorInputs) -> pd.DataFrame:
    """Operating assets (everything but cash and long-term investments) less operating liabilities (everything but debt): what the business itself has tied up (Nissim and Penman 2001)."""
    investments = x.field("long_term_investments").fillna(0.0) if x.fund.has("long_term_investments") else 0.0
    return (x.field("total_assets") - x.field("cash").fillna(0.0) - investments) - (x.field("total_liabilities") - _total_debt(x))


@factor("rnoa", "quality", ("operating_income", "total_assets", "total_liabilities", "cash", "short_term_debt", "long_term_debt"), +1,
        "after-tax operating income over average net operating assets")
def rnoa(x: FactorInputs, tax: float = 0.25) -> pd.DataFrame:
    rate = x.field("tax_rate").fillna(tax) if x.fund.has("tax_rate") else tax
    noa = net_operating_assets(x)
    return ratio(x.field("operating_income") * (1.0 - rate), (noa + noa.shift(YEAR)) / 2.0)


def cfroi_irr(gross_cash_flow, gross_investment, life, nondepreciating, lo: float = -0.5, hi: float = 2.0, iterations: int = 80) -> np.ndarray:
    """The internal rate of return ``r`` that equates the gross investment with a level gross cash flow earned for ``life`` years plus the non-depreciating assets recovered at the end::

        gross_investment = gross_cash_flow * (1 - (1 + r)^-life) / r + nondepreciating / (1 + r)^life

    (HOLT's CFROI without its inflation adjustments). Solved by bisection on ``[lo, hi]``; NaN where the equation has no root there (no positive cash flow, or the assets alone are worth more than the investment)."""
    gcf, gi, n, nda = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(gross_cash_flow, gross_investment, life, nondepreciating))

    def present_value(r):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            disc = (1.0 + r) ** (-n)
            annuity = np.where(np.abs(r) < 1e-9, n, (1.0 - disc) / np.where(np.abs(r) < 1e-9, 1.0, r))
            return gcf * annuity + np.where(nda > 0, nda * disc, 0.0)                                  # no recovery value is no term, not 0 times infinity

    a, b = np.full(gcf.shape, lo), np.full(gcf.shape, hi)
    fa, fb = present_value(a) - gi, present_value(b) - gi
    valid = (fa > 0) & (fb < 0) & (n > 0) & (gi > 0)                                    # NaN inputs fail every comparison; an infinite value at the lower bound (a very long life) is simply large
    for _ in range(iterations):
        mid = 0.5 * (a + b)
        fm = present_value(mid) - gi
        up = fm > 0                                                          # the present value falls as the rate rises: still above the investment, so the root is higher
        a, b = np.where(up, mid, a), np.where(up, b, mid)
    return np.where(valid, 0.5 * (a + b), np.nan)


@factor("cfroi", "quality", ("net_income", "depreciation", "interest_expense", "gross_plant", "net_plant", "total_assets"), +1, "cash flow return on investment (the IRR of the asset base)")
def cfroi(x: FactorInputs) -> pd.DataFrame:
    gcf = x.field("net_income") + x.field("depreciation") + x.field("interest_expense")
    plant, net_plant, assets = x.field("gross_plant"), x.field("net_plant"), x.field("total_assets")
    life = ratio(plant, x.field("depreciation")).clip(3.0, 40.0)
    gross_investment = assets + (plant - net_plant)                            # add back what has been depreciated
    out = cfroi_irr(gcf.to_numpy(), gross_investment.to_numpy(), life.to_numpy(), (assets - net_plant).to_numpy())
    return pd.DataFrame(out, index=gcf.index, columns=gcf.columns)


@factor("operating_leverage", "quality", ("cogs", "sga", "total_assets"), +1, "(cost of goods sold + selling and administrative expense) over total assets")
def operating_leverage(x: FactorInputs) -> pd.DataFrame:
    return ratio(x.field("cogs") + x.field("sga"), x.field("total_assets"))


@factor("operating_leverage_change", "quality", ("cogs", "sga", "total_assets"), -1, "the change in operating leverage over a year")
def operating_leverage_change(x: FactorInputs) -> pd.DataFrame:
    ol = operating_leverage(x)
    return ol - ol.shift(YEAR)


@factor("wc_inc", "quality", ("current_assets", "current_liabilities", "cash", "short_term_debt", "total_assets"), -1, "increase in non-cash working capital over average assets (Sloan's accruals)")
def wc_inc(x: FactorInputs) -> pd.DataFrame:
    wc = (x.field("current_assets") - x.field("cash").fillna(0.0)) - (x.field("current_liabilities") - x.field("short_term_debt").fillna(0.0))
    return ratio(wc - wc.shift(YEAR), x.average("total_assets"))


@factor("nco_inc", "quality", ("total_assets", "current_assets", "total_liabilities", "current_liabilities", "long_term_debt"), -1,
        "increase in net non-current operating assets over average assets")
def nco_inc(x: FactorInputs) -> pd.DataFrame:
    investments = x.field("long_term_investments").fillna(0.0) if x.fund.has("long_term_investments") else 0.0
    nco = (x.field("total_assets") - x.field("current_assets") - investments) - (x.field("total_liabilities") - x.field("current_liabilities") - x.field("long_term_debt").fillna(0.0))
    return ratio(nco - nco.shift(YEAR), x.average("total_assets"))


@factor("icapx", "quality", ("capex",), -1, "capital expenditure over its average of the three years before, less one (abnormal investment)")
def icapx(x: FactorInputs) -> pd.DataFrame:
    capex = x.field("capex")
    history = (capex.shift(YEAR) + capex.shift(2 * YEAR) + capex.shift(3 * YEAR)) / 3.0
    return ratio(capex, history.where(history > 0)) - 1.0


@factor("capxg", "quality", ("capex",), -1, "capital expenditure growth over a year")
def capxg(x: FactorInputs) -> pd.DataFrame:
    capex = x.field("capex")
    return ratio(capex, capex.shift(YEAR).where(capex.shift(YEAR) > 0)) - 1.0


@factor("xf", "quality", ("total_assets", FINANCING), -1, "net external financing over average assets")
def xf(x: FactorInputs) -> pd.DataFrame:
    return ratio(_net_external_financing(x), x.average("total_assets"))


@factor("share_inc", "quality", ("shares_outstanding",), -1, "growth in shares outstanding over a year")
def share_inc(x: FactorInputs) -> pd.DataFrame:
    shares = x.field("shares_outstanding")
    return ratio(shares, shares.shift(YEAR)) - 1.0


# ------------------------------------------------------------------------------------------------------------------ momentum
@factor("ret1", "momentum", (), -1, "the last month's return (short-term reversal: the model buys last month's losers)")
def ret1(x: FactorInputs) -> pd.DataFrame:
    return x.prices / x.prices.shift(MONTH) - 1.0


@factor("ret9", "momentum", (), +1, "the nine-month return ending a month ago (intermediate-term continuation)")
def ret9(x: FactorInputs) -> pd.DataFrame:
    return x.prices.shift(MONTH) / x.prices.shift(MONTH + WINDOW9) - 1.0


@factor("adj_ret9", "momentum", (), +1, "the nine-month return ending a month ago per unit of the volatility it was earned with")
def adj_ret9(x: FactorInputs) -> pd.DataFrame:
    vol = x.returns.shift(MONTH).rolling(WINDOW9, min_periods=WINDOW9 // 2).std() * np.sqrt(WINDOW9)
    return ratio(ret9(x), vol)


# ------------------------------------------------------------------------------------------------------------------ estimates
@factor("earn_rev9", "estimates", ("eps_fy1",), +1, "nine-month revision of the consensus EPS forecast, over price")
def earn_rev9(x: FactorInputs) -> pd.DataFrame:
    eps = x.field("eps_fy1")
    return ratio(eps - eps.shift(WINDOW9), x.prices)


@factor("earn_diff9", "estimates", ("n_up", "n_down", "n_estimates"), +1, "analysts revising up less those revising down, as a share of all analysts, averaged over nine months")
def earn_diff9(x: FactorInputs) -> pd.DataFrame:
    net = ratio(x.field("n_up").fillna(0.0) - x.field("n_down").fillna(0.0), x.field("n_estimates"))
    return net.rolling(WINDOW9, min_periods=WINDOW9 // 3).mean()


@factor("ltg_rev9", "estimates", ("ltg",), +1, "nine-month revision of the consensus long-term growth forecast")
def ltg_rev9(x: FactorInputs) -> pd.DataFrame:
    g = x.field("ltg")
    return g - g.shift(WINDOW9)


STYLES = ("value", "quality", "momentum", "estimates", "valuation")


def names(style: str | None = None) -> list[str]:
    return [n for n, s in FACTORS.items() if style is None or s.style == style]


def available(inputs: FactorInputs, style: str | None = None) -> tuple[list[str], dict[str, list[str]]]:
    """The factors of a style that the supplied statements can support, and for each of the others the fields it lacks."""
    ok, lacking = [], {}
    for n in names(style):
        gap = inputs.fund.missing(*FACTORS[n].needs)
        (lacking.__setitem__(n, gap) if gap else ok.append(n))
    return ok, lacking


def compute(name: str, inputs: FactorInputs, **params) -> pd.DataFrame:
    """One factor's raw table. A factor whose fields are missing from the file raises a ``KeyError`` that names them."""
    if name not in FACTORS:
        raise KeyError(f"unknown factor '{name}'; known: {', '.join(FACTORS)}")
    spec = FACTORS[name]
    gap = inputs.fund.missing(*spec.needs)
    if gap:
        raise KeyError(f"the factor {name} needs the columns {gap}, which the fundamentals file does not have")
    return spec.fn(inputs, **params)


def compute_all(inputs: FactorInputs, style: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Every factor (of a style) the data supports, and the ones skipped with the columns they lack."""
    ok, lacking = available(inputs, style)
    return {n: FACTORS[n].fn(inputs) for n in ok}, lacking


from . import dcf as _dcf  # noqa: E402,F401  (registers the valuation factors, so the library is the same whichever module was imported first)
