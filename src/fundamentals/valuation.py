"""Yardsticks for what a share might be worth: published rules applied to one company's statements, shown as ranges beside the last price.

A yardstick answers "what would the share be worth *if this rule's assumption held*", never "what is it worth". Each one is a different assumption (assets only, no growth at all, growth for ever,
a fair multiple of growth), so they disagree, and the disagreement is the information: a price far above every bar asks the buyer to believe something none of the rules does. They are per-share
amounts in the statements' own currency and are set against the last close only when the price is quoted in that same currency (a mismatch is said, never converted at a guessed rate).

The rules, with where they come from:

* **Graham number** - the price at which a share is at 15 times three-year average earnings and 1.5 times book value at once (Graham, *The Intelligent Investor*, chapter 14).
* **Net current asset value** - current assets less every liability and the preferred stock; Graham bought at two-thirds of it or less (Graham and Dodd, *Security Analysis*; Oppenheimer 1986).
* **Earnings power value** - normalised after-tax operating profit capitalised as a perpetuity with no growth, plus cash, less debt (Greenwald, Kahn, Sonkin and van Biema 2001).
* **Owner earnings and free cash flow, capitalised** - the average of three years, growing for ever at the rate chosen, discounted at the rate chosen (the Gordon growth model; Buffett 1986 for owner earnings).
* **The Lynch line** - earnings per share times the growth rate in per cent: a P/E equal to the growth rate (Lynch, *One Up on Wall Street*).
* **Book value per share.**

The two inputs a reader may change are the discount rate and the perpetual growth rate; every range moves the discount rate one point each way. The growth must stay at least two points under the rate, or
a perpetuity has no finite value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .errors import FundamentalsError
from .metrics import DEFAULT_TAX_RATE
from .scores import GRAHAM_MULTIPLIER, GRAHAM_NCAV_BUY

DEFAULT_RATE = 0.09
DEFAULT_GROWTH = 0.02
BAND = 0.01                         # each range moves the discount rate this far either way
MIN_SPREAD = 0.02                   # the rate must exceed the growth by at least this much, so that even the lowest rate of a range leaves a spread
RATE_RANGE = (0.03, 0.25)
GROWTH_RANGE = (-0.05, 0.08)
SMOOTH = 3                          # years averaged for owner earnings and free cash flow
HISTORY = 5                         # years averaged for the operating margin of the earnings power value
MIN_MARGIN_YEARS = 3
LYNCH_MAX_PE = 40.0                 # the Lynch line never asks for a multiple of earnings beyond this
GRAHAM_PE, GRAHAM_PB = 15.0, 1.5


def check_inputs(rate: float, growth: float) -> None:
    if not RATE_RANGE[0] <= rate <= RATE_RANGE[1]:
        raise FundamentalsError(f"the discount rate must be between {RATE_RANGE[0]:.0%} and {RATE_RANGE[1]:.0%}")
    if not GROWTH_RANGE[0] <= growth <= GROWTH_RANGE[1]:
        raise FundamentalsError(f"the perpetual growth must be between {GROWTH_RANGE[0]:.0%} and {GROWTH_RANGE[1]:.0%}")
    if rate - growth < MIN_SPREAD - 1e-12:
        raise FundamentalsError(f"the discount rate must be at least {MIN_SPREAD * 100:.0f} points above the growth rate (a perpetuity growing as fast as the rate that discounts it is worth an unbounded amount)")


def perpetuity(flow: float, rate: float, growth: float) -> float | None:
    """The value now of ``flow`` (this year's) growing at ``growth`` for ever, discounted at ``rate``: ``flow * (1 + growth) / (rate - growth)``. ``None`` unless the flow is positive and the rate exceeds the growth."""
    if flow is None or not math.isfinite(flow) or flow <= 0 or not rate > growth:
        return None
    return flow * (1.0 + growth) / (rate - growth)


def _last(rows: pd.DataFrame, name: str) -> float:
    if name not in rows.columns or rows.empty:
        return float("nan")
    return float(rows[name].iloc[-1])


def _mean_of_last(rows: pd.DataFrame, name: str, years: int) -> float:
    """The mean of the last ``years`` fiscal years of a figure, or NaN unless every one of them is there."""
    if name not in rows.columns or len(rows) < years:
        return float("nan")
    tail = rows[name].tail(years).astype(float)
    return float(tail.mean()) if tail.notna().all() else float("nan")


def _row(key: str, label: str, text: str, low=None, high=None, point=None, missing: str | None = None, inputs: list[str] | None = None) -> dict:
    return {"key": key, "label": label, "text": text, "low": low, "high": high, "point": point, "missing": missing, "inputs": inputs or []}


def _money(v: float) -> str:
    return f"{v:,.2f}"


def build(rows: pd.DataFrame, price: float | None, price_date: str | None, trading_currency: str | None, rate: float = DEFAULT_RATE, growth: float = DEFAULT_GROWTH) -> dict:
    """The yardsticks for one company. ``rows`` are its fiscal years in order (the derived metrics of :mod:`metrics` and :mod:`scores` already attached); ``price`` is the last unadjusted close."""
    check_inputs(rate, growth)
    if rows.empty:
        raise FundamentalsError("this company has no annual statements to value")
    currency = str(rows["reporting_currency"].iloc[-1]) if "reporting_currency" in rows.columns and pd.notna(rows["reporting_currency"].iloc[-1]) else None
    shares = _last(rows, "shares_outstanding")
    if not math.isfinite(shares) or shares <= 0:
        shares = _last(rows, "weighted_avg_shares_diluted")
    has_shares = math.isfinite(shares) and shares > 0
    year = int(rows["fiscal_year"].iloc[-1])
    out: list[dict] = []
    no_shares = "the number of shares is not in the statements"

    # --- Graham number
    gn = _last(rows, "graham_number")
    out.append(_row("graham_number", "Graham number",
                    "The price at which a share is at 15 times its average earnings of the last three years and 1.5 times its book value at once: the square root of 22.5 times earnings per share times book value per share.",
                    point=gn if math.isfinite(gn) else None, low=gn if math.isfinite(gn) else None, high=gn if math.isfinite(gn) else None,
                    missing=None if math.isfinite(gn) else "needs three years of earnings per share whose average is positive, and a positive book value",
                    inputs=[f"average earnings per share {_money(_last(rows, 'eps_avg_3y'))}", f"book value per share {_money(_last(rows, 'book_value_per_share'))}"] if math.isfinite(gn) else []))

    # --- net current asset value
    ncav = _last(rows, "ncav_per_share")
    ok = math.isfinite(ncav) and ncav > 0
    out.append(_row("ncav", "Net current asset value",
                    "Current assets less every liability and the preferred stock, per share. Graham bought a share at two-thirds of this or less, so the bar runs from two-thirds to the whole.",
                    low=GRAHAM_NCAV_BUY * ncav if ok else None, high=ncav if ok else None, point=ncav if ok else None,
                    missing=None if ok else ("the current assets do not cover the liabilities: there is no net current asset value to buy at a discount" if math.isfinite(ncav) else "needs current assets, total liabilities and the shares"),
                    inputs=[f"{_money(ncav)} per share"] if math.isfinite(ncav) else []))

    # --- earnings power value
    out.append(_earnings_power(rows, rate, shares if has_shares else None, no_shares))

    # --- the perpetuities
    for key, label, name, text in (("owner_earnings", "Owner earnings, capitalised", "owner_earnings", "Net income plus depreciation less capital spending (Buffett's owner earnings)"),
                                   ("free_cash_flow", "Free cash flow, capitalised", "fcf", "Operating cash flow less capital spending")):
        flow = _mean_of_last(rows, name, SMOOTH)
        values = [perpetuity(flow, r, growth) for r in (rate + BAND, rate, rate - BAND)]
        ok = has_shares and all(v is not None for v in values)
        why = None if ok else (no_shares if not has_shares else f"needs {SMOOTH} years of figures whose average is positive")
        out.append(_row(key, label, f"{text}, averaged over the last {SMOOTH} years, growing at {growth:.1%} for ever and discounted at {rate:.1%}; the bar moves the rate a point either way.",
                        low=values[0] / shares if ok else None, point=values[1] / shares if ok else None, high=values[2] / shares if ok else None, missing=why,
                        inputs=[f"average {_money(flow)} a year"] if math.isfinite(flow) else []))

    # --- the Lynch line
    eps, g = _last(rows, "eps_diluted"), _last(rows, "eps_cagr_3y")
    ok = math.isfinite(eps) and eps > 0 and math.isfinite(g) and g > 0
    fair = eps * min(100.0 * g, LYNCH_MAX_PE) if ok else None
    out.append(_row("lynch", "The Lynch line",
                    f"Earnings per share times the three-year growth in earnings per share in per cent (at most {LYNCH_MAX_PE:.0f} times): the price at which the P/E equals the growth rate.",
                    low=fair, high=fair, point=fair, missing=None if ok else "needs positive earnings per share now and three years ago", inputs=[f"earnings per share {_money(eps)}", f"growth {g:.1%}"] if ok else []))

    # --- book value
    bv = _last(rows, "book_value_per_share")
    ok = math.isfinite(bv) and bv > 0
    out.append(_row("book_value", "Book value per share", "Shareholders' equity per share: what the balance sheet says the owners own.", low=bv if ok else None, high=bv if ok else None, point=bv if ok else None,
                    missing=None if ok else "equity is not positive, or the shares are not known"))

    comparable = price is not None and math.isfinite(price) and price > 0 and bool(trading_currency) and bool(currency) and trading_currency == currency
    if price is None:
        why = "no price is loaded for this company"
    elif not trading_currency or not currency:
        why = "the currency of the price or of the statements is not known"
    elif trading_currency != currency:
        why = f"the price is in {trading_currency} and the statements are in {currency}: they are not compared"
    else:
        why = None
    ref = {"value": float(price), "date": price_date, "currency": trading_currency} if comparable else None
    tests = _graham_price_tests(rows, price) if comparable else []
    return {"inputs": {"rate": rate, "growth": growth, "band": BAND, "min_spread": MIN_SPREAD}, "fiscal_year": year, "currency": currency, "price": ref, "price_note": why, "rows": out, "graham_tests": tests,
            "notes": ["Each bar is what the share would be worth if one rule's assumption held; they are not forecasts, and they disagree on purpose.",
                      "Debt that the statements do not report counts as none; cash that they do not report counts as none."]}


def _earnings_power(rows: pd.DataFrame, rate: float, shares: float | None, no_shares: str) -> dict:
    text = (f"Greenwald's earnings power value: the operating margin of up to the last {HISTORY} years on this year's revenue, taxed at the company's own average rate (else {DEFAULT_TAX_RATE:.0%}) and capitalised at "
            "the discount rate as if it never grew, plus cash, less debt, minority interest and preferred stock. It assumes capital spending equals depreciation.")
    label = "Earnings power value"
    revenue = _last(rows, "revenue")
    if "operating_income" in rows.columns:
        operating = rows["operating_income"].fillna(rows["ebit"] if "ebit" in rows.columns else np.nan)
    else:
        operating = rows["ebit"] if "ebit" in rows.columns else pd.Series(np.nan, index=rows.index)
    margins = (operating / rows["revenue"].where(rows["revenue"] > 0)).dropna().tail(HISTORY)
    if shares is None:
        return _row("epv", label, text, missing=no_shares)
    if len(margins) < MIN_MARGIN_YEARS or not math.isfinite(revenue) or revenue <= 0:
        return _row("epv", label, text, missing=f"needs the operating margin of {MIN_MARGIN_YEARS} years and this year's revenue")
    taxes = (rows["income_tax"] / rows["income_before_tax"].where(rows["income_before_tax"] > 0)).dropna().tail(HISTORY)
    taxes = taxes[(taxes >= 0) & (taxes <= 0.45)]
    tax = float(taxes.mean()) if len(taxes) else DEFAULT_TAX_RATE
    nopat = float(margins.mean()) * revenue * (1.0 - tax)
    if nopat <= 0:
        return _row("epv", label, text, missing="the normalised operating profit is not positive")
    debt, cash = _last(rows, "total_debt"), _last(rows, "cash_and_equivalents")
    if not math.isfinite(cash):
        cash = _last(rows, "cash")
    if not (math.isfinite(debt) or math.isfinite(cash)):
        return _row("epv", label, text, missing="neither debt nor cash is in the statements")
    other = (0.0 if not math.isfinite(_last(rows, "minority_interest")) else _last(rows, "minority_interest")) + (0.0 if not math.isfinite(_last(rows, "preferred_equity")) else _last(rows, "preferred_equity"))
    net = (0.0 if not math.isfinite(cash) else cash) - (0.0 if not math.isfinite(debt) else debt) - other

    def per_share(r: float) -> float:
        return (nopat / r + net) / shares

    return _row("epv", label, text, low=per_share(rate + BAND), point=per_share(rate), high=per_share(rate - BAND),
                inputs=[f"normalised operating profit after tax {_money(nopat)}", f"margin {float(margins.mean()):.1%} over {len(margins)} years", f"tax {tax:.1%}", f"net cash {_money(net)}"])


def _graham_price_tests(rows: pd.DataFrame, price: float) -> list[dict]:
    """Graham's two price tests for the defensive investor: no more than 15 times the average earnings of three years, no more than 1.5 times book value, and the two together no more than 22.5."""
    eps, bv = _last(rows, "eps_avg_3y"), _last(rows, "book_value_per_share")
    pe = price / eps if math.isfinite(eps) and eps > 0 else None
    pb = price / bv if math.isfinite(bv) and bv > 0 else None
    return [{"key": "pe", "label": "Price to three-year average earnings", "value": pe, "limit": GRAHAM_PE, "ok": None if pe is None else pe <= GRAHAM_PE},
            {"key": "pb", "label": "Price to book value", "value": pb, "limit": GRAHAM_PB, "ok": None if pb is None else pb <= GRAHAM_PB},
            {"key": "product", "label": "The two multiplied", "value": None if pe is None or pb is None else pe * pb, "limit": GRAHAM_MULTIPLIER, "ok": None if pe is None or pb is None else pe * pb <= GRAHAM_MULTIPLIER}]
