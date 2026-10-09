"""Discounted cash flow valuation, and its multipath version that carries the uncertainty in the inputs through to a distribution of values.

**DCF.** The free cash flow of a company grows from ``growth`` at first, fading linearly to a terminal rate ``terminal_growth`` over ``years`` years, and is discounted at ``rate``; after the last year
it grows at the terminal rate for ever (the Gordon formula). The enterprise value is the sum, the equity value is that less net debt, and the value per share is that over shares::

    FCF_t = FCF_(t-1) (1 + g_t),   g_t = growth + (terminal_growth - growth) (t - 1) / (years - 1)
    EV    = sum_t FCF_t / (1 + r)^t  +  FCF_N (1 + g_T) / (r - g_T) / (1 + r)^N

With ``growth == terminal_growth == g`` it reduces to ``FCF_0 (1 + g) / (r - g)``, the check the tests use.

**MDCF.** A single DCF is a point estimate built from numbers nobody knows: the growth rate, the discount rate, the terminal rate, where the cash flow starts. The multipath version draws them from
distributions centred on the point estimates (growth and starting cash flow with the widths you give, the discount rate with its own, the terminal rate kept below the discount rate) and values each
draw, so the output is a distribution: its median, the probability that the value exceeds the price, and the dispersion. Because the value is convex in the discount rate, the average of the draws exceeds
the value at the average rate: uncertainty adds value to a long-lived claim, which a point estimate cannot show. The draws are common random numbers (the same standard normals for every company and date),
so two companies with the same inputs get the same answer and the result is deterministic given the seed.

Both are factors: ``dcf_upside`` is value per share over price less one, ``mdcf_upside`` the same for the median of the multipath values and ``mdcf_prob`` the probability that value exceeds price.
They are recomputed once a month (the inputs move slowly and the simulation is the expensive part) from the statements known on that date.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .factors import FACTORS, FactorSpec
from .fundamentals import MONTH, YEAR, FactorInputs, ratio

WARMUP = YEAR + MONTH


def growth_path(growth, terminal_growth, years: int) -> np.ndarray:
    """The growth rate in each of the ``years`` explicit years, fading linearly from ``growth`` (year 1) to ``terminal_growth`` (year ``years``); shape ``(years, ...)``."""
    if years < 2:
        raise ValueError("years must be at least 2")
    g0, gT = np.broadcast_arrays(np.asarray(growth, dtype=float), np.asarray(terminal_growth, dtype=float))
    weight = (np.arange(years, dtype=float) / (years - 1)).reshape((years,) + (1,) * g0.ndim)
    return g0 + (gT - g0) * weight


def dcf_value(fcf0, growth, terminal_growth, rate, years: int = 10) -> np.ndarray:
    """The present value of the cash flows of the firm (enterprise value) for scalars or arrays (broadcast together). NaN where the discount rate does not exceed the terminal growth rate."""
    if years < 2:
        raise ValueError("years must be at least 2")
    fcf0, growth, terminal_growth, rate = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(fcf0, growth, terminal_growth, rate))
    flow, total = fcf0, np.zeros_like(fcf0)
    for t in range(1, years + 1):                                                        # year by year, so the memory is one array however many years there are
        flow = flow * (1.0 + growth + (terminal_growth - growth) * (t - 1) / (years - 1))
        total = total + flow * (1.0 + rate) ** -t
    with np.errstate(divide="ignore", invalid="ignore"):
        terminal = flow * (1.0 + terminal_growth) / (rate - terminal_growth) * (1.0 + rate) ** -years
    return np.where(rate > terminal_growth, total + terminal, np.nan)


def per_share(enterprise_value, net_debt, shares) -> np.ndarray:
    """Equity value per share: enterprise value less net debt (debt, preferred and minority interest less cash) over shares; NaN for no shares."""
    shares = np.asarray(shares, dtype=float)
    return (np.asarray(enterprise_value, dtype=float) - np.asarray(net_debt, dtype=float)) / np.where(shares > 0, shares, np.nan)


def mdcf_values(fcf0, growth, terminal_growth, rate, years: int = 10, paths: int = 500, growth_sd: float = 0.04, rate_sd: float = 0.01, fcf_sd: float = 0.15,
                terminal_sd: float = 0.005, seed: int = 0) -> np.ndarray:
    """The enterprise value in each of ``paths`` draws, shape ``(paths, ...)``: growth and the starting cash flow (lognormal, ``fcf_sd``) vary with the widths given, the discount rate with ``rate_sd``
    and the terminal rate with ``terminal_sd``. A draw whose discount rate does not exceed its terminal rate by at least a percentage point has the terminal rate pulled down to leave one."""
    fcf0, growth, terminal_growth, rate = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(fcf0, growth, terminal_growth, rate))
    z = np.random.default_rng(seed).standard_normal((4, paths)).reshape((4, paths) + (1,) * fcf0.ndim)       # common random numbers: the same shocks for every firm and date
    f = fcf0 * np.exp(fcf_sd * z[0] - 0.5 * fcf_sd ** 2)
    g1 = growth + growth_sd * z[1]
    r = np.maximum(rate + rate_sd * z[2], 0.01)
    gT = np.minimum(terminal_growth + terminal_sd * z[3], r - 0.01)
    return dcf_value(f, g1, gT, r, years)


def beta_estimate(returns: pd.DataFrame, window: int = YEAR, shrink: float = 1.0 / 3.0) -> pd.DataFrame:
    """The rolling beta to the equal-weight market, pulled a third of the way to one (Blume), which is what a discount rate should use: a raw beta is mostly noise."""
    market = returns.mean(axis=1)
    raw = returns.rolling(window, min_periods=window // 2).cov(market).div(market.rolling(window, min_periods=window // 2).var(), axis=0)
    return (1.0 - shrink) * raw + shrink


def _inputs_at(x: FactorInputs, rows: np.ndarray, risk_free: float, equity_premium: float, terminal_growth: float, years: int, growth_cap: float) -> dict:
    """The DCF inputs on the dates ``rows`` (positions in the price index), each of shape (dates, assets)."""
    take = lambda frame: frame.iloc[rows].to_numpy(dtype=float)
    tax = x.field("tax_rate").fillna(0.25) if x.fund.has("tax_rate") else 0.25
    interest = x.field("interest_expense").fillna(0.0) * (1.0 - tax) if x.fund.has("interest_expense") else 0.0
    fcf = x.field("cfo") + interest - x.field("capex")
    if x.fund.has("ltg"):
        growth = x.field("ltg")
    else:
        growth = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    if x.fund.has("sales"):
        realised = ratio(x.field("sales"), x.field("sales").shift(3 * YEAR)) ** (1.0 / 3.0) - 1.0          # three-year compound growth in sales when no forecast was supplied
        growth = growth.fillna(realised)
    growth = growth.clip(-growth_cap, growth_cap)
    debt = x.total_debt
    other = sum(x.field(c).fillna(0.0) for c in ("preferred", "minority_interest") if x.fund.has(c))
    net_debt = debt + other - (x.field("cash").fillna(0.0) if x.fund.has("cash") else 0.0)
    rate = (risk_free + beta_estimate(x.returns) * equity_premium).clip(terminal_growth + 0.02, 0.25)
    return {"fcf0": take(fcf), "growth": take(growth), "net_debt": take(net_debt), "shares": take(x.field("shares_outstanding")), "rate": take(rate)}


def _monthly_rows(x: FactorInputs) -> np.ndarray:
    return np.arange(min(WARMUP, len(x.index) - 1), len(x.index), MONTH)


def _spread_to_days(values: np.ndarray, rows: np.ndarray, x: FactorInputs) -> pd.DataFrame:
    sparse = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    sparse.iloc[rows] = values
    return sparse.ffill(limit=MONTH + 5)


def dcf_upside(x: FactorInputs, risk_free: float = 0.03, equity_premium: float = 0.05, terminal_growth: float = 0.025, years: int = 10, growth_cap: float = 0.30) -> pd.DataFrame:
    """Value per share from the point-estimate DCF over the price, less one. Needs positive free cash flow (otherwise there is nothing to grow); refreshed monthly."""
    rows = _monthly_rows(x)
    v = _inputs_at(x, rows, risk_free, equity_premium, terminal_growth, years, growth_cap)
    ev = dcf_value(np.where(v["fcf0"] > 0, v["fcf0"], np.nan), v["growth"], terminal_growth, v["rate"], years)
    value = _spread_to_days(per_share(ev, v["net_debt"], v["shares"]), rows, x)
    return ratio(value.where(value > 0), x.prices) - 1.0


def mdcf_distribution(x: FactorInputs, risk_free: float = 0.03, equity_premium: float = 0.05, terminal_growth: float = 0.025, years: int = 10, growth_cap: float = 0.30,
                      paths: int = 400, growth_sd: float = 0.04, rate_sd: float = 0.01, fcf_sd: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """The per-share value in every path for each monthly date: shape ``(paths, dates, assets)``, with the dates and the inputs used."""
    rows = _monthly_rows(x)
    v = _inputs_at(x, rows, risk_free, equity_premium, terminal_growth, years, growth_cap)
    ev = mdcf_values(np.where(v["fcf0"] > 0, v["fcf0"], np.nan), v["growth"], terminal_growth, v["rate"], years, paths, growth_sd, rate_sd, fcf_sd, seed=seed)
    return per_share(ev, v["net_debt"], v["shares"]), rows, v


def mdcf_upside(x: FactorInputs, **params) -> pd.DataFrame:
    """The median of the multipath values over the price, less one."""
    values, rows, _ = mdcf_distribution(x, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                     # a company with no free cash flow has no values to take the median of
        median = _spread_to_days(np.nanmedian(np.where(values > 0, values, np.nan), axis=0), rows, x)
    return ratio(median, x.prices) - 1.0


def mdcf_prob(x: FactorInputs, **params) -> pd.DataFrame:
    """The share of the multipath values that exceed the price on the date the values were computed (refreshed monthly): how much of the plausible range says the stock is cheap."""
    values, rows, _ = mdcf_distribution(x, **params)
    price = x.prices.iloc[rows].to_numpy(dtype=float)
    finite = np.isfinite(values)
    p = np.where(finite.any(axis=0), (finite & (values > price[None])).sum(axis=0) / np.maximum(finite.sum(axis=0), 1), np.nan)
    return _spread_to_days(p, rows, x)


_VALUATION_NEEDS = ("cfo", "capex", "shares_outstanding", "total_assets")
for _name, _fn, _text in (("dcf_upside", dcf_upside, "discounted cash flow value per share over price, less one"),
                          ("mdcf_upside", mdcf_upside, "median of the multipath (Monte Carlo) DCF values per share over price, less one"),
                          ("mdcf_prob", mdcf_prob, "probability that the multipath DCF value per share exceeds the price")):
    FACTORS[_name] = FactorSpec(_name, "valuation", _VALUATION_NEEDS, +1, _text, _fn)
