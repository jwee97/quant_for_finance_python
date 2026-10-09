"""A portfolio with money moving in and out of it, followed day by day.

``simulate_cashflows`` holds a portfolio that aims at target weights (a fixed mix, or the weights any strategy in this repository holds) and applies the flows of a ``flow_schedule``: deposits and withdrawals
(handled by the policies of :mod:`src.cashflow.policies`), cash dividends (reinvested in the asset that paid them, treated as one more deposit, or left in cash) and the scheduled rebalances. It reports

* the money-weighted return (``irr``): what the investor, who chose when to add and take money, earned on their dollars;
* the time-weighted return (``twr``): what the portfolio earned whatever the flows, which is what a strategy backtest measures; the two differ by the timing of the flows;
* the cost: turnover, trading cost, and how far the book sat from its target (``deviation``: half the sum of absolute weight differences).

Prices are taken as total-return series (dividends included), the way adjusted closes are. A ``dividend_yield`` therefore splits each day's return into a price part and a dividend part that is
earned daily but paid in cash at the end of each quarter; the total return is unchanged, and what is tested is what is done with the cash. Trades are filled at the day's close at the same price;
``cost_bps`` is charged on every dollar traded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ..utils.dates import rebalance_dates
from .flows import period_starts
from .policies import POLICIES, allocate_flow

DIVIDEND_POLICIES = ("reinvest", "flow", "cash")
REBALANCE = ("none", "daily", "weekly", "monthly", "quarterly", "annual")


@dataclass
class CashflowResult:
    """The path of the portfolio: dollars by asset, cash, trades and the summary figures."""

    nav: pd.Series
    values: pd.DataFrame
    cash: pd.Series
    accrued: pd.Series
    flows: pd.Series
    trades: pd.DataFrame
    costs: pd.Series
    dividends: pd.Series
    twr: pd.Series
    target: pd.DataFrame
    summary: dict

    @property
    def weights(self) -> pd.DataFrame:
        return self.values.div(self.nav.replace(0.0, np.nan), axis=0)

    @property
    def deviation(self) -> pd.Series:
        """Half the sum of absolute differences between the weights held and the target weights, each day (0 = exactly on target; the part of the book in cash counts as a difference)."""
        return 0.5 * (self.weights.fillna(0.0) - self.target).abs().sum(axis=1)


def money_weighted_return(dates, flows, final_value: float, initial: float) -> float:
    """The annualised internal rate of return of an investor who put ``initial`` in on the first date, added ``flows`` (deposits positive) on their dates and holds ``final_value`` on the last.

    ``dates[0]`` is the start and ``dates[-1]`` the end; ``flows`` has one entry per date (the first and last are normally zero). The rate ``r`` solves
    ``initial + sum(flow_k / (1 + r)^t_k) = final / (1 + r)^T`` with times in years. NaN when no rate between -99% and +1000% does."""
    dates = pd.DatetimeIndex(dates)
    t = np.asarray((dates - dates[0]).days, float) / 365.25
    flows = np.asarray(flows, float)
    T = t[-1]

    def gap(r: float) -> float:
        return initial + float((flows / (1.0 + r) ** t).sum()) - final_value / (1.0 + r) ** T

    lo, hi = -0.99, 10.0
    try:
        if gap(lo) * gap(hi) > 0:
            return float("nan")
        return float(brentq(gap, lo, hi, xtol=1e-12))
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def _targets_frame(targets, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    if isinstance(targets, pd.Series):
        frame = pd.DataFrame([targets.reindex(columns).fillna(0.0).to_numpy()] * len(index), index=index, columns=columns)
    else:
        frame = pd.DataFrame(targets).reindex(index=index, columns=columns).ffill()
    return frame.fillna(0.0).astype(float)


def simulate_cashflows(prices: pd.DataFrame, targets, flows: pd.Series | None = None, *, initial: float = 100_000.0, inflow_policy: str = "correct_drift", outflow_policy: str = "correct_drift",
                       rebalance: str = "monthly", dividend_yield=None, dividend_policy: str = "flow", cost_bps: float = 5.0, cash_rate: float = 0.0, dca_months: int = 1,
                       liquidity: pd.Series | None = None) -> CashflowResult:
    """Follow a portfolio through ``flows``; see the module docstring. ``targets`` is a ``{ticker: weight}`` series (a fixed mix) or a date-by-ticker frame of the weights to hold (a strategy's book,
    carried forward between its dates); the first date on which any target is non-zero is the day ``initial`` is invested.

    ``inflow_policy`` and ``outflow_policy`` say how a deposit or a withdrawal is traded (see ``allocate_flow``); ``rebalance`` is how often the whole book is traded back to target regardless of flows
    (``none`` leaves the flows to correct drift). ``dividend_yield`` (a float, or a ticker-to-yield mapping, per year) is paid in cash each quarter-end and handled by ``dividend_policy``:
    ``reinvest`` buys back the asset that paid it, ``flow`` puts it through the flow policies like a deposit (netted against a withdrawal made the same day), ``cash`` leaves it. A deposit can be
    invested over ``dca_months`` month-starts (dollar-cost averaging; the part not yet invested earns ``cash_rate``). A withdrawal larger than the portfolio is limited to what it holds.
    ``cost_bps`` is charged on all dollars traded."""
    for name, policy in (("inflow_policy", inflow_policy), ("outflow_policy", outflow_policy)):
        if policy not in POLICIES:
            raise ValueError(f"{name} must be one of {POLICIES}")
    if dividend_policy not in DIVIDEND_POLICIES or rebalance not in REBALANCE:
        raise ValueError(f"dividend_policy must be one of {DIVIDEND_POLICIES} and rebalance one of {REBALANCE}")
    if initial <= 0 or cost_bps < 0 or dca_months < 1:
        raise ValueError("initial > 0, cost_bps >= 0, dca_months >= 1")
    prices = pd.DataFrame(prices).sort_index().ffill()
    index, assets = prices.index, list(prices.columns)
    target = _targets_frame(targets, index, assets)
    live = (target.abs().sum(axis=1) > 1e-12) & prices.notna().all(axis=1)
    if not live.any():
        raise ValueError("no date has both a target and prices for every asset")
    first = int(np.flatnonzero(live.to_numpy())[0])
    returns = prices.pct_change().fillna(0.0).to_numpy()
    tgt = target.to_numpy()
    n = len(assets)
    if dividend_yield is None:
        y = np.zeros(n)
    elif np.isscalar(dividend_yield):
        y = np.full(n, float(dividend_yield))
    else:
        y = np.array([float(dividend_yield.get(a, 0.0)) for a in assets])
    daily_y = y / 252.0
    external = (flows.reindex(index).fillna(0.0) if flows is not None else pd.Series(0.0, index=index)).to_numpy(dtype=float).copy()
    external[:first + 1] = 0.0                                                                                  # the starting capital is the first deposit
    is_mark = index.isin(rebalance_dates(index, rebalance)) if rebalance != "none" else np.zeros(len(index), bool)
    is_pay = index.isin(rebalance_dates(index, "quarterly"))
    starts = np.flatnonzero(index.isin(period_starts(index, "monthly")))
    cost = cost_bps * 1e-4
    cash_daily = cash_rate / 252.0

    value, accrued, cash = np.zeros(n), np.zeros(n), 0.0
    pending: list[tuple[int, float]] = []                                                                        # (day to invest, dollars) for dollar-cost averaging
    reserved = 0.0                                                                                              # cash set aside for those later tranches
    size = len(index)
    nav_h, cash_h, acc_h = np.full(size, np.nan), np.full(size, np.nan), np.full(size, np.nan)
    val_h, trade_h = np.full((size, n), np.nan), np.zeros((size, n))
    cost_h, div_h, twr = np.zeros(size), np.zeros(size), np.full(size, np.nan)
    prev_nav, ruined = 0.0, False
    for t in range(first, size):
        trades = np.zeros(n)
        paid = np.zeros(n)
        tg = pd.Series(tgt[t], index=assets)
        if t == first:
            cash = float(initial)
            tranche = initial / dca_months
            trades = tgt[t] * tranche
            later = [int(d) for d in starts if d > t][:dca_months - 1]
            pending = [(d, tranche) for d in later]
            reserved = tranche * len(later)
        else:
            gain = value * daily_y
            value = value * (1.0 + returns[t] - daily_y)
            accrued = accrued + gain
            cash *= 1.0 + cash_daily
            if ruined or value.sum() + accrued.sum() + cash <= 0:
                ruined = True
                nav_h[t] = cash_h[t] = acc_h[t] = 0.0
                val_h[t] = 0.0
                twr[t] = -1.0 if prev_nav > 0 else 0.0
                prev_nav = 0.0
                value, accrued, cash, reserved, pending = np.zeros(n), np.zeros(n), 0.0, 0.0, []
                external[t] = 0.0
                continue
            free = cash - reserved                                                                              # cash not set aside for later tranches, before today's dividends and flow
            if is_pay[t] and accrued.sum() > 0:
                paid, accrued = accrued.copy(), np.zeros(n)
                cash += paid.sum()
                div_h[t] = paid.sum()
            f = external[t]
            if f < 0:
                f = -min(-f, float(value.sum()) + max(free, 0.0) + float(paid.sum()))                              # a withdrawal cannot exceed what the portfolio holds
                external[t] = f
            cash += f
            invest = 0.0
            if f > 0 and dca_months > 1:
                tranche = f / dca_months
                later = [int(d) for d in starts if d > t][:dca_months - 1]
                pending += [(d, tranche) for d in later]
                reserved += tranche * len(later)
                invest = tranche
            elif f > 0:
                invest = f
            due = [p for p in pending if p[0] == t]
            if due:
                pending = [p for p in pending if p[0] != t]
                invest += sum(a for _, a in due)
                reserved -= sum(a for _, a in due)
            if paid.sum() > 0 and dividend_policy == "reinvest":
                trades = trades + paid
            elif paid.sum() > 0 and dividend_policy == "flow":
                invest += paid.sum()
            raise_cash = -f if f < 0 else 0.0
            net = invest - raise_cash
            vals = pd.Series(value, index=assets)
            if net > 0:
                trades = trades + allocate_flow(vals, tg, net, inflow_policy, free, liquidity).to_numpy()
            elif net < 0:
                trades = trades + allocate_flow(vals, tg, net, outflow_policy, free, liquidity).to_numpy()
            if is_mark[t]:
                after = value + trades
                free_nav = float(after.sum()) + cash - float(trades.sum()) - reserved                              # what is not set aside for later tranches
                trades = trades + (tg * max(free_nav, 0.0)).to_numpy() - after
        spent = float(trades.sum())
        fee = cost * float(np.abs(trades).sum())
        value = value + trades
        cash = cash - spent - fee
        nav = float(value.sum() + accrued.sum() + cash)
        if t > first:
            twr[t] = (nav - external[t]) / prev_nav - 1.0 if prev_nav > 0 else 0.0
        nav_h[t], cash_h[t], acc_h[t], val_h[t] = nav, cash, accrued.sum(), value
        trade_h[t], cost_h[t] = trades, fee
        prev_nav = nav
    result = CashflowResult(pd.Series(nav_h, index=index), pd.DataFrame(val_h, index=index, columns=assets), pd.Series(cash_h, index=index), pd.Series(acc_h, index=index),
                            pd.Series(external, index=index), pd.DataFrame(trade_h, index=index, columns=assets), pd.Series(cost_h, index=index), pd.Series(div_h, index=index),
                            pd.Series(twr, index=index), target, {})
    result.summary = _summary(result, first, initial)
    return result


def _summary(r: CashflowResult, first: int, initial: float) -> dict:
    idx = r.nav.index
    nav = r.nav.iloc[first:]
    years = max((idx[-1] - idx[first]).days / 365.25, 1e-9)
    deposits, withdrawals = float(r.flows[r.flows > 0].sum()), float(-r.flows[r.flows < 0].sum())
    twr = r.twr.iloc[first + 1:].dropna()
    total = float((1.0 + twr).prod() - 1.0) if len(twr) else 0.0
    when = [i for i in np.flatnonzero(r.flows.to_numpy() != 0.0) if i > first]
    irr = money_weighted_return(pd.DatetimeIndex([idx[first]] + [idx[i] for i in when] + [idx[-1]]), np.array([0.0] + [float(r.flows.iloc[i]) for i in when] + [0.0]), float(nav.iloc[-1]), float(initial))
    traded, mean_nav = float(r.trades.iloc[first + 1:].abs().sum().sum()), float(nav.mean())                  # the first purchase of the starting capital is not turnover
    held = r.deviation.iloc[first:]
    return {"start": str(idx[first].date()), "end": str(idx[-1].date()), "years": years, "initial": float(initial), "deposits": deposits, "withdrawals": withdrawals,
            "final_value": float(nav.iloc[-1]), "profit": float(nav.iloc[-1]) - float(initial) - deposits + withdrawals, "twr_total": total,
            "twr_annual": float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0, "irr": irr,
            "turnover": traded / mean_nav / years if mean_nav > 0 else float("nan"), "costs": float(r.costs.sum()), "dividends": float(r.dividends.sum()),
            "mean_deviation": float(held.mean()), "max_deviation": float(held.max()), "mean_cash": float((r.cash.iloc[first:] / nav.replace(0.0, np.nan)).mean())}
