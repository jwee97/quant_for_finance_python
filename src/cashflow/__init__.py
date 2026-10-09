"""Cash-flow strategies: what to trade when money moves in or out of a portfolio, and what a stream of liabilities or payments asks of it.

    policies      deposit, redemption and dividend handling as dollar trades: pro rata, drift-correcting, full rebalance, cash, most liquid first
    flows         dated external flows: contributions and withdrawals as a schedule
    simulate      a portfolio followed day by day through flows, dividends, rebalances and costs: money-weighted against time-weighted return, turnover, drift
    redemption    what a redemption costs to meet under each policy, with the liquidation cost model of the execution algorithms
    liabilities   liability cash flows, their duration and value, the funding ratio and liability-driven investing with a hedge ratio that follows it
    spending      payments out of a portfolio (fixed real, percent of value, endowment, guardrails) over bootstrapped markets: ruin, sustainable spending
"""

from . import flows, liabilities, policies, redemption, simulate, spending  # noqa: F401
