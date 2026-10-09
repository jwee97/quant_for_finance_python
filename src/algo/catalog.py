"""Where every strategy and algorithm of the taxonomy lives in this repository.

The list below follows the taxonomy of investment, portfolio and trading strategies (alpha, rebalance, risk, cash flow and economic-outlook strategies; trading styles; single-stock, basket, black-box and
liquidity-seeking algorithms; high-frequency strategies; best-execution goals; adaptation tactics; schedule optimisation; advanced execution tactics) item by item, and says for each one

* ``status``: ``built`` (written for this item), ``existing`` (the repository already had it; the entry points to it), ``proxy`` (built on prices and volume alone because the real input, such as news or
  index membership, is not in the repository: the entry says what to supply), ``simulator`` (a research simulator on a stylised market, not a tradable system);
* ``where``: how to find it: ``py:module.object`` for Python code, ``model:name`` for a strategy in the model registry, ``allocator:name`` for a portfolio allocator.

``tests/test_algo_catalog.py`` imports every ``where``, so this file cannot describe something that is not there; ``docs/algorithmic_trading.md`` is written from it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    section: str
    item: str
    status: str
    where: tuple
    note: str = ""


S2, S3, S4, S5, S6, S7, S8 = ("2. Trading algorithm styles", "3. Specific algorithm types", "4. High-frequency trading", "5. Best execution goals", "6. Adaptation tactics",
                              "7. Schedule and portfolio optimisation", "8. Advanced execution and risk tactics")

ITEMS: list[Item] = [
    Item(S2, "Aggressive", "built", ("py:src.algo.simulate.AGGRESSIVE",), "every share crosses the spread: follows the schedule exactly, pays spread and impact"),
    Item(S2, "Working order", "built", ("py:src.algo.simulate.WORKING",), "a mix of limit, dark and market orders, topped up with market orders when behind"),
    Item(S2, "Passive", "built", ("py:src.algo.simulate.PASSIVE",), "mostly limit orders and dark pools: earns the spread, leaks little, may not fill"),
    Item(S3, "VWAP", "built", ("py:src.algo.algos.VWAP",), "slices follow the expected volume profile"),
    Item(S3, "TWAP", "built", ("py:src.algo.algos.TWAP",), "equal slices"),
    Item(S3, "POV / Volume", "built", ("py:src.algo.algos.POV",), "a fixed share of the volume as it prints"),
    Item(S3, "Arrival price", "built", ("py:src.algo.algos.ArrivalPrice",), "front-loaded to stay near the arrival price, by an urgency parameter"),
    Item(S3, "Implementation shortfall", "built", ("py:src.algo.algos.ImplementationShortfall",), "the Almgren-Chriss optimum for a risk aversion, with an optional view on drift"),
    Item(S3, "Basket / portfolio algorithms", "built", ("py:src.algo.basket.basket_schedule", "py:src.algo.basket.simulate_basket"), "cost against the risk of the whole unexecuted list, by a risk aversion"),
    Item(S3, "Black-box: pair trading", "simulator", ("py:src.algo.blackbox.simulate_pair_trading", "model:kalman_pairs", "model:cointegration_pairs"), "the intraday simulator; the daily pairs strategies already existed"),
    Item(S3, "Black-box: auto market making", "existing", ("py:src.algo.hft.auto_market_making", "py:src.microstructure.market_making.simulate_market_making"), "Avellaneda-Stoikov, in the algo vocabulary"),
    Item(S3, "Black-box: statistical arbitrage", "simulator", ("py:src.algo.blackbox.simulate_etf_arbitrage", "model:pca_residual", "model:sparse_basket"), "ETF against its basket intraday; the daily residual strategies already existed"),
    Item(S3, "Liquidity seeking", "built", ("py:src.algo.liquidity.LiquiditySeeking",), "backs off when spreads widen, depth thins or volume dries up, and catches up later"),
    Item(S4, "Auto market making (AMM)", "existing", ("py:src.microstructure.market_making.simulate_market_making", "py:src.algo.hft.auto_market_making"), "inventory-shaded quotes against symmetric quotes"),
    Item(S4, "Quantitative trading / statistical arbitrage", "simulator", ("py:src.algo.blackbox.simulate_etf_arbitrage", "py:src.algo.blackbox.latency_table"), "the premium of an ETF to its net asset value, and what latency does to it"),
    Item(S4, "Rebate / liquidity trading", "simulator", ("py:src.algo.hft.simulate_rebate_trading", "py:src.algo.hft.pressure_table"), "a maker that infers order-flow pressure and withdraws the side about to be hit"),
    Item(S5, "Minimise cost", "built", ("py:src.algo.algos.MinCost", "py:src.algo.optimize.min_cost"), "the cheapest schedule: VWAP unless there is a view on drift"),
    Item(S5, "Minimise cost with a risk constraint", "built", ("py:src.algo.algos.MinCostRisk", "py:src.algo.optimize.min_cost_given_risk"), "the cheapest schedule whose timing risk is under a limit"),
    Item(S5, "Minimise risk with a cost constraint", "built", ("py:src.algo.algos.MinRiskCost", "py:src.algo.optimize.min_risk_given_cost"), "the least risky schedule whose expected cost is under a limit"),
    Item(S5, "Balance cost and risk", "built", ("py:src.algo.algos.Balanced", "py:src.algo.optimize.balanced"), "the standard trade-off for a risk aversion"),
    Item(S5, "Price improvement", "built", ("py:src.algo.algos.PriceImprovement", "py:src.algo.optimize.price_improvement"), "maximises the probability of beating a target cost"),
    Item(S6, "Target cost", "built", ("py:src.algo.tactics.TargetCost",), "re-chooses the schedule so the impact paid stays on the plan's expected figure"),
    Item(S6, "Aggressive in the money (AIM)", "built", ("py:src.algo.tactics.AIM",), "speeds up when the price is in your favour"),
    Item(S6, "Passive in the money (PIM)", "built", ("py:src.algo.tactics.PIM",), "slows down when the price is in your favour and speeds up to limit a loss"),
    Item(S7, "Quadratic programming", "built", ("py:src.algo.optimize.optimal_schedule", "py:src.algo.optimize.solve_qp_eq"), "an exact active-set solver for the cost-plus-risk problem"),
    Item(S7, "Trade schedule exponential", "built", ("py:src.algo.optimize.exponential_trade", "py:src.algo.optimize.fit_exponential_trade"), "the trade rate decays like exp(-kappa t)"),
    Item(S7, "Residual schedule exponential", "built", ("py:src.algo.optimize.exponential_residual", "py:src.algo.optimize.fit_exponential_residual"), "the shares left decay like exp(-kappa t)"),
    Item(S7, "Trade rate parameter", "built", ("py:src.algo.optimize.trade_rate", "py:src.algo.optimize.fit_trade_rate"), "one participation rate describes the strategy"),
    Item(S8, "Minimum trading risk quantity", "built", ("py:src.algo.basket.minimum_trading_risk_quantity",), "which shares to execute so the remaining list has the least risk"),
    Item(S8, "Maximum trading opportunity", "built", ("py:src.algo.basket.maximum_trading_opportunity",), "the most that can be executed now without the remaining list getting riskier"),
    Item(S8, "Program-block decomposition", "built", ("py:src.algo.basket.program_block",), "which block names can go to dark pools without raising risk"),
]


def resolve(where: str):
    """The Python object, registry entry or allocator that ``where`` names; raises if it does not exist."""
    kind, _, target = where.partition(":")
    if kind == "py":
        module, _, attr = target.rpartition(".")
        return getattr(importlib.import_module(module), attr)
    if kind == "model":
        from ..framework import MODELS, load_library

        load_library()
        return MODELS._entries[target].factory
    if kind == "allocator":
        from ..framework import ALLOCATORS, load_library

        load_library()
        return ALLOCATORS._entries[target].factory
    raise ValueError(f"unknown location '{where}'")


def table():
    """The catalogue as a data frame, one row per item."""
    import pandas as pd

    return pd.DataFrame([{"section": i.section, "item": i.item, "status": i.status, "where": ", ".join(w.split(":", 1)[1] for w in i.where), "note": i.note} for i in ITEMS])
