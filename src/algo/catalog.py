"""Where every strategy and algorithm of the taxonomy lives in this repository.

The list below follows the taxonomy of investment, portfolio and trading strategies (alpha, rebalance, risk, cash flow and economic-outlook strategies; trading styles; single-stock, basket, black-box and
liquidity-seeking algorithms; high-frequency strategies; best-execution goals; adaptation tactics; schedule optimisation; advanced execution tactics) item by item, and says for each one

* ``status``: ``built`` (written for this item), ``existing`` (the repository already had it; the entry points to it), ``proxy`` (built on prices and volume alone because the real input, such as news or
  index membership, is not in the repository: the entry says what to supply), ``data`` (built, and reads a file you supply, such as headlines or events, saying so when it is missing),
  ``simulator`` (a research simulator on a stylised market, not a tradable system);
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


S1A, S1B, S1C, S1D, S1E = ("1a. Investment: alpha generating", "1b. Investment: portfolio rebalance", "1c. Investment: risk management", "1d. Investment: cash flow", "1e. Investment: economic outlook")
S2, S3, S4, S5, S6, S7, S8 = ("2. Trading algorithm styles", "3. Specific algorithm types", "4. High-frequency trading", "5. Best execution goals", "6. Adaptation tactics",
                              "7. Schedule and portfolio optimisation", "8. Advanced execution and risk tactics")

ITEMS: list[Item] = [
    Item(S1A, "Long-term", "built", ("model:jensen_alpha", "model:momentum_13612w", "model:value_proxy", "model:quality_proxy", "model:low_volatility"), "persistent market-adjusted return (appraisal ratio) of the last three years; the others are the existing long-horizon factors"),
    Item(S1A, "Short-term", "built", ("model:adaptive_autocorrelation", "model:squeeze_breakout", "model:rsi2", "model:ibs_reversion", "model:short_term_reversal"), "continue or fade yesterday's move by the asset's own autocorrelation; a volatility squeeze breakout; the existing pullback rules"),
    Item(S1A, "Company outlook", "data", ("model:panel_signal",), "any dated score by company (analyst revisions, earnings surprises, guidance) with a publication lag and an expiry: supply data/user/signals.csv"),
    Item(S1A, "Company news", "data", ("model:news_sentiment", "model:abnormal_volume_drift"), "headline tone from data/user/headlines.csv; the proxy from prices and volume alone is abnormal_volume_drift"),
    Item(S1A, "Corporate action", "data", ("model:event_study_drift",), "a walk-forward event study by event type: price shocks by default, or your events file (splits, buybacks, mergers, upgrades)"),
    Item(S1A, "Mispricing", "existing", ("model:cointegration_pairs", "model:kalman_pairs", "model:pca_residual", "model:sparse_basket", "py:src.algo.blackbox.simulate_etf_arbitrage"), "relative-value rules on pairs and baskets, and the ETF premium simulator"),
    Item(S1B, "Asset allocation", "built", ("model:policy_portfolio", "model:model_portfolio", "model:faber_gtaa", "model:paa", "model:vaa", "model:daa", "model:adaptive_asset_allocation"), "strategic weights restored on a calendar and/or a tolerance band; the others are the existing tactical rules"),
    Item(S1B, "Index reconstitution", "data", ("model:event_study_drift",), "index additions and deletions are events: supply them as a file of date, ticker and type (the repository has no index-membership feed)"),
    Item(S1B, "Market outlook", "built", ("model:market_outlook", "model:risk_on_off", "allocator:regime_switch"), "trend, momentum, breadth and calm of the risky assets tilt the book between risk and safety"),
    Item(S1B, "Market neutral", "built", ("allocator:beta_neutral", "model:pca_residual", "model:kalman_pairs"), "the book's market beta removed by projection or with one hedge instrument; the others are existing market-neutral rules"),
    Item(S1B, "Flight to quality", "built", ("model:flight_to_quality", "model:risk_on_off"), "a price-only stress gauge moves the book from risky assets to rates, fixed income and gold"),
    Item(S1B, "Model driven", "existing", ("model:ml_ridge", "model:ml_trees", "model:deep_window", "model:expression", "allocator:black_litterman"), "forecasts from learned models and formulas, turned into weights by an allocator"),
    Item(S1B, "Month-end rebalancing flow", "built", ("model:rebalancing_flow",), "not in the list, the rebalancing of other people: lean against the expected month-end flow of balanced funds"),
    Item(S1C, "Risk reduction", "existing", ("model:vol_managed_long", "model:garch_vol_managed", "model:evt_risk_managed", "allocator:min_variance", "py:src.framework.adaptive.RegimeRiskLimits"), "exposure scaled by volatility or tail risk; minimum variance; regime-dependent caps and drawdown de-risking"),
    Item(S1C, "Hedging", "built", ("allocator:beta_neutral",), "removes the market beta of any book"),
    Item(S1C, "Liquidation costs", "built", ("allocator:liquidity_cap", "py:src.algo.liquidation.liquidation_profile", "py:src.algo.liquidation.liquidation_horizon", "allocator:tca_mvo"), "days and cost to sell a book at a share of volume; positions capped by what can be sold; trading costs inside the optimiser"),
    Item(S1D, "Cash deposit", "built", ("py:src.cashflow.policies.allocate_flow", "py:src.cashflow.simulate.simulate_cashflows"), "pro rata, drift-correcting, rebalance, cash or dollar-cost-averaged deposits, followed day by day"),
    Item(S1D, "Redemption", "built", ("py:src.cashflow.redemption.redemption_cost", "py:src.cashflow.redemption.compare_redemption_policies"), "which assets to sell for a redemption and what it costs under the liquidation model"),
    Item(S1D, "Cash dividend", "built", ("py:src.cashflow.simulate.simulate_cashflows",), "dividends reinvested in their asset, put through the deposit policy or held as cash (the dividend_policy setting)"),
    Item(S1D, "Liabilities", "built", ("py:src.cashflow.liabilities.Liability", "py:src.cashflow.liabilities.simulate_ldi"), "liability value, duration and funding ratio; liability-driven investing with a glide path"),
    Item(S1D, "Payments", "built", ("py:src.cashflow.spending.simulate_spending", "py:src.cashflow.spending.sustainable_rate"), "spending rules (fixed real, percent of value, endowment, guardrails) over bootstrapped markets: ruin and the sustainable rate"),
    Item(S1E, "Yield curve strategy", "built", ("model:curve_quadrant", "model:curve_steepener", "model:butterfly", "model:duration_timing", "model:carry_rolldown", "model:yield_curve_regime"), "the four level-and-slope states with consequences learned from earlier data; the others are the existing curve trades"),
    Item(S1E, "Credit strategy", "built", ("model:credit_cycle_rotation", "model:credit_spread_timing"), "the four phases of the credit cycle with consequences learned from earlier data; credit spread timing already existed"),
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
    Item(S7, "Portfolio optimisation with TCA (third wave)", "built", ("allocator:tca_mvo", "py:src.algo.basket.basket_schedule"), "mean-variance with spread and impact costs inside the optimiser, and the joint schedule of a basket"),
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


INTRO = """# Algorithmic trading and investment strategies: where everything is

This page maps the taxonomy of investment, portfolio and trading strategies, item by item, to the code that implements it. It is generated from `src/algo/catalog.py` (`python -m src.algo.catalog`), and a test imports every location it names, so it cannot describe something that is not there.

**Status.** *built*: written for this item. *existing*: the repository already had it, and the entry points to it. *data*: built, and reads a file you supply (`data/user/headlines.csv`, `signals.csv`, `events.csv`); it refuses to run without it instead of inventing a signal. *proxy*: made from prices and volume because the real input is not here. *simulator*: a research simulator on a stylised market, not a tradable system.

**Where to run things.** Strategies and allocators (sections 1a to 1e) are in the dashboard's strategy and portfolio lists and in `quant backtest --model NAME`. Cash-flow tools are `quant cashflow simulate|spending|redeem|ldi` and the dashboard's Cash flows tab. Execution algorithms and simulators (sections 2 to 8) are `quant algo list|run|frontier|basket|hft` and the dashboard's Execution tab. Guides: [alpha-generating styles](techniques/alpha-generating-styles.md), [portfolio rebalancing styles](techniques/portfolio-rebalancing-styles.md), [economic-outlook strategies](techniques/economic-outlook-strategies.md), [portfolio overlays and liquidation costs](techniques/portfolio-overlays-and-liquidation.md), [cash-flow strategies](techniques/cash-flow-strategies.md), [execution algorithms](techniques/execution-algorithms.md), [basket and liquidity algorithms](techniques/basket-and-liquidity-algorithms.md), [black-box and high-frequency strategies](techniques/black-box-and-high-frequency-strategies.md). The 15-ETF results for the new strategies are in [strategy survey, part two](strategy_survey_2.md).
"""

LIMITS = """## What this does not do

- **No news, analyst or index-membership feed.** The company-news, company-outlook, corporate-action and index-reconstitution strategies read files you supply, and the price-and-volume proxy for news is only a proxy. Dates in those files must be the first day the information could have been traded on.
- **The execution and high-frequency parts are simulators.** They use a stylised market (U-shaped volume, a price walk, a power-law impact whose parameters are illustrative values from the equity literature), no order book, queue, latency model beyond a parameter, or other participants. They compare methods and show which way a parameter pushes; they say nothing about whether an edge exists in a real market. Calibrate the impact parameters to your own executions before trusting a number.
- **Several one-line definitions have more than one reasonable reading** (adaptation tactics such as AIM and PIM, minimum trading risk quantity, maximum trading opportunity, program-block decomposition). The docstring of each says which reading was implemented.
- **Short-horizon rules do not survive costs on liquid ETFs.** The survey shows it; run them on the assets and costs you trade.
"""


def markdown() -> str:
    """The coverage page, written from the catalogue."""
    lines = [INTRO]
    for section in dict.fromkeys(i.section for i in ITEMS):
        lines += [f"## {section}", "", "| Item | Status | Where | What it does |", "|---|---|---|---|"]
        for i in (x for x in ITEMS if x.section == section):
            places = ", ".join(f"`{w.split(':', 1)[1]}`" + ("" if w.startswith("py:") else f" ({w.split(':', 1)[0]})") for w in i.where)
            lines.append(f"| {i.item} | {i.status} | {places} | {i.note} |")
        lines.append("")
    lines.append(LIMITS)
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "docs" / "algorithmic_trading.md"
    out.write_text(markdown(), encoding="utf-8")
    print(f"wrote {out}")
