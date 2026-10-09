# Algorithmic trading and investment strategies: where everything is

This page maps the taxonomy of investment, portfolio and trading strategies, item by item, to the code that implements it. It is generated from `src/algo/catalog.py` (`python -m src.algo.catalog`), and a test imports every location it names, so it cannot describe something that is not there.

**Status.** *built*: written for this item. *existing*: the repository already had it, and the entry points to it. *data*: built, and reads a file you supply (`data/user/headlines.csv`, `signals.csv`, `events.csv`); it refuses to run without it instead of inventing a signal. *proxy*: made from prices and volume because the real input is not here. *simulator*: a research simulator on a stylised market, not a tradable system.

**Where to run things.** Strategies and allocators (sections 1a to 1e) are in the dashboard's strategy and portfolio lists and in `quant backtest --model NAME`. Cash-flow tools are `quant cashflow simulate|spending|redeem|ldi` and the dashboard's Cash flows tab. Execution algorithms and simulators (sections 2 to 8) are `quant algo list|run|frontier|basket|hft` and the dashboard's Execution tab. Guides: [alpha-generating styles](techniques/alpha-generating-styles.md), [portfolio rebalancing styles](techniques/portfolio-rebalancing-styles.md), [economic-outlook strategies](techniques/economic-outlook-strategies.md), [portfolio overlays and liquidation costs](techniques/portfolio-overlays-and-liquidation.md), [cash-flow strategies](techniques/cash-flow-strategies.md), [execution algorithms](techniques/execution-algorithms.md), [basket and liquidity algorithms](techniques/basket-and-liquidity-algorithms.md), [black-box and high-frequency strategies](techniques/black-box-and-high-frequency-strategies.md). The 15-ETF results for the new strategies are in [strategy survey, part two](strategy_survey_2.md).

## 1a. Investment: alpha generating

| Item | Status | Where | What it does |
|---|---|---|---|
| Long-term | built | `jensen_alpha` (model), `momentum_13612w` (model), `value_proxy` (model), `quality_proxy` (model), `low_volatility` (model) | persistent market-adjusted return (appraisal ratio) of the last three years; the others are the existing long-horizon factors |
| Short-term | built | `adaptive_autocorrelation` (model), `squeeze_breakout` (model), `rsi2` (model), `ibs_reversion` (model), `short_term_reversal` (model) | continue or fade yesterday's move by the asset's own autocorrelation; a volatility squeeze breakout; the existing pullback rules |
| Company outlook | data | `panel_signal` (model) | any dated score by company (analyst revisions, earnings surprises, guidance) with a publication lag and an expiry: supply data/user/signals.csv |
| Company news | data | `news_sentiment` (model), `abnormal_volume_drift` (model) | headline tone from data/user/headlines.csv; the proxy from prices and volume alone is abnormal_volume_drift |
| Corporate action | data | `event_study_drift` (model) | a walk-forward event study by event type: price shocks by default, or your events file (splits, buybacks, mergers, upgrades) |
| Mispricing | existing | `cointegration_pairs` (model), `kalman_pairs` (model), `pca_residual` (model), `sparse_basket` (model), `src.algo.blackbox.simulate_etf_arbitrage` | relative-value rules on pairs and baskets, and the ETF premium simulator |

## 1b. Investment: portfolio rebalance

| Item | Status | Where | What it does |
|---|---|---|---|
| Asset allocation | built | `policy_portfolio` (model), `model_portfolio` (model), `faber_gtaa` (model), `paa` (model), `vaa` (model), `daa` (model), `adaptive_asset_allocation` (model) | strategic weights restored on a calendar and/or a tolerance band; the others are the existing tactical rules |
| Index reconstitution | data | `event_study_drift` (model) | index additions and deletions are events: supply them as a file of date, ticker and type (the repository has no index-membership feed) |
| Market outlook | built | `market_outlook` (model), `risk_on_off` (model), `regime_switch` (allocator) | trend, momentum, breadth and calm of the risky assets tilt the book between risk and safety |
| Market neutral | built | `beta_neutral` (allocator), `pca_residual` (model), `kalman_pairs` (model) | the book's market beta removed by projection or with one hedge instrument; the others are existing market-neutral rules |
| Flight to quality | built | `flight_to_quality` (model), `risk_on_off` (model) | a price-only stress gauge moves the book from risky assets to rates, fixed income and gold |
| Model driven | existing | `ml_ridge` (model), `ml_trees` (model), `deep_window` (model), `expression` (model), `black_litterman` (allocator) | forecasts from learned models and formulas, turned into weights by an allocator |
| Month-end rebalancing flow | built | `rebalancing_flow` (model) | not in the list, the rebalancing of other people: lean against the expected month-end flow of balanced funds |

## 1c. Investment: risk management

| Item | Status | Where | What it does |
|---|---|---|---|
| Risk reduction | existing | `vol_managed_long` (model), `garch_vol_managed` (model), `evt_risk_managed` (model), `min_variance` (allocator), `src.framework.adaptive.RegimeRiskLimits` | exposure scaled by volatility or tail risk; minimum variance; regime-dependent caps and drawdown de-risking |
| Hedging | built | `beta_neutral` (allocator) | removes the market beta of any book |
| Liquidation costs | built | `liquidity_cap` (allocator), `src.algo.liquidation.liquidation_profile`, `src.algo.liquidation.liquidation_horizon`, `tca_mvo` (allocator) | days and cost to sell a book at a share of volume; positions capped by what can be sold; trading costs inside the optimiser |

## 1d. Investment: cash flow

| Item | Status | Where | What it does |
|---|---|---|---|
| Cash deposit | built | `src.cashflow.policies.allocate_flow`, `src.cashflow.simulate.simulate_cashflows` | pro rata, drift-correcting, rebalance, cash or dollar-cost-averaged deposits, followed day by day |
| Redemption | built | `src.cashflow.redemption.redemption_cost`, `src.cashflow.redemption.compare_redemption_policies` | which assets to sell for a redemption and what it costs under the liquidation model |
| Cash dividend | built | `src.cashflow.simulate.simulate_cashflows` | dividends reinvested in their asset, put through the deposit policy or held as cash (the dividend_policy setting) |
| Liabilities | built | `src.cashflow.liabilities.Liability`, `src.cashflow.liabilities.simulate_ldi` | liability value, duration and funding ratio; liability-driven investing with a glide path |
| Payments | built | `src.cashflow.spending.simulate_spending`, `src.cashflow.spending.sustainable_rate` | spending rules (fixed real, percent of value, endowment, guardrails) over bootstrapped markets: ruin and the sustainable rate |

## 1e. Investment: economic outlook

| Item | Status | Where | What it does |
|---|---|---|---|
| Yield curve strategy | built | `curve_quadrant` (model), `curve_steepener` (model), `butterfly` (model), `duration_timing` (model), `carry_rolldown` (model), `yield_curve_regime` (model) | the four level-and-slope states with consequences learned from earlier data; the others are the existing curve trades |
| Credit strategy | built | `credit_cycle_rotation` (model), `credit_spread_timing` (model) | the four phases of the credit cycle with consequences learned from earlier data; credit spread timing already existed |

## 2. Trading algorithm styles

| Item | Status | Where | What it does |
|---|---|---|---|
| Aggressive | built | `src.algo.simulate.AGGRESSIVE` | every share crosses the spread: follows the schedule exactly, pays spread and impact |
| Working order | built | `src.algo.simulate.WORKING` | a mix of limit, dark and market orders, topped up with market orders when behind |
| Passive | built | `src.algo.simulate.PASSIVE` | mostly limit orders and dark pools: earns the spread, leaks little, may not fill |

## 3. Specific algorithm types

| Item | Status | Where | What it does |
|---|---|---|---|
| VWAP | built | `src.algo.algos.VWAP` | slices follow the expected volume profile |
| TWAP | built | `src.algo.algos.TWAP` | equal slices |
| POV / Volume | built | `src.algo.algos.POV` | a fixed share of the volume as it prints |
| Arrival price | built | `src.algo.algos.ArrivalPrice` | front-loaded to stay near the arrival price, by an urgency parameter |
| Implementation shortfall | built | `src.algo.algos.ImplementationShortfall` | the Almgren-Chriss optimum for a risk aversion, with an optional view on drift |
| Basket / portfolio algorithms | built | `src.algo.basket.basket_schedule`, `src.algo.basket.simulate_basket` | cost against the risk of the whole unexecuted list, by a risk aversion |
| Black-box: pair trading | simulator | `src.algo.blackbox.simulate_pair_trading`, `kalman_pairs` (model), `cointegration_pairs` (model) | the intraday simulator; the daily pairs strategies already existed |
| Black-box: auto market making | existing | `src.algo.hft.auto_market_making`, `src.microstructure.market_making.simulate_market_making` | Avellaneda-Stoikov, in the algo vocabulary |
| Black-box: statistical arbitrage | simulator | `src.algo.blackbox.simulate_etf_arbitrage`, `pca_residual` (model), `sparse_basket` (model) | ETF against its basket intraday; the daily residual strategies already existed |
| Liquidity seeking | built | `src.algo.liquidity.LiquiditySeeking` | backs off when spreads widen, depth thins or volume dries up, and catches up later |

## 4. High-frequency trading

| Item | Status | Where | What it does |
|---|---|---|---|
| Auto market making (AMM) | existing | `src.microstructure.market_making.simulate_market_making`, `src.algo.hft.auto_market_making` | inventory-shaded quotes against symmetric quotes |
| Quantitative trading / statistical arbitrage | simulator | `src.algo.blackbox.simulate_etf_arbitrage`, `src.algo.blackbox.latency_table` | the premium of an ETF to its net asset value, and what latency does to it |
| Rebate / liquidity trading | simulator | `src.algo.hft.simulate_rebate_trading`, `src.algo.hft.pressure_table` | a maker that infers order-flow pressure and withdraws the side about to be hit |

## 5. Best execution goals

| Item | Status | Where | What it does |
|---|---|---|---|
| Minimise cost | built | `src.algo.algos.MinCost`, `src.algo.optimize.min_cost` | the cheapest schedule: VWAP unless there is a view on drift |
| Minimise cost with a risk constraint | built | `src.algo.algos.MinCostRisk`, `src.algo.optimize.min_cost_given_risk` | the cheapest schedule whose timing risk is under a limit |
| Minimise risk with a cost constraint | built | `src.algo.algos.MinRiskCost`, `src.algo.optimize.min_risk_given_cost` | the least risky schedule whose expected cost is under a limit |
| Balance cost and risk | built | `src.algo.algos.Balanced`, `src.algo.optimize.balanced` | the standard trade-off for a risk aversion |
| Price improvement | built | `src.algo.algos.PriceImprovement`, `src.algo.optimize.price_improvement` | maximises the probability of beating a target cost |

## 6. Adaptation tactics

| Item | Status | Where | What it does |
|---|---|---|---|
| Target cost | built | `src.algo.tactics.TargetCost` | re-chooses the schedule so the impact paid stays on the plan's expected figure |
| Aggressive in the money (AIM) | built | `src.algo.tactics.AIM` | speeds up when the price is in your favour |
| Passive in the money (PIM) | built | `src.algo.tactics.PIM` | slows down when the price is in your favour and speeds up to limit a loss |

## 7. Schedule and portfolio optimisation

| Item | Status | Where | What it does |
|---|---|---|---|
| Quadratic programming | built | `src.algo.optimize.optimal_schedule`, `src.algo.optimize.solve_qp_eq` | an exact active-set solver for the cost-plus-risk problem |
| Trade schedule exponential | built | `src.algo.optimize.exponential_trade`, `src.algo.optimize.fit_exponential_trade` | the trade rate decays like exp(-kappa t) |
| Residual schedule exponential | built | `src.algo.optimize.exponential_residual`, `src.algo.optimize.fit_exponential_residual` | the shares left decay like exp(-kappa t) |
| Trade rate parameter | built | `src.algo.optimize.trade_rate`, `src.algo.optimize.fit_trade_rate` | one participation rate describes the strategy |
| Portfolio optimisation with TCA (third wave) | built | `tca_mvo` (allocator), `src.algo.basket.basket_schedule` | mean-variance with spread and impact costs inside the optimiser, and the joint schedule of a basket |

## 8. Advanced execution and risk tactics

| Item | Status | Where | What it does |
|---|---|---|---|
| Minimum trading risk quantity | built | `src.algo.basket.minimum_trading_risk_quantity` | which shares to execute so the remaining list has the least risk |
| Maximum trading opportunity | built | `src.algo.basket.maximum_trading_opportunity` | the most that can be executed now without the remaining list getting riskier |
| Program-block decomposition | built | `src.algo.basket.program_block` | which block names can go to dark pools without raising risk |

## What this does not do

- **No news, analyst or index-membership feed.** The company-news, company-outlook, corporate-action and index-reconstitution strategies read files you supply, and the price-and-volume proxy for news is only a proxy. Dates in those files must be the first day the information could have been traded on.
- **The execution and high-frequency parts are simulators.** They use a stylised market (U-shaped volume, a price walk, a power-law impact whose parameters are illustrative values from the equity literature), no order book, queue, latency model beyond a parameter, or other participants. They compare methods and show which way a parameter pushes; they say nothing about whether an edge exists in a real market. Calibrate the impact parameters to your own executions before trusting a number.
- **Several one-line definitions have more than one reasonable reading** (adaptation tactics such as AIM and PIM, minimum trading risk quantity, maximum trading opportunity, program-block decomposition). The docstring of each says which reading was implemented.
- **Short-horizon rules do not survive costs on liquid ETFs.** The survey shows it; run them on the assets and costs you trade.
