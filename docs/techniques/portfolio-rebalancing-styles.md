---
title: "Portfolio rebalancing styles: policy bands, month-end flows, flight to quality, market outlook"
slug: portfolio-rebalancing-styles
difficulty: 2
chapter: Ch. 20
prerequisites: [tactical-asset-allocation, risk-managed-exposure]
stages: []
files: [src/strategies/rebalance_styles.py, src/framework/allocators_overlay.py]
figures: []
tests: [tests/test_rebalance_styles.py, tests/test_allocators_overlay.py]
models: [policy_portfolio, rebalancing_flow, flight_to_quality, market_outlook]
---

# Portfolio rebalancing styles: policy bands, month-end flows, flight to quality, market outlook

## In one sentence

How a portfolio is brought back to its intended shape (on a calendar, when it drifts too far, ahead of other people's forced trades, away from risk in a panic, or by a view on the whole market) is a strategy in its own right.

## The idea

**Asset allocation.** `policy_portfolio` holds strategic weights (a preset such as 60/40, or your own `targets`) and decides each month-end whether to restore them: on a calendar (`calendar`: monthly, quarterly, semiannual, annual, never), when any weight has drifted more than `band` from target, or both. Between restores the weights drift with returns, and the weights the strategy states are the drifted ones, so the engine's own drift and cost agree with the rule. Restoring sells what rose and buys what fell, a mild contrarian bet; a wide band or no restoring lets the riskiest asset take over. Leland (1999) shows that with proportional costs the optimal rule is a no-trade region around the target, which is what a band is.

**Market neutral and hedged.** A rebalance can also remove an exposure instead of restoring a weight: the `beta_neutral` allocator (see [portfolio overlays](portfolio-overlays-and-liquidation.md)) removes a book's beta to the market by projection or with one hedge instrument.

**The month-end flow.** A fund that targets a mix finds its risky share too high after risky assets outperform and sells them near the month-end, and the reverse after they underperform. The size of the flow is visible from prices, and Harvey, Mazzoleni and Melone (NBER 33554, 2025) find that when pension funds are overweight stocks, equity returns fall by about 17 basis points over the next day, a pressure that fades within about two weeks. Funds that rebalance on a calendar do so at month- and quarter-ends, which is why the strategy looks at the last days of the month (the paper does not test a month-end window; that part is this repository's hypothesis). `rebalancing_flow` measures the relative return of the equal-weight risky basket over the safe basket since the last month-end, in units of its own history, and in the last `days` trading days of the month holds the opposite of the expected flow. Under the engine's one-day signal lag a position stamped on day s earns day s+2, so `lead=2` places the window so the book earns the last days of the month.

**Flight to quality.** Investors run from risky to safe assets together: Baele, Bekaert, Inghelbrecht and Wei (2020) identify flights to safety on the rare days when bonds beat stocks by a wide margin, and show they coincide with money moving out of equity funds into government bond and money market funds. `flight_to_quality` averages three price-only symptoms into a stress level in [0, 1]: the drawdown of the risky basket from its half-year high, its volatility against its own history, and how unusually negative the stock-bond correlation is (measured against its own history, because a negative correlation can last a decade without being stress). Below `threshold` the book is long risky assets in proportion to calm; above it, it moves to rates, fixed income and gold (gold ETFs are treated as havens, not as commodities) and shorts risky assets unless `short_risky` is off.

**Market outlook.** `market_outlook` makes a manager's market view mechanical: the equal-weight risky basket's distance from its trend average, its twelve-month return, the share of risky assets above their own averages (breadth) and the calm of its volatility are each squashed to [-1, 1] and averaged. The outlook times `tilt` is the score of every risky asset, and when negative the score of every safe asset; `bias` adds an opinion of your own.

**Index reconstitution** is covered by `event_study_drift` with an index-changes file ([alpha-generating styles](alpha-generating-styles.md)); **model-driven** rebalancing is every `ml`, `deep` and `expression` model.

## Why it matters

The rebalancing rule decides turnover, cost and what the portfolio is exposed to between rebalances, and the choice is rarely tested: a quarterly calendar and a 5% band are different strategies with different drawdowns. Stress gauges and market views are the same decision made faster, with the same risk that they cut risk after the loss or add it after the gain.

## How this repo uses it

`policy_portfolio` is a structured model (the weights are the strategy) and runs with the engine's monthly rebalance. The three others are per-asset rules that run as sleeves (`rebalancing_flow` and `flight_to_quality` daily, `market_outlook` weekly) and need a risky and a safe asset, which they find by asset class (equity, real estate, commodity and credit against rates and fixed income); with unlabelled tickers they say so. The tests prove the rebalancing arithmetic against closed forms (never rebalancing equals the drifted weights exactly; a band never leaves a month-end outside it), place the month-end window on the calendar, and check that a planted crash turns the stress gauge on and a steady rally makes the outlook bullish. All four are causal.

## What we found

On the 15 ETFs ([survey part two](../strategy_survey_2.md): 92 strategies counted as trials, 10 bps costs): `policy_portfolio` with its defaults (60/40, quarterly restore, 5% band) earned a net Sharpe of +0.83 with 1.4x turnover a year against +0.65 for equal weight over the same dates, the same as the static `model_portfolio` it is built on, so on these twenty years the restoring rule changed little. `flight_to_quality` earned +0.60 (equal weight +0.95 over its dates) with a 12% maximum drawdown and 5.8% volatility: it protects, and pays for it. `market_outlook` earned +0.14 at 4% volatility and `rebalancing_flow` +0.31 gross but +0.02 net of costs at 7.7x turnover a year: a small effect on liquid funds that costs eat. The month-end flow is an hypothesis about other people's behaviour; it needs a longer sample, individual securities or futures with tiny costs to be tested properly.

## Going deeper

```
policy_portfolio:   at each month-end:  w_drift = w * (1 + r) / sum(w * (1 + r));   restore to target if  months since last restore >= calendar
                    or  max_i | w_drift_i - target_i | > band;   state w_drift otherwise
rebalancing_flow:   run_t = sum of daily (risky - safe) returns since the last month-end;   z = run_t / sd(earlier months' run);
                    score(risky) = -tanh(z / 2) in the window, score(safe) = +tanh(z / 2);  0 elsewhere
flight_to_quality:  stress = mean( tanh(-drawdown / 0.10), tanh(max(vol_z, 0) / 2), tanh(max(bond_bid_z, 0) / 2) );
                    calm = clip(1 - stress / threshold, 0, 1);   flight = clip((stress - threshold) / (1 - threshold), 0, 1)
```

## Pitfalls

- A band has no effect if it is wider than any drift; a calendar has no effect if drift never matters. Check how often the rule actually restored.
- Rebalancing into a falling asset is only a good idea if the asset mean-reverts. In a trending market it adds to the loss.
- Stress gauges lag: by the time drawdown and volatility are high, part of the move has happened. The gain is avoiding the next part, if there is one.
- `policy_portfolio` states drifted weights, so run it with the engine's monthly rebalance; a weekly or daily engine rebalance would re-impose the target and defeat the band.

## Try it

```bash
quant backtest --model policy_portfolio --param preset=60_40 --param calendar=annual --param band=0.10
quant backtest --model flight_to_quality --allocator sleeves --tearsheet
quant backtest --model rebalancing_flow --allocator sleeves
```
