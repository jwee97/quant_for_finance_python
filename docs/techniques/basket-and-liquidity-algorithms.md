---
title: "Basket algorithms, minimum trading risk, program-block decomposition and liquidity seeking"
slug: basket-and-liquidity-algorithms
difficulty: 3
chapter: Ch. 22
prerequisites: [execution-algorithms, mean-variance-and-shrinkage]
stages: []
files: [src/algo/basket.py, src/algo/liquidity.py, src/algo/simulate.py]
figures: []
tests: [tests/test_basket.py, tests/test_algo.py]
models: []
---

# Basket algorithms, minimum trading risk, program-block decomposition and liquidity seeking

## In one sentence

When many stocks are traded together, what is left unexecuted is a portfolio with its own risk, so the order in which names are traded matters; and when the market turns thin and wide, a good algorithm slows down instead of paying for the pace.

## The idea

**Basket (portfolio) algorithms.** A basket is a trade list of signed shares with prices, volumes, volatilities and a correlation matrix. While it is worked, the part not yet executed is a portfolio: a buy of one bank against a sale of another is nearly hedged, and executing only one leg leaves a bet on banks. `basket_schedule` chooses shares for every stock and interval to minimise the stocks' own impact costs plus `risk_aversion` times the variance of the unexecuted exposure, `sum_i w_i e_i' Sigma e_i` with `e_i = sign * price * shares left`. At zero risk aversion every stock follows its own volume (a VWAP basket); as it grows the schedule hedges first, executing the legs that reduce the remaining risk most and keeping offsetting legs together. `independent_schedule` solves each stock alone at the same dollar price of risk; the joint answer is measured against it.

**Minimum trading risk quantity.** Given how much of the list's value you will execute, which shares to execute so the remaining list has the least price risk (executed fractions in [0, 1] minimising the risk of what is left), or the smallest share of the list that brings the remaining risk down to a target. Executing in proportion is the naive answer; the optimum executes the hedged legs together.

**Maximum trading opportunity.** Given what can be executed right now (a dark pool's offer, a block, the participation limit), the most value that can go without the remaining list being riskier than a limit (default: the original list's risk). Executing only the available long legs of a hedged list can leave a naked short; this is the bound on that, with a flag when the limit cannot be met and the lowest-risk fallback.

**Program-block decomposition.** Split the list into block names (large against their volume, too much for the lit market) and program names, and choose which block names may go to dark pools without raising risk: every subset of them that might fill, and nothing else, must leave a remaining list no riskier than the original (the worst case over subsets, exact up to twelve names), because dark fills are uncertain.

**Liquidity seeking.** A volume-following algorithm trades its schedule whatever the market looks like. `liquidity_seeking` reads what it can see each interval, `quality = (volume printing / expected) / (spread / normal spread * depth shortfall)`, and scales the pace by `clip(quality ** sensitivity, floor, cap)`: heavy volume in a tight deep market trades more, a crisis (spreads four times wider, a book two and a half times thinner) trades a `floor` of the base pace and waits. What it holds back is spread over the rest of the plan, so the order is still finished.

## Why it matters

Trading a list stock by stock ignores the one thing a portfolio manager cares about: the risk of the book while it is half done. The three tactics are the difference between a trade list that leaves a day of unintended market exposure and one that does not. Liquidity seeking is the execution-side answer to the crowded exit: it pays for waiting with price exposure instead of paying for speed with impact and spread.

## How this repo uses it

`src/algo/basket.py` has the schedule, a correlated-basket simulator (`simulate_basket`) that reproduces the closed forms the schedule is built from, and the three tactics; `src/algo/liquidity.py` the wrapper (`liquidity_seeking`, or `LiquiditySeeking(ImplementationShortfall(1e-3))`). `quant algo basket` runs a demonstration list and `quant algo run --algos liquidity_seeking --scenario crisis` the crisis day; the dashboard's Execution tab does both with a form (its Basket mode and its Compare mode). The tests state only what is guaranteed: the joint schedule is never worse than stock by stock in the model it optimises (with exact power-law polish, or with linear impact), the minimum-risk quantity never leaves more risk than proportional execution, the dark-safe set is safe for every fill pattern, and liquidity seeking lowers participation in the stress window and finishes the order.

## What we found

An eight-name list of $20m gross with net exposure of $0.1m: the joint schedule had an objective (`cost + risk aversion * risk^2`, risk aversion 0.001) of 7.30 against 7.36 stock by stock, at 6.78 against 6.95 bps of cost and 22.9 against 20.2 bps of risk: a small gain here, because a random list is not very hedged; hedged lists gain more. Executing half the list's value in the best order left $44,094 of risk against $49,644 for proportional execution and $99,288 for the whole list. If only the buys were on offer, 23% of the list's value could be executed before the remaining list was riskier than the original; seven of the eight names were blocks and one of them could go to a dark pool without raising risk. In the crisis scenario, 1,500 simulated days, a seller using `liquidity_seeking` paid 44.5 bps against 41.6 for VWAP, with more dispersion (239 against 207 bps); a buyer in the same scenario paid -16.6 against -10.5 bps. The price falls in a crisis, so waiting hurts a seller and helps a buyer: liquidity seeking trades impact for exposure, and the sign of the drift decides whether the exposure pays.

## Going deeper

```
basket:       minimise  sum_k cost_k(q_k)  +  lambda * sum_i w_i  e_i' Sigma e_i          e_i = s * p * x_i   (sign, price, shares left after interval i)
MTRQ:         minimise  risk( x - theta * x )       s.t.  value( theta * x ) = share * value( x ),   0 <= theta <= 1
MTO:          maximise  value( theta * x )          s.t.  theta * x <= available,   risk( x - theta * x ) <= limit
program-block: dark-safe set D  iff  for every subset S of D:  risk( x - S ) <= cap * risk( x )
liquidity:    pace_i = base_i * clip( quality_i ** sensitivity,  floor,  cap )
```

## Pitfalls

- The correlation matrix of a basket is itself an estimate; a joint schedule that leans on a correlation of 0.9 is as good as that number.
- A dark fill is uncertain; a plan that needs it is not a plan. The program-block check is the discipline.
- Liquidity seeking has no view on direction. It reduces the cost of trading in bad conditions and increases the exposure to the price while it waits.
- The "minimum risk" executions are only as good as the risk model; with few names and one factor they differ little from proportional execution.

## Try it

```bash
quant algo basket --size 12 --risk-aversion 0.002
quant algo run --algos vwap liquidity_seeking --scenario crisis --side sell --paths 1500
quant algo run --algos vwap liquidity_seeking --scenario crisis --side buy --paths 1500
```
