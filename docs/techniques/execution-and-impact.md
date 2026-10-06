---
title: "Execution: spread, impact and capacity"
slug: execution-and-impact
difficulty: 3
chapter: Ch. 22
prerequisites: [backtest-engine-and-costs]
stages: [25]
files: [src/backtest/execution.py, src/backtest/impact.py, experiments/stage25_execution.py]
figures: [50, 51]
tests: [tests/test_impact.py]
models: []
---

# Execution: spread, impact and capacity

## In one sentence

Trading costs are a spread, a market impact that grows with the square root of your size, slippage and commission, and they set how much money a strategy can run.

## The idea

Instead of a flat cost per unit of turnover, the cost of a trade of size Q in an asset with average daily volume ADV is spread/2 plus an impact proportional to volatility times the square root
of Q/ADV, plus commission. At zero assets under management the model reduces exactly to the flat-cost model. Capacity is the assets at which net Sharpe falls to half.

## Why it matters

Impact is why strategies that work at $10m fail at $1bn. A no-trade band (do not trade unless the position is far enough from target) trades tracking error for cost.

## How this repo uses it

`src/backtest/impact.py` has the square-root model; `execution.py` applies it to the book; Stage 25 reports capacity by book, a Sharpe-by-AUM grid and the effect of no-trade bands.

## What we found

At $1bn the impact-inclusive Sharpe of the risk-based books stays within 0.05 of the flat-cost Sharpe (EXP-068), but the momentum book is capacity constrained. A 1% no-trade band did not improve the
three active books after correction (EXP-069).

## Going deeper

```
cost per trade = 0.5 * spread + k * sigma_daily * sqrt( Q / ADV ) + commission          (as a fraction of notional)
Q = trade size in dollars = |delta weight| * AUM
capacity = AUM at which net Sharpe = 0.5 * linear-cost Sharpe
```
Impact grows with the square root of size, so doubling AUM raises the cost per dollar traded by about 41% and the total cost by about 183% when turnover is unchanged.

## Pitfalls

- The impact coefficient is an order-of-magnitude assumption; the stage shows sensitivity.
- ETF liquidity is better than the underlying baskets suggest.
- Capacity is a property of the strategy and the market, not of the code.

## Try it

```bash
python -m experiments.stage25_execution
```
