---
title: "How a backtest turns signals into returns"
slug: backtest-engine-and-costs
difficulty: 1
chapter: Ch. 22
prerequisites: [momentum]
stages: [6]
files: [src/backtest/engine.py, src/backtest/costs.py, experiments/stage06_backtest.py]
figures: [14, 15]
tests: [tests/test_backtest.py]
models: []
---

# How a backtest turns signals into returns

## In one sentence

A backtest engine lags the decision, charges trading costs on what actually changed, lets weights drift between rebalances, and reports net returns.

## The idea

On the decision date the strategy states target weights using only information through that date. The engine applies them one day later (the execution lag), so a signal can never trade at
the price it was computed from. Between rebalances weights drift with returns. Costs are charged on turnover. The result is a daily net return series that every metric reads.

## Why it matters

Most apparent alpha in backtests comes from three sins: look-ahead (using tomorrow's data today), ignoring costs, and survivorship. The engine's structure makes the first two hard to commit
by accident.

## How this repo uses it

`src/backtest/engine.py` is the single engine used by every stage, including the plugin framework and the newcomer CLI. `src/backtest/costs.py` is the linear cost model; Stage 25 replaces
it with a spread-plus-impact model for the capacity study.

## What we found

Gross-versus-net comparisons show that high-turnover strategies lose a large fraction of gross performance to costs, and the breakeven cost for each strategy is reported in Stage 6.

## Going deeper

```
decision at close of day t  ->  weights w_t     (information through day t only)
traded at day t+1:          position held from t+1 on
portfolio return_t+1 = sum_i w_i,t * r_i,t+1     minus cost_t+1
cost = c * sum_i | w_i,t - w_i,t-1_drifted |      c = 10 bps by default, per unit of one-way turnover
w_drifted_i = w_i (1 + r_i) / (1 + sum_j w_j r_j)   between rebalances
```
Turnover of 9.2x a year at 10 bps costs about 92 bps of one-way notional, which is the arithmetic behind M3 momentum's drop from a gross Sharpe of 0.40 to a net 0.31.

## Pitfalls

- Lagging by one day is the minimum; whether you can really trade at the next close or open is an assumption.
- Linear costs understate the cost of trading size; see execution and impact.
- Rebalancing frequency changes both turnover and the information used.

## Try it

```bash
quant backtest --model momentum --tearsheet
```
Read the generated tear sheet's red-flag list; every rule is stated next to its result.
