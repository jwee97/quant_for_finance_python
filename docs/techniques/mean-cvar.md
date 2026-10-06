---
title: "Mean-CVaR optimisation: optimising the tail"
slug: mean-cvar
difficulty: 3
chapter: Ch. 21
prerequisites: [var-cvar-and-backtests]
stages: [7, 9]
files: [src/portfolio/cvar_optimize.py, src/risk/cvar.py]
figures: [19, 20]
tests: [tests/test_portfolio.py, tests/test_risk.py]
models: []
---

# Mean-CVaR optimisation: optimising the tail

## In one sentence

Instead of minimising variance, choose weights that minimise the average loss in the worst few percent of scenarios.

## The idea

Conditional value at risk (CVaR, expected shortfall) is the mean loss beyond the VaR quantile. Rockafellar and Uryasev showed that minimising it over historical scenarios is a linear
programme, so it scales and has a global optimum. Unlike variance it cares about the left tail only and about skew.

## Why it matters

Variance treats a +5% day and a -5% day identically. Investors do not. When returns are fat-tailed (all 15 ETFs reject normality), tail-aware optimisation can allocate differently.

## How this repo uses it

`src/portfolio/cvar_optimize.py` is the scenario linear programme; the adaptive allocator in Stage 31 uses it as the Crisis book.

## What we found

Mean-CVaR books behave defensibly in crises but are not reliably better after costs; in the adaptive study the allocation switch to it was not shown to help beyond what risk parity gives.

## Going deeper

```
CVaR_a(w) = min_z  z + 1/((1-a) S) * sum_s max( -w'r_s - z , 0 )          Rockafellar-Uryasev
minimise CVaR_a(w) subject to w'mu >= target, sum w = 1, w >= 0
```
The `max(., 0)` terms become auxiliary variables `u_s >= -w'r_s - z, u_s >= 0`, which makes the whole problem a linear programme in `(w, z, u)`.

## Pitfalls

- A few hundred scenarios make the 5% tail only a handful of observations.
- Historical scenarios contain no event that has not happened.
- The optimum can concentrate in assets whose tails are merely absent from the sample.

## Try it

```bash
quant backtest --allocator static --model momentum
```
