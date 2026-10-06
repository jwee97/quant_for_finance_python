---
title: "Reading performance honestly"
slug: performance-metrics
difficulty: 1
chapter: Ch. 22
prerequisites: [backtest-engine-and-costs]
stages: [6, 12]
files: [src/backtest/metrics.py, experiments/stage12_results.py]
figures: [25]
tests: [tests/test_backtest.py]
models: []
---

# Reading performance honestly

## In one sentence

A single Sharpe ratio hides drawdowns, tails and luck; a fair summary reports several measures and how uncertain each one is.

## The idea

The Sharpe ratio divides mean excess return by volatility. Sortino penalises only downside volatility; Calmar divides return by the worst drawdown; CVaR is the average loss in the worst
days. Each answers a different question, and each is itself a noisy estimate: over ten years, a true Sharpe of 0.5 is statistically very hard to distinguish from zero.

## Why it matters

Two strategies with the same Sharpe can be very different to own: one may lose 40% once, the other bleed slowly. Reporting only the headline number is how fragile strategies get sold.

## How this repo uses it

`src/backtest/metrics.py` computes the full set (CAGR, volatility, Sharpe, Sortino, Calmar, omega, drawdown, tail ratio, turnover, cost drag). Stage 12 shows the Generation 1 ladder side by
side and opens the final holdout once, after model selection was frozen.

## What we found

Differences in Sharpe between the Generation 1 books are small compared with their estimation error; the paired bootstrap tests in later stages quantify that.

## Going deeper

```
Sharpe  = sqrt(252) * mean(r - rf) / std(r)
Sortino = sqrt(252) * mean(r - rf) / downside_std(r)
Calmar  = CAGR / |max drawdown|
CVaR_5% = - mean( r | r <= 5th percentile )
standard error of Sharpe ~ sqrt((1 + 0.5 SR^2) / T_years)      (for iid returns)
```
For SR = 0.5 over 10 years the standard error is about 0.32, so the 95% interval is roughly (-0.1, 1.1): a good strategy over a decade is statistically close to indistinguishable from nothing.

## Pitfalls

- Annualising short samples makes noise look like skill.
- Skewed and fat-tailed returns make Sharpe flattering; look at drawdown and CVaR.
- Never compare Sharpe ratios from different sample periods.

## Try it

```bash
quant explain sharpe ratio
quant explain deflated sharpe
```
