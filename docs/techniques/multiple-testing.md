---
title: "Multiple testing, false discovery and the deflated Sharpe ratio"
slug: multiple-testing
difficulty: 3
chapter: Ch. 22
prerequisites: [performance-metrics]
stages: [3, 11, 26]
files: [src/validation/multiple_testing.py, src/validation/forecast_tests.py, src/validation/robustness.py]
figures: [52, 53]
tests: [tests/test_multiple_testing.py, tests/test_robustness.py]
models: []
---

# Multiple testing, false discovery and the deflated Sharpe ratio

## In one sentence

If you try enough ideas, the best one will look good even if none works; the remedy is to count the ideas and demand more of the winner.

## The idea

Test 72 signals at the 5% level and about 3.6 will 'work' by chance. Benjamini-Hochberg controls the expected share of false discoveries among those declared. The deflated Sharpe ratio asks
whether the best Sharpe among N trials exceeds what N random strategies would show, given the sample length, skew and kurtosis. The probabilistic Sharpe ratio is the single-trial version.

## Why it matters

This is the single biggest source of fake alpha in published backtests, and the easiest to commit by accident: every parameter tweak is a trial.

## How this repo uses it

Declared decision families use Benjamini-Hochberg throughout. The experiment manager counts distinct specifications in a group and feeds that count to the deflated Sharpe ratio in every
tear sheet. Stage 26 runs White's Reality Check, Hansen's SPA and probability of backtest overfitting on a grid of 1,584 rules.

## What we found

In Stage 3, 15 of 72 momentum tests cleared the naive 5% threshold against 3.6 expected by chance, and 10 survived Benjamini-Hochberg control (EXP-005). Across the 1,584-rule grid, neither
the Reality Check nor SPA found a rule with positive expected net return once the search was accounted for (EXP-071).

## Going deeper

```
Benjamini-Hochberg: sort p_(1) <= ... <= p_(m); find the largest k with p_(k) <= (k/m) * q; reject hypotheses 1..k
Deflated Sharpe:    SR0 = sqrt(V[SR_trials]) * ( (1-g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)) )     g = Euler-Mascheroni
                    DSR = Phi( (SR - SR0) sqrt(T-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
```
`SR0` is the Sharpe you would expect from the best of N random strategies. If the observed Sharpe is below it, the probability that the strategy has skill is below one half, however good the Sharpe looks in isolation.

## Pitfalls

- Do not forget the trials you did not report.
- Deflation assumes trials are exchangeable; correlated trials need fewer 'effective' trials.
- A multiple-testing correction cannot rescue a biased design.

## Try it

```bash
quant backtest --model momentum --group my_ideas
quant backtest --model donchian --group my_ideas       # the group's trial count rises to 2
```
