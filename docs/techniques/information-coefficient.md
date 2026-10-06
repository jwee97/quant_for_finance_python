---
title: "The information coefficient and the fundamental law"
slug: information-coefficient
difficulty: 2
chapter: Ch. 20
prerequisites: [data-integrity]
stages: [5]
files: [experiments/stage05_expected_returns.py, src/signals/transform.py, src/validation/permutation.py]
figures: [9, 10, 13]
tests: [tests/test_signals.py, tests/test_forecast_tests.py]
models: []
---

# The information coefficient and the fundamental law

## In one sentence

The information coefficient is the correlation between a signal today and the return that follows, and it is the first, cheapest honest test of whether a signal knows anything.

## The idea

Each day, rank the assets by the signal and by the return that follows; the IC is the rank correlation. Averaged over time it measures skill per bet. The fundamental law of active
management says information ratio is roughly IC times the square root of breadth (the number of independent bets per year), which explains why a tiny IC can still be valuable at high breadth and
why overlapping forecasts do not count as independent bets.

## Why it matters

It separates "the signal predicts" from "the strategy made money". A strategy can make money from a bull market with no predictive signal at all. The IC also decays with horizon, which
tells you how fast to trade.

## How this repo uses it

Stage 5 reports mean IC, its t-statistic corrected for overlapping observations (the naive t-statistic overstates evidence by the square root of the overlap), the decay with horizon, and
the implied information ratio. `src/validation/permutation.py` supplies the circular-shift test used wherever a placebo is needed.

## What we found

Overlap correction removes most of the apparent significance of the Generation 1 signals; family-level false discovery control (Benjamini-Hochberg) is applied across the many lookbacks.

## Going deeper

```
IC_t = rank_corr( signal_i,t , return_i,t+h )   across assets i
mean IC, IC_IR = mean(IC) / std(IC)
fundamental law:  IR ~ IC * sqrt(breadth)
overlap-adjusted t:  t_naive / sqrt(h)    (daily observations of h-day returns overlap by h-1 days)
```
An IC of 0.03 looks tiny, but 15 independent bets a month is 180 a year: IR ~ 0.03 * sqrt(180) ~ 0.4. Overlapping observations are not extra bets, which is why the naive t-statistic overstates the evidence.

## Pitfalls

- Overlapping returns make adjacent observations nearly identical; naive standard errors are far too small.
- IC is blind to magnitude: a signal can rank well and still lose money on the rare big move.
- Testing many signals needs a correction.

## Try it

```bash
python -m experiments.stage05_expected_returns
quant explain information coefficient
```
