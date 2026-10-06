---
title: "Mean reversion and z-scores"
slug: mean-reversion
difficulty: 1
chapter: Ch. 22
prerequisites: [information-coefficient]
stages: [4, 30]
files: [src/signals/mean_reversion.py, src/features/mean_reversion.py, experiments/stage04_mean_reversion.py]
figures: [11, 12]
tests: [tests/test_signals.py]
models: [mean_reversion]
---

# Mean reversion and z-scores

## In one sentence

When a price is far from its recent average, bet that it moves back, with the distance measured in standard deviations.

## The idea

A z-score is (price minus its rolling mean) divided by its rolling standard deviation. A very negative z-score means the asset is unusually cheap relative to its own recent history. The
strategy goes long such assets and short expensive ones. It is the opposite bet from momentum and works over a different horizon, which is why combining them is attractive.

## Why it matters

Mean reversion tends to pay in range-bound markets and lose in trends. Because it trades often, costs eat a large share of any edge; whether the signal survives costs is the real test.

## How this repo uses it

`src/signals/mean_reversion.py` builds the z-score signal at several lookbacks; Stage 4 tests whether the relationship between z-score and subsequent return is negative, as the hypothesis
requires, and Stage 6 charges costs.

## What we found

Mean reversion is only weakly present on this universe at short horizons, and turnover is high; the library version with a 5-day lookback ranks mid-table net of costs in Stage 30.

## Going deeper

```
z_t = (P_t - mean(P_{t-n+1..t})) / std(P_{t-n+1..t})
position_t = - clip(z_t, -c, c)
```
A z-score of -2 with a 20-day window says the price is two standard deviations under its 20-day mean. Because the denominator is the recent standard deviation, a trending market inflates it and the z-score never gets extreme, which is one reason
mean reversion rarely trades the big move.

## Pitfalls

- A price that looks cheap can be cheap for a reason; mean reversion has no brake in a one-way market.
- Short lookbacks pick up bid-ask bounce, which is not tradable.
- Doubling turnover doubles the cost drag.

## Try it

```bash
quant backtest --model mean_reversion --param lookback=5
```
