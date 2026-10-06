---
title: "Momentum and trend following"
slug: momentum
difficulty: 1
chapter: Ch. 22
prerequisites: [information-coefficient]
stages: [3, 30]
files: [src/signals/momentum.py, src/strategies/time_series.py, src/strategies/cross_sectional.py, experiments/stage03_momentum.py]
figures: [8, 9, 10]
tests: [tests/test_signals.py, tests/test_strategies.py]
models: [momentum, xs_momentum, dual_momentum, donchian, ma_crossover, trend_atr, relative_strength, vol_breakout]
---

# Momentum and trend following

## In one sentence

Assets that have gone up recently tend to keep going up for a while, and the strategies in this family differ only in how they measure 'recently' and 'up'.

## The idea

Time-series momentum asks of each asset on its own: is its past return positive? Cross-sectional momentum ranks assets against each other: which are the strongest? Variants add a skipped
month (to avoid short-term reversal), moving-average crossovers, channel breakouts (Donchian), volatility filters (ATR) and an absolute-versus-relative test (dual momentum). All of them
are a transformation of past prices into a score, then into a position.

## Why it matters

Momentum is among the most documented return patterns across asset classes, and also among the most crowded. It tends to do well in sustained trends and badly in sharp reversals, so its
correlation with other strategies and its drawdowns matter as much as its average return.

## How this repo uses it

The Generation 1 momentum signal lives in `src/signals/momentum.py`; the library variants live in `src/strategies/`. They are measured with the information coefficient (Stage 5), run in
the cost-aware engine (Stage 6), and re-examined as a family in the Stage 30 search, where the number of variants tried is accounted for.

## What we found

On this 15-ETF universe the single 126-day volatility-scaled momentum signal had no reliable cross-sectional information coefficient (EXP-005). Among the library variants, dual momentum
and moving-average crossovers were the best net performers, but the multiple-testing study in Stage 30 treats them as the winners of a search, not as discoveries.

## Going deeper

```
time-series momentum:  score_i,t = P_i,t-skip / P_i,t-lookback - 1      long if score > 0
cross-sectional:       rank_i,t = rank of score_i,t among assets         long top k, short bottom k
dual momentum:         hold the top-k by score, but only those with score > 0, else cash
Donchian:              long when P_t > max(P_{t-55..t-1}); exit when P_t < min(P_{t-20..t-1})
```
Skipping the most recent month (`skip`) removes the short-term reversal that otherwise contaminates the signal. Scores are usually divided by volatility so a volatile asset does not dominate.

## Pitfalls

- Momentum crashes: months after a sharp bear market ends are its worst.
- With few assets, cross-sectional momentum is a very noisy ranking.
- The more lookbacks you try, the better the best one looks by luck alone; see multiple testing.

## Try it

```bash
quant list models                                  # the momentum family and its relatives
quant backtest --model dual_momentum --tearsheet
quant backtest --model dual_momentum --param lookback=126 --param skip=5
```
