---
title: "The strategy library as a search, not a menu"
slug: strategy-library-search
difficulty: 3
chapter: Ch. 22
prerequisites: [multiple-testing, momentum]
stages: [30]
files: [src/strategies/__init__.py, src/strategies/_common.py, experiments/stage30_library.py, config/strategy_library.yaml]
figures: [59, 60, 61]
tests: [tests/test_strategies.py]
models: []
---

# The strategy library as a search, not a menu

## In one sentence

Writing 31 strategies and reporting the best is a search, and the evidence for the best must be judged against the whole search.

## The idea

The library has 31 ETF models and 3 crypto models, each registered as a forecast model. Stage 30 backtests 42 specifications (variants included), reports performance net of costs, and then tests the family: the Reality Check and SPA
ask whether any beats cash or passive equal weight after accounting for the number tried; correlations and an effective count of independent ideas say how much the family really diversifies.

## Why it matters

This is how research actually fails: not one bad backtest but a quiet search. A platform that makes adding strategies easy has to make counting them automatic.

## How this repo uses it

Each model implements `score`, which a shared stack turns into a calibrated forecast and a position. The causality test runs on every model. The experiment manager counts distinct specifications in a group and feeds the deflated Sharpe ratio.

## What we found

At least one specification beat cash after the search (EXP-075, retained), none beat passive equal weight (EXP-076), and the confidence-weighted combination did not beat equal weighting or the Generation 1 momentum specification (EXP-077).

## Going deeper

```
common window starts when the slowest specification has a forecast (first forecast date, not first non-zero position)
Reality Check / SPA on d_k = r_k - r_cash  (and r_k - r_equal_weight):  max over the 42 specifications
effective number of independent ideas = exp(entropy of the eigenvalue spectrum of the correlation of strategy returns)
```
The common window matters: comparing a strategy that starts in 2008 with one that starts in 2011 compares different markets.

## Pitfalls

- A common window must start when the slowest strategy is live, or the comparison is not like for like.
- Sharpe differences between strategies are small relative to their uncertainty.
- Most of the library's apparent diversity comes from a few independent ideas.

## Try it

```bash
quant list models
quant dashboard      # the 'Strategy library' tab
```
