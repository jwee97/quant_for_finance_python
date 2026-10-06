---
title: "Reality Check, SPA and probability of backtest overfitting"
slug: reality-check-spa-pbo
difficulty: 3
chapter: Ch. 22
prerequisites: [multiple-testing]
stages: [26, 30]
files: [src/validation/multiple_testing.py, experiments/stage26_distributed.py, experiments/stage30_library.py]
figures: [52, 53, 59]
tests: [tests/test_multiple_testing.py, tests/test_stage26_grid.py]
models: []
---

# Reality Check, SPA and probability of backtest overfitting

## In one sentence

These three tools ask whether the best strategy found by a search is better than luck would produce from that same search.

## The idea

White's Reality Check bootstraps the maximum performance across all rules under the null that none beats the benchmark. Hansen's SPA improves power by discarding clearly poor rules.
Probability of backtest overfitting (CSCV) splits time into blocks, picks the best rule in half of them and records how often it ranks below the median in the other half.

## Why it matters

They test the search, not the winner. A Sharpe of 0.9 is impressive for one pre-specified rule and unremarkable as the best of 1,584.

## How this repo uses it

The tools are in `src/validation/multiple_testing.py`. Stage 26 applies them to a pre-declared grid in the distributed executor (serial, joblib and dask backends give identical results);
Stage 30 applies them to the 42 library specifications.

## What we found

Across the grid, no rule survived (EXP-071). In the library search, the best strategies beat cash after the search but not passive equal weight.

## Going deeper

```
Reality Check:  d_k,t = performance of rule k minus benchmark at time t;  V = max_k sqrt(T) mean(d_k)
                bootstrap d* (stationary blocks), centre at zero, V* = max_k sqrt(T) mean(d*_k - mean d_k);  p = P(V* >= V)
PBO (CSCV):     split time into S blocks, choose S/2 for 'in sample', pick the best rule, record its out-of-sample rank;
                PBO = share of splits where it ranks below the median
```
SPA studentises each rule's statistic and discards rules that are clearly bad, so one hopeless rule cannot mask a good one.

## Pitfalls

- The null is "no better than the benchmark"; a rule can lose to the benchmark and still be worth owning as a diversifier.
- PBO depends on how many blocks you cut; report the setting.
- The grid itself is a choice.

## Try it

```bash
python -m experiments.stage26_distributed --backend joblib
```
