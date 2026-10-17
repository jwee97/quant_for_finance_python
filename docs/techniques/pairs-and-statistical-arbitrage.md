---
title: "Pairs, cointegration and statistical arbitrage"
slug: pairs-and-statistical-arbitrage
difficulty: 3
chapter: Ch. 22
prerequisites: [mean-reversion, pca-effective-rank]
stages: [14, 30]
files: [src/signals/pairs.py, src/signals/pca_strategy.py, src/strategies/statarb.py, src/probability/ou.py, experiments/stage14_extensions.py]
figures: [26, 27]
tests: [tests/test_extensions.py, tests/test_strategies.py, tests/test_ou.py, tests/test_ou_pairs.py]
models: [cointegration_pairs, kalman_pairs, ou_threshold_pairs, pca_residual, sparse_basket]
---

# Pairs, cointegration and statistical arbitrage

## In one sentence

Trade the gap between related assets, betting it closes, with a hedge that removes the shared market move.

## The idea

Two prices are cointegrated if a combination of them is stationary, so the spread reverts. The hedge ratio can be fixed (Engle-Granger), updated by a Kalman filter, or taken from a PCA decomposition: trade the
residual after removing the first few principal components. A sparse variant regresses each asset on a LASSO-selected basket. ETF NAV arbitrage is the same idea against a fund's net asset value and is only simulated here.

## Why it matters

Market-neutral strategies promise returns uncorrelated with the market, but they are exposed to the relationship breaking, and they trade often.

## How this repo uses it

Stage 14 selects pairs on a development sample only and trades them with that hedge ratio; the library adds Kalman, rolling-cointegration, PCA residual and sparse-basket models as plugins. The
`ou_threshold_pairs` model treats the spread as an Ornstein-Uhlenbeck process, fits it by maximum likelihood on a rolling window and trades the levels that earn the most per day after costs (`probability.ou`).

## What we found

The PCA residual book was factor neutral by construction (residual exposures at most 0.035) but did not survive costs (EXP-023). Cointegrated pairs within this small universe were left as `investigate` (EXP-022). The Kalman pairs
model turned over about 24 times a year in the library.

## Going deeper

```
Engle-Granger: regress A on B -> beta;  spread_t = A_t - beta B_t;  ADF test on spread (reject unit root => cointegrated)
Kalman hedge ratio: state beta_t = beta_{t-1} + w_t;  observation A_t = beta_t B_t + v_t
trade:  z = (spread - mean)/std;  short spread if z > 2, long if z < -2, exit near 0
PCA residual: e_i,t = r_i,t - sum_k loading_ik f_k,t (k = 3);  trade the cumulative residual back toward zero
```
Always trade dollar-neutral (long one leg, short `beta` of the other) so the market move cancels.

If the spread is an Ornstein-Uhlenbeck process, `dX = -theta (X - mu) dt + sigma dW`, there is a best pair of levels. Buy at `mu - d` and sell at `mu + d`: a trade earns `2d - c` (`c` the round-trip cost)
and a full cycle takes `(2 pi / theta) erfi(d sqrt(theta) / sigma)` on average, so the profit per unit of time is `(2d - c) theta / (2 pi erfi(d sqrt(theta) / sigma))`, which is largest at one width `d` that grows
with the cost (Bertram 2010). `ou_threshold_pairs` estimates `theta, mu, sigma` from the exact transition density, requires a Dickey-Fuller t statistic below the Engle-Granger value of -3.4 (a regression of one
random walk on another leaves a residual that looks like reversion; without the test the rule trades it), and closes a pair whose spread runs past `stop_sd` standard deviations.

## Pitfalls

- Cointegration estimated in sample often fails out of sample; always re-test.
- The best levels assume the spread is watched at every instant. On daily closes crossings are missed and the rule trades only 60 to 80% as often as the continuous theory says (`tests/test_ou_pairs.py`).
- Costs on both legs.
- A tiny universe offers few genuine pairs.

## Try it

```bash
quant backtest --model kalman_pairs --tearsheet
```
