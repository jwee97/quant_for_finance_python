---
title: "Combining forecasts: stacks, trust rules and mixtures"
slug: forecast-combination
difficulty: 3
chapter: Ch. 22.5
prerequisites: [probabilistic-forecasting]
stages: [10, 24, 30, 31]
files: [src/framework/forecasting.py, src/signals/alpha_engine.py, src/signals/combine.py, experiments/stage24_combination.py, src/framework/adaptive.py]
figures: [48, 49, 61]
tests: [tests/test_alpha_engine.py, tests/test_framework.py, tests/test_adaptive_integration.py]
models: []
---

# Combining forecasts: stacks, trust rules and mixtures

## In one sentence

Several weak forecasts can be better than one, if the way they are combined does not itself overfit.

## The idea

Each model produces a forecast with a mean and spread. Combination rules decide the weight on each: equal, by confidence, by precision (inverse variance), by recent information coefficient, cost
aware (shrunk by turnover) or conditional on the regime. A Gaussian mixture keeps the disagreement between models as extra spread instead of averaging it away.

## Why it matters

This is how systematic firms actually run: many small alphas, one portfolio. The weakness is that a clever weighting scheme is another set of parameters to overfit.

## How this repo uses it

`src/framework/forecasting.py` has `combine_forecasts` with every rule; the Generation 3 alpha engine in `src/signals/alpha_engine.py` is the cost-aware one; Stage 31 adds regime-conditional
trust. Adding a strategy means registering a forecast model; it joins the combination automatically.

The `decay_weighted` combination rule weights each model by its information coefficient at the holding period implied by an exponential decay fitted to its incremental IC at lags 1 to 20 (matured labels only); the fitted half-lives appear in the tear sheet as "alpha decay". It sits beside `equal`, `confidence`, `precision`, `ic_weighted`, `regime_conditional` and `cost_aware`.

Two rules weigh the models by the covariance of their information coefficients rather than by each one's own: `optimal_ic` uses the weights `Sigma^-1 mu` of the models' matured ICs over a trailing window (non-negative, monthly, shrunk toward the diagonal), which maximise the information ratio of the combination and give a model less when it repeats the others; `orthogonal_ic` makes the models' forecasts orthogonal across stocks first (Gram-Schmidt in the order the models are listed, or symmetric) and returns one combined forecast. See [building an alpha model](alpha-model-construction.md).

## What we found

The cost-aware trust rule did not beat equal weighting or the Generation 1 momentum book (EXP-066), and in the library combination study confidence weighting was worse than equal weighting.

## Going deeper

```
equal:       w_k = 1/K
precision:   w_k proportional to 1 / s_k^2
confidence:  w_k proportional to c_k
IC-weighted: w_k proportional to max(IC_k, 0)  (trailing)
cost-aware:  w_k proportional to (IC_k - lambda * turnover_k)+
mixture:     p(y) = sum_k w_k N(mu_k, s_k^2);   mean = sum w_k mu_k;   var = sum w_k (s_k^2 + mu_k^2) - mean^2
```
The mixture variance has a term for disagreement between models: when two models give opposite means the combined spread widens instead of cancelling.

## Pitfalls

- Models with correlated errors add less than their count suggests.
- Weights estimated on a few years of data are noisy; shrink them toward equal.
- Combination before costs can hide that each alpha turns over fast.

## Try it

```bash
quant backtest --model momentum --combine equal
```
