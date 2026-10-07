---
title: "Bootstrap, block resampling and Monte Carlo simulation"
slug: resampling-and-monte-carlo
difficulty: 2
chapter: Ch. 14
prerequisites: [performance-metrics]
stages: []
files: [src/probability/resampling.py, src/probability/montecarlo.py, src/stats/inference.py]
figures: []
tests: [tests/test_probability.py, tests/test_stats.py]
models: []
---

# Bootstrap, block resampling and Monte Carlo simulation

## In one sentence

When a statistic has no closed-form distribution you simulate one: resample the data you have (the bootstrap), or simulate from a model (Monte Carlo), and use importance sampling when the event you care about is rare.

## The idea

The i.i.d. bootstrap draws observations with replacement. Returns are not independent (volatility clusters, strategies are autocorrelated), so `src.probability.resampling` offers the dependent-data versions: the **moving-block** and **circular-block** bootstraps draw blocks of consecutive observations, the **stationary bootstrap** draws blocks with geometric lengths so the resampled series is itself stationary, and `optimal_block_length` implements the Politis-White rule (with the Patton-Politis-White correction) so the block length is chosen from the data. The **wild bootstrap** multiplies regression residuals by random signs to preserve heteroskedasticity, and **subsampling** (`subsample_statistics`, `subsample_ci`) gives intervals under weaker conditions. `bootstrap_statistic` evaluates any function of a data array on resamples; `src.stats.inference.bootstrap_ci` builds percentile, studentised and BCa intervals on top.

For models, `src.probability.montecarlo` simulates geometric Brownian motion (with antithetic variates), correlated GBM, Ornstein-Uhlenbeck, Merton jump diffusions and GARCH paths, with `mc_estimate` returning a value, its standard error and the effective sample size. **Importance sampling** draws from a proposal `q` and weights by `p/q`; `portfolio_tail_probability_is` tilts the mean of a Gaussian portfolio toward the loss region and `cross_entropy_is` adapts the proposal automatically.

## Why it matters

A Sharpe ratio estimated from five years has a standard error near 0.45 and the i.i.d. formula understates it for autocorrelated strategies. A 1-in-1,000 loss needs a million plain draws to see a thousand hits, but only a few thousand under a good importance-sampling proposal.

## How this repo uses it

The validation stage, the drawdown and ruin tools, the Bayesian tools and the research-operations cross-validation all draw their random indices from this module, and every function takes a seed or a `numpy` generator, so every result is reproducible.

## What we found

The tests check that the resampling indices are valid and reproducible, that the optimal block length grows with the autocorrelation, that the block-bootstrap standard error of a mean matches theory, and that importance sampling reproduces the known tail probability of a Gaussian portfolio with far fewer draws than plain Monte Carlo. The bootstrap interval covers the true mean at about its nominal rate in simulation.

## Pitfalls

- A block that is too short keeps the dependence from being captured; too long leaves few distinct blocks.
- The bootstrap fails for extremes (the maximum, quantiles in the far tail) and for heavy tails with infinite variance; use EVT or subsampling.
- Importance-sampling weights can have infinite variance if the proposal has thinner tails than the target; check the effective sample size.
- Resampling returns from one history only reshuffles what happened; it cannot create a crash that is not in the sample.

## Try it

```python
import numpy as np
from src.probability.resampling import bootstrap_statistic, optimal_block_length

rng = np.random.default_rng(0)
x = np.zeros(2000)
for t in range(1, 2000):
    x[t] = 0.3 * x[t - 1] + rng.standard_normal()
print(round(optimal_block_length(x), 1))
means = bootstrap_statistic(x, np.mean, n_boot=500, seed=1)
print(round(float(means.std()), 3))
```
