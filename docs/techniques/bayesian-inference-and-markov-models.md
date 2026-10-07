---
title: "Bayesian inference and Markov models"
slug: bayesian-inference-and-markov-models
difficulty: 3
chapter: Ch. 17
prerequisites: [bayesian-portfolio-construction, regime-detection]
stages: []
files: [src/probability/bayes.py, src/probability/markov.py]
figures: []
tests: [tests/test_probability.py]
models: []
---

# Bayesian inference and Markov models

## In one sentence

Bayesian methods give a distribution for what you do not know, which is how much of a Sharpe ratio is luck; Markov models describe how states persist, which is how much regimes matter.

## The idea

**Bayes' rule** updates a prior with data: `p(theta | y) is proportional to p(y | theta) p(theta)`. `bayes_linear_regression` has a conjugate normal prior and a closed-form posterior (it equals OLS under a flat prior), and `bayes_factor` compares two models by marginal likelihood. For models without a closed form, `metropolis_hastings` is a random-walk sampler with adaptive scaling and several chains, and `rhat` and `ess` (the Gelman-Rubin statistic and the effective sample size) say whether the chains have converged; `hpd_interval` gives the highest-posterior-density interval. `bayesian_sharpe` puts a Student-t likelihood on returns and samples `(mu, sigma, nu)`, which yields a posterior for the Sharpe ratio that accounts for fat tails; `compare_sharpe` returns the posterior probability that one strategy's Sharpe exceeds another's. `beta_binomial` is the conjugate case for hit rates.

A **Markov chain** has the property that tomorrow's state depends only on today's. `estimate_transition_matrix` counts transitions (with optional smoothing), `MarkovChain` provides the stationary distribution, expected holding times and multi-step probabilities, `transition_posterior` gives a Dirichlet posterior over the rows, `test_markov_order` tests whether a memory of more than one step is needed, `runs_test` tests randomness of a sign sequence, and `discretize` turns a series into states. `fit_markov_switching` fits a regime-switching model (switching mean and variance) and returns filtered, causal probabilities as well as smoothed ones.

## Why it matters

A point estimate of a Sharpe ratio hides its uncertainty; the posterior shows how wide it is and how much the prior or the tails matter. Regime models drive many allocation decisions, and whether they use the filtered or the smoothed probability is the difference between a tradable signal and a look-ahead bug.

## How this repo uses it

The regime stage and `bayes` portfolio construction build on the same ideas; this module adds the general-purpose tools (a sampler with diagnostics, Bayesian regression, Sharpe inference) and the Markov chain analytics. The institutional findings report the Bayesian Sharpe of SPY with its convergence diagnostics.

## What we found

The tests show Bayesian regression equals OLS under a flat prior, the sampler recovers a known Gaussian and its diagnostics flag a deliberately bad chain, the Sharpe posterior behaves as it should (it narrows with more data and widens with heavier tails), the two-state chain analytics match closed forms, and the filtered probabilities of the switching model are causal. Real-data values are in [the findings](../institutional_findings.md).

## Pitfalls

- An MCMC result without R-hat and effective-sample-size checks is not a result.
- A prior is an assumption; check how much the answer moves when it changes.
- Smoothed regime probabilities use future data; only filtered ones are tradable.
- Regime labels from a fitted model are not guaranteed to correspond to the economic regimes you have in mind.

## Try it

```python
import numpy as np
from src.probability.bayes import bayesian_sharpe

rng = np.random.default_rng(0)
r = rng.normal(0.0004, 0.01, 1000)
post = bayesian_sharpe(r, n_samples=1500, burn=1000)
print(round(post["median"], 2), [round(v, 2) for v in post["hpd95"]], round(float(post["rhat"].max()), 2))
```
