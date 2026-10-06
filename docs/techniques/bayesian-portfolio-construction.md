---
title: "Bayesian portfolio construction"
slug: bayesian-portfolio-construction
difficulty: 3
chapter: Ch. 19
prerequisites: [mean-variance-and-shrinkage]
stages: [21]
files: [src/portfolio/bayesian.py, experiments/stage21_bayesian.py, src/framework/allocators_portfolio.py]
figures: [42, 43]
tests: [tests/test_bayesian.py, tests/test_portfolio_allocators.py]
models: []
---

# Bayesian portfolio construction

## In one sentence

Treat expected returns and covariances as uncertain, and optimise against the whole posterior instead of a single estimate.

## The idea

Rather than plugging in a sample mean, draw mean and covariance from a posterior that reflects the sample length, and average the optimal weights or the utility over the draws (the posterior
predictive). This naturally shrinks extreme weights and spreads capital when the data cannot separate assets.

## Why it matters

It addresses the root cause of optimiser failure, estimation error, without ad hoc constraints.

## How this repo uses it

`src/portfolio/bayesian.py` implements plug-in, Bayes-Stein and posterior-predictive books; Stage 21 compares weight stability and net Sharpe.

The `bayesian` allocator exposes the three Stage 21 books (`kind: mvo_sample | bayes_stein | bayes_predictive`) to any specification, so they can be compared with the same costs, benchmarks and tear sheet as every other allocator.

## What we found

Posterior-averaged weights were more stable than plug-in mean-variance weights (EXP-060, retained), but the posterior-predictive book did not beat both the plug-in inputs and the risk-parity book after costs (EXP-059, rejected). Bayes-Stein shrinkage of the means alone did not improve stability.

## Going deeper

```
posterior:  mu | data ~ N( mu_post, Sigma/T_eff );  Sigma | data ~ inverse-Wishart
Bayes-Stein: mu_post = (1 - phi) mu_hat + phi mu_0 1,   phi = (N + 2) / ((N + 2) + T (mu_hat - mu_0 1)' Sigma^-1 (mu_hat - mu_0 1))
posterior predictive weights: w = average over draws d of argmax w'mu_d - (gamma/2) w'Sigma_d w
```
Averaging optimal weights over parameter draws is what smooths them: no single noisy estimate decides the allocation.

## Pitfalls

- The prior is a choice, and with a weak prior the posterior is just the data.
- Beating plug-in mean-variance is a low bar.
- Computation grows with the number of draws.

## Try it

```bash
python -m experiments.stage21_bayesian
```
