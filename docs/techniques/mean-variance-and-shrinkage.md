---
title: "Mean-variance optimisation and why it needs shrinkage"
slug: mean-variance-and-shrinkage
difficulty: 3
chapter: Ch. 19
prerequisites: [risk-parity, pca-effective-rank]
stages: [7, 8, 21]
files: [src/portfolio/mean_variance.py, src/portfolio/covariance.py, src/portfolio/black_litterman.py]
figures: [18]
tests: [tests/test_portfolio.py]
models: []
---

# Mean-variance optimisation and why it needs shrinkage

## In one sentence

Mean-variance optimisation picks the portfolio with the best expected return per unit of risk, and amplifies every error in its inputs unless the inputs are shrunk.

## The idea

Given expected returns and a covariance matrix the optimiser finds weights that maximise return for a risk budget. Estimation error in the means and in the small eigenvalues of the
covariance matrix is exploited: the optimiser puts large weights where the data looks best by chance. Shrinkage (Ledoit-Wolf) pulls the covariance toward a structured target;
constraints and Black-Litterman views regularise the means.

## Why it matters

It is the textbook answer and the textbook failure. Understanding why it fails, and what the fixes buy, is the core of practical portfolio construction.

## How this repo uses it

`src/portfolio/mean_variance.py` and `black_litterman.py` implement the optimisers with the constraints in `src/portfolio/constraints.py`; Stage 8 evaluates covariance estimators out of
sample, and Stage 21 adds parameter uncertainty (Bayesian portfolio construction).

## What we found

Plug-in mean-variance weights are unstable (an average pairwise weight change of about 0.87 between rebalances) and concentrated (effective number of holdings about 4.5). Bayes-Stein shrinkage of the means did not improve that (0.88), while averaging over the posterior predictive did (0.57, effective holdings about 8.8), as Stage 21 reports.

## Going deeper

```
maximise  w'mu - (gamma/2) w'Sigma w      subject to  sum w = 1, w >= 0, w <= w_max
w* = (1/gamma) Sigma^-1 mu                (unconstrained)
Ledoit-Wolf:  Sigma_shrunk = delta F + (1 - delta) S      F = structured target, S = sample covariance
```
`Sigma^-1` multiplies noise in the smallest eigenvalues by `1/l_min`, which is why a condition number of 35,609 (the raw sample covariance here) becomes 9,764 after shrinkage.

## Pitfalls

- Optimised weights are often extreme; always look at them.
- Maximising in-sample Sharpe is overfitting by construction.
- Constraints are not a nuisance, they are the regulariser.

## Try it

```bash
python -m experiments.stage07_portfolio
```
