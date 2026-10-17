---
title: "Mean-variance optimisation and why it needs shrinkage"
slug: mean-variance-and-shrinkage
difficulty: 3
chapter: Ch. 19
prerequisites: [risk-parity, pca-effective-rank]
stages: [7, 8, 21]
files: [src/portfolio/mean_variance.py, src/portfolio/covariance.py, src/portfolio/black_litterman.py, src/portfolio/closed_form.py]
figures: [18]
tests: [tests/test_portfolio.py, tests/test_closed_form.py]
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

`src/portfolio/closed_form.py` adds Markowitz's and Merton's *closed forms* for weights that only have to sum to one. Four numbers `A = 1'S^-1 1`, `B = 1'S^-1 mu`, `C = mu'S^-1 mu` and `D = AC - B^2`
describe the whole frontier, the hyperbola `var(m) = (A m^2 - 2 B m + C) / D`: `frontier_constants`, `global_minimum_variance` (the nose, at return `B/A` and variance `1/A`), `target_return_portfolio`,
`efficient_frontier`, `tangency_portfolio` (the highest Sharpe ratio there is), `capital_market_line` and `utility_portfolio` (a risk-averse investor's holding). The tests check them against the
numerical optimiser with its bounds loosened (they agree), against the frontier's geometry (two-fund separation, the nose) and for the fact that the weights are linear in the expected returns, which
is why noise in the means moves them so much. Long-only and capped problems have no closed form: use `portfolio.mean_variance`.

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
