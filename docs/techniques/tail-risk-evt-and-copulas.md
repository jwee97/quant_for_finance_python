---
title: "Tail risk: extreme value theory and copulas"
slug: tail-risk-evt-and-copulas
difficulty: 3
chapter: Ch. 15
prerequisites: [resampling-and-monte-carlo, risk-managed-exposure]
stages: []
files: [src/probability/evt.py, src/probability/copulas.py]
figures: []
tests: [tests/test_probability.py]
models: []
---

# Tail risk: extreme value theory and copulas

## In one sentence

Value-at-risk at 99% depends on the far tail, where a normal distribution is wrong and an empirical quantile has few points; extreme value theory fits the tail directly and copulas describe how tails move together.

## The idea

By the **Pickands-Balkema-de Haan theorem**, losses above a high threshold `u` follow a **generalised Pareto distribution** (GPD) with shape `xi` and scale `beta`. `fit_gpd` estimates it by maximum likelihood (the threshold is a quantile or a value), `gpd_var_es` converts it into value-at-risk and expected shortfall,

```
VaR_p = u + (beta / xi) * (((1 - p) / (N_u / N)) ** -xi - 1)
ES_p  = (VaR_p + beta - xi u) / (1 - xi)
```

and `profile_var_ci` gives a profile-likelihood confidence interval. `hill_estimator` and `hill_plot` estimate the tail index of a heavy-tailed series, `threshold_diagnostics` and `mean_excess` help choose `u`, `fit_gev` and `block_maxima` fit the block-maxima version, `extremal_index` measures the clustering of extremes and `dynamic_evt_var` does the walk-forward conditional version (a volatility filter, then EVT on the standardised residuals).

A **copula** separates the marginals from the dependence: `F(x1, ..., xd) = C(F1(x1), ..., Fd(xd))`. `fit_copula` supports the Gaussian, Student-t, Clayton, Gumbel and Frank families, `compare_copulas` ranks them by likelihood and AIC, `empirical_tail_dependence` measures the joint tail directly, and `simulate_joint` fits an EVT or empirical marginal per asset and a copula over them to produce scenarios for `portfolio_var_es`.

## Why it matters

Assets that look weakly correlated in ordinary times fall together in a crisis. A Gaussian copula has no tail dependence however high the correlation, so a portfolio VaR from a Gaussian model is optimistic exactly when it matters; the Student-t and Clayton copulas allow joint crashes.

## How this repo uses it

`evt_risk_managed` sizes a long-only book by a conditional-EVT expected-shortfall forecast; the findings script compares Gaussian, historical and EVT VaR on the ETFs and ranks copulas for equity-bond pairs. The scenario generators are available to the risk stage for stress tests.

## What we found

The tests recover the tail index of simulated Student-t data and extrapolate beyond the sample, check that the GPD maximum likelihood matches `scipy` on exact GPD data, check that the dynamic EVT value-at-risk is causal and calibrated, show that copula selection picks the true family, and verify that the elliptical copulas recover correlation and tail dependence (the Gaussian copula has none). Real-data results are in [the findings](../institutional_findings.md); treat them as exploratory since a 20-year sample holds a handful of independent tail events.

## Pitfalls

- The threshold is a bias-variance choice: too low violates the asymptotics, too high leaves too few exceedances. Look at the stability of `xi` across thresholds.
- ES needs `xi < 1`; the estimates are very noisy for `xi` near one.
- Copula fits depend on the marginals; always use pseudo-observations or a good marginal fit.
- Volatility clusters, so unconditional EVT on raw returns overstates risk in calm periods and understates it after shocks.

## Try it

```python
import numpy as np
from src.probability.evt import fit_gpd, gpd_var_es

rng = np.random.default_rng(0)
losses = rng.standard_t(4, 5000) * 0.01
fit = fit_gpd(losses, quantile=0.95)
print(round(fit.xi, 2), {k: round(v, 4) for k, v in gpd_var_es(fit, 0.99).items()})
```
