---
title: "Dynamic covariance: DCC-GARCH and orthogonal GARCH"
slug: dynamic-covariance
difficulty: 3
chapter: Ch. 20.2
prerequisites: [volatility-forecasting]
stages: [17]
files: [src/portfolio/dynamic_covariance.py, experiments/stage17_dynamic_covariance.py, src/framework/allocators_portfolio.py]
figures: [34, 35]
tests: [tests/test_dynamic_covariance.py, tests/test_portfolio_allocators.py]
models: []
---

# Dynamic covariance: DCC-GARCH and orthogonal GARCH

## In one sentence

Let both volatilities and correlations change through time, with correlations that tend to rise in a crisis.

## The idea

DCC-GARCH fits a GARCH to each asset, standardises returns, and then models the correlation matrix of the standardised residuals as a slowly moving average that reacts to shocks. Orthogonal GARCH
applies GARCH to a few principal components and rebuilds the covariance from them. Both are judged by an out-of-sample loss on the covariance forecast and by the book they produce.

## Why it matters

A rolling covariance is slow to see a crisis. A good dynamic covariance matters most exactly when it is hardest to get right.

## How this repo uses it

`src/portfolio/dynamic_covariance.py` has both models; Stage 17 compares them with sample, EWMA and shrinkage estimators, pooled and inside the GFC, COVID and 2022 windows.

The `dynamic_cov` allocator makes the forecast usable: at each month-end it solves the constrained minimum-variance problem on the month-ahead DCC-GARCH, O-GARCH or (control) rolling-shrinkage covariance. Use it as `allocation: {allocator: dynamic_cov, params: {model: dcc}}` in a specification.

## What we found

DCC had the lowest average loss but the advantage came from a few stress months and was not significant after correction (EXP-036); a minimum-variance book from DCC forecasts realised 3.9% lower
volatility than the shrinkage book (p = 0.008), which was retained (EXP-038). Orthogonal GARCH was worse (EXP-039, EXP-041).

## Going deeper

```
DCC:  r_t = H_t^(1/2) e_t, H_t = D_t R_t D_t,  D_t = diag of univariate GARCH volatilities
      Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1};   R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2
O-GARCH: fit GARCH to the k leading principal components f_t;  Sigma_t = V diag(s2_f,t) V' + D_resid
```
`a + b` near 1 means correlation shocks decay very slowly; the stage reports it as 0.97 to 0.995.

## Pitfalls

- DCC correlation persistence near 1 means forecasts revert very slowly to the long-run level.
- Crisis windows have few independent observations.
- Lower volatility is not higher Sharpe.

## Try it

```bash
python -m experiments.stage17_dynamic_covariance
```
