---
title: "Risk parity and inverse-volatility weighting"
slug: risk-parity
difficulty: 2
chapter: Ch. 19
prerequisites: [volatility-forecasting]
stages: [7]
files: [src/portfolio/risk_parity.py, src/portfolio/inverse_vol.py, experiments/stage07_portfolio.py]
figures: [16, 17]
tests: [tests/test_portfolio.py]
models: []
---

# Risk parity and inverse-volatility weighting

## In one sentence

Risk parity sizes each asset so that it contributes the same amount of risk, rather than the same amount of money.

## The idea

An equal-dollar portfolio of stocks and bonds is dominated by stock risk. Inverse-volatility weighting holds less of what is volatile. Full risk parity goes further and equalises each
asset's contribution to total portfolio risk using the covariance matrix, so correlated assets share a budget.

## Why it matters

It needs no return forecasts, which are the noisiest inputs in finance, and is the default sensible allocation when you do not trust your forecasts. It is the benchmark most adaptive
allocators in this repo are tested against.

## How this repo uses it

`src/portfolio/risk_parity.py` solves for equal risk contributions; Stage 7 compares it with equal weight, inverse volatility, mean-variance and mean-CVaR; figure 17 checks that the
contributions really are equal.

## What we found

Over the 2008-2026 window of the adaptive study, risk parity had about 6.8% volatility against 11% for equal weight and a higher Sharpe (0.77 against 0.65), but a difference of that size cannot be told from chance with samples this short (see the power study).

## Going deeper

```
inverse volatility:  w_i = (1/s_i) / sum_j (1/s_j)
risk contribution:   RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w)
risk parity:         find w >= 0, sum w = 1, with RC_i = RC_j for all i, j
```
Risk parity is solved numerically (Spinu's convex formulation: minimise `0.5 w'Sigma w - (1/N) sum ln w_i`, then rescale). If assets were uncorrelated it reduces to inverse volatility.

## Pitfalls

- Risk parity with leverage is an interest-rate bet in disguise: bonds dominate the low-volatility side.
- Weights are only as good as the covariance estimate.
- Equal risk is not minimum risk and not maximum return.

## Try it

```bash
quant backtest --allocator static --model momentum   # compare with the books in `quant list books`
```
