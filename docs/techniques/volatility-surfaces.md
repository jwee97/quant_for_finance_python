---
title: "Volatility surfaces: SVI, SSVI, local volatility and the VIX"
slug: volatility-surfaces
difficulty: 3
chapter: Ch. 18
prerequisites: [option-pricing-and-greeks]
stages: []
files: [src/derivatives/surface.py, src/derivatives/synthetic.py, src/derivatives/volatility.py]
figures: []
tests: [tests/test_derivatives.py]
models: []
---

# Volatility surfaces: SVI, SSVI, local volatility and the VIX

## In one sentence

The implied volatilities of all strikes and expiries form a surface that must obey no-arbitrage rules; `src.derivatives.surface` fits arbitrage-free parametric surfaces and derives from them the risk-neutral density, local volatility, variance-swap strikes and a VIX-style index.

## The idea

Work in **total variance** `w(k, T) = sigma_imp^2 T` against log-moneyness `k = ln(K / F)`. The **SVI** slice of Gatheral is `w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))`; `fit_svi_slice` fits it by least squares with several starts and checks the butterfly condition (`svi_butterfly_density` must be non-negative). **SSVI** parametrises the whole surface from the at-the-money variance `theta_T` and three global parameters through `w = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))`, with a power-law `phi`; `fit_ssvi` fits it and `ssvi_arbitrage_free` checks the Gatheral-Jacquier sufficient conditions for no butterfly and no calendar arbitrage. From any surface function `w_fun`:

- `risk_neutral_density` is the second derivative of the call price in strike (Breeden-Litzenberger), and is non-negative exactly when there is no butterfly arbitrage;
- `local_volatility` is the Dupire volatility `sigma_loc^2 = (dw/dT) / (1 - k w'/w + (w'^2/4)(-1/4 - 1/w + k^2/w^2) + w''/2)`;
- `variance_swap_strike` and `vix_style_index` replicate the model-free variance `2/T * sum (dK / K^2) e^{rT} Q(K)` from out-of-the-money options, interpolated to a constant 30-day maturity as the VIX is;
- `smile_metrics` reports at-the-money level, skew and curvature at fixed deltas.

## Why it matters

An unconstrained interpolation of implied volatilities can imply negative probabilities, and a strategy that trades the "mispricing" is trading the interpolation error. Arbitrage-free parametric forms also summarise a chain in a handful of numbers that can be stored, compared across days and used as features.

## How this repo uses it

`src.derivatives.synthetic` builds the simulation: a stochastic-volatility-with-jumps underlying, quotes from an SSVI surface with bid/ask spreads, listed strikes and expiries, and a **variance premium** (implied variance exceeds the expected realised variance by a stated 25%) so that volatility strategies have something to harvest by construction. `src.derivatives.volatility` computes VIX-style series, term structure, implied correlation and the variance risk premium.

## What we found

The tests check that the SVI fit recovers the parameters of a generated slice and flags a slice with butterfly arbitrage, that SSVI parameters are recovered and its arbitrage conditions hold, that the density, Dupire volatility, variance-swap strike and VIX-style index are right on surfaces with known answers (including the VIX-style index on a flat Black-Scholes chain, which must equal the volatility), and that a surface fitted to the synthetic quotes recovers the skew that generated them. In the synthetic market the measured implied variance exceeds realised variance, as designed; this validates the machinery, not the existence of the premium in real markets.

## Pitfalls

- Fitting wings with few, wide quotes extrapolates SVI badly; weight by vega or spread.
- Calendar arbitrage appears when slices are fitted independently; SSVI avoids it by construction.
- Dupire's formula divides by a quantity that can reach zero, so local volatility is noisy where the surface is noisy.
- The VIX formula assumes a continuum of strikes; truncating the wings biases it down.

## Try it

```python
import numpy as np
from src.derivatives.surface import fit_svi_slice

k = np.linspace(-0.3, 0.3, 21)
T = 0.25
true_w = 0.01 + 0.04 * (-0.5 * (k - 0.0) + np.sqrt((k - 0.0) ** 2 + 0.05 ** 2))
iv = np.sqrt(true_w / T)
fit = fit_svi_slice(k, iv, T)
print(round(float(np.abs(fit.iv(k) - iv).max()), 5))
```
