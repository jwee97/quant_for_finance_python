---
title: "Option pricing, Greeks and implied volatility"
slug: option-pricing-and-greeks
difficulty: 3
chapter: Ch. 18
prerequisites: [performance-metrics]
stages: []
files: [src/derivatives/pricing.py, src/derivatives/greeks.py, src/derivatives/iv.py, src/derivatives/schema.py]
figures: []
tests: [tests/test_derivatives.py]
models: []
---

# Option pricing, Greeks and implied volatility

## In one sentence

An option's price is an expectation under a model, its Greeks are the sensitivities of that price, and its implied volatility is the volatility that makes the model match the market; `src.derivatives` implements the pricers, the Greeks and a robust implied-volatility solver, each checked against an independent method.

## The idea

**Black-Scholes-Merton** with a continuous dividend yield `q` prices a European call as `S e^{-qT} N(d1) - K e^{-rT} N(d2)` with `d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T))`. `black76_price` prices options on forwards and futures, `bachelier_price` handles a normal model (rates, spreads, negative prices), and `binomial_price` (Cox-Ross-Rubinstein) and `american_lsmc` (Longstaff-Schwartz least-squares Monte Carlo) price early exercise. Beyond constant volatility, `heston_price` uses the characteristic function and numerical Fourier inversion for stochastic volatility, `heston_calibrate` fits it to a smile, `merton_price` sums the Poisson mixture of jump diffusions and `sabr_vol` and `sabr_calibrate` give the Hagan approximation and its calibration.

**Greeks**: delta `dV/dS`, gamma `d2V/dS2`, vega `dV/dsigma`, theta `dV/dt`, rho. `bsm_greeks` and `black76_greeks` give closed forms, `numerical_greeks` differentiates any pricer by finite differences, `portfolio_greeks` aggregates positions, and `pnl_explain` splits a one-period P&L into delta, gamma, vega and theta pieces (the Taylor expansion), with an unexplained residual.

**Implied volatility** (`implied_vol`) inverts the pricing function with a safeguarded Newton-bisection hybrid. It returns NaN, not a wrong number, when the price is outside the no-arbitrage bounds or so close to them that vega is zero; variants cover Black-76, Bachelier and American options. `src.derivatives.schema` normalises a raw chain, validates it, extracts out-of-the-money quotes and the forward implied by put-call parity, and reports static-arbitrage violations (`arbitrage_report`).

## Why it matters

Mistakes in options code are silent: a wrong dividend convention or a day-count error gives plausible prices that make a strategy's backtest look profitable. Checking each pricer against an independent method (put-call parity, binomial convergence, Monte Carlo, finite-difference Greeks) is what makes the later strategy results believable.

## How this repo uses it

These are the foundations of the surface, volatility and backtesting modules. The same schema is used for the synthetic option market and for real option CSVs loaded through `load_option_csv`, so a strategy tested on synthetic data runs unchanged on a vendor file.

## What we found

The tests check Black-Scholes against put-call parity and known values, binomial convergence to Black-Scholes, closed-form Greeks against finite differences, Heston against its Black-Scholes limit and against Monte Carlo, Merton against its jump-free limit, American puts against the binomial tree, and the implied-volatility solver round-trips a grid of prices. The solver's maximum round-trip error on the test grid is a few parts in a million, and unidentifiable prices return NaN.

## Pitfalls

- Use the forward implied by the chain (parity), not a spot and a guessed dividend, when inverting prices.
- American options are not European options: do not invert a European model on American quotes except for out-of-the-money calls on non-dividend stocks.
- Greeks of a model are not the market's Greeks: vega is sticky-strike or sticky-delta depending on how the surface moves.
- Stale quotes and wide spreads make implied volatilities meaningless; filter on spread and size.

## Try it

```python
from src.derivatives.pricing import bsm_price
from src.derivatives.greeks import bsm_greeks
from src.derivatives.iv import implied_vol

price = bsm_price(100, 105, 0.5, 0.02, 0.01, 0.25)
print(round(float(price), 4), round(float(implied_vol(price, 100, 105, 0.5, 0.02, 0.01)), 4))
print({k: round(float(v), 4) for k, v in bsm_greeks(100, 105, 0.5, 0.02, 0.01, 0.25).items()})
```
