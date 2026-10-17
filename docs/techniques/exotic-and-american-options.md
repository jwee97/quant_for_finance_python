---
title: "Exotic and American options: closed forms, grids and simulation, and model risk"
slug: exotic-and-american-options
difficulty: 3
chapter: Ch. 18
prerequisites: [option-pricing-and-greeks, option-pricing-models-and-calibration]
stages: []
files: [src/derivatives/exotics.py, src/derivatives/pde.py, src/derivatives/pide.py, src/numerics/linsolve.py, src/assets/option_models.py]
figures: []
tests: [tests/test_exotics.py, tests/test_pde.py, tests/test_pide.py, tests/test_numerics.py, tests/test_option_models.py, tests/test_models_api.py]
models: []
---

# Exotic and American options: closed forms, grids and simulation, and model risk

## In one sentence

A vanilla's price is pinned by its quote, but a digital, a barrier, an average-price or an American option depends on *how* the price moves, so it can be worth a different amount under each model that reproduces the vanillas, and the honest answer is the range, set beside the models' range on the vanilla (which is only how well each fits).

## The idea

**Digital** options pay a fixed amount if the price ends on one side of the strike. Their price is a risk-neutral probability (`e^{-rT} P(S_T > K)`), which is the *slope* of the vanilla price in the strike, so it is the slope of the smile in disguise: under a skewed model a digital call at the money is worth more than Black-Scholes says. **Barrier** options are cancelled (knock-out) or only created (knock-in) if the price touches a level; a knock-in plus the matching knock-out is the vanilla. **Average-price** (Asian) options pay on the average of the price on fixed dates, which damps the tails. **American** options may be exercised at any time, which turns pricing into an *optimal stopping* problem.

Three ways to price them. **Closed forms** exist under Black-Scholes (Reiner-Rubinstein's eight barrier options, digitals, the geometric-average option, the perpetual American option). **Grids** solve the pricing equation: the Black-Scholes PDE by finite differences, and for jump models the *integro*-differential equation (PIDE) in which the jumps appear as a non-local sum. **Monte Carlo** simulates paths under any model and applies the payoff, with a *control variate* (a correlated quantity of known mean, such as the geometric-average option for the arithmetic one) to remove most of the noise.

Early exercise is a **linear complementarity problem** at every time step: the value must stay above the payoff, and wherever it is above, the pricing equation holds. For a tridiagonal system with an M-matrix the **Brennan-Schwartz** sweep solves it exactly in one pass (Jaillet, Lamberton and Lapeyre 1990); **projected SOR** solves it iteratively and makes no assumption about where the exercise region is. The jump term of the PIDE is explicit, so it does not disturb the constraint.

## Why it matters

Listed options on ETFs are American, so the right to exercise early is part of every quote of a deep put. Exotics are where models can disagree: they were all fitted to the same vanillas, so the part of the *spread* of their prices on an exotic that is not already there on the vanilla (where it is only how well each model fits) measures how much of the answer comes from the modelling choice rather than from the data. That excess, not any one model's price, is what to carry into a decision.

## How this repo uses it

`derivatives/exotics.py` has the closed forms and a Monte Carlo engine (`mc_price`, `asian_arithmetic_mc`, `barrier_mc`) under any of the platform's models; `derivatives/pde.py` the finite-difference solver for vanillas (European and American), barriers and digitals; `derivatives/pide.py` the jump-model solver of Cont and Voltchkova (2005), now with early exercise; `numerics/linsolve.py` the linear solvers.

The **Options tab → Model prices** view prices one product under every model you fit and under Black-Scholes, and says in words how far apart they are: a table with standard errors for the simulated products, a bar chart of each price against Black-Scholes, the price by strike for a vanilla or digital, the early-exercise boundary for an American option. `quant models price SPY --product american --right put --strike 105` does the same in the terminal.

## What we found

- **Every identity that must hold, does.** Digital call plus put pays the discount factor under every model; a knock-in plus the matching knock-out is the vanilla (to the Monte Carlo error); a far barrier changes nothing and a near one kills the option; the grid reproduces the Reiner-Rubinstein formulas; the geometric-average option is a Black formula the simulation reproduces; the arithmetic average costs more than the geometric one for a call, and Levy's approximation is close.
- **Discrete monitoring is a real effect, and the Brownian bridge removes it.** A barrier watched on the daily close is a weaker barrier than one watched continuously. For Black-Scholes the bridge probability corrects the bias exactly; for the jump and stochastic-volatility models no such correction exists, so their knock-outs are priced a little high and the page says so beside the price.
- **The American PIDE agrees with an independent method.** Without jumps it reproduces the Black-Scholes American grid to 5e-4 on an 8.6 put; under Merton, Kou, variance gamma and NIG it agrees with a Longstaff-Schwartz simulation to within 1.5 standard errors (the gaps were 0.01 to 0.03), is converged to a fifth of a cent under grid refinement, and gives the early-exercise premium (0.32 to 0.39 on a 6.6 to 7.7 put at the money, one year). Without dividends a call is never exercised early, and the solvers agree.
- **Whether the models disagree on an exotic depends on the exotic, and the vanilla is the yardstick.** On the platform's simulated SPY chain (88 days, five models fitted to the same quotes) the at-the-money call ranged over 10% of the Black-Scholes price across the models, which is only how well each fits (Heston to 0.35 volatility points, Kou and Merton to about 1.5, variance gamma to 2.6). A 90% down-and-out call (10%), an average-price call (11%), a digital put (11%) and an American put (14%) had the same range: no model risk beyond the fit. A knock-in call at 110% of the price did not: 3.59 under Heston, 4.94 under NIG, 6.1 to 6.5 under variance gamma, Kou and Merton and 11.14 under Black-Scholes, a range of 26%. The simulated surface is itself a model, so this shows how much the answer can move with the family, not what the market would pay.

## Going deeper

```
digital call  = -dC/dK = e^{-rT} P(S_T > K)                     (a central difference of the Lewis price)
barrier:      in + out = vanilla;    discrete monitoring bias ~ 1 / sqrt(steps) without the bridge
bridge:       P(no touch between dates) = 1 - exp(-2 ln(S_i/H) ln(S_{i+1}/H) / (sigma^2 dt))
American LCP: min( (I - theta dt L) V - rhs ,  V - payoff ) = 0 at every step
PIDE:         V_tau = (sigma^2/2) V_xx + b V_x - rV + Int [V(x+z) - V(x) - z V_x] nu(dz)
              small jumps (|z| < half a cell) as extra diffusion, large jumps summed on the grid, jump term explicit,
              drift b solved from the *discrete* martingale condition, M-matrix guard: sigma^2 >= |b| dx
```

## Pitfalls

- **A knock-in is not priced by its own grid** (the grid prices the knock-out and a knock-in is the vanilla minus it); and a barrier already crossed is worth nothing (or the vanilla, for a knock-in).
- **Heston has no American price here.** Its variance is a second state variable, so the one-dimensional early-exercise solver does not apply; the table says "not offered" with the reason.
- **Longstaff-Schwartz is a lower bound with noise.** It is used here only as an independent check; the grid is the price.
- A simulated price is a mean with a standard error: compare models on the same seed, as the page does, and quote the range with the noise in mind.
- With a chain that is itself a model (the platform's ETF chains), the spread between models says nothing about the market.

## Try it

```bash
quant models price SPY --product digital --right put --strike 100
quant models price SPY --product barrier --barrier-kind up-in --barrier 110 --paths 100000
quant models price SPY --product american --right put --strike 105 --models merton nig
python -m pytest tests/test_exotics.py tests/test_pde.py tests/test_pide.py -q
```
In the dashboard: **Options → Model prices**, choose a product, press *Price the product*.
