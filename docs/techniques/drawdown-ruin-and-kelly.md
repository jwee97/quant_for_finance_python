---
title: "Drawdowns, ruin probabilities and Kelly sizing"
slug: drawdown-ruin-and-kelly
difficulty: 3
chapter: Ch. 16
prerequisites: [resampling-and-monte-carlo, risk-managed-exposure]
stages: []
files: [src/probability/ruin.py, src/framework/allocators_portfolio.py]
figures: []
tests: [tests/test_probability.py, tests/test_ml_additions.py]
models: []
---

# Drawdowns, ruin probabilities and Kelly sizing

## In one sentence

How much to bet and how deep a loss to expect are two sides of one question; `src.probability.ruin` gives closed forms where they exist, simulation where they do not, and the Kelly allocator turns the answer into portfolio weights.

## The idea

A **drawdown** is the fall from the running peak. `drawdown_series` and `drawdown_episodes` describe the history; `drawdown_distribution` bootstraps the distribution of the maximum drawdown over a horizon (stationary bootstrap by default, so autocorrelation is kept), and `cdar` is the conditional drawdown at risk, the mean of the worst `1 - alpha` of drawdowns. For a Brownian motion with drift `mu` and volatility `sigma` there are exact results: `prob_fall_below` is the probability that wealth ever falls to a given fraction of its starting level, `prob_ruin_brownian` is the same for losing a given share of capital, and `expected_max_drawdown` is exact at zero drift (`sigma sqrt(pi T / 2)`) and grows with the horizon. `gambler_ruin_probability` is the discrete analogue.

**Kelly** maximises the expected log growth. For one asset `f* = (mu - r) / sigma^2`; `kelly_from_returns` estimates it from data with a shrinkage factor, `multi_asset_kelly` solves `f = Sigma^-1 (mu - r)`, and `kelly_growth` reports the growth rate at any fraction. Betting more than Kelly lowers growth and past twice Kelly the growth is negative; because `mu` is estimated, practitioners bet a fraction (half Kelly gives three quarters of the growth for half the variance). `drawdown_constrained_kelly` uses the closed form `P(max drawdown >= x) = x^(2/c - 1)` for leverage `c = f / f*` to choose the largest fraction whose probability of a given drawdown stays under a tolerance.

## Why it matters

Maximum drawdown is the number that ends careers and funds, and it is not a property of the Sharpe ratio alone: it depends on the leverage, the horizon and the path. Kelly links the three, and is the only sizing rule with a growth-optimality proof, but its inputs are the least reliable numbers in the book.

## How this repo uses it

Two allocators use it: `kelly` (fractional Kelly on the forecast mean and volatility, with an optional drawdown cap) and `black_litterman`, which shares the forecast-book plumbing. The risk stage can report the bootstrap drawdown distribution for any strategy's return series.

## What we found

The tests compare the gambler's ruin and Brownian formulas with simulation, check that the expected maximum drawdown is exact at zero drift, and verify the Kelly formulas and the drawdown constraint. Applied to the real ETF returns in [the findings](../institutional_findings.md), the Kelly fraction implied by a historical mean is far above what anyone would hold, which is the practical case for shrinkage.

## Pitfalls

- Full Kelly on an estimated edge over-bets whenever the estimate is too high, which is most of the time; use a fraction.
- Closed forms assume continuous Brownian paths; real drawdowns are deeper because of jumps and volatility clustering.
- A bootstrapped maximum drawdown from one history cannot exceed what the history contained, by much.
- Drawdown measures are path-dependent and noisy: three bad years in twenty make the estimate.

## Try it

```python
from src.probability.ruin import kelly_fraction, drawdown_constrained_kelly, prob_ruin_brownian

f = kelly_fraction(mu=0.06, sigma=0.15)
print(round(f, 2), round(prob_ruin_brownian(0.06, 0.15, 0.5), 3))
print(drawdown_constrained_kelly(0.06, 0.15, max_drawdown=0.3))
```
