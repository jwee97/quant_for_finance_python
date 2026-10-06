---
title: "Diffusion models for scenario generation"
slug: diffusion-scenarios
difficulty: 3
chapter: Ch. 21
prerequisites: [var-cvar-and-backtests]
stages: [36]
files: [src/models/diffusion.py, experiments/stage36_diffusion.py]
figures: [71, 72]
tests: [tests/test_diffusion.py]
models: []
---

# Diffusion models for scenario generation

## In one sentence

A diffusion model learns to turn noise into realistic return vectors, giving unlimited stress-test scenarios with the right fat tails and correlations.

## The idea

Noise is added to a training vector in many small steps; a network learns to undo each step. Sampling runs the learned reverse process from pure noise. Here the objects are 15-dimensional vectors of 21-day
returns, each divided by its volatility forecast, so scenarios can be rescaled to today's volatility. The comparison baselines are a multivariate normal and the empirical distribution of the same training vectors.

## Why it matters

Historical simulation can only replay what happened; a generative model can in principle smooth between events. But it can also invent unrealistic extremes.

## How this repo uses it

`src/models/diffusion.py` is a small DDPM with a cosine schedule; Stage 36 reads off 5% and 1% value at risk of the equal-weight book at each monthly origin and scores them by Kupiec coverage and pinball loss.

## What we found

Diffusion scenarios reproduced the training data's fat tails (mean excess kurtosis 2.2 against 2.15) with a larger correlation error than the baselines and invented extremes in about 3% of scenarios. The 5% VaR
had an exceedance rate near 5% (5.3%, against 7.0% for the others), but no significant pinball-loss advantage (declared hypothesis h_diffusion, rejected).

## Going deeper

```
forward:  x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I),   cosine schedule for abar_t
train:    minimise E || eps - eps_theta(x_t, t) ||^2
reverse:  x_{t-1} = posterior mean( x_t, x0_hat ) + sqrt(beta_tilde_t) * noise,   x0_hat = (x_t - sqrt(1-abar_t) eps_theta) / sqrt(abar_t), clipped
VaR_5% of the equal-weight book = 5th percentile of  mean_i ( sigma_i * z_i )  over 2,000 generated z-vectors
```
Clipping the predicted `x0` keeps the early reverse steps from amplifying network error.

## Pitfalls

- Generated scenarios can look plausible and be wrong in the tail.
- 1% VaR with 187 months has two expected breaches: do not over-read it.
- Keep the volatility scaling out of the learned object so it is not learned wrongly.

## Try it

```bash
python -m experiments.stage36_diffusion
```
