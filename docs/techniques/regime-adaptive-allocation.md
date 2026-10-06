---
title: "Adapting the portfolio to the regime"
slug: regime-adaptive-allocation
difficulty: 3
chapter: Ch. 19
prerequisites: [regime-detection, risk-parity, mean-cvar]
stages: [31]
files: [src/framework/allocation.py, src/framework/risk.py, experiments/stage31_adaptive.py, config/adaptive.yaml]
figures: [62, 63, 64]
tests: [tests/test_framework.py, tests/test_regime_strategies.py]
models: []
---

# Adapting the portfolio to the regime

## In one sentence

Choose a different allocator, tilt and risk target in each regime, weighted by the regime probabilities, and test it against a placebo.

## The idea

The `regime_switch` allocator blends other allocators (risk parity in high volatility, mean-CVaR in crisis, mean-variance in low volatility, commodity tilts in inflation) with weights equal to
the regime probabilities. A regime-dependent risk policy sets the volatility target as the probability-weighted target. Because regimes persist and volatility clusters, you must show the
regime *path* matters: a placebo shifts the regime path in time and re-runs everything.

## Why it matters

It is the textbook institutional move, and the one most likely to look good through hindsight.

## How this repo uses it

Stage 31 declares four decisions in `config/adaptive.yaml` before results: the allocation switch against risk parity and equal weight, a placebo that circularly shifts the regime path, regime-dependent
volatility targets, and regime-conditional trust in the signals.

## What we found

Regime timing was retained against the placebo (EXP-081, p = 0.005) and the adaptive signal trust was retained but fragile (EXP-083, p = 0.068), whereas the allocation switch did not beat risk parity and equal weight (EXP-080) and
the adaptive risk policy did not beat a constant target (EXP-082). Four decisions were declared, so some caution on multiplicity applies.

## Going deeper

```
w_t = sum_regimes  P(regime | data to t) * w_t^(allocator for that regime)  +  tilt
vol target_t = sum_regimes P(regime) * target_regime;     scale_t = target_t / forecast_vol_t
placebo: shift the regime path circularly by a random lag >= 252 days, re-run, collect the Sharpe distribution; p = share >= observed
```
The placebo preserves the regime frequencies and persistence, so beating it means the timing, not the average mix of books, added value.

## Pitfalls

- A placebo is stronger than a benchmark: it keeps everything except the timing.
- Soft (probability) blends avoid whipsaw from hard switches but still turn over.
- Each extra regime-specific rule is a parameter.

## Try it

```bash
python -m experiments.stage31_adaptive
```
