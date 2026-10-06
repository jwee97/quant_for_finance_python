---
title: "Market regimes: HMM, volatility states, change points"
slug: regime-detection
difficulty: 3
chapter: Ch. 20
prerequisites: [volatility-forecasting]
stages: [16, 31]
files: [src/models/regimes.py, src/framework/regimes.py, src/features/regime_rules.py, experiments/stage16_regimes.py, src/framework/adaptive.py]
figures: [30, 31, 32, 62]
tests: [tests/test_regimes.py, tests/test_framework.py, tests/test_adaptive_integration.py]
models: []
---

# Market regimes: HMM, volatility states, change points

## In one sentence

A regime detector labels each day with the state the market is probably in, and outputs a probability rather than a verdict.

## The idea

A hidden Markov model assumes a few hidden states, each with its own return distribution, with persistence between days. Volatility-state models rank today's volatility in its own history.
Rule-based regimes (inflation from CPI, a credit-spread spike) use named thresholds. Bayesian online change-point detection watches for the moment a distribution shifts. The crucial
distinction is filtered probabilities (information up to today, tradable) versus smoothed probabilities (the whole sample, hindsight only).

## Why it matters

If markets behave differently in different states, everything downstream should react. But a backtest on smoothed probabilities overstates results badly, and regimes found in daily returns are
mostly volatility regimes.

## How this repo uses it

Regimes are `Regime(name, probability)` objects. `src/framework/regimes.py` has the registered detectors (HMM, volatility state, macro, composite, static control) and the framework's
look-ahead test runs on each. Stage 16 evaluates them; Stage 31 uses the composite in the adaptive pipeline with a circular-shift placebo.

In the plugin framework two more detectors are registered, `gmm` (a Gaussian mixture on trailing volatility and trend features) and `bocpd` (Bayesian online change-point detection), beside `hmm` and the rule-based ones. A regime feeds the rest of the pipeline in three places: `forecast: {regime_spread: ...}` rescales the forecast spread by what the model's errors actually were in the current regime, `allocation` can be a `regime_switch`, and `risk: {limits: {gross_caps: ..., drawdown: ...}}` caps gross exposure by regime and de-risks in a drawdown.

## What we found

A filtered high-volatility state predicts larger absolute returns next day (EXP-027, a sanity check) but not a different mean return (EXP-028). Regime-aware allocation rules in Stage 16 were not
retained (EXP-031 to EXP-033). The automated look-ahead test flagged all 20 look-ahead controls and passed all 16 honest rules (EXP-034).

## Going deeper

```
HMM: hidden state s_t in {1..K}, P(s_t = j | s_{t-1} = i) = A_ij, return r_t | s_t = k ~ N(mu_k, sigma_k^2)
filtered:  P(s_t | r_1..r_t)         (forward algorithm; tradable)
smoothed:  P(s_t | r_1..r_T)         (forward-backward; uses the future)
vol state: rank of EWMA volatility today within its own trailing history
BOCPD:     run-length posterior P(r_t | data), updated with a hazard rate H = 1/expected spell length
```
Only filtered probabilities may be used in a backtest. The expected duration of a state is `1/(1 - A_kk)`: a persistence of 0.95 means 20-day spells.

## Pitfalls

- Labels fitted on the whole sample leak the future.
- The number of regimes is a choice; more states fit better and generalise worse.
- Persistence is what makes a regime tradable at a monthly horizon.

## Try it

```bash
quant list detectors
python -m experiments.stage16_regimes
```
