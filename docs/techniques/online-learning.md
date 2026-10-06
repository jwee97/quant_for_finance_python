---
title: "Online learning: updating as data arrives"
slug: online-learning
difficulty: 3
chapter: Ch. 20
prerequisites: [probabilistic-forecasting]
stages: [22]
files: [src/models/online.py, experiments/stage22_online.py, src/strategies/online.py]
figures: [44, 45]
tests: [tests/test_online.py, tests/test_online_model.py]
models: [online_ridge]
---

# Online learning: updating as data arrives

## In one sentence

Instead of training once and freezing, an online learner nudges its parameters each time a new observation arrives.

## The idea

Recursive least squares re-estimates a linear model with exponential forgetting; normalised least mean squares is a cheaper gradient step; a Kalman filter treats the coefficients as a
random walk. At the portfolio level, expert-aggregation rules such as Hedge reweight strategies by their recent performance, with a guaranteed regret bound against the best expert in hindsight.

## Why it matters

Markets drift, so a frozen model decays. But a model that adapts quickly also chases noise. The right forgetting rate is the whole problem.

## How this repo uses it

`src/models/online.py` implements the three learners and the aggregation; Stage 22 gives them the same features, volatility forecast and origins as the annually refitted Stage 19 ridge, so
only the update scheme differs.

In the plugin framework the `online_ridge` model wraps the same three updaters (`updater: ridge | nlms | kalman`): at each month-end origin it first absorbs only the origins whose 21-day label is already complete, then predicts. Run it with `quant backtest --model online_ridge --tearsheet`; `forgetting` is the knob that matters.

## What we found

No online model beat the annually refitted ridge (EXP-062), and the Hedge aggregate of strategy sleeves did not beat the inverse-volatility blend (EXP-063).

## Going deeper

```
RLS:   K_t = P_{t-1} x_t / (lambda + x_t' P_{t-1} x_t);  b_t = b_{t-1} + K_t (y_t - x_t' b_{t-1});  P_t = (P_{t-1} - K_t x_t' P_{t-1}) / lambda
NLMS:  b_t = b_{t-1} + mu * x_t (y_t - x_t' b_{t-1}) / (eps + x_t' x_t)
Hedge: w_k,t+1 proportional to w_k,t * exp(-eta * loss_k,t);  regret vs best expert <= sqrt(T ln K / 2)
```
`lambda < 1` makes RLS forget old data exponentially: an effective window of `1/(1 - lambda)` observations.

## Pitfalls

- The refit-versus-online comparison depends on the forgetting factor; the stage reports the sweep.
- Regret bounds are about the worst case, not typical performance.
- Online updating hides when a model has stopped working.

## Try it

```bash
python -m experiments.stage22_online
```
