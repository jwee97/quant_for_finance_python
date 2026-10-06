---
title: "Calibration: do 70% forecasts come true 70% of the time?"
slug: calibration
difficulty: 2
chapter: Ch. 20
prerequisites: [probabilistic-forecasting]
stages: [19, 35]
files: [src/models/probabilistic.py, experiments/stage35_explain.py, src/framework/adaptive.py]
figures: [38, 70]
tests: [tests/test_probabilistic.py, tests/test_explain.py, tests/test_adaptive_integration.py]
models: []
---

# Calibration: do 70% forecasts come true 70% of the time?

## In one sentence

Calibration repairs a forecaster whose stated probabilities do not match how often events actually happen.

## The idea

A reliability diagram bins forecasts and plots the observed frequency in each bin; a calibrated forecaster lies on the diagonal. Platt scaling fits a logistic curve to the raw output;
isotonic regression fits a monotone step function and makes fewer assumptions but needs more data.

## Why it matters

Sizing a bet by a probability only works if the probability is honest. But calibration is itself a fitted model and can make things worse if it is fitted on too little or the wrong data.

## How this repo uses it

Stage 19 applies Platt scaling to a direction classifier. Stage 35 refits Platt and isotonic calibrators each month on only the outcomes that had matured, and compares them with the raw
probabilities by log loss, Brier score and expected calibration error.

`forecast: {confidence: {method: platt | isotonic}}` recalibrates P(up) walk-forward inside the pipeline: each month the map from the forecast's z-score to the probability of a positive 21-day return is refit on matured labels only, and the Brier score and reliability table in the tear sheet use the calibrated probability.

## What we found

In Stage 19 the Platt step made the calibration slope worse (0.35 raw, -0.47 after), because it was fitted on a short late block (EXP-045). In Stage 35 isotonic did not beat Platt (declared hypothesis rejected), and
the raw probability had the lowest log loss of all arms.

## Going deeper

```
Platt:     p_cal = sigmoid( a * logit(p_raw) + b )            fit a, b by logistic regression on held-out (p_raw, outcome)
Isotonic:  p_cal = f(p_raw), f non-decreasing, minimising sum (y - f(p))^2     (pool adjacent violators)
ECE:       sum over bins  (n_bin / n) * | mean(y in bin) - mean(p in bin) |
log loss:  - mean( y ln p + (1 - y) ln(1 - p) )
```
Calibration slope = the coefficient from regressing outcomes on `logit(p)`: 1 is calibrated, below 1 means forecasts too extreme.

## Pitfalls

- Calibrate on data the model has not seen, with enough of it.
- An isotonic fit can output exactly 0 or 1, which a log-loss score punishes without mercy.
- Calibration cannot add information that is not there.

## Try it

```bash
python -m experiments.stage35_explain
```
