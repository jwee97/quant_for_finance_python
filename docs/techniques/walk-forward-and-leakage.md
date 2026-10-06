---
title: "Walk-forward testing and look-ahead leakage"
slug: walk-forward-and-leakage
difficulty: 2
chapter: Ch. 22
prerequisites: [backtest-engine-and-costs]
stages: [11]
files: [src/validation/walk_forward.py, src/validation/leakage.py, experiments/stage11_validation.py]
figures: [22, 23]
tests: [tests/test_leakage.py]
models: []
---

# Walk-forward testing and look-ahead leakage

## In one sentence

Walk-forward testing refits a model only on the past and scores it on the future in rolling steps, and a leakage test proves the code cannot see tomorrow.

## The idea

Instead of one train/test split, the sample is cut into successive folds: fit on everything before date t, evaluate on the next block, move forward. Leakage is information from the
future entering a decision. The most reliable detector is a perturbation test: replace all data after a cutoff with noise and check that every output on or before the cutoff is identical.

## Why it matters

A leak can silently double a Sharpe ratio. A strategy that was never tested for causality has no business being trusted, however good its backtest looks.

## How this repo uses it

`src/validation/leakage.py` implements the perturbation test and includes a deliberately broken control strategy (built from tomorrow's return) that the test must catch. The Generation 5
framework runs the same check on every registered model and regime detector before any result is reported. `src/validation/walk_forward.py` produces the folds.

## What we found

Every production strategy passed (pre-split weights were bit-identical after perturbation) and the broken control was caught at every split date (EXP-016). The out-of-sample best model was
mean-CVaR at a Sharpe of 0.83, with a 90% block-bootstrap interval of [0.38, 1.27]: the interval, not the point estimate, is the finding (EXP-019).

## Going deeper

```
for each fold k:  fit on [0, t_k - embargo];  predict [t_k, t_k + step);  concatenate predictions
causality test:   output(data) on or before c   ==   output(data with everything after c replaced by noise) on or before c
```
The embargo is at least the label horizon (21 days here): a label that spans days t..t+21 cannot be used to train a model that predicts from t+10.

## Pitfalls

- Features that use the whole sample (full-sample standardisation, smoothed HMM probabilities) leak even when each row looks innocent.
- An embargo between training labels and the test start is needed when labels span multiple days.
- Walk-forward is still subject to the researcher's choices; it controls the model, not the search.

## Try it

```bash
python -m experiments.stage11_validation
quant backtest --model momentum        # prints the causality verdict with the result
```
