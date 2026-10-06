---
title: "Machine learning on returns: ridge, forests and honesty"
slug: machine-learning-for-returns
difficulty: 3
chapter: Ch. 23
prerequisites: [walk-forward-and-leakage, information-coefficient]
stages: [13, 19]
files: [src/models/machine_learning.py, src/strategies/ml.py, experiments/stage13_ml.py]
figures: [24]
tests: [tests/test_strategies.py]
models: [ml_ridge]
---

# Machine learning on returns: ridge, forests and honesty

## In one sentence

Flexible models can find nonlinear patterns, and in financial returns the signal is so weak that most of what they find is noise.

## The idea

A ridge regression penalises large coefficients; a random forest averages many trees. Both are trained walk-forward on price features to predict next-period returns or direction. Evaluation
uses three axes that can disagree: classification accuracy (AUC), forecast error, and trading performance net of costs, because a 0.1% day and a 5% day count equally in AUC but not in trading.

## Why it matters

Financial data has low signal-to-noise and few independent observations. The standard result is that regularised linear models are hard to beat, and flexible models have to earn their
complexity.

## How this repo uses it

Stage 13 compares ridge, a random forest and others on the Generation 1 universe; the library's `ml_ridge` model brings a walk-forward ridge into the plugin framework; Stage 19 reuses ridge as the
annually refitted price-only benchmark for every forecast stage.

## What we found

Nonlinear machine learning did not give a genuine out-of-sample economic improvement over the simpler models (EXP-021, rejected), even though the best-AUC model was also the best on net Sharpe among the ML models.

## Going deeper

```
ridge:   b = argmin sum (y - X b)^2 + alpha ||b||^2 = (X'X + alpha I)^-1 X'y
forest:  average of B trees, each fit to a bootstrap sample and a random subset of features at each split
AUC = P( score of a random up-month > score of a random down-month )
```
Always evaluate on three axes: AUC (ranking), error (calibration of size), and net Sharpe (what you can bank).

## Pitfalls

- Hyperparameter tuning on the test period is overfitting.
- Feature standardisation must use training statistics only.
- Accuracy above 50% is not profit: check the size of the wins and losses.

## Try it

```bash
quant backtest --model ml_ridge --tearsheet
```
