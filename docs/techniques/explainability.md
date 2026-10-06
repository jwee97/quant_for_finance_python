---
title: "Explaining a model: permutation importance, Shapley values, integrated gradients"
slug: explainability
difficulty: 3
chapter: Ch. 23
prerequisites: [calibration]
stages: [35]
files: [src/models/explain.py, experiments/stage35_explain.py, src/strategies/ml.py]
figures: [69]
tests: [tests/test_explain.py, tests/test_positioning_explain.py]
models: []
---

# Explaining a model: permutation importance, Shapley values, integrated gradients

## In one sentence

Attribution methods say which inputs drove a prediction, and each comes with an identity you can test.

## The idea

Permutation importance shuffles one feature and measures how much out-of-sample error rises. Shapley values divide a prediction among the features by averaging each feature's marginal
contribution over all subsets; with 12 features they can be computed exactly, and they must sum to the prediction minus the baseline. Integrated gradients integrate a network's gradient along a
path from a baseline; the attributions must sum to the change in output.

## Why it matters

A black box that makes money cannot be risk managed. And an explanation method that has not been checked can be wrong without anyone noticing.

## How this repo uses it

Stage 35 fits a small MLP and a gradient-boosted model on the twelve price features, explains 202 out-of-sample rows with all three methods, tests both identities on every row, and compares the
rankings.

Setting `evaluation: {explain: true}` in a specification asks every model that can explain itself (today `ml_ridge`) for a permutation-importance table of its final fit, which the tear sheet prints under "What drives the forecast". It is descriptive: the rows it is measured on overlap the training set.

## What we found

The identities held on every row (Shapley to rounding error, integrated gradients to about 0.3% of the prediction range), but the global rankings disagreed: Spearman correlations between method-model
pairs ran from -0.43 to 0.78. Permutation importance, which measures usefulness out of sample, was near zero or negative for most features, so the models use features that do not help.

## Going deeper

```
permutation importance_j = MSE(shuffle feature j) - MSE(original)
Shapley_j(x) = sum over subsets S not containing j of |S|!(n-|S|-1)!/n! * [ v(S + j) - v(S) ],    v(S) = mean prediction with features in S fixed to x
efficiency:  sum_j Shapley_j = f(x) - mean(f(background))
integrated gradients_j = (x_j - b_j) * integral_0^1 dF(b + a (x - b))/dx_j da      completeness: sum_j IG_j = F(x) - F(b)
```
With 12 features there are 4,096 subsets, so Shapley values are exact here instead of sampled.

## Pitfalls

- Importance in the model is not importance for the outcome.
- Shapley values depend on the background sample.
- Correlated features share credit unpredictably.

## Try it

```bash
python -m experiments.stage35_explain
```
