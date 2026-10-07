---
title: "Gradient boosting and representation learning on returns"
slug: gradient-boosting-and-representation-learning
difficulty: 3
chapter: Ch. 23
prerequisites: [machine-learning-for-returns, deep-time-series-models]
stages: []
files: [src/strategies/ml.py, src/strategies/representation.py, src/models/representation.py]
figures: []
tests: [tests/test_ml_additions.py]
models: [ml_trees, deep_representation]
---

# Gradient boosting and representation learning on returns

## In one sentence

Two ways to ask whether a flexible learner finds signal that a ridge regression misses: tree ensembles on the same price features, and an unsupervised encoder trained on return windows with a ridge on top.

## The idea

**`ml_trees`** replaces the ridge of `ml_ridge` with a tree ensemble. Trees capture interactions (momentum matters more when volatility is low) that a linear model cannot. The learner is a parameter: scikit-learn's histogram gradient boosting (the default), random forest, extra trees, or, if installed (extra `boosting`), LightGBM, XGBoost or CatBoost. The defaults are deliberately conservative for a signal-to-noise ratio this low: depth 3, learning rate 0.05, leaves of at least 100 rows, strong L2 regularisation. The protocol is the same as for every ML model in the library: features at each month-end, an expanding training window, an embargo equal to the label horizon so no training label overlaps the test period, refit once a year.

**`deep_representation`** uses the fact that labels are scarce while return windows are plentiful. An encoder is trained **without labels** on every window of volatility-normalised returns available at the refit date: an **autoencoder** that reconstructs the window through a narrow bottleneck, or a **contrastive** encoder (SimCLR style, NT-Xent loss) that must map two random augmentations of a window (scaling, jitter, masking) to nearby points. A ridge then maps the embedding to the next 21-day return using only matured labels. If the embedding holds regularities that twelve hand-made features do not, the second stage should forecast better.

## Why it matters

Gradient boosting dominates tabular machine learning, and representation learning is the leading idea for scarce labels in the time-series literature. The question for finance is whether either survives a low signal-to-noise ratio, costs and honest validation; including them with identical protocols makes the answer a number rather than an opinion.

## How this repo uses it

Both are registry models with forecast distributions, so they pass through the same allocators, costs and validation as everything else and appear in the same tables. `ml_trees` reports permutation importance through the explainability module.

```bash
quant backtest --model ml_trees --param learner=random_forest
quant backtest --model deep_representation --param kind=contrastive
```

## What we found

In [the institutional findings](../institutional_findings.md), five variants on the 15 ETFs as cross-sectional long-short books, net of costs, with the same walk-forward protocol: `ml_ridge` -0.40 net Sharpe, `ml_trees` with gradient boosting -0.32 and with a random forest -0.19, `deep_representation` with the autoencoder -0.03 and the contrastive encoder +0.07; the best deflated probability is 0.18. None has a Sharpe worth trading, and the ridge itself loses money. The honest reading is that price-only features carry little cross-sectional signal in 15 liquid ETFs at the one-month horizon, however flexible the learner, and that the learned representations are slightly less bad than hand-made features, not good. Turnover of 12 to 19 times a year at 10 bps explains part of the loss. The tests check that every learner trains and forecasts without look-ahead, that missing optional packages give a clear error, and that the encoders reduce their training loss.

## Pitfalls

- With a monthly horizon and 15 assets there are only a few thousand labelled rows; any flexible model overfits.
- Boosting libraries differ in defaults; compare them with the same depth and regularisation.
- Contrastive augmentations encode an assumption about what is a nuisance (scale, short noise); a wrong choice removes signal.
- Unsupervised pre-training uses windows from the past only, but fitting the encoder on the future would be a leak: the code uses windows that end on or before the refit date.

## Try it

```bash
quant backtest --model ml_trees --param learner=hgb --param max_depth=2
quant backtest --model deep_representation --param kind=autoencoder --param embed_dim=4
```
