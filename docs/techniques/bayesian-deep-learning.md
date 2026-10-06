---
title: "Uncertainty from neural networks: ensembles and MC dropout"
slug: bayesian-deep-learning
difficulty: 3
chapter: Ch. 23
prerequisites: [deep-time-series-models]
stages: [33]
files: [src/models/deep_forecast.py, experiments/stage33_frontier.py]
figures: [67]
tests: [tests/test_frontier_models.py]
models: []
---

# Uncertainty from neural networks: ensembles and MC dropout

## In one sentence

Train several networks (an ensemble) and sample with dropout switched on; where they disagree, the model is telling you it is unsure.

## The idea

A deep ensemble trains the same network from different seeds. Monte Carlo dropout keeps dropout active at prediction time, so repeated passes give different answers. The variance of the answers is an
estimate of epistemic uncertainty (what the model does not know). Adding it to the forecast variance gives a spread that depends on the model's own disagreement.

## Why it matters

A forecast that knows when it does not know can size positions down. But only if the disagreement carries information about the error.

## How this repo uses it

Stage 33 computes 3 seeds by 30 dropout passes for each model and widens the forecast standard deviation by the factor sqrt(1 + epistemic variance of z), with no free parameter, then asks whether CRPS improves.

## What we found

The widening was tiny (about 0.2%) and made CRPS slightly worse, not better (declared hypothesis h_bayesian, rejected): the networks, regularised heavily, agree with each other and are mostly near the mean, so their disagreement does not predict
their error.

## Going deeper

```
ensemble: K networks with different seeds -> predictions f_1..f_K
MC dropout: T stochastic forward passes with dropout on -> g_1..g_T
epistemic variance of z = Var over all K*T predictions
widened forecast std = s * sqrt(1 + epistemic variance)
```
The widening is relative: if the networks disagree by 0.06 in z units (variance 0.0036) the std grows by 0.18%, which is about what was observed.

## Pitfalls

- Ensembles of near-identical networks underestimate uncertainty.
- Dropout rate is a hyperparameter that sets the size of the uncertainty.
- Check coverage of the resulting intervals, not only the score.

## Try it

```bash
python -m experiments.stage33_frontier
```
