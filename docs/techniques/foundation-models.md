---
title: "Foundation models for time series (zero-shot Chronos)"
slug: foundation-models
difficulty: 3
chapter: Ch. 23
prerequisites: [deep-time-series-models]
stages: [34]
files: [experiments/stage34_foundation.py]
figures: [68]
tests: [tests/test_frontier_models.py]
models: []
---

# Foundation models for time series (zero-shot Chronos)

## In one sentence

A foundation model is pre-trained on huge numbers of series and can forecast a new series with no training at all.

## The idea

Chronos-Bolt takes a context window and outputs forecast quantiles directly. Used zero-shot, none of its parameters ever see this universe's data. The main methodological risk is contamination: its
pre-training corpus is large, partly documented and overlaps the test period, so a good score could reflect memorised prices.

## Why it matters

If a general model forecast markets well with no fitting it would be a big result; the point of running it is to find out cheaply and to state the contamination asymmetry in advance: a good result would
not be clean evidence, a bad result is.

## How this repo uses it

Stage 34 feeds the 252 normalised returns before each origin to Chronos-Bolt (small), sums the median path over 21 days and scales by volatility. The stage is descriptive by declaration, and the model revision hash is
stored with the results.

## What we found

Zero-shot Chronos forecasts were significantly worse than the historical mean on CRPS (difference +0.0016, p about 0.00002), mostly because it extrapolates a positive drift (79% of its forecasts are positive,
with absolute size about three times the historical mean's), although its rank information coefficient was about 0.08, similar to the best trained models.

## Going deeper

```
context = last 252 normalised returns  ->  Chronos-Bolt predicts quantiles of the next 21 steps
return forecast = ( sum of the 21 median-path values ) * EWMA daily vol at the origin
```
The model is applied as is: no gradient step, no scaling parameter fitted on this data. Its zero-shot output is a drift estimate and a rank signal, not a calibrated distribution, which is why the same EWMA spread is attached to it.

## Pitfalls

- Ranking skill and probabilistic skill are different things.
- Zero-shot results depend on the context length and scaling.
- Hub models can be updated; pin the revision.

## Try it

```bash
pip install chronos-forecasting einops
python -m experiments.stage34_foundation
```
