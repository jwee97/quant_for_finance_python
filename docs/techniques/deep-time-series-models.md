---
title: "Deep learning for time series: transformers, mixers, N-BEATS, N-HiTS"
slug: deep-time-series-models
difficulty: 3
chapter: Ch. 23
prerequisites: [machine-learning-for-returns, probabilistic-forecasting]
stages: [27, 33]
files: [src/models/deep_forecast.py, src/models/deep_family.py, experiments/stage27_deep.py, experiments/stage33_frontier.py, src/strategies/deep.py]
figures: [54, 55, 66]
tests: [tests/test_deep_forecast.py, tests/test_frontier_models.py, tests/test_deep_models.py]
models: [deep_window]
---

# Deep learning for time series: transformers, mixers, N-BEATS, N-HiTS

## In one sentence

Modern neural forecasters read a window of past returns and output a forecast; on financial returns their main risk is learning noise.

## The idea

A patch transformer splits the last 252 normalised days into 21-day patches and applies attention across patches. An MLP-mixer alternates mixing across time and across features. N-BEATS stacks
blocks that each subtract what they explain (a backcast) and add to the forecast; N-HiTS gives each block a differently pooled view of the window. A TimeMixer-style model reads the window at several
scales. All are trained with early stopping, three seeds, and the target is the 21-day return in units of forecast volatility.

## Why it matters

They dominate time-series forecasting benchmarks in other domains. Whether that carries to monthly asset returns, where the signal is tiny, is exactly what a careful benchmark must decide, and the benchmark
matters: beating a weak benchmark proves nothing.

## How this repo uses it

Stage 27 tests a patch transformer and an MLP-mixer against the annually refitted ridge. Stage 33 re-tests four models (N-BEATS, N-HiTS, TimeMixer-style, graph attention) against the asset's historical
mean, declared as the primary benchmark because Stage 27 showed that the ridge was the weak one.

The `deep_window` model puts the five networks (`kind: patchtst | tsmixer | nbeats | nhits | timemixer`) behind the plugin interface: refit yearly on matured origins with a 21-day embargo, early stopping on the last fifth, forecast = predicted z-score times the EWMA volatility. `quant backtest --model deep_window --param kind=nbeats` gives it the same costs, tear sheet and causality check as any strategy.

## What we found

In Stage 27 the MLP-mixer beat the ridge significantly (EXP-073), but the ridge was worse than a plain historical mean. In Stage 33 no model beat the historical mean after Benjamini-Hochberg control (the best,
N-BEATS, had a CRPS difference of -0.00009, p = 0.32), while all four still beat the ridge. Rank information coefficients were 0.05 to 0.10 across models.

## Going deeper

```
input:   x = last 252 daily returns / EWMA daily vol at the origin, clipped at +-5      target: z = 21-day return / EWMA 21-day vol
N-BEATS block: h = MLP([x, asset embedding]);  backcast, forecast = split(h);  x <- x - backcast;  total forecast += forecast
N-HiTS block:  x_pooled = average-pool(x, p) with p in (12, 3, 1); backcast upsampled by repetition
forecast of return = z_hat * EWMA vol      -> a Gaussian N(z_hat * s, s^2) with the SAME s as every other model
```
Because the spread is shared, any CRPS difference between models is purely due to the mean `z_hat`.

## Pitfalls

- A benchmark chosen after seeing results is a different benchmark; here the change was declared before the new run.
- Seed variance can be as large as the difference between models.
- A model can rank assets (IC) without improving the probabilistic score.

## Try it

```bash
python -m experiments.stage33_frontier
```
