# deep_window

*Family: machine learning*

## What it bets on

A small neural network (patch transformer, MLP-mixer, N-BEATS, N-HiTS or TimeMixer-style) on the volatility-normalised 252-day return window, refit yearly

If a flexible sequence model finds structure in the last year of returns that a ridge on twelve features cannot, it should beat the ridge out of sample.
Stage 27 and 33 found it did not; the plug-in makes that test repeatable on any bundle.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `kind` = 'nbeats', `min_train` = 1260, `refit_every` = 252, `embargo` = 21, `max_epochs` = 15, `patience` = 3, `seeds` = (0,), `d_model` = 16

## Run it

```bash
quant backtest --model deep_window --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Deep learning for time series: transformers, mixers, N-BEATS, N-HiTS](../techniques/deep-time-series-models.md)
