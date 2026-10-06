# chronos

*Family: machine learning*

## What it bets on

Chronos-Bolt foundation model used zero-shot on the volatility-normalised 252-day return window (optional dependency, weights from the Hugging Face hub)

A pre-trained time-series model needs no fitting, so there is nothing to leak. Its pre-training corpus overlaps the sample and may contain these prices,
so a good result here is weaker evidence than a good result from a model trained walk-forward.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `model` = 'amazon/chronos-bolt-small', `batch` = 256, `min_history` = 315

## Run it

```bash
quant backtest --model chronos --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Foundation models for time series (zero-shot Chronos)](../techniques/foundation-models.md)
