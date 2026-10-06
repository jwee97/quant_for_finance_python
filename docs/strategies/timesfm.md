# timesfm

*Family: machine learning*

## What it bets on

TimesFM 2.5 (200M) foundation model used zero-shot on the volatility-normalised 252-day return window (optional dependency, weights from the Hugging Face hub)

Same protocol and same caveat as ``chronos``: nothing is fitted, and the pre-training corpus may overlap the sample.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `model` = 'google/timesfm-2.5-200m-pytorch', `batch` = 64, `min_history` = 315

## Run it

```bash
quant backtest --model timesfm --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Foundation models for time series (zero-shot Chronos)](../techniques/foundation-models.md)
