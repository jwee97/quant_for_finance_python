# online_ridge

*Family: machine learning*

## What it bets on

Monthly-updated linear forecast of the next 21-day return (recursive ridge, NLMS or Kalman coefficients) that follows a drifting signal

If the link between price features and the next return drifts, a model that forgets old months adapts faster than one refit yearly.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `updater` = 'ridge', `forgetting` = 0.97, `ridge` = 50.0, `step` = 0.05, `process_to_observation` = 0.001, `min_updates` = 36, `features` = 'price'

## Run it

```bash
quant backtest --model online_ridge --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Online learning: updating as data arrives](../techniques/online-learning.md)
