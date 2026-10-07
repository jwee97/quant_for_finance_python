# deep_representation

*Family: machine learning*

## What it bets on

Autoencoder or contrastive encoder trained WITHOUT labels on volatility-normalised return windows, then a ridge from the embedding to the next 21-day return

Labels are scarce but windows are plentiful: learn what a year of returns typically looks like, then ask whether where a window sits in that learned space predicts the next month.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `kind` = 'autoencoder', `embed_dim` = 8, `hidden` = 64, `epochs` = 15, `ridge_alpha` = 100.0, `min_train` = 1260, `refit_every` = 252, `embargo` = 21, `seed` = 0

## Run it

```bash
quant backtest --model deep_representation --tearsheet
```

## Caveats

Walk-forward ridge; the signal-to-noise ratio of monthly returns is very low.

## Learn more

[Gradient boosting and representation learning on returns](../techniques/gradient-boosting-and-representation-learning.md)
