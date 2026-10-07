# bvar_lead_lag

*Family: cross-sectional*

## What it bets on

Cross-asset lead-lag: a Minnesota-shrinkage VAR(1) on monthly returns forecasts each asset's next month from every asset's last month

Some assets move first and others follow (credit before equities, bonds before the dollar); a heavily shrunk VAR can pick that up without overfitting.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `lags` = 1, `tightness` = 0.1, `min_months` = 60, `refit_every` = 3

## Run it

```bash
quant backtest --model bvar_lead_lag --tearsheet
```

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.
