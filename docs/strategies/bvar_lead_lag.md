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

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.10, CAGR 0.4%, volatility 6.0%, max drawdown -17.4%, turnover 18.8 times a year, deflated Sharpe probability 0.02 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Econometric strategies: Kalman trend, GARCH and EVT sizing, and a shrunk VAR](../techniques/econometric-strategies.md)
