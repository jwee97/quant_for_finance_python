# characteristic_regression

*Family: cross-sectional*

## What it bets on

Cross-sectional characteristics model: regress each month's returns on the previous month-end momentum, reversal, volatility, beta and distance from the high; forecast with the trailing average slopes

Fama and MacBeth's design as a forecaster: every month, regress the following month's returns of all assets on their characteristics at the month-end (momentum, reversal, low volatility, low beta, no
extreme day, closeness to the one-year high), and forecast with the average slope of the last ``window`` months whose returns are known times today's characteristics. A characteristic that has not paid
gets a slope near zero and so little weight. The same machinery the fundamental models use, with characteristics that need only prices.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `characteristics` = 'mom,rev,lowvol,lowbeta,nomax,high', `window` = 60, `min_obs` = 24

## Run it

```bash
quant backtest --model characteristic_regression --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.32, CAGR -2.3%, volatility 6.7%, max drawdown -37.9%, turnover 15.3 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Learn more

[Factor models and machine learning: statistical APT, macroeconomic factors, cross-sectional characteristics, LASSO to neural networks](../techniques/factor-models-and-machine-learning.md)
