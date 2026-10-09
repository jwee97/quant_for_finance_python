# macro_factor_model

*Family: macro*

## What it bets on

Macroeconomic factor model: each asset's sensitivity to changes in yields, spreads, volatility, inflation expectations and the dollar, priced by the premium per unit of sensitivity paid in earlier months

Chen, Roll and Ross: returns move with unexpected changes in a few macroeconomic variables, and assets that are more exposed to a risk that is priced earn more. Here the factors are daily changes of the
macro series you name (default: the 10-year yield, the term spread, the credit spread, the VIX, breakeven inflation and the dollar), made unexpected by an AR(1) filter if ``innovations``, and
standardised by their own volatility. Each asset's loadings on all of them are estimated by multivariate regression over the trailing ``window`` days. At each month-end the premium per unit of loading is the average,
over the last ``premium_window`` months whose returns are known, of the cross-sectional regression slope of the next month's return on the loadings (Fama and MacBeth), and the score is loadings times
premia. Needs the macro series in the bundle; uses what is there, and at least two.

## Inputs

- Macro or alternative series required: DGS10, T10Y3M, BAA10Y, VIX, T10YIE, DTWEXBGS (any 2 of them)
- Parameters: `series` = 'DGS10,T10Y3M,BAA10Y,VIX,T10YIE,DTWEXBGS', `window` = 504, `premium_window` = 60, `min_months` = 24, `innovations` = True

## Run it

```bash
quant backtest --model macro_factor_model --tearsheet
```

## Caveats

Macro series enter with their publication lags; revisions are not modelled.

## Learn more

[Factor models and machine learning: statistical APT, macroeconomic factors, cross-sectional characteristics, LASSO to neural networks](../techniques/factor-models-and-machine-learning.md)
