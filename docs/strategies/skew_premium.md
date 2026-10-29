# skew_premium

*Family: cross-sectional*

## What it bets on

Realised-skewness premium (Amaya et al., Fernandez-Perez et al.): favour the assets whose recent daily returns are the most negatively skewed, avoid the most positively skewed

Investors overpay for lottery-like positive skewness, so assets with positively skewed returns are expensive and those with negative skewness earn a premium for it.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252

## Run it

```bash
quant backtest --model skew_premium --tearsheet
```
This rule declares a weekly rebalance; pass `execution: {rebalance: monthly}` in a spec to override it.


## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.00, CAGR -0.1%, volatility 5.4%, max drawdown -19.9%, turnover 10.8 times a year, deflated Sharpe probability 0.01 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Caveats

Ranks 15 ETFs across asset classes, so the ranking is partly a ranking of asset classes.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Amaya, Christoffersen, Jacobs and Vasquez (2015), Does realized skewness predict the cross-section of equity returns?, Journal of Financial Economics 118(1)](https://doi.org/10.1016/j.jfineco.2015.02.009)
- [Fernandez-Perez, Frijns, Fuertes and Miffre (2018), The skewness of commodity futures returns, Journal of Banking & Finance 86](https://doi.org/10.1016/j.jbankfin.2017.06.015)

## Learn more

[Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance](../techniques/book-rules-systematic-trading.md)
