# pca_residual

*Family: stat-arb*

## What it bets on

PCA residual trading: fade each asset's deviation from what three principal components explain

After removing the common factors, what is left in an asset's return is noise that reverts.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `components` = 3, `window` = 252, `zscore_window` = 21

## Run it

```bash
quant backtest --model pca_residual --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.33, CAGR -1.1%, volatility 3.3%, max drawdown -19.0%, turnover 22.7 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.33 (-0.45 over its own, longer live window), CAGR -1.5%, volatility 4.4%, max drawdown -27.2%, turnover 21.6 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Avellaneda and Lee (2010), Statistical arbitrage in the US equities market, Quantitative Finance 10(7)](https://doi.org/10.1080/14697680903124632)

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
