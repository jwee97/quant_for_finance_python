# sparse_basket

*Family: stat-arb*

## What it bets on

Sparse basket mean reversion: regress each asset on a LASSO-selected basket of the others and fade the cumulative residual

The few other assets that actually explain an ETF define its fair value; the residual from that sparse basket reverts.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `window` = 252, `refit_every` = 21, `alpha` = 0.05, `z_window` = 21

## Run it

```bash
quant backtest --model sparse_basket --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.28, CAGR -1.1%, volatility 3.7%, max drawdown -24.6%, turnover 23.1 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe -0.18 (-0.16 over its own, longer live window), CAGR -0.2%, volatility 1.0%, max drawdown -5.2%, turnover 1.7 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
