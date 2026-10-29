# cointegration_pairs

*Family: stat-arb*

## What it bets on

Rolling-cointegration pairs: trade a pair only while an Engle-Granger test on the last year says it is cointegrated

Trade only relationships that currently look stationary; pairs that fail the test are left alone.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `pairs` = (('EEM', 'EFA'), ('GLD', 'SLV'), ('IEF', 'TLT'), ('LQD', 'HYG'), ('SPY', 'QQQ'), ('IWM', 'SPY')), `window` = 252, `refit_every` = 21, `p_value` = 0.1, `z_window` = 63, `clip` = 3.0

## Run it

```bash
quant backtest --model cointegration_pairs --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.04, CAGR -0.2%, volatility 3.4%, max drawdown -10.6%, turnover 8.0 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey.md) for how to read it.

## What happened in this repository

Net of costs on the window the search used: Sharpe nan, CAGR 0.0%, volatility 0.0%, max drawdown 0.0%, turnover 0.0 times a year. This is one backtest among those in a search; read the search-aware tests in the Generation 5 report before treating any row as evidence.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Gatev, Goetzmann and Rouwenhorst (2006), Pairs trading: performance of a relative-value arbitrage rule, Review of Financial Studies 19(3)](https://doi.org/10.1093/rfs/hhj020)
- [Engle and Granger (1987), Co-integration and error correction: representation, estimation, and testing, Econometrica 55(2)](https://doi.org/10.2307/1913236)
- Vidyamurthy (2004), Pairs Trading: Quantitative Methods and Analysis, Wiley

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
