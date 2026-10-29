# ou_threshold_pairs

*Family: stat-arb*

## What it bets on

Pairs trading at the best levels for an Ornstein-Uhlenbeck spread: refit the spread's reversion by maximum likelihood and trade the levels that earn the most per day after costs (Bertram)

Most pairs rules enter at a round number of standard deviations. If the spread really is a mean-reverting (Ornstein-Uhlenbeck) process there is a best pair of levels, and it depends on how
fast the spread reverts and what a round trip costs: wide enough that each trade clears its cost, narrow enough that trades come often. This rule estimates the process on a rolling window,
solves for those levels (``probability.ou.optimal_thresholds``) and holds each pair until the spread reaches the opposite level (``exit_scale = 1``) or the mean (``exit_scale = 0``).
A pair trades only while the window's spread passes a Dickey-Fuller test (``df_critical``, the Engle-Granger value) and reverts with a half-life between ``min_half_life`` and
``max_half_life`` days; a spread that runs past ``stop_sd`` standard deviations is closed, and the pair stays out until the spread is back inside its entry level.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `pairs` = (('EEM', 'EFA'), ('GLD', 'SLV'), ('IEF', 'TLT'), ('LQD', 'HYG'), ('SPY', 'QQQ'), ('IWM', 'SPY')), `window` = 252, `refit_every` = 21, `cost_bps` = 5.0, `min_half_life` = 2.0, `max_half_life` = 40.0, `exit_scale` = 1.0, `stop_sd` = 4.0, `df_critical` = -3.4

## Run it

```bash
quant backtest --model ou_threshold_pairs --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe -0.16, CAGR -0.1%, volatility 0.7%, max drawdown -3.0%, turnover 1.5 times a year, deflated Sharpe probability 0.00 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Caveats

Needs pairs that are genuinely related; with 15 ETFs there are few. Trades both legs, so costs are doubled.

## Sources

Where the rule comes from. A source is not evidence that it works on this repository's data: the survey above and the application's own result pages say that.

- [Bertram (2010), Analytic solutions for optimal statistical arbitrage trading, Physica A 389(11)](https://doi.org/10.1016/j.physa.2010.01.045)
- [Avellaneda and Lee (2010), Statistical arbitrage in the US equities market, Quantitative Finance 10(7)](https://doi.org/10.1080/14697680903124632)

## Learn more

[Pairs, cointegration and statistical arbitrage](../techniques/pairs-and-statistical-arbitrage.md)
