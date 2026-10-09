# policy_portfolio

*Family: allocation*

## What it bets on

Policy portfolio with rebalancing discipline: strategic weights restored on a calendar (monthly to annual) and/or whenever an asset drifts outside a tolerance band

Most of a strategic allocation's behaviour comes from how it is rebalanced. Between rebalances the weights drift with returns (winners grow); restoring them every month sells winners and
buys losers (a mild contrarian bet that earns when assets mean-revert and loses when they trend), while never rebalancing lets the riskiest asset take over. A tolerance band restores the policy
only when it is needed (Leland 1999: under proportional costs the optimal rule is a no-trade region around the target). ``preset`` picks a mix from ``model_portfolio`` (60_40, permanent,
all_weather, bogleheads) or ``targets`` gives your own ``{ticker: weight}``. ``calendar`` is how often the policy is restored whatever the drift (``never`` for the band alone); ``band`` is the
drift in weight (0.05 = five points) that restores it. Decisions are taken on the last trading day of each month, and the weights stated when nothing is restored are the drifted ones, so the
engine's own drift and trading cost agree with the rule. Run it with the engine's monthly rebalance.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `preset` = '60_40', `targets` = None, `calendar` = 'quarterly', `band` = 0.05

## Run it

```bash
quant backtest --model policy_portfolio --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.83, CAGR 9.1%, volatility 11.4%, max drawdown -26.6%, turnover 1.4 times a year, deflated Sharpe probability 0.85 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_2.md) for how to read it.

## Learn more

[Portfolio rebalancing styles: policy bands, month-end flows, flight to quality, market outlook](../techniques/portfolio-rebalancing-styles.md)
