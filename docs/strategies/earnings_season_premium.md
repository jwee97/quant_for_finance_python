# earnings_season_premium

*Family: event-driven*

## What it bets on

Seasonal earnings announcement premium: favour stocks expected to announce earnings next month (they announced in the same month a year ago), with the premium learned from earlier months

Stocks earn more in the months in which they announce earnings (Frazzini and Lamont 2007, Barber, De George, Lehavy and Trueman 2013), and the months are predictable: companies keep their
reporting calendar. The model flags, at each month-end, the stocks that announced in the next calendar month a year earlier, and forecasts with the average premium the flag has earned in earlier
months (it is learned, so a flag that has not paid is switched off or reversed). Announcements come from ``data/user/earnings_dates.csv`` (``date, ticker``) if it exists, else are inferred from spikes in trading volume,
which needs a bundle with volume and works only for companies, not funds.

## Inputs

- Macro or alternative series required: none (prices only)
- Parameters: `source` = 'auto', `path` = 'earnings_dates.csv', `window` = 0, `min_obs` = 24, `volume_multiple` = 3.0, `shrink` = 0.0

## Run it

```bash
quant backtest --model earnings_season_premium --tearsheet
```

## In the strategy survey

Default parameters on the 15 ETFs, net of costs, traded as written: net Sharpe +0.55, CAGR 2.9%, volatility 5.5%, max drawdown -17.3%, turnover 13.9 times a year, deflated Sharpe probability 0.35 counting every strategy in the survey as a trial. One run, not a test: see [the survey](../strategy_survey_3.md) for how to read it.

## Learn more

[Factor timing: the calendar, the macroeconomy, the market's state and the earnings season](../techniques/factor-timing.md)
