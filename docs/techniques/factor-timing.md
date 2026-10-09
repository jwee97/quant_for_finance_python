---
title: "Factor timing: the calendar, the macroeconomy, the market's state and the earnings season"
slug: factor-timing
difficulty: 3
chapter: Platform
prerequisites: [cross-sectional-factors, seasonality-and-calendar-effects, macro-and-alternative-data, factor-models-and-machine-learning]
stages: []
files: [src/strategies/factor_timing.py, src/equity/synthetic.py, src/framework/data.py, experiments/timing_world.py]
figures: []
tests: [tests/test_factor_timing.py]
models: [calendar_factor_timing, macro_factor_timing, earnings_season_premium]
---

# Factor timing: the calendar, the macroeconomy, the market's state and the earnings season

## In one sentence

A factor does not pay the same every month, so a timing model forecasts what a factor will pay from the state of the world (January or not, the month within the quarter, whether the Fed is tightening, how fast money, output or prices are growing, whether the market is up, whether a stock is about to report earnings), learning from earlier months only, and an asset's forecast is its exposure to each factor times the factor's forecast premium.

## The idea

**A premium forecast.** Each month, regress the next month's returns on a factor's exposures across the assets (in standard deviations); the slope is the *premium* the factor paid that month. A plain factor forecast uses the average of the premiums so far. A *timing* model uses, instead, the average of the premiums earned in earlier months *in the same state as the coming month*, moved toward the overall average by `n / (n + shrink)` when the state has been seen only `n` times. The shrinkage is what stops a state seen twice from flipping a factor, and what makes a state that does not matter cost almost nothing. The forecast can switch a factor off, or reverse it: no sign is built in, so the finding that momentum loses in January is something the model finds in the data or does not. Only months whose returns are known enter (the month `s` premium is known at `s + 1`).

**Calendar states** (`calendar_factor_timing`). `january` (the month the return is earned in is January or not), `month` (twelve states: most to learn), `quarter` (the month within the calendar quarter, which is how this repository reads "quarterly horizons": the first, second or third month, the last being the quarter-end month) and `halloween` (November to April against May to October). The calendar is known in advance, so a rule that depends on it is not look-ahead, but it is among the most data-mined ideas in finance.

**Macroeconomic states** (`macro_factor_timing`). `fed`: the six-month change in the federal funds rate, easing, flat (within a band of 0.25 points) or tightening. `m1`, `gdp`, `inflation`, `ppi`: the year-on-year growth of M1, real GDP, CPI or producer prices, at or above the median of its own history so far, or below. `market`: the equal-weighted return of the universe over the last twelve months, positive (an up market) or not; this is the state behind the finding that momentum earns after up markets and crashes after down markets (Cooper, Gutierrez and Hameed 2004), and it needs no series. Growth against the median of everything *so far*, never against the full sample, keeps the state free of look-ahead.

**The earnings announcement season** (`earnings_season_premium`). Stocks earn more in the months in which they announce earnings (Frazzini and Lamont 2007; Barber, De George, Lehavy and Trueman 2013), and the months are predictable because companies keep their reporting calendar. The model flags, at each month-end, the stocks that announced in the next calendar month a year earlier, and forecasts with the average premium that flag has earned in earlier months.

## Why it matters

Factors are the building blocks of equity strategies, and the standing question about each is whether to hold it always. A timing rule is a bet that a factor's premium varies in a way the state predicts; it is easy to find in-sample (twelve calendar months, six macro series, a handful of factors: hundreds of ways to be right by luck) and hard to keep out of sample, so the point of a machine for it is that it learns from the past only, shrinks, and can be compared with the plain factor on the same data.

## How this repo uses it

The factors are the six price-based characteristics of the [factor model guide](factor-models-and-machine-learning.md) (`mom`, `rev`, `lowvol`, `lowbeta`, `nomax`, `high`; one, a list, or `all`). Each is timed separately and the timed forecasts are summed. All three models are monthly decisions on data through the month-end, checked by the generic causality test and by tests that change the data after a date and require every earlier score to be identical. `model.premia(bundle)` shows the premium forecast of every factor at every month-end, which is the thing to look at before trusting a score.

The macroeconomic series come from the platform's point-in-time macro panel with their publication lags. `DFF` (the federal funds rate) and `CPIAUCNS` (CPI, not seasonally adjusted, never revised) were already there; `M1SL`, `GDPC1` and `PPIACO` are new (declared in `src/framework/data.py`, not in the configuration, which is part of the pinned identity of earlier results) and their raw files are committed in `data/raw/framework/`. M1 and GDP are seasonally adjusted and revised after publication, and FRED serves the latest versions, so the lags (45 days for M1 and producer prices, 150 days for GDP, the second estimate) reduce that look-ahead without removing it; a state built from GDP is the least trustworthy of the six. M1 has a definitional break in 2020 (savings deposits were added), which makes its growth rate meaningless for a year. `market` needs no series.

`earnings_season_premium` reads `data/user/earnings_dates.csv` (`date, ticker`: one row per announcement, the date being the first day the news could be traded on). Without the file it infers announcements as spikes in trading volume (three times the median of the previous sixty days and the largest within 45 days either side), which needs a bundle with volume and works for companies, not funds. The rank correlation of a score with returns depends only on the *sign* of the premium forecast, never its size; the size matters to how the pipeline sizes positions, which is why the experiment also reports what acting on the forecast earns.

## What we found

All simulated unless stated (`python -m experiments.timing_world`, about a minute), so these are checks of what each model finds and what a wrong choice costs, not evidence that timing works in a market.

*The calendar.* Four worlds of 50 stocks for 16 years in which momentum earns 0.4% a month per standard deviation, except in January, when it loses 2%. Mean monthly rank IC, the IC in the Januaries, and what acting on the premium forecast earns (forecast times the premium actually paid, in basis points a month per standard deviation squared):

| model | IC | IC in Januaries | forecast x premium |
|---|---|---|---|
| plain factor (no state) | 0.024 | -0.230 | 0.069 |
| state: January | 0.074 | +0.263 | 0.281 |
| state: month of the year | 0.067 | +0.263 | 0.232 |
| state: month within the quarter | 0.032 | +0.030 | 0.119 |
| state: November-April | 0.024 | -0.057 | 0.080 |

The January state turns a loss in the Januaries into a gain and quadruples what the forecast earns. The twelve-month state finds the same thing with a little more noise. States that overlap January only in part find only a part of it: January is the first month of a quarter, so the quarter state gets 0.119, and it is in November-April, which gets 0.080, against 0.281 for the state that is the truth. A state that does not matter is no worse than the plain factor, because the shrinkage pulls it back. In worlds with the same premium every month all five agree (IC 0.053 for every state but the twelve-month one, 0.055; forecast times premium 0.199 to 0.209, against 0.209 for the plain factor): timing a factor that does not vary costs almost nothing.

*Macroeconomic states.* Momentum is paid 1.2% a month per standard deviation after up markets and loses the same after down markets (state: the market), or pays when policy tightens and loses when it eases (state: the Fed), four worlds each. The plain factor averages a premium that changes sign: IC 0.023 (t 1.7) in the market world and -0.009 (t -0.7) in the Fed world. The timed models find it: 0.148 (t 13.8) and 0.148 (t 12.0), and what acting on the forecast earns rises from 0.05 and -0.01 to 1.06 and 1.13.

*The earnings season.* Companies that report in the same month of every quarter and earn 2% more in the months they report. With dates from a file the rank IC is 0.132 (t 8.3); with announcements inferred from volume spikes it is 0.127 (t 8.0), the spikes having found 94% of the announcements with almost no false ones. In the control world with no premium the IC is -0.006 and -0.007 (t -0.3 and -0.4). (The effect is large in the simulation because a binary flag on a third of the stocks needs a premium of that size to show in a rank correlation of 60 stocks.)

*On the 15 ETFs, 2006 to 2026.* The machinery runs on real prices and the point-in-time macro series, and finds nothing. The six price factors combined have an IC of 0.095 (t 2.8) untimed; with the January state 0.093, the month of the year 0.044, the month within the quarter 0.088, November-April 0.028; and with the macro states (market, Fed, M1, GDP, inflation, producer prices) the IC is 0.058 to 0.076 against 0.086 to 0.113 for the untimed factors over the same months. What acting on the forecast earns is between -0.02 and 0.05 basis points a month in every case, against 0.01 to 0.02 for the plain factors. Fifteen assets, mixing stocks, bonds and commodities, and twenty years have almost no power to detect a timing effect, and the states that carry real content (the month, November-April) did worse than none; this is a sample too small to say anything, and it does not support the idea.

## Going deeper

```
premium_k(s)   = slope of  r(i, s+1)  on the standardised exposure z(k, i, s) across assets i        (the factor's return in month s+1, per standard deviation)
forecast_k(t)  = m + n/(n + shrink) * (m_state - m)     m = mean of premium_k(s), s < t;   m_state = the same over the s < t in the state of t;   n = how many
score(i, t)    = sum_k  forecast_k(t) * z(k, i, t)
states         january: month(t+1) = 1;  month: month(t+1);  quarter: (month(t+1) - 1) mod 3;  halloween: month(t+1) in Nov..Apr
               fed: DFF(t) - DFF(t - 6 months)  >  band, < -band, else flat;   m1, gdp, inflation, ppi: yoy growth(t) >= median of yoy growth(s <= t);   market: prod(1 + r_ew) over 12 months > 1
earnings       flag(i, t) = 1 if i announced in month(t+1) - 12 months;   z = standardise(flag);   forecast = mean premium of the flag so far
```

## Pitfalls

- Timing is where backtests lie most easily. With twelve calendar months, six macro series and six factors there are hundreds of rules; one will have worked. Count what you tried as trials, fix the states before looking, and compare with the plain factor on the same data, as the experiment does.
- A state that carries no information costs a little (the twelve-month state's forecast times premium was 0.20 against 0.21 where nothing varied). A state seen only a few times is shrunk almost entirely to the overall mean; that is the design, and it means the model needs years of history before any calendar state speaks.
- The macro series are revised. M1 and GDP are seasonally adjusted and FRED serves the latest vintage; lags reduce the look-ahead and cannot remove it. Treat a GDP or M1 result as less reliable than one from the funds rate, CPI (never revised) or the market's own state.
- "Up market" and "down market" are defined by a trailing return, so the state changes just after the turning points it is meant to detect; the premium it forecasts is the premium after the regime has been visible for some months.
- Announcement months move: companies change reporting dates, and the volume proxy mistakes other volume events (index changes, news) for announcements. A file of true dates is better; the proxy is for when there is none.
- The IC of a score depends only on the sign of the premium forecast; judge a timing model also by what acting on the forecast earns (the experiment's last column), which depends on its size.
- Cross-sectional timing is not market timing. These models forecast which assets will do better than others, not whether the market will rise.

## Try it

```bash
python -m experiments.timing_world
quant backtest --model calendar_factor_timing --param factor=mom --param state=january --tearsheet
quant backtest --model calendar_factor_timing --param factor=all --param state=quarter
quant backtest --model macro_factor_timing --param factor=mom --param state=market
quant backtest --model macro_factor_timing --param factor=all --param state=fed
quant backtest --model earnings_season_premium --param source=file --param path=earnings_dates.csv
```
