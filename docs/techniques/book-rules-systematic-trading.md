---
title: "Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance"
slug: book-rules-systematic-trading
difficulty: 2
chapter: Ch. 20
prerequisites: [momentum, plugin-framework, trend-following-toolkit]
stages: []
files: [src/strategies/systematic.py, src/references.py]
figures: []
tests: [tests/test_systematic_exact.py, tests/test_references.py]
models: [carver_ewmac, clenow_momentum, tstat_momentum, variance_ratio_switch, connors_rsi, double_seven, skew_premium, cppi]
---

# Rules from the practitioner books: Carver, Clenow, Connors, Lo-MacKinlay, portfolio insurance

## In one sentence

Eight rules that practitioners and textbooks made popular and that the strategy library did not yet contain, each written the way its source defines it, tested against an independent calculation, and shown with the books and papers it comes from.

## The idea

**Carver's EWMAC forecast** (`carver_ewmac`; Carver 2015, 2019, 2023). A trend rule is the gap between a fast and a slow exponentially weighted average of the price, divided by the volatility of daily price changes. Six speeds (fast spans 2, 4, 8, 16, 32 and 64 days, slow span four times as long) are each multiplied by a *forecast scalar* that makes their average absolute value 10, capped at plus or minus 20, averaged, and multiplied by a *forecast diversification multiplier*, `1 / sqrt(w' C w)` with `w = 1/6` and the correlations `C` of the six forecasts floored at zero, because six imperfectly correlated forecasts average to something smaller than each. The scalars here are **estimated from the data's own history** (an expanding average over all assets), which is how Carver says to set them; the repository does not copy a table of constants it could not check against the book.

**Clenow's momentum rank** (`clenow_momentum`; Clenow 2015). The annualised slope of the exponential regression of the last 90 days of log prices (`exp(slope x 250) - 1`) times the R-squared of the fit, so a steep and straight climb outranks a steep and jagged one. A name is eligible only above its 100-day average and when no daily move in the window exceeded 15%.

**t-statistic trend** (`tstat_momentum`; Baltas and Kosowski 2013, 2020). In place of the sign of the year's return, the t-statistic of a year of daily log returns, capped at one. A steady drift gets the full position, a lucky jump a partial one, and the position moves gradually.

**The variance-ratio switch** (`variance_ratio_switch`; Lo and MacKinlay 1988, 1989; Campbell, Lo and MacKinlay 1997; Chan 2013). Over a trailing year, the variance of q-day returns divided by q times the variance of daily ones is one for a random walk, above one when moves persist and below one when they reverse. The heteroscedasticity-robust statistic `z*(q)` says whether the departure is real. The rule follows the last q-day move where `z*` is above 1.645, fades it where it is below -1.645, and holds nothing in between.

**ConnorsRSI and Double 7s** (`connors_rsi`, `double_seven`; Connors and Alvarez 2009). Two short-term pullback rules for an up-trending market. ConnorsRSI averages the RSI(3) of the price, the RSI(2) of the up or down streak and the percent rank of the day's return over the last 100; it buys below 10 above the 200-day average and sells above 70. Double 7s buys a 7-day closing low above the 200-day average and sells a 7-day closing high.

**The skewness premium** (`skew_premium`; Amaya et al. 2015; Fernandez-Perez et al. 2018). Rank by the skewness of the last year of daily returns and favour the most negative. Investors overpay for positive skewness, so lottery-like assets are expensive.

**Portfolio insurance** (`cppi`; Black and Jones 1987, Perold and Sharpe 1988, Black and Perold 1992; with `ratchet=True`, the TIPP of Estep and Kritzman 1988). Hold a multiple of the *cushion* (account value less a floor) in the risky asset and the rest in cash; the floor is 85% of the starting value by default. A fall shrinks the cushion and so the exposure, which is what stops the account falling through the floor, and what makes the strategy buy high and sell low in a market that whipsaws.

## Why it matters

These are the rules a reader of the books will ask for by name, and each hides a detail that decides whether an implementation is the rule or a lookalike: Carver's scalar and multiplier, Clenow's gap and trend filters, the robust form of the variance-ratio statistic, whether the percent rank in ConnorsRSI counts today. Naming the rule and pinning its definition with a test is what lets a result be compared with the source.

## How this repo uses it

All eight are registered strategies (`quant list models`) with the sources in the strategy picker and on each strategy card. Per-asset rules (`carver_ewmac`, `tstat_momentum`, `variance_ratio_switch`, `connors_rsi`, `double_seven`, `cppi`) run as independent sleeves; `clenow_momentum` and `skew_premium` rank the universe. The extra rule statistics are exposed so that they can be charted: `src.strategies.systematic.variance_ratio_stats`, `ewmac_raw`, `connors_rsi`, `cppi_exposure`. Every score on a date uses data through that date only (the exact tests, and the platform's look-ahead check, replace the future with noise and require the same output).

## What we found

One run of the strategy survey, the 15 ETFs, 2006 to 2026, default parameters, net of costs, no tuning ([part three](../strategy_survey_3.md)): `cppi` net Sharpe +0.70 against +0.67 for equal weight over the same days; `double_seven` +0.49 against +0.86; `connors_rsi` +0.25 against +0.67; `tstat_momentum` +0.22 against +0.89; `skew_premium` 0.00 against +0.89; `carver_ewmac` -0.05 against +0.76; `clenow_momentum` -0.19 against +0.66; `variance_ratio_switch` -0.37 against +0.89. Only the insurance rule matched its benchmark, and its deflated Sharpe probability is 0.67, well short of 0.95. That is a statement about 15 liquid ETFs in a period of strong equity returns, not about the books: Clenow's rule was written for hundreds of stocks, Carver's for dozens of futures markets, and Connors' for equity indices.

On the same ETFs the fastest Carver speeds cost the most: dropping the 2-day speed takes the turnover of the sleeves from 25.9 times a year to 18.6, dropping the 4-day as well to 12.9, and keeping only 16, 32 and 64 to 9.3. The `fast_spans` parameter is there for that trade-off; the Sharpe ratios of those variants are not reported because choosing among them on the same data would be the tuning the survey avoids.

## Going deeper

```
EWMAC raw:   (EMA_f - EMA_4f) / sd_35(daily price change)
scalar_k:    10 / mean |raw_k|            (expanding, all assets, through the date)
forecast:    clip( mean_k clip(scalar_k raw_k, +-20) x FDM , +-20 );  score = forecast / 10
FDM:         1 / sqrt( w' C w ),  w = 1/K,  C = correlations of the capped forecasts, floored at 0, FDM <= 2.5
Clenow:      (exp(250 slope) - 1) R^2,  slope and R^2 of OLS of log P on the day index over 90 days
t-stat:      mean(r) / ( sd(r) / sqrt(252) ),  r = daily log return;  score = clip(t, -1, 1)
VR(q):       sigma_c^2(q) / sigma_a^2,  m = q (n - q + 1)(1 - q / n)
z*(q):       sqrt(n) (VR - 1) / sqrt(theta),  theta = 4 sum_j (1 - j/q)^2 delta_j,  delta_j = n sum d_t^2 d_(t-j)^2 / (sum d_t^2)^2
ConnorsRSI:  ( RSI_3(P) + RSI_2(streak) + PercentRank_100(P_t / P_(t-1) - 1) ) / 3
CPPI:        e_t = clip( m (V_t - F_t) / V_t , 0, max );  V_(t+1) = V_t (1 + e_t r_(t+1) + (1 - e_t) rf)
```

## Pitfalls

- The sources describe rules for the markets they were written for; a different universe is a different test, and a negative row here does not refute them.
- A statistic with a threshold (the variance-ratio switch, ConnorsRSI) is a rule with a free parameter. The defaults are the sources', not tuned.
- Estimated scalars need history: `carver_ewmac` gives no forecast until `min_history` dates have every speed, by design.
- CPPI cannot lose more than the floor unless the asset gaps by more than one over the multiple in a single day, and it stays in cash after a breach unless the ratchet or a new account resets it.
- Several of these rules are the same idea (trend) measured differently: counting them as independent confirmation overstates the evidence.

## Try it

```bash
quant backtest --model carver_ewmac --allocator sleeves --tearsheet
quant backtest --model variance_ratio_switch --allocator sleeves
quant backtest --model cppi --allocator sleeves
```

## Sources

- Carver (2015), *Systematic Trading*, Harriman House; Carver (2019), *Leveraged Trading*, Harriman House; Carver (2023), *Advanced Futures Trading Strategies*, Harriman House
- Clenow (2015), *Stocks on the Move*, Equilateral Publishing
- Baltas and Kosowski (2020), [Demystifying time-series momentum strategies](https://doi.org/10.1002/9781119599364.ch3), in *Market Momentum*, Wiley; Baltas and Kosowski (2013), [Momentum strategies in futures markets and trend-following funds](https://doi.org/10.2139/ssrn.1968996), SSRN 1968996
- Lo and MacKinlay (1988), [Stock market prices do not follow random walks](https://doi.org/10.1093/rfs/1.1.41), Review of Financial Studies; Lo and MacKinlay (1989), [The size and power of the variance ratio test in finite samples](https://doi.org/10.1016/0304-4076(89)90083-3), Journal of Econometrics; Campbell, Lo and MacKinlay (1997), *The Econometrics of Financial Markets*, Princeton University Press; Chan (2013), *Algorithmic Trading*, Wiley
- Connors and Alvarez (2009), *Short Term Trading Strategies That Work*, TradingMarkets Publishing
- Amaya, Christoffersen, Jacobs and Vasquez (2015), [Does realized skewness predict the cross-section of equity returns?](https://doi.org/10.1016/j.jfineco.2015.02.009), Journal of Financial Economics; Fernandez-Perez, Frijns, Fuertes and Miffre (2018), [The skewness of commodity futures returns](https://doi.org/10.1016/j.jbankfin.2017.06.015), Journal of Banking & Finance
- Black and Jones (1987), [Simplifying portfolio insurance](https://doi.org/10.3905/jpm.1987.409131); Perold and Sharpe (1988), [Dynamic strategies for asset allocation](https://doi.org/10.2469/faj.v44.n1.16); Black and Perold (1992), [Theory of constant proportion portfolio insurance](https://doi.org/10.1016/0165-1889(92)90043-e); Estep and Kritzman (1988), [TIPP: insurance without complexity](https://doi.org/10.3905/jpm.1988.409172)
