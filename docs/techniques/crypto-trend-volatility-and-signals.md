---
title: "Crypto: trend rules, the Mayer multiple, the cone of volatility and what every rule says"
slug: crypto-trend-volatility-and-signals
difficulty: 3
chapter: Platform
prerequisites: [crypto-carry, crypto-funding-basis-and-venues, futures-signals-factors-and-curve-shape]
stages: []
files: [src/assets/crypto_analytics.py, src/assets/strategy_runs.py, src/strategies/trend.py, src/webapp/asset_api.py, src/webapp/static/crypto.js, src/references.py]
figures: []
tests: [tests/test_crypto_views_exact.py, tests/test_webapp_assets_ml.py, tests/test_webapp_workspaces_browser.py]
models: [mayer_multiple]
---

# Crypto: trend rules, the Mayer multiple, the cone of volatility and what every rule says

## In one sentence

The Crypto tab now runs fifteen strategies in four groups with their own settings, adds a rule that holds a coin until its price is stretched over its 200-day average (the Mayer multiple), draws the price against its own history (the multiple, the cone of its volatility, its drawdowns, its ratio and correlation with the other coin), and shows what every rule says about BTC and ETH today; on the platform's Deribit data none of the thirteen spot rules beat holding the two coins equally over the same dates, and only the funding-carry rule, which is market-neutral, had a high Sharpe ratio.

## The idea

**Trend in crypto.** Liu and Tsyvinski (2021) and Liu, Tsyvinski and Wu (2022) found that a coin's recent return, over a week to a few weeks, predicts its next returns and that momentum is one of the three factors that price the cross-section of coins; Gerritsen, Bouri, Ramezanifar and Roubaud (2020) and Detzel, Liu, Strauss, Zhou and Zhu (2021) tested moving-average rules on Bitcoin and found that they beat buy and hold in their samples (Brock, Lakonishok and LeBaron 1992 had done the same for stocks). The tab runs the trend rules of the Backtest tab on the platform's two coins: Carver's six-speed crossover, the t-statistic of a year's returns, the three-horizon momentum, the moving-average ensemble, Kaufman's adaptive average, Wilder's directional movement and the Donchian breakout.

**The Mayer multiple.** The price over its 200-day average (Trace Mayer popularised it for Bitcoin). A multiple above about 2.4 has been called overheated and below 1 cheap; those thresholds come from practice and no paper tests them. `mayer_multiple` holds the coin (score 1) while the multiple is between a floor and a cap and sits out otherwise: with the default floor of 0 it is buy and hold with an exit from euphoria, with a floor of 1 it also needs the price to be above its average, a trend filter with an exit.

**The cone of volatility.** Realised volatility over the last 7, 14, 30, 60 and 90 days against every window of the same length in the coin's history, with the percentiles of that history (Burghardt and Lane 1990 draw the same cone for options): is volatility high or low for this coin, and is the short window above or below the long one?

**Against another coin.** The ratio of one coin to the other and its z-score against the last 60 days, and the rolling correlation of their daily returns (90 days): how stretched the pair is, and how much diversification the two give each other. Makarov and Schoar (2020) and Griffin and Shams (2020) document how prices and flows differ across venues and coins, which is why the page does not pretend that one price is the market.

**What every rule says today.** As on the Futures tab: each rule's score on the last day over the average size of its own absolute score in the last two years, with the funding and basis rules read on the coin's long-spot, short-perpetual index.

## Why it matters

A coin's price has few drivers and many opinions. The multiple, the cone and the pair say where the price stands against its own history in one picture each, and the backtest says what a rule would have done, after fees and slippage, against the plain alternative of holding.

## How this repo uses it

`src/assets/crypto_analytics.py` has `volatility_cone`, `long_average_view`, `relative_view`, `drawdown_view` and `technical_view` (served on the overview), the catalogue `CRYPTO_STRATEGIES` (each with a group, the strategy behind it and its defaults), `run_crypto_strategy` (which accepts the strategy's own `settings`) and `run_crypto_signals`; the board itself is `signals_board` in `src/assets/strategy_runs.py`, shared with the Futures tab. `mayer_multiple` is in `src/strategies/trend.py`. The result of every run carries the risk measures, the monthly returns, the distribution of days and the correlation of the assets (`src/risk/performance.py`). The routes are `GET /api/crypto/overview`, `POST /api/crypto/strategy/backtest` (with `params`) and `POST /api/crypto/signals`.

## What we found

The platform's data are Deribit's BTC and ETH, daily, 2019 to 2026, sampled on weekdays; the tab backtests only on them. Each strategy was run once with its default settings, weekly rebalancing, 5 basis points of fees and 2 of slippage each way, and the volatility target of the platform. The strategies start at different dates because each needs its own history before it trades (`from` below); the comparison column is equal weight in BTC and ETH, rebalanced daily, over the same dates.

| Strategy | From | Net Sharpe | Equal-weight BTC and ETH, same dates | Return a year | Worst fall | Deflated Sharpe |
|−−−|−−−|−−−:|−−−:|−−−:|−−−:|−−−:|
| Spot momentum | 2021−10 | −0.03 | +0.33 | −1.0% | −23% | 0.47 |
| Funding carry | 2021−08 | +3.72 | +0.38 | +3.2% | −1% | 1.00 |
| Basis trade | 2021−10 | −1.12 | +0.35 | −1.0% | −6% | 0.01 |
| Volatility breakout | 2021−08 | +0.09 | +0.38 | +0.3% | −22% | 0.58 |
| Stablecoin flow | 2021−12 | +0.02 | +0.35 | −0.5% | −27% | 0.51 |
| Regime-filtered trend | 2021−10 | +0.14 | +0.29 | +1.0% | −15% | 0.63 |
| Multi-speed trend (Carver) | 2024−04 | +0.13 | +0.30 | −5.4% | −41% | 0.58 |
| Trend by t−statistic (Baltas and Kosowski) | 2022−05 | −0.04 | +0.58 | −9.2% | −58% | 0.47 |
| Multi-horizon trend (Hurst, Ooi and Pedersen) | 2022−05 | +0.20 | +0.58 | +0.9% | −51% | 0.67 |
| Moving-average ensemble (Baz et al.) | 2022−09 | −0.18 | +0.80 | −9.0% | −42% | 0.36 |
| Adaptive trend (Kaufman) | 2021−08 | +0.11 | +0.38 | −0.2% | −56% | 0.60 |
| Directional movement (Wilder) | 2021−08 | +0.33 | +0.38 | +5.8% | −37% | 0.78 |
| Breakout (Donchian) | 2021−08 | +0.09 | +0.38 | −5.0% | −68% | 0.58 |
| Mayer multiple | 2022−03 | +0.43 | +0.43 | +9.5% | −67% | 0.83 |
| Volatility-managed hold (Moreira and Muir) | 2022−06 | +0.73 | +0.82 | +8.2% | −18% | 0.94 |

- **No spot rule beat holding.** Of the thirteen rules that trade the coins themselves none had a higher net Sharpe ratio than holding both over its own dates; the closest were the Mayer multiple (+0.43, level with holding) and the volatility-managed hold (+0.73 against +0.82, with a worst fall of 18%). The two coins rose a great deal over the sample, and a rule that is out of the market for part of it gives that up.
- **The one high Sharpe ratio is a market-neutral rule.** Funding carry (long spot, short the perpetual, earning the funding) had a Sharpe ratio of +3.7 on a return of 3.2% a year and a worst fall of 1%: the ratio is high because the volatility is tiny, which is what a carry trade looks like until the day it does not (the Basis tab shows its tail). The basis rule, which times the same position, lost (-1.1).
- **Read the deflated Sharpe with the count of tries.** The last column assumes each run was the only one tried; fifteen strategies were tried here, so a value of 0.94 (volatility-managed) or 0.83 (Mayer) is less than it looks.
- **On the data's last day (2 October 2026)** BTC's multiple of its 200-day average was 1.21 (higher than 70% of the days since 2019, above 2.4 on 0.9% of them); its 30-day volatility, 32% a year, was in the 13th percentile of its history; it was 31% below its high (its deepest falls were 77% from November 2021 and 63% from July 2019); and its 90-day correlation with ETH was 0.86 against an average of 0.83. Those are descriptions, not forecasts.

## Pitfalls

- Two coins that move together (a correlation of 0.86) are close to one bet: a board with eight rules long on each coin is one view counted sixteen times.
- A short sample of a huge uptrend favours holding. The comparison column is there so that a rule that merely avoided a fall is not mistaken for one that forecast it.
- A trend rule's result is one path: the six-speed rule's scalars are estimated from the data's own history (about two years), so it starts late and its sample is the shortest.
- The Mayer thresholds are a practitioner's: moving them to fit the past is how a rule gets a good-looking backtest.
- The weekday marks of the backtests are not the daily marks of the overview (which use every day): the numbers differ in the second decimal.

## Try it

```python
from src.assets import crypto_analytics as ca
from src.utils.config import load_config

config = load_config()
out = ca.run_crypto_strategy(config, "mayer", ["BTC", "ETH"], "weekly", None, None, 5.0, 2.0, None, {"floor": 1.0})
print(round(out["metrics"]["sharpe"], 2), out["spec"]["models"][0]["params"])
print(ca.run_crypto_signals(config)["agreement"])                                              # for each coin, how many rules are long, short or without a view
```

Sources: [Liu and Tsyvinski (2021)](https://doi.org/10.1093/rfs/hhaa113), [Liu, Tsyvinski and Wu (2022)](https://doi.org/10.1111/jofi.13119), [Gerritsen et al. (2020)](https://doi.org/10.1016/j.frl.2019.08.011), [Detzel et al. (2021)](https://doi.org/10.1111/fima.12310), [Brock, Lakonishok and LeBaron (1992)](https://doi.org/10.1111/j.1540-6261.1992.tb04681.x), [Burghardt and Lane (1990)](https://doi.org/10.3905/jpm.1990.409259), [Makarov and Schoar (2020)](https://doi.org/10.1016/j.jfineco.2019.07.001), [Griffin and Shams (2020)](https://doi.org/10.1111/jofi.12903), [Moreira and Muir (2017)](https://doi.org/10.1111/jofi.12513), [Schmeling, Schrimpf and Todorov (2023)](https://www.bis.org/publ/work1087.htm).
