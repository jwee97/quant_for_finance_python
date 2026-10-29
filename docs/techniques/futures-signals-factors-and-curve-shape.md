---
title: "Futures: what every rule says, what moves a curve, and how its shape changes"
slug: futures-signals-factors-and-curve-shape
difficulty: 3
chapter: Platform
prerequisites: [futures-curves-carry-and-rolls, cross-asset-carry, book-rules-systematic-trading]
stages: []
files: [src/assets/futures_analytics.py, src/assets/strategy_runs.py, src/strategies/multi_asset.py, src/risk/performance.py, src/webapp/asset_api.py, src/webapp/static/futures.js, src/references.py]
figures: []
tests: [tests/test_futures_curve_exact.py, tests/test_futures_strategies_exact.py, tests/test_strategy_runs_exact.py, tests/test_performance_exact.py, tests/test_webapp_assets_ml.py]
models: []
---

# Futures: what every rule says, what moves a curve, and how its shape changes

## In one sentence

The Futures tab reads a curve three more ways (the principal components of its daily changes, its shape through time, and its carry by calendar month and against the range of its own year), puts fifteen strategies with their own settings behind one backtest form, and adds a board that shows what every rule says about every root today.

## The idea

**What moves a curve.** Litterman and Scheinkman (1991) took the daily changes of the yields at each maturity and found that three principal components explain nearly all of them, and that they are a *level* move (every maturity up together), a *slope* move (the front against the back) and a *curvature* move (the middle against the ends). Tolmasky and Hindanov (2002) did the same for the futures curve of petroleum. Here the changes are those of the contract that held each rank the day before (the change in `F2` is that of the contract that was second yesterday, so a roll between ranks is not mistaken for a move), the covariance of the columns is decomposed, and each component is named by the pattern of its loadings: one sign across the curve is a level, one change of sign a slope, two a curvature, and anything else "higher order". Signs are fixed (level up, slope more on the back than the front, curvature more in the middle than at the ends).

**How the shape changed.** For every month back in time, the log of each contract month's price over the front's: positive is contango (later contracts dearer, a long pays the roll), negative backwardation. A market that spends years in one colour and then turns is changing regime; the picture says when.

**Carry against its own range, and by month.** The roll yield through time with the range of the year before each day (10th to 90th and 25th to 75th percentiles, and the median), so that a carry outside the dark band is unusual *for this market*; and the average carry on the days of each calendar month, which is flat for a market without seasonality and has a shape for one with it (natural gas has a winter premium: Routledge, Seppi and Spatt (2000) derive such a seasonal forward curve from storage).

**What every rule says today.** For a set of roots the page computes the score each rule gives on the last day, divides it by the average size of that rule's *own* absolute score over the last two years (its *strength*: 1 is a typical position, the sign is the side, capped at 3), and shows rules by roots as a grid, with the count of rules long, short and without a view on each root. The rules are the platform's strategies: trend of several kinds (Moskowitz, Ooi and Pedersen 2012; Hurst, Ooi and Pedersen 2017; Carver 2015), carry (Koijen, Moskowitz, Pedersen and Vrugt 2018), basis momentum, skewness, the variance-ratio switch, and trend and carry together.

**Strategies with their own settings.** The backtest form lists fifteen strategies in four groups and, under the one chosen, every setting its constructor takes, with its default; only what is changed is sent. A new strategy, `momentum_carry`, averages a bounded trend signal and a bounded carry signal for each root (Fuertes, Miffre and Rallis 2010 sort commodities on momentum and the slope of the curve together; Carver combines trend and carry as separate forecasts).

## Why it matters

A curve is a high-dimensional object and a trader's questions about it are few: has the whole thing moved or has it tilted? Is the carry I am being paid unusual? Do my rules all say the same thing, in which case I hold one bet counted six times? The principal components, the shape history and the board answer those by drawing; the settings and the risk views of the backtest say what a rule would have done and how, after costs.

## How this repo uses it

`src/assets/futures_analytics.py` has `rank_changes`, `pca_of_changes`, `term_structure_pca`, `curve_shape_history`, `carry_by_month`, the bands inside `carry_dashboard`, `signals_board` and `run_signals`, and the catalogue `FUTURES_STRATEGIES` (each entry has a group, the strategy behind it, and what it needs of the roots); `run_futures_strategy` accepts the strategy's own `params`. `src/strategies/multi_asset.py` has `momentum_carry`. Every backtest result also carries the risk measures, the deepest falls, the volatility through time, the months, the distribution of days, the correlation of the roots and how the book was spread (`src/assets/strategy_runs.py`, `src/risk/performance.py`). The routes are `GET /api/futures/curve` (with `spacing` and `dates` for the curves drawn), `POST /api/futures/signals` and `POST /api/futures/strategy/backtest`.

## What we found

The repository holds no futures prices, so there is no finding about real markets here: every number below is from the simulated roots (a Schwartz-Smith model) and says that the machinery recovers what the model has, not how markets behave.

- **The factors.** On the simulated curves the first component is a level move and holds 98.1% of the variance for crude oil and 97.1% for natural gas; the slope holds 0.8% and 2.1%; for gold the second and third components have no clear shape ("higher order") because its curve barely moves. The tests plant a level, a slope and a hump of known size in the daily changes and recover the loadings to five decimals and the variance shares exactly.
- **Seasonality.** The simulated natural-gas carry swings from about +0.3 a year in March to about -0.9 in September (a winter premium put in by the model); crude oil's monthly averages stay between -0.03 and +0.16, and gold's are flat at -0.05.
- **The rules, on simulated draws.** Run through the pipeline on three simulated cross-asset universes, `momentum_carry` averaged a net Sharpe of +0.09, between `carry_ts` (+0.14) and `tsmom` (-0.17); the skewness premium +0.29, the t-statistic trend +0.02, Carver's six-speed trend -0.01 and the variance-ratio switch -0.48 ([the table](../institutional_findings.md), section 7). Three draws are too few to rank rules, and the premium that carry earns is built into the simulation.

## Pitfalls

- A curve's components are unstable when the market is quiet: gold's second and third are noise. The share of the variance is printed beside each so that a 0.2% factor is not read as a finding.
- Agreement is not accuracy. Six trend rules agree because they are six readings of one thing; the board's counts are of rules, not of independent opinions.
- The strength divides by the rule's own typical size over two years, so it says how unusual today's view is for that rule, not how profitable.
- A strategy's settings are only checked when it is built: a value the strategy refuses (a weight above 1) comes back as an error naming the strategy.
- Carry by calendar month on four years of history is four observations a month: it is a description of the sample.

## Try it

```python
from src.assets import futures_analytics as fa

curve = fa.simulated_curve("CL")
print({c["label"]: round(c["explained"], 3) for c in fa.term_structure_pca(curve, 8)["components"]})     # level, slope, curvature and their shares
print(fa.signals_board(fa.futures_bundle_for(["CL", "NG", "GC", "HG", "ES"]))["agreement"])              # for each root, how many rules are long, short or without a view
```

Sources: [Litterman and Scheinkman (1991)](https://doi.org/10.3905/jfi.1991.692347), [Tolmasky and Hindanov (2002)](https://doi.org/10.1002/fut.10046), [Routledge, Seppi and Spatt (2000)](https://doi.org/10.1111/0022-1082.00248), [Moskowitz, Ooi and Pedersen (2012)](https://doi.org/10.1016/j.jfineco.2011.11.003), [Hurst, Ooi and Pedersen (2017)](https://doi.org/10.3905/jpm.2017.44.1.015), [Koijen, Moskowitz, Pedersen and Vrugt (2018)](https://doi.org/10.1016/j.jfineco.2017.11.002), [Fuertes, Miffre and Rallis (2010)](https://doi.org/10.1016/j.jbankfin.2010.04.009), [Gorton and Rouwenhorst (2006)](https://doi.org/10.2469/faj.v62.n2.4083), [Erb and Harvey (2006)](https://doi.org/10.2469/faj.v62.n2.4084), and Carver (2015), *Systematic Trading*, Harriman House.
