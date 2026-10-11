---
title: "Reading a futures curve: carry, rolls, basis and margin"
slug: futures-curves-carry-and-rolls
difficulty: 2
chapter: Ch. 19
prerequisites: [futures-and-commodity-curves]
stages: []
files: [src/assets/futures_analytics.py, src/assets/futures.py, src/webapp/asset_api.py, src/webapp/static/futures.js, src/assets/free_data/futures.py, src/webapp/data_api.py]
figures: []
tests: [tests/test_assets_workspaces.py, tests/test_free_data_futures.py, tests/test_webapp_data.py]
models: []
---

# Reading a futures curve: carry, rolls, basis and margin

## In one sentence

A futures curve is the price of the same thing for delivery on different dates, and its slope tells you what holding the contract earns or costs, when to move to the next one and how much capital the position ties up.

## The idea

A futures contract expires, so a position held for months is a chain of contracts, each sold a few days before expiry and replaced by the next. If the later contract is cheaper (the curve slopes down, *backwardation*), a long position buys the next contract below the price it sold the old one for and earns the difference as it rolls up the curve: positive **carry**, also called roll yield. If the later contract is dearer (*contango*), the same roll costs money. Carry is annualised as `ln(F_near / F_far) / (T_far - T_near)`. The **basis** is the gap between a futures price and the spot price of the underlying, which shrinks to zero at expiry. **Margin** is the collateral the exchange or broker requires: an initial amount to open, a maintenance amount below which equity triggers a call, both a fraction of the contract's notional value (price times the multiplier).

## Why it matters

Two traders with the same view on oil can earn different returns because one rolled on the last day, when the front contract was thin, and the other rolled when open interest had moved to the next contract. A trend strategy that ignores carry can look profitable on a price chart and lose money to contango. A leveraged book can be correct about direction and still be closed out because margin usage crossed the line.

## How this repo uses it

The **Futures** tab (`src/assets/futures_analytics.py`, served by `src/webapp/asset_api.py`) has five views. *Curve* draws price by contract month, how the curve moved over the last two months, the roll yield of each contract, the curve's state (contango, backwardation or mixed) and the contract specification. *Roll* names the contract held and the next, the calendar roll date (a set number of days before expiry), the date open interest moved to the next contract, the suggested roll date (the earlier of the two), what the roll costs (the calendar spread in dollars per contract plus the spread crossed and commission) and the rolls of the past. *Carry and basis* follows carry and basis through time against their own history, with z-scores. *Margin* takes a book of positions and reports notional, initial and maintenance margin, leverage and the distance to a margin call, with warnings for margin use above 70% of equity, concentration in one root and contracts close to expiry. *Backtest* runs carry, trend, breakout and momentum strategies across several roots through the platform's own pipeline, so costs, the one-day signal lag and volatility targeting are the same as in any other backtest.

**Provenance is part of the data.** The repository holds no futures contract prices. The default curves are *simulated*: a Schwartz-Smith two-factor model per root with an exchange-style expiry calendar and a simulated open-interest profile, and the contract specifications and margins are illustrative round numbers, not current exchange values. Every payload says `simulated`. A file in `data/user/futures/<ROOT>.csv` replaces the simulation for that root and is labelled "Your file".

**Real curves without sourcing them.** The **Real futures curves (free)** card downloads the curve of CL, NG, GC, SI, HG, ES, NQ, ZN, ZB, 6E, ZC, ZS or ZW from Yahoo Finance, which serves a daily price for each contract month (2 to 16 contracts, 6 months to 5 years of history), and Bitcoin and Ether futures from Deribit. The file is saved under `data/user/downloads/futures/` and the root is marked "(downloaded)" in the menu. Yahoo does not say when a contract expires, so the expiry is **computed from the exchange's rule** (the third-last business day of the delivery month for gold, the third Friday for the equity indexes, three business days before the 25th of the month before delivery for crude oil, and so on) without a holiday calendar: it can be a day off. Yahoo has **no open interest**, so the open-interest chart and the open-interest roll date are left out and the roll is by the calendar alone; margins stay the illustrative figures. Yahoo stops serving a contract when it expires, so a later download is merged with the earlier file and the history of expired contracts is kept: the rolls of the past appear as the file ages. Deribit's futures carry open interest and volume. [What is downloaded and what it is not](../free_data.md).

## What we found

The tests pin the arithmetic. The curve snapshot matches the term-structure panel; the state classification (contango, backwardation, mixed) follows the sign and consistency of the roll yields; open interest moves from one contract to the next before expiry; the carry and basis dashboards agree with the curve; the margin dashboard's notional and margin figures reproduce by hand and its warnings fire at the stated thresholds; a curve file round-trips and is labelled as yours; simulated curves are deterministic and an unknown root is refused. The futures strategies run through the same pipeline as every other backtest. Because the curves are a model, a profitable simulated carry backtest is a check that the machinery works, not evidence that carry pays. For downloaded curves the tests pin each exchange's last-trading-day rule against known dates, the way the next contracts skip what has expired and name their Yahoo symbols, the one priced bar per exchange day that a real Yahoo answer gives, the stacking of contracts into a curve the workspace loader reads, the skipping and reporting of a contract Yahoo lacks, the merge that keeps a contract the feed has dropped and lets new rows win, and Deribit's strip with its open interest.

## Going deeper

```
carry (annualised)   =  ln(F_near / F_far) / (T_far - T_near)               positive in backwardation
suggested roll date  =  min( expiry - roll_days ,  first date open interest in next >= open interest in front )
roll cost per contract  =  (F_next - F_front) * multiplier  +  spread_ticks * tick_value  +  2 * commission
initial margin       =  margin_per_contract * contracts * margin_scale        margin use = initial margin / equity
```
Two-factor models (Schwartz-Smith) split the log price into a short-lived deviation that decays back and a long-run level that drifts: the first drives the slope of the curve, the second drives the level.

## Pitfalls

- A simulated curve is not a quote; the carry it shows is whatever the model's parameters imply.
- Backwardation does not make a long position profitable on its own: the spot price moves, and usually moves more than the roll yield.
- Rolling by open interest and rolling by the calendar can be weeks apart on thinly traded contracts.
- Margin requirements change with volatility, and brokers can raise them without notice; the figures here are illustrative.
- A downloaded curve is delayed end-of-day data: the last bar can be a session in progress, a computed expiry can be a day off, and with no open interest a roll "by open interest" is not available. Backtests run on the simulated roots only, because they share one calendar and one model; downloaded curves feed the Curve, Roll, Carry and Margin views.
- A continuous price series made by stitching contracts together hides the roll gap and misstates returns; the strategy backtests use an excess-return index built contract by contract (`assets.futures.excess_return_series`), which earns or pays the roll yield as it happens.

## Try it

```bash
quant serve            # the Futures tab: Curve, then Roll, then Margin with an example book
python -m pytest tests/test_assets_workspaces.py -q -k "curve or roll or margin or carry"
```
To analyse a real curve without any file, tick a root in the **Real futures curves (free)** card and press *Download real data*. To bring your own, save a long table (`date, contract, expiry, price`, optionally `open_interest` and `volume`) as `data/user/futures/CL.csv` and reload the tab.
