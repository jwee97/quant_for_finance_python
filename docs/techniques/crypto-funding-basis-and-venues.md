---
title: "Crypto funding, basis, venue spreads and liquidity"
slug: crypto-funding-basis-and-venues
difficulty: 2
chapter: Ch. 22
prerequisites: [crypto-carry]
stages: []
files: [src/assets/crypto_analytics.py, src/data/crypto.py, src/webapp/asset_api.py, src/webapp/static/crypto.js, src/assets/free_data/crypto.py, src/webapp/data_api.py, src/webapp/data_itick.py]
figures: []
tests: [tests/test_assets_workspaces.py, tests/test_crypto.py, tests/test_free_data_crypto.py, tests/test_webapp_data.py, tests/test_webapp_data_itick.py]
models: []
---

# Crypto funding, basis, venue spreads and liquidity

## In one sentence

A perpetual swap trades near spot because longs and shorts pay each other a funding rate, and that rate, the gap to spot, the price differences between exchanges and the depth of each order book are the four numbers that decide whether a crypto trade is worth making.

## The idea

A **perpetual swap** is a futures contract with no expiry. To keep its price near the spot index it uses **funding**: every few hours the side that is crowded pays the other. When longs pay shorts (the usual state in a rising market) the funding rate is positive and a short position earns it. The **basis** is `perp / spot - 1`. Holding spot long against a perpetual short is delta-neutral and earns funding: the *cash-and-carry* trade, whose risks are margin calls on the short leg in a rally, exchange failure and funding turning negative. **Cross-exchange spreads** are differences between the same coin's prices on different venues; a spread is only an opportunity if it survives both taker fees and the slippage of crossing the book twice. **Liquidity** is how much can be traded before the price moves: the bid-ask spread, the dollars resting within a few basis points of the mid and the price impact of an order of a given size.

## Why it matters

Annualised funding of 10% on a 5 bps fee sounds like free money until the break-even is computed: how many days of carry pay for opening and closing both legs. A 6 bps gap between two exchanges looks like an arbitrage until fees of 9 bps and 2 bps of slippage turn it into a loss. And a $10 million order does not trade at the quoted price. Each number is simple; reading them together keeps a trader from confusing a quote with something they could actually do.

## How this repo uses it

The **Crypto** tab (`src/assets/crypto_analytics.py`, served by `src/webapp/asset_api.py`) has six views. *Overview* shows six cards (trend, funding, momentum, volatility, liquidity, on-chain stablecoin supply), spot against the perpetual and the basis and funding through time. *Funding* is a coin-by-venue table of annualised funding with its 7- and 30-day averages, its z-score against 90 days and its percentile against a year, plus each coin's funding through time by venue. *Basis and carry* measures the basis against its history, the carry of the delta-neutral pair (growth, volatility, worst fall, Sharpe) and the break-even in days for a fee you set. *Across exchanges* lists every buy-sell pair of venues with its gross gap, fees, slippage and net result in basis points and dollars. *Liquidity* gives spread, depth within 10, 25 and 50 bps and the price impact of $100 thousand, $1 million and $10 million orders. *Backtest* runs spot momentum, funding carry, basis reversion, volatility breakout, stablecoin flow and regime-filtered trend through the platform's pipeline, with fees and slippage.

**Provenance is part of the data.** The repository holds Deribit's BTC and ETH perpetual daily closes, funding history and index price, and DefiLlama's total stablecoin supply: that is `platform` data. It holds no order books, no quotes from other venues, no exchange net flows and no history for other coins. Where the page needs them it simulates them and says so: other venues' funding and quotes are Deribit's with venue offsets and noise; SOL, BNB and XRP are generated from BTC with a beta and their own noise; the order book is a parametric depth profile with assumed round numbers. Every table is badged with the source of each part ("Platform data", "Simulated: a model", "Mixed sources"), and a file in `data/user/crypto/<SYMBOL>.csv` replaces the simulation for that coin.

**Real prices, books and funding without sourcing them.** The **Real crypto data (free)** card downloads, for a coin: its daily history (spot, perpetual and funding from OKX, or spot alone from Kraken, Bitstamp or Coinbase when OKX does not list it); the real quotes and order books of the coin at this moment on Coinbase, Kraken, Bitstamp, OKX (spot and perpetual) and Deribit's perpetual; and OKX's funding history. Files go under `data/user/downloads/crypto/`. A downloaded coin is badged "Downloaded: free data" where it is data and still "Simulated: a model" where it is not: the funding table says, row by row, which venues' funding is data (Deribit's, OKX's) and which is simulated. With real books the *Across exchanges* view compares only like with like (spot with spot, perpetual with perpetual) on real bids and asks, and *Liquidity* walks the real book for the cost of each size, showing n/a where the saved levels (up to 400 a side, within 5% of the mid) cannot fill it. Bitcoin and Ether keep the platform's longer Deribit history; the card adds their venues and funding. With an iTick key the card can instead download a coin's daily closes against USDT from iTick: spot only (no perpetual, funding or order books, which the exchanges provide), which exchange's prices they are is not documented, and a coin the exchanges do not list can still be had this way. A coin downloaded that way has spot prices, momentum and volatility from data, and its basis, funding and venue views stay modelled or say they have nothing. [What is downloaded and what it is not](../free_data.md).

## What we found

The tests pin the arithmetic and the labels. Deribit's funding stays real while the other venues are marked simulated; simulated coins come from BTC and say so; the break-even in the basis view is the fee over the funding rate; a cross-exchange edge is the gross gap less both fees and the slippage on each leg; the order-book impact grows with size and is labelled as a model; a user's file replaces the simulation. For downloaded data the tests pin that every venue's book is normalised to coins, best first, in the unit it quotes in; that far levels are dropped and the count capped; that the summary's spread, depth and the cost of size come from walking the levels; that a snapshot keeps what worked, says what did not and fails when nothing did; that a coin's history is spot, perpetual and funding on the same days; that OKX's funding is summed per UTC day with the part day at the start dropped; and that the cross-exchange view uses real prices and compares like with like. The strategies run on Deribit BTC and ETH through the same pipeline as every other backtest. On the default simulated data no venue pair covered its fees when this guide was written, which is what noise around one price produces: the lesson is the method, not a finding about exchanges.

## Going deeper

```
funding (annualised, short side)  =  daily funding * 365
basis                             =  perp / spot - 1
break-even days                   =  (4 * fee_bps / 10000) / (mean daily funding over 30 days)         open and close two legs
net spread, buy at A, sell at B   =  (bid_B / ask_A - 1) in bps  -  taker_A  -  taker_B  -  2 * slippage
depth within d bps (per side)     =  depth_10bp * (d / 10)^0.8                                         the parametric book
impact of an order of size S      =  spread / 2  +  10 * (S / depth_10bp)^(1 / 0.8)   bps from the mid
```
Crypto trades every day, so the overview annualises by 365; the strategy backtests use weekday marks and 252, like every other backtest here, so their numbers compare with the rest of the platform.

## Pitfalls

- Funding that is high is usually high because longs are crowded: the trade that earns it is the one that gets squeezed.
- A delta-neutral pair is not risk-free: the short leg is margined and a sharp rally can liquidate it before the spot leg is sold.
- Simulated venues and the parametric book are for exercising the page; they are not quotes and no conclusion about a real exchange follows from them.
- The platform holds no exchange net-flow data, so a strategy that needs it says it cannot run; the on-chain signal that is available is stablecoin supply.
- Taker fees change with a venue's volume tiers; the defaults here are round numbers, and for real venues they are entry-level **assumptions**, not your tier.
- OKX quotes in USDT, not dollars: a few basis points of a spot-perpetual basis can be the USDT/USD rate. The downloaded daily closes are at 00:00 UTC while the platform's own BTC and ETH series use the 08:00 UTC mark, so the two are not on one clock.
- A book is one moment. A depth that looks ample in a calm minute says little about the same venue in a crash, and OKX keeps only about three months of funding history whatever you ask.

## Try it

```bash
quant serve            # the Crypto tab: Overview, then Funding, then Across exchanges
python -m pytest tests/test_assets_workspaces.py tests/test_crypto.py -q
```
To download a coin without any file, tick it in the **Real crypto data (free)** card and press *Download real data*. To use your own history, save a CSV (`time, spot` and optionally `perp`, `funding_daily` or `funding_8h`) as `data/user/crypto/SOL.csv` and reload the tab.
