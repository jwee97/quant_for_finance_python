---
title: "Black-box and high-frequency strategies: pair trading, ETF arbitrage, market making, rebate trading"
slug: black-box-and-high-frequency-strategies
difficulty: 3
chapter: Ch. 22
prerequisites: [market-making-and-order-flow, pairs-and-statistical-arbitrage, execution-algorithms]
stages: []
files: [src/algo/blackbox.py, src/algo/hft.py, src/microstructure/market_making.py]
figures: []
tests: [tests/test_blackbox_hft.py]
models: []
---

# Black-box and high-frequency strategies: pair trading, ETF arbitrage, market making, rebate trading

## In one sentence

Simulators of the strategies that live on spreads, rebates and short-lived mispricings, built to show how much edge a given half-life, cost and delay leave, not whether such an edge exists.

## The idea

A black-box strategy applies a coded rule at high frequency; a high-frequency strategy is one whose edge is measured in basis points and decays in milliseconds to seconds. The honest way to study either without tick data is a market with a stated structure and the strategy applied exactly as a desk would code it.

**Pair trading** (`simulate_pair_trading`). Log prices obey `log A = alpha + beta log B + s` with the spread `s` an Ornstein-Uhlenbeck process (or, as the control, a random walk with the same variability: no edge). The rule is the classic one (Gatev, Goetzmann and Rouwenhorst 2006): the z-score of `s` against its trailing window, enter beyond `entry` standard deviations, leave inside `exit` or at a `stop`, be flat at the close. Orders reach the market `latency` bars after the decision and each change of position pays `cost_bps` on both legs.

**ETF versus basket arbitrage** (`simulate_etf_arbitrage`, `latency_table`). An ETF trades at its net asset value times `exp(d)` with a mean-reverting premium `d`. Sell the ETF and buy the basket above `entry_bps`, reverse below, leave near zero. With latency, the premium has partly gone by the time the order lands, so profit per trade falls with delay.

**Auto market making** (`auto_market_making`). The Avellaneda-Stoikov (2008) market maker: quotes shaded by inventory against symmetric quotes with the same average spread, with a share of fills coming from informed traders. The decomposition of profit (spread, price moves on inventory, losses to informed flow) is the point.

**Rebate and liquidity trading** (`simulate_rebate_trading`, `pressure_table`). A maker living on exchange rebates and spread, whose problem is that fills are not random. Signed order flow is persistent, moves the price a step later and decides which of our quotes gets hit: buyers lift our ask, sellers hit our bid. A passive ask fills most when the price is about to rise: adverse selection (Glosten and Milgrom 1985). A naive maker quotes both sides always and pays that tax. A pressure-aware maker estimates the pressure from the flow it can see (`latency` steps old) and withdraws the side about to be hit.

## Why it matters

These strategies are where the gap between a backtest and a trade is widest and where a mistake in the cost or the delay is the whole result. A simulator that makes the dependence on latency, half-life and cost explicit tells you what you would have to believe, and measure, before building one.

## How this repo uses it

`src/algo/blackbox.py` and `src/algo/hft.py`, with the existing `src/microstructure/market_making.py` for the market maker; `quant algo hft --sim pairs|etf|rebate|amm` runs each and the dashboard's Execution tab shows the latency tables. They are research simulators on stylised markets: one share per fill, no queue priority, no order book, parameters in place of measurements. The tests check each against its mechanism: the random-walk control earns nothing, the premium strategy's edge falls monotonically with latency, the aware maker's edge fades as its view of the flow ages and is gone when the view is as old as the flow's memory, and the inventory-shaded market maker holds less inventory and takes less risk than the symmetric one.

## What we found

Sixty simulated days of pair trading with an OU spread: 152 round trips, a 66% win rate, a mean of 1.6 bps a day (standard error 1.0) and a Sharpe of 3.2 on those days, with 11.1 bps of cost against 12.8 bps gross: costs took 87% of the edge, so a 20% rise in costs would have ended it. The premium strategy's edge by latency: 116 bps a day at zero delay (97% of trades profitable), 76 at one bar, 43 at two, and -20 at four: the half-life of the premium sets how fast the edge goes, and at four bars of delay nothing is left. The rebate maker: the naive maker earned 669 (standard error 52) in the units of the simulation, with rebate and spread worth 1,411 and adverse selection plus inventory losses taking 742; the pressure-aware maker earned 1,272 at zero latency, 873 at four and 670 at sixteen, the same as the naive one. Avellaneda-Stoikov inventory shading earned about the same mean as symmetric quotes (56 against 58) with 7.5 against 20.1 of standard deviation (Sharpe 7.5 against 2.9) and a third of the inventory.

## Going deeper

```
pairs:     z_t = (s_t - mean_n(s)) / sd_n(s);   enter short spread if z > entry, long if z < -entry;   exit when |z| < exit;   flat at the close;   fills at t + latency
ETF:       d_t = log(ETF / NAV),  d_(t+1) = (1 - k) d_t + noise,  half-life ln 2 / k bars;   profit per trade ~ |d_entry| - |d_(entry + latency)| - cost
maker:     adverse selection = price move after a fill against our side;   edge(aware) = avoided adverse fills - missed rebate and spread
A-S:       reservation price r = mid - q gamma sigma^2 (T - t);   spread = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)
```

## Pitfalls

- A strategy that works in the simulation it was designed for has demonstrated nothing about the world. The aware maker's edge is built into the world it is tested in (flow does move prices); the lesson is its size against rebate and spread and how latency erodes it.
- Costs and delay decide these strategies. Move `cost_bps` or `latency` before believing any row.
- Real queues, hidden liquidity, fee tiers and other fast participants are absent here and each can erase the edge.

## Try it

```bash
quant algo hft --sim pairs
quant algo hft --sim etf
quant algo hft --sim rebate
quant algo hft --sim amm
```
