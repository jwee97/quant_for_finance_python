---
title: "Seasonality and calendar effects"
slug: seasonality-and-calendar-effects
difficulty: 2
chapter: Ch. 20
prerequisites: [plugin-framework, multiple-testing]
stages: []
files: [src/strategies/seasonal.py]
figures: []
tests: [tests/test_strategy_library_v2.py]
models: [turn_of_month, sell_in_may, seasonal_rank]
---

# Seasonality and calendar effects

## In one sentence

Returns have not been spread evenly through the calendar: some days and months have paid much more than others, and a rule can hold only those.

## The idea

The **turn-of-the-month effect** (`turn_of_month`) holds equities from the last trading day of a month through the third trading day of the next; pension and payroll inflows and month-end settlement concentrate buying there, and a large share of the historical equity premium has accrued in those four days. The **Halloween indicator** (`sell_in_may`) holds equities November to April and stays in cash May to October (Bouman and Jacobsen 2002). **Same-month seasonality** (`seasonal_rank`; Keloharju, Linnainmaa and Nyberg 2016) ranks each asset by its own average return in the current calendar month over earlier years, using only years already completed.

## Why it matters

Calendar effects are the clearest example of a pattern that is easy to find and easy to fool yourself with: with twelve months and thirty days there are hundreds of calendar rules, and a few always look good in any sample. They are included because they are famous and because the framework can say what each one earned after costs and how many other rules you tried.

## How this repo uses it

`turn_of_month` and `sell_in_may` are date-only rules (the exchange calendar is public in advance, so they are not look-ahead) applied to the equity-class assets, or to every asset when the universe has no class labels; they run as sleeves and `turn_of_month` rebalances daily. `seasonal_rank` shifts each calendar month's history by one year before averaging, so the month being traded never contributes to its own score (a test changes the current month's returns and checks the score does not move).

## What we found

All numbers below come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 71 strategies counted as trials in the deflated Sharpe ratio, equal weight at about 0.67. They are exploratory, not tested hypotheses. Halloween earned a net Sharpe of +0.45 (deflated probability 0.32) with a maximum drawdown of 12% (it is out of the market half the year, in cash earning nothing in this implementation), below equal weight (0.67). The turn-of-the-month rule earned +0.01: holding four days a month and trading in and out of the equity ETFs every month cost about 8x of turnover a year at 10 bps, which absorbs the effect; on a stock-index future with a fraction of a basis point of cost it would look different. `seasonal_rank` (-0.42) did not work on 15 ETFs. The result is a fair summary of what calendar effects are: real in the papers' samples, small, and easy to lose to costs.

## Going deeper

```
turn_of_month:   score_t = 1 if (trading day of month <= 3) or (days left in month < 1) else 0
sell_in_may:     score_t = 1 if month in {Nov, Dec, Jan, Feb, Mar, Apr} else 0
seasonal_rank:   score(asset, month m, year y) = mean of that asset's returns in month m over years before y   (at least 5 years)
```

## Pitfalls

- A calendar rule found by looking at the calendar is a data-mined rule; count every one you looked at.
- Well-known effects tend to weaken once they are published and widely traded.
- Costs matter more than the effect for rules that trade a few days each month.
- Month-end days coincide with index rebalancing and window-dressing: liquidity is not the same as an ordinary day.

## Try it

```bash
quant backtest --model sell_in_may --allocator sleeves --tearsheet
quant backtest --model turn_of_month --allocator sleeves --aum 100000000
```
