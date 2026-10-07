---
title: "Low risk, lottery demand, illiquidity and value with momentum"
slug: low-risk-and-multi-factor
difficulty: 3
chapter: Ch. 20
prerequisites: [plugin-framework, cross-sectional-factors]
stages: []
files: [src/strategies/factors.py]
figures: []
tests: [tests/test_strategy_library_v2.py]
models: [bab, low_idio_vol, max_effect, amihud_illiquidity, value_momentum]
---

# Low risk, lottery demand, illiquidity and value with momentum

## In one sentence

Assets that are risky in ways investors dislike to hold, or boring in ways they overpay to avoid, have earned different returns than their risk alone predicts; these are the standard ways to trade that.

## The idea

**Betting against beta** (`bab`; Frazzini and Pedersen 2014) goes long low-beta assets and short high-beta assets, each leg scaled to a beta of one, because investors who cannot use leverage bid up high-beta assets. **Low idiosyncratic volatility** (`low_idio_vol`; Ang, Hodrick, Xing and Zhang 2006) favours assets whose returns are least driven by noise the market does not share. The **MAX effect** (`max_effect`; Bali, Cakici and Whitelaw 2011) avoids assets with an extreme recent daily gain, a lottery preference. **Illiquidity** (`amihud_illiquidity`; Amihud 2002) favours assets whose price moves most per dollar traded, since they must offer a higher return. **Value and momentum together** (`value_momentum`; Asness, Moskowitz and Pedersen 2013) averages the rank of five-year reversal (a price-based stand-in for cheapness) and 12-1 month momentum, which are negatively correlated.

## Why it matters

These factors were found in thousands of stocks and are the backbone of equity factor investing. On a handful of ETFs they have far less to work with: few assets, no stock-specific lottery, and a universe where liquidity is high everywhere. Testing them here shows how much of a factor survives the move from the paper's universe to a small, liquid one.

## How this repo uses it

`bab` is a structured, beta-neutral trade: its weights are the strategy (long leg beta one, short leg beta minus one, rebalanced monthly with betas shrunk 40% toward one). The others are scores for the ranked book. `amihud_illiquidity` needs traded volume and says so when it is missing.

## What we found

All numbers below come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 79 strategies counted as trials in the deflated Sharpe ratio. Each rule starts on its own first day, so it is compared with equal weight **over the same dates** (the survey's comparison column), which ranges from 0.64 to 0.94 depending on the start. The numbers are exploratory, not tested hypotheses. On 15 ETFs none of these worked. `bab` earned a net Sharpe of +0.06 with a drawdown of 44%, `low_idio_vol` +0.02, `max_effect` +0.03, `value_momentum` +0.03 and `amihud_illiquidity` -0.39, against equal weight over the same dates at 0.64 to 0.88. That is a statement about this universe and sample, not about the factors: with 15 assets, a handful of them equity funds, the cross-section is too small to sort on beta or lottery demand, and the funds that screen as illiquid here (commodity and credit) are not the ones that earned a premium. Run the same rules on thirty or more stocks in the dashboard to see them with breadth.

## Going deeper

```
BAB:   beta_i = 0.6 * beta_ts + 0.4 * 1;   z = rank(beta);   w_H = k (z - mean z)+,  w_L = k (z - mean z)-,  k = 2 / sum|z - mean z|
       weights = w_L / (w_L . beta)  -  w_H / (w_H . beta)           # each leg has beta one; net beta zero; leveraged long
Amihud: ILLIQ = mean_63( |r| / dollar volume );  score = log ILLIQ
```

## Pitfalls

- A factor needs breadth: ranking 15 assets into halves is a coarse sort.
- BAB is levered long low-beta assets: financing and margin costs are not in the backtest.
- Value proxies built from price alone are momentum's mirror image, not accounting value.
- Factors are cyclical: low risk has long droughts when speculative assets rally.

## Try it

```bash
quant backtest --model bab --tearsheet
quant backtest --model value_momentum --allocator score_stack
quant serve        # add 30 stocks as tickers and run the same factors on a universe with breadth
```
