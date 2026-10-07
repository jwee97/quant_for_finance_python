---
title: "Trend following and momentum: the indicator toolkit"
slug: trend-following-toolkit
difficulty: 2
chapter: Ch. 20
prerequisites: [momentum, plugin-framework]
stages: []
files: [src/strategies/trend.py, src/strategies/time_series.py]
figures: []
tests: [tests/test_strategy_library_v2.py]
models: [tsmom, tsmom_multi, ma_ensemble, breakout_ensemble, kama_trend, adx_trend, supertrend, keltner_breakout, macd_trend, ichimoku_trend, high_52w, smooth_momentum, momentum_13612w, accelerating_dual_momentum, residual_momentum]
---

# Trend following and momentum: the indicator toolkit

## In one sentence

Trend following buys what has been rising and sells what has been falling; the strategies here are the same idea measured in fifteen different ways, from the academic baseline to the indicators traders use.

## The idea

**Time-series momentum** (Moskowitz, Ooi and Pedersen 2012) goes long an asset whose past twelve-month return is positive and short one whose return is negative, sizing each by 40% over its own volatility (`tsmom`); averaging the signs over one, three and twelve months (`tsmom_multi`) keeps the premium with less turnover. The **crossover ensemble** (`ma_ensemble`, after Baz et al. 2015) takes exponential-average crossovers at three speeds, divides by price volatility and by the signal's own recent size, and passes the result through a response that fades extreme trends. **Breakout ensembles** (`breakout_ensemble`, after Carver) score where the price sits inside its 10- to 320-day range. **Adaptive and filtered trend**: Kaufman's average speeds up in clean trends and slows in noise (`kama_trend`); Wilder's ADX switches the strategy off when no trend exists (`adx_trend`); SuperTrend (`supertrend`) and Keltner channels (`keltner_breakout`) trail price with an ATR band; MACD (`macd_trend`) and the Ichimoku cloud (`ichimoku_trend`) are the best known charting rules. On the **cross-sectional** side: assets near their 52-week high keep outperforming (`high_52w`, George and Hwang 2004); momentum that arrives in many small steps lasts longer than momentum from a few jumps (`smooth_momentum`, Da, Gurun and Warachka 2014); momentum of the part of the return the market does not explain is steadier (`residual_momentum`, Blitz, Huij and Martens 2011); Keller's 13612W weights recent months heavily (`momentum_13612w`); and `accelerating_dual_momentum` applies the dual-momentum rule to an averaged one-, three- and six-month return.

## Why it matters

Trend following is the most robust return premium in the literature: it shows up in every asset class, in samples going back a century, and it tends to pay in crises when equities fall. It is also the strategy most often over-fitted, because a hundred indicators say nearly the same thing and picking the one that did best is a data-mining exercise. Putting all of them behind one interface, on the same costs, is how you see how much of the difference between them is real.

## How this repo uses it

Each rule is a registered strategy; per-asset rules run as independent **sleeves** (equal slice of capital times the signal, cash when flat) and declare their own rebalance frequency (weekly or daily), cross-sectional rules run as a ranked book. Indicators that need a high and a low (ADX, SuperTrend, Keltner, Ichimoku) fall back to the close when the bundle has none. Every score uses data through the date only, which the framework's look-ahead check verifies for each one.

## What we found

All numbers below come from one run of [the strategy survey](../strategy_survey.md): the platform's 15 ETFs, default parameters, net of 10 bps costs, 79 strategies counted as trials in the deflated Sharpe ratio. Each rule starts on its own first day, so it is compared with equal weight **over the same dates** (the survey's comparison column), which ranges from 0.64 to 0.94 depending on the start. The numbers are exploratory, not tested hypotheses. On this universe and sample trend following did not beat buying everything: only `dual_momentum` (+0.88) matched equal weight over its dates (+0.88, from February 2009, after the crisis), and its deflated probability is 0.89. The rest of the family earned net Sharpe ratios from +0.41 (`tsmom`, 13x turnover, against +0.89 for equal weight over the same dates) and +0.37 (the existing 50/200 crossover, against +0.85) down through +0.26 for the crossover ensemble and +0.34, +0.32 and +0.33 for the cross-sectional 52-week-high, smooth and residual variants, which only matched plain cross-sectional momentum (+0.32). The fast indicator systems lost money after costs: `macd_trend` (-0.10), `adx_trend` (-0.24), `supertrend` (-0.31), `kama_trend` (-0.32) and `ichimoku_trend` (-0.34). Over these twenty years equities, bonds and gold mostly rose, so a rule that sits out part of the time pays for the protection; trend following earns its keep in the years a long-only book falls, and 2008, 2020 and 2022 are three years in a sample of twenty.

## Going deeper

```
tsmom:             score = sign(r_12m) * min(0.40 / sigma_ewma, 4)
ma_ensemble:       x = (EMA_f - EMA_s) / std_63(price);  y = x / std_252(x);  signal = y * exp(-y^2/4) / 0.89     averaged over (8,24), (16,48), (32,96)
breakout:          raw_L = 40 * (P - (max_L + min_L)/2) / (max_L - min_L); smoothed with span L/4; mean over L in {10,...,320}; clipped to [-20, 20] / 20
KAMA:              ER = |P_t - P_(t-n)| / sum |dP|;  sc = (ER*(2/3 - 2/31) + 2/31)^2;  K_t = K_(t-1) + sc*(P_t - K_(t-1))
13612W:            (12 r_1m + 4 r_3m + 2 r_6m + r_12m) / 19
```

## Pitfalls

- Trend rules are one idea tested many ways: counting each as an independent confirmation overstates the evidence.
- The premium is concentrated in a few large trends; a decade without one is normal, and the cost of whipsaws is certain.
- Signals that flip often (SuperTrend, KAMA, ADX) need low costs or a slower rebalance.
- Long-only equity ETFs are a hard test for a rule built for diversified futures.

## Try it

```bash
quant backtest --model tsmom --allocator sleeves --tearsheet
quant backtest --model ma_ensemble --allocator sleeves
quant backtest --model high_52w --allocator score_stack
```
