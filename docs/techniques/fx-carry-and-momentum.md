---
title: "FX: forwards, carry, the forward-premium puzzle and crash risk"
slug: fx-carry-and-momentum
difficulty: 2
chapter: Ch. 19
prerequisites: [performance-metrics]
stages: []
files: [src/assets/fx.py, src/assets/bundles.py]
figures: []
tests: [tests/test_assets.py]
models: []
---

# FX: forwards, carry, the forward-premium puzzle and crash risk

## In one sentence

Currencies with high interest rates have tended to earn more than the low-rate ones even though theory says they should depreciate by the difference; `src.assets.fx` supplies the conventions, the tests and a synthetic market with the crash risk that carry is paid for.

## The idea

A pair `EURUSD = 1.10` is the number of dollars per euro (base EUR, quote USD). **Covered interest parity** fixes the forward at `F = S e^{(r_quote - r_base) T}`, so the forward discount is the interest differential (`forward_rate`, `forward_points`, `implied_rate_differential`). A long base, short quote position earns the spot return plus the carry: `excess_return = S_t / S_{t-1} - 1 + (r_base - r_quote) dt`. `cross_rate` builds a cross from two dollar pairs and `triangular_gap` checks for triangular arbitrage.

**Uncovered interest parity** says the spot rate is expected to move by the differential, so carry earns nothing on average. The **Fama regression** (`fama_regression`) regresses the realised depreciation on the forward discount: parity predicts a slope of one, and the data typically give a slope near zero or negative, the **forward-premium puzzle** (Fama 1984). A **carry portfolio** (`carry_portfolio_returns`) is long the highest-yielding currencies and short the lowest. Its excess return is explained by exposure to global volatility and to crash risk (Lustig, Roussanov and Verdelhan 2011; Brunnermeier, Nagel and Pedersen 2009): the high yielders fall together when volatility spikes.

## Why it matters

The carry trade is the textbook example of a premium that looks like a free lunch in the Sharpe ratio and is a short-crash position in the drawdown. Whether a carry or momentum overlay helps depends on how its losses line up with the rest of the portfolio's.

## How this repo uses it

`fx_bundle` converts a market (spot, rates, USD rate) into a bundle with currency excess returns and the carry signal, so `carry_xs`, `carry_ts` and momentum run on it unchanged. `synthetic_fx_market` generates a G10-style set: mean-reverting interest rates, spot changes that offset only a fraction `uip_beta` of the differential, so carry earns `1 - uip_beta` of it on average, and a crash factor, which depreciates the high yielders when global volatility spikes (the crash drift is compensated so the premium is what is stated). Real data needs a vendor for spot and for deposit or forward rates; the loader contract is a table of spot and short rates per currency.

## What we found

The tests check covered interest parity round-trips, the triangular-arbitrage gap on constructed crosses, that excess returns include the carry with the right timing, that the Fama regression recovers a planted slope, and that on the synthetic market carry earns the non-parity share of the differential while its return skewness is below -0.5 (the premium comes with crashes). These are properties of the simulator and the code, not of the real market.

## Pitfalls

- Spot returns alone omit carry; always use total or excess returns.
- Forward rates from vendors embed bid/ask and basis (the cross-currency basis since 2008): parity no longer holds exactly.
- The carry premium is concentrated in rare episodes; ten years may hold none.
- High-carry currencies are often illiquid and expensive to trade in stress.

## Try it

```python
from src.assets import fx

m = fx.synthetic_fx_market(n_days=1500, seed=1)
r = fx.carry_portfolio_returns(m["excess_returns"], m["rates"], m["usd_rate"], n_long=2, n_short=2)
print(round(float(r.mean() * 252), 3), round(float(r.std() * 252 ** 0.5), 3))
```
