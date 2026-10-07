---
title: "Futures, continuous contracts, roll yield and commodity curves"
slug: futures-and-commodity-curves
difficulty: 3
chapter: Ch. 19
prerequisites: [performance-metrics]
stages: []
files: [src/assets/futures.py, src/assets/commodities.py, src/assets/bundles.py]
figures: []
tests: [tests/test_assets.py]
models: []
---

# Futures, continuous contracts, roll yield and commodity curves

## In one sentence

A futures "price series" is stitched together from contracts that expire, and how it is stitched changes every statistic you compute from it; `src.assets.futures` builds continuous series correctly and measures carry, roll yield and basis momentum, and `src.assets.commodities` estimates the curve model behind them.

## The idea

Every contract expires, so a long history is a sequence of contracts. A raw splice of the front contract contains the **roll gap** (the difference between the expiring and the next contract) as a fake return. `continuous_series` rolls `roll_days` before expiry and removes the gap by **ratio back-adjustment** (percentage returns are right), **difference adjustment** (Panama canal, point moves are right) or not at all. `excess_return_series` goes further: it measures the return of holding the front contract and rolling, contract by contract, so the roll never appears and the series is the tradable excess return, including the **roll yield** (positive in backwardation, negative in contango).

**Carry** is what a position earns if the curve does not move: `carry = ln(F1 / F2) / (T2 - T1)` annualised (Koijen, Moskowitz, Pedersen and Vrugt 2018). `roll_yield` is the same quantity as a signal, and `basis_momentum` (Boons and Prado 2019) is the trailing mean of the return difference between the first and second contract. `contract_calendar` generates expiry dates for common exchange rules.

For commodities the **Schwartz-Smith** model writes the log price as a mean-reverting short-term deviation `chi` plus a random-walk long-run level `xi`, so that `ln F(T) = e^{-kappa T} chi + xi + A(T)`; the risk premium bends the curve into contango or backwardation. `fit_schwartz_smith` estimates it by a Kalman filter whose measurement matrix changes daily as contracts age, and the filtered `chi` and `xi` are the quantities a trader wants: how tight the market is and where the long-run price is. `curve_factors` gives level, slope and curvature of the observed curve, and `seasonal_factors` measures calendar-month seasonality with HAC inference.

## Why it matters

Computing momentum or volatility on a raw splice produces spikes at every roll; computing it on the wrong adjustment makes returns wrong by the carry. Carry and the curve shape are among the best documented return predictors in commodities and across futures, and they are only measurable from the contract-level data.

## How this repo uses it

`futures_bundle` turns contract tables into a `MarketBundle` with one excess-return index per instrument plus carry, curve-slope and basis-momentum signals in the macro panel, so `carry_xs`, `carry_ts` and `basis_momentum` ([cross-asset carry](cross-asset-carry.md)) run on futures through the same pipeline as on ETFs. No free source gives clean contract-level data for these markets, so the repository ships a synthetic generator (`synthetic_term_structure`) with a known Schwartz-Smith truth, and a loader contract (a long table of date, contract, expiry, price) for real vendor data.

## What we found

The tests check that the calendars follow the exchange rules, that ratio adjustment removes the roll gap exactly and its returns equal the tradable excess return, that a constant spot in contango pays the roll yield to shorts, that carry has the right sign by curve shape, and that the Kalman estimation recovers the parameters and factors of a simulated Schwartz-Smith curve. On the synthetic curves the carry signal earns what the generator builds into the risk premium, which demonstrates the machinery and not a market premium.

## Pitfalls

- Never compute returns on an unadjusted splice; the roll gap is not a return.
- Ratio adjustment fails when prices cross zero (the 2020 crude oil contract); use differences or excess returns there.
- Roll timing matters: rolling at expiry trades the illiquid, converging contract.
- Carry can be negative-skewed: backwardation earns until a supply shock reverses it.

## Try it

```python
from src.assets import futures as F

curves = F.synthetic_term_structure(n_days=600, seed=1)
cont = F.continuous_series(curves["long"], roll_days=5, adjust="ratio")
print(len(cont.roll_dates), round(float(cont.returns.std() * 252 ** 0.5), 3))
```
