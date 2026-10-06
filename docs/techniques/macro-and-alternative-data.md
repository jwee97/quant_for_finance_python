---
title: "Macro and alternative data without look-ahead"
slug: macro-and-alternative-data
difficulty: 3
chapter: Ch. 20
prerequisites: [walk-forward-and-leakage]
stages: [15, 23, 30]
files: [src/data/macro.py, src/data/altdata.py, src/features/macro.py, experiments/stage15_macro.py, experiments/stage23_altdata.py, src/strategies/macro.py]
figures: [28, 29, 46, 47]
tests: [tests/test_macro.py, tests/test_macro_forecast.py, tests/test_altdata.py, tests/test_positioning_explain.py]
models: [yield_curve_regime, risk_on_off, inflation_rotation, dollar_strength, commodity_supercycle, cftc_positioning]
---

# Macro and alternative data without look-ahead

## In one sentence

Macro series are published late and revised, so using them correctly means using only what was known on each date.

## The idea

Each macro observation has a date it refers to and a date it became public (the publication lag). A point-in-time panel uses a value only from its release date on. Alternative data here means series institutional
researchers use: credit spreads, options-implied volatility, CFTC positioning, jobless claims. Features are standardised with trailing statistics and tested for whether they add information beyond price.

## Why it matters

Ignoring release dates makes macro features look predictive when they are not. The Clark-West test asks whether a larger model's forecast improvement is real when it nests a smaller one.

## How this repo uses it

`src/data/macro.py` and `src/features/macro.py` build the lagged panel; Stage 15 and Stage 23 test macro and non-price features against price-only forecasts; five macro strategies (yield-curve regime, risk-on/off, inflation rotation, dollar strength,
commodity supercycle) plug into the framework.

The default bundle also carries the six CFTC positioning series (`cftc_es`, `cftc_nq`, `cftc_ust10`, `cftc_gold`, `cftc_silver`, `cftc_crude`) after their four-day release lag when the Stage 23 table has been downloaded, and the `cftc_positioning` model turns each into a contrarian (or, with `sign: 1`, trend-following) score for the matching ETF. Stage 23 found no predictive power, so the model is a template and a null result, not an edge.

## What we found

Macro features did not improve out-of-sample monthly forecasts of asset-class returns (EXP-024), and credit, implied-volatility, positioning and jobless-claims features added nothing after correction (EXP-064, EXP-065). The publication-lag control
showed that ignoring lags inflated apparent skill only slightly here (EXP-025).

## Going deeper

```
value used at date t for series s:  the latest observation with release_date <= t     (point in time)
standardised feature:  (x_t - trailing mean) / trailing std,  window >= 756 days
Clark-West:  f_t = (y - yhat_restricted)^2 - (y - yhat_full)^2 + (yhat_restricted - yhat_full)^2;   t-test on mean(f_t)
```
Clark-West corrects the bias that nested models create: the larger model estimates parameters that are truly zero and pays a noise penalty the plain test ignores.

## Pitfalls

- Revised data is not the data you had.
- Dozens of macro series with a few hundred months invite data mining.
- Standardising over the full sample leaks.

## Try it

```bash
python -m experiments.stage15_macro
quant backtest --model yield_curve_regime
```
