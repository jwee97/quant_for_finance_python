---
title: "Yield curves, bond analytics and carry with roll-down"
slug: yield-curves-and-fixed-income
difficulty: 2
chapter: Ch. 19
prerequisites: [fixed-income-and-volatility-strategies]
stages: []
files: [src/assets/rates.py, src/assets/bundles.py]
figures: []
tests: [tests/test_assets.py]
models: []
---

# Yield curves, bond analytics and carry with roll-down

## In one sentence

A bond's price, risk and expected return all follow from the yield curve; `src.assets.rates` fits the curve with Nelson-Siegel, forecasts it with Diebold-Li, computes duration, convexity and key-rate durations, and measures the carry and roll-down of a position.

## The idea

The **Nelson-Siegel** curve `y(m) = b0 + b1 (1 - e^{-lm}) / (lm) + b2 ((1 - e^{-lm}) / (lm) - e^{-lm})` summarises a whole curve in three numbers that have names: `b0` is the level (the long yield), `b1` the slope (short minus long, with the sign convention of the formula) and `b2` the curvature (`fit_nelson_siegel` fits them by least squares over a grid of decay parameters `l`, and `svensson` adds a second hump). **Dynamic Nelson-Siegel** (Diebold and Li 2006, `dynamic_nelson_siegel`) fits the curve every day and treats the three betas as a VAR(1), so `forecast_curve` forecasts the whole curve. `bootstrap_zero_curve` converts par yields into zero rates and `forward_rate` gives the forward between two maturities.

For a fixed-coupon bond, `bond_price` discounts the cash flows (`cashflow_schedule` lists them, `bond_price_from_curve` discounts on a zero curve), `yield_to_maturity` inverts the price, and the risk measures are `duration` (Macaulay or modified), `convexity`, `dv01` (the price change for one basis point) and `key_rate_durations`, which bump each point of the zero curve in turn so that a hedge can match the curve's shape and not only its parallel moves.

`carry_rolldown` is the expected return of holding the bond for `horizon` years if the curve does not move: the coupon accrued net of financing (carry) plus the price gain from rolling down a steep curve to a lower yield (roll-down).

## Why it matters

Bonds are the diversifier in most multi-asset portfolios, and their risk is rate risk: a 10-year note and a 2-year note have very different durations, so equal dollar positions are very different bets. Carry and roll-down are the predictable part of bond returns and the basis of the fixed-income strategies in the library.

## How this repo uses it

`multi_asset_demo_bundle` builds constant-maturity bond indices from a simulated dynamic Nelson-Siegel curve, with carry as the signal, so `carry_xs` and `carry_ts` run on bonds next to commodities, FX and equities. For real data, the platform's Treasury ETFs and any FRED yield panel work with the same functions: the loader contract is a table of yields by maturity per date.

## What we found

The tests check that the analytics are internally consistent (a par bond prices at 100, the yield-to-maturity inverts the price, duration equals the numerical derivative of price in yield, Macaulay duration of a zero-coupon bond is its maturity, convexity matches the second difference, dv01 matches duration), that the zero-curve bootstrap reprices the par bonds and the forwards are consistent, that Nelson-Siegel recovers the parameters of a generated curve and the dynamic version recovers its factors, and that carry and roll-down depend on the slope of the curve in the expected direction.

## Pitfalls

- A parallel shift is not the only risk: two portfolios with equal duration can have very different key-rate exposures.
- Carry assumes an unchanged curve; the roll-down gain disappears if the curve shifts up by the amount it implies.
- Nelson-Siegel is smooth by construction and can miss kinks at the short end; do not fit to bills with it alone.
- Constant-maturity indices are not tradable securities; real portfolios hold bonds that age.

## Try it

```python
from src.assets import rates

print(round(rates.bond_price(10, 0.04, 0.05), 3), round(rates.duration(10, 0.04, 0.05), 3), round(rates.dv01(10, 0.04, 0.05), 4))
curve = rates.synthetic_curve_panel(n_days=300, seed=1)
fit = rates.fit_nelson_siegel(curve.columns.astype(float), curve.iloc[-1].values)
print({k: round(float(v), 3) for k, v in fit.items()})
```
