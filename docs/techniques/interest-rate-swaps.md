---
title: "Interest rate swaps: schedules, curves, cashflows, PV01, carry and roll-down"
slug: interest-rate-swaps
difficulty: 3
chapter: Platform
prerequisites: [contract-lifecycle, fixed-income-and-volatility-strategies]
stages: []
files: [src/swaps/schedules.py, src/swaps/curves.py, src/swaps/contracts.py, src/swaps/cashflows.py, src/swaps/pricing.py, src/swaps/risk.py, src/swaps/integration.py, src/swaps/strategies.py]
figures: []
tests: [tests/test_swaps.py]
models: []
---

# Interest rate swaps: schedules, curves, cashflows, PV01, carry and roll-down

## In one sentence

`src/swaps` prices and manages swaps from first principles: it generates the accrual schedule, builds discount and projection curves, produces every cashflow with its present value, measures PV01 by full repricing and splits expected P&L into carry and roll-down.

## The idea

A swap is two streams of cashflows. **Schedules** come from the contract: effective and maturity dates, frequency, stub rule, business-day convention, payment and reset calendars, fixing lag and day count (ACT/360, ACT/365F, 30/360, 30E/360, ACT/ACT). **Curves** are zero-rate curves with log-linear discount factors; discounting and projecting use different curves (the multi-curve world), a par-swap bootstrap builds a single curve, and a tenor-basis curve adds a spread for another floating tenor. **Cashflows**: fixed coupons are `notional x rate x accrual`; floating coupons are the forward rate of the projection curve over the period (or the published fixing once it is known), each discounted. The present value is the sum; the **par rate** sets it to zero.

**Risk by repricing.** PV01 bumps the curves by one basis point and reprices, key-rate PV01 bumps one tenor at a time, and scenarios shift or twist the curve. **Carry and roll-down** answer "what do I earn if nothing moves?": carry is the net coupon accrual at current rates, roll-down is the rest of the change in value when the curve keeps its shape in tenor space as time passes (a ten-year swap on a steep curve becomes a nine-and-a-half-year swap that prices lower). Trades sized in PV01 rather than notional (DV01-neutral curve trades, butterflies) make the risk budget explicit.

## Why it matters

Swap P&L in a backtest is easy to get wrong in boring ways: a coupon booked on the wrong day, a floating leg valued at the wrong curve, a carry number that mixes accrual with value change. Making every cashflow visible and every identity testable (par swap has zero value, pay and receive are opposites, carry plus roll-down equals the total) catches those.

## How this repo uses it

A swap is an instrument like any other (`make_irs`, `BasisSwap`, `CrossCurrencyBasisSwap`). The engine marks it from the point-in-time curves of its currency (`{ccy}-DISCOUNT`, `{ccy}-PROJ-3M`) with a bid and ask a fraction of a basis point of annuity either side of the present value, pays coupons from the schedule on their dates, and uses published fixings when available. `CurveTradeStrategy`, `CarryRolldownStrategy` and `TenorBasisStrategy` trade swaps on the common strategy API. `synthetic_rates_market` provides Nelson-Siegel curves, fixings and basis for offline tests.

## What we found

The tests check day counts and stubs, that the bootstrapped curve reprices the par swaps it was built from, that a par swap is worth zero, that pay-fixed and receive-fixed values are exact opposites, that the floating leg on one curve is worth `DF(start) - DF(end)` up to calendar alignment, that PV01 matches a bump of the curves and key-rate PV01 sums to it, that carry plus roll-down equals the total and a flat curve has far less roll-down, and that a swap traded in the engine pays one coupon on the right date and reconciles.

## Pitfalls

- Floating legs on a swap whose projection curve is missing silently fall back to the discount curve in a single-curve world; supply the projection curve.
- Roll-down depends on the convention (here: curve shape fixed in tenor space); another convention gives another split.
- Cross-currency swaps need the spot rate and exchange notionals; this is modelled but only exercised synthetically.
- Swap strategies are the slowest part of the platform: every valuation reprices cashflow tables.

## Try it

```python
import numpy as np
import pandas as pd
from src.instruments import make_irs
from src.swaps.curves import CurveSet, DiscountCurve
from src.swaps.pricing import par_rate, price
from src.swaps.risk import carry_rolldown, pv01

val = pd.Timestamp("2024-01-02")
tenors = (0.25, 1, 2, 5, 10, 30)
disc = DiscountCurve(val, tenors, tuple(0.03 + 0.01 * np.log1p(t) / 3.4 for t in tenors), name="USD-DISCOUNT")
cs = CurveSet({"USD-DISCOUNT": disc}, {"USD-PROJ-3M": disc})
swap = make_irs("USD", "2024-01-04", "2034-01-04", 0.0, 10_000_000.0)
k = par_rate(swap, cs, val)
swap = swap.replace(fixed_rate=round(k, 7))
print(round(k * 1e4, 1), "bp par;", round(price(swap, cs, val).pv, 2), "pv;", round(pv01(swap, cs, val), 1), "PV01;", {a: round(b) for a, b in carry_rolldown(swap, cs, val, 91).items() if a in ("carry", "rolldown")})
```
