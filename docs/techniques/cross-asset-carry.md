---
title: "Cross-asset carry, basis momentum and long-term reversal"
slug: cross-asset-carry
difficulty: 3
chapter: Ch. 19
prerequisites: [futures-and-commodity-curves, fx-carry-and-momentum]
stages: []
files: [src/strategies/multi_asset.py, src/assets/bundles.py, src/assets/futures.py]
figures: []
tests: [tests/test_assets.py]
models: [carry_xs, carry_ts, basis_momentum, long_term_reversal]
---

# Cross-asset carry, basis momentum and long-term reversal

## In one sentence

Carry, the return an instrument earns if prices do not move, has predicted returns in every liquid asset class; these four models rank or time instruments by carry, by the shape of the futures curve, or by five-year reversal, within each asset class.

## The idea

Koijen, Moskowitz, Pedersen and Vrugt (2018) define **carry** for any instrument as its expected return if its price is unchanged: the roll yield for a future, the interest differential for a currency, the yield over cash for a bond, the dividend yield for an equity index. **`carry_xs`** divides each instrument's carry by its volatility, z-scores that inside its asset class (so the signal is not dominated by the class with the largest carries) and goes long the high-carry instruments against the low-carry ones. **`carry_ts`** is the time-series version: long an instrument while its carry is positive, short while negative, sized by carry over volatility. **`basis_momentum`** (Boons and Prado 2019) ranks commodity futures by the trailing return difference between the first and second contracts, which carries information about the curve beyond its slope. **`long_term_reversal`** is a value proxy (Asness, Moskowitz and Pedersen 2013): within each asset class, buy what has fallen over five years and sell what has risen.

## Why it matters

These strategies diversify trend and equity risk and are available in markets that the platform's ETFs only approximate. They also test the multi-asset architecture: one signal definition, applied through the same pipeline to commodities, currencies, bonds and equity indices, with the model blind to which is which except through the asset-class label.

## How this repo uses it

The models read per-instrument signals from the macro panel: series named `CARRY_<asset>` and `BASISMOM_<asset>`, built by `src.assets.bundles` from contract tables (futures), spot and rate panels (FX) and yield curves (bonds). On a bundle without the signals the models raise a clear error rather than inventing one; `long_term_reversal` needs only prices. `multi_asset_demo_bundle` provides a fully synthetic cross-asset universe so everything runs offline.

## What we found

On the platform's ETFs the carry models cannot run (there is no carry series for ETFs; the survey lists them as not runnable) and `long_term_reversal` earned a net Sharpe of -0.37 against +0.74 for equal weight over the same dates ([the survey](../strategy_survey.md), exploratory). On the **simulated** universe of [the institutional findings](../institutional_findings.md), mean net Sharpe over three draws was +0.56 for `carry_xs` and +0.14 for `carry_ts`, with a wide spread across draws (from 0.04 to 0.96 for `carry_xs`). That carry works in that table is true by construction, because the FX and commodity simulations pay a carry premium; the table shows that the pipeline recovers a built-in premium and that three draws are too few to rank the variants. No claim about real markets follows. The tests also check that the models are causal, forecast, and make money when carry is the return, and that they refuse bundles without signals.

## Pitfalls

- Carry earns a premium for crash risk; the average masks rare, large losses.
- Within-class standardisation needs at least two live instruments per class; small classes drop out.
- A long-term reversal signal needs five years of history: the universe's early dates are empty.
- Real contract-level data are needed for the futures signals; continuous series from a data vendor rarely carry the second contract.

## Try it

```python
from src.assets.bundles import multi_asset_demo_bundle
from src.framework import Pipeline, PipelineSpec, load_library
from src.utils.config import load_config

load_library()
bundle, _ = multi_asset_demo_bundle(n_days=1500, seed=1)
spec = PipelineSpec.from_dict({"name": "carry_xs", "models": [{"name": "carry_xs"}], "allocation": {"allocator": "score_stack"}})
result = Pipeline(spec, load_config(), bundle).run()
print(round(result.metrics["sharpe"], 2), len(result.net_returns.dropna()))
```
