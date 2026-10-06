---
title: "Value at risk, expected shortfall and how to test them"
slug: var-cvar-and-backtests
difficulty: 2
chapter: Ch. 21
prerequisites: [performance-metrics]
stages: [9, 36]
files: [src/risk/var.py, src/risk/cvar.py, src/models/diffusion.py, experiments/stage09_risk.py]
figures: [19, 71]
tests: [tests/test_risk.py, tests/test_diffusion.py]
models: []
---

# Value at risk, expected shortfall and how to test them

## In one sentence

VaR is a quantile of the loss distribution, and the only honest way to judge it is to count how often reality breaches it.

## The idea

A 5% one-month VaR of 6% says: in 95 months out of 100 you lose less than 6%. Methods differ in how they estimate the quantile: normal (variance only), historical simulation (the
empirical distribution), filtered historical (rescaled by current volatility) or model-based scenarios. Kupiec's test asks whether the breach rate equals the promised rate; the pinball
loss scores the quantile itself.

## Why it matters

Risk limits, margin and regulatory capital are built on these numbers. A VaR model that breaches twice as often as promised is not conservative or aggressive, it is wrong.

## How this repo uses it

`src/risk/var.py` implements the estimators; Stage 9 backtests them with coverage and independence tests; Stage 36 adds diffusion-generated scenarios, scored by pinball loss and Kupiec
against the Gaussian and the empirical distribution.

## What we found

Daily ETF returns are far from normal, so historical simulation is the default method in Stage 9. Every daily 95% VaR method there has a breach rate close to the promised one (5.3% to 5.8%), but the breaches are clustered in time, which the independence test rejects, and every method fails at 99% (historical simulation breaches 1.46% of days against 1% promised); EXP-013 is the most useful negative result in the project. In Stage 36 the diffusion quantile had the best Kupiec behaviour but no significantly
lower pinball loss than the empirical distribution (see the Generation 5 report).

## Going deeper

```
VaR_a  = - quantile_(1-a)( portfolio return )
Kupiec: LR = -2 ln[ (1-p)^(n-x) p^x ] + 2 ln[ (1-x/n)^(n-x) (x/n)^x ]  ~ chi2(1)    x breaches in n days, promised rate p
pinball loss_tau(y, q) = (tau - 1[y < q]) * (y - q)
```
Christoffersen's independence test asks whether a breach today makes a breach tomorrow more likely. A model can pass Kupiec (right number of breaches) and fail independence (breaches clustered in crises).

## Pitfalls

- At 1% with about 190 monthly observations you expect two breaches; the test has almost no power.
- Overlapping horizons make breaches cluster.
- A good unconditional breach rate can hide clustering in crises.

## Try it

```bash
python -m experiments.stage09_risk
python -m experiments.stage36_diffusion
```
