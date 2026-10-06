---
title: "Power: what could the tests have found?"
slug: power-analysis
difficulty: 3
chapter: Ch. 22
prerequisites: [paired-sharpe-bootstrap, multiple-testing]
stages: [39]
files: [experiments/stage39_power.py, config/diagnostics.yaml]
figures: [77, 78]
tests: [tests/test_power.py]
models: []
---

# Power: what could the tests have found?

## In one sentence

Statistical power is the chance a test notices an effect that is really there, and without it a 'not significant' result tells you very little.

## The idea

Simulate data with a known edge, run the real test, and count how often it rejects. Repeat across edge sizes, sample lengths and tracking errors. The smallest edge detected 80% of the time is the
minimum detectable effect. At zero edge the rejection rate is the false-positive rate and should equal the stated level.

## Why it matters

Most of this repo's decisions are 'reject'. Whether that means 'no effect' or 'could not tell' depends entirely on power, and it is rarely reported.

## How this repo uses it

Stage 39 runs the harness on the platform's own paired Sharpe test (using the real equal-weight return series as the benchmark) and on the Diebold-Mariano CRPS test, and prints earlier
comparisons next to the minimum detectable differences.

## What we found

With 15.7 years of data and a 6% tracking error, the paired Sharpe test needs a true difference of about 0.35 to reach 80% power; at 3% tracking error, about 0.18. The CRPS test with 187
monthly origins needs a true out-of-sample R-squared of about 1%. The Stage 31 allocation comparisons (observed differences 0.07 and 0.19) could therefore not have been detected even if real,
while the reinforcement-learning book's shortfall of 0.54 was far outside the noise.

## Going deeper

```
for each (effect size, sample length, tracking error):
    repeat R times: simulate a benchmark and an alternative with a known edge; run the test; record reject / not
    power = rejections / R;   size = power at effect 0 (should equal alpha)
minimum detectable effect = smallest effect with power >= 0.8
```
The designed Sharpe difference is computed exactly from the population formula `(mu_b + alpha)/sqrt(var_b + te^2) - mu_b/sd_b`, so the grid's x-axis is the truth, not an estimate.

## Pitfalls

- Power depends on assumptions about the world; the grid states them.
- A significant result in an underpowered test is more likely to be an exaggerated effect.
- Do not run a power analysis after the fact on the observed effect; use plausible effects.

## Try it

```bash
python -m experiments.stage39_power
quant dashboard      # the 'What can we detect?' tab
```
