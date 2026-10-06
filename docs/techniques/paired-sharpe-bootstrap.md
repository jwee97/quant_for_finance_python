---
title: "Comparing two strategies fairly: the paired bootstrap"
slug: paired-sharpe-bootstrap
difficulty: 2
chapter: Ch. 22
prerequisites: [performance-metrics]
stages: [18, 31]
files: [src/validation/robustness.py]
figures: [43]
tests: [tests/test_robustness.py]
models: []
---

# Comparing two strategies fairly: the paired bootstrap

## In one sentence

To ask whether strategy A really has a higher Sharpe than strategy B, resample the two return streams together so their correlation is preserved.

## The idea

Two risk-based books are correlated at 0.9 or more, so the difference between their Sharpe ratios is estimated much more precisely than either Sharpe on its own. A stationary block
bootstrap resamples blocks of days (random block lengths, mean 21) from both series at the same dates, recomputes both Sharpe ratios on each resample, and uses the distribution of the
difference.

## Why it matters

Comparing two confidence intervals that overlap throws away the pairing and understates the evidence; comparing point estimates ignores uncertainty altogether.

## How this repo uses it

`paired_sharpe_test` is the comparison used for nearly every decision in the repo, with Benjamini-Hochberg control across the declared family. Stage 39 measures its size and power.

## What we found

Its size is close to the nominal 10% (between 0.10 and 0.125 across the simulated grid), and its power is modest: with 15 years of data and a 6% tracking error, a true Sharpe difference of about 0.35 is
needed for 80% power.

## Going deeper

```
repeat B times:
    draw block start dates with geometric lengths (mean 21), the SAME dates for both series
    SR_a*, SR_b* from the resampled paired returns
    d* = SR_a* - SR_b*
CI = percentiles of d*;   p = 2 * min( P(d* <= 0), P(d* >= 0) ) around the centred bootstrap distribution
```
Resampling the same dates keeps the correlation. That is the whole trick: two books correlated at 0.9 have a difference that is far less noisy than either Sharpe.

## Pitfalls

- A high-correlation pair gives a precise difference; a low-correlation pair gives a noisy one.
- Block length matters when returns are autocorrelated.
- A non-rejection is not proof of equality.

## Try it

```python
from src.validation.robustness import paired_sharpe_test
paired_sharpe_test(returns_a, returns_b, n_samples=2000, block_length=21)
```
