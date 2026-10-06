---
title: "Causal inference: from correlation to cause"
slug: causal-inference
difficulty: 3
chapter: Ch. 23
prerequisites: [multiple-testing]
stages: [37]
files: [src/causal/estimators.py, src/causal/simulate.py, experiments/stage37_causal.py]
figures: [73, 74]
tests: [tests/test_causal.py]
models: []
---

# Causal inference: from correlation to cause

## In one sentence

Correlation says two things move together; causal methods try to say what would happen to one if you changed the other.

## The idea

Double machine learning removes the part of treatment and outcome explained by controls with flexible models, cross-fitted so the same data never fits and scores a model, then regresses residuals on residuals.
The R-learner uses the same residuals to estimate how the effect varies. Instrumental variables use something that moves the treatment but affects the outcome only through it. Difference in differences compares
the change in a treated group with the change in a control group, assuming they would have moved in parallel.

## Why it matters

Every backtest signal is a correlation. Strategies built on spurious relationships fail when the relationship was never causal, and the estimators here are the standard tools for asking.

## How this repo uses it

Stage 37 first validates each estimator on simulated data with a known effect, next to the naive estimate, then applies double machine learning to FOMC statement tone and same-day ETF returns. IV is
validated but not applied (no credible instrument); difference in differences is applied to one announcement-day design only.

## What we found

2SLS and difference in differences recovered the truth when their assumptions held and difference in differences was biased when trends diverged. Double machine learning with the declared learners failed the
validation at n = 500 (bias 0.58 on a true effect of 0.5, learner-limited: oracle nuisance functions gave no bias). In the application the effect on TLT was negative and the effect on SPY positive, both marginal (p about 0.05)
and not robust to the choice of learner, so the declared rule retained the hypothesis while the report calls it fragile (declared hypothesis h_tone).

## Going deeper

```
DML (partially linear): y = theta d + g(X) + e,  d = m(X) + v
   r_y = y - E^[y|X],  r_d = d - E^[d|X]  (cross-fitted);   theta^ = sum r_d r_y / sum r_d^2
2SLS:  d^ = Z pi^ ;  theta^ = (d^'y) / (d^'d)  with first-stage F as a strength check
DiD:   theta^ = (Ybar_treated,post - Ybar_treated,pre) - (Ybar_control,post - Ybar_control,pre)
```
The estimate is only as good as the two nuisance regressions: if `E^[y|X]` and `E^[d|X]` miss the same structure, the residual correlation survives as bias, which is what the n = 500 simulation showed.

## Pitfalls

- Double machine learning is only as good as the nuisance models.
- Unconfoundedness is an assumption, not a result.
- With 168 statements, any estimate is weak.

## Try it

```bash
python -m experiments.stage37_causal
```
