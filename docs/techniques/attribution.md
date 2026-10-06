---
title: "Attribution: where did the return come from?"
slug: attribution
difficulty: 2
chapter: Ch. 22
prerequisites: [backtest-engine-and-costs]
stages: [20, 40]
files: [src/backtest/attribution.py, experiments/stage20_attribution.py, experiments/stage40_factors.py]
figures: [40, 41, 79]
tests: [tests/test_attribution.py, tests/test_factors.py]
models: []
---

# Attribution: where did the return come from?

## In one sentence

Attribution splits a return into pieces (allocation, selection, interaction, costs, factors) that add back to the total.

## The idea

Brinson-Fachler splits an active return against a benchmark into allocation (being in the right asset class), selection (choosing within it) and interaction; Carino linking makes monthly effects
sum to the cumulative figure. Risk can be split by Euler contribution. A factor regression on equity factors splits a book's variance into exposures and a residual intercept.

## Why it matters

A strategy you cannot explain is a strategy you cannot defend, and an explanation that does not add up to the total is not one.

## How this repo uses it

Stage 20 asserts every identity to 1e-9 on the live books and reconciles monthly effects with the engine; Stage 40 regresses the Generation 1 books on the Fama-French five factors plus momentum.

## What we found

Against equal weight, the differences of the risk-based books came mainly from allocation, above all the rates sleeve (EXP-053 to EXP-056). In the factor regression the momentum books loaded
positively on the momentum factor (t about 11), and the equal-weight book had a market beta of about 0.46 with R-squared 0.73.

## Going deeper

```
Brinson-Fachler per asset class c:  allocation_c = (wp_c - wb_c)(Rb_c - Rb);   selection_c = wb_c (Rp_c - Rb_c);   interaction_c = (wp_c - wb_c)(Rp_c - Rb_c)
Carino link:  scale each period's effects by  [ln(1 + Rp) - ln(1 + Rb)] / (Rp - Rb)  so the sum equals the cumulative active return
Euler risk:   component_i = w_i (Sigma w)_i / sqrt(w' Sigma w);  components sum to portfolio volatility
factor regression:  r_t - rf_t = alpha + sum_k beta_k F_k,t + e_t
```

## Pitfalls

- Equity factors do not span bonds, gold or commodities, so an intercept is not unexplained skill.
- Attribution depends on the benchmark chosen.
- Interaction terms are not zero and are often ignored.

## Try it

```bash
python -m experiments.stage20_attribution
python -m experiments.stage40_factors
```
