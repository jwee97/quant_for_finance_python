---
title: "Hierarchical risk parity (HRP) and HERC"
slug: hierarchical-risk-parity
difficulty: 3
chapter: Ch. 19
prerequisites: [risk-parity, pca-effective-rank]
stages: [18]
files: [src/portfolio/hierarchical.py, experiments/stage18_hierarchical.py]
figures: [36, 37]
tests: [tests/test_hierarchical.py]
models: []
---

# Hierarchical risk parity (HRP) and HERC

## In one sentence

Cluster the assets by correlation and split risk down the tree, so that similar assets share a budget and no matrix inversion is needed.

## The idea

HRP orders assets by a hierarchical clustering of the correlation matrix and allocates by recursive bisection, giving each cluster weight inversely to its variance. HERC goes further and equalises
risk contribution across clusters. Because it never inverts the covariance matrix, it is robust to the near-singularity that wrecks mean-variance.

## Why it matters

It was proposed as a cure for the instability of optimisers. Whether it cures it is an empirical question this repo tests with a rule fixed in advance.

## How this repo uses it

`src/portfolio/hierarchical.py` implements both; Stage 18 compares weight stability and net Sharpe with minimum variance and risk parity.

## What we found

On this 15-ETF universe the hierarchical methods were not more stable than minimum variance by the pre-declared 75%-of-windows rule (EXP-043) and did not deliver a better Sharpe than risk parity (EXP-044).

## Going deeper

```
1. distance d_ij = sqrt( (1 - corr_ij)/2 );  cluster with single/ward linkage
2. quasi-diagonalise: reorder assets so similar ones are adjacent
3. recursive bisection: split the ordered list in half; cluster variance V_c = w_c' Sigma_c w_c with inverse-variance w_c;
   alpha = 1 - V_left/(V_left + V_right);  scale left weights by alpha, right by 1 - alpha;  recurse
```
No matrix inversion appears anywhere, which is the point.

## Pitfalls

- With 15 assets there is little hierarchy to find.
- Linkage method and distance metric change the tree.
- The advantage reported in papers is often on large universes.

## Try it

```bash
python -m experiments.stage18_hierarchical
```
