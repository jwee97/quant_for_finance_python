---
title: "How many independent bets do I own? (PCA)"
slug: pca-effective-rank
difficulty: 2
chapter: Ch. 8
prerequisites: [data-integrity]
stages: [2]
files: [src/features/pca.py, experiments/stage02_eda.py]
figures: [5, 7]
tests: [tests/test_returns.py]
models: []
---

# How many independent bets do I own? (PCA)

## In one sentence

Principal component analysis rotates correlated returns into uncorrelated components, and the number of components that matter is how many independent risks a portfolio really holds.

## The idea

The covariance matrix of 15 ETFs has 15 eigenvalues. If one eigenvalue is huge, most of the variance is a single shared factor (here, mostly equity versus duration). The effective rank,
the exponential of the entropy of the normalised eigenvalues, turns the spectrum into a single "number of independent bets". Random-matrix theory (Marchenko-Pastur) tells you which
eigenvalues are distinguishable from noise given your sample length.

## Why it matters

Owning 15 assets is not 15 bets. Optimisers that treat the sample covariance as truth over-allocate to the tiny eigenvalues, which are mostly noise. Knowing the true dimensionality is the
argument for shrinkage, factor models and risk parity before you optimise.

## How this repo uses it

`src/features/pca.py` computes the spectrum, the effective rank and the noise bound. Stage 2 reports them overall and on rolling windows; the PCA residual stat-arb strategy later reuses the
same decomposition.

## What we found

The first component explains about 41% of variance and the first three about 77%; the effective rank is roughly six, and falls to about four in the worst rolling window, so diversification
shrinks in crises (EXP-002). Only a handful of eigenvalues clear the Marchenko-Pastur bound.

## Going deeper

```
C = V diag(l_1..l_N) V'              eigen-decomposition of the correlation matrix
p_i = l_i / sum(l)                   share of variance in component i
effective rank = exp( - sum p_i ln p_i )
```
If all 15 components were equal the effective rank would be 15; if one component carried everything it would be 1. The Marchenko-Pastur upper bound for N assets and T days is `(1 + sqrt(N/T))^2` times the average eigenvalue: components below it are
indistinguishable from the spectrum of pure noise.

## Pitfalls

- PCA on returns depends on scale: use correlations or standardised returns, not raw returns across asset classes of very different volatility.
- Components are statistical, not economic; naming PC1 "equity" is an interpretation.
- Rolling windows trade noise against responsiveness.

## Try it

```bash
python -m experiments.stage02_eda
quant explain effective rank
```
Open figure 7 and ask how far the cumulative-variance curve is from the straight line of a 15-bet universe.
