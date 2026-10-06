---
title: "Cross-sectional factors: low volatility, value, quality, carry, defensive beta"
slug: cross-sectional-factors
difficulty: 2
chapter: Ch. 22
prerequisites: [momentum, information-coefficient]
stages: [30]
files: [src/strategies/cross_sectional.py, src/strategies/_common.py, experiments/stage30_library.py]
figures: [59, 60]
tests: [tests/test_strategies.py]
models: [carry, defensive_beta, low_volatility, quality_proxy, value_proxy]
---

# Cross-sectional factors: low volatility, value, quality, carry, defensive beta

## In one sentence

A factor is a characteristic that ranks assets by expected return, and these strategies go long the top of a ranking and short (or underweight) the bottom.

## The idea

Low volatility favours assets with the lowest recent volatility; defensive beta (betting against beta) favours assets with the lowest market beta; carry favours the bond ETFs that yield the most; value favours assets that
have fallen furthest over a long horizon; quality favours assets with smooth, steady price paths. In equities these come from accounting data; this repo has only ETF prices and rates, so value and quality are *price-based proxies* and
are labelled that way everywhere.

## Why it matters

Factors are how institutions describe and diversify their exposures. The honest question for each is whether it adds anything to a market portfolio on this particular universe.

## How this repo uses it

`src/strategies/cross_sectional.py` builds the scores; the shared stack in `src/strategies/_common.py` turns each score into positions in the same way for every model, so differences in results are due to the score, not to
different handling. Because the universe is 15 multi-asset ETFs, a cross-sectional ranking mixes asset classes: low volatility mostly means "own bonds".

## What we found

The carry model earned a small positive net Sharpe in the library, low volatility and defensive beta behaved like long-bond tilts, and the value proxy (long-horizon reversal) had a small Sharpe at a very low volatility. None survived the
search-aware tests against passive equal weight (Stage 30).

## Going deeper

```
low volatility:  score_i = - std of daily returns over 252 days
defensive beta:  score_i = - beta_i,   beta_i = cov(r_i, r_market proxy) / var(r_market proxy) over 252 days
carry (bonds):   score_i = yield of the matching Treasury (2y, 5y, 10y, 20y) - fed funds rate, for SHY, AGG, IEF, TLT
value proxy:     score_i = - ( P_i,t / P_i,t-1260 - 1 )          reversal over five years
quality proxy:   score_i = R-squared of log price on time over 252 days * sign of the one-year change      (smooth gainers)
positions:       rank -> z-score -> clip -> scale to a volatility target -> cap per asset   (the shared stack)
```

## Pitfalls

- A rank across asset classes is dominated by the class: low volatility is not a stock factor here.
- Proxies are not the factors; do not cite these results as evidence about value or quality in equities.
- Factor returns come in long, painful droughts.

## Try it

```bash
quant backtest --model low_volatility --tearsheet
quant backtest --model carry
```
