---
title: "Your own strategies: formulas and Python files"
slug: custom-strategies
difficulty: 2
chapter: Ch. 22
prerequisites: [plugin-framework]
stages: []
files: [src/strategies/expression.py, src/strategies/user.py, src/webapp/api.py]
figures: []
tests: [tests/test_custom_strategies.py, tests/test_webapp.py]
models: [expression]
---

# Your own strategies: formulas and Python files

## In one sentence

A strategy is a rule that gives every asset a score each day; you can write that rule as a one-line formula or as a small Python class, and either way it is tested with the same costs, benchmarks and look-ahead checks as the built-in strategies.

## The idea

A score is a number per asset per day where higher means "expect a higher return". Momentum is last year's return, mean reversion is minus last week's move, a trend filter is one when the price is above its 200-day average. The `expression` model evaluates such a formula over the price table: `rank(mom(126, 21)) - rank(vol(63))` is "rank by medium-term momentum, subtract the rank of volatility", so it prefers rising assets that are calm. The formula is parsed, never executed, and only backward-looking functions exist, so a formula cannot see the future whatever it says. Anything more elaborate is a Python file in `user_strategies/`, written from a template that `quant new-strategy` creates.

## Why it matters

Most ideas do not need infrastructure; they need an honest test. What makes a test honest is not the formula but the machinery around it: costs charged on every trade, signals acting the next day, comparison with equal weight and risk parity, and a deflated Sharpe ratio that counts how many variants you tried. A formula that looks brilliant after twenty tries is usually luck, and the trial counter makes that visible.

## How this repo uses it

`src/strategies/expression.py` holds the parser and the registered `expression` model; `src/strategies/user.py` loads `user_strategies/*.py` and writes the template. The dashboard (`quant serve`) offers both routes: a formula editor with a function reference and a preview of today's scores, and a Reload button for Python files. Formulas are traded as written by default (the `score_stack` allocator: rank and scale the score, no calibration), because a calibrated forecast would hold nothing for a fade that has not paid in the past.

## What we found

Nothing about whether any particular formula works: that is the point of testing yours. What the tests establish is the machinery: every formula in the shipped examples passes the look-ahead check, hostile input (attribute access, indexing, lambdas, comprehensions, windows out of range, wrong argument types) is refused with a reason, and a template produced by `quant new-strategy` is valid, registers, reloads after an edit and is causal.

## Going deeper

```
score(asset, day) = f(prices up to and including day)      # only trailing windows: rolling(n), shift(k >= 0)
cross-sectional:  position proportional to rank/z-score of the score across assets each day
time series:      position proportional to the score of each asset alone, scaled to a volatility target
```
The parser accepts numbers, quoted series names, the operators `+ - * / ** & |` and unary `- + ~`, comparisons, and calls of the whitelisted functions with positional arguments; it caps the formula at 400 characters and 120 nodes, windows at 1000 days and exponents at 4. Cross-sectional functions (`rank`, `zs`, `demean`) use only the same day's values across assets.

## Pitfalls

- A formula that works on the 15 ETFs may only be an equity-beta bet; look at the attribution.
- Every variant is a trial. Ten formulas tried on the same tickers raise the bar for all of them.
- `where(close > sma(200), 1, -1)` trades often when the price hovers around the average; check turnover and the cost drag.
- User Python files run with your permissions; the browser never sends code.

## Try it

```bash
quant serve                                        # Strategy builder tab
quant backtest --model expression --param "expr=mom(252, 21)" --allocator score_stack --tearsheet
quant new-strategy my_idea                         # then edit user_strategies/my_idea.py
```
