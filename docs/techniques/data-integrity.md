---
title: "Clean data you can trust"
slug: data-integrity
difficulty: 1
chapter: Ch. 7
prerequisites: []
stages: [1]
files: [src/data/clean.py, src/data/validation.py, experiments/stage01_data.py]
figures: [1]
tests: [tests/test_data.py]
models: []
---

# Clean data you can trust

## In one sentence

Before any model runs, every price is checked, repaired only by declared rules, and nothing is silently deleted.

## The idea

Market data arrives with splits, missing days, bad ticks and survivorship. Cleaning it by eye does not scale and cannot be audited, so the rules are written down and run in code:
forward-fill at most a few days, flag any move over a threshold and check it against peer returns and volume, and log every change. The raw files are never edited; the
cleaned data is a function of the raw data and the rules, and carries a version hash.

## Why it matters

A backtest is a claim about the past, and the past you test on is whatever your data says it was. A single unadjusted split looks like a 50% crash and can make a strategy look brilliant
(it shorts the "crash") or terrible. Every later result inherits the data's errors.

## How this repo uses it

`src/data/clean.py` applies the repair rules; `src/data/validation.py` produces the checks and the report in `reports/data_quality_report.md`. The data version hash flows into every
registry entry, so a result always says which data produced it. The universe is 15 ETFs spanning equities, bonds, gold, silver, commodities and real estate.

## What we found

The panel passed validation with zero blocking errors, and every large move was corroborated by peer returns and volume, so none was deleted (experiment EXP-001). The known residual bias is
survivorship: the universe is ETFs that exist today.

## Going deeper

The repair rules, in order: (1) reindex to the union trading calendar; (2) forward-fill gaps of at most `max_ffill_days` (3 here), never more; (3) flag any one-day return above `jump_threshold` (20%); (4) a flagged move is kept only if
peers moved the same way or volume spiked, otherwise it is an error to be fixed or removed with a log line. The data version is a hash of the cleaned panel: if one price changes, every downstream result's `data_version` changes with it.

## Pitfalls

- Survivorship bias cannot be cleaned away; it can only be stated.
- Adjusted prices include dividends, so returns are total returns; check which series a number refers to.
- A repair rule that is tuned to make a backtest look better is a model, not cleaning.

## Try it

```bash
quant data check                       # the checks on the default panel
quant data check --prices my.csv       # the same checks on your own file
```
