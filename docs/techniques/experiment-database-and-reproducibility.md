---
title: "The research database and reproducibility"
slug: experiment-database-and-reproducibility
difficulty: 2
chapter: Ch. 7
prerequisites: []
stages: [29]
files: [src/research_db/builder.py, src/framework/experiments.py, experiments/stage29_research_db.py, src/utils/config.py]
figures: [58]
tests: [tests/test_research_db.py, tests/test_config_identity.py]
models: []
---

# The research database and reproducibility

## In one sentence

Every experiment is stored with its hypothesis, data version, configuration fingerprint and decision, so results can be queried and reproduced.

## The idea

A registry entry holds the hypothesis in plain language, parameters, results, the decision (retain, reject, investigate or record) and notes. The configuration fingerprint is a hash of the declared configuration, and the data version a hash
of the cleaned data. Declaring the rule in committed configuration before running is what makes a decision pre-registered rather than post hoc.

## Why it matters

Without a ledger of everything tried, including failures, no one can tell how much of the result was luck. A research record that only contains successes is marketing.

## How this repo uses it

`quant sql "SELECT ..."` queries the SQLite database rebuilt from the files in Stage 29; `quant leaderboard` queries your own experiment runs; the dashboard shows the whole ledger. Configuration fingerprints for each generation are pinned by a test.

## What we found

The ledger lists every hypothesis with its decision; most were rejected, and that is by design.

## Going deeper

```
config fingerprint = hash of the sorted, canonical YAML of the declared namespaces in a scope
data version       = hash of the cleaned panel
run id             = hash of (specification, data version, config fingerprint)       -> a rerun of the same thing returns the stored result
n_trials(group)    = number of distinct specification hashes in the group           -> input to the deflated Sharpe ratio
```
Pre-registration is just ordering: the configuration with the decision rule is committed before the stage that uses it is run, and the commit history shows it.

## Pitfalls

- A fingerprint changes when any declared value changes; that is the point.
- The database is derived and rebuilt; the files are the record.
- Post-hoc analyses are labelled and cannot overturn a declared decision.

## Try it

```bash
quant sql "SELECT decision, COUNT(*) FROM experiments GROUP BY decision"
quant dashboard
```
