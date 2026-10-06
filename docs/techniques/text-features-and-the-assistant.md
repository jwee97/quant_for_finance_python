---
title: "Text as data: FOMC statements and the research assistant"
slug: text-features-and-the-assistant
difficulty: 3
chapter: Ch. 23
prerequisites: [macro-and-alternative-data]
stages: [28]
files: [src/assistant/documents.py, src/assistant/extraction.py, src/assistant/sqlguard.py, experiments/stage28_text.py, src/cli.py]
figures: [56, 57]
tests: [tests/test_assistant.py, tests/test_cli_tools.py]
models: []
---

# Text as data: FOMC statements and the research assistant

## In one sentence

Turn documents into structured numbers (tone, change, action) with a method you can audit, and keep language models away from predicting prices.

## The idea

FOMC statements are downloaded once, stamped with the date they became available, and scored with an offline lexicon: hawkish and dovish term counts give a tone; text difference from the previous statement gives
a change score; the announced rate action is parsed. An optional language-model backend can extract structured fields, with a fixed schema, a cache and the source text always kept.

## Why it matters

Text carries information numbers do not, but it is easy to leak (the statement time vs the return date) and easy to over-claim (language models hallucinate). Using them for extraction, not prediction, keeps the claims checkable.

## How this repo uses it

Stage 28 builds the corpus (171 statements) and tests the features against price and macro forecasts. The assistant's SQL tool is read-only and guarded by `sqlguard.py`. No language model was called in any result in this repository: the
environment has no API credentials, and the offline lexicon is the only backend that was run.

`quant ask "which hypotheses were retained?"` answers recognised questions from the research database with templates and refuses anything else. `--llm MODEL` lets a model write the SQL (validated read-only like any other statement); that path needs the optional `anthropic` package and an API key and has not been exercised against a live model in this repository.

## What we found

Tone, change and action did not add out-of-sample information for monthly sleeve returns after correction (EXP-074). A post-hoc matched-sample control is reported separately and does not change the decision.

## Going deeper

```
tone   = (hawkish term count - dovish term count) per 1,000 words        (offline lexicon)
change = 1 - cosine similarity( tf-idf(statement_t), tf-idf(statement_{t-1}) )
action = -1, 0 or +1: the target-rate decision announced
available_at = statement date + 1 calendar day                              (point-in-time stamp for the forecast stages)
```
For the causal application, the statement day itself is used (the release is 2pm US Eastern, before the close), while the forecasting stages use the next day to be conservative.

## Pitfalls

- A dictionary tone score is crude (and the lexicon was written for a different corpus).
- Statements come out mid-afternoon; same-day versus next-day returns is a modelling choice.
- Always keep the source text next to the extracted fields.

## Try it

```bash
python -m experiments.stage28_text
```
