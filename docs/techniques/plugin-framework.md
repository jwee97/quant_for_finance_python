---
title: "The plugin framework: every strategy is a forecast model"
slug: plugin-framework
difficulty: 2
chapter: Ch. 22
prerequisites: [probabilistic-forecasting]
stages: [30, 31]
files: [src/framework/pipeline.py, src/framework/registry.py, src/framework/types.py, src/framework/forecasting.py]
figures: []
tests: [tests/test_framework.py]
models: []
---

# The plugin framework: every strategy is a forecast model

## In one sentence

Market data flows through regime detection, forecast models, confidence, combination, allocation, risk and validation, and adding a strategy is writing one forecast model that plugs in.

## The idea

A forecast model turns a market bundle into a score; a shared calibration step converts the score into a `Forecast(mean, std, confidence)`; a combiner merges forecasts; an allocator turns forecasts and regimes into weights; a
risk policy scales them; the Generation 1 engine backtests them; validation runs the causality, benchmark and deflated-Sharpe checks. Models, regime detectors and allocators register by name with decorators.

## Why it matters

Capabilities, not strategies, are what a platform is made of. Each stage can be improved or swapped without touching the others, and every strategy gets the same honest treatment.

## How this repo uses it

The registry in `src/framework/registry.py` powers `quant list models`. `src/framework/pipeline.py` runs the whole chain from a specification (a YAML or a Python dict), and `src/framework/experiments.py` records the run in a persistent
database so that the number of specifications tried is known.

## What we found

Thirty-four models, five regime detectors and six allocators are registered, and every model passes the causality test.

## Going deeper

```python
@MODELS.register("my_model", family="my family", description="what it bets on")
class MyModel(ForecastModel):
    def score(self, bundle):            # a DataFrame, dates x assets; uses only information up to each date
        return bundle.prices.pct_change(63)
```
Everything else (calibration into a forecast, combination, allocation, backtest, causality test, tear sheet) is provided by the pipeline.

## Pitfalls

- A model that quietly reads global state breaks causality; the test catches this.
- A new model needs a cost assumption, not only a signal.
- Calibration is fitted on matured outcomes only.

## Try it

See `docs/how_to_add_a_strategy.md` for a worked example.
