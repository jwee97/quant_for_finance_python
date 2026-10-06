# A tour of one backtest day

Follow one decision through the pipeline: the last trading day of a month, close of business. The code names below are real; open them as you read.

## 1. The market bundle

`load_default_bundle` (`src/framework/data.py`) returns a `MarketBundle`: adjusted prices, returns, an `investable` mask (is the asset live and tradable on this date?) and, when asked, the macro series with their publication lags. Nothing after today
is in it as far as today's decision is concerned: that is enforced by tests, not by good intentions.

## 2. Regimes (optional)

If the specification names a detector, `DETECTORS.create(...).detect(bundle)` returns a `RegimeSeries`: for each date, a probability for each named regime. Only *filtered* probabilities, which use information up to the date, are allowed
([regime-detection](techniques/regime-detection.md)).

## 3. Forecast models

Each model in `spec.models` implements `score(bundle)` and returns a table of dates by assets. The shared step `score_to_forecast` (`src/framework/forecasting.py`) calibrates the score: it regresses the realised 21-day return on the score using only
pairs whose outcome was complete on that date (*matured* pairs), clips a negative slope to zero, and multiplies. The output is a `Forecast(mean, std, confidence)` per asset, with the spread from an EWMA volatility ([probabilistic-forecasting](techniques/probabilistic-forecasting.md)).

## 4. Combination

If several models are named, `combine_forecasts` merges them by the rule in `spec.combination` (equal, confidence, precision, IC-weighted, cost-aware, regime-conditional) into one forecast per asset
([forecast-combination](techniques/forecast-combination.md)).

## 5. Portfolio construction

The allocator in `spec.allocation` turns forecasts (and regimes) into target weights: the default `forecast_stack` winsorises, z-scores, clips, scales to risk and caps; `regime_switch` blends other allocators by regime probability; `static` is one of
the classic books ([risk-parity](techniques/risk-parity.md), [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)).

## 6. Risk

A risk policy scales the book to a volatility target, constant or probability-weighted by regime.

## 7. Execution

`BacktestEngine.run` stamps the weights at the decision date and **applies them one day later**. Between rebalances the weights drift with returns. Costs are charged on what traded. If an assets-under-management figure is given, a spread-plus-impact model replaces the flat
cost ([backtest-engine-and-costs](techniques/backtest-engine-and-costs.md), [execution-and-impact](techniques/execution-and-impact.md)).

## 8. Validation

`evaluate` (`src/framework/validate.py`) runs the battery: the causality test on every model and detector, paired bootstrap comparisons against benchmarks, the deflated Sharpe ratio with the number of trials in the group, and a sample split into development,
validation and a final holdout ([walk-forward-and-leakage](techniques/walk-forward-and-leakage.md), [multiple-testing](techniques/multiple-testing.md)).

## 9. Attribution, tear sheet, database

The result carries its own tear sheet with red-flag rules, and the experiment manager stores the run so it can be compared and counted ([attribution](techniques/attribution.md),
[experiment-database-and-reproducibility](techniques/experiment-database-and-reproducibility.md)).

## Try the whole chain in one command

```bash
quant backtest --model dual_momentum --regime vol_state --tearsheet
```

Open `reports/tearsheets/<run id>/tearsheet.md` and read the red-flag list first.
