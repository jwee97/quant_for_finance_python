# Start here

This repository is a research platform for systematic multi-asset investing, and also a teaching tool. It is built around one idea that is easy to say and hard to practise:

> **Most backtests that look good are not telling you the truth. The job is to find out which ones are.**

Everything here (the strategies, the tests, the reports that mostly say "this did not work") is organised around that.

## Fifteen minutes

```bash
pip install -e .                  # installs the `quant` command
quant demo                        # a five-minute tour on the cached data: one strategy, its tear sheet, a regime view, an honest verdict
quant dashboard                   # writes reports/dashboard.html: open it in a browser
quant explain deflated sharpe     # plain-language explanations for any term or experiment
```

Then open `reports/dashboard.html` and look at the **Decision ledger**: every hypothesis the project ever declared, with the rule that decided it. Most are rejections. That is the point.

## Choose your path

**You know almost no finance.** Read the [glossary](glossary.md) entries for *return, volatility, Sharpe ratio, drawdown, backtest*, then the guides in this order:
[data-integrity](techniques/data-integrity.md), [performance-metrics](techniques/performance-metrics.md), [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md), [momentum](techniques/momentum.md),
[information-coefficient](techniques/information-coefficient.md), [multiple-testing](techniques/multiple-testing.md). Run `quant backtest --model momentum --tearsheet` after each one.

**You know finance but not code.** Read [the tour of a backtest day](tour_of_a_backtest_day.md), then [how to add a strategy](how_to_add_a_strategy.md), and change one number in the example.

**You know machine learning but not finance.** The difference is signal-to-noise: monthly asset returns are almost pure noise, samples are tiny, and every "feature" you try is a trial. Read [multiple-testing](techniques/multiple-testing.md),
[power-analysis](techniques/power-analysis.md), [probabilistic-forecasting](techniques/probabilistic-forecasting.md), then [deep-time-series-models](techniques/deep-time-series-models.md) to see why a model that wins in other domains has to fight for a tie here.

**You want to backtest futures, FX, crypto, options or swaps in one portfolio.** Read the [architecture](architecture.md#the-multi-asset-engine-one-engine-one-ledger-one-strategy-interface), then [the instrument model](techniques/unified-instrument-model.md), [the engine and ledger](techniques/event-driven-engine-and-ledger.md), [contract lifecycle](techniques/contract-lifecycle.md) and [the strategy API](techniques/multi-asset-strategy-api.md), and run `from src.engine.demo import run_mixed_asset_demo; print(run_mixed_asset_demo(200).report())`.

**You are interviewing for a quant role.** Be able to explain, from this repository, [attribution](techniques/attribution.md), [multiple-testing](techniques/multiple-testing.md), [execution-and-impact](techniques/execution-and-impact.md) and
[regime-adaptive-allocation](techniques/regime-adaptive-allocation.md), and to say what each result does and does not show.

## How the platform is organised

```
Market data -> Feature factory -> Regime detection -> Forecast models -> Forecast confidence -> Forecast combination
            -> Portfolio construction -> Execution model -> Risk engine -> Walk-forward validation -> Attribution -> Experiment database
```

A strategy is a forecast model that plugs into this pipeline: see [the plugin framework](techniques/plugin-framework.md). The code for each box is listed in the [chapter map](chapter_map.md), one row per technique, with the stage that runs it,
the figures that show it and the tests that pin it.

## The rules that make the results worth reading

1. **Declare before you look.** The decision rule for each hypothesis is committed in a configuration file before the code that tests it is run. Changes after seeing results are disclosed, never silent.
2. **Count your trials.** Every specification tried is counted and the count feeds the deflated Sharpe ratio.
3. **Compare against the dumb alternative.** The historical mean, equal weight, a placebo that shifts the regime path.
4. **Read the power.** A "not significant" result means something only if the test could have found the effect; Stage 39 says what these tests could find.
5. **Separate post-hoc from declared.** Post-hoc analyses are labelled and never overturn a declared decision.

## What this repository cannot tell you

No result here is evidence that a strategy will make money. The universe is 15 ETFs that survived to today; the samples are 5 to 18 years; value and quality are price-based proxies; no live language model was ever called; the crypto branch is one exchange. The
[roadmap coverage](roadmap_coverage.md) lists what was built, what was built in part and what was not built, and why.
