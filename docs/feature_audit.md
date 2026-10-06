# Feature audit: Generation 2-4 roadmap

A strict audit against seven completion criteria. A feature is **COMPLETE** only if it has (1) production code, (2) no placeholder logic, (3) tests, (4) integration with the
pipeline, (5) reachability through the normal experiment workflow (`quant backtest`, `quant run`, or the stage runner for stage-level experiments), (6) documentation, and (7) a simple reproducible example.
A filename, a class or a README sentence does not count. Statuses: ✅ COMPLETE, 🟡 PARTIAL (reason), ❌ NOT IMPLEMENTED; DEFERRED means a documented decision not to build it now.

**Reading the "After" column.** Where a feature needs data, credentials or compute this environment does not have, it is not marked complete even if the code is. No live language-model call, Ray run, Docker build or CI run
was made. Statistical *findings* about a technique (does it make money?) live in the generation reports; this document is about whether the technique is built, integrated and reproducible.

## Phase 1: the audit as found (before this pass)

| Feature | Status | Evidence | Missing pieces | Action |
|---|---|---|---|---|
| Market regime detection | 🟡 | `src/framework/regimes.py` (rule, composite, HMM), `src/models/regimes.py`, `experiments/stage16_regimes.py`, `tests/test_regimes.py` | GMM and BOCPD existed only in the Stage 16 script; regimes did not feed forecast confidence or risk limits | COMPLETE EXISTING IMPLEMENTATION |
| Adaptive allocation | ✅ | `src/framework/allocation.py::RegimeSwitch`, `experiments/stage31_adaptive.py`, `tests/test_framework.py` | none | KEEP |
| Forecast confidence | 🟡 | `src/framework/types.py::Forecast`, `src/models/probabilistic.py`, `experiments/stage19_probabilistic.py` | confidence did not depend on regime; calibration (Platt, isotonic) was a stage-only analysis, not applied inside a pipeline run | COMPLETE EXISTING IMPLEMENTATION |
| Alpha combination engine | 🟡 | `src/framework/forecasting.py::combine_forecasts`, `src/signals/alpha_engine.py`, `experiments/stage24_combination.py` | decay estimates not consumed by any combination rule | COMPLETE EXISTING IMPLEMENTATION |
| Hierarchical risk parity | ✅ | `src/portfolio/hierarchical.py`, `experiments/stage18_hierarchical.py`, `tests/test_hierarchical.py`, allocator `static` (`hrp`, `herc`) | none (no CLI example of parameters) | KEEP |
| Dynamic covariance (DCC-GARCH, O-GARCH) | 🟡 | `src/portfolio/dynamic_covariance.py`, `experiments/stage17_dynamic_covariance.py`, `tests/test_dynamic_covariance.py` | not an allocator: could not be run through `Pipeline`/`quant` | CREATE NEW IMPLEMENTATION (allocator) |
| Portfolio attribution | 🟡 | `src/backtest/attribution.py`, `experiments/stage20_attribution.py`, tear-sheet asset-class table | Brinson and model-level attribution not in the pipeline report | COMPLETE EXISTING IMPLEMENTATION |
| Execution / market impact | 🟡 | `src/backtest/impact.py`, `experiments/stage25_execution.py`, `tests/test_impact.py`, `execution.aum` | no cost breakdown or capacity-by-AUM in the tear sheet or CLI | COMPLETE EXISTING IMPLEMENTATION |
| Research database | 🟡 | `src/research_db/builder.py`, `experiments/stage29_research_db.py`, `tests/test_research_db.py`, `src/framework/experiments.py` | runs did not store model outputs or the commit; no SQL over runs | COMPLETE EXISTING IMPLEMENTATION |
| Experiment manager | 🟡 | `src/framework/experiments.py`, `src/cli.py` | essentially untested; no parallel sweep | COMPLETE EXISTING IMPLEMENTATION |
| Online learning | 🟡 | `src/models/online.py`, `experiments/stage22_online.py`, `tests/test_online.py` | not a plug-in model | CREATE NEW IMPLEMENTATION |
| Macro feature layer | 🟡 | `src/data/macro.py`, `src/features/macro.py`, `src/strategies/macro.py`, `tests/test_macro.py` | ISM PMI is proprietary (a Philadelphia Fed survey is the proxy) | DEFER WITH REASON (data) |
| Alternative data | 🟡 | `src/data/altdata.py`, `experiments/stage23_altdata.py`, `tests/test_altdata.py` | positioning series not in the bundle; no model reads them; ETF flows, analyst revisions, CDS not sourced | CREATE NEW IMPLEMENTATION (bundle + model); DEFER the rest |
| Bayesian portfolio construction | 🟡 | `src/portfolio/bayesian.py`, `experiments/stage21_bayesian.py`, `tests/test_bayesian.py` | not an allocator | CREATE NEW IMPLEMENTATION (allocator) |
| Explainable AI | 🟡 | `src/models/explain.py`, `experiments/stage35_explain.py`, `tests/test_explain.py` | not available from a pipeline run or the tear sheet | COMPLETE EXISTING IMPLEMENTATION |
| Forecast calibration | 🟡 | `experiments/stage35_explain.py`, `experiments/stage19_probabilistic.py` | stage-only | COMPLETE EXISTING IMPLEMENTATION |
| Distributed experiment execution | 🟡 | `src/distributed/executor.py`, `experiments/stage26_distributed.py` | only used by a stage; the experiment manager ran variants serially; Ray never run | COMPLETE EXISTING IMPLEMENTATION; DEFER Ray (not installed) |
| Research dashboard | ✅ | `src/framework/dashboard.py`, `tests/test_dashboard.py` | user runs not included | KEEP (small extension) |
| PatchTST, TSMixer | 🟡 | `src/models/deep_forecast.py`, `experiments/stage27_deep.py`, `tests/test_deep_forecast.py` | stage-only | CREATE NEW IMPLEMENTATION (plug-in) |
| N-BEATS, N-HiTS | 🟡 | `src/models/deep_family.py`, `experiments/stage33_frontier.py`, `tests/test_frontier_models.py` | stage-only | CREATE NEW IMPLEMENTATION (plug-in) |
| TimeMixer | 🟡 | `src/models/deep_family.py` | a simplified multiscale mixer "in the spirit of" the paper, not the paper's decomposable past/future mixing | DEFER WITH REASON |
| Chronos | 🟡 | `experiments/stage34_foundation.py` | stage-only | CREATE NEW IMPLEMENTATION (plug-in) |
| TimesFM | ❌ | none | everything | CREATE NEW IMPLEMENTATION if reachable |
| Graph neural network | 🟡 | `src/models/graph_forecast.py`, `experiments/stage33_frontier.py` | stage-only | DEFER WITH REASON |
| Bayesian deep learning | 🟡 | `src/models/deep_forecast.py::mc_dropout_predict`, `experiments/stage33_frontier.py` | stage-only | DEFER WITH REASON |
| Diffusion models | 🟡 | `src/models/diffusion.py`, `experiments/stage36_diffusion.py`, `tests/test_diffusion.py` | stage-only; a scenario generator, not a forecaster | DEFER WITH REASON |
| Causal inference | 🟡 | `src/causal/estimators.py`, `experiments/stage37_causal.py`, `tests/test_causal.py` | no command; causal forests not built | COMPLETE EXISTING IMPLEMENTATION (command); DEFER forests |
| LLM research assistant | 🟡 | `src/assistant/`, `experiments/stage28_text.py`, `tests/test_assistant.py` | no `quant` command; no live model call has ever been made | COMPLETE EXISTING IMPLEMENTATION (command); DEFER live use (credentials) |
| Reinforcement learning (lowest priority) | 🟡 | `src/models/rl_allocation.py`, `experiments/stage38_rl.py`, `tests/test_rl_allocation.py` | not an allocator | CREATE NEW IMPLEMENTATION (allocator) |

## Phase 5: final verification (after this pass)

"Tests added" counts test functions (parametrised cases not multiplied) in the named files.

| Feature | Before | After | Files added / modified | Tests added | Notes |
|---|---|---|---|---|---|
| Market regime detection | 🟡 | ✅ | `src/framework/adaptive.py` (new: `regime_spread_scale`, `RegimeRiskLimits`), `src/framework/regimes.py` (`gmm`, `bocpd` registered), `src/framework/pipeline.py`, `src/framework/tearsheet.py`, `examples/adaptive_pipeline.yaml` | `tests/test_adaptive_integration.py` | Regime now feeds forecast spread, volatility target, gross caps, drawdown de-risking and the by-regime report. Detectors are causality-checked. The "Bull" label is still absent (see roadmap_coverage) |
| Adaptive allocation | ✅ | ✅ | `src/framework/allocation.py` (unchanged) | existing | Whether it *beats* risk parity is a Generation 5 finding: it does not |
| Forecast confidence | 🟡 | ✅ | `src/framework/adaptive.py::calibrate_confidence`, `src/framework/types.py` (`p_up`), `src/framework/validate.py` | `tests/test_adaptive_integration.py` | `forecast: {confidence: {method: platt \| isotonic}}`; matured labels only; Brier and reliability use the calibrated probability |
| Alpha combination engine | 🟡 | ✅ | `src/framework/adaptive.py::fit_decay`, `holding_period_ic`, `decay_trust_weights`, `src/framework/pipeline.py` | `tests/test_adaptive_integration.py` | `combination.rule: decay_weighted`; consumes momentum, mean-reversion, ML forecasts, confidence and decay estimates. Decay fit is a grid NNLS over lags 1-20 and says "no claim" when the IC is not distinguishable from zero |
| Hierarchical risk parity | ✅ | ✅ | `src/cli.py` (`--alloc-param`) | `tests/test_adaptive_integration.py` | `quant backtest --allocator static --alloc-param book=hrp` |
| Dynamic covariance | 🟡 | 🟡 | `src/framework/allocators_portfolio.py::DynamicCovarianceMinVar` | `tests/test_portfolio_allocators.py` | Allocator `dynamic_cov` (DCC, O-GARCH, static control). The universe is the assets with complete history over the first training window (so selection never looks ahead): later-listed assets are excluded. Kalman-filter covariance and factor stochastic volatility are not built (see roadmap_coverage) |
| Portfolio attribution | 🟡 | ✅ | `src/framework/analytics.py` (`brinson_vs_benchmark`, `model_attribution`), `src/framework/pipeline.py`, `src/framework/tearsheet.py` | `tests/test_adaptive_integration.py` | Brinson-Fachler against equal weight (cash as a sector) and model-level attribution that reconciles exactly. Factor attribution stays a stage (Stage 40) |
| Execution / market impact | 🟡 | 🟡 | `src/framework/analytics.py` (`cost_breakdown`, `capacity_by_aum`), `src/cli.py` (`quant capacity`, `--aum`), `src/framework/pipeline.py` | `tests/test_adaptive_integration.py` | Impact feeds net returns, strategy comparison (capacity curve) and attribution (cost split). Slippage as a separate term is DEFERRED: it is not separable from spread and impact in daily ETF data. Needs intraday or fill data |
| Research database | 🟡 | ✅ | `src/framework/experiments.py` (forecasts, `meta.json`, commit), `src/research_db/__main__.py`, `src/cli.py` (`quant sql --db runs`) | `tests/test_experiment_manager.py` | Stores spec, config fingerprint, git commit, metrics, tables, model forecasts. SQLite only (DuckDB/Postgres not used, by decision) |
| Experiment manager | 🟡 | ✅ | `src/framework/experiments.py` (`sweep`, `expand_grid`), `src/cli.py` (`quant sweep`), `src/framework/dashboard.py` | `tests/test_experiment_manager.py` (10) | Idempotent runs, trial counting for the deflated Sharpe ratio, parallel sweep, comparison |
| Online learning | 🟡 | ✅ | `src/strategies/online.py` | `tests/test_online_model.py` | Model `online_ridge` (ridge/NLMS/Kalman updaters); only matured labels are absorbed |
| Macro feature layer | 🟡 | 🟡 | `src/framework/data.py` (positioning loader) | `tests/test_positioning_explain.py` | PARTIAL: ISM PMI is proprietary, so a regional Fed survey stands in. Everything else the roadmap lists is in the bundle |
| Alternative data | 🟡 | 🟡 | `src/framework/data.py::_with_positioning`, `src/strategies/macro.py::CftcPositioning` | `tests/test_positioning_explain.py` | CFTC positioning is in the bundle after its release lag and drives the `cftc_positioning` model. PARTIAL: ETF flows, analyst/earnings revisions and CDS spreads are DEFERRED: no free point-in-time source was reachable. A positioning table must have been downloaded (Stage 23) |
| Bayesian portfolio construction | 🟡 | ✅ | `src/framework/allocators_portfolio.py::BayesianMeanVariance` | `tests/test_portfolio_allocators.py` | Allocator `bayesian` (`mvo_sample`, `bayes_stein`, `bayes_predictive`) |
| Explainable AI | 🟡 | 🟡 | `src/strategies/ml.py::MLRidge.explain`, `src/framework/forecasting.py`, `src/framework/pipeline.py`, `src/framework/tearsheet.py` | `tests/test_positioning_explain.py` | `evaluation: {explain: true}` gives a permutation-importance table for `ml_ridge`. PARTIAL: exact Shapley values and integrated gradients (for the MLP and the networks) remain Stage 35 only; the table is descriptive (rows overlap training) |
| Forecast calibration | 🟡 | ✅ | see "Forecast confidence" | see above | Platt and isotonic in the pipeline, reliability table in the tear sheet |
| Distributed experiment execution | 🟡 | 🟡 | `src/framework/experiments.py::sweep` using `src/distributed/executor.py` | `tests/test_experiment_manager.py` | Serial, joblib and dask agree; PARTIAL: the Ray adapter has never been run (Ray not installed here) |
| Research dashboard | ✅ | ✅ | `src/framework/dashboard.py` | `tests/test_dashboard.py` | A static HTML page by design, not a live server; now includes the user's own runs |
| PatchTST, TSMixer, N-BEATS, N-HiTS | 🟡 | ✅ | `src/strategies/deep.py::DeepWindow` | `tests/test_deep_models.py` | Model `deep_window` (`kind` selects the network); yearly refit on matured origins; CPU only. N-HiTS uses repetition rather than interpolation |
| TimeMixer | 🟡 | 🟡 | `src/strategies/deep.py` (`kind: timemixer`) | `tests/test_deep_models.py` | PARTIAL: the "style" mixer, not the paper's architecture. DEFERRED: needs the decomposable past/future mixing and a reference implementation to check against |
| Chronos | 🟡 | ✅ | `src/strategies/deep.py::ChronosZeroShot` | `tests/test_deep_models.py` | Optional dependency; weights from the Hugging Face hub; causality-checked on the real weights. Pre-training may overlap the sample |
| TimesFM | ❌ | 🟡 | `src/strategies/deep.py::TimesFMZeroShot` | `tests/test_deep_models.py` | PARTIAL: plug-in works (TimesFM 2.5, 200M, CPU, causality-checked on the real weights) and one exploratory run is below; no pre-registered study was run. The `timesfm` package is installed ad hoc here (not in `requirements.txt`) |
| Graph neural network | 🟡 | 🟡 (DEFERRED integration) | unchanged | existing | Stage 33 only. Not a plug-in because it needs a cross-sectional trainer and added nothing over the historical mean; building it would add a model, not integration. Minimum viable version: wrap `train_and_predict_graph` like `deep_window` |
| Bayesian deep learning | 🟡 | 🟡 (DEFERRED integration) | unchanged | existing | Monte-Carlo-dropout spread is Stage 33 only; minimum viable version: expose it as `deep_window`'s `std` (Stage 33 found ensemble uncertainty carries no information about error) |
| Diffusion models | 🟡 | 🟡 (DEFERRED integration) | unchanged | existing | A scenario generator, run as Stage 36. Minimum viable version: a `quant scenarios` command |
| Causal inference | 🟡 | 🟡 | `src/causal/check.py`, `src/cli.py` (`quant causal`) | `tests/test_cli_tools.py` | `quant causal` validates the estimators against simulated truth (including a violated-assumption world). The applied FOMC question is Stage 37 only. Causal forests are DEFERRED (no forest effect model; the R-learner with a ridge effect model exists) |
| LLM research assistant | 🟡 | 🟡 | `src/cli.py` (`quant ask`) | `tests/test_cli_tools.py` | PARTIAL, cannot be completed here: no API credentials, so **no language model has ever been called**; the template backend and the read-only SQL guard are tested, and `--llm` exists but is unexercised. Earnings-call and news summarisation need data and credentials |
| Reinforcement learning | 🟡 | ✅ | `src/framework/allocators_portfolio.py::EvolutionStrategyPolicy` | `tests/test_portfolio_allocators.py` | Allocator `es_policy`. Stage 38 found no out-of-sample gain over equal weight; shipped as a reproducible baseline, lowest priority |

## Exploratory runs (one sample, not pre-registered, not evidence of an edge)

Each new model or allocator run once through `Pipeline` on the 15-ETF bundle with costs, no causality re-check (done in the tests). The windows start where each model first has a forecast, so Sharpe ratios are **not** comparable across rows.
Equal weight over the same sample is about 0.7 net Sharpe. These runs show the components execute end to end on real data; they do not show they are worth trading.

| Run | Net Sharpe | Gross Sharpe | CAGR | Volatility | Max drawdown | Turnover (x/yr) | First day | Seconds |
|---|---|---|---|---|---|---|---|---|
| `ml_ridge` | -0.40 | -0.23 | -2.4% | 5.6% | -39.9% | 13.8 | 2011-01-31 | 2 |
| `online_ridge` | -0.35 | -0.13 | -2.1% | 5.7% | -34.4% | 19.6 | 2010-12-31 | 2 |
| `deep_window:nbeats` | -0.30 | -0.07 | -2.0% | 6.1% | -31.5% | 22.7 | 2011-01-31 | 9 |
| `deep_window:patchtst` | -0.19 | -0.02 | -1.4% | 6.3% | -21.7% | 17.4 | 2011-01-31 | 16 |
| `cftc_positioning` | +0.34 | +0.36 | 1.5% | 4.5% | -11.6% | 1.3 | 2009-07-20 | 2 |
| `chronos` | +0.40 | +0.51 | 2.5% | 6.7% | -19.0% | 12.0 | 2007-04-30 | 37 |
| `timesfm` | +0.14 | +0.27 | 0.7% | 6.8% | -24.4% | 13.1 | 2007-04-30 | 830 |
| `alloc:dynamic_cov(dcc)` | +0.85 | +0.87 | 3.3% | 3.9% | -17.4% | 1.7 | 2006-02-01 | 20 |
| `alloc:dynamic_cov(static)` | +0.79 | +0.80 | 3.2% | 4.0% | -17.7% | 0.6 | 2006-02-01 | 12 |
| `alloc:bayesian(bayes_stein)` | +0.55 | +0.57 | 6.0% | 12.0% | -19.8% | 4.7 | 2006-02-01 | 9 |
| `alloc:es_policy` | +0.09 | +0.14 | 0.3% | 13.8% | -48.8% | 11.3 | 2006-02-01 | 10 |
| `alloc:static(risk_parity)` | +0.79 | +0.79 | 5.0% | 6.5% | -18.5% | 0.4 | 2006-02-01 | 3 |

## Reproducing

```bash
quant run examples/adaptive_pipeline.yaml --tearsheet --group examples     # the whole integrated chain
quant backtest --model online_ridge --tearsheet
quant backtest --model deep_window --param kind=nbeats
quant backtest --allocator dynamic_cov --alloc-param model=dcc --model momentum
quant capacity --spec examples/adaptive_pipeline.yaml
quant sweep --spec examples/adaptive_pipeline.yaml --grid combination.rule=equal,decay_weighted
quant causal; quant ask which hypotheses were retained
pytest -q tests/test_feature_audit.py                                      # the evidence paths in this file exist
```
