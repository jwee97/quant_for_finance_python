# Systematic Multi-Asset Alpha, Portfolio Construction & Risk Platform

An end-to-end quantitative research platform built on *Quantitative Finance with
Case Studies in Python*, covering the full arc from raw market data to a
validated research conclusion.

**The research question**

> Can economically interpretable systematic signals generate persistent
> out-of-sample returns across liquid asset classes, and can robust portfolio
> construction improve their risk-adjusted performance after realistic
> transaction costs?

**The answer this project reached, in one line:** no for the signals, yes for
the portfolio construction — and the negative half of that is the more useful
result.

**New here?** `pip install -e .`, then `quant demo` (thirteen seconds on the cached data), `quant dashboard` (an offline
explorer of every hypothesis the project declared, most of them rejected) and `quant explain deflated sharpe` (any term,
technique, strategy or experiment, in plain language). Read [`docs/START_HERE.md`](docs/START_HERE.md) for a learning path
by background, the [technique guides](docs/techniques/index.md) (73 of them, from PCA to diffusion models and from HAC errors to option surfaces), the
[chapter map](docs/chapter_map.md) from the book to the code, and [what was and was not built](docs/roadmap_coverage.md).
Adding a strategy is writing one forecast model: [how to](docs/how_to_add_a_strategy.md).

---

## The design principle

This project is deliberately **not** "build the strategy with the highest
Sharpe ratio". It is:

> Build a defensible quantitative research process capable of determining
> whether an apparent strategy is likely to represent genuine signal or
> research overfitting.

```
research quality  >  backtest attractiveness
```

An out-of-sample Sharpe ratio of 0.7 backed by evidence is a better result
than a suspicious 3.0. Every rejected hypothesis is recorded in
[`experiments/registry.md`](experiments/registry.md) alongside the accepted
ones, because a research process that only reports its successes is not a
research process.

---

## Headline results

Full sample: **15 liquid ETFs, 5,211 trading days, 2006-01-03 to 2026-09-21.**
All figures are net of transaction costs at the baseline assumption, on the engine as corrected in the Generation 2 build (see the erratum in the research report).

| Model | CAGR | Vol | Sharpe | Max DD | Turnover |
|-------|-----:|----:|-------:|-------:|---------:|
| SPY buy & hold | 11.1% | 19.2% | 0.65 | −55.2% | 0.0x |
| M0 Equal weight | 7.1% | 10.9% | 0.69 | −33.4% | 0.4x |
| M1 Inverse volatility | 5.4% | 6.4% | **0.85** | −17.7% | 0.5x |
| M2 Risk parity | 5.0% | 6.5% | 0.79 | −18.5% | 0.4x |
| M3 Momentum | 1.9% | 6.7% | 0.31 | −14.8% | 9.2x |
| M4 Mean reversion | −1.9% | 6.8% | −0.24 | −34.7% | 22.4x |
| M5 Momentum + reversion | 0.2% | 6.7% | 0.07 | −19.9% | 18.4x |
| M7 Combined alpha + shrinkage MVO | 3.5% | 7.5% | 0.50 | −21.1% | 10.8x |
| M8 Black-Litterman | 4.2% | 11.2% | 0.42 | −29.5% | 11.9x |
| M9 Mean-CVaR | 3.3% | 4.2% | 0.81 | −14.2% | 0.6x |

The full table, including in-sample/out-of-sample splits and every metric, is
in [`reports/tables/stage12_final_comparison.csv`](reports/tables/).

### Six findings worth stating plainly

1. **The alpha signals do not survive costs.** Momentum and mean reversion
   both have statistically detectable information, but it lives at the 1–5 day
   horizon. Harvesting it requires 9–22x annual turnover, which costs more
   than the signal is worth. Mean reversion breaks even at a transaction cost
   far below what these instruments actually trade at.

2. **Newey-West changes the conclusions.** With overlapping forward returns,
   naive standard errors understate uncertainty by 1.8–3.3x. A 252-day
   momentum signal goes from t = 2.57 ("significant") to t = 0.77 (not) once
   the overlap is corrected. That single number is the argument for Chapter
   20's serial-correlation treatment.

3. **Fifteen ETFs are about five independent bets.** PC1 explains 40.7% of
   variance and only three eigenvalues clear the Marchenko-Pastur noise bound.
   The effective rank falls to 4.1 in March 2020 — diversification thins out
   exactly when it is needed.

4. **The risk model fails its own backtest, and that is the useful part.**
   Every VaR method breaches too often at 99% (historical: 1.46% of days
   against 1% promised) and every method fails the Christoffersen
   independence test at 95%: the breaches cluster. An unconditional VaR is not
   a risk limit.

5. **Four strategy types, one conclusion.** Momentum, mean reversion, machine
   learning and relative value (pairs and PCA statistical arbitrage) were all
   built with the same discipline, and all four land in the same place: a real
   but small edge, a high required turnover, and costs that close the gap.
   Zero of ten candidate pairs were even cointegrated.

6. **What does work is the part that estimates the least.** The models that
   never touch expected returns — inverse volatility, risk parity, mean-CVaR
   — beat every model that does. The estimation-error experiment shows why:
   unconstrained mean-variance holds 3 of 15 assets at a 77% maximum weight,
   and perturbing expected returns by 25% of their cross-sectional dispersion
   moves up to 92% of the book.

---

## Generation 2

Six extensions, each held to a decision rule committed before its result existed (the one exception, Stage 18, is
disclosed in the report): macro features, regime detection, dynamic covariance, hierarchical risk parity,
probabilistic forecasting and performance attribution. The full account, including every deviation and post-hoc
analysis, is in [`reports/generation2_report.md`](reports/generation2_report.md).

**The result is the Generation 1 result again.** Of 23 decision-bearing hypotheses, three were retained, each with a
qualification: a volatility-clustering sanity check; a DCC-GARCH minimum-variance book that is 3.9% less volatile
for twice the turnover; and fractional-Kelly sizing, whose pass is confounded (it loses to passive equal weight and
does not beat sign-only sizing of its own signal). Macro features do not forecast beyond price; regimes found in daily
returns are a few days long and an overlay does not beat what it overlays; HRP and HERC are not better than risk
parity; Platt calibration made the probabilities worse.

**An engine bug was found and fixed.** The weight-drift step was one day stale. Fixing it moved every Generation 1 Sharpe
ratio by at most 0.016, changed no ranking and no decision, and every Generation 1 number was regenerated. The
before/after record is in [`reports/errata/`](reports/errata/).

**Not built in Generation 2:** Wishart and factor stochastic-volatility covariance models, factor-model attribution, and
the software-engineering items of the roadmap's Priority 20.

---

## Generation 3

Five extensions (Stages 21-25), again each held to a rule committed before its result existed: Bayesian portfolio
construction, online learning, non-price data, an alpha-combination engine, and an execution model with market impact.
The full account, including every deviation and post-hoc analysis, is in
[`reports/generation3_report.md`](reports/generation3_report.md).

**The result is, once more, the Generation 1 result.** Of 9 decision-bearing hypotheses, two were retained, each with a
qualification: posterior-averaged Bayesian weights are more stable than plug-in weights (but averaging is mechanical, and
the Bayesian book's Sharpe is no better), and the six allocators lose at most 0.042 of net Sharpe at $1bn of assets (a thin
margin that fails at twice the impact coefficient). Online learners forecast no better than an annually refitted ridge;
credit, implied-volatility structure, jobless claims and CFTC positioning add nothing out of sample; combining the alphas
at the forecast level does not beat momentum alone; a 1% no-trade band does not help.

**The new finding is about size.** With square-root market impact, the allocators keep their Sharpe to beyond $10bn,
while the momentum book halves its Sharpe at about $100m, and mean reversion, the combined book and every alpha
combination are not viable at any size worth running.

**Not built in Generation 3:** ETF flows, earnings revisions, creation and redemption data and options flow (not available
free and not proxied); explainable machine learning, factor-model attribution, an Almgren-Chriss optimal trajectory, an
estimated impact coefficient.

---

## Generation 4

Distributed experimentation, modern deep learning, a text-and-assistant stage, a research database (Stages 26-29), and the
engineering around them. Three research questions with rules committed before their results; the database and the
engineering have acceptance checks instead. Full account: [`reports/generation4_report.md`](reports/generation4_report.md).

**One of three hypotheses retained, and its qualification is the finding.** A grid of 1,584 simple rules run in parallel
has 76% of its members with a positive net Sharpe and a best of 0.45, but a search that size produces a best of about 0.77 from
noise: White's Reality Check p = 0.50, Hansen's SPA p = 0.34, probability of backtest overfitting 0.60, and the in-sample
winner has no out-of-sample edge. An MLP-mixer has a significantly lower CRPS than the Stage 19 ridge (p = 0.019), but the ridge
is worse than predicting zero, and the mixer is no better than each asset's historical mean. FOMC statement tone, change and
action add nothing to price and macro.

**What the engineering did and did not do.** The research record is queryable (`python -m src.research_db "SELECT ..."`,
`python -m src.assistant "which hypotheses were retained?"`) with six acceptance checks that pass. A Dockerfile, a compose
file, a CI workflow, pre-commit, a Makefile and a lint gate exist; the image and the workflow were **not run** (no Docker
daemon or Actions runner here). **No language model was called to produce any result**; the hosted-model paths are tested
against fake clients only.

**Not built:** zero-shot foundation models (their pre-training overlaps the sample), TFT / TimeMixer / N-HiTS / N-BEATS,
any live language-model run, a cloud deployment, the ray backend's test, Hydra / MLflow / Weights & Biases, explainable
machine learning, factor-model attribution and reinforcement learning.

---

## Generation 5

A plugin framework and a newcomer layer around the earlier research, a library of 34 registered strategies (80 models after the integration audit, the formula strategy and the second batch below), and thirteen
stages of new evidence (Stages 30-40), each decision rule committed before its result. The full account, with every commit,
deviation and post-hoc analysis, is in [`reports/generation5_report.md`](reports/generation5_report.md).

**The framework.** `Forecast(mean, std, confidence)` and `Regime(name, probability)` objects, registries for forecast models,
regime detectors and allocators, calibration against matured outcomes, six forecast-combination rules, a regime-switching
allocator and risk policy, a causality test run on every plugin, a persistent experiment database that counts trials for the
deflated Sharpe ratio, and a tear sheet with fixed red-flag rules. `quant backtest --model dual_momentum --tearsheet` runs one
strategy through all of it; a new strategy is one class (see `docs/how_to_add_a_strategy.md`).

**Strategy library.** 80 registered strategies, including a second batch of the best-known systematic rules (time-series momentum and the
moving-average and breakout ensembles, KAMA, ADX, SuperTrend, Keltner, Ichimoku and MACD trend, the 52-week high, residual and smooth
momentum, RSI(2), IBS, Bollinger and stochastic pullbacks, the Ornstein-Uhlenbeck and range filters, turn-of-the-month and Halloween
calendar rules, betting against beta, low idiosyncratic volatility, MAX, illiquidity, value with momentum, volatility-managed exposure,
buy-the-VIX-spike, credit risk appetite, Faber's GTAA, Protective, Vigilant and Defensive Asset Allocation, adaptive asset allocation and
the classic model portfolios) plus `min_variance`, `max_diversification` and `sleeves` allocators. [`docs/strategy_survey.md`](docs/strategy_survey.md)
runs all of them once on the 15 ETFs with the deflated Sharpe ratio counting every one as a trial; the honest reading is that few beat
equal weight and none clears 0.95 (with the same-dates comparison added in Generation 6, only the static 60/40 mix clearly beats it and `dual_momentum` ties).

**Dashboard.** `quant serve` opens a local dashboard ([`docs/dashboard.md`](docs/dashboard.md)): add any Yahoo Finance tickers next to the
platform's 15 ETFs, pick a strategy (or type a one-line formula, or drop a Python file into `user_strategies/`), and see net growth
against equal weight and risk parity, drawdowns, rolling Sharpe, annual and monthly returns, exposure, attribution, red flags and a
deflated Sharpe ratio that counts how many ideas you tried. It also compares runs and renders the guides, including
[how to add a strategy](docs/how_to_add_a_strategy.md). Runs on localhost with a per-launch token; the browser never sends code.

**Integration audit.** A strict feature audit of the Generation 2-4 roadmap ([`docs/feature_audit.md`](docs/feature_audit.md), with a
before/after status for every feature) found that most of it lived only in stage scripts. The pieces are now reachable from one
specification ([`docs/platform_integration.md`](docs/platform_integration.md)): regimes feed the forecast spread, the
volatility target and risk limits; confidence is calibrated walk-forward; combination can weight by fitted alpha decay;
impact costs, capacity, Brinson and model-level attribution appear in every tear sheet; runs store their forecasts and git
commit; `quant sweep` runs parallel grids as counted trials; and online, deep, Chronos, TimesFM, CFTC-positioning,
dynamic-covariance, Bayesian and evolution-strategy components are plugins. `quant run examples/adaptive_pipeline.yaml --tearsheet`
runs the whole chain.

**Fourteen decision-bearing hypotheses, four retained, each with its qualification.** A search over 42 library specifications
finds one that beats cash after the search (dual momentum, Sharpe 0.90; Reality Check p = 0.0015) and none that beats passive
equal weight (0.74; p = 0.61). The regime path beats 99.5% of shifted placebo paths, but the allocator it drives does not beat
risk parity. Regime-conditional trust in twelve models (0.88 against 0.78, p = 0.0675) and a double-machine-learning effect of
FOMC tone (p about 0.05, from an estimator whose declared validation failed) are retained and fragile. Against the historical
mean, none of N-BEATS, N-HiTS, a TimeMixer-style mixer or graph attention has a lower CRPS; zero-shot Chronos is significantly
worse; ensemble uncertainty carries no information about error; isotonic calibration does not beat Platt; diffusion
scenarios do not beat the empirical distribution for 5% VaR; and a reinforcement-learned allocator turns a training-period
gain into a Sharpe of 0.21 against 0.75 for equal weight.

**Two diagnostics.** Stage 39 measures the power of the platform's own tests: with 15.7 years and 6% tracking error the paired
Sharpe test needs a true difference of about 0.35, so most earlier rejections could not have been otherwise. Stage 40 regresses
the Generation 1 books on the Fama-French five factors plus momentum.

**Not built:** language-model results (no model was called), TimeGPT, causal forests, dispersion trading,
cross-exchange crypto spreads, accounting-based value and quality, ETF flows and analyst revisions, a run of the Ray backend,
Docker and CI runs.

---

## Generation 6: an institutional multi-asset research toolkit

The platform was widened from "15 ETFs" to the methods and asset classes a multi-asset research desk uses, with the same discipline: no look-ahead, trials counted, every estimator checked against a reference or a known truth, and results reported as they came out. Start with the [capability matrix](docs/capability_matrix.md) (what is COMPLETE, PARTIAL or NOT IMPLEMENTED and why), the [research survey](docs/research_survey.md) and the [architecture](docs/architecture.md); the exploratory results are in [`docs/institutional_findings.md`](docs/institutional_findings.md).

- **`src/stats`**: OLS/WLS/GLS with HC, HAC, one- and two-way clustered covariances, rolling regression and diagnostics; panel regression, Hausman, Fama-MacBeth with Shanken, GRS; ADF, Phillips-Perron, KPSS, variance ratios, Engle-Granger, Granger; event studies; Benjamini-Hochberg, Storey, Romano-Wolf, haircut and deflated Sharpe, BCa and block bootstrap.
- **`src/probability`**: block and stationary bootstrap with Politis-White block length, Monte Carlo with variance reduction, importance sampling, extreme value theory (GPD, Hill, GEV, conditional EVT), copulas, drawdown, ruin and Kelly, MCMC with R-hat and ESS, Bayesian Sharpe, Markov chains and Markov switching.
- **`src/econometrics`**: exact-likelihood ARIMA, VAR, Bayesian (Minnesota) VAR, Johansen and VECM, GARCH/GJR/EGARCH, state space and Kalman filters, dynamic factors.
- **`src/derivatives`**: Black-Scholes, Black-76, Bachelier, binomial, Longstaff-Schwartz, Heston, Merton, SABR; Greeks; an implied-volatility solver; SVI and SSVI with arbitrage checks; local volatility; a VIX-style index; a synthetic option market with a stated variance premium; an options backtester (bid/ask fills, hedging, settlement, Greek attribution); eight volatility strategies and a dispersion study.
- **`src/assets`**: futures calendars, continuous contracts and roll yield, Schwartz-Smith commodity curves, FX forwards and carry, Nelson-Siegel curves and bond analytics, and market bundles that make futures, FX, bonds and a mixed book run through the unchanged pipeline.
- **`src/microstructure`**: order-flow imbalance, Kyle and Amihud measures, a limit-order-book simulator, Avellaneda-Stoikov market making, Almgren-Chriss execution.
- **`src/ops`**: a content-addressed artifact store, a model registry, a hyperparameter optimiser that reports the selection bias of its own winner, purged and combinatorial cross-validation, profiling; `quant tune`, `quant benchmark`, `quant registry`.
- **Ten new models** (90 in all): `kalman_trend`, `garch_vol_managed`, `evt_risk_managed`, `bvar_lead_lag`, `carry_xs`, `carry_ts`, `basis_momentum`, `long_term_reversal`, `ml_trees` (gradient boosting and forests) and `deep_representation` (autoencoder and contrastive), plus the `kelly` and `black_litterman` allocators and a Temporal Fusion Transformer in `deep_window`.

**What the evidence says, honestly.** On the 15 ETFs none of the new learners earned a tradable net Sharpe ratio (the gradient-boosting, forest and representation models range from -0.32 to +0.07), the EVT-sized long book earned +0.68 against +0.76 for equal weight over the same dates, with the same drawdown, and conditional EVT passed a 99% VaR backtest where an EWMA-normal model did not (breach rates 0.97% and 2.14%). Futures, FX, options and market-making results come from **synthetic** markets built with a stated truth: they show that the code recovers what was built in, and say nothing about real markets. Real option chains, contract-level futures data, tick data and fundamentals are not available here; the loaders and schemas for them exist and are documented.

## Repository layout

```
config/          every parameter that affects a result, in YAML
data/raw/        immutable per-ticker CSVs + provenance manifest
data/processed/  cleaned wide panels (rebuilt, not committed)
src/
  data/          download, validation, cleaning, loading, macro and non-price series  (Ch. 7)
  features/      returns, volatility, momentum, reversion, PCA (Ch. 8, 20)
  signals/       forecasts, position stack, blending, pairs, PCA stat-arb, alpha-combination engine (Ch. 22)
  distributed/   parallel executor (serial, joblib, dask, ray) with results identical to a serial run
  assistant/     FOMC corpus, extraction backends with schema and cache, read-only SQL assistant
  research_db/   SQLite research database builder and its acceptance checks
  portfolio/     EW, inverse vol, risk parity, MVO, BL, CVaR, covariance (Ch. 19, 20),
                 HRP / HERC, DCC-GARCH / O-GARCH, regime-aware overlays, Bayesian (NIW, Bayes-Stein)
  backtest/      engine, execution, costs, impact and capacity, metrics, attribution (Ch. 22)
  risk/          VaR, CVaR, contributions, stress              (Ch. 21)
  validation/    walk-forward, robustness, leakage detection, permutation tests,
                 Reality Check / SPA / probability of backtest overfitting
  models/        regression with HAC errors, ML ladder, macro forecasts,
                 regimes (HMM / GMM / BOCPD), probabilistic forecasts, online learners (RLS / NLMS /
                 Kalman / Hedge), patch-transformer, MLP-mixer, N-BEATS / N-HiTS / TimeMixer-style, graph attention,
                 diffusion, evolution-strategies policy, explanation methods
  framework/     Generation 5 plugin pipeline: types, registries, data bundle, regimes, calibration and combination, allocators,
                 regime risk, validation, experiment manager, tear sheet, dashboard, demo, generated docs
  strategies/    the 90 registered forecast models (time-series, cross-sectional, stat-arb, volatility, fixed income, macro, ML, crypto, econometric, multi-asset carry)
  stats/         Generation 6: regression and HAC, panels, unit roots, cointegration, event studies, multiple testing, Sharpe inference
  probability/   Generation 6: resampling, Monte Carlo, EVT, copulas, drawdown and Kelly, Bayesian inference, Markov models
  econometrics/  Generation 6: ARIMA, VAR/VECM, GARCH family, state space, dynamic factors
  derivatives/   Generation 6: pricing, Greeks, implied vol, SVI/SSVI, synthetic option market, options backtester, volatility strategies
  assets/        Generation 6: futures, commodity curves, FX, rates and bonds, market bundles
  microstructure/ Generation 6: order flow, limit-order-book simulator, market making, optimal execution
  ops/           Generation 6: artifact store, model registry, hyperparameter optimisation, purged/CPCV cross-validation, profiling
  causal/        double machine learning, R-learner, 2SLS, difference in differences, simulated worlds
  cli.py         the `quant` command
  utils/         config, logging, dates, plotting, experiment registry
docs/            START_HERE, 73 technique guides, glossary, chapter map, strategy cards, capability matrix, research survey, architecture, notebooks, roadmap coverage (mkdocs.yml)
experiments/     numbered stage scripts + the experiment registry
reports/         figures, tables, the data-quality report, the research paper,
                 the Generation 2-5 reports, errata/ (before/after record of the drift fix), the dashboard
tests/           973 tests
Dockerfile, docker-compose.yml, Makefile, .github/workflows/ci.yml and benchmark.yml, .pre-commit-config.yaml
```

---

## Running it

```bash
pip install -r requirements.txt

python -m experiments.run_all --download      # all 40 stages
python -m experiments.run_all --generation 2  # Generation 2 only (stages 15-20)
python -m experiments.run_all --generation 3  # Generation 3 only (stages 21-25)
python -m experiments.run_all --generation 4  # Generation 4 only (stages 26-29)
python -m experiments.run_all --generation 5  # Generation 5 only (stages 30-40)
python -m experiments.run_all --from 6 --to 9 # a range of stages
python -m experiments.run_all --fresh         # clear derived artefacts first
python -m experiments.stage01_data            # a single stage
pytest -q                                      # 973 tests
quant demo; quant dashboard; quant explain risk parity   # the newcomer layer
quant serve                                    # the interactive dashboard: your tickers, your strategies
quant new-strategy my_idea                     # a template for your own strategy in user_strategies/
quant docs build                               # regenerate strategy cards, chapter map, findings digest
make install test lint cov                     # the same through the Makefile; docker compose run --rm tests for the container
```

Stage 1 writes `data/raw/*.csv` once and refuses to overwrite them without
`--force`: raw data is immutable, and everything downstream is rebuilt from
it. The dataset's identity is the sha256-derived `data_version` in
`data/metadata/manifest.json`; the research configuration's identity is the
fingerprint printed by every stage.

---

## How the pipeline is structured

```
RAW DATA -> VALIDATION -> CLEAN DATA -> FEATURES
              |                            |
              |                  MOMENTUM / MEAN REVERSION
              |                            |
              |                    SIGNAL RESEARCH (IC, regression)
              |                            |
              |                      SIGNAL ENGINE
              |                            |
              |                 PORTFOLIO CONSTRUCTION
              |                  (EW / inv-vol / RP / MVO / BL / CVaR)
              |                            |
              |                       RISK ENGINE
              |                            |
              |                    BACKTEST + COSTS
              |                            |
              |                  WALK-FORWARD VALIDATION
              |                            |
              +------------->  STRESS / ROBUSTNESS -> REPORT
```

| Stage | Subject | Book basis |
|------:|---------|------------|
| 1 | Data collection, validation, cleaning | Ch. 1 §1.4, Ch. 7 |
| 2 | EDA, PCA, volatility estimators | Ch. 8 §8.2, §8.5; Ch. 20 §20.2 |
| 3 | Momentum alpha research | Ch. 22 §22.3.1, §22.3.9 |
| 4 | Mean-reversion alpha research | Ch. 22 §22.3.1 |
| 5 | Expected returns, IC, signal decay | Ch. 20 §20.1 |
| 6 | Backtesting and transaction costs | Ch. 22 §22.2 |
| 7 | Portfolio construction, estimation error | Ch. 19 §19.2–§19.9 |
| 8 | Covariance and volatility modelling | Ch. 20 §20.2.5–§20.2.11 |
| 9 | VaR, CVaR, stress testing | Ch. 21 §21.2–§21.3 |
| 10 | Combining strategies | Ch. 22 §22.5 |
| 11 | Walk-forward, robustness, leakage | Ch. 22 §22.2.4, §22.2.6–7 |
| 12 | Final comparison | Ch. 22 §22.2.5 |
| 13 | Machine learning extension | Ch. 23 |
| 14 | Pairs trading and PCA stat-arb | Ch. 22 §22.3.3–§22.3.7 |
| 15 | Macro features, nested-model predictability (Gen 2) | Ch. 20 §20.1 |
| 16 | Regime detection: HMM, GMM, BOCPD, regime overlays (Gen 2) | Ch. 8, Ch. 20 |
| 17 | Dynamic covariance: DCC-GARCH, O-GARCH (Gen 2) | Ch. 20 §20.2 |
| 18 | Hierarchical risk parity: HRP, HERC (Gen 2) | Ch. 19 |
| 19 | Probabilistic forecasts and confidence-aware sizing (Gen 2) | Ch. 20, Ch. 22 |
| 20 | Performance attribution: Brinson-Fachler, Carino, Euler (Gen 2) | Ch. 21, Ch. 22 |
| 21 | Bayesian portfolio construction: NIW posterior, Bayes-Stein, weights as a distribution (Gen 3) | Ch. 19, Ch. 20 |
| 22 | Online learning: RLS, NLMS, Kalman, Hedge aggregation with regret (Gen 3) | Ch. 20, Ch. 22 |
| 23 | Non-price data: credit spread, implied-volatility structure, claims, CFTC positioning (Gen 3) | Ch. 7, Ch. 20 |
| 24 | Alpha-combination engine: forecast-level combination, trust rules, alpha descriptors (Gen 3) | Ch. 20, Ch. 22 |
| 25 | Execution model: square-root impact, capacity, no-trade bands, scheduling (Gen 3) | Ch. 22 |
| 26 | Distributed experimentation: 1,584-rule grid, Reality Check, SPA, PBO (Gen 4) | Ch. 22 §22.2.4 |
| 27 | Deep learning: patch transformer, MLP-mixer against the Stage 19 ridge (Gen 4) | Ch. 23 |
| 28 | Text features and research assistant: FOMC statements, extraction schema, SQL guard (Gen 4) | Ch. 20 |
| 29 | Research database: SQLite, acceptance checks, query CLIs (Gen 4) | Ch. 7 |
| 30 | The strategy library as a search: 42 specifications, Reality Check, SPA, PBO, combination (Gen 5) | Ch. 22 |
| 31 | The adaptive pipeline: regime-switching allocation with placebo, regime risk, regime-conditional trust (Gen 5) | Ch. 19, Ch. 20 |
| 32 | The optional crypto branch: funding carry, basis, stablecoin flows (Gen 5) | Ch. 22 |
| 33 | N-BEATS, N-HiTS, TimeMixer-style, graph attention and Bayesian deep learning vs the historical mean (Gen 5) | Ch. 23 |
| 34 | Zero-shot foundation model: Chronos-Bolt, descriptive (Gen 5) | Ch. 23 |
| 35 | Explainability (permutation, Shapley, integrated gradients) and calibration (Platt vs isotonic) (Gen 5) | Ch. 23 |
| 36 | Diffusion-model scenarios for tail risk (Gen 5) | Ch. 21 |
| 37 | Causal inference: DML, R-learner, IV, difference in differences; FOMC application (Gen 5) | Ch. 23 |
| 38 | Reinforcement-learning allocation by evolution strategies (Gen 5) | Ch. 19 |
| 39 | Statistical power of the platform's own tests (Gen 5) | Ch. 22 |
| 40 | Factor attribution on Fama-French five factors plus momentum (Gen 5) | Ch. 22 |

---

## What defends this project against fooling itself

| Trap | Defence | Where |
|------|---------|-------|
| Look-ahead bias | Automated test: scramble all future data, require every past weight to be bit-identical. Includes a deliberately broken control strategy, so the test can fail. | `src/validation/leakage.py` |
| Survivorship / universe selection | Universe fixed ex ante in config, never edited after seeing results; residual ETF-survival bias stated as a limitation. | `config/universe.yaml` |
| Overlapping observations | Newey-West HAC standard errors everywhere, lag chosen from the horizon; naive-vs-HAC comparison reported. | `src/models/regression.py` |
| Parameter mining | Whole parameter families evaluated, never one point; plateau-vs-spike diagnostic; deflated Sharpe ratio. | `src/validation/robustness.py` |
| Multiple testing | Benjamini-Hochberg FDR control across all 72 tests per signal family. | `experiments/alpha_research.py` |
| Ignoring costs | Per-asset spreads, cost sweeps, and a breakeven cost for every strategy. | `src/backtest/costs.py` |
| Unrealistic turnover | Weights drift with returns between rebalances rather than being silently reset. | `src/backtest/execution.py` |
| Quiet data deletion | Anomalies are flagged and adjudicated from evidence, never dropped. | `src/data/validation.py` |
| Cherry-picked results | Append-only experiment registry recording rejections. | `experiments/registry.md` |

---

## Reports

- [`reports/research_report.md`](reports/research_report.md) — the full paper
- [`reports/generation2_report.md`](reports/generation2_report.md) — Generation 2: regimes, dynamic covariance, forecasts, attribution, and the engine erratum
- [`reports/generation3_report.md`](reports/generation3_report.md) — Generation 3: Bayesian portfolios, online learning, non-price data, an alpha engine, execution and capacity
- [`reports/generation4_report.md`](reports/generation4_report.md) — Generation 4: search-aware statistics, deep learning, text, the research database
- [`reports/generation5_report.md`](reports/generation5_report.md) — Generation 5: the framework, the strategy library as a search, adaptive allocation, frontier models, power and factor attribution
- [`docs/START_HERE.md`](docs/START_HERE.md) — a learning path, the technique guides, the glossary and the strategy cards
- [`reports/data_quality_report.md`](reports/data_quality_report.md) — Stage 1 findings
- [`reports/figure_index.md`](reports/figure_index.md) — every figure and the question it answers
- [`experiments/registry.md`](experiments/registry.md) — every experiment, including the rejected ones

---

## Limitations

Stated in full in the research report; the ones that most constrain the
conclusions:

- The universe consists of ETFs that exist and are liquid **today**. A
  universe chosen in 2006 would have included funds that later closed. This
  biases results upward and cannot be removed with this dataset.
- Generation 1 models transaction costs as linear in traded notional, which
  flatters the high-turnover strategies; Generation 3 adds a square-root impact
  model and measures capacity, with an impact coefficient that is an
  order-of-magnitude assumption.
- Twenty years is one macro cycle and a bit. It contains a single deflationary
  crisis and a single inflationary one, so regime conclusions rest on very
  few independent episodes.
- The final holdout has now been examined. It is no longer untouched, and any
  further work on this data cannot claim a clean out-of-sample test.
- These are backtests. They are not a live track record, and no result here
  includes slippage, financing, taxes or the effect of trading at scale.
- Generation 2 spends the same twenty years a second time, applies no
  multiple-testing correction across stages, starts its regime sample in
  December 2008, and uses macro data at today's vintages with modelled release
  lags rather than real-time vintages.

- Generation 5 adds thirteen stages on the same twenty years, with fourteen
  declared decisions and no correction across stages. Several of its null results
  are underpowered (Stage 39 says by how much), the causal and foundation-model
  stages rest on small or possibly contaminated samples, and no language model was
  ever called.

## Licence

MIT.
