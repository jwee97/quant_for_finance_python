# Generation 4: Distributed Search, Deep Learning, Text, a Research Database, and the Engineering Around Them

### Four extensions to the Generation 1-3 platform, the three research questions held to rules written down before their results existed

**Universe and sample:** unchanged (15 ETFs, 2006-01-03 to 2026-09-21)
**Data version:** `f13af1fed1f7` · **FOMC statement corpus:** `41cf8d5c35c3` (171 statements) · **Earlier data:** macro `0cb282ac19f2`, non-price `d416cae26f99`, CFTC `b2ec476d7703`
**Config fingerprints:** core `12f9cfd14055`, Generations 1-2 `c25e66aa003a`, Generation 3 `4a4c779b0c8d` (all three unchanged and pinned by a test), Generation 4 scope `ea5f48a282e9`
**Reproduce with:** `python -m experiments.run_all --fresh` (all 29 stages, about 80 minutes on 4 cores) or `--generation 4` (about 10 minutes)

---

## 1. Executive summary

### 1.1 What was asked, and what was built

The instruction was "Continue with the last generation". The roadmap's Generation 4 lists five items: transformers, an LLM research assistant,
distributed experimentation, a research database and cloud deployment. All five were addressed, in different ways and to different depths, and
the differences matter:

| Roadmap item | What exists | What does not |
|---|---|---|
| Distributed experimentation (Priority 15) | A parallel executor (serial, joblib; dask and ray adapters), a 1,584-rule grid run on it, and the statistics a large search needs: White's Reality Check, Hansen's SPA, probability of backtest overfitting (Stage 26) | Ten thousand combinations (the grid is 1,584); the dask adapter passes a unit test on a local cluster but was not used for the grid; the ray adapter has not been run (ray is not installed here) |
| Modern deep learning (17) | A patch transformer and an MLP-mixer, trained walk-forward and tested against the Stage 19 ridge (Stage 27) | Zero-shot foundation models (Chronos, TimeGPT), the Temporal Fusion Transformer, TimeMixer, N-HiTS, N-BEATS |
| LLM research assistant (16) | A FOMC statement corpus, a validated extraction schema with a content-addressed cache, an offline lexicon backend, a hosted-model backend, a read-only SQL guard and a question-to-query assistant (Stage 28) | **No language model was called to produce any result.** There are no model credentials in this environment; the hosted-model paths are tested against fake clients only |
| Research database (14) | SQLite, rebuilt from the files, with acceptance checks and two command-line entry points (Stage 29) | DuckDB or PostgreSQL (SQLite needs no server) |
| Cloud deployment (and the software-engineering part of Priority 20) | A Dockerfile, a compose file, a GitHub Actions workflow, pre-commit, a Makefile, a lint gate and coverage reporting | **None of it was run end to end**: there is no Docker daemon or Actions runner here. No cloud provider is involved. Hydra, MLflow, Weights & Biases, a documentation site and a dashboard were not built |

Stages 26, 27 and 28 each pose a question with a decision rule committed before any result (Stage 26 `d625490`, Stage 27 `25d118d`, Stage 28 `047ca07`);
Stage 29's design and acceptance checks were committed before it was built (`51c056f`). Stage 26's declaration contained a misnamed option
(`price` for `raw`) that failed on its first run before any grid result existed; it was corrected in its own commit (Section 11).

### 1.2 The answer

**Three decision-bearing hypotheses were declared; one was retained, and its qualification is the finding.**

| Hypothesis | Decision | Numbers |
|---|---|---|
| Stage 26: after accounting for the size of the search, some rule in a grid of 1,584 has a positive expected net return | **Rejected** | Best rule net Sharpe 0.448; Reality Check p = 0.503; SPA p = 0.344 |
| Stage 27: a deep model forecasts 21-day returns better than the annually refitted ridge | **Retained**, with a qualification | MLP-mixer: CRPS −0.00056 (p = 0.019, BH-significant). The ridge is the benchmark that is bad: it is worse than predicting zero and worse than each asset's historical mean, and the mixer is not better than either |
| Stage 28: FOMC statement tone, change and action add information to price and macro | **Rejected** | No sleeve significant (smallest p = 0.27); mean out-of-sample R² −0.3% |

### 1.3 Seven findings worth stating plainly

1. **The search is the problem, not the rules.** 76% of the 1,584 rules have a positive net Sharpe over 2007-2026, the median is 0.12 and the best is 0.45, which looks like an
   answer until the search is priced in: a search of this size produces a best Sharpe of about **0.77 from pure noise**, the Reality Check p-value is 0.50, and the deflated Sharpe probability of the best rule is 0.08.

2. **The in-sample winner is, on average, a coin flip out of sample.** Probability of backtest overfitting 0.60; the rule that wins in-sample (mean Sharpe 0.64) has a mean out-of-sample Sharpe of 0.02, and across the 12,870 splits the
   relation between in-sample and out-of-sample Sharpe of the winner is **negative** (slope −1.08).

3. **The grid fades with time.** The share of rules with a positive net Sharpe is 82% in the development sample, 55% in validation and 44% in the final holdout.

4. **Parallelism changed the time and nothing else.** Serial and joblib runs of the same 24 rules agree exactly (maximum difference 0.0), and the 1,584-rule grid takes about 4 minutes on four cores against an extrapolated 15 serially (a speed-up of about 3.8 on the grid, 2.0 on the short 24-rule check, which includes process start-up).

5. **The mixer "beats" the ridge because the ridge is poor.** Against predicting zero, the ridge's CRPS is +0.00044 worse (p = 0.088); against each asset's historical mean, +0.00061 worse (p = 0.003). The mixer is −0.00012 against zero (p = 0.45) and +0.00006 against the historical mean (p = 0.56): no better than knowing nothing but an average.
   The patch transformer is *worse* than the historical mean (+0.00027, p = 0.043).

6. **A linear model on the same 252-day window is worse than either network** (CRPS +0.00050 against the Stage 19 ridge; both deep models beat it, p = 0.002 and 0.0001), and the 3-seed ensembles average over large seed-to-seed differences in rank IC (0.009 to 0.063 across the single patch-transformer seeds).

7. **FOMC text carries no forecasting information beyond price and macro here.** A transparent lexicon reads the announced rate action correctly for 94.2% of 171 statements, which makes it a usable baseline, and its tone, change and action features give a
   mean out-of-sample R² of −0.3% across five sleeves. This says nothing about what a language model could extract; it says what such a model has to beat.

---

## 2. Scope, assumption and method

**Assumption, stated once.** "The last generation" was taken to be Generation 4 as the roadmap lists it. The roadmap says of Generation 4 only five nouns; how each was built is a judgement, laid out in Section 1.1.
Priority 19 (Reality Check, SPA, probability of backtest overfitting) was pulled forward from the roadmap's list of "modern statistical methods" because a distributed search without them is misleading.

**Pre-registration.** `config/distributed.yaml`, `deeplearning.yaml`, `assistant.yaml` and `research_db.yaml`. Post-hoc diagnostics carry `posthoc` in the table name and the registry text and never overturn a declared decision. Section 11 lists every deviation.

**Identity of a result.** The earlier fingerprints are unchanged and the Generation 3 scope was redefined (it now names its namespaces explicitly) so that adding Generation 4 files cannot rename a Generation 3 result; a test pins all three.

**Timing.** Unchanged for returns. New: text features are available the day after a statement's date and enter the monthly panel at the latest month-end on or after that; the deep models see the 252 daily returns up to and including the origin, each divided by the EWMA volatility at the origin, and are trained only on rows whose labels were realised at least 21 days before the refit.

---

## 3. Distributed experimentation and honest search (Stage 26)

**Method.** 1,584 simple rules: 1,440 momentum rules (three variants, lookbacks 21 to 252 days, skips 0 to 21 days, volatility lookbacks) and 144 mean-reversion rules, each at three rebalance frequencies, two signal lags and with or without cross-sectional standardisation, all through the
Generation 1 position stack, a 10% volatility target and per-asset linear costs. Each rule is a pure function of its parameters and the read-only market data, so the grid runs as independent tasks in `src/distributed/executor.py`, and the results come back in task order. The common window begins when the slowest rule has a live book (2007-04-02, 4,899 days).

**Declared rule.** `h_search`: retained only if **both** White's Reality Check and Hansen's SPA reject, at 10%, the null that no candidate has a positive expected net return (benchmark: cash; stationary bootstrap, 2,000 draws, mean block 21 days).

| | |
|---|---|
| Rules with positive net Sharpe | 76% (momentum 81%, mean reversion 31%) |
| Median / best net Sharpe | 0.12 / 0.45 (ranked momentum, 189-day lookback, skip 5, monthly, lag 2) |
| Best of 1,584 noise rules (expected) | 0.77 |
| White's Reality Check p | 0.503 |
| Hansen's SPA p (lower / consistent / upper) | 0.279 / 0.344 / 0.369 |
| Probability of backtest overfitting (16 blocks, 12,870 splits) | 0.60 |
| In-sample winner: in-sample / out-of-sample mean Sharpe | 0.64 / 0.02 |
| Deflated Sharpe probability of the best rule (1,584 trials) | 0.08 |
| Against equal weight as the benchmark (reported) | Reality Check p = 1.000, SPA p = 1.000 |

**Outcome.** **Rejected.** The positive average of the grid is real in the sense that most momentum rules earn something after costs on this sample, and it fades from 82% of rules positive in the development sample to 44% in the final holdout; no rule can be distinguished from what a search of this size finds in noise.

**Engineering identity.** Serial and parallel runs of 24 randomly chosen rules agree exactly (asserted to 1e-12 in the run, pinned by a test). Wall-clock: 241 s for the grid on four cores, against 926 s extrapolated from the serial timing of the check.

*Figures 52-53.*

---

## 4. Modern deep learning (Stage 27)

**Method.** A channel-independent patch transformer (PatchTST-style: 21-day patches, 12 tokens, two layers, 18,753 parameters) and an MLP-mixer (TSMixer-style, 13,529 parameters) read the 252 normalised daily returns before each origin, with an asset embedding. They predict the 21-day return divided by the EWMA volatility forecast; the forecast is that times the **same** volatility forecast Stage 19 used, so only the conditional mean differs. Walk-forward on Stage 19's schedule (first fit on 1,260 days, annual refits, 21-day embargo), trained on every fifth day with the 15 ETFs pooled, early stopping on the last fifth of the training days after a 42-day gap, three seeds averaged. A ridge on the same window is a control. Each refit is an independent task in the Stage 26 executor.

**Declared rule.** `h_deep`: retained if at least one of the two deep models has a significantly lower mean CRPS than the Stage 19 price-only ridge (Diebold-Mariano on 187 monthly cross-asset means, two-sided, BH across the two, negative difference).

| 187 monthly origins from 2011-01 | Mean CRPS | Difference to ridge | p | Mean rank IC |
|---|---:|---:|---:|---:|
| Stage 19 ridge | 0.02301 | | | 0.045 |
| Patch transformer | 0.02267 | −0.00034 | 0.152 | 0.029 |
| **MLP-mixer** | 0.02245 | **−0.00056** | **0.019** | 0.049 |
| Linear on the window (control) | 0.02350 | +0.00050 | 0.140 | 0.022 |

**Outcome.** **Retained** by the rule: the MLP-mixer has a significantly lower CRPS than the ridge. **Qualification (post-hoc controls, `stage27_posthoc_controls.csv`; they do not overturn the decision):**

| CRPS, a minus b (negative favours a) | Difference | p |
|---|---:|---:|
| Ridge against predicting zero | +0.00044 | 0.088 |
| Ridge against each asset's historical mean | +0.00061 | 0.003 |
| MLP-mixer against zero | −0.00012 | 0.45 |
| MLP-mixer against the historical mean | +0.00006 | 0.56 |
| Patch transformer against the historical mean | +0.00027 | 0.043 |

The ridge is worse than doing nothing, and the mixer is not better than doing nothing; it beats the ridge by shrinking its forecasts toward zero (mean absolute forecast 0.0057 against 0.0082), not by finding information (rank IC 0.049 against 0.045, indistinguishable). A comparable thing happened in Stage 19, where Kelly sizing beat a classifier that itself lost; the lesson is to declare a historical-mean benchmark alongside any learned one.

Both deep models beat the linear control (p = 0.002 and 0.0001); the control is itself worse than the Stage 19 ridge, so it is not a strong reference either.

*Figures 54-55.*

---

## 5. Text features and the research assistant (Stage 28)

**What was built.** (1) `src/assistant/documents.py`: 171 FOMC statements from 2006-01-31 to 2026-09-16, cached as text with the source URL and sha256 of each page, each marked available the day after its date. (2) `extraction.py`: a schema (tone, change, action) with validation and clipping; a content-addressed cache keyed by text, schema and backend identity; an offline lexicon backend; and a hosted-model backend (temperature 0, JSON output, model and prompt hash in its identity) that is tested against a fake client. (3) `sqlguard.py`: a guard that admits one SELECT or WITH statement over six whitelisted tables, and a question-to-query assistant over the research database with a template backend (recognised questions only, otherwise a refusal) and a hosted-model SQL backend whose output is treated as untrusted and run on a read-only connection.

**No language model was called to produce any result in this repository.** The reported features come from the lexicon declared in `config/assistant.yaml` before any result.

**Declared rule.** `h_text`: for each of five sleeves, a nested Clark-West test of price + macro + text against price + macro (the Stage 15 and 23 machinery, first test year 2010); retained if at least one sleeve is significant at FDR 10% **and** the mean out-of-sample R² is positive.

**Outcome.** **Rejected.** No sleeve is significant (smallest p = 0.27, commodities); the mean out-of-sample R² is −0.3% over 166 months. Post-hoc matched-sample control: also negative (−0.2%) and not significant. The lexicon reads the announced rate action correctly for 94.2% of 171 statements (compared with the effective funds rate ten days either side), so the null is not an artefact of a broken reader. Rank correlations of the three features with next month's excess return of SPY, IEF, TLT, HYG and GLD are all between −0.11 and +0.11 (reported, not judged).

*Figures 56-57.*

---

## 6. The research database (Stage 29)

SQLite, rebuilt from the files on every run (`data/processed/research.db`, not committed). Tables: `experiments` (one row per registry entry), `metrics` (one row per numeric result), `result_files` (every CSV with its sha256), `figures`, `books` (every row of every performance table that has a Sharpe column) and `grid_results` (the Stage 26 grid). Six acceptance checks declared in advance all pass: every registry entry is present; every numeric result is a row in `metrics` with the same value; every CSV is indexed with a matching hash; every figure caption is present; the roadmap's example query (momentum rules with annual turnover below 8, best net Sharpe first) returns exactly the rows a direct pandas filter returns, in order; and a write through the read-only connection fails.

```bash
python -m src.research_db "SELECT stage, COUNT(*) n FROM experiments WHERE decision='retain' GROUP BY stage"
python -m src.assistant "which hypotheses were retained?"
```

**Caveat found while building it.** `books` is a raw harvest of every performance table, and a Sharpe ratio measured inside a crisis window (3.4) sits next to a full-sample one (0.85). An unfiltered "best books" query returned the former. Queries over `books` must name their source table; the example and the template question do (Section 11).

*Figure 58.*

---

## 7. Engineering (not a research stage)

Added: `Dockerfile` (Python 3.11, CPU PyTorch, runs the test suite by default), `docker-compose.yml`, `.github/workflows/ci.yml` (tests with coverage, lint, an image build and test), `.pre-commit-config.yaml`, a `Makefile`, `pyproject.toml` extras and a lint gate that the code passes (undefined names, unused imports, f-strings without placeholders). Test coverage of `src/` is 70% by line. The suite grew from 339 to 424 tests, including import smoke tests that would catch a broken stage script.

**What was not run.** There is no Docker daemon and no GitHub Actions runner in this environment, so the image was not built and the workflow did not execute; the workflow's YAML and the pre-commit configuration parse, and every command in them was run locally. No cloud provider is involved: "cloud deployment" here means that the environment and the pipeline are now reproducible in a container, not that anything is hosted.

**One regression caught.** Removing unused imports automatically also removed an import that was in fact used (`new_axes` in Stage 9, imported twice under two names); the linter's undefined-name check caught it before the full run, and a test now imports every stage script.

---

## 8. Reading the generation together

Generation 4's research stages follow the pattern of Generations 2 and 3, with one difference of kind. The search (Stage 26) and the text features (Stage 28) are nulls. The deep-learning result is a pass of a declared rule that dissolves under the controls, because the declared benchmark was weak. What Generation 4 contributes that the earlier generations did not is **a way to price a search**: the Reality Check, the SPA and the probability of backtest overfitting, run on a grid of the size the roadmap describes, say that the apparent edge of the best simple rule (Sharpe 0.45) is smaller than what chance alone produces from a search that size (0.77), and that the in-sample winner has no out-of-sample edge on average.

---

## 9. What Generation 4 establishes, and what it does not

**Establishes.** (1) A parallel executor whose results are identical to a serial run, with a test. (2) Search-aware statistics (Reality Check, SPA, PBO) tested on noise, a planted edge, and the Reality Check's known weakness. (3) On this universe and sample no simple rule in a grid of 1,584 survives its search. (4) A small patch transformer and mixer are not better than a historical-mean forecast. (5) An FOMC lexicon adds nothing out of sample. (6) The experiment record is queryable and verified, and a model-written query cannot write to it.

**Does not establish.** Anything about language models on text (none was run); anything about foundation models (not run, and their pre-training overlaps the sample); anything about larger or differently trained deep models (two small architectures, 15 series, about 15,000 training rows); anything about grids ten times the size or with other families of rule. The 1,584 rules are simple, correlated and mostly variations of one idea, so the effective number of independent trials is smaller than 1,584; the deflated Sharpe probability uses 1,584 anyway, the harsher choice.

**Limitations specific to Generation 4.** The same twenty years, now spent a fourth time; no multiple-testing correction across stages (three decisions were declared in Generation 4; the one retained dissolves under its own controls); the 94.2% action agreement is measured against a crude rule (effective funds rate change over twenty days); the research database has no concurrency or history and is rebuilt, not migrated.

---

## 10. Cumulative ledger

| Generation | Decision-bearing hypotheses | Retained | Rejected |
|---|---:|---:|---:|
| 2 | 23 | 3 | 20 |
| 3 | 9 | 2 | 7 |
| 4 | 3 | 1 | 2 |

Each retained result in Generations 2 to 4 carries a qualification stated in its own report; none is an economically exploitable edge.

---

## 11. Disclosed deviations and post-hoc analyses

Everything below happened after a first look at a result or after a failed first implementation. None overturned a declared decision.

| Where | What changed or was added | Why |
|---|---|---|
| Stage 26 | The declaration said `price_basis: [log, price]`; the code accepts `raw`. Corrected in its own commit | The run failed on it before any grid result existed |
| Stage 26 | A first run's common window began with every rule's idle-cash days; it was changed to begin when the slowest rule has a live book | The declaration said "when the slowest rule has a full history"; the first implementation did not follow it. The first run's headline (Reality Check p = 0.498, best Sharpe 0.447) differs from the final (0.503, 0.448) in the third digit |
| Stage 27 | Post-hoc controls: zero forecast and historical-mean forecast, and deep against the linear control | After the MLP-mixer passed the declared rule, to ask whether it was skill or shrinkage; the controls show the latter and the decision stays as declared |
| Stage 27 | The patch transformer's unit test needed 40 epochs to learn a planted signal; the declared training budget is 30 epochs with patience 5 | Test-only; the declared budget was not changed after seeing results |
| Stage 28 | `features.burn_in_statements` (16) is a default in the code, matching the declared text, but is not a key in the YAML | The declaration states the 16-statement burn-in in words |
| Stage 28 | Matched-sample control and rank correlations | Post-hoc, labelled |
| Stage 29 | The `best_books` example query and template question were restricted to the full-sample Stage 12 table | The first draft returned in-regime Sharpe ratios (Section 6) |
| Engineering | Automated removal of unused imports touched 35 files | One removal broke a stage; fixed, and an import test added (Section 7) |

---

## 12. Not built

- **Roadmap Generation 4:** zero-shot foundation models (Chronos, TimeGPT; their pre-training overlaps the sample); TFT, TimeMixer, N-HiTS, N-BEATS; any run of a language model; DuckDB or PostgreSQL; an actual cloud deployment.
- **Untested here:** the ray executor backend (the dask backend has a unit test on a local cluster); the Docker image; the GitHub Actions workflow; the hosted-model extraction and SQL backends against a real model.
- **Roadmap items outside Generation 4:** explainable machine learning (Priority 9), factor-model attribution (12), reinforcement learning (18), Bayesian model averaging (Priority 19, the rest of which was built), Hydra, MLflow, Weights & Biases, a documentation site and an experiment dashboard (Priority 20), Wishart and factor stochastic-volatility covariance models (Priority 4).
- **The rename.** The project keeps its name.

---

## Appendix A: Reproducing this report

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python -m experiments.run_all --fresh --download   # all 29 stages, ~80 minutes on 4 cores
python -m experiments.run_all --generation 4       # Generation 4 only
pytest -q
```

## Appendix B: Figures

Figures 52-58, each with the research question it answers in `reports/figures/*.txt` and `reports/figure_index.md`.

| Stage | Figures |
|---|---|
| 26 distributed search | 52, 53 |
| 27 deep learning | 54, 55 |
| 28 text features | 56, 57 |
| 29 research database | 58 |
