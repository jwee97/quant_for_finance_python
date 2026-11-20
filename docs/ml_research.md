# ML research workspaces: comparison, features, simulation and evidence

Four pieces that sit around the [Strategy Explorer](ml_strategy.md): a like-for-like **Model Comparison**, the **feature registry** with ablations, a **scheduled walk-forward simulation**, and the
**Research Evidence** checklist. Each reads or runs through the same pipeline as every other experiment, and each says what it does not show.

## Model Comparison (`src/ml/strategy_compare.py`, `POST /api/ml/compare-strategies`)

Up to six models are trained on the **same dataset and splits**, traded by the **same position rule, costs and lag**, and set beside baselines run through the same engine **over the same days**:

| Baseline | What it is |
| --- | --- |
| Equal weight, rebalanced monthly / held | The universe, with and without rebalancing |
| 12-1 momentum, top 30% | The best 30% by return over the last year skipping the last month |
| 5-day reversal, top 30% | The 30% that fell most over the last five days |
| Trend | Names whose 50-day average is above their 200-day average |
| Equal weight, 10% volatility target | The universe scaled to a volatility target |
| Random-signal books (mean of n) | Random scores at a model-like pace: what no skill earns |

**Predictive quality** (rank IC, its information ratio, share of positive days, calibration slope, the error of the score against the error of a forecast of nothing) and **trading quality** (CAGR,
Sharpe, Sortino, maximum drawdown, information ratio against equal weight, beta, return over a beta-matched benchmark, turnover, cost drag, exposure) are separate tables: a model can rank well and not
trade profitably after costs, and the reverse. Capacity is not estimated (it needs volume and an order size). Several models on one history is a multiple comparison; the best row is not corrected for being
the best.

## Feature registry and ablations (`src/ml/feature_registry.py`, `GET /api/ml/features`, `POST /api/ml/features/study`)

Every feature has its **data source**, **when it becomes known** (the rule that keeps a feature at the close of day `t` from using anything later), **how it is calculated**, and the **hypothesis and
reference** that make it worth testing (Jegadeesh and Titman 1993 for momentum, Jegadeesh 1990 for reversal, Ang et al. 2006 for volatility, Amihud 2002 and Gervais et al. 2001 for volume, Koijen et al. 2018
for carry, Lopez de Prado 2018 for fractional differencing, and so on; a practitioner hypothesis with no academic reference says so). A hypothesis is a reason to look, not a result.

The study measures each feature on a dataset: its mean rank information coefficient against the target, its t-statistic, and the share of years in which it kept its overall sign. The **ablation** retrains the
same model on the same splits with all features, without each group, and on **noise alone**, and reports the change in rank IC and net Sharpe. A change smaller than the gap between the full model and the
noise run cannot be told from nothing, and removing a group cannot create information: a model that does better without one was being hurt by its noise. One run per variant on one history; the differences are
not tested for significance.

## Scheduled walk-forward simulation (`src/ml/simulation.py`, `POST /api/ml/simulate`, "scheduled simulation" in the Strategy Explorer builder)

One call runs the loop a live system would run and records it:

1. the dataset's point-in-time `investable` mask decides who can be held on each date;
2. features and labels are built causally; targets that look `lag + horizon` days ahead are purged from every training window, plus the embargo;
3. the model is refitted on a schedule (`split.refit_every`, expanding or rolling); each fit is a recorded **model version** with its fitting day, the last day its labels reached, a hash of exactly the rows it
   trained on, the dates it predicted, the model and its parameter hash, the feature-list hash and the seed;
4. at each refit the **decision policy is chosen from a grid using only the days before that date** (less the embargo and the horizon): the candidate with the best Sharpe ratio over a trailing window, ties
   to the default, the default kept when there is too little history. The grid is limited to 12 candidates and each candidate counts as a trial;
5. the chosen policy goes into the run as a **dated schedule** (`signal.decision_schedule`): from its date the policy changes, and the book, entry prices and cooldowns carry over unchanged, so the saved run is
   one coherent book and ledger that the Strategy Explorer reads like any other;
6. every trade, cost, cash balance and holding is recorded.

The simulation block saved with the report holds the model versions, the choices and the score each was chosen on, the framework's own out-of-sample check (`oos_violations`, which must be empty), the
configuration hash, dataset key and seed, and the scheduled result beside each fixed policy over the same days. The best fixed policy is hindsight and is not what a person could have chosen in advance.

**Reproducibility**: the configuration (including the schedule), the dataset key and the seed are in the report; running the same configuration on the same data reproduces the run. The model versions' row
hashes let a later refit be checked against exactly what it trained on.

## Research Evidence (`src/ml/evidence.py`, `GET /api/ml/evidence`)

A checklist for one experiment. **Every line applies a stated rule to a number the run already recorded** and names where the number came from; nothing is recomputed. The lines: features use only what was
known (the dataset audit); every prediction came from a model that had not seen its date (the fold lineage); the final test history was kept out of the choices (the holdout ledger); the result survives the
number of things tried (the deflated Sharpe); it kept working on data it was not tuned on (the segment Sharpe ratios); it survives higher costs, delays, noise and other starts (the stress tests); costs are
modelled at the order level; the benchmark is the same kind of bet (net exposure); it beats random signals (the stored control, if it was run); the retraining loop was out of sample (simulations); the run
can be reproduced; the data are real; the run raised no warnings. Statuses are pass, warn, fail or **unknown**: an unknown is something nobody has checked, never a pass. There is no combined score: a
strategy with ten passes and one fail is not nine-tenths of a strategy, and the list says which line is the problem.

## Tests (the validation layer)

Each test builds a synthetic world whose answer is known and checks the component against it, apart from the others. A perfect-foresight signal appears only as a diagnostic control.

| Component | Test file |
| --- | --- |
| Timing, cash, costs, compounding, benchmark, trimming | `test_ml_pipeline_reference.py` |
| Signal generation, entry, exit, sizing, stops, causality, the ledger reconciliation, the dated schedule | `test_ml_decisions.py`, `test_ml_simulation.py` |
| Eligibility and historical membership, screens, portfolio construction, walk-forward, leakage | `test_universelab.py` |
| Feature registry, per-feature evidence, ablation against noise | `test_ml_feature_registry.py` |
| Model and baseline comparison | `test_ml_strategy_compare.py` |
| Evidence rules | `test_ml_evidence.py` |
| API routes and the workspaces in a browser | `test_ml_strategy_api.py`, `test_ml_research_api.py`, `test_universelab_api.py`, `test_webapp_strategy_browser.py`, `test_webapp_universe_browser.py`, `test_webapp_research_pages_browser.py` |

## Limits

* One history. Every difference here is measured on a single path of the market; the bootstrap, the deflated Sharpe and the random-signal control say how much of it could be luck, not that it is not.
* Choosing the policy grid, the models or the features on the same history you report is a selection process the trial counts only partly capture.
* The comparison and the ablation train models several times: on the full 20-year ETF universe each model takes about a minute.
