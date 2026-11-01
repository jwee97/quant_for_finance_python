# Requirement traceability

One row for each of the 86 original requirement IDs. Status words are the specification's: `not_started`, `in_progress`, `implemented_unverified`, `verified`, `blocked`. `verified` means the behaviour exists and an independent check ran and passed; a partly delivered item stays `in_progress` with its gap in the last column.

| ID | Requirement | Status | Implementation | Independent tests / evidence | API / UI / DB | Limitations / blockers |
| --- | --- | --- | --- | --- | --- | --- |
| P0-01 | RL holdings, cash, drift, financing, leverage, shorts and ruin reconcile with an independent quantity ledger | in_progress | src/ml/rl/{env,ledger,agent}.py | tests/test_ml_rl_accounting_reference.py (shadow ledger, golden fixtures, random paths) | report: model_extras.rewards; `ml-verify` | single-currency cash-equity accounting only: futures margin, FX and corporate actions are not modelled in the RL ledger (Release 4 routes them through src/ledger) |
| P0-02 | Same observations and actions give the same states in training, evaluation, reload and chunked replay | in_progress | src/ml/rl/{env,ledger,agent}.py | tests/test_ml_rl_accounting_reference.py::observation_and_equity_... ; tests/test_ml_rl.py::books_do_not_depend_on_how_many_rows... | state_schema in diagnostics | prospective step-by-step stepping with persisted recurrent state is not implemented: every call replays from the start of its chain (deterministic, but O(days)) |
| P0-03 | Reward arrays (return, risk-adjusted, differential Sharpe) against independent references | verified | src/ml/rl/ledger.py (differential_sharpe, drawdown_increase), env.py | tests/test_ml_rl_accounting_reference.py (independent loops: constant, alternating, near-zero variance, 1e6 values) | - | - |
| P0-04 | Actions cannot violate hard constraints; forced actions are explicit events | in_progress | src/ml/rl/env.py (check_action, enforce_limits, _execute) | tests/test_ml_rl_accounting_reference.py (rejected/clipped actions, forced liquidation, cap/gross after the turnover budget, 400 random draws) | info['events'] | events are returned by the environment and the rollout trace; they are not yet persisted as registry audit rows |
| P0-05 | Ranker groups recomputed after the label filter; excluded dates and reasons recorded | verified | src/ml/boosting.py (RankerModel.fit) | tests/test_ml_ranker_groups.py | diagnostics()['excluded'] | - |
| P0-06 | Graph fitted on eligible historical returns and PIT relationships; cutoff, asset order, relationship version recorded | in_progress | src/ml/graph.py (fit, graph_info_) | tests/test_ml_graph_masks.py::the_graph_is_built_from_returns_up_to_the_fit_cutoff... | diagnostics()['graph'] | sector/asset-class relationships are the dataset's current static map, labelled `current classification, not point in time`: no PIT relationship source exists in the repository |
| P0-07 | Absent nodes excluded at every message-passing layer (GCN degrees, GAT keys and queries, biases); all-masked finite | verified | src/ml/graph.py (GraphNet) | tests/test_ml_graph_masks.py (induced-subgraph oracle for GCN and GAT, junk and NaN in absent slots, added absent nodes, all-masked forward and backward) | - | - |
| P0-08 | Architecture manifest and architecture-specific reference/property tests | in_progress | src/ml/deep/manifest.py | tests/test_ml_architecture_contracts.py (patching, N-BEATS residual unrolling, N-HiTS multi-rate pooling and interpolation, TCN, TFT layout); tests/test_ml_deep.py | manifest in the catalogue is not yet in the API | LSTM/GRU have shape/axes tests only; PatchTST, N-BEATS, N-HiTS and the TFT are labelled REDUCED/COMPACT and say what they leave out |
| P0-09 | Probability, scale and quantile validators; crossing rate recorded | in_progress | src/ml/contracts.py; boosting.py, regimes.py, deep/model.py | tests/test_ml_output_contracts.py | diagnostics()['raw_quantile_crossing_rate'] | calibration quality is a Release 5 item (P1-22); NGBoost has the validator but no independent distribution test in the base environment |
| P0-10 | HMM forward recursion in log space against an independent implementation; filtering causal; ordering and reload | in_progress | src/ml/regimes.py | tests/test_ml_hmm_reference.py | - | the Markov-switching engine is checked against statsmodels' own filter only (no independent oracle); fit seeding and retry are tested |
| P0-11 | Output contracts enforced across construction, signals, optimizer, backtest and forward interfaces | in_progress | src/ml/contracts.py; experiments.py (run_experiment, evaluate_run) | tests/test_ml_output_contracts.py; tests/test_ml_integration.py | report['output_contract']; signal.reinterpret_outputs | enforced for the position rule at experiment time and re-backtest; the optimizer (Release 6), forward step and ensemble member mixing do not check kinds yet |
| P0-12 | Save and reload every family with preprocessing, order, uncertainty heads and state; corrupt and mismatched artifacts refused | in_progress | src/ml/models.py (QFM1 envelope, ModelArtifactError, ModelSchemaError) | tests/test_ml_serialization_matrix.py (12 families, truncation, bit flip, foreign file, legacy artifact, missing and reordered columns) | - | 12 of the catalogue's 21 new kinds plus two base kinds are in the matrix; the other networks are covered by tests/test_ml_deep.py |
| P0-13 | Per-row label intervals and execution convention | not_started | - | - | - | - |
| P0-14 | Feature dependency/availability audit; stale and imputed values | not_started | - | - | - | - |
| P0-15 | Point-in-time fundamentals and macro vintages | not_started | - | - | - | - |
| P0-16 | Session-aware cross-market alignment | not_started | - | - | - | - |
| P0-17 | Purge by actual label/event intervals | not_started | - | - | - | - |
| P0-18 | Fit-only preprocessing, selection, calibration, early stopping and HPO | not_started | - | - | - | - |
| P0-19 | Causal OOF stacking and meta-training | not_started | - | - | - | - |
| P0-20 | Sequence windows and inference parity | not_started | - | - | - | - |
| P0-21 | Event concurrency, uniqueness and sample weights | not_started | - | - | - | - |
| P0-22 | Inference bundle: names, order, dtypes, preprocessing, state | not_started | - | - | - | - |
| P0-23 | Complete study and trial lineage | not_started | - | - | - | - |
| P0-24 | Full provenance hashing and reproducibility | not_started | - | - | - | - |
| P0-25 | Protected holdouts and access ledger | not_started | - | - | - | - |
| P0-26 | Null-control pipeline study | not_started | - | - | - | - |
| P0-27 | Selection-test eligibility layer | not_started | - | - | - | - |
| P0-28 | Adaptive feature-discovery registration | not_started | - | - | - | - |
| P0-29 | Unit, reference, integration, property, golden and leakage tests in blocking CI | in_progress | .github/workflows/ci.yml (ml-correctness, ml-base, replit-python) | src/ml/verification.py; `ml-verify --run --require-pass` | /api/ml/verification | the causal/leakage part belongs to Release 2; scheduled heavy lanes (null studies, multi-seed RL) do not exist yet |
| P0-30 | Fills, positions, cash, equity, P&L, costs and leverage reconciled with shadow calculations | in_progress | src/ml/rl/{env,ledger,agent}.py | tests/test_ml_rl_accounting_reference.py | - | RL ledger only; the vectorised and event-driven engines are reconciled in Release 4 |
| P1-01 | Dated universe membership and delistings | not_started | - | - | - | - |
| P1-02 | Corporate actions | not_started | - | - | - | - |
| P1-03 | PIT liquidity and tradability screens | not_started | - | - | - | - |
| P1-04 | Decision/submission/fill timing | not_started | - | - | - | - |
| P1-05 | Itemised costs | not_started | - | - | - | - |
| P1-06 | Configurable impact with calibration provenance | not_started | - | - | - | - |
| P1-07 | Order lifecycle states | not_started | - | - | - | - |
| P1-08 | Borrow availability and recalls | not_started | - | - | - | - |
| P1-09 | ADV participation and capacity | not_started | - | - | - | - |
| P1-10 | Closed sessions, stale marks, halts, delistings | not_started | - | - | - | - |
| P1-11 | Existing ledger as authority | not_started | - | - | - | - |
| P1-12 | Multi-currency accounting | not_started | - | - | - | - |
| P1-13 | Futures and crypto derivatives conventions | not_started | - | - | - | - |
| P1-14 | Coverage and capability report | not_started | - | - | - | - |
| P1-15 | Nested chronological selection | not_started | - | - | - | - |
| P1-16 | HAC and block-bootstrap inference | not_started | - | - | - | - |
| P1-17 | Eligible selection tests (DSR, PBO, RC, SPA, FDR) | not_started | - | - | - | - |
| P1-18 | Matched benchmarks and factor exposure | not_started | - | - | - | - |
| P1-19 | Predeclared regime and stress evaluation | not_started | - | - | - | - |
| P1-20 | Neighbourhood robustness | not_started | - | - | - | - |
| P1-21 | Delay and capacity decay on the real timeline | not_started | - | - | - | - |
| P1-22 | Calibration and conformal evaluation | not_started | - | - | - | - |
| P1-23 | RL multi-seed evaluation | not_started | - | - | - | - |
| P1-24 | Prospective commitments and settlement ledger | not_started | - | - | - | - |
| P1-25 | Server-side gate and risk governance | not_started | - | - | - | - |
| P1-26 | Cross-study holdout access tracking | not_started | - | - | - | - |
| P2-01 | Raw, relative, sector-relative and residual targets | not_started | - | - | - | - |
| P2-02 | Rolling IC, decay, coverage, stability, redundancy | not_started | - | - | - | - |
| P2-03 | Stable feature selection inside training | not_started | - | - | - | - |
| P2-04 | Cost-aware portfolio optimization | not_started | - | - | - | - |
| P2-05 | No-trade regions | not_started | - | - | - | - |
| P2-06 | Calibrated sizing | not_started | - | - | - | - |
| P2-07 | Covariance estimators | not_started | - | - | - | - |
| P2-08 | Trade filtering by net edge | not_started | - | - | - | - |
| P2-09 | Multi-horizon comparison | not_started | - | - | - | - |
| P2-10 | Causal regime exposure | not_started | - | - | - | - |
| P2-11 | Meta-labelling with settled outcomes | not_started | - | - | - | - |
| P2-12 | Regularised ensemble weights | not_started | - | - | - | - |
| P2-13 | Signal and portfolio correlation | not_started | - | - | - | - |
| P2-14 | Perturbation grid | not_started | - | - | - | - |
| P2-15 | Delay sensitivity | not_started | - | - | - | - |
| P2-16 | Capital- and liquidity-aware allocation | not_started | - | - | - | - |
| P2-17 | Simplicity tie-break | not_started | - | - | - | - |
| P2-18 | Incremental signal contribution | not_started | - | - | - | - |
| P3-01 | Drift monitoring | not_started | - | - | - | - |
| P3-02 | Retraining triggers and rollback | not_started | - | - | - | - |
| P3-03 | Bayesian model averaging | not_started | - | - | - | - |
| P3-04 | Distributionally robust allocation | not_started | - | - | - | - |
| P3-05 | Multi-horizon combination | not_started | - | - | - | - |
| P3-06 | Cross-asset transfer | not_started | - | - | - | - |
| P3-07 | Feature/strategy discovery controls | not_started | - | - | - | - |
| P3-08 | Off-policy RL estimators | not_started | - | - | - | - |
| P3-09 | After-tax accounting | not_started | - | - | - | - |
| P3-10 | Champion/challenger | not_started | - | - | - | - |
| P3-11 | Calibration and uncertainty monitoring | not_started | - | - | - | - |
| P3-12 | Reproducible governance reports | not_started | - | - | - | - |
