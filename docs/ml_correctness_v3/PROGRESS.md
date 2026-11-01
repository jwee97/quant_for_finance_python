# ML v3 implementation progress

Current release: **1 — numerical and model correctness** (delivered in part; see the traceability table for what is and is not done). Releases 2 to 8 are `not_started`.
Checkout: branch `claude/multi-asset-quant-platform-uwqect`, built on commit `31bcae6`. Source tree hash of the code the checks hashed: see each check's `subject_hash` in the registry (`ml-verify --json`).
Updated: 2026-11-01.

## What exists now

- **RL portfolio ledger and environment** (`src/ml/rl/ledger.py`, `env.py`, `agent.py`): a quantity-and-cash ledger, one state transition for training and for playing a policy forward, an event timeline with pending orders, a corrected drawdown
  penalty, a stated differential-Sharpe convention, rejected/clipped/forced actions with reasons, mandatory eligibility liquidation and hard position/exposure limits that outrank the optional turnover budget, ruin handling, and a state schema
  version (2) so that agents trained under the old state are refused with a retrain message.
- **Rankers** (`boosting.py`): query groups and relevance recomputed after the label filter; excluded dates and reasons recorded.
- **Graph networks** (`graph.py`): absent nodes are removed from both ends of every edge at every layer (GCN degrees, GAT keys and queries, biases, residuals); the graph's cutoff, window, asset order and relationship version are recorded.
- **Output contracts** (`contracts.py`): typed output kinds, a position-rule/output-kind compatibility check that runs before training and again on re-backtest, `signal.reinterpret_outputs` as the one explicit override, validators for
  probabilities, quantiles (with the raw crossing rate), predictive distributions and pinball loss, and `report["output_contract"]`.
- **Serialization** (`models.py`): a checksummed `QFM1` envelope (legacy artifacts still load), `ModelArtifactError` for damaged files, `ModelSchemaError` for missing features.
- **Architecture manifest** (`deep/manifest.py`) with property tests; reduced variants are labelled as reduced.
- **Registry migrations and verification evidence** (`registry.py`, `verification.py`, `webapp/ml_verify.py`): additive versioned migrations, an append-only `verification_checks` table, `ml-verify` on the command line and
  `GET /api/ml/verification` / `POST /api/ml/verification/run` in the API. Only a run of the tests writes a status.
- **CI**: a blocking `ml-correctness` job (`ml-verify --run --require-pass`), the new files in the `ml-base` and Python 3.10 jobs.

## Actual test runs

| Command | Environment | Exit code | Passed/failed/skipped | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `python -m src.cli ml-verify --run --require-pass --registry <tmp>` | Python 3.11, pandas 3.1.0, numpy 2.4.6, torch 2.14, SB3 2.9, statsmodels 0.15 (a copy of the Python 3.11 CI environment, NGBoost importable) | 0 | 13 of 13 requirements `pass`; 33 + 15 + 4 + 10 + 14 + 18 + 4 + 17 tests across the eight files | registry `verification/*.xml` (JUnit), `verification_checks` rows | the first full run after the last code change to the verified sources |
| ML regression group: `tests/test_ml_*.py` (existing and new), `test_webapp_assets_ml.py`, `test_optional_packages.py`, `test_hpo_search.py`, `test_docs.py`, `test_webapp_ml_models_browser.py`, `-n 4` | same | 0 | 487 passed, 0 failed, 0 skipped, 264 s | pytest output of the run | run after the last code change; an earlier run of the same group found one failure, a test that expected the old warning for an agent ranked by another rule, which is now a refusal and the test was rewritten |

CI results are not claimed here: they are recorded when the run completes.

## Gate evidence

| Model / strategy version | Algorithm Verified | Strategy Validated | Operationally Ready | Reasons |
| --- | --- | --- | --- | --- |
| every model kind | pending | not started | not started | Gate A needs the Release 2 causal-view and Release 3 research-control evidence as well as the Release 1 numerical checks; the Release 1 checks above are inputs to it, not the gate. No gate evaluator exists yet. |

## Compatibility and migration

- Existing experiments, registry rows and model files are untouched. The registry gains `schema_migrations` and `verification_checks` (additive; opening an old registry upgrades it in place, in one transaction per migration).
- Model files saved before this release have no checksum envelope and still load; new ones carry it.
- RL agents trained before this release have state schema 1 and say `retrain it` when asked to predict. Their saved results are unchanged and readable.
- An experiment that uses a position rule the model's output does not support is now refused (`target_weights` with a non-agent model, a rank rule with an agent, a volatility or drawdown forecast ranked as if it were attractiveness). The
  refusal names the replacement and `signal.reinterpret_outputs` as the explicit override. Saved experiments are not re-evaluated.
- The RL environment's observation grew by one vector (the pending order) and its reward timing changed (a book's return is credited in the step in which it is earned). Reward magnitudes are comparable to before; training curves of old and new agents are not.

## Data and optional dependency limitations

See `ASSUMPTIONS_AND_LIMITATIONS.md`.

## Next executable actions

1. Release 2: restricted dataset views for every family that binds history; label/event intervals and purging on the real calendar; point-in-time joins for fundamentals and macro vintages; session-aware alignment.
2. Release 3: study lineage for every entry point, full provenance hashing, the holdout access ledger, the null-control runner, the selection-test eligibility layer and the gate evaluator.
3. Finish the Release 1 gaps listed against P0-01, P0-02, P0-04, P0-06, P0-08, P0-09, P0-10, P0-11, P0-12 and P0-30 in `TRACEABILITY.md`.
4. A Model Verification view in the dashboard (the API and CLI exist; the page does not).
