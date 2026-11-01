# Assumptions and limitations (Releases 1 and 2)

## What the RL ledger models, and what it does not

- One currency, cash-equity accounting: cash, quantities, marks, a proportional cost plus an optional quadratic impact term, financing on debit cash and a borrow fee on short notional (both default to zero).
- Not modelled: futures margin and variation settlement, contract multipliers and rolls, FX and settlement calendars, corporate actions, borrow availability and recalls, partial fills, order rejection by a broker. Release 4 routes evaluations that
  need them through the existing event-driven engine and ledger.
- Prices are synthetic marks built from returns (the ledger starts at 1 and compounds returns); a return below -100% is a data error and stops the price at zero.
- The agent's private ledger starts flat at the first day of each chain. The platform backtest then charges the real transition between consecutive folds' books. The two are not the same account.
- Rewards are training signals, in percent per session. They do not sum to final equity.

## Data

- The ML futures universe is simulated and stays labelled so. Nothing in Release 1 uses it as evidence of a strategy.
- Sector and asset-class relationships used by the graph models are the dataset's current static map. They are not point in time, and the recorded relationship version says so (`static_group_map:<hash>`; `current classification, not point in time`).
- No historical constituents, borrow data, costs or fills were invented. None are used by Release 1.

## Statistics

- Release 1 makes no statistical claim about any strategy. Calibration quality, selection-bias control and uncertainty are Release 3 to 5 items.
- The HMM filter is checked against an independent log-space recursion; the Markov-switching filter is checked only against statsmodels' own. Both engines' fits start from random parameters, and the Markov-switching fit now passes its seed to statsmodels
  (which ignores numpy's global seed) and retries three seeds.

## Optional dependencies

- NGBoost depends on lifelines, which requires pandas below 3. On Python 3.11 or later it cannot be installed beside pandas 3, so the blocking `ml-correctness` job leaves it out and its serialization-matrix family is recorded as
  `NOT_APPLICABLE`; the Python 3.10 job runs it. A family skipped for any other reason makes its requirement `blocked`, not `pass`.
- The RL checks need Gymnasium and Stable-Baselines3; without them the environment tests skip, the ledger and reward tests still run, and the requirement is `blocked`.
- The GPU path has never been run.

## What this release does not do

Study lineage, holdout reservation, null controls, selection-test eligibility and the gate evaluator (Release 3); execution realism (Release 4); statistical and prospective validation (Release 5); alpha research,
strategy-improvement experiments and the advanced modules (Releases 6 to 8). Gate A, B and C are all `pending` or `not started` for every model.


# Release 2: causal data and training integrity

## Labels and calendars

- The label convention is daily close to close (`CONVENTION`). Intraday bars and executable-price conventions other than the close are not implemented; the convention is recorded, not selectable.
- Holidays are the data's own: a date with no row is not a trading day. Where a closure is carried as a price (the futures bundle with `hold_days`), the cell is a mark and is not tradable. The other loaders carry nothing over.
- The fold plan (`segment_plan`) cuts the history by fractions of the number of labelled dates, so appending data moves the fold boundaries. Prediction causality is a statement about a fixed plan on data of fixed length; the tests change values, never the length.
- Purged K-fold and CPCV exist as functions on label intervals. They train on later blocks and are research diagnostics (`chronological=False`); no experiment uses them yet.

## Point in time and sessions

- The platform has no macro vintage history: FRED serves the latest vintage, so the macro series are reported as `estimated` (a fixed release lag) except market prints and series computed from prices. Validation claims that rest on `estimated` or `unavailable` inputs are marked restricted. Nothing was backfilled or invented.
- The ML feature library has no fundamentals group. The fundamentals engine is reused for the availability join and its quality vocabulary is mapped, but no fundamentals feature enters a model yet.
- Sessions are a fixed table with daily bars. A ticker with no suffix, index symbol or asset class hint is assumed to trade US equity hours, and the report lists those assets. The macro loader applies release lags in days; release instants exist (`sessions.release_instant`) but it does not use them.
- Tokyo's close is modelled at 15:30 JST (the time since 5 November 2024). Before that the exchange closed at 15:00, so the model treats the bar as final half an hour later than it was: conservative, never early.

## Training and inference

- Perturbation checks run through six kinds (ridge, hmm_regime, lstm, gcn, model_ensemble, ppo). The other kinds of each family share the class and the binding but are not each run through the whole-walk-forward comparison.
- The pipeline has no data-driven feature-selection, dimensionality-reduction or probability-calibration stage. If one is added, it must fit on the eligible rows and be added to the isolation test.
- One-row-at-a-time inference equals the batch for the models without state through a call. The ensemble's dynamic weights and the PPO account carry state through a call: for them, a prefix of the block and a live replay (a view that ends on the decision date, given the block so far) give the batch values; a single isolated row would start the account flat.
- Sample weights are used by the linear, boosted-tree and regime models only; the others report `requested, not applied`. Sequential bootstrap is not implemented.
- The registry's `register_model` still stores plain model bytes; the inference bundle is written beside the experiment's model file. Experiments saved before this release have no bundle, no fold lineage and no audit, and say `not recorded`.
- A forward signal for a date the saved model had already seen is flagged as a replay and counted; it is not refused, because replaying history is a legitimate way to look at a model.

## Fixed in passing (behaviour that changed)

- Strategy combination by online learning back-filled the first volatility from later days and filled gaps with the whole-sample volatility; it now makes no update until 20 days of trailing volatility exist.
- The regime model's diagnostics described the prediction block as training data; they now stop at the fit window.
- `idio_vol_63` declared a 63-day lookback but needs 126.
