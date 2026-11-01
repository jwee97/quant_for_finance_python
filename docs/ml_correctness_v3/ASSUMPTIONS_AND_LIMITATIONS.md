# Assumptions and limitations (Release 1)

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

Causal dataset views (Release 2); study lineage, holdout reservation, null controls, selection-test eligibility and the gate evaluator (Release 3); execution realism (Release 4); statistical and prospective validation (Release 5); alpha research,
strategy-improvement experiments and the advanced modules (Releases 6 to 8). Gate A, B and C are all `pending` or `not started` for every model.
