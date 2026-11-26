# Accounting and measurement conventions

What each engine means by a return, a cost and a position, where they differ on purpose, and how a saved result is measured when it is compared with another. Every statement below is backed by a test named in the last column of the tables; a convention with no test is marked as such.

## 1. Which engine is the authority

| Question | Authority | Why |
|---|---|---|
| What cash, positions and profit does an account hold? | `src/ledger` (the journal; every operation obeys a per-entry identity and `ledger.reconcile` recomputes cash, equity and quantities from it) | It is the only component that posts every movement (trade, fee, variation margin, funding, dividend, split, settlement, currency translation) as an explained entry. |
| What would a weight-based daily strategy have returned? | `src/backtest/engine.py` | Fast, vectorised, used by the Backtest tab and the Alpha Lab. It is checked against the ledger, not the other way round. |
| What happens to an order over time? | `src/engine` (event-driven) | Posts to the ledger; its reconciliation report is part of every run. |

## 2. Conventions of the weight engine and how each differs from the ledger

| Topic | Weight engine (`BacktestEngine`) | Ledger | Difference | Test |
|---|---|---|---|---|
| Return of a day | `w_{t-1}' r_t` on the equity at the start of the day | positions marked to market | none while the weights are the same (agree to 1e-12 with zero costs) | `test_accounting_cross_engine::test_a_daily_rebalanced_book_is_the_same_money...` |
| Cost | `cost x |traded weight|` taken from the day's return, i.e. a fraction of the previous close's equity | `cost x traded notional` at trade-time equity (which already contains the day's return) | cost x turnover x the day's return per trade: second order, bounded and tested | `...differs_between_the_engine_and_the_ledger_only_by_cost_times_turnover...` |
| Weight drift between rebalances | `drift_basis="gross"` (default, the historical convention): rescale to the same gross exposure after each return. `drift_basis="equity"`: divide by the growth of equity | an untouched account drifts by equity | **exact only for a fully invested long-only book.** With cash, leverage or shorts the gross convention re-trades the book to its gross exposure every day at no cost and with no turnover. Use `"equity"` when the book is not fully invested. The default is unchanged so saved research is reproducible. | `...drift_on_an_equity_basis_is_what_the_ledger_does...`, `...gross_basis_drift_overstates...` |
| Dividend | included in a total-return series (reinvested the same day) | a `dividend` cashflow, reinvested when the account is next rebalanced | none when the account is rebalanced to fully invested at the close | `...a_dividend_is_a_total_return...` |
| Split | the return series must be split-adjusted before the engine sees it | `split()` rescales quantity and price, equity unchanged | an unadjusted series shows a false loss | `...a_split_leaves_equity_alone...` |
| Perpetual funding | `short_carry` added to the return of the short | a `funding` cashflow on the notional | first order the same; the engine re-sizes to equity daily, the ledger holds its first size | `...a_short_perpetual_that_collects_funding...` |
| Futures roll | `holding_cost` (a yearly proxy) | close the near contract, open the far one: only the fees move equity, never the calendar spread | the engine's roll cost is an input, the ledger's is what was paid | `...a_futures_roll_changes_equity_only_by_its_costs...` |
| Option expiry | not modelled | position closed at intrinsic; equity change = intrinsic - premium | engine-only users cannot hold options | `...a_long_option_that_expires...` |
| Partial fills | not modelled (full target is traded) | each fill is booked as it happens; nothing is booked for the unfilled part | the engine assumes the whole target trades | `...partial_fill...` |
| Foreign currency | not modelled (one currency) | local return and currency translation are separate entries that add to the base-currency return | -- | `...a_foreign_holding_splits...` |

## 3. How a saved result is measured in the Research Comparison

* **Levels, not returns.** A saved result is a set of dated equity levels. The comparison measures **only on the dates every chosen result observed**. A return is the change between two consecutive common levels, so a result that did not trade on a date, started later or has a hole in its history is never given a zero return there (the old arithmetic reindexed to the first result's dates and filled the gaps with zero). Each result's own whole history is shown separately (`individual`), with a coverage table.
* **Calendar and annualisation.** Declared in the series metadata (`frequency`, `calendar`, `periods_per_year`, `return_type`, `currency`, `risk_free`) or inferred from the dates and flagged `inferred`: business days 252, a continuous (weekend-observing) calendar 365, weekly 52, monthly 12, quarterly 4, anything uneven annualised by elapsed time. The common grid's calendar decides the annualisation of the comparison; a note says when calendars or frequencies differ.
* **CAGR** is `(L_end / L_start)^(1 / years) - 1` with `years` = elapsed calendar days / 365.25.
* **Volatility, Sharpe, Sortino** use the periods per year of the common grid; for irregular spacing, variance per year of elapsed time. Sortino uses the downside deviation `sqrt(mean(min(excess, 0)^2))`.
* **Risk-free rate**: an annual effective rate accrued by elapsed calendar time between observations (a Friday-to-Monday return accrues three days of it).
* **Ranking** sorts by a metric in the registry (`src/research/metrics.py`) in that metric's own direction, and is refused when the metric does not apply to a kind, the horizons differ, or the units differ.

Caveat that no code can remove: scaling a per-period volatility by `sqrt(periods per year)` assumes returns are not serially correlated and that the periods are alike. For a strategy with autocorrelated returns the annualised volatility is mis-stated. The page says so in the conventions panel.

## 4. Measurements elsewhere in the code that still use a fixed 252

`src/backtest/metrics.py`, `src/framework/validate.py`, `src/alphalab/*`, `src/risk/performance.py` (`benchmark_relative` and the rolling functions), `src/research/regime_lab.py`, `src/assets/spread_lab.py` and `src/strategies/*` annualise with 252. They run on business-day series (equity ETFs and futures) where 252 is right. The crypto lab (`src/assets/crypto_*.py`) uses its own 365. Mixing the two in a comparison is exactly what the Research Comparison now handles; **inside each engine the constant is correct for the data it is fed but is not read from the data**, so a 7-day series pushed through an equity-calendar engine would be mis-annualised. That is recorded as a risk, not changed here.
