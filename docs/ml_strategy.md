# ML strategies: explicit decisions and the Strategy Explorer

A model's prediction is not a trade. Between them sit a *decision policy* (what to buy, sell and hold, how much, and when to get out) and an *execution* (what the engine actually did, when, and at what
cost). This page describes the three layers, the policy's settings, the ledger that reconciles them to the engine, and how to read the **ML Strategy Explorer** tab. The audit that motivated the work,
and what it found wrong with the shared pipeline, is in [the pipeline audit](ml_pipeline_audit.md).

## Three layers, kept apart

| Layer | What it is | Where it is recorded |
| --- | --- | --- |
| **Prediction** | The model's score for an asset on a date, and that score's cross-sectional rank (1 = best) | `predictions.parquet`, and `score` / `rank_pct` on every decision |
| **Decision** | BUY, SELL, SHORT, COVER, INCREASE, REDUCE, HOLD or a forced exit, with its reason, from that day's information only | `decisions.parquet` |
| **Execution** | The trade the backtest engine made `lag` days later, at that day's close, at the configured cost | `executions.parquet`, `round_trips.parquet` |

Selling a long position and opening a short are different actions (`SELL` and `SHORT`); closing a short is `COVER`. Shorts exist only when `decision_allow_short` is on.

## The decision policy (`signal.method = decision_policy`)

Scores are converted to a percentile rank each day, so a return forecast, a ranking score and a class probability are read the same way and the thresholds mean the same for every model.

| Setting (`signal.decision_*`) | Default | Meaning |
| --- | ---: | --- |
| `entry_quantile` | 0.2 | Open a position on an asset in the best (worst, for a short) share of the day's ranks |
| `exit_quantile` | 0.4 | Close it only when it falls out of this wider share. The gap is the hysteresis: a name is not sold for a one-place change of rank |
| `min_confidence` | 0 | Confidence needed to open (a held position is not closed for low confidence alone) |
| `max_positions`, `max_position` | 5, 0.25 | Most positions per side; largest weight of one |
| `sizing` | `equal` | `equal`, `inverse_vol` or `score` (by how extreme the rank is) |
| `long_exposure`, `short_exposure`, `cash_floor` | 1.0, 0, 0 | Gross weights when fully invested; the rest is cash |
| `allow_short` | false | Whether the SHORT and COVER actions exist |
| `stop_loss_vol`, `take_profit_vol` | none | Exit when the position's loss (gain) reaches this many times the entry-day volatility scaled to the horizon |
| `max_holding`, `cooldown` | none, 5 | Longest holding in days; days before a stopped-out name may be bought again |
| `rebalance_days`, `resize_band` | 5, 0.25 | Entries, rule exits and size changes happen on this schedule; stops and time exits are daily. A held position is resized only when its target moves by more than this share of itself |
| `min_edge_bps`, `forecast_is_return` | none, false | If set, a position opens only when the forecast return exceeds this edge plus the round-trip cost |

Timing is the platform's convention: a decision from the close of `t` is traded `lag` days later. Stop and take-profit levels are measured from the *executed* entry price (the close `lag` days after the
decision) and tested on each close after that, so a stop that fires on `t` is itself executed at `t + lag`. The policy never sees a price after the day it decides on; a test rewrites the whole future and
checks that no earlier decision changes.

Use `backtest.rebalance = on_change` with the policy (an engine mode added for it): the book trades only on days a target changed and drifts with prices in between. With a calendar schedule, a decision
between rebalance dates waits for the next one and the report says so.

## The ledger

`src/ml/trade_log.py` reads the engine's own book and trades and builds:

* **executions**: each trade, its kind (OPEN_LONG, CLOSE_SHORT, FLIP_TO_LONG, TRIM_LONG ...), the book before and after, the cost, and the decision behind it. A trade with no decision is the engine putting a
  drifted position back to target on a day it traded the book; it is labelled `DRIFT`.
* **round trips**: each position from its opening to its closing, with its own return, its contribution to the book, its costs, and why it was opened and closed.
* **analytics**: win rate, average win and loss, payoff ratio, profit factor, holding-period histogram, exposure, cash, turnover, the gross-to-net gap, and results by asset.

The ledger reconciles to the engine: the round trips' net contributions sum to the engine's net return and their costs to the engine's costs, to 1e-10 (a test).

## The ML Strategy Explorer tab

* **Build and run a strategy**: a model, the policy settings above and the costs. It trains walk-forward and saves the run; nothing about the policy is chosen on the evaluation period.
* **Tiles**: round trips, win rate, profit factor, average win and loss, holding time, exposure and cash, turnover, cost drag, gross and net return.
* **Price, position and trades**: the asset's close with a marker on every executed trade (triangles for buys and covers, inverted triangles for sells and shorts, a cross for a stop, target or time exit;
  hover for the reason) and the position held underneath.
* **Equity against the benchmark, drawdown, exposure and cash**, holding-period and profit-by-asset charts. Every chart has a Table view and CSV.
* **Decision log**: searchable (reasons, asset, action, dates), paged, holds on request.
* **Trades and round trips**.
* **Replay**: pick a date to see every asset's score and rank, the book going in, what the policy decided and why. The trades that followed are listed apart, because they were not known at the decision.
  The features behind each score are not stored with the run, so the replay shows the score and its rank, not the inputs.

## Is it better than random signals?

The **random-signal control** replays the run's decision policy, costs and trading pace with random scores in place of the model's, 40 times by default, and reports the share of those draws that did at
least as well (`src/ml/controls.py`, `POST /api/ml/strategy/controls`). The random scores are AR(1) processes whose coefficient is the model's own rank autocorrelation, so they change as slowly as the
model's do; a day-by-day shuffle would trade far more than the model and make it look better than it is. Beside it is a **beta-matched benchmark** (the market scaled by the book's causal beta): a long-only
book earns the market's return for being exposed to it, and that is not skill. The replay is checked against the saved run first (it reproduces the saved Sharpe ratio to 1e-9 or the page says it did not).
A share near 0.5 means the model is indistinguishable from noise at this pace of trading. Runs made before the control existed lack the saved price history and cannot be replayed.

## API

`GET /api/ml/strategy/runs`, `/api/ml/strategy?id=&asset=`, `/api/ml/strategy/decisions?id=&asset=&action=&q=&from=&to=&limit=&offset=&hold=`, `/api/ml/strategy/trades?id=&kind=executions|round_trips`,
`/api/ml/strategy/replay?id=&date=`, and `POST /api/ml/strategy/controls` (a job). The reads use what the run saved; nothing is recomputed.

## Limits

* The comparison with the equal-weight benchmark is descriptive. A long-only policy is comparable with it; a policy that holds cash or shorts is not like-for-like (the audit shows this on real numbers).
  Beta- and exposure-matched benchmarks exist (`src/ml/benchmarks.py`).
* Costs here are the engine's linear cost on traded notional. The order-level model (partial fills, participation limits, impact) is `execution.mode = realistic_cash_equity`.
* The policy's settings are yours. Choosing them on the same period you then report is a selection process that must be counted as such (the Tune tab and the trial counts do this); the Explorer
  does not search.
* A stop at `k` volatilities is a risk rule, not a source of return; the analytics report what it did to the result and nothing more.
