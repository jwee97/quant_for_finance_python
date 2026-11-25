# Options and Crypto dashboards: what the review changed

This page records how the critical review of the Options and Crypto dashboards was implemented, what each part rests on, and what is **not** available and why. Nothing here predicts a market.

## Data rules (both dashboards)

* **Observed-only is the default.** The page header of the Options, Futures and Crypto tabs has a *Data* switch: *Observed only* (default) or *Model data allowed*. Every request carries it (`data_mode=observed|model`). In observed-only mode the server refuses the platform's simulated chains, futures curves, coins, quotes and order books with an `observed-only mode:` error, and the page offers an *Allow model data* button. A refused view is never filled with a simulation; a view that cannot be computed returns `{"state": "unavailable", "reason": ..., "needs": [...]}` and the page prints it.
* **Server default.** `QUANT_OBSERVED_ONLY=0` makes the server default to model data (the test-suite and demos use this); a reader's own choice in the page wins.
* **Point-in-time.** Strategy validation decides at the close of a day and trades the next day's return; a jump on the last day does not change any earlier return (tested).
* **Direct exchange feeds stay.** Funding, order books and perpetual data come from the exchanges; Massive is the primary source for the market data it supports and is not used as a substitute for them.

## Crypto

Five workflows (`crypto.js` groups): **Market Overview** (Market, Coin deep dive), **Derivatives Intelligence** (Funding intelligence, Funding by venue, Basis, Carry P&L), **Liquidity & Arbitrage** (Liquidity, Across exchanges, Execution simulator), **Strategy Laboratory** (Strategy validation, the legacy weekday backtest, Signals) and **Portfolio Risk** (Liquidation risk).

| Item | Where | What it does |
|---|---|---|
| 24/7 calendar | `assets/crypto_research.calendar_audit`, `GET /api/crypto/calendar` | Counts missing calendar days and weekend marks; annualisation is 365. The older pipeline (`framework/crypto_bundle.py`) keeps weekday marks and 252 and is labelled *legacy* in the page. |
| Carry economics | `crypto_research.carry_pnl`, `GET /api/crypto/carry` | P&L = funding received + basis change + unhedged delta + collateral revaluation (coin-margined) + trading costs + financing + **residual**, checked against an independent total. Margin is tracked daily with no top-ups: margin calls, liquidation price, shock table, funding reversals (paid days, longest run, worst cumulative drawdown). |
| Executable arbitrage | `crypto_research.executable_cross`, `GET /api/crypto/executable` | Walks both real order books for the notional; reports quoted vs executable edge, price impact, fill fraction, fees (taker/maker), stale-quote detection by snapshot age, the inventory or collateral each pair needs, and a verdict that is never "confirmed". |
| Execution simulator | `crypto_research.simulate_execution`, `GET /api/crypto/execution` | Market, limit, and hypothetical TWAP/VWAP on a stored book; the depth-replenishment and volume-profile assumptions are parameters and are printed. |
| Derivatives intelligence | `crypto_research.funding_intelligence`, `GET /api/crypto/derivatives` | Funding per interval and annualised (365), percentiles, z-scores, regimes and their dates, cross-venue dispersion, the payment calendar, the basis percentile. |
| Strategy validation | `assets/crypto_validation.py`, `POST /api/crypto/validate` | Directional (time-series momentum, MA cross, breakout), relative value (BTC/ETH spread with an Engle-Granger test, cross-sectional momentum), derivatives (funding carry, basis convergence, funding dispersion), a volatility-target overlay, costs, leverage and short constraints, benchmarks, a block bootstrap and a walk-forward. |
| Liquidation risk | `crypto_research.portfolio_liquidation`, `POST /api/crypto/liquidation` | Cross-margin shocks with betas measured on observed data; isolated liquidation level per position. |

**Not stored, so shown as unavailable:** open interest by exchange, liquidation observations, the dated-futures term structure, daily traded volume. Each needs an exchange integration or a vendor history; the page lists what is needed instead of drawing an estimate.

## Options

The Strategy Studio is four workflows: **Opportunity scanner** (`options_studio.scan`), **Strategy builder** (real listed contracts priced at the bid, ask and your limits, plus *Ideas from a view*), **Scenario laboratory** (spot x volatility, spot x skew and spot x time heatmaps and an attribution to the underlying, volatility, skew, time, costs and an explicit residual) and **Trade validation** (listed, two-sided, fresh, narrow, fillable, bounded loss, margin).

* *Ideas from a view* ranks structures by what they pay **if the stated hypothesis is right** and shows what they lose if it is wrong; excluded structures list their reasons. It does not choose the hypothesis.
* **Synthetic validation vs historical backtest.** The rolling-strategy backtest is labelled *Synthetic validation (model market)* and carries its limitations (`BACKTEST_LIMITATIONS`) and a margin block (peak, average, margin-call days, return on margin; measured, never charged). `GET /api/options/backtest-modes` reports a historical backtest as **unavailable**: it needs at least 60 dated bid/ask chains (the Snapshot button builds them) or a vendor history with historical quotes. The two are never ranked together.

## Experiments

`assets/experiments.py` gained the kinds `options_studio`, `options_scenarios`, `options_validation`, `crypto_validation`, `crypto_carry`, `crypto_execution`, `crypto_liquidation` and `crypto_derivatives`; each view has a *Save this run* card and a `metricsOf` entry so a saved run can be repeated and compared.

## Limitations

* Nothing here was run against a live exchange or a live Massive key; the tests use synthetic series, fake books and fake feeds.
* The carry decomposition treats the inverse (Deribit) perpetual's convexity as residual rather than modelling it in coin.
* Cross-venue edges use one snapshot read venue by venue, not simultaneously; transfer time, inventory limits and latency are listed as not modelled.
