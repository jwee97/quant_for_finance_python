# Options and Futures research workspace: audit, corrections and limits

## What was audited and what changed

| Area | Finding | Change |
|---|---|---|
| Pricing, Greeks, IV | Black-Scholes, Black-76, Bachelier, binomial American, Heston/Merton/Lévy checked against identities (parity, put-call, finite-difference Greeks, IV round trip, model limits) | No formula errors found; `tests/test_derivatives_audit.py` now pins them. `bsm_greeks` handled a scalar call flag wrongly and is fixed |
| Arbitrage | European parity applied to American quotes | American chains now test parity as bounds (`F e^{-rT}-K ≤ C-P ≤ S-K e^{-rT}`); calendar and butterfly checks kept |
| Gamma exposure | Open interest was presented as if it revealed dealer positions | Gross gamma (observed) is separate from net gamma under a stated dealer-positioning scenario (`calls_long_puts_short`, `dealers_short_all`, `dealers_long_all`, `calls_short_puts_long`, `custom:<c>,<p>`) and a dealer share; the result says when the sign depends on the scenario |
| Futures roll | "Roll cost" mixed a calendar spread (a price difference) with execution cost | Decomposed into calendar spread, implied roll-down (a scenario), execution cost, realised excess return and realised roll contribution |
| Expiry rules | One `last_business_day` style rule for every root | `cme_crude`, `cme_natgas`, `third_last_business_day` (holidays are not modelled; the response says so) |
| Provenance | Each view labelled its own data | One schema (`observed_live`, `observed_delayed`, `historical_observed`, `calibrated_model`, `synthetic`, `mixed`) with freshness and assumptions, shown as a badge on every view |

## New analytics

* **Surface research** (`GET /api/options/surface/research`): per-expiry SVI with bid/ask-weighted fit, residuals against bid/ask bands, butterfly g(k), calendar total-variance monotonicity, joint SSVI with its sufficient conditions.
* **Model validation** (`POST /api/models/validate`): calibration with iv / vega / bid-ask-band loss, out-of-sample quotes (by strike or expiry), parameter stability, residual grid, BIC, scorecard that does not rank on in-sample RMSE alone.
* **Backtest validation**: stationary block bootstrap intervals, walk-forward setting selection, and a realism label that stays `simplified` unless every essential friction is modelled (`/api/options/strategy/walkforward`, `validation` block on the options and futures backtests).
* **Term-structure research** (`GET /api/futures/research`): calendar-spread matrix, spread history and z-score, contango/backwardation regimes, curve change decomposition, liquidity, expiry and roll calendar, curve-shape comparison.
* **Spread lab** (`GET /api/futures/spread`): hedge ratio (OLS, total least squares, rolling), ADF and Engle-Granger tests, half-life, rolling correlation, seasonality, volatility regimes, walk-forward mean-reversion backtest (no look-ahead; tested). Relationships are reported as hypotheses; a spread that fails the stationarity test is flagged. Simulated curves are refused unless `simulated=1`.
* **Experiments** (`/api/experiments*`): saved runs keep the request, metrics, provenance, software version, `config_hash` and `result_hash`; a comparison states whether a repeat reproduced.

## Added after the first report

* **Portfolio**: vanna and volga (analytic, checked against finite differences), exposure by underlying, expiry and strategy tag, P&L attribution into delta, gamma, vega, vanna, volga, theta and the smile moving with the spot, and every stress grid shown under both volatility rules (sticky-strike, sticky-delta). A position in another underlying needs its own spot and iv; stress moves every underlying alike (a correlation of one, flagged).
* **Strategy studio**: sticky-delta T+0 and half-way lines, early-exercise notes for short legs (European value below intrinsic, dividend assignment risk), and a probability-of-profit audit: the measure (risk-neutral unless you set a view), the distribution assumed, a 3x3 table of volatility and drift, and the caveats.
* **Risk-neutral density**: the density comes from an SVI fit (no numerical differentiation of noisy quotes); a test checks it integrates to one, is a martingale and agrees with the second difference of the fitted-smile prices.
* **Dated chains and stability**: `POST /api/options/snapshot` keeps a real chain per date (simulated chains refused); `GET /api/options/history` follows a model's parameters through the kept dates and says how many more are needed.
* **3D surface**: a canvas viewer (`surface3d.js`, no graphics library; rotate, zoom, keyboard) beside the heatmap.
* **Saved runs**: Save buttons on Surface research, Model validation, the Strategy builder and the Spread lab; a Saved runs view in Options and Futures lists, compares (metrics, differing settings, reproducibility) and deletes them.

## Known limits

* Plotly, echarts-gl and AG Grid are not vendored; the 3D viewer is a small canvas painter (no lighting, no picking). Tables are the existing ones.
* Chain history exists only for the dates a person keeps; a free source gives today's chain only.
* The default data are simulated unless real data are downloaded or supplied; every view says which.
* Backtests omit bid-ask crossing, margin financing, contract-multiplier effects on sizing, and liquidity limits; the realism label names them.
* American exercise is flagged, not priced, in the portfolio and strategy revaluation (Black-Scholes); the binomial pricer is in Model prices.
* A saved run stores its request and metrics; repeating it is done by running the same view again (no one-click repeat).
* Holidays are not modelled in expiry dates (a date can be one business day off).
