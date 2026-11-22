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

* **Portfolio**: vanna and volga (analytic, checked against finite differences), exposure by underlying, expiry and strategy tag, P&L attribution into delta, gamma, vega, vanna, volga, theta and the smile moving with the spot, and every stress grid shown under both volatility rules (sticky-strike, sticky-delta). A position in another underlying needs its own spot and iv; each row has a beta (its share of the market move, 1 by default: with every beta 1 the stress is a correlation of one, and says so), and the book can be revalued as American.
* **Strategy studio**: sticky-delta T+0 and half-way lines, early-exercise notes for short legs (European value below intrinsic, dividend assignment risk), and a probability-of-profit audit: the measure (risk-neutral unless you set a view), the distribution assumed, a 3x3 table of volatility and drift, and the caveats.
* **Risk-neutral density**: the density comes from an SVI fit (no numerical differentiation of noisy quotes); a test checks it integrates to one, is a martingale and agrees with the second difference of the fitted-smile prices.
* **Dated chains and stability**: `POST /api/options/snapshot` keeps a real chain per date (simulated chains refused); `GET /api/options/history` follows a model's parameters through the kept dates and says how many more are needed.
* **3D surface**: a canvas viewer (`surface3d.js`, no graphics library; rotate, zoom, keyboard) beside the heatmap.
* **Saved runs**: a Repeat button resends the stored request and judges the new numbers against the saved ones (a changed result with the same request means the data, the seed or the code changed). Save buttons on Surface research, Model validation, the Strategy builder and the Spread lab; a Saved runs view in Options and Futures lists, compares (metrics, differing settings, reproducibility) and deletes them.

## The volatility surface view (Options tab: Volatility surface)

Built in `src/assets/vol_surface.py`, served by `GET /api/options/vol-surface`, drawn by `src/webapp/static/vol_surface.js`.

* **From the quotes, in the data.** Quotes are cleaned and every rejection is counted by reason (no bid or a crossed market, no implied volatility, volatility above 300%, a price too small to carry information, the other
  side of the chain, duplicates, expired). Coordinates: `ln(K/F)` with each expiry's own forward, or the call delta (a put is placed on the call-delta scale at its own volatility); days to expiry on the other axis.
* **One grid, masked where there is no quote.** All expiries share the nodes. A node is filled only between two neighbouring quotes of its own expiry that are no further apart than `max(4 x median spacing, 0.10)` in
  `k` (`0.15` in delta); the value is the linear interpolation of total variance, converted back to volatility. Outside the quoted strikes, inside a gap, and in an expiry with fewer than four quotes the node is empty
  (`null`): never extrapolated, never bridged, never smoothed. This replaces the staircase of the old canvas viewer at the source.
* **Raw and fitted apart.** The fitted surface is an SVI slice per expiry, evaluated only at nodes that are supported by quotes, with the error in volatility points, Gatheral's butterfly condition inside and beyond the
  quoted range, and the calendar condition between neighbouring slices (raw and fitted), counted and located. It exists in the `ln(K/F)` coordinate only.
* **Too little data.** With fewer than two usable expiries the answer is `smiles_only` (or `empty`) with the reason; the page shows the discrete quotes, not a manufactured surface.
* **Four synchronised views** (3D surface, 2D heatmap, volatility smiles, term structure) of one payload; volatility and total variance grids are both returned, so the units switch without a request. Server choices
  (date, side, coordinate, expiry range, comparison date) are in the left panel; display choices (units, raw or fitted, colour range, camera, view) are in the toolbar and survive view changes. The result is cached by
  underlying, date and settings.
* **Drawing.** Plotly.js when vendored (`pip install plotly`, `python scripts/vendor_libs.py`): `Plotly.react` updates in place, `uirevision` keeps the camera and zoom, the camera is kept when filters change and a
  resized or newly shown tab is resized. Without it the app's own ECharts and canvas charts are used and the page says so. The Plotly path is tested with a stand-in object that records what it is given (axes, colour
  scale, hover text, camera, `react` calls); the real library has not been run in this environment.

## Known limits

* Plotly and AG Grid are optional: `pip install plotly dash-ag-grid` then `python scripts/vendor_libs.py` copies `plotly.min.js` and the AG Grid bundle (and its style sheets, if the version has them) into `src/webapp/static/vendor/`. The Options surface then uses Plotly's 3D surface with hover and the Saved runs view uses an AG Grid table (sort, filter, tick rows). Without the files the canvas viewer and the plain tables are used. The hand-over is tested with stand-in files; the real bundles have not been run in this environment. echarts-gl is not used (Plotly covers the surface).
* Chain history exists only for the dates a person keeps; a free source gives today's chain only.
* The default data are simulated unless real data are downloaded or supplied; every view says which.
* Backtests omit bid-ask crossing, margin financing, contract-multiplier effects on sizing, and liquidity limits; the realism label names them.
* American revaluation of a book uses a binomial tree for values; the Greeks stay Black-Scholes, so early-exercise value appears as unexplained in the attribution. The strategy builder prices legs as European and flags the legs at risk.
* Holidays are not modelled in expiry dates (a date can be one business day off).
