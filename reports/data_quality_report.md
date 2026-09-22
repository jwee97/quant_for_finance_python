# Data quality report

- **Provider**: yahoo
- **Data version**: `f13af1fed1f7`
- **Downloaded**: 2026-09-22T12:34:40+00:00
- **Requested window**: 2006-01-01 to 2026-09-22
- **Universe**: 15 ETFs (multi_asset_etf_v1 v1)
- **Trading calendar**: 5211 days, 2006-01-03 to 2026-09-21
- **Config fingerprint**: `12f9cfd14055`

## 1. Provider adjustment convention (Ch. 7 §7.5.1)

Yahoo Finance via yfinance. 'close' is the exchange closing price; 'adj_close' is back-adjusted for splits and cash distributions (dividends reinvested at the ex-date close). Splits affect both series; distributions affect only adj_close. Adjustment is applied retroactively, so the adjusted history for a given date can change after a future distribution -- which is why the download date is part of the data version.

Both price series are preserved in `data/processed`: `prices_close.csv`
(exchange close) and `prices_adjusted.csv` (split- and distribution-
adjusted). Research returns use the adjusted series; the unadjusted
series is retained so that a mechanical price change caused by a
distribution is never read as an investment loss.

### Evidence that the treatment matters

Annualised mean total return minus annualised mean price-only return is
the distribution contribution. If we had naively used the unadjusted
close, we would have discarded the following return per year:

| Ticker | Total return (ann.) | Price-only return (ann.) | Distribution contribution |
|--------|--------------------:|-------------------------:|--------------------------:|
| SPY | +12.43% | +10.61% | +1.82% |
| QQQ | +17.15% | +16.38% | +0.77% |
| IWM | +11.23% | +9.91% | +1.32% |
| EFA | +7.83% | +4.94% | +2.89% |
| EEM | +9.84% | +7.84% | +2.00% |
| SHY | +1.95% | +0.07% | +1.87% |
| IEF | +3.28% | +0.64% | +2.64% |
| TLT | +3.73% | +0.53% | +3.20% |
| AGG | +3.08% | -0.07% | +3.15% |
| LQD | +4.23% | +0.23% | +4.00% |
| HYG | +5.35% | -0.85% | +6.20% |
| GLD | +11.46% | +11.46% | +0.00% |
| SLV | +12.84% | +12.84% | +0.00% |
| DBC | +4.45% | +3.34% | +1.11% |
| VNQ | +10.42% | +6.29% | +4.13% |

The largest distribution contribution is **HYG (+6.20% per year)**; the credit and REIT sleeves would be
materially mis-measured on price returns alone. GLD and SLV show exactly
0.00% because physically backed metal trusts make no distributions -- a
useful sanity check that the adjustment ratio is being read correctly.

## 2. Validation results (spec §9)

Total issues flagged: **1745** (0 errors, 6 warnings, 1739 informational).

| Check | Severity | Count |
|-------|----------|------:|
| price_jump | warning | 3 |
| stale_price | warning | 3 |
| adjustment_break | info | 1736 |
| late_inception | info | 3 |

Checks run: duplicate dates, invalid/future/weekend dates, non-positive
prices, OHLC internal consistency (static-arbitrage identities, §7.5.3),
suspicious jumps, calendar gaps, missing observations against the
universe calendar, stale prices, zero volume, adjustment-factor breaks
and differing inception dates (§7.5.2).

## 3. Anomalies: flagged, investigated, retained

> flag anomaly != delete anomaly

Every return above the 20% threshold was investigated using evidence
available inside the panel: the same-day median return of the asset's
own asset class, the median return of the whole universe, and the
z-score of that day's volume against its trailing 60-day history.

| Ticker | Date | Return | Peer median | Universe median | Volume z | Verdict | Action |
|--------|------|-------:|------------:|----------------:|---------:|---------|--------|
| SLV | 2026-01-30 | -28.54% | -6.86% | -0.56% | 4.8 | market_event_confirmed | retain |
| EEM | 2008-10-13 | +22.77% | +13.34% | +6.36% | 2.4 | market_event_confirmed | retain |
| EEM | 2008-10-28 | +20.93% | +11.37% | +2.26% | 2.5 | market_event_confirmed | retain |

No observation was deleted. These are the tail events the risk engine
(Ch. 21) exists to measure; removing them would flatter every
drawdown, VaR and CVaR number in this project.

## 4. Missing data (spec §11, Ch. 7 §7.6)

The first question is never *which* imputation method to use, it is
**why the observation is missing**. The pipeline classifies every gap
before touching it:

| Ticker | Calendar days | Pre-inception | Interior gaps | Forward filled | Unfilled gaps | Investable days |
|--------|--------------:|--------------:|--------------:|---------------:|--------------:|----------------:|
| AGG | 5211 | 0 | 0 | 0 | 0 | 5211 |
| DBC | 5211 | 23 | 0 | 0 | 0 | 5188 |
| EEM | 5211 | 0 | 0 | 0 | 0 | 5211 |
| EFA | 5211 | 0 | 0 | 0 | 0 | 5211 |
| GLD | 5211 | 0 | 0 | 0 | 0 | 5211 |
| HYG | 5211 | 318 | 0 | 0 | 0 | 4893 |
| IEF | 5211 | 0 | 0 | 0 | 0 | 5211 |
| IWM | 5211 | 0 | 0 | 0 | 0 | 5211 |
| LQD | 5211 | 0 | 0 | 0 | 0 | 5211 |
| QQQ | 5211 | 0 | 0 | 0 | 0 | 5211 |
| SHY | 5211 | 0 | 0 | 0 | 0 | 5211 |
| SLV | 5211 | 80 | 0 | 0 | 0 | 5131 |
| SPY | 5211 | 0 | 0 | 0 | 0 | 5211 |
| TLT | 5211 | 0 | 0 | 0 | 0 | 5211 |
| VNQ | 5211 | 0 | 0 | 0 | 0 | 5211 |

**Policy applied.** Pre-inception NaNs are never filled: HYG did not
exist before 2007-04-11, so the honest statement is that the asset was
not investable, not that its price is unknown. Interior gaps on a
universe trading day are forward filled for at most
3 days and the fill is recorded
in `filled_mask.csv`; `MarketData.returns()` blanks any return that
touches a filled price, so a provider gap can never enter the research
set as a fabricated 0% return.

### Imputation methods compared on artificially masked data

The book's methods (Ch. 7 §7.6) were scored by masking observed
returns at random and measuring the error of each reconstruction, in
basis points. This is the relevant test: returns, not prices, are the
research input. Note that `zero` is the return-space equivalent of
forward filling a *price* -- which is what the production policy does
for a short interior gap -- so it scores the policy we actually use.

| Method | Imputed | MAE (bps) | RMSE (bps) | Bias (bps) | Corr with truth |
|--------|--------:|----------:|-----------:|-----------:|----------------:|
| zero | 400 | 72.5 | 117.7 | -0.5 | undefined (constant) |
| ffill | 400 | 114.7 | 194.9 | -10.2 | -0.222 |
| linear_interpolate | 400 | 100.8 | 165.2 | -1.1 | -0.242 |
| knn | 400 | 42.1 | 66.2 | +2.1 | 0.834 |
| cross_sectional_regression | 400 | 62.5 | 93.7 | +4.0 | 0.612 |

The lowest-error method is **knn**. Two conclusions follow, and they
point in opposite directions from the naive reading of the chapter.

1. **Time-series fills of returns are worse than useless.** `ffill` and
   `linear_interpolate` both achieve a *negative* correlation with the
   truth, because daily returns carry no exploitable persistence to
   extrapolate: copying yesterday's return injects noise with the wrong
   sign. `zero` -- the production policy for a short price gap -- has no
   correlation with the truth by construction, but it is unbiased and it
   never invents a move that did not happen.
2. **Cross-sectional fills do carry information.** `knn` and the
   one-factor `cross_sectional_regression` reach a materially positive
   correlation because contemporaneous asset returns are genuinely
   correlated. That is the defensible way to fill a missing daily
   observation *if* one must be filled.

Even so, the best RMSE (66 bps) is a large fraction of the
cross-sectional daily return standard deviation (124 bps). Filling is
therefore reserved for genuinely missing observations of a series that
did trade; it is never used to manufacture a return, and every filled
value is masked out of the research return set anyway.

## 5. Survivorship and inception bias (Ch. 7 §7.5.2)

The universe is fixed ex ante in `config/universe.yaml` and never
modified in response to results. Three honest limitations remain:

1. **Inception staggering.** Assets enter as they list; the investability
   mask means the cross-section grows from 12 assets at the start of
   the sample to 15 today. All 15 are available from 2007-04-11.
2. **Selection by survival.** These 15 ETFs are liquid and alive *today*.
   An ETF universe chosen in 2006 would have included funds that later
   closed. This is a real upward bias in the results and cannot be
   removed with this dataset; it is restated in the report's limitations.
3. **Retroactive adjustment.** The adjusted history for any date can
   change when a future distribution occurs, so the data version is
   pinned to the download date.

## 6. Corporate actions detected

1736 adjustment-ratio breaks were identified across the universe
and classified by whether the unadjusted price moved with them.

| Ticker | distribution_like |
|--------|---:|
| AGG | 246 |
| DBC | 9 |
| EEM | 39 |
| EFA | 40 |
| HYG | 232 |
| IEF | 227 |
| IWM | 83 |
| LQD | 247 |
| QQQ | 74 |
| SHY | 127 |
| SPY | 83 |
| TLT | 247 |
| VNQ | 82 |

Distribution-like events dominate, as expected for dividend-paying
ETFs. GLD and SLV show none, consistent with non-distributing trusts.

## 7. Verdict

The dataset is fit for research: **0 blocking errors**,
no duplicate dates, no OHLC inconsistencies, no non-positive prices, and
0 forward-filled values in total across
15 series and 5211 trading days. Every warning above
is either a genuine market event or a low-volatility instrument printing
the same price twice, and each is documented rather than removed.

Generated by `experiments/stage01_data.py`.
