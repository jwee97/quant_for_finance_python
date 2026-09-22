# Systematic Multi-Asset Alpha, Portfolio Construction and Risk

### A research process, and what it concluded

**Universe:** 15 liquid ETFs spanning equities, rates, credit, commodities and
real estate
**Sample:** 5,211 trading days, 2006-01-03 to 2026-09-21
**Data version:** `f13af1fed1f7` · **Config fingerprint:** `12f9cfd14055`
**Reproduce with:** `python -m experiments.run_all --fresh`

---

## 1. Executive summary

### 1.1 The question

> Can economically interpretable systematic signals generate persistent
> out-of-sample returns across liquid asset classes, and can robust portfolio
> construction improve their risk-adjusted performance after realistic
> transaction costs?

### 1.2 The answer

**No to the first half. Yes, modestly, to the second.** And the process that
established the "no" is worth more than the "yes".

Momentum and mean reversion both carry genuine, statistically detectable
information on this universe. That information lives almost entirely at the
one-to-five-day horizon. Harvesting it requires turning the book over 9 to 22
times a year, and at that rate transaction costs consume more than the signal
is worth. Mean reversion does not survive even at zero cost. Momentum survives
at zero cost and dies somewhere between 10 and 28 basis points, which is
inside the plausible range for these instruments.

What does work is the part of the process that estimates the least. Inverse
volatility (Sharpe 0.87), risk parity (0.80) and mean-CVaR (0.81) all beat
both equal weighting (0.70) and SPY buy-and-hold (0.65), and they do it with
one-third to one-half of SPY's drawdown. None of them uses an expected-return
forecast. Every model that does — constrained mean-variance, Black-Litterman,
the four machine-learning models — lands below them.

### 1.3 The five findings

1. **The alpha signals do not survive costs.** Gross Sharpe 0.38 becomes net
   0.30 for momentum; −0.04 becomes −0.24 for mean reversion. Mean reversion's
   breakeven transaction cost is **−1.1 bps**: it loses money before a single
   basis point of cost is charged.

2. **Honest inference changes the conclusions.** With overlapping forward
   returns, naive standard errors understate uncertainty by 1.8× to 3.3×.
   A 252-day momentum signal goes from t = 2.57 ("significant") to t = 0.77
   (not) once Newey-West corrects the overlap.

3. **Fifteen ETFs are about five independent bets.** PC1 explains 40.7% of
   variance; only three eigenvalues clear the Marchenko-Pastur noise bound.
   The effective rank of the covariance spectrum falls from 5.3 on average to
   **4.11 on 2020-03-16** — diversification thins out exactly when it is
   needed.

4. **The VaR model fails its own backtest.** Every method breaches too often
   at 99% (historical: 1.48% of days against 1% promised), and every method
   fails the Christoffersen independence test at 95%: the breach *count* is
   roughly right, but the breaches arrive in clusters. CVaR, by contrast, is
   well calibrated (realised/predicted = 1.03).

5. **Nothing here survives multiple-testing deflation.** The best out-of-sample
   Sharpe ratio is 0.84, and after accounting for the 42 parameter variants
   tried, its deflated Sharpe probability is 0.86 — below the 0.95 threshold.
   The honest statement is that this research cannot distinguish its best
   model from a good draw.

### 1.4 The headline table

Full sample, net of costs at the baseline assumption:

| Model | CAGR | Vol | Sharpe | Sortino | Max DD | CVaR 95% | Turnover | Cost drag |
|-------|-----:|----:|-------:|--------:|-------:|---------:|---------:|----------:|
| SPY buy & hold | 11.1% | 19.2% | 0.65 | 0.61 | −55.2% | 2.87% | 0.0x | 0 bp |
| M0 Equal weight | 7.3% | 10.9% | 0.70 | 0.66 | −33.0% | 1.65% | 0.4x | 2 bp |
| **M1 Inverse volatility** | 5.5% | 6.4% | **0.87** | 0.82 | −17.6% | 0.96% | 0.5x | 3 bp |
| M2 Risk parity | 5.2% | 6.5% | 0.80 | 0.75 | −18.4% | 0.96% | 0.4x | 3 bp |
| M3 Momentum | 1.8% | 6.7% | 0.30 | 0.29 | −15.2% | 1.02% | 9.2x | 57 bp |
| M4 Mean reversion | −1.9% | 6.8% | −0.24 | −0.24 | −34.4% | 1.06% | 22.4x | 139 bp |
| M5 Momentum + reversion | 0.2% | 6.7% | 0.06 | 0.06 | −20.5% | 1.03% | 18.4x | 115 bp |
| M6 Combined alpha + MVO | 3.6% | 7.5% | 0.51 | 0.47 | −21.2% | 1.15% | 10.9x | 68 bp |
| M7 + shrinkage covariance | 3.6% | 7.5% | 0.50 | 0.47 | −21.2% | 1.15% | 10.8x | 67 bp |
| M8 Black-Litterman | 4.3% | 11.2% | 0.43 | 0.39 | −29.3% | 1.71% | 11.9x | 74 bp |
| M9 Mean-CVaR | 3.3% | 4.2% | 0.81 | 0.76 | −14.2% | 0.67% | 0.6x | 4 bp |
| ML0–ML3 (best) | 0.1% | 3.4% | 0.06 | — | −14.8% | — | 5.4x | 54 bp |

The ordering is the finding. Complexity is monotonically *unrewarded*: every
step up the ladder that requires estimating an expected return performs worse
than the steps that do not.

---

## 2. Motivation

Multi-asset systematic investing is the right laboratory for this question for
three reasons.

**The risk premia are genuinely different.** Equity, duration, credit and
commodity risk are compensated for different economic reasons. A momentum
signal within US equities is largely a bet on one factor; across asset classes
it has to work against genuinely heterogeneous dynamics.

**The diversification is real but limited, and measurable.** Section 4 shows
that fifteen instruments deliver roughly five independent bets. That is a much
more useful statement than "we hold fifteen ETFs", and it is only available
because the analysis was done.

**The instruments are clean.** Liquid ETFs trade continuously, have no
survivorship complications within the sample window, and have transaction
costs that can be estimated within a few basis points. This removes confounds
so that a negative result can be attributed to the signal rather than to the
data.

The project's governing principle is worth restating, because it determined
every design decision:

> Build a defensible research process capable of determining whether an
> apparent strategy is genuine signal or research overfitting.

A strategy with an out-of-sample Sharpe of 0.7 supported by evidence is a
better outcome than a suspicious 3.0. Every experiment run — including the
rejections — is recorded in `experiments/registry.md`.

---

## 3. Data

### 3.1 Sources and universe

Daily OHLCV and adjusted close for 15 ETFs from Yahoo Finance via `yfinance`.
The universe is fixed *ex ante* in `config/universe.yaml` and was never
modified in response to a result:

| Asset class | Tickers |
|-------------|---------|
| Equity | SPY, QQQ, IWM, EFA, EEM |
| Rates | SHY, IEF, TLT |
| Fixed income | AGG |
| Credit | LQD, HYG |
| Commodity | GLD, SLV, DBC |
| Real estate | VNQ |

ETFs are a deliberate simplification. Chapter 7 shows that stock, FX, futures,
options and fixed-income datasets each carry distinct cleaning problems;
keeping the instrument type homogeneous means solving one data problem
properly rather than five badly, while still exposing the strategy to
genuinely different economic risk.

### 3.2 Provenance

Every series carries a provenance record: provider, requested window, actual
window, download timestamp, observation count and a sha256 of the file. The
hash of those hashes is the dataset's `data_version`. Raw files are written
once and the downloader refuses to overwrite them without an explicit
`--force`. Everything downstream is rebuilt from raw.

This matters more than it sounds. Yahoo's adjusted prices are computed
*retroactively*: the adjusted history for any past date changes when a future
distribution occurs. Pinning the data version to the download date is what
makes a result reproducible at all.

### 3.3 Corporate actions

Both price series are preserved: the exchange close and the split- and
distribution-adjusted close, plus their ratio. Research returns use the
adjusted series so that distributions are income rather than a price loss.

The treatment is not cosmetic. Annualised total return minus annualised
price-only return, per asset:

| Ticker | Total return | Price-only | Distribution contribution |
|--------|-------------:|-----------:|--------------------------:|
| HYG | +5.35% | **−0.85%** | +6.20% |
| VNQ | +10.42% | +6.29% | +4.13% |
| LQD | +4.23% | +0.23% | +4.00% |
| TLT | +3.73% | +0.53% | +3.20% |
| AGG | +3.08% | −0.07% | +3.15% |
| SPY | +12.43% | +10.61% | +1.82% |
| GLD | +11.46% | +11.46% | **0.00%** |
| SLV | +12.84% | +12.84% | **0.00%** |

On price returns alone, high-yield credit appears to *lose* 0.85% a year while
actually returning 5.35%. GLD and SLV show exactly zero because physically
backed metal trusts make no distributions — a sanity check that the adjustment
ratio is being read correctly rather than merely applied.

1,736 adjustment-ratio breaks were detected across the universe and classified
as distribution-like or split-like by whether the unadjusted price moved with
them.

### 3.4 Validation: flag, never delete

The governing rule:

> flag anomaly ≠ delete anomaly

A 20% one-day move is either a data error or the most informative observation
in the sample. Deleting it automatically destroys exactly the tail behaviour
the risk engine exists to measure.

Eleven checks ran over the panel: duplicate dates, invalid/future/weekend
dates, non-positive prices, OHLC internal consistency (the static-arbitrage
identities of §7.5.3), suspicious jumps, calendar gaps, missing observations
against the universe calendar, stale prices, zero volume, adjustment-factor
breaks and inception staggering.

**Result: 1,745 issues, of which 0 are blocking errors.** Three returns
exceeded 20%. Each was adjudicated automatically from evidence inside the
panel — the same-day median return of the asset's own class, the median of the
whole universe, and the z-score of that day's volume:

| Ticker | Date | Return | Peer median | Volume z | Verdict |
|--------|------|-------:|------------:|---------:|---------|
| SLV | 2026-01-30 | −28.5% | −6.9% | 4.8 | market event confirmed |
| EEM | 2008-10-13 | +22.8% | +13.3% | 2.4 | market event confirmed |
| EEM | 2008-10-28 | +20.9% | +11.4% | 2.5 | market event confirmed |

All three retained. The SLV move followed a 42% run-up in January 2026 and
coincided with GLD −10.3% on 510 million shares; the EEM moves are the October
2008 rally days, and the surrounding days moved ±10–16%. None of these is a
bad tick, and removing them would have flattered every drawdown, VaR and CVaR
number in this report.

Three stale-price runs were flagged, all in SHY — a short-duration Treasury
ETF whose daily volatility is 1.5% annualised, so printing the same price
twice is expected rather than anomalous.

### 3.5 Missing data

The first question is never *which* imputation method to use; it is **why the
observation is missing**. Every gap is classified before anything touches it:

- **Pre-inception** (HYG before 2007-04-11, SLV before 2006-04-28, DBC before
  2006-02-06): never filled. The honest statement is that the asset was not
  investable, not that its price is unknown. The investability mask carries
  this into every downstream stage.
- **Interior gaps** on a universe trading day: forward filled for at most
  three days, with every fill recorded in a mask. `MarketData.returns()`
  blanks any return that touches a filled price, so a provider gap can never
  enter the research set as a fabricated 0%.
- **Longer runs**: left missing and reported.

On this panel the policy never fired: **zero values required filling.**

The book's methods (§7.6) were nevertheless scored, by masking observed
returns at random and measuring reconstruction error in basis points:

| Method | MAE (bps) | RMSE (bps) | Correlation with truth |
|--------|----------:|-----------:|-----------------------:|
| zero (≡ forward-filling a price) | 72.5 | 117.7 | undefined (constant) |
| forward fill | 114.8 | 194.9 | **−0.22** |
| linear interpolation | 100.8 | 165.2 | **−0.24** |
| cross-sectional regression | 62.5 | 93.7 | +0.61 |
| KNN | **42.1** | **66.2** | **+0.83** |

Two conclusions, pointing in opposite directions from the naive reading of the
chapter. **Time-series fills of returns are worse than useless**: `ffill` and
interpolation achieve *negative* correlation with the truth, because daily
returns carry no persistence to extrapolate, so copying yesterday's return
injects noise with the wrong sign. **Cross-sectional fills do carry
information**: KNN reaches +0.83 because contemporaneous asset returns are
genuinely correlated. Even so, the best RMSE of 66 bps is a large fraction of
the cross-sectional daily return standard deviation, which is why filling is
reserved for genuinely missing observations of a series that did trade.

### 3.6 Data limitations

1. **Survivorship.** These 15 ETFs are liquid and alive *today*. A universe
   chosen in 2006 would have included funds that later closed. This biases
   results upward and cannot be removed with this dataset.
2. **Inception staggering.** The investable cross-section grows from 11 assets
   at the start to 15 from 2007-04-11.
3. **Retroactive adjustment.** As noted, adjusted history is not stable
   through time; only the pinned data version is.
4. **One provider.** No cross-source reconciliation was performed.

---

## 4. Exploratory analysis

### 4.1 Distributions

| Statistic | Range across the 15 assets |
|-----------|---------------------------|
| Annualised return | +1.95% (SHY) to +17.15% (QQQ) |
| Annualised volatility | 1.50% (SHY) to 33.34% (SLV) |
| Skew | −1.86 (AGG) to +0.76 (HYG) |
| **Excess kurtosis** | **+2.76 (IEF) to +59.16 (LQD)** |
| Maximum drawdown | −5.7% (SHY) to −76.4% (DBC) |

**Every asset rejects normality at any conventional level** (Jarque-Bera
p < 10⁻¹⁰ for all 15). This is not a stylised fact quoted from a textbook; it
is the evidence that determines the risk methodology in Section 10. A
parametric normal VaR on this universe is knowably wrong before it is
computed, and Section 10 quantifies how wrong.

The volatility range of 22× between SHY and SLV is the other structural fact:
it makes equal *notional* weighting a fiction of diversification, and it is
why every signal is risk-scaled before it becomes a position.

### 4.2 Correlation structure

Average pairwise correlation is 0.256 over the full sample, but that average
conceals the behaviour that matters:

| Regime | Mean pairwise correlation |
|--------|--------------------------:|
| Full sample | 0.256 |
| Global financial crisis | 0.166 |
| Taper tantrum | 0.425 |
| COVID crash | 0.305 |
| **Post-COVID inflation shock** | **0.416** |

The 2008 figure is *lower* than the full sample, because the crisis was
deflationary: equities collapsed while Treasuries rallied, and the negative
equity-bond correlation held. 2013 and 2022 are the opposite — rate-driven
episodes in which both legs fell together and the average correlation rose
above 0.4. This single table explains why the risk-based allocators, which
rely on that equity-bond hedge, had their worst regime in 2022 rather than in
2008.

### 4.3 How many independent bets?

Eigen-decomposition of the correlation matrix (correlation rather than
covariance, so PC1 does not simply rediscover that silver is volatile):

| Component | Explained variance | Cumulative | Interpretation |
|-----------|-------------------:|-----------:|----------------|
| PC1 | 40.7% | 40.7% | equity risk vs duration (SPY +0.39, EFA +0.37 / TLT −0.16) |
| PC2 | 25.0% | 65.7% | rates level (AGG +0.46, IEF +0.45, TLT +0.41) |
| PC3 | 11.7% | 77.4% | precious metals (GLD +0.61, SLV +0.59) |
| PC4 | 4.8% | 82.2% | credit spread (DBC −0.57, HYG −0.43 / VNQ +0.31) |

Seven components are needed for 90% of variance. But with 15 assets and a
252-day window, the Marchenko-Pastur bound says only **three** eigenvalues are
distinguishable from noise. The effective rank — the exponential of the
entropy of the eigenvalue spectrum — averages **5.32** and falls to **4.11 on
2020-03-16**.

So: fifteen instruments, roughly five independent bets, dropping to four in a
crisis. This has two direct consequences for the rest of the report. It sets a
ceiling on how much diversification any allocation method can extract. And it
means the sample covariance matrix is mostly noise in its lower eigenvalues,
which is why Section 9 shrinks it before any optimiser inverts it.

*Figures 2–7: cumulative returns, return distributions, rolling volatility,
correlation matrices, rolling correlations, PCA.*

---

## 5. Alpha research

### 5.1 Protocol

Both families were evaluated identically, on the **development sample only**
(2006-04-10 to 2015-12-31), with the validation and holdout blocks closed:

- Whole parameter families, never a single lookback. Momentum: 3 variants ×
  4 lookbacks × 6 horizons = 72 tests. Mean reversion: 3 variants × 4
  lookbacks × 6 horizons = 72 tests.
- Spearman IC, because the excess kurtosis in Section 4 means a Pearson IC
  mostly reports which asset had the biggest move.
- t-statistics adjusted for the overlap induced by multi-day horizons.
- **Benjamini-Hochberg FDR control across each 72-test family**, because at a
  naive 5% threshold 3.6 of 72 tests are expected to look significant even if
  every signal is pure noise.

### 5.2 Momentum

The pre-declared primary specification — 126-day volatility-scaled momentum at
a 21-day horizon — is **rejected**: IC +0.0122, p = 0.78. The pooled panel
regression gives β = −0.00044 (t = −1.36), with only 60% of assets showing a
positive beta and 13% significant.

At the family level, 15 of 72 tests clear a naive 5% threshold against 3.6
expected by chance, and **10 survive FDR control**. But every survivor sits at
a 1–10 day horizon with a long lookback:

`ranked_252@h1, raw_252@h1, raw_126@h1, ranked_126@h1, raw_252@h5,
ranked_252@h5, ranked_63@h1, raw_63@h1, raw_252@h10, ranked_252@h10`

So there *is* information in past returns. It predicts tomorrow, not next
month.

Cross-sectional momentum across heterogeneous asset classes is a much weaker
effect than the within-equity version the literature reports, and the reason
is visible in the loadings: the cross-section here is dominated by the
volatility gap between bonds and commodities rather than by relative trend.

### 5.3 Mean reversion

Tested on the sign-neutral z-score, with the hypothesis β < 0 stated before
the test, and at a horizon (5 days) **declared in config before any IC was
computed** — chosen from variance-ratio evidence, which uses only the price
series' own autocovariance and never forward returns.

The structural evidence is strong: **14 of 15 assets have a 5-day variance
ratio below 1**, and the equity sleeve is between 0.75 and 0.90 at 5–21 days.
Prices do mean revert on this universe.

The IC evidence is weaker and decays immediately:

| Horizon | Mean IC of the z-score | t (overlap-adjusted) | p |
|--------:|-----------------------:|---------------------:|---:|
| 1 | −0.0206 | −3.53 | 0.0004 |
| 5 | −0.0176 | −1.34 | 0.18 |
| 10 | −0.0109 | −0.59 | 0.55 |
| 21 | −0.0140 | −0.52 | 0.60 |

Five family tests survive FDR control — `zscore_ranked_5@h1, zscore_5@h1,
zscore_10@h1, zscore_ranked_10@h1, reversal_5@h1` — and **every one is at the
one-day horizon**. The pooled 5-day regression gives β = −0.00025, t = −1.96,
p = 0.050: the right sign, at the edge of significance.

### 5.4 The decisive pattern

Both families tell the same story. The information is real and it is
concentrated where turnover is highest. This is not a coincidence — it is the
market being efficient at exactly the horizons where it is cheap to trade.
Section 7 shows what that costs.

*Figures 8–12: signal behaviour, momentum IC, IC decay, reversion signal,
reversion IC.*

---

## 6. Expected-return modelling

### 6.1 Overlapping observations

A 21-day forward return sampled daily is 21-fold overlapping. The point
estimate is fine; the naive standard error is not.

| Signal | Coefficient | t (naive) | t (Newey-West) | SE inflation |
|--------|------------:|----------:|---------------:|-------------:|
| momentum 126 vol-scaled (h=21) | −0.0017 | −7.09 | −3.34 | 2.12× |
| **momentum 252 raw (h=21)** | +0.0025 | **+2.57** | **+0.77** | **3.31×** |
| reversion 5 z-score (h=1) | +0.0004 | +8.77 | +4.87 | 1.80× |
| reversion 21 z-score (h=1) | +0.0002 | +4.89 | +2.43 | 2.02× |

The second row is the whole argument for Chapter 20's serial-correlation
treatment in one line: a signal that is "significant at the 1% level" under
naive standard errors is **not significant at all** once the overlap is
corrected. Every regression in this project uses HAC standard errors with the
lag chosen from the forecast horizon.

### 6.2 Signal decay and the Fundamental Law

Implied information ratio, IC × √breadth, before any costs:

| Horizon | momentum 252 raw | reversion 5 z-score | 50/50 blend |
|--------:|-----------------:|--------------------:|------------:|
| 1 | 2.24 | 1.72 | **2.55** |
| 5 | 1.40 | 0.71 | 1.39 |
| 10 | 1.10 | 0.31 | 0.87 |
| 21 | 0.81 | 0.17 | 0.61 |
| 63 | 0.48 | 0.00 | 0.28 |

Two things stand out. The decay is steep and monotone: by three weeks, two
thirds of the implied IR is gone. And the blend beats both components at h=1
(IC 0.0415, t = 7.07), because the two signals are negatively correlated in
the cross-section (−0.08 to −0.29).

An implied IR above 2 looks spectacular. It is entirely pre-cost, and it is
available only at a one-day holding period. Section 7 is where that collides
with reality.

### 6.3 Expected returns for the optimiser

Signal-implied expected returns fix the cross-sectional dispersion of μ
explicitly (target spread × (1 − shrinkage) = 2.5% annualised) rather than
inheriting whatever scale the signal happens to have, and shrink towards the
cross-sectional mean before the optimiser sees them. The historical
alternative's dispersion instead wanders between 1.8% and 18.2% — and every
one of those swings feeds straight into the weights. Section 8 quantifies the
damage.

*Figure 13: naive vs HAC inference and the implied information ratio.*

---

## 7. Backtesting methodology and the cost of trading

### 7.1 The timing convention

```
information available through t
        ↓        signal computed from data ≤ t
weights decided at t
        ↓        signal_lag = 1 trading day
book is HELD from t+1
        ↓
return over (t+1, t+2] accrues to that book
```

`R_t = w_{t−1}' r_t`, with an additional configurable lag on top. The
assumption is deliberately conservative: you cannot trade the close you
measured. Weights drift with realised returns between monthly rebalances
rather than being silently reset, because re-imposing targets daily would
assume a daily rebalance and charge none of its turnover.

The engine computes no signals and chooses no parameters. That separation is
what makes the leakage test in Section 12 meaningful.

### 7.2 Costs

`TC_t = c · Σ|w_{i,t} − w_{i,t−1}|`, with per-asset spreads ranging from 3 bps
(SPY) to 12 bps (DBC) so that liquidity differences appear in net returns.

### 7.3 What costs do

| Model | Gross Sharpe | Net Sharpe | Turnover | Cost drag | **Breakeven cost** |
|-------|-------------:|-----------:|---------:|----------:|-------------------:|
| M3 Momentum | 0.38 | 0.30 | 9.2×/yr | 57 bp/yr | **27.8 bps** |
| M4 Mean reversion | −0.04 | −0.24 | 22.4×/yr | 139 bp/yr | **−1.1 bps** |
| M5 Combined | 0.23 | 0.06 | 18.4×/yr | 115 bp/yr | **8.5 bps** |

The breakeven column is the one that settles the question.

**Mean reversion breaks even at −1.1 bps**, which is to say it does not work
at all: its gross return is already negative, and costs merely deepen the
loss. The signal-level IC was real; the strategy built on it is not. The gap
between those two statements is the reason a backtest is run at all.

**The combined strategy breaks even at 8.5 bps**, below the 10 bps baseline
and far below what a real desk would pay after impact.

**Momentum breaks even at 27.8 bps.** That is the only alpha result in this
project with any daylight, and the margin is thin enough that a market-impact
term — which this cost model does not have — could plausibly close it.

Under a 50 bps assumption, mean reversion returns −11.1% a year with a Sharpe
of −1.48. Cost assumptions are not a footnote on strategies that trade this
much; they are the result.

*Figures 14–15: gross vs net, cost sensitivity and breakeven costs.*

---

## 8. Portfolio construction and estimation error

### 8.1 The ladder

Each step adds exactly one idea, so improvement can be attributed:

| | Model | Uses μ? | Uses Σ? | Net Sharpe |
|-|-------|:-------:|:-------:|-----------:|
| M0 | Equal weight | no | no | 0.70 |
| M1 | Inverse volatility | no | diagonal only | **0.87** |
| M2 | Risk parity | no | full | 0.80 |
| M6 | Combined alpha + MVO | yes | sample | 0.51 |
| M7 | Combined alpha + shrinkage MVO | yes | shrunk | 0.50 |
| M8 | Black-Litterman | equilibrium + views | shrunk | 0.43 |
| M9 | Mean-CVaR | yes | scenarios | 0.81 |

The ranking is almost exactly inverse to the amount of estimation required.

### 8.2 Why: the estimation-error experiment

Perturb μ by a fraction of its own cross-sectional dispersion, re-run the full
pipeline on the perturbed input, and measure what moves:

| Method | Holdings | Max weight | Effective N | Mean turnover from noise | Worst case |
|--------|---------:|-----------:|------------:|-------------------------:|-----------:|
| Unconstrained MVO | **3 of 15** | **77.4%** | **1.60** | 24.6% | **91.8%** |
| Constrained MVO | 5 | 25.0% | 4.55 | 37.2% | 98.8% |
| Shrunk μ | 4 | 68.9% | 1.84 | 48.2% | 103.4% |
| Shrunk μ + constrained | 6 | 25.0% | 5.00 | 61.3% | 123.4% |
| **Minimum variance (no μ)** | 5 | 25.0% | 4.42 | **0.0%** | **0.0%** |

Unconstrained mean-variance optimisation on a fifteen-asset universe holds
**three assets**, with 77% in one of them, and a perturbation of μ worth 25%
of its cross-sectional dispersion rewrites up to 92% of the book. This is
Chapter 19 §19.3's error-maximisation property, measured rather than asserted.

One caveat the table makes unavoidable, and which is easy to report
dishonestly: **turnover sensitivity alone is a misleading stability metric.**
The unconstrained solution shows the *lowest* sensitivity — but only because
it is pinned against its bounds in a three-asset corner, not because it is
well estimated. Constraints fix the concentration (effective N rises from 1.6
to 4.6) while *increasing* measured sensitivity, because the solution is now
interior and free to move. The concentration columns and the sensitivity
columns have to be read together, and the walk-forward in Section 12 is what
actually settles it.

Only minimum variance is genuinely immune, because it uses no expected returns
at all. That is the cleanest available statement of where the estimation error
lives: not in the covariance matrix, in μ.

### 8.3 Risk parity in practice

Risk contributions equalise exactly (spread between maximum and minimum share
< 10⁻⁶; Euler's identity Σ RC_i = σ_p holds to machine precision; the
coordinate-descent and SLSQP solvers agree to 10⁻⁶).

The resulting book allocates 33% to SHY and 0.7% to SLV — which is the
intended behaviour, and also the method's main practical weakness: equalising
*risk* contributions on a universe with a 22× volatility range produces a book
dominated by short-duration bonds. Its 5.2% CAGR reflects that.

Diversification ratios: equal weight 1.50, inverse volatility 1.54, risk
parity **1.75**, mean-CVaR 1.82. Risk parity does extract more diversification
per unit of risk — it simply takes less risk overall.

### 8.4 Extensions

**Black-Litterman** (M8) behaves as designed: starting from market-implied
equilibrium returns and tilting on momentum views produces an effective N of
10.7 versus 4.5 for raw MVO — far better diversified. Its net Sharpe of 0.43
still trails the no-μ models, because the views themselves carry no edge.

**Mean-CVaR** (M9) produces a genuinely different book from mean-variance
(L1 distance 1.79) once units are handled correctly, shifting decisively
towards low-tail-risk fixed income. It achieves the **lowest drawdown of any
model (−14.2%)** and the lowest CVaR (0.67% daily), at 4.2% volatility. On
risk-adjusted terms it is the second-best model in the project.

*Figures 16–17: portfolio weights, risk contributions.*

---

## 9. Covariance and volatility modelling

### 9.1 Which estimator forecasts risk best?

Scored out of sample, forecasting 21-day realised portfolio volatility:

| Estimator | Correlation | RMSE | QLIKE | Condition number |
|-----------|------------:|-----:|------:|-----------------:|
| EWMA | **0.567** | **0.053** | **0.550** | 28,824 |
| PCA-denoised | 0.414 | 0.059 | 0.578 | 7,143 |
| Sample | 0.419 | 0.059 | 0.586 | 35,609 |
| Shrinkage | 0.418 | 0.059 | 0.589 | **9,764** |

The answer is genuinely two-sided, and reporting only one side would be
misleading. **EWMA forecasts most accurately. Shrinkage conditions best** —
it cuts the condition number by 3.6× — but it does not forecast better.

These are different goods, and the distinction determines the production
configuration. An optimiser *inverts* the covariance matrix, so it is hurt far
more by ill-conditioning than by slightly worse point accuracy. The platform
therefore uses EWMA for position sizing and shrinkage for anything that gets
inverted. The Ledoit-Wolf intensity the data asks for averages 0.09 — modest,
because 252 observations for 15 assets is not a severely over-parameterised
problem, merely a noisy one.

### 9.2 Single-asset volatility, including GARCH

| Estimator | Correlation | RMSE | QLIKE | R² |
|-----------|------------:|-----:|------:|---:|
| GARCH(1,1) | 0.601 | **0.063** | **0.303** | 0.259 |
| EWMA (half-life 11) | **0.665** | 0.070 | 0.375 | **0.358** |
| EWMA (half-life 40) | 0.602 | 0.074 | 0.390 | 0.302 |
| Rolling 63-day | 0.593 | 0.077 | 0.428 | 0.245 |
| Parkinson 21-day | 0.644 | 0.078 | 0.963 | 0.151 |

> **Does GARCH materially improve volatility forecasts?** (spec §32)

Partly. GARCH wins on QLIKE and RMSE; a short-half-life EWMA wins on
correlation and R², at a fraction of the complexity and with no optimiser to
fail. Mean persistence α+β is 0.993, implying volatility shocks that decay
over months, and the fit is *integrated* (α+β ≥ 1, no long-run variance) for
SHY and HYG. EWMA remains the production estimator; GARCH is documented as an
extension, not adopted.

Parkinson's estimator is instructive: second-best on correlation, worst by far
on QLIKE, because it systematically under-forecasts (mean ratio 0.90) by
ignoring overnight gaps — which for ETFs is where a large share of the
variance lives.

*Figure 18: covariance evaluation.*

---

## 10. Risk modelling

### 10.1 VaR and CVaR by method

Equal-weight portfolio, daily, as positive losses:

| Method | VaR 95% | VaR 99% | CVaR 95% | CVaR 99% |
|--------|--------:|--------:|---------:|---------:|
| Historical | 1.02% | 1.96% | 1.66% | 2.91% |
| Parametric normal | 1.12% | 1.60% | 1.41% | 1.83% |
| Parametric t | 0.96% | 1.87% | 1.59% | 2.88% |
| Monte Carlo (t, joint) | 1.06% | 1.79% | 1.55% | 2.45% |

The normal model *overstates* VaR at 95% and *understates* it at 99% —
precisely what fat tails do to a two-parameter fit. The CVaR/VaR ratio of 1.63
at 95% quantifies how much worse the tail is than its threshold.

### 10.2 Out-of-sample validation — the model fails

Rolling 500-day forecasts, tested over 4,941 days:

| Method | α | Breaches | Expected | Rate | Kupiec p | Christoffersen p | Verdict |
|--------|--:|---------:|---------:|-----:|---------:|-----------------:|---------|
| Historical | 95% | 268 | 247 | 5.42% | 0.177 | **0.000** | fail: clustered |
| Parametric normal | 95% | 257 | 247 | 5.20% | 0.519 | **0.000** | fail: clustered |
| Parametric t | 95% | 284 | 247 | 5.75% | **0.018** | 0.000 | fail: understates |
| EWMA normal | 95% | 231 | 247 | 4.68% | 0.290 | **0.001** | fail: clustered |
| Historical | 99% | 73 | 49 | **1.48%** | **0.002** | 0.000 | fail: understates |
| Parametric normal | 99% | 100 | 49 | **2.02%** | **0.000** | 0.000 | fail: understates |
| Parametric t | 99% | 71 | 49 | 1.44% | **0.004** | 0.000 | fail: understates |
| EWMA normal | 99% | 84 | 49 | 1.70% | **0.000** | 0.001 | fail: understates |

**Zero of eight configurations pass.** This is the most useful negative result
in the project, and it has two distinct parts.

At 95%, the breach *count* is defensible for three of four methods (Kupiec
p > 0.05) but **every method fails the independence test at p < 0.001**. The
breaches arrive in clusters. That is exactly what an unconditional model does
to a world with volatility clustering: it is right on average and wrong when it
matters.

At 99% nothing passes even on count. The normal model breaches at twice its
promised rate.

The practical consequence: **a single VaR number is not a risk limit on this
portfolio.** It must be paired with CVaR and with regime analysis.

### 10.3 CVaR, by contrast, is well calibrated

Comparing realised tail loss on breach days against the predicted conditional
mean:

| Book | Realised / predicted |
|------|---------------------:|
| M1 inverse volatility | 1.01 |
| M9 mean-CVaR | 1.02 |
| M5 combined | 1.03 |
| M0 equal weight | 1.03 |
| M2 risk parity | 1.04 |
| SPY buy & hold | 1.07 |

Within 1–7% across every book. The coherent, whole-tail measure works where
the single-quantile measure does not — a concrete argument for CVaR beyond
the theoretical sub-additivity one.

### 10.4 Where the risk actually is

| Book | Effective N | Diversification ratio | Top risk contributor | **PC1 share of variance** |
|------|------------:|----------------------:|---------------------|--------------------------:|
| M0 equal weight | 15.0 | 1.50 | SLV (31.3%) | **91.8%** |
| M1 inverse volatility | 8.2 | 1.54 | SLV (11.5%) | 74.9% |
| M2 risk parity | 8.5 | 1.75 | SLV (11.7%) | 75.2% |
| M9 mean-CVaR | 4.6 | 1.82 | DBC (39.1%) | **40.5%** |

The equal-weight book holds fifteen assets and puts **91.8% of its variance on
a single principal component**. Holding fifteen tickers is not the same as
holding fifteen bets — the point of Section 4.3, now measured at the portfolio
level. Mean-CVaR, which holds the fewest names, has by far the lowest factor
concentration.

### 10.5 Stress testing

Total return through each regime, defined ex ante from published macro
chronology, never from strategy drawdowns:

| Regime | SPY | M0 EW | M1 Inv-vol | M2 RP | M5 Alpha | M9 CVaR |
|--------|----:|------:|-----------:|------:|---------:|--------:|
| GFC (2007-10 → 2009-03) | **−54.8%** | −26.4% | −10.6% | −10.7% | −5.4% | **+0.0%** |
| Euro crisis (2011) | −14.4% | −5.7% | −2.6% | −1.9% | **+7.0%** | +4.5% |
| Taper tantrum (2013) | −0.2% | −3.4% | −3.0% | −3.1% | −0.3% | −3.8% |
| Feb-2018 vol shock | −9.1% | −5.3% | −3.4% | −3.6% | −1.7% | −1.7% |
| Q4-2018 selloff | −18.9% | −7.6% | −3.4% | −4.3% | −4.7% | −1.0% |
| COVID crash (2020) | −33.4% | −18.5% | −10.9% | −11.4% | −11.7% | **−5.4%** |
| COVID recovery | **+69.8%** | +40.0% | +25.1% | +24.5% | +7.4% | +10.0% |
| **Inflation shock (2022)** | −23.8% | −19.9% | **−16.7%** | **−16.1%** | **+2.5%** | −12.9% |

Two observations.

**The risk-based books did their job in the deflationary crises.** Inverse
volatility lost 10.6% in the GFC against SPY's 54.8%. Mean-CVaR was flat.

**They did not in 2022 — and that is the case study the book closes on.** The
post-COVID inflation shock cost inverse volatility 16.7% and risk parity
16.1%, only marginally better than SPY's 23.8% and worse, proportionally, than
anything else in the table. The reason is in Section 4.2: average pairwise
correlation rose to 0.416 as equities and bonds fell together, and a book
built on the equity-bond hedge had nowhere to hide. Risk parity's structural
overweight to duration made it *more* exposed to a rates shock, not less.

Notably, the alpha strategy M5 is the only book positive in 2022 (+2.5%). Its
dismal full-sample record does not erase the fact that a trend-responsive
strategy was the one thing that helped in the regime that broke everything
else — an argument for diversifying across *strategy types*, not just assets.

*Figures 19–20: VaR forecasts and breaches, crisis performance.*

---

## 11. Combining strategies

Blending the five return streams (mean pairwise correlation +0.31):

| Book | Return | Vol | Sharpe | Max DD |
|------|-------:|----:|-------:|-------:|
| M1 inverse volatility (best component) | 5.6% | 6.4% | **0.865** | −17.6% |
| M2 risk parity | 5.3% | 6.5% | 0.805 | −18.4% |
| M9 mean-CVaR | 3.4% | 4.2% | 0.810 | −14.2% |
| M3 momentum | 2.0% | 6.7% | 0.296 | −15.2% |
| M4 mean reversion | −1.6% | 6.8% | −0.239 | −34.4% |
| **Combined (equal)** | 2.9% | 3.7% | 0.778 | **−11.6%** |
| **Combined (inverse vol)** | 2.9% | 3.8% | 0.775 | −12.9% |
| Combined (alpha only) | 0.4% | 4.1% | 0.095 | −11.9% |

The honest reading: combining beats the **average** component (0.78 vs 0.51, a
gain of +0.27) but **not the best** component (0.87). It does produce the
lowest drawdown of any book in the table (−11.6%) at the lowest volatility.

Combination is not alchemy. It cannot turn strategies with no edge into one
that has an edge — the alpha-only blend achieves 0.095, which is what you get
from averaging 0.30 and −0.24. What it does is reduce the variance of the
total, which is worth having when the components are genuinely positive and
worth nothing when they are not.

*Figure 21: strategy-level diversification.*

---

## 12. Out-of-sample validation and robustness

### 12.1 Look-ahead testing

The test: take dataset D, build D′ identical up to time t but with all
post-t data replaced by a random lognormal path (and, separately, time-reversed),
then require every pre-t weight to be **bit-identical**.

24 tests (4 strategies × 3 split dates × 2 perturbation modes):

| Strategy | Tests | Result |
|----------|------:|--------|
| M1 inverse volatility | 6 | **PASS** — 0 differing cells |
| M3 momentum | 6 | **PASS** — 0 differing cells |
| M4 mean reversion | 6 | **PASS** — 0 differing cells |
| CONTROL (built from tomorrow's return) | 6 | **correctly caught, all 6** |

The control is the point. A test that cannot fail proves nothing, so a
deliberately broken strategy is part of the suite permanently. The
implementation also refuses to report a pass when the perturbation failed to
change *any* post-split weight — a strategy that ignores the scrambled data
returns INCONCLUSIVE rather than a vacuous PASS.

A complementary alignment diagnostic flagged something worth recording: the
21-day z-score has a correlation of 0.32 with the **same-day** return, because
a price-level feature computed at the close of t necessarily contains the
return of t. That is legitimate — but it is exactly why the one-day execution
lag is non-negotiable for that signal. Trading the close you measured would
capture a mechanical, untradable correlation.

### 12.2 Parameter sensitivity

| Family | Sets | Best | Median | Worst | % positive | Verdict |
|--------|-----:|-----:|-------:|------:|-----------:|---------|
| Momentum | 24 | 0.302 | 0.076 | −0.410 | 75% | **plateau** |
| Mean reversion | 18 | 0.208 | −0.152 | −0.610 | 39% | **spike — treat as parameter mining** |

Momentum shows the shape you want: three quarters of the parameter
combinations produce a positive net Sharpe and the best is not far above the
median. Mean reversion shows the shape you fear: the median is *negative*, and
the single positive result sits far above it. On this evidence the one
attractive mean-reversion parameterisation should be treated as a lucky draw,
not a discovery — which is consistent with its −1.1 bps breakeven cost.

### 12.3 Walk-forward

Expanding window, 16 folds, 12-month test blocks, 21-day embargo between train
and test to kill horizon overlap. Only test returns are concatenated.

| Model | OOS CAGR | OOS Vol | OOS Sharpe | Max DD | Folds positive | Fold Sharpe σ |
|-------|---------:|--------:|-----------:|-------:|---------------:|--------------:|
| M1 inverse volatility | 4.9% | 6.0% | **0.835** | −17.3% | 13/16 | 1.25 |
| M2 risk parity | 5.0% | 6.1% | 0.834 | −16.9% | 13/16 | 1.24 |
| M9 mean-CVaR | 3.1% | 3.8% | 0.832 | −14.2% | 11/16 | 1.36 |
| M0 equal weight | 6.9% | 9.5% | 0.755 | −20.7% | 12/16 | 1.16 |
| M3 momentum | 2.3% | 6.3% | 0.389 | −15.2% | 10/16 | 0.85 |
| M5 combined alpha | 1.3% | 6.2% | 0.237 | −12.4% | 11/16 | 0.87 |

The per-fold standard deviation of 1.16–1.36 against mean Sharpe ratios of
0.24–0.84 is the number to keep in view: **year-to-year dispersion is larger
than the effect being measured.**

### 12.4 In-sample versus out-of-sample

Comparing the development block (2006–2015) against **disjoint** walk-forward
folds (2016–2026):

| Model | IS Sharpe | OOS Sharpe | Slippage | Retention |
|-------|----------:|-----------:|---------:|----------:|
| M0 equal weight | 0.53 | 0.90 | −0.37 | 1.70 |
| M1 inverse volatility | 0.79 | 0.94 | −0.15 | 1.19 |
| M2 risk parity | 0.66 | 0.93 | −0.27 | 1.41 |
| **M3 momentum** | 0.37 | **0.21** | **+0.15** | **0.58** |
| M5 combined alpha | −0.02 | 0.15 | −0.17 | — |
| M9 mean-CVaR | 0.87 | 0.75 | +0.12 | 0.86 |

Most models did *better* out of sample. That looks like a triumph and is
nothing of the kind: **the two windows cover different market regimes.** The
development block contains the 2008 crisis; the out-of-sample block does not.
A positive slippage here is a statement about regimes, not about overfitting,
and reporting it as evidence of robustness would be exactly the kind of
self-flattery this project is designed to avoid. The per-fold dispersion and
the bootstrap intervals below are the more honest reads.

Momentum is the exception that behaves as theory predicts: it retains 58% of
its in-sample Sharpe.

### 12.5 Statistical uncertainty

Stationary block bootstrap, 2,000 samples, 21-day blocks:

| Model | OOS Sharpe | 90% CI | p vs 0 | Deflated Sharpe prob. |
|-------|-----------:|--------|-------:|----------------------:|
| M1 inverse volatility | 0.835 | [0.41, 1.28] | 0.0005 | 0.864 |
| M2 risk parity | 0.834 | [0.41, 1.26] | 0.0010 | 0.863 |
| M9 mean-CVaR | 0.832 | [0.39, 1.28] | 0.0000 | 0.861 |
| M0 equal weight | 0.755 | [0.35, 1.17] | 0.0020 | 0.783 |
| M3 momentum | 0.389 | [0.02, 0.77] | 0.0375 | 0.252 |
| M5 combined alpha | 0.237 | [−0.12, 0.61] | 0.1405 | 0.102 |

Two conclusions that should temper everything above.

**The intervals are wide.** Twenty years of daily data supports a 90% interval
of roughly ±0.44 on a Sharpe ratio. The top three models — 0.835, 0.834, 0.832
— are statistically indistinguishable from one another, and from equal
weighting. Any claim that inverse volatility "beats" risk parity is not
supported by this sample.

**Nothing survives deflation.** Accounting for the 42 parameter variants
tried, no model reaches the 0.95 threshold; the best reaches 0.86. The
risk-based models are *close*, and their p-values against zero are genuinely
small (0.0005–0.002), so the effect is probably real. But this research cannot
claim, at conventional standards, that its best model is more than a good
draw from the set it searched.

### 12.6 The final holdout, opened once

Examined only after every model choice was frozen. **It is now no longer
untouched, and any further work on this data cannot claim a clean test.**

2022-01-03 to 2026-09-21:

| Model | CAGR | Vol | Sharpe | Max DD | P(Sharpe > 0) |
|-------|-----:|----:|-------:|-------:|--------------:|
| SPY buy & hold | 12.4% | 17.4% | **0.76** | −24.5% | 0.95 |
| M0 equal weight | 7.4% | 10.7% | 0.73 | −19.6% | 0.94 |
| M2 risk parity | 4.9% | 7.3% | 0.69 | −16.0% | 0.93 |
| M1 inverse volatility | 4.3% | 7.0% | 0.63 | −16.4% | 0.92 |
| M7 shrinkage MVO | 3.9% | 8.0% | 0.52 | −18.7% | 0.87 |
| M9 mean-CVaR | 2.4% | 5.0% | 0.49 | −13.1% | 0.86 |
| M8 Black-Litterman | 3.5% | 11.8% | 0.35 | −26.8% | 0.78 |
| **M5 combined alpha** | −1.0% | 6.4% | **−0.12** | −11.9% | 0.39 |
| **M3 momentum** | −1.3% | 6.5% | **−0.17** | −15.2% | 0.36 |
| **M4 mean reversion** | −2.3% | 6.6% | **−0.31** | −13.4% | 0.25 |

The holdout is unambiguous on the question that matters. **All three alpha
models are negative.** The risk-based allocators hold up, retaining 0.63–0.73
against their full-sample 0.80–0.87. And SPY buy-and-hold has the highest
Sharpe of anything in the table — over this particular window, a passive
equity holding beat every systematic construction in the project.

Sharpe by sample block, which puts the holdout in context:

| Model | Development (2006–15) | Validation (2016–21) | Holdout (2022–26) |
|-------|----------------------:|---------------------:|------------------:|
| M0 equal weight | 0.53 | 1.07 | 0.73 |
| M1 inverse volatility | 0.80 | 1.25 | 0.63 |
| M2 risk parity | 0.67 | 1.18 | 0.69 |
| M9 mean-CVaR | 0.88 | 1.10 | 0.49 |
| M3 momentum | 0.37 | 0.57 | −0.17 |
| M4 mean reversion | −0.30 | −0.01 | −0.31 |
| SPY buy & hold | 0.43 | 0.98 | 0.76 |

The validation block flatters everything — Sharpe ratios of 1.0–1.25 across
the board, in a period of falling rates and rising equities. Any research that
had stopped at 2021 would have concluded that all of these models work well.
The holdout is the correction, and mean reversion is the only model that was
negative in all three blocks — consistently, honestly bad.

*Figures 22–23, 25: parameter sensitivity, IS vs OOS, final comparison.*

---

## 13. Machine learning extension

Four models on thirteen ex-ante features, evaluated with purged, embargoed
cross-validation over unique dates, then walk-forward.

| Model | Accuracy | AUC | CV IC | **WF IC** | **WF t** | Net Sharpe | Gross Sharpe | Turnover |
|-------|---------:|----:|------:|----------:|---------:|-----------:|-------------:|---------:|
| ML0 Logistic | 0.534 | 0.545 | 0.067 | 0.080 | 2.59 | −0.008 | 0.119 | 5.0× |
| ML1 Logistic L2 | 0.533 | 0.545 | 0.067 | 0.079 | 2.58 | −0.007 | 0.119 | 5.0× |
| **ML2 Random forest** | 0.533 | **0.553** | **0.077** | **0.096** | **3.42** | **+0.058** | 0.218 | 5.4× |
| ML3 Gradient boosting | 0.529 | 0.539 | 0.058 | 0.054 | 2.08 | **−0.300** | −0.001 | 10.2× |

> **Does nonlinear ML deliver genuine out-of-sample economic improvement?**

**No — and the way it fails is the interesting part.**

Every model achieves a **statistically significant** walk-forward IC
(t = 2.08 to 3.42). The random forest's IC of 0.096 is *higher than anything
the classical signals produced*. Every model has a positive IC in 5 of 5
cross-validation folds. By the standards of Section 5, these are the best
signals in the project.

And the best net Sharpe among them is **0.058**.

This is precisely why spec §48 demands three evaluation axes. Classification
accuracy weights a 0.1% day and a 5% day identically; trading does not.
Gradient boosting demonstrates the point in its sharpest form: a significant
IC of 0.054, and a net Sharpe of **−0.30**, because it trades at 10.2× a year
and its edge cannot pay for that.

ML is not rejected here because it is ML. It is rejected because the
underlying predictability at this horizon is too small to pay for the turnover
required to harvest it, and no functional form changes that. The feature
importances are consistent with the rest of the report: the top four features
are volatility measures (vol_21, ewma_vol, vol_63) and 252-day momentum — the
model is mostly learning to size by risk, which is what the no-μ allocators do
directly and far more cheaply.

*Figure 24: the ML ladder on all three axes.*

---

## 13A. Extensions: pairs trading and PCA statistical arbitrage

Two further strategy types from Chapter 22, run as a **separate research
branch** (`python -m experiments.stage14_extensions`) rather than folded into
the headline ladder. They are different animals with different assumptions,
and mixing them into the core comparison would muddy the question it answers.

### 13A.1 Pairs trading (Extension E, §22.3.3–§22.3.4)

The discipline here is the order of operations: establish that the
relationship *exists* before building a rule on it. Ten economically
motivated candidate pairs were declared in advance, and each was tested for
cointegration by Engle-Granger on the **development sample only**, so the pair
is selected without seeing the period it will be traded in.

| Pair | Return correlation | ADF p-value | Half-life (days) | Cointegrated? |
|------|-------------------:|------------:|-----------------:|:-------------:|
| LQD/HYG | 0.39 | 0.053 | 61 | no (just) |
| SPY/IWM | 0.92 | 0.073 | 65 | no |
| AGG/IEF | 0.65 | 0.090 | 68 | no |
| IEF/TLT | 0.91 | 0.135 | 134 | no |
| SPY/VNQ | 0.78 | 0.229 | 83 | no |
| **GLD/SLV** | **0.81** | **0.426** | **254** | **no** |
| SPY/QQQ | 0.92 | 0.610 | 255 | no |
| EFA/EEM | 0.89 | 0.843 | 614 | no |

**Zero of ten pairs are cointegrated.** The most striking rows are SPY/QQQ and
SPY/IWM: return correlations of 0.92, and spreads that wander off with
half-lives of 255 and 65 days. This is exactly the distinction the screen
exists to enforce — correlation is not cointegration, and trading a spread
that never comes back is a bet on nothing.

GLD/SLV, the closest analogue in this universe to the book's gold-versus-gold-
miners example, has an ADF p-value of 0.43 and a 254-day half-life. It is not
tradable on this evidence.

Trading them anyway, using hedge ratios fitted on development data only,
confirms the screen was right: the best net Sharpe across all ten is **0.17**
(SPY/VNQ), the median is **−0.28**, and the pair with the strongest prior
(GLD/SLV) returns −0.30 with a 66% drawdown. All ten turn over 9–18× a year,
so costs take roughly 0.25–0.85 off each gross Sharpe.

The finding: on a liquid multi-asset ETF universe there are no pairs worth
trading. That is unsurprising — cointegration is far more plausible between
two firms in the same industry than between two broad index funds — but it is
now established rather than assumed.

### 13A.2 PCA statistical arbitrage (Extension F, §22.3.5–§22.3.7)

The progression the specification describes: PCA for *understanding* risk
becomes PCA for *generating* alpha. Each asset's returns are regressed on the
first three principal components estimated from a trailing 252-day window,
refit every 21 days and applied forward. The residual is what the factor
exposures do not explain; a cumulative residual far from zero is the signal.

> **How many components explain the universe?** (§22.3.7) PC1 explains 50.8%
> of variance on the most recent window, the first three 80.2%, and the first
> five 90.1%.

| Horizon | Mean IC | t (overlap-adjusted) | p |
|--------:|--------:|---------------------:|---:|
| 1 | +0.0048 | 1.28 | 0.20 |
| 5 | +0.0084 | 0.99 | 0.32 |
| 21 | +0.0190 | 1.11 | 0.27 |
| 63 | +0.0192 | 0.66 | 0.51 |

No horizon is significant. The backtest is correspondingly poor: gross Sharpe
+0.06, **net Sharpe −0.27**, turnover 13.2× a year, and a breakeven
transaction cost of **1.1 bps**.

One part of it did work exactly as designed. The residual factor exposures of
the final book are PC1 −0.035, PC2 +0.011, PC3 +0.026 — the construction is
genuinely factor neutral, not a disguised beta bet. The machinery is correct;
there is simply no residual alpha in it to harvest.

### 13A.3 What the extensions add to the conclusion

Both are rejected, and both are rejected for the *same reason as everything
else in this project*: a small edge, a high required turnover, and costs that
close the gap. The extensions therefore strengthen rather than complicate the
headline finding. Four independent strategy types — cross-sectional momentum,
mean reversion, machine learning, and now relative value in two forms — were
built with the same discipline and all landed in the same place.

---

## 14. Failure analysis

What did not work, and why — recorded because a research process that reports
only its successes is not a research process. All 30+ experiments, including
every rejection, are in `experiments/registry.md`.

**1. Cross-sectional momentum at a monthly horizon (REJECTED).** The
pre-declared specification produced IC +0.012, p = 0.78. *Why:* the
cross-section of this universe is dominated by the volatility gap between
bonds and commodities rather than by relative trend. The within-equity
momentum of the literature does not transfer to heterogeneous asset classes
without neutralisation.

**2. Mean reversion as a strategy (REJECTED).** The *hypothesis* survived: 14
of 15 assets have a 5-day variance ratio below 1, and five family tests
survive FDR control. The *strategy* did not: −0.24 net Sharpe, and a breakeven
cost of −1.1 bps, meaning it loses money gross. *Why:* every surviving test is
at the one-day horizon, and a one-day holding period on a fifteen-asset book
generates 22× annual turnover. The parameter surface confirms it — the median
parameterisation has a *negative* Sharpe.

**3. Mean-variance optimisation with signal-implied μ (REJECTED as an
improvement).** M6 and M7 achieved 0.51 and 0.50 against inverse volatility's
0.87. *Why:* Section 8.2. The optimiser faithfully maximises a quantity it
cannot estimate. Shrinking the covariance matrix (M7 vs M6) changed the Sharpe
ratio by 0.006 — the covariance was never the binding problem.

**4. Black-Litterman (REJECTED as an improvement).** 0.43 net Sharpe. The
machinery worked exactly as designed — effective N of 10.7 versus 4.5 for raw
MVO — but the views fed into it came from a momentum signal with no
demonstrated edge. Better plumbing does not fix a bad input.

**5. Every VaR configuration (REJECTED).** Zero of eight passed. *Why:*
unconditional models cannot represent volatility clustering, so the breaches
cluster even when the count is right.

**6. The entire ML ladder (REJECTED).** Significant ICs, ~zero net Sharpe, as
above.

**7. Strategy combination as a route past the best component (REJECTED).**
Combining beat the average component but not the best one.

**8. Pairs trading (REJECTED at the screen).** Zero of ten candidate pairs
were cointegrated on the development sample. *Why:* broad index ETFs share a
common factor but nothing pins their spread; SPY/QQQ has a 0.92 return
correlation and a 255-day spread half-life. The screen rejected them before
any trading rule was built, which is the point of running it first.

**9. PCA statistical arbitrage (REJECTED).** Factor-neutral by construction
(residual exposures ≤ 0.035) and correct in every mechanical respect, but the
residual signal is not significant at any horizon and the strategy breaks even
at 1.1 bps.

**What worked:** risk-based allocation that never estimates an expected
return. That is a thin conclusion relative to the machinery built to reach it
— and it is the conclusion the evidence supports.

---

## 15. Limitations

1. **Survivorship.** The universe consists of ETFs liquid *today*. An upward
   bias that cannot be removed with this dataset.

2. **No market impact.** Costs are linear in traded notional. Real impact is
   convex in participation, so the 9–22× turnover strategies are, if anything,
   flattered. Momentum's 27.8 bps breakeven would likely narrow further.

3. **One macro cycle.** Twenty years contains one deflationary crisis and one
   inflationary one. The regime conclusions in Section 10.5 rest on very few
   independent episodes, and the 90% bootstrap intervals of ±0.44 on a Sharpe
   ratio reflect that.

4. **The holdout is spent.** It has now been examined. Any further iteration
   on this data cannot claim a clean out-of-sample test.

5. **Deflation is not passed.** No model clears the 0.95 deflated-Sharpe
   threshold after accounting for the 42 variants tried.

6. **Long-only, unlevered, single-currency.** No shorting, no financing, no FX,
   no taxes, no capacity analysis.

7. **These are backtests.** Not a live track record. No slippage, no borrow
   costs, no operational friction, no effect of trading at scale.

8. **The negative results are the robust ones.** The finding that momentum and
   mean reversion do not survive costs is supported across the development
   sample, the parameter surface, the walk-forward and the final holdout. The
   positive finding — that risk-based allocation helps — is supported, but its
   confidence interval overlaps every competing model in the table.

---

## 16. Conclusion

### What this research establishes

**On the primary question.** Economically interpretable systematic signals do
contain genuine, statistically detectable information about future returns on
this universe. That information is concentrated at the one-to-five-day
horizon, and it is too small to pay for the turnover required to harvest it.
Both alpha families were negative in the final holdout. The answer is no.

**On the secondary question.** Robust portfolio construction does improve
risk-adjusted performance — but only the portion of it that avoids estimating
expected returns. Inverse volatility, risk parity and mean-CVaR beat equal
weighting and SPY on Sharpe ratio, with a third to half the drawdown, and they
held up out of sample. The improvement is real, modest, and statistically
indistinguishable between the three.

### What it does not establish

That any of these models will work in future. The bootstrap intervals span
±0.44 in Sharpe, the deflated Sharpe ratio does not clear its threshold, and
the validation block showed that a favourable regime can make every model in
the project look good.

### The result that generalises

The pattern across all thirteen stages is one thing, stated three ways:

- Signals with the most information had the least tradable information.
- Models requiring the most estimation performed the worst.
- The machine-learning models with the highest information coefficients
  produced the lowest net returns.

Four independent strategy types — cross-sectional momentum, mean reversion,
machine learning, and relative value in two forms — were built with identical
discipline and all landed in the same place.

**Every step that added estimation added error faster than it added edge.**
The project's most useful output is not a strategy. It is the set of
instruments — HAC inference, FDR control, breakeven costs, automated
look-ahead testing with a failing control, parameter surfaces, block bootstraps
and deflated Sharpe ratios — that were able to establish that, honestly,
against a set of ideas that all looked promising at the signal level.

A Sharpe ratio of 0.84 that survives this process is worth more than a 3.0
that has not been asked to.

---

## Appendix A — Reproducing this report

```bash
pip install -r requirements.txt
python -m experiments.run_all --fresh --download   # stages 1-13, ~25 minutes
python -m experiments.run_all --only 14            # optional extensions branch
pytest -q                                           # 130 tests
```

Outputs: `reports/figures/` (25 figures, 27 with the extensions branch), `reports/tables/` (~70 CSVs),
`reports/data_quality_report.md`, `experiments/registry.md`.

Identity of a result = git commit + `data_version` + config fingerprint, all
three printed by every stage.

## Appendix B — Book coverage

| Stage | Chapter and sections |
|------:|----------------------|
| 1 | Ch. 1 §1.4; Ch. 7 §7.2–§7.6 |
| 2 | Ch. 8 §8.2, §8.5; Ch. 20 §20.2 |
| 3–4 | Ch. 22 §22.1, §22.3.1, §22.3.9 |
| 5 | Ch. 20 §20.1.1–§20.1.8 |
| 6 | Ch. 22 §22.2.1–§22.2.5, §22.3.10 |
| 7 | Ch. 11 §11.5; Ch. 19 §19.2–§19.9 |
| 8 | Ch. 20 §20.2.5–§20.2.11 |
| 9 | Ch. 21 §21.2–§21.3 |
| 10 | Ch. 22 §22.5 |
| 11 | Ch. 22 §22.2.4, §22.2.6–§22.2.7 |
| 12 | Ch. 22 §22.2.5 |
| 13 | Ch. 23 §23.2–§23.3 |
| 14 (optional) | Ch. 22 §22.3.3–§22.3.7 |

## Appendix C — Figures

All 25 figures, and the research question each answers, are indexed in
`reports/figure_index.md`.
