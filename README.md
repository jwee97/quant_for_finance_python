# Systematic Multi-Asset Alpha, Portfolio Construction & Risk Platform

An end-to-end quantitative research platform built on *Quantitative Finance with
Case Studies in Python*, covering the full arc from raw market data to a
validated research conclusion.

**The research question**

> Can economically interpretable systematic signals generate persistent
> out-of-sample returns across liquid asset classes, and can robust portfolio
> construction improve their risk-adjusted performance after realistic
> transaction costs?

**The answer this project reached, in one line:** no for the signals, yes for
the portfolio construction — and the negative half of that is the more useful
result.

---

## The design principle

This project is deliberately **not** "build the strategy with the highest
Sharpe ratio". It is:

> Build a defensible quantitative research process capable of determining
> whether an apparent strategy is likely to represent genuine signal or
> research overfitting.

```
research quality  >  backtest attractiveness
```

An out-of-sample Sharpe ratio of 0.7 backed by evidence is a better result
than a suspicious 3.0. Every rejected hypothesis is recorded in
[`experiments/registry.md`](experiments/registry.md) alongside the accepted
ones, because a research process that only reports its successes is not a
research process.

---

## Headline results

Full sample: **15 liquid ETFs, 5,211 trading days, 2006-01-03 to 2026-09-21.**
All figures are net of transaction costs at the baseline assumption.

| Model | CAGR | Vol | Sharpe | Max DD | Turnover |
|-------|-----:|----:|-------:|-------:|---------:|
| SPY buy & hold | 11.1% | 19.2% | 0.65 | −55.2% | 0.0x |
| M0 Equal weight | 7.3% | 10.9% | 0.70 | −33.0% | 0.4x |
| M1 Inverse volatility | 5.5% | 6.4% | **0.87** | −17.6% | 0.5x |
| M2 Risk parity | 5.2% | 6.5% | 0.80 | −18.4% | 0.4x |
| M3 Momentum | 1.8% | 6.7% | 0.30 | −15.2% | 9.2x |
| M4 Mean reversion | −1.9% | 6.8% | −0.24 | −34.4% | 22.4x |
| M5 Momentum + reversion | 0.2% | 6.7% | 0.06 | −20.5% | 18.4x |
| M7 Combined alpha + shrinkage MVO | 3.6% | 7.5% | 0.50 | −21.2% | 10.8x |
| M8 Black-Litterman | 4.3% | 11.2% | 0.43 | −29.3% | 11.9x |
| M9 Mean-CVaR | 3.3% | 4.2% | 0.81 | −14.2% | 0.6x |

The full table, including in-sample/out-of-sample splits and every metric, is
in [`reports/tables/stage12_final_comparison.csv`](reports/tables/).

### Five findings worth stating plainly

1. **The alpha signals do not survive costs.** Momentum and mean reversion
   both have statistically detectable information, but it lives at the 1–5 day
   horizon. Harvesting it requires 9–22x annual turnover, which costs more
   than the signal is worth. Mean reversion breaks even at a transaction cost
   far below what these instruments actually trade at.

2. **Newey-West changes the conclusions.** With overlapping forward returns,
   naive standard errors understate uncertainty by 1.8–3.3x. A 252-day
   momentum signal goes from t = 2.57 ("significant") to t = 0.77 (not) once
   the overlap is corrected. That single number is the argument for Chapter
   20's serial-correlation treatment.

3. **Fifteen ETFs are about five independent bets.** PC1 explains 40.7% of
   variance and only three eigenvalues clear the Marchenko-Pastur noise bound.
   The effective rank falls to 4.1 in March 2020 — diversification thins out
   exactly when it is needed.

4. **The risk model fails its own backtest, and that is the useful part.**
   Every VaR method breaches too often at 99% (historical: 1.53% of days
   against 1% promised) and every method fails the Christoffersen
   independence test at 95%: the breaches cluster. An unconditional VaR is not
   a risk limit.

5. **Four strategy types, one conclusion.** Momentum, mean reversion, machine
   learning and relative value (pairs and PCA statistical arbitrage) were all
   built with the same discipline, and all four land in the same place: a real
   but small edge, a high required turnover, and costs that close the gap.
   Zero of ten candidate pairs were even cointegrated.

6. **What does work is the part that estimates the least.** The models that
   never touch expected returns — inverse volatility, risk parity, mean-CVaR
   — beat every model that does. The estimation-error experiment shows why:
   unconstrained mean-variance holds 3 of 15 assets at a 77% maximum weight,
   and perturbing expected returns by 25% of their cross-sectional dispersion
   moves up to 92% of the book.

---

## Repository layout

```
config/          every parameter that affects a result, in YAML
data/raw/        immutable per-ticker CSVs + provenance manifest
data/processed/  cleaned wide panels (rebuilt, not committed)
src/
  data/          download, validation, cleaning, loading      (Ch. 7)
  features/      returns, volatility, momentum, reversion, PCA (Ch. 8, 20)
  signals/       forecasts, position stack, blending, pairs, PCA stat-arb (Ch. 22)
  portfolio/     EW, inverse vol, risk parity, MVO, BL, CVaR, covariance (Ch. 19, 20)
  backtest/      engine, execution, costs, metrics             (Ch. 22)
  risk/          VaR, CVaR, contributions, stress              (Ch. 21)
  validation/    walk-forward, robustness, leakage detection
  models/        regression with HAC errors, ML ladder         (Ch. 20, 23)
  utils/         config, logging, dates, plotting, experiment registry
experiments/     numbered stage scripts + the experiment registry
reports/         figures, tables, the data-quality report, the research paper
tests/           114 tests
```

---

## Running it

```bash
pip install -r requirements.txt

python -m experiments.run_all --download      # stages 1-13, ~25 minutes
python -m experiments.run_all --from 6 --to 9 # a range of stages
python -m experiments.run_all --only 14       # optional pairs / PCA stat-arb branch
python -m experiments.stage01_data            # a single stage
pytest -q                                      # 114 tests
```

Stage 1 writes `data/raw/*.csv` once and refuses to overwrite them without
`--force`: raw data is immutable, and everything downstream is rebuilt from
it. The dataset's identity is the sha256-derived `data_version` in
`data/metadata/manifest.json`; the research configuration's identity is the
fingerprint printed by every stage.

---

## How the pipeline is structured

```
RAW DATA -> VALIDATION -> CLEAN DATA -> FEATURES
              |                            |
              |                  MOMENTUM / MEAN REVERSION
              |                            |
              |                    SIGNAL RESEARCH (IC, regression)
              |                            |
              |                      SIGNAL ENGINE
              |                            |
              |                 PORTFOLIO CONSTRUCTION
              |                  (EW / inv-vol / RP / MVO / BL / CVaR)
              |                            |
              |                       RISK ENGINE
              |                            |
              |                    BACKTEST + COSTS
              |                            |
              |                  WALK-FORWARD VALIDATION
              |                            |
              +------------->  STRESS / ROBUSTNESS -> REPORT
```

| Stage | Subject | Book basis |
|------:|---------|------------|
| 1 | Data collection, validation, cleaning | Ch. 1 §1.4, Ch. 7 |
| 2 | EDA, PCA, volatility estimators | Ch. 8 §8.2, §8.5; Ch. 20 §20.2 |
| 3 | Momentum alpha research | Ch. 22 §22.3.1, §22.3.9 |
| 4 | Mean-reversion alpha research | Ch. 22 §22.3.1 |
| 5 | Expected returns, IC, signal decay | Ch. 20 §20.1 |
| 6 | Backtesting and transaction costs | Ch. 22 §22.2 |
| 7 | Portfolio construction, estimation error | Ch. 19 §19.2–§19.9 |
| 8 | Covariance and volatility modelling | Ch. 20 §20.2.5–§20.2.11 |
| 9 | VaR, CVaR, stress testing | Ch. 21 §21.2–§21.3 |
| 10 | Combining strategies | Ch. 22 §22.5 |
| 11 | Walk-forward, robustness, leakage | Ch. 22 §22.2.4, §22.2.6–7 |
| 12 | Final comparison | Ch. 22 §22.2.5 |
| 13 | Machine learning extension | Ch. 23 |
| 14 | Pairs trading and PCA stat-arb (optional) | Ch. 22 §22.3.3–§22.3.7 |

---

## What defends this project against fooling itself

| Trap | Defence | Where |
|------|---------|-------|
| Look-ahead bias | Automated test: scramble all future data, require every past weight to be bit-identical. Includes a deliberately broken control strategy, so the test can fail. | `src/validation/leakage.py` |
| Survivorship / universe selection | Universe fixed ex ante in config, never edited after seeing results; residual ETF-survival bias stated as a limitation. | `config/universe.yaml` |
| Overlapping observations | Newey-West HAC standard errors everywhere, lag chosen from the horizon; naive-vs-HAC comparison reported. | `src/models/regression.py` |
| Parameter mining | Whole parameter families evaluated, never one point; plateau-vs-spike diagnostic; deflated Sharpe ratio. | `src/validation/robustness.py` |
| Multiple testing | Benjamini-Hochberg FDR control across all 72 tests per signal family. | `experiments/alpha_research.py` |
| Ignoring costs | Per-asset spreads, cost sweeps, and a breakeven cost for every strategy. | `src/backtest/costs.py` |
| Unrealistic turnover | Weights drift with returns between rebalances rather than being silently reset. | `src/backtest/execution.py` |
| Quiet data deletion | Anomalies are flagged and adjudicated from evidence, never dropped. | `src/data/validation.py` |
| Cherry-picked results | Append-only experiment registry recording rejections. | `experiments/registry.md` |

---

## Reports

- [`reports/research_report.md`](reports/research_report.md) — the full paper
- [`reports/data_quality_report.md`](reports/data_quality_report.md) — Stage 1 findings
- [`reports/figure_index.md`](reports/figure_index.md) — every figure and the question it answers
- [`experiments/registry.md`](experiments/registry.md) — every experiment, including the rejected ones

---

## Limitations

Stated in full in the research report; the ones that most constrain the
conclusions:

- The universe consists of ETFs that exist and are liquid **today**. A
  universe chosen in 2006 would have included funds that later closed. This
  biases results upward and cannot be removed with this dataset.
- Transaction costs are modelled as linear in traded notional. There is no
  market-impact term, so the high-turnover strategies are, if anything,
  flattered.
- Twenty years is one macro cycle and a bit. It contains a single deflationary
  crisis and a single inflationary one, so regime conclusions rest on very
  few independent episodes.
- The final holdout has now been examined. It is no longer untouched, and any
  further work on this data cannot claim a clean out-of-sample test.
- These are backtests. They are not a live track record, and no result here
  includes slippage, financing, taxes or the effect of trading at scale.

## Licence

MIT.
