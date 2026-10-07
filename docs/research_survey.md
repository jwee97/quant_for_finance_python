# Research survey: what an institutional multi-asset research platform needs

Phase 1 of the Generation 6 upgrade. For each area this page records what the academic literature, practitioner practice and open-source tools say, and the **design decision** it drove in this repository. It is a map of sources, not a literature review: the references are the ones the code was written against, each cited where the corresponding module documents its mathematics.

**How this survey was done, and its limits.** It is a desk survey from the author's knowledge of the literature and of the documentation of the open-source projects named below. No live web crawl or citation-database query was run in the session that wrote it, and the versions of the open-source tools were not pinned. Treat the references as pointers to check, not as a bibliography that has been verified entry by entry. The *implementations* are verified: every module is tested against a reference implementation, a closed form or a simulation with a known truth (see [the capability matrix](capability_matrix.md)).

## 1. Where research platforms go wrong

Three themes recur across the literature and across practitioner post-mortems, and they shaped the priorities (methodology and architecture over model count):

| Failure | Source | Consequence for this repository |
|---|---|---|
| Selection bias from many trials | Bailey and Lopez de Prado (2014), *The deflated Sharpe ratio*; Harvey, Liu and Zhu (2016), *...and the cross-section of expected returns*; White (2000) reality check; Hansen (2005) SPA | Every search (parameter sweep, `quant tune`, the strategy survey) counts its trials; the deflated Sharpe, probability of backtest overfitting, Romano-Wolf and haircut Sharpe are computed from all of them (`src.stats.inference`, `src.ops.hpo`, `src.validation`). |
| Information leakage in time series | Lopez de Prado (2018), *Advances in Financial Machine Learning* (purging, embargo, CPCV); Arlot and Celisse (2010) on cross-validation | Walk-forward protocols with an embargo equal to the label horizon for every learned model; purged k-fold and combinatorial purged CV in `src.ops.cv`; a causality check (change the future, the past must not move) on every model in the registry. |
| Unrealistic costs and fills | Almgren and Chriss (2001); Frazzini, Israel and Moskowitz (2018) on trading costs; Cont, Kukanov and Stoikov (2014) | Spread, commission and square-root impact in the daily engine; options filled at the bid and ask with a one-day lag; execution and market-making simulators for the intraday mechanics. |

## 2. Statistics and inference

**Academic.** Newey and West (1987) and Andrews (1991) on HAC covariances; White (1980); Cameron, Gelbach and Miller (2011) on multiway clustering; Driscoll and Kraay (1998) for panels with cross-sectional dependence; Fama and MacBeth (1973) with the Shanken (1992) errors-in-variables correction; Gibbons, Ross and Shanken (1989); Hausman (1978); Dickey and Fuller (1979), Phillips and Perron (1988), Kwiatkowski et al. (1992) for unit roots; Lo and MacKinlay (1988) variance ratios; Engle and Granger (1987) and Johansen (1991) for cointegration; MacKinlay (1997) and Boehmer, Musumeci and Poulsen (1991) for event studies; Politis and Romano (1992, 1994), Politis and White (2004) with Patton, Politis and White (2009) for dependent-data resampling; Efron (1987) for BCa intervals; Benjamini and Hochberg (1995), Storey (2002) and Romano and Wolf (2005) for multiple testing; Diebold and Mariano (1995) for forecast comparison.

**Practitioner.** Cochrane, *Asset Pricing*, for the Fama-MacBeth and GRS machinery; Harvey (2017) on the t-statistic hurdle; the common practice of reporting Newey-West t-statistics with automatic lags.

**Open source.** `statsmodels` (the reference for OLS, HAC, panel, unit-root, VAR and Johansen), `arch` (Phillips-Perron, variance ratio, GARCH), `linearmodels` (panel and Fama-MacBeth), `scipy.stats`.

**Decision.** Implement each estimator from its formulas in `src.stats`, with every formula visible, and *test against the references above to numerical precision* rather than wrapping them. The own implementations let the same code run on rolling windows, inside walk-forward loops and in the corrections the libraries lack (two-way clustering inside panels, Romano-Wolf, haircut Sharpe, minimum track-record length). **Considered and not done:** wrapping `linearmodels`/`statsmodels` as the implementation, which is shorter but ties numerical behaviour to a dependency.

## 3. Probability and risk

**Academic.** Pickands (1975) and Balkema and de Haan (1974) for the generalised Pareto limit; McNeil and Frey (2000) for conditional EVT; Hill (1975) for the tail index; Embrechts, Kluppelberg and Mikosch (1997); Sklar (1959) and Joe (1997) for copulas; Embrechts, McNeil and Straumann (2002) on correlation pitfalls; Kelly (1956), Thorp (2006) and MacLean, Thorp and Ziemba (2011) for Kelly sizing; Grossman and Zhou (1993) and Cvitanic and Karatzas (1995) for drawdown-constrained growth; Magdon-Ismail et al. (2004) for the expected maximum drawdown; Glasserman, *Monte Carlo Methods in Financial Engineering* for variance reduction and importance sampling; Gelman et al., *Bayesian Data Analysis* for MCMC diagnostics (R-hat, effective sample size); Hamilton (1989) for Markov switching.

**Open source.** `scipy.stats` (genpareto, t), `arch`, `statsmodels` (MarkovRegression), `PyMC`/`Stan` (not used).

**Decision.** Closed forms where they exist (Brownian ruin and drawdown probabilities, GPD quantiles, Kelly with a drawdown constraint) cross-checked against simulation; a small adaptive Metropolis sampler with R-hat and ESS reported rather than a probabilistic-programming dependency; *filtered* (causal) regime probabilities by default. **Considered and not done:** vine copulas and a full Bayesian hierarchical model, which add dimensions of dependence structure that 15-asset daily data cannot identify.

## 4. Econometrics

**Academic.** Box and Jenkins (1970); Harvey (1989) and Durbin and Koopman (2012) for state space and the Kalman filter; Sims (1980) and Litterman (1986) for VARs and the Minnesota prior; Engle and Granger (1987), Johansen (1991) for error correction; Bollerslev (1986), Glosten, Jagannathan and Runkle (1993), Nelson (1991) for GARCH, GJR and EGARCH; Engle (2002) for DCC (already in the repository); Stock and Watson (2002) and Bai and Ng (2002) for dynamic factors; Diebold and Li (2006) for the dynamic Nelson-Siegel yield curve; Patton (2006) on volatility-forecast loss functions (QLIKE).

**Open source.** `statsmodels.tsa` (ARIMA, VAR, VECM, state space, dynamic factor), `arch`.

**Decision.** Exact Gaussian likelihood through a Kalman filter for ARMA (with a steady-state shortcut that equals the full filter to 1e-12), the Minnesota prior for large systems, GJR, TGARCH and EGARCH as the threshold and asymmetric variants, QLIKE for comparing variance forecasts. **Considered and not done:** stochastic-volatility particle filters, FIGARCH and realised GARCH.

## 5. Derivatives

**Academic.** Black and Scholes (1973), Merton (1973, 1976), Black (1976), Bachelier (1900); Cox, Ross and Rubinstein (1979); Longstaff and Schwartz (2001) for American options; Heston (1993) with the Albrecher et al. (2007) form of the characteristic function; Hagan et al. (2002) for SABR; Gatheral (2004) and Gatheral and Jacquier (2014) for SVI, SSVI and the static-arbitrage conditions; Dupire (1994) for local volatility; Breeden and Litzenberger (1978); Carr and Madan (1998), Demeterfi et al. (1999) and CBOE (the VIX white paper) for variance swaps and the VIX; Carr and Wu (2009) and Bollerslev, Tauchen and Zhou (2009) for the variance risk premium; Coval and Shumway (2001), Bakshi and Kapadia (2003) and Bondarenko (2014) on the returns of option strategies; Driessen, Maenhout and Vilkov (2009) for dispersion.

**Practitioner.** Sinclair, *Volatility Trading*, and Bennett, *Trading Volatility*, for the strategy menu; Natenberg for Greeks; the standard criticism that option backtests on mid prices overstate returns.

**Open source.** QuantLib (pricing library), `py_vollib`, `mibian`, and backtesters such as `optopsy` (option strategies).

**Decision.** Implement the pricers, Greeks, an implied-volatility solver that returns NaN when the price does not identify a volatility, SVI/SSVI fits with their arbitrage conditions and an options backtester with explicit fills, hedging, settlement and Greek attribution. Because **no free source gives historical option chains**, build the data layer in full (schema, validation, vendor-CSV loader) and a *synthetic* market whose variance premium and jump risk are stated parameters, so that the engine can be tested against a known truth. **Considered and not done:** an exchange-traded-options data feed (proprietary or paid), exotic derivatives, and a full local-stochastic-volatility model.

## 6. Futures, FX, commodities and rates

**Academic.** Koijen, Moskowitz, Pedersen and Vrugt (2018) on carry; Asness, Moskowitz and Pedersen (2013) on value and momentum everywhere; Moskowitz, Ooi and Pedersen (2012) on time-series momentum; Gorton and Rouwenhorst (2006), Erb and Harvey (2006), Szymanowska et al. (2014) and Boons and Prado (2019) on commodity futures, roll yield and basis momentum; Schwartz and Smith (2000) for the two-factor commodity curve; Fama (1984) for the forward-premium puzzle; Lustig, Roussanov and Verdelhan (2011) and Brunnermeier, Nagel and Pedersen (2009) on carry-trade risk; Nelson and Siegel (1987), Svensson (1994), Diebold and Li (2006) for yield curves; Fabozzi on bond analytics; Ilmanen, *Expected Returns*, for carry and roll-down across asset classes.

**Open source.** QuantLib (curves, bonds), `ccxt` (crypto exchange data), `fredapi`, `nelson-siegel-svensson`.

**Decision.** Treat the *contract* as the unit for futures (continuous series, roll yield and excess returns measured contract by contract), make every other asset class a `MarketBundle` of total-return indices plus a macro panel of carry signals, so that all existing models, allocators and validators apply unchanged. Provide a Schwartz-Smith synthetic generator with a known truth and a Kalman estimator that recovers it. **Considered and not done:** real futures, FX-forward and option data (vendor data), cross-currency-basis modelling, bond-level analytics for callable or inflation-linked securities, and credit curves.

## 7. Microstructure

**Academic.** Kyle (1985); Glosten and Milgrom (1985); Roll (1984); Amihud (2002); Corwin and Schultz (2012); Lee and Ready (1991); Easley, Lopez de Prado and O'Hara (2012) for VPIN; Cont, Stoikov and Talreja (2010) for the Poisson order-book model; Cont, Kukanov and Stoikov (2014) for order-flow imbalance; Avellaneda and Stoikov (2008) and Guéant, Lehalle and Fernandez-Tapia (2013) for market making; Almgren and Chriss (2001) for optimal execution; Bouchaud et al. (2018), *Trades, Quotes and Prices*.

**Open source.** `hftbacktest`, `abides` (agent-based market simulation), `Hummingbot` (market-making bots).

**Decision.** Implement the order-flow estimators, a Poisson limit-order-book simulator, the Avellaneda-Stoikov quotes with a simulator that decomposes the market maker's P&L into spread capture, adverse selection and inventory, and Almgren-Chriss schedules, all tested against closed forms. **Considered and not done:** tick or level-2 data (proprietary), queue-position and latency modelling, and a multi-agent simulator.

## 8. Machine learning

**Academic.** Gu, Kelly and Xiu (2020), *Empirical asset pricing via machine learning*; Friedman (2001) for gradient boosting; Chen and Guestrin (2016), Ke et al. (2017), Prokhorenkova et al. (2018) for XGBoost, LightGBM and CatBoost; Nie et al. (2023, PatchTST); Chen et al. (2023, TSMixer); Oreshkin et al. (2020, N-BEATS); Challu et al. (2023, N-HiTS); Wang et al. (2024, TimeMixer); Lim et al. (2021, Temporal Fusion Transformer); Ansari et al. (2024, Chronos); Das et al. (2024, TimesFM); Chen et al. (2020, SimCLR) for contrastive learning; Gal and Ghahramani (2016) for dropout as Bayesian inference; Kelly, Malamud and Zhou (2024) on the virtue of complexity, and the sobering results in Gu et al. that signal-to-noise in returns is very low.

**Open source.** scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, `neuralforecast`, `pytorch-forecasting`, `chronos-forecasting`, Microsoft Qlib (a quant research platform with a model zoo).

**Decision.** One protocol for every learned model (walk-forward, matured labels only, an embargo equal to the horizon, a yearly refit), so that models differ only in the learner and comparisons are fair; optional boosting libraries behind a clear error; a compact Temporal Fusion Transformer; an unsupervised-pretraining model (autoencoder or contrastive) whose labels enter only in the second stage. The honest finding, reported in [the institutional findings](institutional_findings.md), is that none of the learners earns a tradable Sharpe on 15 ETFs. **Considered and not done:** foundation-model fine-tuning (compute), graph networks on the cross-section beyond the existing stage, and a large hyperparameter search per learner (it would multiply the trial count).

## 9. Platform and research operations

**Academic and practitioner.** Lopez de Prado (2018) and Lopez de Prado and Lewis (2019) on research as a production process; Sculley et al. (2015) on technical debt in machine-learning systems; the MLflow, Weights & Biases and DVC designs for experiment tracking, model registry and data versioning; Bergstra et al. (2011) for TPE; Jamieson and Talwalkar (2016) for successive halving; Li et al. (2017), Hyperband.

**Open source.** MLflow, DVC, Optuna, Ray Tune, Dask, QuantConnect Lean, Zipline, vectorbt, Backtrader, Nautilus Trader, Riskfolio-Lib, skfolio, PyPortfolioOpt, `quantstats`, `empyrical`.

**Decision.** Keep the existing plugin framework and experiment database (the repository's own equivalents of tracking and orchestration) and add what was missing: a content-addressed artifact store whose references prove what was saved, a versioned model registry with stages and lineage, a hyperparameter optimiser that records every trial and reports the selection bias of its own winner, purged and combinatorial cross-validation, and profiling with scaling exponents, all reachable from `quant tune`, `quant benchmark` and `quant registry`. **Considered and not done:** wrapping MLflow or Optuna (the repository's constraint is offline reproducibility with SQLite and files), a cloud deployment of the research database, and Ray execution (the adapter exists but has never been run here).

## 10. What the survey says about the repository's priorities

1. **Methodology before models.** The largest risks in the sources are selection bias, leakage and cost realism; the largest additions here are inference tools, trial accounting, leakage-safe validation and honest simulators.
2. **Know the truth before trusting the machine.** Where free data does not exist (options, futures curves, FX forwards, order books) each capability is built against a synthetic world with a known answer, which tests the *code* and is labelled as saying nothing about markets.
3. **A cheap, honest baseline beats a clever model.** The repository's own results are consistent with this: equal weight and a 60/40 mix are hard to beat net of costs, and the most complex learners did worst.

## 11. Contract-level multi-asset engines

What the literature and practice say a mixed futures, FX, crypto, options and swaps backtest has to get right, and what the engine here does about each:

- **Contract mechanics are the P&L.** For futures the return is the change in the held contract's price, and the roll is a trade between two contracts, not a return (Gorton and Rouwenhorst; Koijen et al. on carry). For perpetuals the funding rate is a large and persistent term (He, Manela, Ross and von Wachter on perpetual futures pricing). The engine therefore books variation margin, roll trades and funding as separate journal categories, so each can be attributed and tested.
- **A ledger beats a returns table.** Cash, margin, collateral and financing in each currency are state, not a return. One journal with a per-entry accounting identity makes "no money from nowhere" a checkable property rather than a hope.
- **Point-in-time discipline.** The look-ahead failures documented in empirical finance (publication lags, revisions, survivorship) are about WHEN a value became known. The event schema carries both times and the store refuses to answer from the future.
- **Costs are structural.** The square-root impact law (Almgren and Chriss; Toth et al.), bid-ask spreads for options and the carry cost of financing decide whether carry and volatility premia survive; they are modelled per instrument class, not as one basis-point charge.
- **Meta-labelling and walk-forward learning** (Lopez de Prado) keep a learner's training labels realised and its decisions out of sample; the ML strategies here train only on labels whose horizon has elapsed.
- **Deterministic replay** is how a research system is audited: the engine hashes its own event log, so a result can be reproduced bit for bit and a change of code or data is visible as a change of digest.

What the survey could not settle without data: how large the equity, volatility and carry premia are in real markets after these costs. The synthetic market built for the engine has a known generating process, so tests can check that machinery recovers what was built in; the sizes of its premia are assumptions, not estimates.

