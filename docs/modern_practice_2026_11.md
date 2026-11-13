# How close the app is to current quantitative practice (November 2026)

What this is: every tab was read against what systematic firms and the research of roughly 2023 to 2026 actually do, and the gaps that could be closed on one person's computer were closed. This page says what was already in line, what was added, how to use it, and what is still out of reach here and why. It is not a claim that any of it makes money: the newer a method, the shorter its record.

How the comparison was made: the code of each tab's models was searched for about thirty methods that recur in recent papers and in public descriptions of how systematic funds work (training losses, architectures, labels, validation, volatility models, hedging, calibration). A method counted as present only if a tab or the command line reaches it, not if a file merely mentions it.

## What was already in line

These are the things a research team would look for first, and the app had them before this pass.

| Practice | Where |
|---|---|
| Walk-forward training with a purge gap between training and test rows, and combinatorial purged cross-validation | Machine learning, Tune |
| Counting every trial and deflating the Sharpe ratio for it; null (shuffled-label) controls; holdout guards | Machine learning, Tune, Backtest |
| Learning to rank (LambdaMART, XGBoost rankers) and cross-sectional targets (rank, sector-relative, beta-residual) | Machine learning |
| Probabilistic outputs: quantile boosting, NGBoost, a Temporal Fusion Transformer, conformal intervals on tabular models | Machine learning |
| Sequence and graph networks: LSTM, GRU, TCN, PatchTST, N-BEATS, N-HiTS, GCN, GAT | Machine learning |
| Stacked and regime-aware ensembles, meta-labelling, sample weights by label uniqueness, stable feature selection | Machine learning |
| Reinforcement-learning allocators (PPO, SAC) in an environment with costs and turnover limits | Machine learning |
| Shrinkage covariance, hierarchical risk parity, Black-Litterman, CVaR and cost-aware mean-variance allocators | Backtest |
| Regime detection (hidden Markov, Bayesian change point), volatility targeting, trend and carry | Backtest, Futures |
| Costs on every trade, a lag between signal and trade, roll costs, funding, borrow fees | Every backtest (see the [algorithm review](algorithm_review_2026_11.md)) |
| Forecasts scored as whole distributions (CRPS), tested against a random walk with corrections for many methods | Forecast |
| Jump, stochastic-volatility and infinite-activity option models; Leland and Whalley-Wilmott hedging with costs | Options |
| Generative scenarios (WGAN-GP, factor VAE, diffusion) with a quality report | Scenarios |

## What was added

### Machine learning tab

**Losses that train a network on what a long-short book is paid for.** A squared error asks a network to get every asset's return right, and most of that error is the market's common move, which a long-short book does not hold. The sequence networks (`lstm`, `gru`, `tcn`, `patchtst`, `nbeats`, `nhits`, and the two new ones) now have a `loss` setting:

- `mse`: the squared error, as before. Still the default, so nothing you have already run changes.
- `ic`: one minus the correlation of score and return across the assets of each day, averaged over days (the information coefficient), plus a small squared-error term that keeps the score in the target's units.
- `mse_rank`: the squared error plus `rank_weight` times a penalty on every pair of assets of a day that the scores order the wrong way round (Feng et al. 2019; the ranking term of StockMixer, Fan and Shen 2024).

The last two need each batch to hold whole days, so the trainer now builds batches from whole days when either is chosen, and thins a large dataset by dropping whole days, never some assets of a day. Tests: `tests/test_ml_deep_modern.py`.

Caution that matters for you: the platform's own universe is a few dozen ETFs. A daily correlation across ten assets is a noisy number, so these losses will help most on a wide universe (your own file of stocks). On a narrow one, compare against `mse` before believing either.

**iTransformer** (`itransformer`, Liu et al., ICLR 2024). The transformer turned on its side: each feature's whole window is one token and attention runs among the features. It is light on a CPU because the number of tokens is the number of features, whatever the lookback. The manifest labels it as reduced and says what differs from the paper (the paper forecasts each series' own future; here the tokens are mixed into the panel target).

**DLinear** (`dlinear`, Zeng et al., AAAI 2023). A moving-average trend and the remainder, one linear layer on each, no hidden layer. It is the baseline every time-series transformer since 2023 is measured against, and on noisy returns it is often the one to beat. Run it beside any network you train: if the network does not beat it, the network's depth is not earning anything.

**Random-feature ridge** (`rff_ridge`): the model of "The Virtue of Complexity in Return Prediction" (Kelly, Malamud and Zhou, Journal of Finance 2024). A ridge regression on a thousand or more random sine and cosine features of the inputs, which is a kernel regression at linear cost. The paper's claim is that more features than observations helps provided the penalty stays. The claim is disputed (Nagel 2025; Buncic 2025), so the catalogue entry tells you to compare it with the plain ridge on the same data. A test checks that it converges to the exact Gaussian-kernel ridge as the features grow. Tests: `tests/test_ml_modern_tabular.py`.

**Triple-barrier target** (`barrier_5`, `barrier_10`, ...; Lopez de Prado 2018). Two price barriers are set one trailing standard deviation of an *h*-day move above and below the entry price. The label is 1 if the upper one is touched first, 0 if the lower one is, and the sign of the *h*-day return if neither is. It asks what a trade with a profit target and a stop asks; `direction_h` only looks at the last day. A row is labelled only once its whole window exists, so which rows have labels never depends on what happened.

**Fractional differencing features** (group `fracdiff`: orders 0.3, 0.5 and 0.7 over 126 days). A return forgets the price level entirely; a price is not stationary. An order in between keeps part of the level's memory. The feature is measured from the window's first price, so it has no unit and a split or a currency does not change it. The group is not in the default dataset, so existing experiments are unchanged; tick it in the dataset form to use it.

### Forecast tab

**Path-dependent volatility** (`pdv`; Guyon and Lekeufack, Quantitative Finance 2023). Their finding is that most of the level of volatility can be read from the past returns alone: a weighted average of the returns (after falls, volatility is higher) and of the squared returns (after busy days, it is higher), each on a fast and a slow clock. A forecast is a simulation in which every path updates its own volatility as it goes, so a path that falls becomes more volatile. That is where the fat left tail of a month's return comes from, and a GARCH(1,1) cannot produce it. The method reports the fitted loadings and whether falls raised volatility in your data. Tests: `tests/test_forecast_pdv.py`.

**Stretch to hold 90%** (split conformal calibration; Romano, Patterson and Candès 2019). For every method the evaluation now reports the factor its 90% range needed to be widened by, about its middle, to have held 90% of the past outcomes. 1.00 is the right width; 1.25 means the honest range is a quarter wider than the fan shows. It is in the scores table and the calibration tiles. The forecasts overlap in time, so the guarantee of the theory (independent cases) does not strictly hold: treat it as a guide.

### Options tab

**Deep hedging** in *Hedging cost* (Buehler, Gonon, Teichmann and Wood 2019, with the no-transaction band network of Imaki and others 2021). The three classical rules each rest on an approximation. Deep hedging makes none: it states the problem as it is (sell the option, trade on the given dates, pay the cost on every trade, be judged on the result at expiry) and finds the rule by gradient descent on simulated paths. The network places a no-trade band around the delta. It is then run on the same paths as the other rules, which it never saw in training, and all rules are scored on one number, the **certainty equivalent** (what an investor with your risk aversion would value each rule's uncertain result at).

In the cases checked while building it (a three-month at-the-money option, costs of 0 to 20 basis points, 21 to 63 hedging dates), the network matched or beat the Whalley-Wilmott band and beat the delta hedge wherever trading cost anything; with free trading it closed its band and became the delta hedge. The largest gain was with few hedging dates and high risk aversion, where the band's small-cost formula is furthest from its assumptions. Two charts show the band it learned beside the theory's, and the training curve. It needs PyTorch; without it the report shows the three classical rules and says so. Tests: `tests/test_deep_hedging.py`.

What it does not do: it learns for one option, one cost and one volatility, on paths from a constant-volatility model. On real prices it is only as good as that model.

## How to try the new pieces

1. **A network trained to rank.** Machine learning → build a dataset on a wide universe → Train → model `dlinear` (fast) or `itransformer` → set `loss` to `ic` → run. Then run the same model with `loss` `mse` and compare the two experiments' rank IC and net Sharpe.
2. **The complexity debate on your data.** Train `ridge` and `rff_ridge` on the same dataset and compare. Raise `components` and see whether the out-of-sample result keeps improving, as the paper says, or not.
3. **Profit target and stop.** In the dataset form set the target to `barrier_10` and train `logistic` or `lightgbm`.
4. **A volatility forecast that reacts to falls.** Forecast → add *Path-dependent volatility* to the methods → run the evaluation, then read its skill, its *Stretch to hold 90%* and whether "falls raise volatility" came out true.
5. **What a learned hedge is worth.** Options → Hedging cost → raise the cost to 20 basis points → *Cost the hedge*. Read the certainty-equivalent column and the two new charts. Training takes about half a minute.

## What is still different from a firm, and why

| A firm has | Here |
|---|---|
| Thousands of stocks with point-in-time fundamentals, analyst estimates, short interest and alternative data | ETFs, futures and crypto from free sources, plus whatever files you bring. Cross-sectional methods (rank losses, neutralisation) are only as good as the width of the universe |
| Intraday and order-book data; execution algorithms measured against arrival price | Daily closes. Costs are modelled, fills are not. Same-day-expiry (0DTE) option analytics need intraday option data and are not built |
| Language models reading filings, calls and news | No text data source is connected, so there is no text signal. This is the largest single gap to 2025-2026 practice |
| Pretrained time-series foundation models (Chronos, TimesFM, Moirai) | Not in the tabs: the weights are a large download from a model hub and the published evidence on daily returns is mixed. The platform's own networks train from scratch on your data |
| GPUs and models with millions of parameters trained on decades of a wide cross-section | CPU-sized networks, labelled as reduced where they are. `device` uses a GPU if PyTorch sees one |
| Risk models with industry and style factors, and a live book reconciled to a broker | A statistical risk model and a paper portfolio. No broker connection, by design |
| Deep hedging under a calibrated stochastic-volatility or market simulator, for whole books | One option under constant volatility: a demonstration of the method, honest about its limits |

## Sources

- Buehler, Gonon, Teichmann and Wood (2019), "Deep hedging", *Quantitative Finance* 19(8).
- Buncic (2025), "Simplified: a closer look at the virtue of complexity in return prediction", working paper.
- Fan and Shen (2024), "StockMixer: a simple yet strong MLP-based architecture for stock price forecasting", AAAI.
- Feng, He, Wang, Luo, Liu and Chua (2019), "Temporal relational ranking for stock prediction", *ACM Transactions on Information Systems* 37(2).
- Guyon and Lekeufack (2023), "Volatility is (mostly) path-dependent", *Quantitative Finance* 23(9).
- Imaki, Imajo, Ito, Minami and Nakagawa (2021), "No-transaction band network: a neural network architecture for efficient deep hedging", working paper.
- Kelly, Malamud and Zhou (2024), "The virtue of complexity in return prediction", *Journal of Finance* 79(1).
- Liu, Hu, Zhang, Wu, Wang, Ma and Long (2024), "iTransformer: inverted transformers are effective for time series forecasting", ICLR.
- Lopez de Prado (2018), *Advances in Financial Machine Learning*, Wiley (triple-barrier labels, fractional differencing).
- Nagel (2025), "Seemingly virtuous complexity in return prediction", working paper.
- Romano, Patterson and Candès (2019), "Conformalized quantile regression", NeurIPS.
- Zeng, Chen, Zhang and Xu (2023), "Are transformers effective for time series forecasting?", AAAI.
