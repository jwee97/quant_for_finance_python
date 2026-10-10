# Glossary

Plain-language definitions of every term the guides and reports use. Each entry says where in the guides to read more. `quant explain <term>` prints an entry from the terminal.

### Abnormal return

The actual return minus the return a model expected for the same day, such as a market-model prediction. Cumulated over a window it gives the cumulative abnormal return.

See: [event-studies](techniques/event-studies.md)

### ADV

*Also: average daily volume*

Average daily volume: the typical amount of an asset traded per day, which sets how much you can trade before moving the price.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Adverse selection

The loss a liquidity provider suffers when the traders it fills are better informed, so the price moves against it after the trade.

See: [market-making-and-order-flow](techniques/market-making-and-order-flow.md)

### AIM and PIM

*Also: aggressive in the money, passive in the money*

Adaptation tactics that change the pace of a running order with the price. Aggressive in the money speeds up when the price has moved in your favour (for markets that overshoot and come back); passive in the money slows down then, and speeds up when the price moves against you (for markets that trend and to limit a loss).

See: [execution-algorithms](techniques/execution-algorithms.md)

### Allocator

In this repo, a component that turns forecasts and regimes into portfolio weights (static books, forecast stack, confidence, regime switch).

See: [plugin-framework](techniques/plugin-framework.md), [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)

### Almgren-Chriss

The optimal-execution model that trades off temporary impact against price risk, giving a closed-form liquidation schedule between immediate and equal-sized trading.

See: [market-making-and-order-flow](techniques/market-making-and-order-flow.md), [execution-and-impact](techniques/execution-and-impact.md)

### Alpha

The part of a return that is not explained by exposure to known risks or factors; in practice, the return a strategy earns beyond what its benchmark or factor exposures predict.

See: [attribution](techniques/attribution.md)

### Annualised volatility

*Also: ann vol*

Daily return standard deviation multiplied by the square root of 252 trading days.

See: [performance-metrics](techniques/performance-metrics.md)

### Antithetic sampling

Drawing random perturbations in plus/minus pairs so their sampling noise cancels; used in evolution strategies.

See: [reinforcement-learning](techniques/reinforcement-learning.md)

### Appraisal ratio

Alpha divided by the volatility of the part of the return the market does not explain. It ranks assets by return per unit of idiosyncratic risk, so a lucky streak in a noisy asset ranks below steady alpha.

See: [alpha-generating-styles](techniques/alpha-generating-styles.md)

### ARIMA

*Also: ARMA*

A time-series model in which the next value depends on past values (autoregressive terms) and past shocks (moving-average terms), after differencing d times to remove a unit root.

See: [time-series-econometrics](techniques/time-series-econometrics.md)

### Arrival price

The market price when an order was received. Implementation shortfall measures the cost of a trade against it.

See: [execution-algorithms](techniques/execution-algorithms.md)

### Attention

A neural-network operation that lets each element of a sequence or graph take a learned weighted average of the others.

See: [deep-time-series-models](techniques/deep-time-series-models.md), [graph-neural-networks](techniques/graph-neural-networks.md)

### AUC

Area under the ROC curve: the probability that a randomly chosen positive case receives a higher score than a randomly chosen negative case; 0.5 is chance.

See: [machine-learning-for-returns](techniques/machine-learning-for-returns.md)

### Availability time

*Also: available_at*

The time a value could first have been known to a trader, as opposed to the time it describes. The engine delivers every market event at its availability time, never earlier, which is what keeps a backtest free of look-ahead.

See: [point-in-time-market-data](techniques/point-in-time-market-data.md)

### Backcast

In N-BEATS, the part of the input window a block explains; it is subtracted before the next block sees the window.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Backtest

A simulation of how a strategy would have performed on past data, under stated rules for timing and costs.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### Base rate

How often an event happens regardless of any signal, for example equities rise in about 60% of months; a forecast must beat it to have skill.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Basis

The difference between a futures or perpetual price and the spot price.

See: [crypto-carry](techniques/crypto-carry.md)

### Bayes-Stein shrinkage

Pulling noisy estimated mean returns toward a common value, with a weight that depends on how noisy they are.

See: [bayesian-portfolio-construction](techniques/bayesian-portfolio-construction.md)

### Bellman equation

The condition that makes dynamic programming work: the value of being in a state is the best action's reward plus the discounted expected value of the state it leads to. Value iteration applies the right side repeatedly until it stops changing; backward induction applies it once per period from the end.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Benjamini-Hochberg

*Also: BH, false discovery rate, FDR*

A procedure that controls the expected share of false discoveries among the hypotheses you declare significant when testing many at once.

See: [multiple-testing](techniques/multiple-testing.md)

### Best execution

Executing an order on the best terms reasonably available. As an algorithm goal it means one of: lowest cost; lowest risk for a cost limit; lowest cost for a risk limit; the best trade-off for a risk aversion; or the best chance of beating a target cost.

See: [execution-algorithms](techniques/execution-algorithms.md)

### Beta

The sensitivity of an asset's return to a market return: a beta of 0.5 means it tends to move half as much.

See: [attribution](techniques/attribution.md), [cross-sectional-factors](techniques/cross-sectional-factors.md)

### Beta-neutral

*Also: market neutral*

A book whose net exposure to the market is zero, so its profit and loss comes from selection and not from the market's move. It is reached by projecting the market out of the weights or by selling a hedge instrument.

See: [portfolio-overlays-and-liquidation](techniques/portfolio-overlays-and-liquidation.md)

### Black-Litterman

A way to combine market-implied expected returns with an investor's views, weighted by their confidence, to get steadier optimiser inputs.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Black-Scholes

*Also: BSM*

The closed-form price of a European option when the underlying follows geometric Brownian motion with constant volatility. Its inverse maps an option price to an implied volatility.

See: [option-pricing-and-greeks](techniques/option-pricing-and-greeks.md)

### Block bootstrap

*Also: stationary bootstrap*

Resampling blocks of consecutive observations rather than single days, so autocorrelation and volatility clustering are preserved.

See: [paired-sharpe-bootstrap](techniques/paired-sharpe-bootstrap.md)

### Block trade and program trade

*Also: basket trade*

A block is a large order in one stock, too big for the lit market in one go; a program (basket) trade is a list of stocks bought and sold together. The risk of the unexecuted part of a list is the reason to schedule it jointly.

See: [basket-and-liquidity-algorithms](techniques/basket-and-liquidity-algorithms.md)

### BOCPD

*Also: change point*

Bayesian online change-point detection: tracks the probability that the time since the last distribution change is any given length.

See: [regime-detection](techniques/regime-detection.md)

### Breadth

The number of independent bets per year; in the fundamental law, information ratio grows with the square root of it.

See: [information-coefficient](techniques/information-coefficient.md)

### Brier score

The mean squared error between predicted probabilities and 0/1 outcomes; lower is better.

See: [calibration](techniques/calibration.md)

### Brinson-Fachler

An attribution method that splits active return versus a benchmark into allocation, selection and interaction effects.

See: [attribution](techniques/attribution.md)

### Calibration

The match between stated probabilities and observed frequencies: events forecast at 70% should happen about 70% of the time.

See: [calibration](techniques/calibration.md)

### Calmar ratio

Compound annual growth divided by the size of the worst drawdown.

See: [performance-metrics](techniques/performance-metrics.md)

### Capacity

The amount of money a strategy can run before its trading costs erase its edge; here, the assets at which net Sharpe halves.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Carry

The return you earn from holding an asset if prices do not change, such as a bond's yield above cash or a perpetual's funding rate.

See: [cross-sectional-factors](techniques/cross-sectional-factors.md), [crypto-carry](techniques/crypto-carry.md)

### Cash style

How a position's value and cash relate: pay in full (shares), premium (options), variation margin (futures and perpetuals: every mark-to-market change is settled in cash), over-the-counter at present value (swaps and forwards) or currency exchange (spot FX and crypto: the balances are the position). The ledger's arithmetic follows the cash style alone.

See: [unified-instrument-model](techniques/unified-instrument-model.md), [event-driven-engine-and-ledger](techniques/event-driven-engine-and-ledger.md)

### Cash-and-carry

*Also: reverse cash-and-carry*

Buying an asset and selling its future or perpetual in the same notional to earn the basis (and, for a perpetual, the funding the short receives) until the prices converge. The reverse trade sells the asset and buys the derivative when the derivative is cheap.

See: [multi-asset-strategy-api](techniques/multi-asset-strategy-api.md)

### Causality test

*Also: look-ahead test*

A check that replaces all data after a cutoff with noise and requires every output on or before the cutoff to be identical, proving no future information leaks in.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### CDS spread

The price of insurance against a borrower's default; an alternative-data indicator of credit stress.

See: [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Christoffersen test

A test that value-at-risk breaches are independent through time rather than clustered.

See: [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Clark-West test

A test of whether a larger forecasting model improves on a smaller model nested inside it, corrected for the noise cost of the extra parameters.

See: [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Clique centrality

How embedded an asset is in a filtered correlation graph: the total correlation over the four-cliques it belongs to. High means a hub that moves with many others, low a peripheral diversifier.

See: [graph-portfolios-and-hierarchical-sensitivity-parity](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md)

### Cointegration

A long-run relationship between non-stationary prices such that some combination of them is stationary and tends to revert.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md)

### Condition number

The ratio of the largest to the smallest eigenvalue of a matrix; a large one means inverting it magnifies noise.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Confidence

In this repo, |2 Phi(mean/std) - 1|: how many forecast standard deviations the mean is from zero, scaled to 0-1.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Configuration fingerprint

A short hash of the declared configuration; it changes if any declared value changes and is stored with every result.

See: [experiment-database-and-reproducibility](techniques/experiment-database-and-reproducibility.md)

### Contamination

When a model's training data includes the period or series you test it on, so a good result may be memory rather than forecasting.

See: [foundation-models](techniques/foundation-models.md)

### Content-addressed store

*Also: artifact store*

A store that names each object by the hash of its content, so identical content is stored once and a reference proves what was saved.

See: [research-operations](techniques/research-operations.md)

### Continuous contract

*Also: back-adjustment*

A single price series stitched from successive futures contracts; ratio or difference adjustment removes the gap at each roll so that returns are right.

See: [futures-and-commodity-curves](techniques/futures-and-commodity-curves.md)

### Contract multiplier

The currency value of one price point of one contract: 50 dollars per index point for an equity-index future, 100 shares for a listed option. Notional is quantity times multiplier times price.

See: [unified-instrument-model](techniques/unified-instrument-model.md)

### Copula

A function that joins separate return distributions into a joint one, so that the dependence can be modelled apart from the marginals. Tail dependence, how often assets crash together, depends on the copula family.

See: [tail-risk-evt-and-copulas](techniques/tail-risk-evt-and-copulas.md)

### Covariance matrix

A table of how every pair of assets' returns vary together; the foundation of portfolio risk.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Covered interest parity

The no-arbitrage relation that the forward exchange rate equals spot adjusted by the interest differential, so a forward discount is the rate difference.

See: [fx-carry-and-momentum](techniques/fx-carry-and-momentum.md)

### CPCV

*Also: combinatorial purged cross-validation*

Cross-validation that cuts the history into groups, tests every combination of a few of them, purges training observations whose labels overlap the test groups and embargoes the days after, and assembles the out-of-sample predictions into many complete paths. The spread over the paths shows how much a result depends on the order of events.

See: [scenario-generation-and-backtesting](techniques/scenario-generation-and-backtesting.md)

### CRPS

Continuous ranked probability score: a proper scoring rule for a whole forecast distribution that generalises absolute error; lower is better.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### CSCV

Combinatorially symmetric cross-validation: the splitting scheme behind the probability of backtest overfitting.

See: [reality-check-spa-pbo](techniques/reality-check-spa-pbo.md)

### CVaR

*Also: expected shortfall*

Conditional value at risk, or expected shortfall: the average loss on the days worse than the VaR quantile.

See: [mean-cvar](techniques/mean-cvar.md), [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Data contract

The promises a data feed makes, written down and enforced: time zone, calendar and session, publication lags, how stale a quote may be, what to do about gaps and revisions. Applying a contract converts a raw file to UTC and reports what it changed.

See: [point-in-time-market-data](techniques/point-in-time-market-data.md)

### DCC-GARCH

Dynamic conditional correlation: GARCH volatilities combined with a time-varying correlation matrix.

See: [dynamic-covariance](techniques/dynamic-covariance.md)

### Deflated Sharpe ratio

*Also: DSR*

The probability that a strategy's Sharpe exceeds what the best of N random trials would show, given sample length, skew and kurtosis.

See: [multiple-testing](techniques/multiple-testing.md)

### Degeneracy ordering

An ordering of a graph's vertices from repeated removal of one of smallest degree. The largest degree met is the graph's degeneracy (3 for a TMFG), and each vertex's running maximum is its core number.

See: [graph-portfolios-and-hierarchical-sensitivity-parity](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md)

### Delta hedging

Trading the underlying so that the option position's delta is close to zero; the remaining profit comes from gamma and the difference between realised and implied volatility.

See: [option-strategies-and-vol-premium](techniques/option-strategies-and-vol-premium.md)

### Desirability

In linearly solvable control, exp(-v) where v is the cost to go. The Bellman equation is linear in it, and the optimal controlled dynamics are the passive dynamics reweighted by it.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Deterministic replay

Running the same data, configuration and strategy twice and getting the same result bit for bit. The engine hashes its event log (a digest); a different digest means different data, code or settings.

See: [event-driven-engine-and-ledger](techniques/event-driven-engine-and-ledger.md), [paper-trading-and-replay](techniques/paper-trading-and-replay.md)

### Diebold-Mariano test

*Also: DM test*

A test of whether two forecasts have different average loss, with a robust standard error.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Difference in differences

*Also: DiD*

A causal estimator comparing the change over time in a treated group with the change in a control group.

See: [causal-inference](techniques/causal-inference.md)

### Diffusion model

*Also: DDPM*

A generative model trained to reverse a gradual noising process, so it can turn noise into realistic samples.

See: [diffusion-scenarios](techniques/diffusion-scenarios.md)

### Dollar-cost averaging

*Also: DCA*

Investing a sum in equal parts over several dates instead of at once. It wins in a falling market and loses to investing at once in a rising one; it reduces regret, not expected cost.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### Double machine learning

*Also: DML*

A causal estimator that removes the effect of controls from treatment and outcome using flexible models, cross-fitted, then regresses the residuals.

See: [causal-inference](techniques/causal-inference.md)

### Drawdown

The fall from a running peak to a later trough, as a share of the peak.

See: [performance-metrics](techniques/performance-metrics.md)

### Dropout

*Also: MC dropout*

Randomly zeroing parts of a network during training to reduce overfitting; kept on at prediction time it gives Monte Carlo uncertainty.

See: [bayesian-deep-learning](techniques/bayesian-deep-learning.md)

### Duration

*Also: DV01, convexity*

A bond's sensitivity to yield: duration is the percentage price change per unit yield, DV01 the dollar change per basis point, and convexity the second-order correction.

See: [yield-curves-and-fixed-income](techniques/yield-curves-and-fixed-income.md)

### DV01

The change in a bond's value for a one-basis-point move in yield; used to make bond trades neutral to parallel rate moves.

See: [fixed-income-and-volatility-strategies](techniques/fixed-income-and-volatility-strategies.md)

### Effective rank

The exponential of the entropy of normalised eigenvalues: the number of independent risk dimensions a covariance matrix really contains.

See: [pca-effective-rank](techniques/pca-effective-rank.md)

### Embargo

A gap between the end of training labels and the start of testing that stops overlapping outcomes from leaking.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Ensemble

Several models trained separately whose predictions are averaged; disagreement between them estimates uncertainty.

See: [bayesian-deep-learning](techniques/bayesian-deep-learning.md)

### Epistemic uncertainty

Uncertainty from not knowing the right model or parameters, as opposed to irreducible noise in the data.

See: [bayesian-deep-learning](techniques/bayesian-deep-learning.md)

### Event study

A method that measures the abnormal return of assets around the date of an event, relative to what a model fitted on an earlier window expected, and tests whether the average is more than noise.

See: [event-studies](techniques/event-studies.md)

### Evolution strategies

*Also: ES*

A way to optimise a policy by perturbing its parameters randomly and moving toward the perturbations that scored best.

See: [reinforcement-learning](techniques/reinforcement-learning.md)

### EWMA

Exponentially weighted moving average: an average where recent observations weigh more, controlled by a half-life.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Expectancy

The average profit of one round trip, wins and losses together: the win rate times the average win, minus the loss rate times the average loss. A strategy that wins rarely can still have a positive expectancy if its wins are big enough.

See: [performance-metrics](techniques/performance-metrics.md)

### Expected calibration error

*Also: ECE*

The weighted average gap between forecast probability and observed frequency across probability bins.

See: [calibration](techniques/calibration.md)

### Expected shortfall

*Also: ES, CVaR*

The average loss on the days when the loss exceeds the value-at-risk level. It looks at how bad the tail is, not only where it starts.

See: [tail-risk-evt-and-copulas](techniques/tail-risk-evt-and-copulas.md)

### Extreme value theory

*Also: EVT*

Statistics of the far tail. Losses above a high threshold follow approximately a generalised Pareto distribution, which gives value-at-risk and expected shortfall beyond the range of the sample.

See: [tail-risk-evt-and-copulas](techniques/tail-risk-evt-and-copulas.md)

### Factor

A systematic source of return shared across many assets, such as market, size, value, momentum.

See: [attribution](techniques/attribution.md), [cross-sectional-factors](techniques/cross-sectional-factors.md)

### Fama-MacBeth regression

Run one cross-sectional regression of returns on characteristics per date, then average the slopes over time and use the time-series standard error: the standard way to estimate a priced factor or characteristic premium.

See: [regression-and-panel-statistics](techniques/regression-and-panel-statistics.md)

### Flight to quality

*Also: flight to safety*

The move of money from risky to safe assets in a panic, seen as bonds rising while stocks fall and volatility jumping.

See: [portfolio-rebalancing-styles](techniques/portfolio-rebalancing-styles.md)

### Forecast

In this repo, an object with a mean, a standard deviation and a confidence, rather than a single number.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md), [plugin-framework](techniques/plugin-framework.md)

### Forward premium puzzle

The empirical finding that high-interest currencies do not depreciate by the interest differential, as uncovered interest parity says, so the carry trade has earned a premium.

See: [fx-carry-and-momentum](techniques/fx-carry-and-momentum.md)

### Foundation model

A large model pre-trained on a vast range of data that can be applied to new tasks with little or no further training.

See: [foundation-models](techniques/foundation-models.md)

### Fundamental law of active management

Information ratio is approximately the information coefficient times the square root of breadth.

See: [information-coefficient](techniques/information-coefficient.md)

### Funding rate

The periodic payment between longs and shorts of a perpetual future that keeps its price near the spot price. When the rate is positive longs pay shorts. It is the main carrying cost or income of a crypto perpetual position.

See: [contract-lifecycle](techniques/contract-lifecycle.md), [crypto-carry](techniques/crypto-carry.md)

### Funding ratio

A plan's assets divided by the present value of its liabilities. Below one the plan cannot meet its promises from what it holds.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### G-learning

Reinforcement learning with a penalty, in units of 1/beta, on the relative entropy of the policy from a prior policy. The optimal policy is the prior times exp(beta times the action value), so a large beta gives Q-learning's greedy policy and a small one gives back the prior.

See: [reinforcement-learning-from-tables-to-networks](techniques/reinforcement-learning-from-tables-to-networks.md)

### GARCH

A volatility model in which today's variance depends on yesterday's squared return and yesterday's variance, with reversion to a long-run level.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Gaussian mixture

A distribution made of several weighted normal distributions; used to combine forecasts while keeping their disagreement.

See: [forecast-combination](techniques/forecast-combination.md)

### GIRL

*Also: G-learning inverse reinforcement learning*

Finding the reward weights that make observed behaviour most likely if the agent followed the soft-optimal policy of G-learning. The reward is identified only up to what leaves the policy unchanged.

See: [reinforcement-learning-from-tables-to-networks](techniques/reinforcement-learning-from-tables-to-networks.md)

### Gradient boosting

An ensemble of shallow trees fitted one after another, each correcting the errors of the previous; the default learner for tabular data and prone to overfitting at low signal-to-noise.

See: [gradient-boosting-and-representation-learning](techniques/gradient-boosting-and-representation-learning.md)

### Graph attention network

*Also: GAT*

A neural network that lets each node of a graph weight its neighbours' features by learned attention.

See: [graph-neural-networks](techniques/graph-neural-networks.md)

### Greeks

Sensitivities of an option's value: delta to the underlying, gamma to delta, vega to volatility, theta to time and rho to interest rates.

See: [option-pricing-and-greeks](techniques/option-pricing-and-greeks.md)

### Guardrails (spending)

*Also: Guyton-Klinger*

A spending rule that raises or cuts withdrawals by a fixed step when the withdrawal rate moves outside a band around its starting level.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### HAC standard errors

*Also: Newey-West*

A covariance estimate for regression coefficients that stays valid when the errors are autocorrelated and heteroskedastic, as they are for overlapping or serially dependent returns. Newey-West is the usual Bartlett-weighted version.

See: [regression-and-panel-statistics](techniques/regression-and-panel-statistics.md)

### Half-life

The age at which an observation's weight in an exponential average has fallen to one half.

See: [volatility-forecasting](techniques/volatility-forecasting.md)

### Hamilton-Jacobi-Bellman equation

*Also: HJB*

The continuous-time Bellman equation: a partial differential equation for the value function, with a maximisation over the control inside. Often the value function is not smooth, and the viscosity solution is the one a monotone numerical scheme converges to.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Hedge (algorithm)

An online expert-aggregation rule that multiplicatively down-weights experts with recent losses.

See: [online-learning](techniques/online-learning.md)

### Hidden Markov model

*Also: HMM*

A model where an unobserved state follows a Markov chain and each state generates returns from its own distribution.

See: [regime-detection](techniques/regime-detection.md)

### Hierarchical sensitivity parity

*Also: HSP*

A generalisation of hierarchical risk parity: at each split of a dendrogram the two branches are weighted so their sensitivity (risk contribution, or exposure to factor shocks) is equal. The definition used here is the author's reading of the name.

See: [graph-portfolios-and-hierarchical-sensitivity-parity](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md)

### HRP

*Also: HERC*

Hierarchical risk parity: allocates by recursively splitting a correlation-based tree of assets by inverse cluster variance.

See: [hierarchical-risk-parity](techniques/hierarchical-risk-parity.md)

### I-Star

*Also: Kissell*

Kissell's pre-trade cost model: market impact in basis points from the size of the order against volume and the stock's volatility, for a trade done at once and for one done at a participation rate.

See: [execution-algorithms](techniques/execution-algorithms.md)

### Impact

*Also: market impact*

The price movement caused by your own trading, approximated here as proportional to volatility times the square root of size over volume.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Implementation shortfall

*Also: IS*

The difference between the value of a trade at the arrival price and what it actually cost to execute: spread, market impact and the price drift while waiting. The Almgren-Chriss optimum minimises its expected value plus a penalty on its variance.

See: [execution-algorithms](techniques/execution-algorithms.md)

### Implied volatility

The volatility that makes an option model match the market price of the option. The implied volatilities of all strikes and expiries form the volatility surface.

See: [option-pricing-and-greeks](techniques/option-pricing-and-greeks.md), [volatility-surfaces](techniques/volatility-surfaces.md)

### Importance sampling

A Monte Carlo method for rare events: draw from a distribution that makes the event common and reweight each draw by the likelihood ratio.

See: [resampling-and-monte-carlo](techniques/resampling-and-monte-carlo.md)

### Information coefficient

*Also: IC*

The rank correlation between a signal and subsequent returns.

See: [information-coefficient](techniques/information-coefficient.md)

### Information ratio

*Also: IR*

Excess return over a benchmark divided by the volatility of that excess (tracking error).

See: [information-coefficient](techniques/information-coefficient.md)

### Instrumental variable

*Also: IV, 2SLS*

A variable that moves the treatment but affects the outcome only through it, used to estimate causal effects when confounding is present.

See: [causal-inference](techniques/causal-inference.md)

### Integrated gradients

A neural-network attribution method that integrates the gradient along a path from a baseline input to the actual input.

See: [explainability](techniques/explainability.md)

### Isotonic regression

Fitting a non-decreasing step function; used to recalibrate probabilities without assuming a shape.

See: [calibration](techniques/calibration.md)

### Journal

The ledger's record of every movement of cash and value, one entry per cause (a fill, a cost, a mark, a coupon, a funding payment). Each entry satisfies an accounting identity, and attribution is a grouping of the journal.

See: [event-driven-engine-and-ledger](techniques/event-driven-engine-and-ledger.md)

### Kalman filter

A recursive estimator of a hidden state that updates as each observation arrives; used here for time-varying hedge ratios and coefficients.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md), [online-learning](techniques/online-learning.md)

### Kelly criterion

The bet size that maximises long-run growth; fractional Kelly uses a safer fraction of it.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Kupiec test

A test that the proportion of value-at-risk breaches equals the promised rate.

See: [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Leakage

*Also: look-ahead bias*

Any path by which future information reaches a decision made in the past.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Ledoit-Wolf shrinkage

A formula that blends the sample covariance matrix with a structured target to reduce estimation error.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Liability-driven investing

*Also: LDI*

Holding assets that move with the present value of liabilities, usually long bonds matched in duration, so that interest rates move both sides together and the funding ratio is stable.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### Limit order book

The list of resting buy and sell orders at each price. Its depth and the arrival of market orders determine spread and price impact.

See: [market-making-and-order-flow](techniques/market-making-and-order-flow.md)

### Liquidity seeking

An execution algorithm that scales its pace with the quality of liquidity it sees (volume, spread, depth): more when the market is deep, less when it is thin. What it holds back is traded later.

See: [basket-and-liquidity-algorithms](techniques/basket-and-liquidity-algorithms.md)

### Local volatility

The instantaneous volatility, a function of price and time, that reproduces all vanilla option prices; computed from the surface with Dupire's formula.

See: [volatility-surfaces](techniques/volatility-surfaces.md)

### Log loss

The negative average log probability assigned to what actually happened; a proper score for probability forecasts.

See: [calibration](techniques/calibration.md)

### Marchenko-Pastur

The distribution of eigenvalues of a random covariance matrix; its upper edge separates signal from noise eigenvalues.

See: [pca-effective-rank](techniques/pca-effective-rank.md)

### Market regime

*Also: regime*

A persistent state of the market (such as high volatility or inflation) in which returns, risks or correlations behave differently.

See: [regime-detection](techniques/regime-detection.md)

### Markov chain

A model in which the next state depends only on the current one. Transition probabilities give expected holding times and the long-run share of time in each state.

See: [bayesian-inference-and-markov-models](techniques/bayesian-inference-and-markov-models.md)

### Markov decision process

*Also: MDP, POMDP*

A model of sequential decisions: states, actions, transition probabilities, rewards and a discount. In a partially observed MDP the state is hidden and the agent holds a belief about it, updated by Bayes' rule.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### MCMC

*Also: Markov chain Monte Carlo*

A way to sample from a posterior distribution that has no closed form by running a chain whose long-run distribution is the posterior. R-hat and the effective sample size say whether it has converged.

See: [bayesian-inference-and-markov-models](techniques/bayesian-inference-and-markov-models.md)

### Mean-variance optimisation

*Also: MVO*

Choosing weights to maximise expected return for a given variance.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Meta-labelling

A second model that decides whether to act on a primary model's signal, trained on whether the primary's past calls were profitable. The bet size follows the estimated probability of being right.

See: [multi-asset-strategy-api](techniques/multi-asset-strategy-api.md)

### Minimum detectable effect

*Also: MDE*

The smallest true effect a test detects with a chosen probability (usually 80%) at a chosen significance level.

See: [power-analysis](techniques/power-analysis.md)

### Minimum spanning tree

*Also: MST*

The cheapest set of N-1 links connecting N assets when the distance is sqrt(2(1-correlation)): the skeleton of the correlation matrix.

See: [graph-portfolios-and-hierarchical-sensitivity-parity](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md)

### Minnesota prior

A prior for vector autoregressions that shrinks every coefficient toward a random walk, more strongly for distant lags and for other variables' lags. It makes large systems estimable on short samples.

See: [time-series-econometrics](techniques/time-series-econometrics.md), [econometric-strategies](techniques/econometric-strategies.md)

### Model registry

A versioned record of trained models with their parameters, data, code version, metrics and lifecycle stage, so that any production number can be traced to its origin.

See: [research-operations](techniques/research-operations.md)

### Momentum

The tendency of recent winners to keep winning over the following weeks to months.

See: [momentum](techniques/momentum.md)

### Money-weighted return

*Also: internal rate of return, IRR*

The rate that makes the investor's dated cash flows into and out of a portfolio worth zero. It depends on when money was added or taken out, unlike the time-weighted return.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### N-BEATS

*Also: N-HiTS*

A neural forecaster of stacked fully connected blocks that each model part of a series and subtract it before the next.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Nelson-Siegel

A three-factor model of the yield curve whose coefficients are the level, slope and curvature. The dynamic version forecasts the betas as a VAR.

See: [yield-curves-and-fixed-income](techniques/yield-curves-and-fixed-income.md)

### Net return

Return after trading costs.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### Newey-West

*Also: HAC*

A standard-error estimator that is robust to autocorrelation and heteroskedasticity up to a chosen lag.

See: [attribution](techniques/attribution.md)

### No-trade band

A rule not to trade unless a position is further than a set distance from its target, saving costs at the price of tracking error.

See: [execution-and-impact](techniques/execution-and-impact.md)

### No-trade region

*Also: no-trade band*

The range around the ideal holding inside which a trader with trading costs does nothing, trading to its edge only when the holding drifts out. It widens with the cost, roughly as its cube root.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Online learning

Updating a model after each new observation instead of training once.

See: [online-learning](techniques/online-learning.md)

### Order-flow imbalance

*Also: OFI*

The net change in resting buy and sell interest at the best quotes over an interval. It explains much of short-horizon mid-price change.

See: [market-making-and-order-flow](techniques/market-making-and-order-flow.md)

### Overfitting

Fitting noise in the sample so well that performance on new data is much worse.

See: [multiple-testing](techniques/multiple-testing.md), [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Pairs trading

Taking opposite positions in two related assets to profit when the gap between them closes.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md)

### Paper trading

Running a strategy against live or streamed data with simulated fills and no real money. Here it is the backtest engine fed a stream, so research code runs unchanged.

See: [paper-trading-and-replay](techniques/paper-trading-and-replay.md)

### Participation rate

*Also: POV, percent of volume*

The share of the market's volume that an order makes up. A POV algorithm trades a fixed participation rate, so it speeds up when the market does.

See: [execution-algorithms](techniques/execution-algorithms.md)

### Payoff ratio

The average winning trade divided by the size of the average losing trade. With the win rate it sets the expectancy: winning 30% of the time needs a payoff of about 2.3 to break even.

See: [performance-metrics](techniques/performance-metrics.md)

### PBO

Probability of backtest overfitting: how often the best in-sample rule ranks below the median out of sample.

See: [reality-check-spa-pbo](techniques/reality-check-spa-pbo.md)

### PCA

*Also: principal component*

Principal component analysis: rotating correlated variables into uncorrelated components ordered by variance.

See: [pca-effective-rank](techniques/pca-effective-rank.md)

### Permutation importance

How much error rises when one input is shuffled; a model-agnostic measure of use.

See: [explainability](techniques/explainability.md)

### Perpetual future

*Also: perp*

A futures contract without expiry that stays near spot through funding payments.

See: [crypto-carry](techniques/crypto-carry.md)

### Physics-informed neural network

*Also: PINN*

A network trained to make a differential equation's residual small at random points, plus its boundary conditions, instead of fitting data. Used here to price options without a grid.

See: [deep-learning-generative-scenarios-and-pinns](techniques/deep-learning-generative-scenarios-and-pinns.md)

### PIT

Probability integral transform: the forecast's cumulative distribution evaluated at the outcome; uniform if the forecast is calibrated.

See: [probabilistic-forecasting](techniques/probabilistic-forecasting.md)

### Platt scaling

Calibrating probabilities by fitting a logistic curve to the raw scores.

See: [calibration](techniques/calibration.md)

### Point in time

Data arranged so that on each date only what was known then is visible, including publication lags and not revisions.

See: [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Policy gradient

A family of reinforcement learning methods that adjust a policy's parameters in the direction that raises expected return, using the likelihood ratio (REINFORCE, actor-critic) or the slope of a critic in the action (deterministic policy gradient).

See: [reinforcement-learning-from-tables-to-networks](techniques/reinforcement-learning-from-tables-to-networks.md)

### Policy portfolio

*Also: strategic asset allocation*

The long-run target allocation an investor sets and returns to, rebalancing on a calendar, when weights drift beyond a band, or both.

See: [portfolio-rebalancing-styles](techniques/portfolio-rebalancing-styles.md)

### Pontryagin's principle

*Also: maximum principle*

A necessary condition for optimal control along one path: the control minimises the Hamiltonian at every instant and the costate follows an equation driven by the Hamiltonian's slope in the state, giving a boundary value problem.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Post-event drift

*Also: post-earnings-announcement drift*

The tendency of prices to keep moving in the direction of news for days or weeks, because information is absorbed gradually.

See: [alpha-generating-styles](techniques/alpha-generating-styles.md), [event-studies](techniques/event-studies.md)

### Posterior predictive

The distribution of a future observation averaged over the posterior uncertainty about parameters.

See: [bayesian-portfolio-construction](techniques/bayesian-portfolio-construction.md)

### Power

The probability a test rejects the null when a specified true effect exists.

See: [power-analysis](techniques/power-analysis.md)

### Pre-registration

Declaring hypotheses, rules and decision criteria before looking at results; here, committed configuration precedes the stage that uses it.

See: [experiment-database-and-reproducibility](techniques/experiment-database-and-reproducibility.md)

### Probabilistic Sharpe ratio

*Also: PSR*

The probability that the true Sharpe exceeds a threshold, given the estimate's uncertainty, skew and kurtosis.

See: [multiple-testing](techniques/multiple-testing.md)

### Probability of ruin

The probability that wealth ever falls to a given fraction of its starting level. It has closed forms for Brownian motion and depends on leverage, drift and volatility.

See: [drawdown-ruin-and-kelly](techniques/drawdown-ruin-and-kelly.md)

### Profit factor

Everything made in the winning periods divided by everything lost in the losing ones (the dashboard shows it by month and by day, and by trade). Above 1 means a net gain. Over a short sample, or after many tries, a high profit factor is easy to get by luck.

See: [performance-metrics](techniques/performance-metrics.md), [multiple-testing](techniques/multiple-testing.md)

### Purged cross-validation

*Also: CPCV*

Cross-validation for overlapping labels: training rows whose labels overlap the test period are removed, with an embargo after it. The combinatorial version builds many out-of-sample paths.

See: [research-operations](techniques/research-operations.md)

### PV01

*Also: DV01*

The change in a position's present value for a one basis point rise in rates. Swap trades are sized in PV01 so that the risk, not the notional, is the budget.

See: [interest-rate-swaps](techniques/interest-rate-swaps.md)

### Q-learning

*Also: SARSA, DQN*

Learning the value of each action in each state by moving it toward the reward plus the discounted best next value. SARSA uses the action actually taken next instead, so it learns the value of the policy it follows; DQN uses a neural network.

See: [reinforcement-learning-from-tables-to-networks](techniques/reinforcement-learning-from-tables-to-networks.md)

### QLIKE

A loss function for comparing variance forecasts that is robust to noise in the realised-variance proxy; lower is better.

See: [time-series-econometrics](techniques/time-series-econometrics.md)

### Reality Check

*Also: SPA*

White's bootstrap test of whether the best of many strategies beats a benchmark after accounting for the search.

See: [reality-check-spa-pbo](techniques/reality-check-spa-pbo.md)

### Rebate

*Also: maker rebate*

A payment an exchange makes to the side that adds liquidity (a resting order that is filled). A strategy that lives on rebates and spread pays for it in adverse selection.

See: [black-box-and-high-frequency-strategies](techniques/black-box-and-high-frequency-strategies.md)

### Reconciliation

The independent re-computation of what the ledger says: cash equals starting cash plus the journal, equity equals start plus P&L plus transfers, positions equal the sum of fills, and no expired contract is held.

See: [event-driven-engine-and-ledger](techniques/event-driven-engine-and-ledger.md)

### Regime-switching allocation

A portfolio rule that blends different allocators according to the probability of each regime.

See: [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)

### Reliability diagram

A plot of observed frequency against forecast probability; the diagonal is perfect calibration.

See: [calibration](techniques/calibration.md)

### Representation learning

*Also: contrastive learning, autoencoder*

Learning a compact description of the inputs without labels, by reconstructing them (autoencoder) or by pulling similar views together and pushing others apart (contrastive learning).

See: [gradient-boosting-and-representation-learning](techniques/gradient-boosting-and-representation-learning.md)

### Reservation price

The price at which a market maker is indifferent to trading, shifted from the mid-price against its inventory so that it tends to unwind a position.

See: [market-making-and-order-flow](techniques/market-making-and-order-flow.md)

### Ridge regression

Linear regression with a penalty on the size of coefficients, which stabilises estimates when predictors are noisy or correlated.

See: [machine-learning-for-returns](techniques/machine-learning-for-returns.md)

### Risk parity

Sizing assets so each contributes equally to portfolio risk.

See: [risk-parity](techniques/risk-parity.md)

### RLS

Recursive least squares: an exact online update of a linear regression with forgetting.

See: [online-learning](techniques/online-learning.md)

### Roll

Replacing an expiring futures contract with the next one. The roll is a trade (with costs), not a return; a continuous series must be adjusted so the gap between the two contracts does not appear as profit.

See: [contract-lifecycle](techniques/contract-lifecycle.md), [futures-and-commodity-curves](techniques/futures-and-commodity-curves.md)

### Roll yield

The return from rolling a futures position when the curve is not flat: positive in backwardation (the next contract is cheaper) and negative in contango.

See: [futures-and-commodity-curves](techniques/futures-and-commodity-curves.md)

### Roll-down

The change in a bond or swap's value as it ages along an unchanged yield curve: a ten-year swap on a steep curve becomes a nine-and-a-half-year swap that prices at a lower rate. Together with carry it is what a position earns if nothing moves.

See: [interest-rate-swaps](techniques/interest-rate-swaps.md), [yield-curves-and-fixed-income](techniques/yield-curves-and-fixed-income.md)

### Round trip

One stretch in which an asset is held on the same side: it opens when the position appears (a purchase, or a short sale) and closes when the position goes to zero or flips. The dashboard's Trades tab lists each one with its dates, prices and profit.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### Safe withdrawal rate

The highest starting spending rate a portfolio can sustain over a horizon with a chance of running out below a chosen limit. It depends on the asset mix, the horizon, the spending rule and the returns assumed.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### Scenario generator

A method that produces many plausible paths of returns for a set of assets: replayed history, bootstrap, copula, factor model, GARCH, or a trained network. Each preserves some features of the data and loses others.

See: [scenario-generation-and-backtesting](techniques/scenario-generation-and-backtesting.md)

### Schroedinger bridge

The process closest in relative entropy to a reference process that starts at one distribution and ends at another. Found by alternately rescaling to match each end (Sinkhorn's iteration).

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Sequence-of-returns risk

The danger that poor returns arrive early in the life of a portfolio that is being drawn down, which a later recovery cannot repair because the money is already gone.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### Shapley value

*Also: SHAP*

A fair division of a prediction among inputs, averaging each input's marginal contribution over all orders.

See: [explainability](techniques/explainability.md)

### Sharpe ratio

Mean excess return divided by its standard deviation, annualised.

See: [performance-metrics](techniques/performance-metrics.md)

### Shrinkage

Pulling an estimate toward a simpler target to reduce its variance at the price of some bias.

See: [mean-variance-and-shrinkage](techniques/mean-variance-and-shrinkage.md)

### Sinkhorn iteration

Alternately rescaling the rows and columns of a positive matrix until its sums match given targets. It solves the static Schroedinger problem and entropy-regularised optimal transport.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Slippage

The difference between the price you expected and the price you got, beyond spread and impact.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Sortino ratio

Mean excess return divided by downside deviation.

See: [performance-metrics](techniques/performance-metrics.md)

### Spread (bid-ask)

The gap between the best price to buy and the best price to sell.

See: [execution-and-impact](techniques/execution-and-impact.md)

### Squeeze (Bollinger)

*Also: volatility squeeze*

A spell when Bollinger bandwidth is at the low end of its recent range. Volatility clusters, so a squeeze tends to be followed by a larger move, and the direction of the first close outside the band is the best guess of its direction.

See: [alpha-generating-styles](techniques/alpha-generating-styles.md)

### Stationary

A series whose statistical properties do not change over time; required for a spread to revert.

See: [pairs-and-statistical-arbitrage](techniques/pairs-and-statistical-arbitrage.md)

### Survivorship bias

Testing only on assets that still exist, which overstates past performance.

See: [data-integrity](techniques/data-integrity.md)

### SVI

*Also: SSVI*

The stochastic volatility inspired parametrisation of total implied variance in log-moneyness, with a surface version (SSVI) whose parameters have simple conditions for no static arbitrage.

See: [volatility-surfaces](techniques/volatility-surfaces.md)

### Tail dependence

The probability that one asset is in its extreme tail given that another is. A Gaussian copula has none; Student-t and Clayton copulas do.

See: [tail-risk-evt-and-copulas](techniques/tail-risk-evt-and-copulas.md)

### Tear sheet

A one-page summary of a strategy with a rule-based red-flag list.

See: [plugin-framework](techniques/plugin-framework.md)

### Temporal Fusion Transformer

*Also: TFT*

A forecasting network that combines variable selection, gated residual blocks, a recurrent encoder and interpretable attention.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Time-weighted return

*Also: TWR*

The return of a portfolio with the effect of cash flows removed, by chaining the returns of the periods between them. It is what a backtest reports.

See: [cash-flow-strategies](techniques/cash-flow-strategies.md)

### TMFG

*Also: triangulated maximally filtered graph*

A planar graph with 3(N-2) links built from the four most correlated assets by repeatedly inserting the remaining asset into the triangle where its total correlation is largest. It keeps the loops a spanning tree drops.

See: [graph-portfolios-and-hierarchical-sensitivity-parity](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md)

### Tracking error

The volatility of the difference between a strategy's return and its benchmark's.

See: [paired-sharpe-bootstrap](techniques/paired-sharpe-bootstrap.md), [power-analysis](techniques/power-analysis.md)

### Transformer

*Also: PatchTST*

A neural network built on attention, now standard for sequences.

See: [deep-time-series-models](techniques/deep-time-series-models.md)

### Tree-structured Parzen estimator

*Also: TPE*

A hyperparameter search that models good and bad trials with density estimates and proposes the candidate with the highest ratio of the two.

See: [research-operations](techniques/research-operations.md)

### Turnover

The amount traded per year as a multiple of the portfolio; it drives trading cost.

See: [backtest-engine-and-costs](techniques/backtest-engine-and-costs.md)

### TWAP and VWAP

*Also: time-weighted average price, volume-weighted average price*

Execution algorithms that slice an order equally over time (TWAP) or in proportion to the day's expected volume (VWAP). VWAP is the cheapest schedule when impact is linear and price risk is ignored.

See: [execution-algorithms](techniques/execution-algorithms.md)

### Unit root

A series that accumulates shocks forever (a random walk) instead of returning to a mean. Prices usually have one, returns do not, and regressing one unit-root series on another can give a spurious relationship.

See: [unit-roots-and-cointegration](techniques/unit-roots-and-cointegration.md)

### Value at risk

*Also: VaR*

A quantile of the loss distribution: the loss exceeded only (1 - confidence) of the time.

See: [var-cvar-and-backtests](techniques/var-cvar-and-backtests.md)

### Variance ratio

The variance of q-period returns divided by q times the one-period variance. Above one signals momentum, below one mean reversion, and one a random walk.

See: [unit-roots-and-cointegration](techniques/unit-roots-and-cointegration.md)

### Variance risk premium

*Also: VRP*

The amount by which implied variance exceeds the variance later realised. It is positive on average in equity indices and is what short-volatility strategies collect, with crash risk.

See: [option-strategies-and-vol-premium](techniques/option-strategies-and-vol-premium.md), [volatility-surfaces](techniques/volatility-surfaces.md)

### Variation margin

The daily cash settlement of a futures or perpetual position's gain or loss. A futures contract has no value on the balance sheet: it is a stream of daily cash payments.

See: [unified-instrument-model](techniques/unified-instrument-model.md), [event-driven-engine-and-ledger](techniques/event-driven-engine-and-ledger.md)

### Vector autoregression

*Also: VAR*

A model in which every series in a vector depends on the lags of all of them. It gives forecasts, impulse responses and Granger tests, and needs shrinkage when there are many series.

See: [time-series-econometrics](techniques/time-series-econometrics.md)

### Viscosity solution

The generalised solution of a Hamilton-Jacobi-Bellman equation that exists even where the value function has kinks, such as at an option's exercise boundary. Monotone, stable and consistent schemes converge to it.

See: [dynamic-programming-and-optimal-control](techniques/dynamic-programming-and-optimal-control.md)

### Volatility surface

Implied volatility as a function of strike and expiry. It must satisfy no-arbitrage conditions; SVI and SSVI are parametric forms that can.

See: [volatility-surfaces](techniques/volatility-surfaces.md)

### Volatility targeting

*Also: vol target*

Scaling positions so the portfolio runs at a chosen level of volatility.

See: [risk-parity](techniques/risk-parity.md), [regime-adaptive-allocation](techniques/regime-adaptive-allocation.md)

### Walk-forward

Testing by repeatedly fitting on the past and predicting the next block of the future.

See: [walk-forward-and-leakage](techniques/walk-forward-and-leakage.md)

### Walk-forward learning

Refitting a model on a rolling window of data known at the time, using only labels whose horizon has already elapsed, and predicting the next period. It is the only honest way to evaluate a learner on time series.

See: [multi-asset-strategy-api](techniques/multi-asset-strategy-api.md)

### Wasserstein GAN

*Also: WGAN, WGAN-GP*

A generative adversarial network whose critic estimates the Wasserstein distance between real and generated data, with a penalty on its gradient (GP) to keep it well behaved. Steadier to train than the original GAN.

See: [deep-learning-generative-scenarios-and-pinns](techniques/deep-learning-generative-scenarios-and-pinns.md)

### Win rate

The share of trades (or of days or months) that made money. It says nothing alone: a strategy can win 80% of the time and still lose money if the losses are much larger than the wins; read it with the payoff ratio.

See: [performance-metrics](techniques/performance-metrics.md)

### Yield curve

Yields of bonds of different maturities; its slope and shape are classic macro signals and the basis of steepener and butterfly trades.

See: [fixed-income-and-volatility-strategies](techniques/fixed-income-and-volatility-strategies.md), [macro-and-alternative-data](techniques/macro-and-alternative-data.md)

### Z-score

A value minus its mean, divided by its standard deviation: how many standard deviations from normal.

See: [mean-reversion](techniques/mean-reversion.md)

### Zero-shot

Applying a pre-trained model to a new task without any training on that task's data.

See: [foundation-models](techniques/foundation-models.md)
