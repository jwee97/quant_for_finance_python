# The two books: where everything is

This page maps the topics of two books, *Quantitative Equity Portfolio Management: Modern Techniques and Applications* (Book 1) and *Quantitative Portfolio Optimization: Advanced Techniques and Applications* (Book 2), item by item, to the code that implements them. It is generated from `src/books/catalog.py` (`python -m src.books.catalog`), and a test imports every location it names, so it cannot describe something that is not there.

**Status.** *built*: written for this item. *existing*: the repository already had it, and the entry points to it. *data*: built, and reads a file you supply (`data/user/fundamentals.csv`, `earnings_dates.csv`) or a macro series; it refuses to run without it instead of inventing a signal. *simulator*: it runs on a simulated or stylised market, or its solver stands in for the one named (simulated annealing for quantum annealing). *interpretation*: the item names a method whose definition was not available here, so what is built is the author's documented reading.

**Where to run things.** Models are in `quant backtest --model NAME` and the dashboard's strategy list; allocators are `--allocator NAME` (with `--alloc-param key=value`). The numerical modules (`src/control`, `src/rl`, `src/scenarios`, `src/equity`) are Python libraries with the experiments `experiments/control_world.py`, `deep_world.py`, `portfolio_world.py`, `trading_world.py`, `timing_world.py`, `bayes_world.py` and `equity_world.py`, which print the numbers the guides quote. Guides: [building an alpha model](techniques/alpha-model-construction.md), [fundamental factors](techniques/fundamental-factors.md), [portfolio theory and constrained books](techniques/portfolio-theory-and-constrained-books.md), [factor models and machine learning](techniques/factor-models-and-machine-learning.md), [trading with costs](techniques/multi-period-trading-and-order-routing.md), [factor timing](techniques/factor-timing.md), [hierarchical Bayes and Gaussian processes](techniques/hierarchical-bayes-decisions-and-gaussian-processes.md), [dynamic programming and optimal control](techniques/dynamic-programming-and-optimal-control.md), [reinforcement learning](techniques/reinforcement-learning-from-tables-to-networks.md), [deep networks, generative scenarios and PINNs](techniques/deep-learning-generative-scenarios-and-pinns.md), [graph portfolios](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md), [scenario generation and backtesting](techniques/scenario-generation-and-backtesting.md).

## Book 1, 1. Alpha generation

| Item | Status | Where | What it does |
|---|---|---|---|
| Value factors | data | `fundamental_value` (model), `src.equity.factors.compute` | cash flow, EBITDA, earnings, forward earnings, payout, net financing, book and sales over enterprise value; needs data/user/fundamentals.csv |
| Quality factors | data | `fundamental_quality` (model), `src.equity.factors.rnoa`, `src.equity.factors.cfroi` | RNOA, CFROI (an IRR), operating leverage, accruals-type increases, capital expenditure, external financing, share issuance; needs data/user/fundamentals.csv |
| Momentum factors | built | `fundamental_momentum` (model), `momentum` (model), `residual_momentum` (model) | one-month reversal, nine-month and risk-adjusted return, earnings revisions (these need the file); the price-only ones do not |
| Discounted cash flow | data | `fundamental_dcf` (model), `src.equity.dcf.dcf_value` | fading-growth DCF whose upside over the price is the signal; needs data/user/fundamentals.csv |
| Multipath DCF | data | `src.equity.dcf.mdcf_values`, `src.equity.dcf.mdcf_distribution` | Monte Carlo over growth, cash flow, discount and terminal rates: a distribution of values and the probability of being above the price; needs data/user/fundamentals.csv |
| Contextual models | data | `src.equity.contextual.contextual_alpha` | the alpha model run separately in terciles of value, growth, earnings variability or size; needs data/user/fundamentals.csv |
| Nonlinear models | data | `fundamental_nonlinear` (model), `src.equity.contextual.nonlinear_features`, `src.equity.contextual.fama_macbeth_forecast` | squares, products and conditional terms with a Fama-MacBeth forecast; needs data/user/fundamentals.csv |

## Book 1, 2. Alpha model construction

| Item | Status | Where | What it does |
|---|---|---|---|
| Optimal (IC-based) alpha model | built | `src.equity.alpha_model.optimal_alpha`, `fundamental_alpha` (model), `src.framework.ic_combination.optimal_ic_weights` | weights proportional to the inverse covariance of the factors' information coefficients times their means, shrunk, optionally non-negative, walk-forward |
| Z-scores | built | `src.equity.alpha_model.standardize`, `src.equity.alpha_model.standardize_all` | winsorised cross-sectional z-scores over the investable names |
| Gram-Schmidt orthogonalisation | built | `src.equity.alpha_model.gram_schmidt`, `src.equity.alpha_model.symmetric_orthogonalize` | sequential (order matters) and symmetric (Loewdin) orthogonalisation of factors |
| Fama-MacBeth marginal contributions | built | `src.equity.alpha_model.marginal_contributions`, `src.stats.panel.fama_macbeth` | what each factor adds after the others, from cross-sectional regressions by period |

## Book 1, 3. Portfolio construction

| Item | Status | Where | What it does |
|---|---|---|---|
| Mean-variance optimisation | existing | `src.portfolio.mean_variance.mean_variance_weights`, `tca_mvo` (allocator) | also with trading costs inside the optimiser |
| Black-Litterman | existing | `black_litterman` (allocator), `src.portfolio.black_litterman.black_litterman_posterior` | market-implied returns tilted by views from the forecasts |
| Equal risk contribution (risk parity) | existing | `src.portfolio.risk_parity.risk_parity_weights` | Newton and SLSQP solvers, risk budgets |
| Hierarchical risk parity | existing | `src.portfolio.hierarchical.hrp_weights`, `src.portfolio.hierarchical.herc_weights` | HRP and HERC on a correlation dendrogram |
| 130/30 and market-neutral books | built | `constrained_long_short` (allocator), `src.equity.portfolio.optimise_book`, `src.equity.portfolio.preset` | one quadratic programme on split long and short weights: 130/30, 120/20, market and dollar neutral, long-only |
| Sector, beta and dollar neutrality | built | `src.equity.portfolio.BookSpec`, `beta_neutral` (allocator) | exposure bounds on any matrix of exposures (sector dummies, betas), a net-exposure target |
| Turnover limits | built | `src.equity.portfolio.BookSpec`, `multi_period` (allocator) | a hard turnover limit and proportional cost in the book; the multi-period allocator trades the cost-optimal amount |
| Long-only | existing | `src.equity.portfolio.preset`, `min_variance` (allocator) | the long-only preset and the existing long-only books |

## Book 1, 4. Trading and implementation

| Item | Status | Where | What it does |
|---|---|---|---|
| Almgren-Chriss | existing | `src.algo.optimize.optimal_schedule`, `src.control.pontryagin.liquidation_pmp`, `src.control.pontryagin.almgren_chriss_closed_form` | the quadratic-programme schedule, and the same problem as a boundary value problem checked against the closed-form sinh schedule |
| Multi-period optimisation (Mei et al.) | built | `src.equity.multiperiod.lq_solve`, `src.equity.multiperiod.stationary_trade_rate`, `multi_period` (allocator) | quadratic-cost dynamic programme in closed form: the aim portfolio and the trade rate |
| Constrained multi-period (Skaf and Boyd) | built | `src.equity.multiperiod.mpc_plan`, `src.equity.multiperiod.mpc_step`, `src.equity.multiperiod.no_trade_region` | model-predictive control with limits and proportional costs as a QP with an approximate-dynamic-programming terminal reward |
| VWAP | existing | `src.algo.algos.VWAP` | slices follow the expected volume profile |
| POV | existing | `src.algo.algos.POV` | a fixed share of the volume |
| Trade-schedule optimisation | existing | `src.algo.optimize.optimal_schedule`, `src.algo.optimize.frontier`, `src.algo.optimize.exponential_trade` | cost-risk frontier, exponential schedules |
| Trade-rate parameter | existing | `src.algo.optimize.trade_rate`, `src.algo.optimize.fit_trade_rate` | a constant fraction of the order per period, fitted to the objective |
| Limit-order models (limit/market mix) | simulator | `src.algo.limit_orders.solve`, `src.algo.limit_orders.simulate` | dynamic programme over the number of limit and market units per interval; fills are drawn from the assumed distribution, there is no order book |
| Smart order routing | simulator | `src.algo.routing.greedy_allocation`, `src.algo.routing.kaplan_meier_tail`, `src.algo.routing.simulate_routing` | send to the venues most likely to fill, learning fill probabilities from censored fills; venues are simulated |

## Book 1, 5. Factor timing

| Item | Status | Where | What it does |
|---|---|---|---|
| Calendar timing: January effect, quarterly horizons | built | `calendar_factor_timing` (model) | premium forecast from earlier months in the same calendar state (January, month, month within the quarter, November-April) |
| Seasonal earnings-announcement timing | data | `earnings_season_premium` (model) | announcement-month premium; needs data/user/earnings_dates.csv, or infers announcements from volume spikes |
| Macro timing: Fed rates, M1, GDP, inflation, PPI, up and down markets | built | `macro_factor_timing` (model) | the same premium forecast conditioned on policy-rate changes, growth of M1, GDP, CPI or PPI against their median, or the market's trailing return |

## Book 2, 1. Portfolio theory and optimisation

| Item | Status | Where | What it does |
|---|---|---|---|
| Modern portfolio theory, mean-variance, capital market line, tangency | built | `src.equity.theory.frontier_constants`, `src.equity.theory.tangency_portfolio`, `src.equity.theory.capital_market_line`, `tangency_cml` (allocator) | frontier constants A, B, C, D, the tangency portfolio, the CML and its allocation |
| CAPM: standard and zero-beta | built | `src.equity.theory.capm_regression`, `src.equity.theory.security_market_line`, `src.equity.theory.zero_beta_portfolio` | time-series betas, the security market line with Shanken's correction, with and without a riskless rate |
| Arbitrage pricing theory | built | `src.equity.theory.apt_two_pass`, `src.equity.theory.apt_arbitrage_portfolio`, `apt_alpha` (model) | two-pass estimation, statistical factors, the cheapest arbitrage portfolio |
| Risk parity: ERC and HRP | existing | `src.portfolio.risk_parity.risk_parity_weights`, `src.portfolio.hierarchical.hrp_weights` | see Book 1, 3 |
| Mean-variance with CVaR | built | `src.equity.cvar_portfolio.mean_variance_cvar`, `mv_cvar` (allocator) | mean-variance under a CVaR limit (Rockafellar-Uryasev), cutting planes with the scenario programme as the reference |
| Quantum annealing and QUBO portfolios | simulator | `src.equity.qubo.weight_qubo`, `src.equity.qubo.cardinality_qubo`, `src.equity.qubo.simulated_annealing`, `qubo_select` (allocator) | the QUBO formulations are exact; the solver is classical simulated annealing, checked against exhaustive search. No quantum hardware |

## Book 2, 2. Bayesian and probabilistic methods

| Item | Status | Where | What it does |
|---|---|---|---|
| Black-Litterman | existing | `black_litterman` (allocator) | see Book 1, 3 |
| Hierarchical Bayes | built | `src.portfolio.hierarchical_bayes.hierarchical_posterior`, `hierarchical_bayes` (allocator) | expected returns shrunk toward their group and groups toward the whole, with the scales inferred on a grid |
| Bayesian decision theory | built | `src.portfolio.decision_theory.bayes_weights`, `src.portfolio.decision_theory.regret`, `src.portfolio.decision_theory.value_of_information`, `bayes_expected_utility` (allocator) | weights maximising expected utility over the posterior predictive, regret and the value of information |
| Gaussian process regression | built | `src.models.gaussian_process.GaussianProcess`, `gp_factor_model` (model) | exact GP with ARD kernels and marginal-likelihood hyperparameters, as a monthly return forecaster with its uncertainty |

## Book 2, 3. Factor investing and machine learning

| Item | Status | Where | What it does |
|---|---|---|---|
| Statistical factors (PCA) | built | `src.equity.theory.statistical_factors`, `apt_alpha` (model) | principal-component factors and the APT alpha built on them |
| Macroeconomic factor models | data | `macro_factor_model` (model) | returns on macro series declared on the model card; needs the macro series (FRED, point in time) |
| Cross-sectional factor models | built | `characteristic_regression` (model) | Fama-MacBeth style forecasts from price characteristics |
| LASSO, ridge and elastic net | built | `ml_factor_model` (model) | kind=ridge, lasso, enet, ols |
| PCR and PLS | built | `ml_factor_model` (model) | kind=pcr, pls |
| Random forests and CART | built | `ml_factor_model` (model), `ml_trees` (model) | kind=cart, forest |
| Neural networks | built | `ml_factor_model` (model), `deep_window` (model) | kind=mlp on characteristics; the window networks of Book 2, 5 on returns |

## Book 2, 4. Dynamic programming and reinforcement learning

| Item | Status | Where | What it does |
|---|---|---|---|
| Markov decision processes: fully observed, finite and infinite horizon | built | `src.control.mdp.MDP`, `src.control.mdp.value_iteration`, `src.control.mdp.policy_iteration`, `src.control.mdp.backward_induction` | exact solutions for small finite MDPs |
| Partially observed MDPs | built | `src.control.mdp.POMDP`, `src.control.mdp.pbvi`, `src.control.mdp.expectimax` | belief updates, exact tree search and point-based value iteration, checked on the tiger problem |
| Risk-sensitive MDPs | built | `src.control.mdp.risk_sensitive_value_iteration` | entropic risk |
| Optimal control: Bellman, HJB | built | `src.control.mdp.value_iteration`, `src.control.hjb.merton_hjb` | the Bellman equation in discrete time, the HJB equation by a monotone scheme with policy iteration |
| Pontryagin | built | `src.control.pontryagin.solve_pmp`, `src.control.pontryagin.lqr_pmp`, `src.control.pontryagin.ramsey_pmp` | boundary value problem, checked against Riccati, the sinh schedule and the Ramsey steady state |
| Merton's problem | built | `src.control.merton.merton_fraction`, `src.control.merton.CostDP`, `src.control.hjb.merton_hjb` | closed forms, the proportional-cost no-trade band by dynamic programming, and the HJB solution |
| Viscosity solutions | built | `src.control.hjb.american_put_hjb` | an obstacle problem solved by a monotone scheme, converging to the binomial price |
| Schroedinger control | built | `src.control.schrodinger.markov_bridge`, `src.control.schrodinger.sinkhorn`, `src.control.schrodinger.solve_lmdp_first_exit` | Schroedinger bridges by Sinkhorn and linearly solvable control |
| SARSA, expected SARSA | built | `src.rl.tabular.train` | method=sarsa, expected_sarsa |
| REINFORCE | built | `src.rl.policy_gradient.reinforce` | linear softmax policy with a baseline |
| PPO | built | `src.rl.deep.PPO` | clipped surrogate with GAE (torch) |
| Q-learning, double Q-learning, DQN | built | `src.rl.tabular.train`, `src.rl.deep.DQN` | method=q_learning, double_q; DQN with double=True for double DQN (torch) |
| Deterministic policy gradient, DDPG | built | `src.rl.policy_gradient.deterministic_policy_gradient`, `src.rl.deep.DDPG` | compatible-critic DPG with linear features; DDPG in torch |
| A2C, SAC, TD3 | built | `src.rl.policy_gradient.a2c`, `src.rl.deep.SAC`, `src.rl.deep.TD3` | n-step actor-critic with linear features; SAC and TD3 in torch |
| G-learning and GIRL | built | `src.rl.glearning.g_learning`, `src.rl.glearning.soft_value_iteration`, `src.rl.glearning.girl` | entropy-regularised RL relative to a prior policy, and maximum-likelihood inverse reinforcement learning |
| RL for exposure | simulator | `q_learning_exposure` (allocator), `src.rl.exposure.fit_exposure_policy` | fitted Q-iteration on the empirical transitions of a book's own history; not in the list, the use of the above on the platform's data |

## Book 2, 5. Deep learning

| Item | Status | Where | What it does |
|---|---|---|---|
| Feed-forward, convolutional, recurrent and Transformer networks | built | `deep_window` (model), `src.models.deep_sequence.build_sequence_network` | kind=fnn, cnn, lstm, gru, transformer (and the earlier patch transformer, mixer, N-BEATS, N-HiTS, TimeMixer-style, TFT) on the normalised 252-day window |
| GAN and WGAN | built | `src.models.generative.WGANGP`, `src.scenarios.generators.wgan_gp` | Wasserstein GAN with gradient penalty as a scenario generator |
| VAE and NeuralFactors | interpretation | `src.models.generative.FactorVAE`, `src.scenarios.generators.factor_vae` | a variational autoencoder with a K-factor decoder and Student-t noise, in the spirit of NeuralFactors; not the paper's model |
| Physics-informed networks: BSM, Vasicek, Heston, Bates | built | `src.models.pinn.BSMPINN`, `src.models.pinn.VasicekPINN`, `src.models.pinn.HestonPINN`, `src.models.pinn.bates_price` | PDE residual training checked against Black-Scholes, the affine bond price and Fourier prices; accuracy about 0.5% of the strike |

## Book 2, 6. Graph-based portfolios

| Item | Status | Where | What it does |
|---|---|---|---|
| Minimum spanning tree | built | `src.portfolio.graph_portfolio.mst` | the MST of the correlation distance |
| TMFG | built | `src.portfolio.graph_portfolio.tmfg` | triangulated maximally filtered graph |
| Degeneracy ordering | built | `src.portfolio.graph_portfolio.degeneracy_ordering` | smallest-last ordering, core numbers, degeneracy |
| Clique centrality | built | `src.portfolio.graph_portfolio.centrality`, `graph_centrality` (allocator) | centrality (degree, strength, eigenvector, clique) and peripheral or central portfolios |
| Hierarchical sensitivity parity | interpretation | `src.portfolio.graph_portfolio.hsp_weights`, `hierarchical_sensitivity_parity` (allocator) | the published definition was not available: built as HRP with each split equalising the branches' risk contribution or factor sensitivity |

## Book 2, 7. Backtesting and scenario generation

| Item | Status | Where | What it does |
|---|---|---|---|
| Walk-forward backtesting | existing | `src.validation.walk_forward.run_walk_forward`, `src.validation.walk_forward.WalkForwardSplitter` | expanding or rolling folds with an embargo |
| Resampling: cross-validation and bootstrap | built | `src.validation.walk_forward.purged_kfold_indices`, `src.validation.cpcv.cpcv_splits`, `src.validation.cpcv.assemble_paths`, `src.probability.resampling.bootstrap_statistic` | purged k-fold, combinatorial purged cross-validation with its paths, block bootstraps |
| Monte Carlo | existing | `src.probability.montecarlo.mc_estimate`, `src.scenarios.backtest.scenario_backtest` | Monte Carlo estimation and importance sampling; a rule's results over generated scenarios |
| Generative models for scenarios | built | `src.scenarios.generators.generate` | WGAN-GP, factor VAE and diffusion generators with a common interface |
| Historical scenarios | built | `src.scenarios.generators.historical` | random windows of the history |
| Bootstrap scenarios | built | `src.scenarios.generators.bootstrap` | iid, moving, circular or stationary block bootstrap of whole rows |
| Copula scenarios | existing | `src.scenarios.generators.copula`, `src.probability.copulas.simulate_joint` | fitted copula with empirical or EVT marginals |
| Risk-factor scenarios | built | `src.scenarios.generators.risk_factor` | principal-component factors bootstrapped in blocks plus residuals |
| ARIMA-GARCH scenarios | built | `src.scenarios.generators.arima_garch` | AR(1) mean and GJR-GARCH volatility with jointly resampled standardised residuals |
| GAN and VAE scenarios | built | `src.scenarios.generators.wgan_gp`, `src.scenarios.generators.factor_vae`, `src.scenarios.generators.diffusion` | learned generators (torch) |
| Scenario quality | built | `src.scenarios.quality.quality_report`, `src.scenarios.quality.compare` | marginals, volatility, correlation, tails, tail dependence, volatility clustering, expected shortfall, horizon volatility |

## What this does not do

- **Simulated worlds are plumbing checks, not market evidence.** Every new method was tested against a world whose truth was planted, and most were run on the platform's 15 ETFs; those runs are reported as they came out, and several of the new methods did not beat simple baselines (the guides say which).
- **Fundamentals need your file.** The value, quality, momentum-with-revisions, DCF and nonlinear equity models read `data/user/fundamentals.csv` and refuse to run without it; the earnings-season model reads `data/user/earnings_dates.csv` or, failing that, infers announcements from volume spikes. The repository has no fundamentals feed.
- **No quantum hardware.** The QUBO portfolio formulations are exact; the solver is simulated annealing, checked against exhaustive search.
- **Reinforcement learning, PINNs and generative models are small research implementations.** They run in minutes on a CPU, are checked against problems with known answers, and are accurate to a fraction of a percent of a strike or a few hundredths of a correlation, not to production tolerance.
- **Two items are interpretations.** The VAE follows the *idea* of NeuralFactors (a latent factor structure with heavy-tailed noise), not the paper's model; hierarchical sensitivity parity is a documented generalisation of HRP because the published definition was not available.
- **Execution models are simulators** (see [algorithmic trading](algorithmic_trading.md)): no order book, queue or latency.
