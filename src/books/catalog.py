"""Where every item of two books' taxonomies lives in this repository.

**Book 1**, *Quantitative Equity Portfolio Management: Modern Techniques and Applications* (alpha generation, alpha model construction, portfolio construction, trading and implementation, factor timing), and
**Book 2**, *Quantitative Portfolio Optimization: Advanced Techniques and Applications* (portfolio theory, Bayesian methods, factor investing and machine learning, dynamic programming and reinforcement learning,
deep learning, graph-based methods, backtesting and scenario generation), are listed item by item as they were requested. For each item the catalogue says

* ``status``: ``built`` (written for this item), ``existing`` (the repository already had it; the entry points to it), ``data`` (built, and reads a file you supply: it refuses to run without it),
  ``simulator`` (runs on a stylised or simulated market, or by a method that stands in for the named one: the note says which), ``interpretation`` (the item names a method whose definition was not
  available, so what is built is the author's documented reading of it);
* ``where``: ``py:module.object`` for Python code, ``model:name`` for a strategy in the model registry, ``allocator:name`` for a portfolio allocator.

``tests/test_books_catalog.py`` imports every ``where``, so this file cannot describe something that is not there; ``docs/quant_books_coverage.md`` is written from it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    section: str
    item: str
    status: str
    where: tuple
    note: str = ""


B1 = {1: "Book 1, 1. Alpha generation", 2: "Book 1, 2. Alpha model construction", 3: "Book 1, 3. Portfolio construction", 4: "Book 1, 4. Trading and implementation", 5: "Book 1, 5. Factor timing"}
B2 = {1: "Book 2, 1. Portfolio theory and optimisation", 2: "Book 2, 2. Bayesian and probabilistic methods", 3: "Book 2, 3. Factor investing and machine learning",
      4: "Book 2, 4. Dynamic programming and reinforcement learning", 5: "Book 2, 5. Deep learning", 6: "Book 2, 6. Graph-based portfolios", 7: "Book 2, 7. Backtesting and scenario generation"}

FUND = "needs data/user/fundamentals.csv"

ITEMS: list[Item] = [
    # ------------------------------------------------------------------------------------------------------------------ Book 1
    Item(B1[1], "Value factors", "data", ("model:fundamental_value", "py:src.equity.factors.compute"), f"cash flow, EBITDA, earnings, forward earnings, payout, net financing, book and sales over enterprise value; {FUND}"),
    Item(B1[1], "Quality factors", "data", ("model:fundamental_quality", "py:src.equity.factors.rnoa", "py:src.equity.factors.cfroi"), f"RNOA, CFROI (an IRR), operating leverage, accruals-type increases, capital expenditure, external financing, share issuance; {FUND}"),
    Item(B1[1], "Momentum factors", "built", ("model:fundamental_momentum", "model:momentum", "model:residual_momentum"), "one-month reversal, nine-month and risk-adjusted return, earnings revisions (these need the file); the price-only ones do not"),
    Item(B1[1], "Discounted cash flow", "data", ("model:fundamental_dcf", "py:src.equity.dcf.dcf_value"), f"fading-growth DCF whose upside over the price is the signal; {FUND}"),
    Item(B1[1], "Multipath DCF", "data", ("py:src.equity.dcf.mdcf_values", "py:src.equity.dcf.mdcf_distribution"), f"Monte Carlo over growth, cash flow, discount and terminal rates: a distribution of values and the probability of being above the price; {FUND}"),
    Item(B1[1], "Contextual models", "data", ("py:src.equity.contextual.contextual_alpha",), f"the alpha model run separately in terciles of value, growth, earnings variability or size; {FUND}"),
    Item(B1[1], "Nonlinear models", "data", ("model:fundamental_nonlinear", "py:src.equity.contextual.nonlinear_features", "py:src.equity.contextual.fama_macbeth_forecast"), f"squares, products and conditional terms with a Fama-MacBeth forecast; {FUND}"),
    Item(B1[2], "Optimal (IC-based) alpha model", "built", ("py:src.equity.alpha_model.optimal_alpha", "model:fundamental_alpha", "py:src.framework.ic_combination.optimal_ic_weights"), "weights proportional to the inverse covariance of the factors' information coefficients times their means, shrunk, optionally non-negative, walk-forward"),
    Item(B1[2], "Z-scores", "built", ("py:src.equity.alpha_model.standardize", "py:src.equity.alpha_model.standardize_all"), "winsorised cross-sectional z-scores over the investable names"),
    Item(B1[2], "Gram-Schmidt orthogonalisation", "built", ("py:src.equity.alpha_model.gram_schmidt", "py:src.equity.alpha_model.symmetric_orthogonalize"), "sequential (order matters) and symmetric (Loewdin) orthogonalisation of factors"),
    Item(B1[2], "Fama-MacBeth marginal contributions", "built", ("py:src.equity.alpha_model.marginal_contributions", "py:src.stats.panel.fama_macbeth"), "what each factor adds after the others, from cross-sectional regressions by period"),
    Item(B1[3], "Mean-variance optimisation", "existing", ("py:src.portfolio.mean_variance.mean_variance_weights", "allocator:tca_mvo"), "also with trading costs inside the optimiser"),
    Item(B1[3], "Black-Litterman", "existing", ("allocator:black_litterman", "py:src.portfolio.black_litterman.black_litterman_posterior"), "market-implied returns tilted by views from the forecasts"),
    Item(B1[3], "Equal risk contribution (risk parity)", "existing", ("py:src.portfolio.risk_parity.risk_parity_weights",), "Newton and SLSQP solvers, risk budgets"),
    Item(B1[3], "Hierarchical risk parity", "existing", ("py:src.portfolio.hierarchical.hrp_weights", "py:src.portfolio.hierarchical.herc_weights"), "HRP and HERC on a correlation dendrogram"),
    Item(B1[3], "130/30 and market-neutral books", "built", ("allocator:constrained_long_short", "py:src.equity.portfolio.optimise_book", "py:src.equity.portfolio.preset"), "one quadratic programme on split long and short weights: 130/30, 120/20, market and dollar neutral, long-only"),
    Item(B1[3], "Sector, beta and dollar neutrality", "built", ("py:src.equity.portfolio.BookSpec", "allocator:beta_neutral"), "exposure bounds on any matrix of exposures (sector dummies, betas), a net-exposure target"),
    Item(B1[3], "Turnover limits", "built", ("py:src.equity.portfolio.BookSpec", "allocator:multi_period"), "a hard turnover limit and proportional cost in the book; the multi-period allocator trades the cost-optimal amount"),
    Item(B1[3], "Long-only", "existing", ("py:src.equity.portfolio.preset", "allocator:min_variance"), "the long-only preset and the existing long-only books"),
    Item(B1[4], "Almgren-Chriss", "existing", ("py:src.algo.optimize.optimal_schedule", "py:src.control.pontryagin.liquidation_pmp", "py:src.control.pontryagin.almgren_chriss_closed_form"), "the quadratic-programme schedule, and the same problem as a boundary value problem checked against the closed-form sinh schedule"),
    Item(B1[4], "Multi-period optimisation (Mei et al.)", "built", ("py:src.equity.multiperiod.lq_solve", "py:src.equity.multiperiod.stationary_trade_rate", "allocator:multi_period"), "quadratic-cost dynamic programme in closed form: the aim portfolio and the trade rate"),
    Item(B1[4], "Constrained multi-period (Skaf and Boyd)", "built", ("py:src.equity.multiperiod.mpc_plan", "py:src.equity.multiperiod.mpc_step", "py:src.equity.multiperiod.no_trade_region"), "model-predictive control with limits and proportional costs as a QP with an approximate-dynamic-programming terminal reward"),
    Item(B1[4], "VWAP", "existing", ("py:src.algo.algos.VWAP",), "slices follow the expected volume profile"),
    Item(B1[4], "POV", "existing", ("py:src.algo.algos.POV",), "a fixed share of the volume"),
    Item(B1[4], "Trade-schedule optimisation", "existing", ("py:src.algo.optimize.optimal_schedule", "py:src.algo.optimize.frontier", "py:src.algo.optimize.exponential_trade"), "cost-risk frontier, exponential schedules"),
    Item(B1[4], "Trade-rate parameter", "existing", ("py:src.algo.optimize.trade_rate", "py:src.algo.optimize.fit_trade_rate"), "a constant fraction of the order per period, fitted to the objective"),
    Item(B1[4], "Limit-order models (limit/market mix)", "simulator", ("py:src.algo.limit_orders.solve", "py:src.algo.limit_orders.simulate"), "dynamic programme over the number of limit and market units per interval; fills are drawn from the assumed distribution, there is no order book"),
    Item(B1[4], "Smart order routing", "simulator", ("py:src.algo.routing.greedy_allocation", "py:src.algo.routing.kaplan_meier_tail", "py:src.algo.routing.simulate_routing"), "send to the venues most likely to fill, learning fill probabilities from censored fills; venues are simulated"),
    Item(B1[5], "Calendar timing: January effect, quarterly horizons", "built", ("model:calendar_factor_timing",), "premium forecast from earlier months in the same calendar state (January, month, month within the quarter, November-April)"),
    Item(B1[5], "Seasonal earnings-announcement timing", "data", ("model:earnings_season_premium",), "announcement-month premium; needs data/user/earnings_dates.csv, or infers announcements from volume spikes"),
    Item(B1[5], "Macro timing: Fed rates, M1, GDP, inflation, PPI, up and down markets", "built", ("model:macro_factor_timing",), "the same premium forecast conditioned on policy-rate changes, growth of M1, GDP, CPI or PPI against their median, or the market's trailing return"),
    # ------------------------------------------------------------------------------------------------------------------ Book 2
    Item(B2[1], "Modern portfolio theory, mean-variance, capital market line, tangency", "built", ("py:src.equity.theory.frontier_constants", "py:src.equity.theory.tangency_portfolio", "py:src.equity.theory.capital_market_line", "allocator:tangency_cml"), "frontier constants A, B, C, D, the tangency portfolio, the CML and its allocation"),
    Item(B2[1], "CAPM: standard and zero-beta", "built", ("py:src.equity.theory.capm_regression", "py:src.equity.theory.security_market_line", "py:src.equity.theory.zero_beta_portfolio"), "time-series betas, the security market line with Shanken's correction, with and without a riskless rate"),
    Item(B2[1], "Arbitrage pricing theory", "built", ("py:src.equity.theory.apt_two_pass", "py:src.equity.theory.apt_arbitrage_portfolio", "model:apt_alpha"), "two-pass estimation, statistical factors, the cheapest arbitrage portfolio"),
    Item(B2[1], "Risk parity: ERC and HRP", "existing", ("py:src.portfolio.risk_parity.risk_parity_weights", "py:src.portfolio.hierarchical.hrp_weights"), "see Book 1, 3"),
    Item(B2[1], "Mean-variance with CVaR", "built", ("py:src.equity.cvar_portfolio.mean_variance_cvar", "allocator:mv_cvar"), "mean-variance under a CVaR limit (Rockafellar-Uryasev), cutting planes with the scenario programme as the reference"),
    Item(B2[1], "Quantum annealing and QUBO portfolios", "simulator", ("py:src.equity.qubo.weight_qubo", "py:src.equity.qubo.cardinality_qubo", "py:src.equity.qubo.simulated_annealing", "allocator:qubo_select"), "the QUBO formulations are exact; the solver is classical simulated annealing, checked against exhaustive search. No quantum hardware"),
    Item(B2[2], "Black-Litterman", "existing", ("allocator:black_litterman",), "see Book 1, 3"),
    Item(B2[2], "Hierarchical Bayes", "built", ("py:src.portfolio.hierarchical_bayes.hierarchical_posterior", "allocator:hierarchical_bayes"), "expected returns shrunk toward their group and groups toward the whole, with the scales inferred on a grid"),
    Item(B2[2], "Bayesian decision theory", "built", ("py:src.portfolio.decision_theory.bayes_weights", "py:src.portfolio.decision_theory.regret", "py:src.portfolio.decision_theory.value_of_information", "allocator:bayes_expected_utility"), "weights maximising expected utility over the posterior predictive, regret and the value of information"),
    Item(B2[2], "Gaussian process regression", "built", ("py:src.models.gaussian_process.GaussianProcess", "model:gp_factor_model"), "exact GP with ARD kernels and marginal-likelihood hyperparameters, as a monthly return forecaster with its uncertainty"),
    Item(B2[3], "Statistical factors (PCA)", "built", ("py:src.equity.theory.statistical_factors", "model:apt_alpha"), "principal-component factors and the APT alpha built on them"),
    Item(B2[3], "Macroeconomic factor models", "data", ("model:macro_factor_model",), "returns on macro series declared on the model card; needs the macro series (FRED, point in time)"),
    Item(B2[3], "Cross-sectional factor models", "built", ("model:characteristic_regression",), "Fama-MacBeth style forecasts from price characteristics"),
    Item(B2[3], "LASSO, ridge and elastic net", "built", ("model:ml_factor_model",), "kind=ridge, lasso, enet, ols"),
    Item(B2[3], "PCR and PLS", "built", ("model:ml_factor_model",), "kind=pcr, pls"),
    Item(B2[3], "Random forests and CART", "built", ("model:ml_factor_model", "model:ml_trees"), "kind=cart, forest"),
    Item(B2[3], "Neural networks", "built", ("model:ml_factor_model", "model:deep_window"), "kind=mlp on characteristics; the window networks of Book 2, 5 on returns"),
    Item(B2[4], "Markov decision processes: fully observed, finite and infinite horizon", "built", ("py:src.control.mdp.MDP", "py:src.control.mdp.value_iteration", "py:src.control.mdp.policy_iteration", "py:src.control.mdp.backward_induction"), "exact solutions for small finite MDPs"),
    Item(B2[4], "Partially observed MDPs", "built", ("py:src.control.mdp.POMDP", "py:src.control.mdp.pbvi", "py:src.control.mdp.expectimax"), "belief updates, exact tree search and point-based value iteration, checked on the tiger problem"),
    Item(B2[4], "Risk-sensitive MDPs", "built", ("py:src.control.mdp.risk_sensitive_value_iteration",), "entropic risk"),
    Item(B2[4], "Optimal control: Bellman, HJB", "built", ("py:src.control.mdp.value_iteration", "py:src.control.hjb.merton_hjb"), "the Bellman equation in discrete time, the HJB equation by a monotone scheme with policy iteration"),
    Item(B2[4], "Pontryagin", "built", ("py:src.control.pontryagin.solve_pmp", "py:src.control.pontryagin.lqr_pmp", "py:src.control.pontryagin.ramsey_pmp"), "boundary value problem, checked against Riccati, the sinh schedule and the Ramsey steady state"),
    Item(B2[4], "Merton's problem", "built", ("py:src.control.merton.merton_fraction", "py:src.control.merton.CostDP", "py:src.control.hjb.merton_hjb"), "closed forms, the proportional-cost no-trade band by dynamic programming, and the HJB solution"),
    Item(B2[4], "Viscosity solutions", "built", ("py:src.control.hjb.american_put_hjb",), "an obstacle problem solved by a monotone scheme, converging to the binomial price"),
    Item(B2[4], "Schroedinger control", "built", ("py:src.control.schrodinger.markov_bridge", "py:src.control.schrodinger.sinkhorn", "py:src.control.schrodinger.solve_lmdp_first_exit"), "Schroedinger bridges by Sinkhorn and linearly solvable control"),
    Item(B2[4], "SARSA, expected SARSA", "built", ("py:src.rl.tabular.train",), "method=sarsa, expected_sarsa"),
    Item(B2[4], "REINFORCE", "built", ("py:src.rl.policy_gradient.reinforce",), "linear softmax policy with a baseline"),
    Item(B2[4], "PPO", "built", ("py:src.rl.deep.PPO",), "clipped surrogate with GAE (torch)"),
    Item(B2[4], "Q-learning, double Q-learning, DQN", "built", ("py:src.rl.tabular.train", "py:src.rl.deep.DQN"), "method=q_learning, double_q; DQN with double=True for double DQN (torch)"),
    Item(B2[4], "Deterministic policy gradient, DDPG", "built", ("py:src.rl.policy_gradient.deterministic_policy_gradient", "py:src.rl.deep.DDPG"), "compatible-critic DPG with linear features; DDPG in torch"),
    Item(B2[4], "A2C, SAC, TD3", "built", ("py:src.rl.policy_gradient.a2c", "py:src.rl.deep.SAC", "py:src.rl.deep.TD3"), "n-step actor-critic with linear features; SAC and TD3 in torch"),
    Item(B2[4], "G-learning and GIRL", "built", ("py:src.rl.glearning.g_learning", "py:src.rl.glearning.soft_value_iteration", "py:src.rl.glearning.girl"), "entropy-regularised RL relative to a prior policy, and maximum-likelihood inverse reinforcement learning"),
    Item(B2[4], "RL for exposure", "simulator", ("allocator:q_learning_exposure", "py:src.rl.exposure.fit_exposure_policy"), "fitted Q-iteration on the empirical transitions of a book's own history; not in the list, the use of the above on the platform's data"),
    Item(B2[5], "Feed-forward, convolutional, recurrent and Transformer networks", "built", ("model:deep_window", "py:src.models.deep_sequence.build_sequence_network"), "kind=fnn, cnn, lstm, gru, transformer (and the earlier patch transformer, mixer, N-BEATS, N-HiTS, TimeMixer-style, TFT) on the normalised 252-day window"),
    Item(B2[5], "GAN and WGAN", "built", ("py:src.models.generative.WGANGP", "py:src.scenarios.generators.wgan_gp"), "Wasserstein GAN with gradient penalty as a scenario generator"),
    Item(B2[5], "VAE and NeuralFactors", "interpretation", ("py:src.models.generative.FactorVAE", "py:src.scenarios.generators.factor_vae"), "a variational autoencoder with a K-factor decoder and Student-t noise, in the spirit of NeuralFactors; not the paper's model"),
    Item(B2[5], "Physics-informed networks: BSM, Vasicek, Heston, Bates", "built", ("py:src.models.pinn.BSMPINN", "py:src.models.pinn.VasicekPINN", "py:src.models.pinn.HestonPINN", "py:src.models.pinn.bates_price"), "PDE residual training checked against Black-Scholes, the affine bond price and Fourier prices; accuracy about 0.5% of the strike"),
    Item(B2[6], "Minimum spanning tree", "built", ("py:src.portfolio.graph_portfolio.mst",), "the MST of the correlation distance"),
    Item(B2[6], "TMFG", "built", ("py:src.portfolio.graph_portfolio.tmfg",), "triangulated maximally filtered graph"),
    Item(B2[6], "Degeneracy ordering", "built", ("py:src.portfolio.graph_portfolio.degeneracy_ordering",), "smallest-last ordering, core numbers, degeneracy"),
    Item(B2[6], "Clique centrality", "built", ("py:src.portfolio.graph_portfolio.centrality", "allocator:graph_centrality"), "centrality (degree, strength, eigenvector, clique) and peripheral or central portfolios"),
    Item(B2[6], "Hierarchical sensitivity parity", "interpretation", ("py:src.portfolio.graph_portfolio.hsp_weights", "allocator:hierarchical_sensitivity_parity"), "the published definition was not available: built as HRP with each split equalising the branches' risk contribution or factor sensitivity"),
    Item(B2[7], "Walk-forward backtesting", "existing", ("py:src.validation.walk_forward.run_walk_forward", "py:src.validation.walk_forward.WalkForwardSplitter"), "expanding or rolling folds with an embargo"),
    Item(B2[7], "Resampling: cross-validation and bootstrap", "built", ("py:src.validation.walk_forward.purged_kfold_indices", "py:src.validation.cpcv.cpcv_splits", "py:src.validation.cpcv.assemble_paths", "py:src.probability.resampling.bootstrap_statistic"), "purged k-fold, combinatorial purged cross-validation with its paths, block bootstraps"),
    Item(B2[7], "Monte Carlo", "existing", ("py:src.probability.montecarlo.mc_estimate", "py:src.scenarios.backtest.scenario_backtest"), "Monte Carlo estimation and importance sampling; a rule's results over generated scenarios"),
    Item(B2[7], "Generative models for scenarios", "built", ("py:src.scenarios.generators.generate",), "WGAN-GP, factor VAE and diffusion generators with a common interface"),
    Item(B2[7], "Historical scenarios", "built", ("py:src.scenarios.generators.historical",), "random windows of the history"),
    Item(B2[7], "Bootstrap scenarios", "built", ("py:src.scenarios.generators.bootstrap",), "iid, moving, circular or stationary block bootstrap of whole rows"),
    Item(B2[7], "Copula scenarios", "existing", ("py:src.scenarios.generators.copula", "py:src.probability.copulas.simulate_joint"), "fitted copula with empirical or EVT marginals"),
    Item(B2[7], "Risk-factor scenarios", "built", ("py:src.scenarios.generators.risk_factor",), "principal-component factors bootstrapped in blocks plus residuals"),
    Item(B2[7], "ARIMA-GARCH scenarios", "built", ("py:src.scenarios.generators.arima_garch",), "AR(1) mean and GJR-GARCH volatility with jointly resampled standardised residuals"),
    Item(B2[7], "GAN and VAE scenarios", "built", ("py:src.scenarios.generators.wgan_gp", "py:src.scenarios.generators.factor_vae", "py:src.scenarios.generators.diffusion"), "learned generators (torch)"),
    Item(B2[7], "Scenario quality", "built", ("py:src.scenarios.quality.quality_report", "py:src.scenarios.quality.compare"), "marginals, volatility, correlation, tails, tail dependence, volatility clustering, expected shortfall, horizon volatility"),
]


def resolve(where: str):
    """The Python object, registry entry or allocator that ``where`` names; raises if it does not exist."""
    kind, _, target = where.partition(":")
    if kind == "py":
        module, _, attr = target.rpartition(".")
        return getattr(importlib.import_module(module), attr)
    if kind == "model":
        from ..framework import MODELS, load_library

        load_library()
        return MODELS._entries[target].factory
    if kind == "allocator":
        from ..framework import ALLOCATORS, load_library

        load_library()
        return ALLOCATORS._entries[target].factory
    raise ValueError(f"unknown location '{where}'")


def table():
    import pandas as pd

    return pd.DataFrame([{"section": i.section, "item": i.item, "status": i.status, "where": ", ".join(w.split(":", 1)[1] for w in i.where), "note": i.note} for i in ITEMS])


INTRO = """# The two books: where everything is

This page maps the topics of two books, *Quantitative Equity Portfolio Management: Modern Techniques and Applications* (Book 1) and *Quantitative Portfolio Optimization: Advanced Techniques and Applications* (Book 2), item by item, to the code that implements them. It is generated from `src/books/catalog.py` (`python -m src.books.catalog`), and a test imports every location it names, so it cannot describe something that is not there.

**Status.** *built*: written for this item. *existing*: the repository already had it, and the entry points to it. *data*: built, and reads a file you supply (`data/user/fundamentals.csv`, `earnings_dates.csv`) or a macro series; it refuses to run without it instead of inventing a signal. *simulator*: it runs on a simulated or stylised market, or its solver stands in for the one named (simulated annealing for quantum annealing). *interpretation*: the item names a method whose definition was not available here, so what is built is the author's documented reading.

**Where to run things.** Models are in `quant backtest --model NAME` and the dashboard's strategy list; allocators are `--allocator NAME` (with `--alloc-param key=value`). The numerical modules (`src/control`, `src/rl`, `src/scenarios`, `src/equity`) are Python libraries with the experiments `experiments/control_world.py`, `deep_world.py`, `portfolio_world.py`, `trading_world.py`, `timing_world.py`, `bayes_world.py` and `equity_world.py`, which print the numbers the guides quote. Guides: [building an alpha model](techniques/alpha-model-construction.md), [fundamental factors](techniques/fundamental-factors.md), [portfolio theory and constrained books](techniques/portfolio-theory-and-constrained-books.md), [factor models and machine learning](techniques/factor-models-and-machine-learning.md), [trading with costs](techniques/multi-period-trading-and-order-routing.md), [factor timing](techniques/factor-timing.md), [hierarchical Bayes and Gaussian processes](techniques/hierarchical-bayes-decisions-and-gaussian-processes.md), [dynamic programming and optimal control](techniques/dynamic-programming-and-optimal-control.md), [reinforcement learning](techniques/reinforcement-learning-from-tables-to-networks.md), [deep networks, generative scenarios and PINNs](techniques/deep-learning-generative-scenarios-and-pinns.md), [graph portfolios](techniques/graph-portfolios-and-hierarchical-sensitivity-parity.md), [scenario generation and backtesting](techniques/scenario-generation-and-backtesting.md).
"""

LIMITS = """## What this does not do

- **Simulated worlds are plumbing checks, not market evidence.** Every new method was tested against a world whose truth was planted, and most were run on the platform's 15 ETFs; those runs are reported as they came out, and several of the new methods did not beat simple baselines (the guides say which).
- **Fundamentals need your file.** The value, quality, momentum-with-revisions, DCF and nonlinear equity models read `data/user/fundamentals.csv` and refuse to run without it; the earnings-season model reads `data/user/earnings_dates.csv` or, failing that, infers announcements from volume spikes. The repository has no fundamentals feed.
- **No quantum hardware.** The QUBO portfolio formulations are exact; the solver is simulated annealing, checked against exhaustive search.
- **Reinforcement learning, PINNs and generative models are small research implementations.** They run in minutes on a CPU, are checked against problems with known answers, and are accurate to a fraction of a percent of a strike or a few hundredths of a correlation, not to production tolerance.
- **Two items are interpretations.** The VAE follows the *idea* of NeuralFactors (a latent factor structure with heavy-tailed noise), not the paper's model; hierarchical sensitivity parity is a documented generalisation of HRP because the published definition was not available.
- **Execution models are simulators** (see [algorithmic trading](algorithmic_trading.md)): no order book, queue or latency.
"""


def markdown() -> str:
    lines = [INTRO]
    for section in dict.fromkeys(i.section for i in ITEMS):
        lines += [f"## {section}", "", "| Item | Status | Where | What it does |", "|---|---|---|---|"]
        for i in (x for x in ITEMS if x.section == section):
            places = ", ".join(f"`{w.split(':', 1)[1]}`" + ("" if w.startswith("py:") else f" ({w.split(':', 1)[0]})") for w in i.where)
            lines.append(f"| {i.item} | {i.status} | {places} | {i.note} |")
        lines.append("")
    lines.append(LIMITS)
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[2] / "docs" / "quant_books_coverage.md"
    out.write_text(markdown(), encoding="utf-8")
    print(f"wrote {out}")
