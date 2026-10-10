"""The coverage catalogue of the two books is true (every location imports), complete against the topics as they were requested, and the page is the catalogue written out."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.books import catalog

# Each topic as the request listed it, mapped to the catalogue item that covers it.
REQUESTED = {
    "Book 1, 1. Alpha generation": {
        "value factors": "Value factors", "quality factors": "Quality factors", "momentum factors": "Momentum factors", "DCF": "Discounted cash flow", "multipath DCF": "Multipath DCF",
        "contextual models": "Contextual models", "nonlinear models": "Nonlinear models"},
    "Book 1, 2. Alpha model construction": {
        "optimal IC-based model": "Optimal (IC-based) alpha model", "z-scores": "Z-scores", "Gram-Schmidt": "Gram-Schmidt orthogonalisation", "Fama-MacBeth marginal contributions": "Fama-MacBeth marginal contributions"},
    "Book 1, 3. Portfolio construction": {
        "MVO": "Mean-variance optimisation", "Black-Litterman": "Black-Litterman", "ERC risk parity": "Equal risk contribution (risk parity)", "HRP": "Hierarchical risk parity",
        "130/30": "130/30 and market-neutral books", "market-neutral": "130/30 and market-neutral books", "sector neutral": "Sector, beta and dollar neutrality", "beta neutral": "Sector, beta and dollar neutrality",
        "dollar neutral": "Sector, beta and dollar neutrality", "turnover": "Turnover limits", "long-only": "Long-only"},
    "Book 1, 4. Trading and implementation": {
        "Almgren-Chriss": "Almgren-Chriss", "Mei et al.": "Multi-period optimisation (Mei et al.)", "Skaf and Boyd 2009": "Constrained multi-period (Skaf and Boyd)", "VWAP": "VWAP", "POV": "POV",
        "trade schedule optimisation": "Trade-schedule optimisation", "trade-rate parameter": "Trade-rate parameter", "limit-order models": "Limit-order models (limit/market mix)",
        "smart order routing": "Smart order routing"},
    "Book 1, 5. Factor timing": {
        "January effect and quarterly horizons": "Calendar timing: January effect, quarterly horizons", "seasonal earnings announcements": "Seasonal earnings-announcement timing",
        "macro: Fed, M1, GDP, inflation, PPI, up/down markets": "Macro timing: Fed rates, M1, GDP, inflation, PPI, up and down markets"},
    "Book 2, 1. Portfolio theory and optimisation": {
        "MPT, MVO, CML, tangency": "Modern portfolio theory, mean-variance, capital market line, tangency", "CAPM zero-beta and standard": "CAPM: standard and zero-beta", "APT": "Arbitrage pricing theory",
        "risk parity ERC/HRP": "Risk parity: ERC and HRP", "mean-variance with CVaR": "Mean-variance with CVaR", "quantum annealing/QUBO": "Quantum annealing and QUBO portfolios"},
    "Book 2, 2. Bayesian and probabilistic methods": {
        "Black-Litterman": "Black-Litterman", "hierarchical Bayes": "Hierarchical Bayes", "Bayesian decision theory": "Bayesian decision theory", "Gaussian process regression": "Gaussian process regression"},
    "Book 2, 3. Factor investing and machine learning": {
        "PCA statistical factors": "Statistical factors (PCA)", "macro factor models": "Macroeconomic factor models", "cross-sectional models": "Cross-sectional factor models",
        "LASSO, ridge, elastic net": "LASSO, ridge and elastic net", "PCR, PLS": "PCR and PLS", "random forests, CART": "Random forests and CART", "neural nets": "Neural networks"},
    "Book 2, 4. Dynamic programming and reinforcement learning": {
        "MDPs fully observed, finite and infinite horizon": "Markov decision processes: fully observed, finite and infinite horizon", "POMDP": "Partially observed MDPs", "risk-sensitive MDP": "Risk-sensitive MDPs",
        "Bellman and HJB": "Optimal control: Bellman, HJB", "Pontryagin": "Pontryagin", "Merton": "Merton's problem", "viscosity solutions": "Viscosity solutions", "Schroedinger control": "Schroedinger control",
        "SARSA": "SARSA, expected SARSA", "REINFORCE": "REINFORCE", "PPO": "PPO", "Q-learning, DQN, double Q": "Q-learning, double Q-learning, DQN", "DPG, DDPG": "Deterministic policy gradient, DDPG",
        "A2C, SAC, TD3": "A2C, SAC, TD3", "G-learning, GIRL": "G-learning and GIRL"},
    "Book 2, 5. Deep learning": {
        "FNN, CNN, RNN/LSTM, Transformer": "Feed-forward, convolutional, recurrent and Transformer networks", "GAN/WGAN": "GAN and WGAN", "VAE/NeuralFactors": "VAE and NeuralFactors",
        "PINNs: BSM, Vasicek, Heston, Bates": "Physics-informed networks: BSM, Vasicek, Heston, Bates"},
    "Book 2, 6. Graph-based portfolios": {
        "MST": "Minimum spanning tree", "TMFG": "TMFG", "degeneracy ordering": "Degeneracy ordering", "clique centrality": "Clique centrality", "Hierarchical Sensitivity Parity": "Hierarchical sensitivity parity"},
    "Book 2, 7. Backtesting and scenario generation": {
        "walk-forward": "Walk-forward backtesting", "resampling (CV, bootstrap)": "Resampling: cross-validation and bootstrap", "Monte Carlo": "Monte Carlo", "generative models": "Generative models for scenarios",
        "historical scenarios": "Historical scenarios", "bootstrap scenarios": "Bootstrap scenarios", "copula scenarios": "Copula scenarios", "risk-factor scenarios": "Risk-factor scenarios",
        "ARIMA-GARCH scenarios": "ARIMA-GARCH scenarios", "GAN/VAE scenarios": "GAN and VAE scenarios"},
}


def test_every_requested_topic_is_in_the_catalogue_and_nothing_is_listed_twice():
    have = {(i.section, i.item) for i in catalog.ITEMS}
    assert len(have) == len(catalog.ITEMS)
    for section, topics in REQUESTED.items():
        for topic, item in topics.items():
            assert (section, item) in have, (topic, item)
    assert {i.status for i in catalog.ITEMS} <= {"built", "existing", "data", "simulator", "interpretation"}
    assert all(i.where and i.note for i in catalog.ITEMS)
    assert {i.section for i in catalog.ITEMS} == set(REQUESTED)


def test_every_location_the_catalogue_names_really_exists():
    for item in catalog.ITEMS:
        for where in item.where:
            assert catalog.resolve(where) is not None, (item.item, where)
    with pytest.raises(ValueError):
        catalog.resolve("nowhere:thing")
    with pytest.raises(KeyError):
        catalog.resolve("model:no_such_model")
    with pytest.raises(AttributeError):
        catalog.resolve("py:src.control.mdp.NoSuchThing")


def test_the_coverage_page_is_the_catalogue_written_out():
    page = (Path(__file__).resolve().parents[1] / "docs" / "quant_books_coverage.md").read_text(encoding="utf-8")
    assert page == catalog.markdown(), "docs/quant_books_coverage.md is stale: run `python -m src.books.catalog`"
    for item in catalog.ITEMS:
        assert f"| {item.item} |" in page
    assert len(catalog.table()) == len(catalog.ITEMS)


def test_the_items_that_need_data_name_models_that_refuse_to_run_without_it():
    import numpy as np
    import pandas as pd

    from src.framework import MODELS, bundle_from_prices, load_library

    load_library()
    prices = pd.DataFrame(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (800, 6)), axis=0)), index=pd.bdate_range("2018-01-01", periods=800), columns=list("ABCDEF"))
    bundle = bundle_from_prices(prices, min_history=60, name="x")
    for name in ("fundamental_value", "fundamental_quality", "fundamental_dcf", "fundamental_nonlinear"):
        with pytest.raises(Exception, match="needs the file|fundamentals"):
            MODELS.create(name).score(bundle)
    for item in catalog.ITEMS:
        if item.status == "data":
            assert "needs" in item.note or "file" in item.note or "data" in item.note, item.item


def test_the_interpretations_say_so():
    for item in catalog.ITEMS:
        if item.status == "interpretation":
            assert "not" in item.note or "spirit" in item.note, item.item
