"""How many tickers a strategy needs: the declarations, the combined check, and (the part that keeps the declarations honest) what every shipped strategy actually does on one ticker."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.framework import ALLOCATORS, MODELS, Pipeline, PipelineSpec, bundle_from_prices, load_library
from src.framework.requirements import minimum_assets, universe_problem
from src.utils.config import load_config

load_library()
SLOW = {"deep_window", "chronos", "timesfm", "es_policy", "deep_representation"}


def model(name, **params):
    return MODELS.create(name, **params)


# ------------------------------------------------------------------------------------------------------------------- the declarations
@pytest.mark.parametrize("name,params,need", [
    ("tsmom", {}, 1), ("ma_crossover", {}, 1), ("rsi2", {}, 1), ("faber_gtaa", {}, 1), ("model_portfolio", {}, 1), ("dual_momentum", {}, 1),
    ("momentum", {}, 2), ("relative_strength", {}, 2), ("low_volatility", {}, 2), ("ml_ridge", {}, 2),
    ("kalman_pairs", {}, 2), ("cointegration_pairs", {}, 2), ("paa", {}, 2), ("vaa", {}, 2), ("daa", {}, 2), ("adaptive_asset_allocation", {}, 2), ("credit_spread_timing", {}, 2),
    ("sparse_basket", {}, 3), ("bvar_lead_lag", {}, 3), ("bab", {}, 4), ("bab", {"min_assets": 6}, 6), ("pca_residual", {}, 4), ("pca_residual", {"components": 5}, 6),
    ("expression", {"expr": "mom(60)", "mode": "cross_sectional"}, 2), ("expression", {"expr": "where(close > sma(50), 1, -1)", "mode": "time_series"}, 1),
])
def test_each_strategy_declares_how_many_tickers_it_needs(name, params, need):
    assert model(name, **params).required_assets() == need


def test_a_ranked_strategy_needs_two_even_if_it_declares_less_and_a_declared_minimum_wins_over_the_mode():
    from src.framework.forecasting import ForecastModel

    class Ranked(ForecastModel):
        name = "ranked"

    class PerAsset(ForecastModel):
        name, position_mode = "per_asset", "time_series"

    class Basket(PerAsset):
        min_assets = 5

    assert Ranked().required_assets() == 2 and PerAsset().required_assets() == 1 and Basket().required_assets() == 5


@pytest.mark.parametrize("name,params,need", [
    ("static", {"book": "equal_weight"}, 1), ("static", {"book": "inverse_vol"}, 1), ("static", {"book": "risk_parity"}, 2), ("static", {"book": "hrp"}, 2), ("static", {"book": "mvo"}, 2),
    ("forecast_stack", {"mode": "cross_sectional"}, 2), ("forecast_stack", {"mode": "time_series"}, 1), ("score_stack", {"mode": "cross_sectional"}, 2), ("score_stack", {"mode": "time_series"}, 1),
    ("sleeves", {}, 1), ("confidence", {}, 1), ("min_variance", {}, 3), ("max_diversification", {}, 3), ("kelly", {}, 3), ("black_litterman", {}, 3), ("dynamic_cov", {}, 3),
    ("bayesian", {}, 2), ("es_policy", {}, 2), ("min_variance", {"min_assets": 5}, 5),
])
def test_each_allocator_declares_how_many_tickers_it_needs(name, params, need):
    assert ALLOCATORS.create(name, **params).required_assets() == need


def test_every_registered_allocator_answers():
    for entry in ALLOCATORS.entries():
        if entry.name in ("regime_switch",):
            continue
        try:
            allocator = ALLOCATORS.create(entry.name)
        except TypeError:
            continue
        assert allocator.required_assets() >= 1


# ----------------------------------------------------------------------------------------------------------------------- the combined check
def test_the_combined_requirement_follows_the_allocator_the_pipeline_would_choose():
    assert minimum_assets([model("tsmom")], {"allocator": "sleeves"}) == (1, [])
    assert minimum_assets([model("tsmom")], None) == (1, [])                                       # automatic: the forecast stack in the strategy's own (per-asset) mode
    assert minimum_assets([model("momentum")], None)[0] == 2
    assert minimum_assets([model("tsmom")], {"allocator": "forecast_stack", "params": {"mode": "cross_sectional"}})[0] == 2
    assert minimum_assets([model("faber_gtaa")], None)[0] == 1                                     # a structured model supplies its own weights
    assert minimum_assets([model("tsmom")], {"allocator": "min_variance"})[0] == 3
    assert minimum_assets([model("tsmom"), model("bab")], None)[0] == 4
    need, reasons = minimum_assets([model("momentum")], None)
    assert need == 2 and reasons and "ranks the tickers" in reasons[0]


def test_the_message_says_what_is_wrong_and_what_to_do_about_it():
    one = universe_problem([model("momentum")], None, 1)
    assert "'momentum' ranks the tickers against each other" in one and "at least 2 tickers and you have 1" in one and "tsmom" in one
    allocation = universe_problem([model("tsmom")], {"allocator": "score_stack", "params": {"mode": "cross_sectional"}}, 1)
    assert "this allocation scores the tickers against each other" in allocation and "Independent sleeves" in allocation
    assert "Add 1 more ticker." in universe_problem([model("bab")], None, 3)
    assert "Add 2 more tickers." in universe_problem([model("bab")], None, 2)
    assert universe_problem([model("momentum")], None, 2) is None and universe_problem([model("tsmom")], {"allocator": "sleeves"}, 1) is None
    assert universe_problem([model("momentum")], None, 15) is None


def test_the_suggested_one_ticker_strategies_really_work_on_one_ticker():
    from src.framework.requirements import ONE_TICKER_EXAMPLES

    for name in ONE_TICKER_EXAMPLES:
        assert name in MODELS and model(name).required_assets() == 1


# ----------------------------------------------------------------------------------------------- the declarations match what the strategies do
N = 2000
IDX = pd.bdate_range("2012-01-02", periods=N)


def _one_ticker_bundle():
    rng = np.random.default_rng(5)
    drift = np.where(np.sin(np.arange(N) / 150.0) > 0, 0.0008, -0.0006)                       # trends that reverse, so trend rules and reversion rules both have something to trade
    prices = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01, N))
    macro = pd.DataFrame({"VIX": 20 + 6 * np.sin(np.arange(N) / 40.0) + rng.normal(0, 2, N)}, index=IDX)
    return bundle_from_prices(pd.DataFrame({"ONE": prices}, index=IDX), asset_class={"ONE": "equity"}, macro=macro, name="one")


def _held_days(name, bundle):
    cls = MODELS._entries[name].factory
    spec = {"name": name, "models": [{"name": name}], "execution": {"min_assets": 1}, "evaluation": {"benchmarks": [], "causality": False}}
    if getattr(cls, "book", None) == "sleeves":
        spec["allocation"] = {"allocator": "sleeves"}
    result = Pipeline(PipelineSpec.from_dict(spec), load_config(), bundle).run(validate=False)
    weights = result.weights.loc[result.start:] if result.start is not None else result.weights
    return int((weights.abs().sum(axis=1) > 1e-9).sum())


def test_strategies_that_declare_one_ticker_take_positions_on_one_ticker_and_the_others_cannot():
    """The honesty check: a strategy declared as needing one ticker must be able to trade on one (with the allocation the dashboard picks for it), and one declared as needing more must
    not pretend to (an empty book, or an error). Strategies that need macro series this synthetic bundle lacks, or a particular asset class, are skipped."""
    bundle = _one_ticker_bundle()
    wrong_one, wrong_many, checked = [], [], 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for entry in MODELS.entries():
            if entry.family == "crypto" or entry.name in SLOW or not getattr(entry.factory, "__module__", "").startswith("src.strategies") or entry.name == "commodity_supercycle":
                continue
            need = model(entry.name).required_assets()
            try:
                held = _held_days(entry.name, bundle)
            except KeyError as error:
                if "macro series" in str(error) or "needs one of" in str(error) or "volume" in str(error) or "asset of class" in str(error):
                    continue
                raise
            except ValueError:
                held = 0                                                                         # a rule that cannot even fit on one ticker
            checked += 1
            if need == 1 and held < 20:
                wrong_one.append((entry.name, held))
            if need > 1 and held >= 20:
                wrong_many.append((entry.name, held))
    assert checked >= 60
    assert not wrong_one, f"declared as one-ticker strategies but held (almost) nothing on one ticker: {wrong_one}"
    assert not wrong_many, f"declared as needing several tickers but traded one: {wrong_many}"


def test_the_strategies_the_dashboard_guide_names_as_buying_and_selling_exist_and_run_on_one_ticker():
    """The guide tells the reader which library strategies enter and leave positions on their own signal and run on one ticker; keep that list true."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "dashboard.md").read_text(encoding="utf-8")
    paragraph = text.split("**Strategies that buy and sell, not just hold.**", 1)[1].split("\n\n", 1)[0]
    named = re.findall(r"`([a-z0-9_]+)`", paragraph)
    assert len(named) >= 20
    for name in named:
        assert name in MODELS, name
        assert model(name).required_assets() == 1, name
