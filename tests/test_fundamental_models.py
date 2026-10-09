"""The fundamental-factor strategies on a world with statements and a planted truth: each style finds the premium that was planted, the file is read point-in-time, and nothing looks ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import alpha_model as am
from src.equity import fundamentals as fm
from src.equity.synthetic import simulate_fundamental_world
from src.framework import MODELS, Pipeline, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies import fundamental_factors as ff
from src.utils.config import load_config
from src.utils.dates import rebalance_dates

load_library()
PREMIA = {"value": 0.006, "quality": 0.005, "investment": -0.004, "momentum": 0.004, "reversal": 0.004, "revision": 0.004}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = simulate_fundamental_world(n_assets=60, n_years=12, seed=1, premia=PREMIA)
    path = tmp_path_factory.mktemp("fund") / "fundamentals.csv"
    w.table.to_csv(path, index=False)
    bundle = bundle_from_prices(w.prices, min_history=60, name="fundamental-world")
    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    return {"world": w, "path": str(path), "bundle": bundle, "grid": grid, "forward": px.shift(-1) / px - 1.0}


def monthly_ic(env, score, skip=36):
    ic = am.information_coefficients({"x": score.loc[env["grid"]]}, env["forward"], "spearman", 20)["x"].iloc[skip:]
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic))


def make(name, env, **params):
    return MODELS.create(name, path=env["path"], **params)


# ------------------------------------------------------------------------------------------------------------------ registration and the file
def test_the_models_are_registered_documented_and_declare_what_they_are():
    names = ["fundamental_value", "fundamental_quality", "fundamental_momentum", "fundamental_dcf", "fundamental_alpha", "fundamental_nonlinear"]
    for n in names:
        e = next(e for e in MODELS.entries() if e.name == n)
        assert e.family == "fundamental" and len(e.description) > 40 and (e.factory.__doc__ or "").strip()
        assert e.factory.position_mode == "cross_sectional" and e.factory.required_assets(MODELS.create(n)) >= 2


def test_a_model_that_needs_the_file_refuses_without_it_and_the_price_factors_do_not(world, tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "USER_DATA", tmp_path / "nowhere")
    monkeypatch.setattr(ff, "USER_DATA", tmp_path / "nowhere")
    bundle = world["bundle"]
    for name in ("fundamental_value", "fundamental_quality", "fundamental_alpha"):
        with pytest.raises(KeyError, match="need the file"):
            MODELS.create(name).score(bundle)
    assert MODELS.create("fundamental_momentum").score(bundle).iloc[-1].notna().sum() > 40                 # the composite of the three price factors needs nothing else
    assert MODELS.create("fundamental_momentum", factor="ret9").score(bundle).iloc[-1].notna().sum() > 40
    with pytest.raises(KeyError, match="need the file"):
        MODELS.create("fundamental_momentum", factor="earn_rev9").score(bundle)


def test_a_file_without_the_columns_a_factor_needs_says_which(world, tmp_path):
    table = world["world"].table.drop(columns=["book_equity", "net_income"])
    p = tmp_path / "thin.csv"
    table.to_csv(p, index=False)
    with pytest.raises(KeyError, match="book_equity"):
        MODELS.create("fundamental_value", factor="b2p", path=str(p)).score(world["bundle"])
    composite = MODELS.create("fundamental_value", path=str(p)).score(world["bundle"])                     # the composite uses what it can: no b2p or earnings yield, six others
    assert composite.iloc[-1].notna().sum() > 40
    thin = table.drop(columns=[c for c in table.columns if c not in ("ticker", "period_end", "available", "sales")])
    thin.to_csv(p, index=False)
    with pytest.raises(KeyError, match="nothing can be computed"):
        MODELS.create("fundamental_value", path=str(p)).score(world["bundle"])


def test_the_statements_file_is_read_again_only_when_it_changes(world, tmp_path):
    p = tmp_path / "f.csv"
    world["world"].table.to_csv(p, index=False)
    a = ff.load_fundamentals(str(p), 60, 400)
    assert ff.load_fundamentals(str(p), 60, 400) is a and ff.load_fundamentals(str(p), 30, 400) is not a
    world["world"].table.head(100).to_csv(p, index=False)
    assert ff.load_fundamentals(str(p), 30, 400) is not a


def test_parameters_are_checked_when_the_model_is_made(world):
    for name, params in [("fundamental_value", {"lag_days": -1}), ("fundamental_value", {"expiry": 0}), ("fundamental_dcf", {"factor": "b2p"}), ("fundamental_dcf", {"paths": 5}),
                         ("fundamental_dcf", {"terminal_growth": 0.2}), ("fundamental_alpha", {"weights": "magic"}), ("fundamental_alpha", {"orthogonalize": "qr"}),
                         ("fundamental_alpha", {"context": "beauty"}), ("fundamental_alpha", {"window": 3}), ("fundamental_nonlinear", {"terms": "cubic:a"}), ("fundamental_nonlinear", {"window": 2})]:
        with pytest.raises(ValueError):
            MODELS.create(name, **params)
    with pytest.raises(ValueError, match="factor must be"):
        make("fundamental_value", world, factor="rnoa").score(world["bundle"])
    with pytest.raises(ValueError, match="unknown factor or style"):
        make("fundamental_alpha", world, factors="b2p,magic").score(world["bundle"])
    with pytest.raises(ValueError, match="refers to"):
        make("fundamental_nonlinear", world, factors="b2p", terms="quadratic:rnoa").score(world["bundle"])


# ------------------------------------------------------------------------------------------------------------------ the planted premia
def test_the_styles_find_the_premium_that_was_planted(world):
    env = world
    # (model, parameters, smallest mean IC, smallest t-statistic): about half of what this world gives, which was measured, not hoped for
    for name, params, floor, t_floor in [("fundamental_value", {"factor": "b2p"}, 0.015, 2.0), ("fundamental_value", {}, 0.05, 4.0), ("fundamental_quality", {"factor": "rnoa"}, 0.025, 2.5),
                                         ("fundamental_quality", {}, 0.01, 1.5), ("fundamental_momentum", {}, 0.06, 5.0), ("fundamental_momentum", {"factor": "ret1"}, 0.0, 1.5),
                                         ("fundamental_momentum", {"factor": "earn_rev9"}, 0.05, 4.0), ("fundamental_dcf", {"factor": "dcf_upside"}, 0.04, 3.0),
                                         ("fundamental_dcf", {"factor": "mdcf_upside", "paths": 100}, 0.04, 3.0)]:
        ic, t = monthly_ic(env, make(name, env, **params).score(env["bundle"]))
        assert ic > floor and t > t_floor, (name, params, ic, t)                                          # ret1 is oriented: the model buys last month's losers, which is what was planted


def test_orientation_turns_a_factor_whose_high_end_is_bad_and_can_be_switched_off(world):
    raw = make("fundamental_momentum", world, factor="ret1", oriented=False).score(world["bundle"])
    oriented = make("fundamental_momentum", world, factor="ret1").score(world["bundle"])
    assert np.allclose(raw.fillna(0).to_numpy(), -oriented.fillna(0).to_numpy())
    assert monthly_ic(world, oriented)[0] > 0 > monthly_ic(world, raw)[0]


def test_the_alpha_model_beats_each_style_and_a_fixed_blend(world):
    env = world
    styles = {s: monthly_ic(env, make(f"fundamental_{s}", env).score(env["bundle"]))[0] for s in ("value", "quality", "momentum")}
    optimal = make("fundamental_alpha", env).score(env["bundle"])
    equal = make("fundamental_alpha", env, weights="equal").score(env["bundle"])
    ic_opt, t_opt = monthly_ic(env, optimal)
    ic_eq = monthly_ic(env, equal)[0]
    assert ic_opt > max(styles.values()) + 0.005 and t_opt > 6                                            # weighing the styles finds more than the best of them
    assert ic_opt > ic_eq - 0.005 and ic_eq > min(styles.values())                                        # and is not worse than a fixed blend of them
    assert optimal.iloc[-1].notna().sum() > 40


def test_the_alpha_model_runs_in_every_orthogonal_and_contextual_form(world):
    env = world
    base = monthly_ic(env, make("fundamental_alpha", env, factors="value,quality,momentum,estimates").score(env["bundle"]))[0]
    for params in ({"orthogonalize": "none"}, {"orthogonalize": "symmetric"}, {"nonnegative": True}, {"context": "value"}, {"context": "growth"}, {"context": "earnings_variability", "weights": "optimal"},
                   {"weights": "equal", "orthogonalize": "symmetric"}, {"factors": "b2p,rnoa,ret9,earn_rev9"}):
        score = make("fundamental_alpha", env, **params).score(env["bundle"])
        ic, t = monthly_ic(env, score)
        assert ic > 0.06 and t > 4, (params, ic, t)
    assert base > 0.07


def test_a_quadratic_term_finds_the_inverted_u_in_capital_expenditure_that_was_planted(world):
    env = world
    plain = make("fundamental_nonlinear", env, terms="").score(env["bundle"])
    curved = make("fundamental_nonlinear", env).score(env["bundle"])
    mixed = make("fundamental_nonlinear", env, factors="value,quality,icapx", terms="quadratic:icapx,interaction:value:quality,conditional:icapx:value:high").score(env["bundle"])
    assert monthly_ic(env, curved)[0] > monthly_ic(env, plain)[0] + 0.004                                  # the square of capital expenditure adds to a model that has only the straight lines
    assert monthly_ic(env, mixed)[0] > 0.05 and curved.iloc[-1].notna().sum() > 40


# ------------------------------------------------------------------------------------------------------------------ no look-ahead
@pytest.mark.parametrize("name,params", [("fundamental_value", {}), ("fundamental_alpha", {}), ("fundamental_nonlinear", {}), ("fundamental_dcf", {"factor": "dcf_upside"}),
                                         ("fundamental_momentum", {})])
def test_every_model_is_causal(world, name, params):
    bundle = world["bundle"]
    check = check_causality(make(name, world, **params), bundle, cutoff=bundle.index[int(0.7 * len(bundle.index))])
    assert check["ok"], (name, check)


def test_a_filing_published_after_a_date_cannot_change_the_score_on_that_date(world, tmp_path):
    cut = world["bundle"].index[1800]
    table = world["world"].table.copy()
    later = pd.to_datetime(table["available"]) > cut
    mangled = table.copy()
    numeric = [c for c in mangled.columns if c in fm.FIELDS]
    mangled.loc[later, numeric] = mangled.loc[later, numeric] * 37.0
    p = tmp_path / "mangled.csv"
    mangled.to_csv(p, index=False)
    for name, params in (("fundamental_value", {}), ("fundamental_quality", {}), ("fundamental_dcf", {"factor": "dcf_upside"}), ("fundamental_alpha", {"factors": "value,quality"})):
        a = make(name, world, **params).score(world["bundle"]).loc[:cut]
        b = MODELS.create(name, path=str(p), **params).score(world["bundle"]).loc[:cut]
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), name


# ------------------------------------------------------------------------------------------------------------------ the pipeline
def test_the_pipeline_runs_fundamental_models_alone_and_combined_by_the_ic_covariance_rule(world):
    config, bundle = load_config(), world["bundle"]
    alone = Pipeline({"name": "fa", "models": [{"name": "fundamental_alpha", "params": {"path": world["path"]}}]}, config, bundle).run()
    assert np.isfinite(alone.metrics["sharpe"]) and alone.validation["causality"]["fundamental_alpha"]["ok"]
    spec = {"name": "fv", "models": [{"name": "fundamental_value", "params": {"path": world["path"]}}, {"name": "fundamental_momentum", "params": {"path": world["path"]}}],
            "combination": {"rule": "optimal_ic", "min_obs": 126, "window": 756}}
    both = Pipeline(spec, config, bundle).run()
    assert np.isfinite(both.metrics["sharpe"]) and abs(both.tables["ic_combination"]["mean_weight"].sum() - 1) < 1e-9
    assert both.metrics["sharpe"] > 0.3                                                                # a world with planted value and momentum premia: the strategy should find them
