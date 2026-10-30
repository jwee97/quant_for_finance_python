"""Phase 1 models: rankers, quantile and NGBoost distributions, the HMM and Markov-switching regime models. Fitting, prediction, intervals, serialization and look-ahead."""

import dataclasses
import warnings

import numpy as np
import pytest

from src.ml import config as mc, datasets as dsm, evaluation as ev, models as md, training as tr
from src.ml.boosting import query_order, relevance_buckets
from src.ml.targets import forward_return
from src.utils.optional import unavailable_reason
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
SPLIT = {"min_train_days": 400, "refit_every": 200}


def _need(*packages):
    for pkg in packages:
        reason = unavailable_reason(pkg)
        if reason:
            pytest.skip(f"{pkg} is not available here: {reason}")


@pytest.fixture(scope="module")
def cfg():
    return mc.experiment_config({"dataset": {"feature_groups": FEATURES}, "split": SPLIT}, defaults=mc.DEFAULTS)


@pytest.fixture(scope="module")
def ds(cfg):
    return dsm.build_dataset(make_bundle(10, 1300, seed=21, alpha=1.0), cfg["dataset"], "phase1")


def _train(ds, cfg, kind, params=None):
    return tr.walk_forward_train(ds, {"kind": kind, "params": params or {}}, cfg["split"], 0)


def _ic(ds, res):
    fwd = forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
    return ev.rank_ic_wide(res.scores_wide(ds), fwd).dropna().mean()


def test_the_new_kinds_are_in_the_catalogue_with_their_settings_and_why_one_is_missing():
    cat = {m["kind"]: m for m in md.model_catalog()}
    for kind in ("lgbm_ranker", "xgb_ranker", "lgbm_quantile", "ngboost", "hmm_regime", "markov_switching"):
        assert kind in cat and cat[kind]["params"] and "available" in cat[kind]
    assert cat["xgb_ranker"]["specs"]["objective"]["choices"][0] == "rank:ndcg" and cat["lgbm_quantile"]["outputs"] == ["distribution"]
    assert md.search_space("lgbm_ranker") and md.search_space("hmm_regime")
    with pytest.raises(ValueError, match="no parameter"):
        md.make_model("lgbm_ranker", {"bogus": 1})
    with pytest.raises(ValueError, match="alpha_low"):
        md.make_model("lgbm_quantile", {"alpha_low": 0.7})


def test_relevance_buckets_order_the_assets_within_each_date_and_queries_group_by_date():
    keys = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([0.1, -0.2, 0.3, 0.0, 5.0, 4.0, 6.0, 3.0])
    rel = relevance_buckets(y, keys, 4)
    assert list(rel) == [2, 0, 3, 1, 2, 1, 3, 0]
    order, counts = query_order(np.array([2, 1, 2, 1, 1]))
    assert list(counts) == [3, 2] and list(np.array([2, 1, 2, 1, 1])[order]) == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("kind, params, packages, floor", [("lgbm_ranker", {"n_estimators": 40}, ("lightgbm",), 0.04), ("xgb_ranker", {"n_estimators": 40}, ("xgboost",), 0.04),
                                                           ("xgb_ranker", {"n_estimators": 40, "objective": "rank:pairwise"}, ("xgboost",), 0.0)])
def test_a_ranker_finds_the_planted_order_and_a_rows_score_does_not_depend_on_the_rows_scored_with_it(ds, cfg, kind, params, packages, floor):
    _need(*packages)
    res = _train(ds, cfg, kind, params)
    assert _ic(ds, res) > floor
    assert abs(res.predictions.mean()) < 0.5 and 0.3 < res.predictions.std() < 3.0                                       # standardised with the training scores, so on the scale of 1
    model = res.models[0]
    again = md.Model.from_bytes(model.to_bytes())
    X = ds.X.iloc[:300]
    assert np.allclose(model.predict(X), again.predict(X)) and set(model.importance().index) == set(ds.features)
    assert np.allclose(model.predict(X.iloc[50:120]), model.predict(X)[50:120])                                       # a row alone scores the same as in company


def test_a_ranker_takes_a_classification_target_and_refuses_too_few_dates(ds, cfg):
    _need("lightgbm")
    m = md.make_model("lgbm_ranker", {"n_estimators": 20}, "classification")
    y = (ds.Y[ds.target] > 0).astype(float)
    ok = y.notna()
    m.fit(ds.X[ok].iloc[:4000], y[ok].iloc[:4000], ds.positions()[ok.to_numpy()][:4000])
    assert len(m.predict(ds.X.iloc[:50])) == 50
    with pytest.raises(ValueError, match="too few"):
        md.make_model("lgbm_ranker").fit(ds.X.iloc[:20], ds.Y[ds.target].iloc[:20])


def test_quantile_lightgbm_gives_an_ordered_interval_that_covers_about_its_level(ds, cfg):
    _need("lightgbm")
    res = _train(ds, cfg, "lgbm_quantile", {"n_estimators": 60})
    d = res.extras["distribution"]
    assert (d["lower"] <= d["mean"] + 1e-12).all() and (d["mean"] <= d["upper"] + 1e-12).all() and (d["std"] > 0).all()
    y = ds.Y[ds.target].reindex(d.index)
    ok = y.notna()
    cover = float(((y[ok] >= d.loc[ok, "lower"]) & (y[ok] <= d.loc[ok, "upper"])).mean())
    assert 0.55 < cover < 0.97                                                                # nominal 0.8; a noisy panel and a short fit leave room
    assert res.diagnostics[0]["interval_level"] == pytest.approx(0.8) and _ic(ds, res) > 0.04


def test_ngboost_gives_a_normal_distribution_per_row(ds, cfg):
    _need("ngboost")
    res = _train(ds, cfg, "ngboost", {"n_estimators": 25, "max_rows": 6000})
    d = res.extras["distribution"]
    assert (d["std"] > 0).all() and ((d["upper"] - d["lower"]) > 0).all() and np.allclose(d["mean"], res.predictions.reindex(d.index))
    z = 1.2815515655446004
    assert np.allclose((d["upper"] - d["mean"]) / d["std"], z, atol=1e-6)
    again = md.Model.from_bytes(res.models[0].to_bytes())
    assert np.allclose(again.predict(ds.X.iloc[:200]), res.models[0].predict(ds.X.iloc[:200]))


@pytest.mark.parametrize("kind, params", [("hmm_regime", {"n_states": 3}), ("markov_switching", {"n_states": 2})])
def test_regime_models_report_filtered_probabilities_and_the_states_of_the_market(ds, cfg, kind, params):
    res = _train(ds, cfg, kind, params)
    reg = res.extras["regimes"]
    assert reg.shape[1] == params["n_states"] and np.allclose(reg.sum(axis=1), 1.0) and (reg >= 0).all().all()
    diag = res.diagnostics[0]
    assert diag["engine"] == ("hmm" if kind == "hmm_regime" else "markov_switching") and len(diag["regimes"]) == params["n_states"]
    assert np.allclose(np.array(diag["transition"]).sum(axis=1), 1.0)
    vols = [r["ann_vol"] for r in diag["regimes"] if r["ann_vol"] is not None]
    assert vols == sorted(vols) or kind == "markov_switching"                              # the HMM numbers its states by volatility
    again = md.Model.from_bytes(res.models[0].to_bytes()).bind(ds)
    X = ds.X.iloc[3000:3200]
    assert np.allclose(again.predict(X), res.models[0].predict(X))


@pytest.mark.parametrize("kind", ["hmm_regime", "markov_switching"])
def test_a_regime_model_never_uses_a_later_day_than_the_one_it_scores(ds, cfg, kind):
    """Scramble every return after a cut date: the probabilities and the scores on or before the cut must not move."""
    cut = ds.dates[1000]
    model = md.make_model(kind, {"n_states": 2}, "regression", 0).bind(ds)
    pos = ds.positions()
    train_rows = np.flatnonzero((pos >= 300) & (pos < 800) & ds.Y[ds.target].notna().to_numpy())
    model.fit(ds.X.iloc[train_rows], ds.Y[ds.target].iloc[train_rows], pos[train_rows])
    rows = ds.X.iloc[np.flatnonzero((pos >= 850) & (pos <= 1000))]
    before_p, before = model.probabilities(cut).copy(), model.predict(rows)
    rng = np.random.default_rng(5)
    later = ds.dates > cut
    r2 = ds.returns.copy()
    r2.loc[later] = rng.normal(0.0, 0.05, (int(later.sum()), r2.shape[1]))
    other = dataclasses.replace(ds, returns=r2)
    model.bind(other)
    assert np.allclose(model.probabilities(cut).to_numpy(), before_p.to_numpy(), equal_nan=True)
    assert np.allclose(model.predict(rows), before)
    assert not np.allclose(model.probabilities(ds.dates[-1]).iloc[-50:].to_numpy(), model.bind(ds).probabilities(ds.dates[-1]).iloc[-50:].to_numpy())      # the later days do change later beliefs


def test_a_markov_switching_fit_repeats_exactly_and_a_start_that_fails_is_replaced_then_the_fit_is_refused(ds, monkeypatch):
    """statsmodels draws the random starts of its search from the ``rng`` it is given (newer versions ignore numpy's global seed), so the fit must pass its seed on: the same data gives the same parameters."""
    from src.ml import regimes as rg

    Z = rg.market_state(ds).dropna().to_numpy()[:600]
    a, b = rg.MarkovSwitchingEngine(2, seed=3).fit(Z), rg.MarkovSwitchingEngine(2, seed=3).fit(Z)
    assert np.array_equal(a.params_, b.params_) and np.array_equal(a.trans_, b.trans_)
    calls = []

    def fails_twice(self, y, seed):
        calls.append(seed)
        if len(calls) < 3:
            raise RuntimeError("Steady-state probabilities could not be constructed.")
        return real(self, y, seed)

    real = rg.MarkovSwitchingEngine._fit_once
    monkeypatch.setattr(rg.MarkovSwitchingEngine, "_fit_once", fails_twice)
    assert rg.MarkovSwitchingEngine(2, seed=3).fit(Z).params_.shape == a.params_.shape and calls == [3, 4, 5], "each attempt starts from the next seed"
    monkeypatch.setattr(rg.MarkovSwitchingEngine, "_fit_once", lambda self, y, seed: (_ for _ in ()).throw(RuntimeError("Steady-state probabilities could not be constructed.")))
    with pytest.raises(ValueError, match="could not be fitted: Steady-state"):
        rg.MarkovSwitchingEngine(2, seed=3).fit(Z)


def test_the_regime_models_need_a_dataset_and_enough_history(ds):
    with pytest.raises(ValueError, match="bound to a dataset"):
        md.make_model("hmm_regime").fit(ds.X.iloc[:500], ds.Y[ds.target].iloc[:500])
    short = md.make_model("hmm_regime").bind(ds)
    with pytest.raises(ValueError, match="too few"):
        short.fit(ds.X.iloc[:300], ds.Y[ds.target].iloc[:300])
