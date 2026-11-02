"""Release 7, P2-08: regularised ensemble weights trained on out-of-sample streams, with explicit windows and member coverage.

The stacking solve is checked against numbers whose answer is known and against a second solver. The ensemble is then fitted on a small market and its weights are traced to the held-out stage; a member
that scores too few rows is left out, a member that misses some rows has its weight handed to the others for those rows only, and a model saved before this release still predicts.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear

from src.ml import config as mc, datasets as dsm, ensemble_models as em, models as md, training as tr
from tests.ml_helpers import make_bundle

SPLIT = {"min_train_days": 400, "refit_every": 300}
MEMBERS = "ridge,lasso[alpha=0.05],elasticnet[alpha=0.05]"


@pytest.fixture(scope="module")
def cfg():
    return mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility", "relative"]}, "split": SPLIT}, defaults=mc.DEFAULTS)


@pytest.fixture(scope="module")
def ds(cfg):
    return dsm.build_dataset(make_bundle(8, 1200, seed=41, alpha=1.0), cfg["dataset"], "stacked")


def fit_rows(ds, lo=100, hi=800):
    pos = ds.positions()
    rows = np.flatnonzero((pos >= lo) & (pos < hi) & ds.Y[ds.target].notna().to_numpy())
    return rows, pos[rows]


def fitted(ds, **params):
    rows, pos = fit_rows(ds)
    model = md.make_model("model_ensemble", {"members": MEMBERS, "weighting": "stacked", **params}, "regression", 0).bind(ds)
    return model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)


# ------------------------------------------------------------------------------------------------------------------ the stacking solve
def test_the_stacking_weights_recover_a_known_blend_and_are_shrunk_toward_equal_weights():
    rng = np.random.default_rng(0)
    c1, c2 = rng.normal(0.0, 1.0, 500), rng.normal(0.0, 1.0, 500)
    y = 0.7 * c1 + 0.3 * c2
    cal = np.column_stack([c1, c2])
    w0, info = em.stack_weights(cal, y, shrink=0.0)
    assert w0 == pytest.approx([0.7, 0.3], abs=1e-9) and info["rows"] == 500 and info["nnls"] == pytest.approx([0.7, 0.3], abs=1e-9)
    assert em.stack_weights(cal, y, shrink=0.5)[0] == pytest.approx([0.6, 0.4], abs=1e-9), "half way between the fit and one over two"
    assert em.stack_weights(cal, y, shrink=1.0)[0] == pytest.approx([0.5, 0.5])
    shifted, _ = em.stack_weights(cal + 3.0, y + 3.0, shrink=0.0)
    assert shifted == pytest.approx(w0, abs=1e-9), "a common level changes nothing: the fit is on centred values"


def test_a_member_that_points_the_wrong_way_gets_no_weight_and_members_that_add_nothing_give_equal_weights():
    rng = np.random.default_rng(1)
    c1, c2 = rng.normal(size=400), rng.normal(size=400)
    y = c1 + 0.5 * c2
    w, info = em.stack_weights(np.column_stack([c1, c2, -c1 + 0.1 * rng.normal(size=400)]), y, shrink=0.0)
    assert w[2] == 0.0 and w[0] > w[1] > 0 and w.sum() == pytest.approx(1.0), "non-negative: the anti-signal is dropped, not shorted"
    w, info = em.stack_weights(np.column_stack([c1, c2]), -(c1 + c2), shrink=0.0)
    assert w == pytest.approx([0.5, 0.5]) and "no member has a positive coefficient" in info["rule"]
    w, info = em.stack_weights(np.column_stack([c1[:10], c2[:10]]), y[:10], shrink=0.0)
    assert w == pytest.approx([0.5, 0.5]) and "fewer than" in info["rule"], "too few rows to estimate anything: equal weights, and it says so"
    nan = np.column_stack([c1, c2])
    nan[:50, 0] = np.nan
    assert em.stack_weights(nan, y, shrink=0.0)[1]["rows"] == 350, "rows a member does not score are not in the fit"


def test_the_non_negative_solution_matches_a_second_solver():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(600, 4))
    y = A @ np.array([0.5, 0.0, 0.3, 0.2]) + 0.2 * rng.normal(size=600)
    ours = em.stack_weights(A, y, shrink=0.0)[0]
    ybar = y.mean()
    ref = lsq_linear(A - ybar, y - ybar, bounds=(0.0, np.inf)).x
    assert ours == pytest.approx(ref / ref.sum(), abs=1e-6)


# ------------------------------------------------------------------------------------------------------------------ the ensemble
@pytest.mark.parametrize("weighting", ["stacked", "stacked_regime"])
def test_an_ensemble_with_stacked_weights_reports_where_its_weights_came_from(ds, cfg, weighting):
    res = tr.walk_forward_train(ds, {"kind": "model_ensemble", "params": {"members": MEMBERS, "weighting": weighting, "shrink": 0.25}}, cfg["split"], 0)
    w = res.extras["weights"]
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= -1e-12).all().all() and list(w.columns) == ["ridge", "lasso", "elasticnet"]
    d = res.diagnostics[0]
    info, lin = d["weights"], d["lineage"]
    assert info["rule"] == weighting and info["training_window"]["stage"].startswith("held-out") and info["training_window"]["rows"] == lin["stage_2"]["rows"]
    assert info["training_window"]["first"] == lin["stage_2"]["first"] and lin["stage_1_labels_settled_before_stage_2"] is True, "the weights come from the held-out stage, whose base predictions are out of sample"
    assert "out of sample" in lin["stage_2"]["base_predictions"] and info["pooled"]["rows"] > 100 and info["pooled"]["shrink"] == 0.25
    pooled = np.array(d["stacked_weights"])
    eq = np.full(3, 1 / 3)
    assert pooled == pytest.approx(0.75 * np.array(info["pooled"]["nnls"]) + 0.25 * eq, abs=1e-12) and pooled.sum() == pytest.approx(1.0)
    if weighting == "stacked":
        assert np.allclose(w.iloc[0], pooled) and len(w.round(12).drop_duplicates()) <= len(res.diagnostics), "one weight vector per fitted model: the first fold's is the first block's, and they change only at a refit"
    else:
        assert np.array(d["regime_weights"]).shape == (2, 3) and np.allclose(np.array(d["regime_weights"]).sum(axis=1), 1.0)
        assert {r["regime"] for r in info["regimes"]} == {0, 1} and all(("rows" in r) or ("pooled" in r["rule"]) for r in info["regimes"])
    assert info["coverage"] == {"ridge": 1.0, "lasso": 1.0, "elasticnet": 1.0} and info["excluded"] == []


def test_equal_shrinkage_is_the_equal_weight_ensemble_and_weights_are_not_trained_on_the_block_being_scored(ds):
    model = fitted(ds, shrink=1.0)
    assert np.allclose(model.static_weights_, 1 / 3)
    stacked = fitted(ds, weighting="stacked_regime")
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 1000))]
    base = stacked.predict_extras(block)["weights"]
    cut = ds.dates[900]
    p2 = ds.prices.copy()
    p2.loc[p2.index > cut] = p2.loc[p2.index > cut].to_numpy()[::-1] * 1.7                          # the world after the cut is replaced
    other = dataclasses.replace(ds, prices=p2, returns=p2.pct_change())
    stacked.bind(other)
    again = stacked.predict_extras(block)["weights"]
    early = base.index <= cut
    assert early.any() and np.allclose(base[early], again[early], atol=1e-12), "a day's weights use its own regime and the held-out stage, nothing after the day"
    first_stage = stacked.weights_info_["training_window"]
    assert pd.Timestamp(first_stage["last"]) < block.index.get_level_values(0).min(), "the training window of the weights ends before the block they score begins"
    again_model = md.Model.from_bytes(stacked.to_bytes()).bind(ds)
    assert np.allclose(again_model.predict(block), stacked.bind(ds).predict(block)), "a saved ensemble predicts what it did"


def test_a_model_saved_before_this_release_still_predicts(ds):
    model = fitted(ds, weighting="dynamic")
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 860))]
    expected = model.predict(block)
    for name in ("usable_", "coverage_", "static_weights_", "weights_info_", "regime_weights_"):
        model.__dict__.pop(name, None)                                                                  # what an earlier version never stored
    assert np.allclose(model.predict(block), expected) and "weights" not in model.diagnostics()


def test_the_settings_are_checked():
    for bad, msg in (({"shrink": 1.5}, "shrink"), ({"min_coverage": -0.1}, "min_coverage"), ({"weighting": "magic"}, "weighting must be one of")):
        with pytest.raises(ValueError, match=msg):
            md.make_model("model_ensemble", {"members": "ridge,lasso", **bad})
    assert "stacked_regime" in em.WEIGHTINGS


# ------------------------------------------------------------------------------------------------------------------ member coverage
class Holey(md.Model):
    """A member that scores only part of the rows (a deterministic share chosen by a hash of the row's index)."""

    needs_dataset = False

    def __init__(self, inner, frac, salt=0):
        self.inner, self.frac, self.salt = inner, float(frac), int(salt)
        self.kind, self.task, self.params, self.seed = inner.kind, inner.task, inner.params, inner.seed

    def fit(self, X, y, dates=None):
        self.inner.fit(X, y, dates)
        return self

    def predict(self, X):
        p = np.asarray(self.inner.predict(X), dtype=float)
        h = ((pd.util.hash_pandas_object(X.index, index=False).to_numpy() + self.salt) % 1000) / 1000.0
        return np.where(h < self.frac, np.nan, p)

    def predict_extras(self, X):
        return {}

    def importance(self):
        return self.inner.importance()

    def diagnostics(self):
        return {}


def patch_members(monkeypatch, fracs: dict):
    real = em.make_model
    monkeypatch.setattr(em, "make_model", lambda kind, params, task, seed: Holey(real(kind, params, task, seed), fracs[kind]) if kind in fracs else real(kind, params, task, seed))


@pytest.mark.parametrize("weighting", ["equal", "dynamic", "stacked"])
def test_a_member_below_the_coverage_floor_is_left_out_whatever_the_weighting(ds, monkeypatch, weighting):
    patch_members(monkeypatch, {"lasso": 0.6})                                                           # lasso scores 40% of the rows: below the 80% floor
    rows, pos = fit_rows(ds)
    model = md.make_model("model_ensemble", {"members": MEMBERS, "weighting": weighting}, "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    info = model.diagnostics()["weights"]
    assert [e["member"] for e in info["excluded"]] == ["lasso"] and info["excluded"][0]["coverage"] < 0.5 and "below the 80%" in info["excluded"][0]["reason"]
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 880))]
    ex = model.predict_extras(block)
    assert np.allclose(ex["weights"]["lasso"], 0.0) and np.allclose(ex["weights"].sum(axis=1), 1.0)
    assert np.isfinite(model.predict(block)).all(), "the member that scores too little is out, so its missing rows do not blank the ensemble's"


def test_where_a_member_misses_a_row_the_others_weights_are_renormalised_for_that_row(ds, monkeypatch):
    patch_members(monkeypatch, {"lasso": 0.1})                                                           # 90% coverage: kept
    rows, pos = fit_rows(ds)
    model = md.make_model("model_ensemble", {"members": MEMBERS, "weighting": "equal"}, "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    assert model.diagnostics()["weights"]["excluded"] == [] and 0.85 < model.diagnostics()["weights"]["coverage"]["lasso"] < 0.95
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 900))]
    S = np.column_stack([m.predict(block[model.features_]) for m in model.members_])
    cal = model.a_ + model.b_ * S
    pred = model.predict(block)
    missing = ~np.isfinite(S[:, 1])
    assert missing.any() and np.isfinite(pred).all()
    want = np.where(missing, cal[:, [0, 2]].mean(axis=1), np.nanmean(cal, axis=1))
    np.testing.assert_allclose(pred, want, atol=1e-12, err_msg="a row lasso did not score is the mean of the other two; the rest the mean of all three")


def test_a_row_no_member_scores_has_no_prediction_and_too_few_scoring_members_is_refused(ds, monkeypatch):
    patch_members(monkeypatch, {"ridge": 0.1, "lasso": 0.1, "elasticnet": 0.1})                          # the same hash drops the same rows from all three
    rows, pos = fit_rows(ds)
    model = md.make_model("model_ensemble", {"members": MEMBERS, "weighting": "equal"}, "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 900))]
    pred = model.predict(block)
    assert np.isnan(pred).any() and np.isfinite(pred[~np.isnan(pred)]).all(), "a row nobody scores has no prediction: not a zero, not a guess"
    patch_members(monkeypatch, {"ridge": 0.6, "lasso": 0.6})
    with pytest.raises(ValueError, match="fewer than two members score"):
        md.make_model("model_ensemble", {"members": MEMBERS, "weighting": "stacked"}, "regression", 0).bind(ds).fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)


def test_the_members_frame_keeps_each_members_calibrated_forecast_and_stays_missing_where_a_member_did_not_score(ds, monkeypatch):
    patch_members(monkeypatch, {"lasso": 0.1})
    rows, pos = fit_rows(ds)
    model = md.make_model("model_ensemble", {"members": MEMBERS, "weighting": "equal"}, "regression", 0).bind(ds)
    model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos)
    block = ds.X.iloc[np.flatnonzero((ds.positions() >= 820) & (ds.positions() < 900))]
    ex = model.predict_extras(block)
    members = ex["members"]
    assert list(members.columns) == list(ex["weights"].columns) and members.index.equals(block.index)
    S = np.column_stack([m.predict(block[model.features_]) for m in model.members_])
    np.testing.assert_allclose(members.to_numpy(), model.a_ + model.b_ * S, equal_nan=True, err_msg="the calibrated forecast a + b * score, in return units, member by member")
    miss = members["lasso"].isna().to_numpy()
    assert miss.any() and not members["ridge"].isna().any(), "a member's missing score is a missing value, not a zero"
    full = ~members.isna().any(axis=1).to_numpy()
    np.testing.assert_allclose(model.predict(block)[full], members[full].mean(axis=1).to_numpy(), atol=1e-12, err_msg="with equal weights and every member present the ensemble is the mean of its members")
    # the extras the training loop persists include it, indexed like the distribution
    from src.ml import training as tr

    merged = tr.collect_extras([ex])
    assert set(merged) == {"distribution", "weights", "members"} and merged["members"].index.equals(merged["distribution"].index)


def test_an_agent_cannot_be_an_ensemble_member_because_its_output_is_a_set_of_positions_not_a_return():
    for agent in ("ppo", "sac", "recurrent_ppo"):
        with pytest.raises(ValueError, match="agent whose output is a set of portfolio weights"):
            md.make_model("model_ensemble", {"members": f"ridge,{agent}"}, "regression", 0)
    md.make_model("model_ensemble", {"members": "ridge,lasso"}, "regression", 0)                           # return forecasts mix
    md.make_model("model_ensemble", {"members": "ridge,lgbm_ranker"}, "regression", 0)                     # a ranker's score is calibrated to return units before it is mixed, as every member's is
