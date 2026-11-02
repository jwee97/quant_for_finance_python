"""Release 6, P2-03: stable feature selection inside the training window.

The panel has three informative features and nine noise features, so the right answer is known. The error-control bound is checked against its published formula, a date-level feature is shown to be unassessable
rather than chosen or dropped silently, and the in-fold property is checked by changing everything after the first fold's training window and observing that the first fold's choice does not move.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as ds, featsel as fs, models as md, training as tr
from tests.ml_helpers import make_bundle

N_DATES, N_ASSETS = 360, 20
SIGNAL = ["s1", "s2", "s3"]


def panel(seed=0, signal=1.0, n_dates=N_DATES):
    rng = np.random.default_rng(seed)
    dates = np.repeat(np.arange(n_dates), N_ASSETS)
    cols = {f"s{i + 1}": rng.normal(size=len(dates)) for i in range(3)} | {f"n{i + 1}": rng.normal(size=len(dates)) for i in range(9)}
    X = pd.DataFrame(cols)
    y = pd.Series(signal * (0.6 * X["s1"] + 0.4 * X["s2"] - 0.4 * X["s3"]) + rng.normal(size=len(dates)), name="y")
    return X, y, dates


def test_the_informative_features_are_found_and_the_rest_are_rejected_with_their_frequency_and_reason():
    X, y, dates = panel(1)
    out = fs.stability_selection(X, y, dates, selector="ic", top_k=3, pi=0.9, subsamples=40, block_days=20, seed=3)
    assert out["eligibility"] == "applied" and sorted(out["chosen"]) == SIGNAL
    assert all(out["frequencies"][f] == 1.0 for f in SIGNAL) and all(out["frequencies"][f] < 0.9 for f in out["frequencies"] if f not in SIGNAL)
    assert len(out["rejected"]) == 9 and all("below the stability threshold 0.90" in r["reason"] for r in out["rejected"]) and {r["feature"] for r in out["rejected"]} == {f"n{i + 1}" for i in range(9)}
    assert out["sign_agreement"]["s1"] == 1.0 and out["sign_agreement"]["s3"] == 1.0
    q = out["error_control"]
    assert q["mean_selected_per_subsample"] == pytest.approx(3.0) and q["expected_false_selections_bound"] == pytest.approx(9.0 / ((2 * 0.9 - 1.0) * 12)), "q^2 / ((2 pi - 1) p)"
    assert q["meaningful"] is True and "no error control" in q["note"]
    assert json.dumps(out), "the record is plain data and can be persisted as it is"
    lasso = fs.stability_selection(X, y, dates, selector="lasso", pi=0.8, subsamples=30, block_days=20, selector_alpha=0.05, seed=3)
    assert sorted(lasso["chosen"]) == SIGNAL and lasso["sign_agreement"]["s1"] == 1.0
    again = fs.stability_selection(X, y, dates, selector="ic", top_k=3, pi=0.9, subsamples=40, block_days=20, seed=3)
    assert again["frequencies"] == out["frequencies"], "the same seed and settings give the same record"


def test_a_loose_setting_is_reported_as_giving_no_error_control_and_noise_alone_rarely_clears_a_strict_threshold():
    X, y, dates = panel(2, signal=0.0)
    loose = fs.stability_selection(X, y, dates, selector="ic", top_k=6, pi=0.55, subsamples=40, block_days=20, seed=1)
    assert loose["error_control"]["expected_false_selections_bound"] == pytest.approx(36.0 / (0.1 * 12)) and loose["error_control"]["meaningful"] is False
    strict = fs.stability_selection(X, y, dates, selector="ic", top_k=2, pi=0.95, subsamples=40, block_days=20, seed=1)
    assert strict["error_control"]["meaningful"] is True and len(strict["chosen"]) <= 1, "pure noise: at most one chance selection even at a strict threshold, and the bound says why"
    assert fs.stability_selection(X, y, dates, pi=0.5, top_k=2)["error_control"]["expected_false_selections_bound"] is None, "no bound exists at pi <= 0.5"


def test_a_feature_that_does_not_vary_across_assets_on_a_date_cannot_rank_them_and_is_named_as_such():
    X, y, dates = panel(4)
    X["macro"] = np.repeat(np.random.default_rng(9).normal(size=N_DATES), N_ASSETS)                    # one value per date: no cross-sectional information
    out = fs.stability_selection(X, y, dates, selector="ic", top_k=3, pi=0.8, subsamples=30, block_days=20, seed=2)
    assert "macro" not in out["chosen"] and out["frequencies"]["macro"] == 0.0
    assert next(r for r in out["rejected"] if r["feature"] == "macro")["reason"].startswith("no cross-sectional variation")


def test_too_few_blocks_to_draw_different_halves_is_insufficient_and_the_model_refuses():
    X, y, dates = panel(5, n_dates=60)
    out = fs.stability_selection(X, y, dates, block_days=20)
    assert out["eligibility"] == "insufficient" and out["chosen"] == [] and "at least 6 are needed" in out["reason"]
    with pytest.raises(ValueError, match="could not run"):
        md.make_model("stable_select", {"block_days": 20}, "regression").fit(X, y, dates)
    with pytest.raises(ValueError, match="date of every training row"):
        md.make_model("stable_select", {}, "regression").fit(X, y, None)
    with pytest.raises(ValueError, match="unknown selector"):
        fs.stability_selection(X, y, dates, selector="mutual_info")


def test_the_model_fits_the_base_on_the_chosen_features_only_and_persists_what_it_chose_and_rejected():
    X, y, dates = panel(6)
    m = md.make_model("stable_select", {"top_k": 3, "pi": 0.8, "block_days": 20, "alpha": 5.0}, "regression", seed=3).fit(X, y, dates)
    assert sorted(m.features_) == SIGNAL and list(m.importance().index) == m.features_ and m.n_train_ == len(X)
    pred = m.predict(X)
    ref = md.make_model("ridge", {"alpha": 5.0}, "regression").fit(X[SIGNAL], y, dates).predict(X[SIGNAL])
    assert np.allclose(pred, ref), "the same predictions as the base model fitted on the chosen columns"
    assert np.corrcoef(pred, y)[0, 1] > 0.5
    diag = m.diagnostics()["stable_selection"]
    assert diag["chosen"] == m.features_ and len(diag["rejected"]) == 9 and json.dumps(diag) and diag["settings"]["seed"] == 3
    back = md.Model.from_bytes(m.to_bytes())
    assert np.allclose(back.predict(X.iloc[:50]), pred[:50])
    with pytest.raises(md.ModelSchemaError, match="lacks"):
        m.predict(X.drop(columns=["s2"]))
    m2 = md.make_model("stable_select", {"top_k": 3, "pi": 0.8, "block_days": 20}, "regression", seed=3)
    m2.sample_weight_ = np.ones(len(X))
    m2.fit(X, y, dates)
    assert m2.weights_applied_ is True, "the dependence weights the trainer attaches reach the base model"
    with pytest.raises(ValueError, match="no stable signal"):
        md.make_model("stable_select", {"top_k": 3, "pi": 0.95, "block_days": 20}, "regression", seed=1).fit(*panel(7, signal=0.0)[:2], panel(7, signal=0.0)[2])
    with pytest.raises(ValueError, match="cannot itself"):
        md.make_model("stable_select", {"base": "stable_select"}, "regression")


def test_selection_stability_counts_how_often_each_feature_is_chosen_and_how_much_neighbouring_folds_agree():
    rec = lambda *c: {"chosen": list(c)}                                                                # noqa: E731
    out = fs.selection_stability([rec("a", "b"), rec("a", "b", "c"), None, rec("a", "d")])
    assert out["folds"] == 4 and out["folds_selected"] == 3 and out["inclusion_rate"] == {"a": 1.0, "b": 2 / 3, "c": 1 / 3, "d": 1 / 3}
    assert out["mean_size"] == pytest.approx((2 + 3 + 2) / 3) and out["consecutive_jaccard"] == pytest.approx((2 / 3 + 1 / 4) / 2), "the fold that selected nothing is left out of both"
    none = fs.selection_stability([None, {"chosen": []}])
    assert none["eligibility"] == "insufficient" and none["folds"] == 2 and none["folds_selected"] == 0


def test_a_walk_forward_run_selects_inside_each_fold_and_changing_the_future_does_not_move_the_first_choice():
    b = make_bundle(12, 1500, seed=31, alpha=1.0)
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "momentum", "mean_reversion", "volatility"]}, "split": {"min_train_days": 500, "refit_every": 63},
                                "model": {"kind": "stable_select", "params": {"pi": 0.5, "top_k": 4, "block_days": 21, "subsamples": 20}}}, defaults=mc.DEFAULTS)
    d = ds.build_dataset(b, cfg["dataset"], "sel")
    res = tr.walk_forward_train(d, cfg["model"], cfg["split"], seed=2)
    picked = [x for x in res.diagnostics if "stable_selection" in x]
    assert picked and all(set(x["stable_selection"]["chosen"]) <= set(d.X.columns) for x in picked)
    first = res.folds[[i for i, f in enumerate(res.folds) if not f.get("skipped")][0]]
    cut = pd.Timestamp(first["train_end"]) + pd.Timedelta(days=60)                                    # everything from here on lies after the first training window and its label ends
    later = d.dates[d.dates > cut]
    assert len(later) > 100
    rows = d.X.index.get_level_values(0) > cut
    d2 = ds.Dataset(d.name, d.X.copy(), d.Y.copy(), d.prices, d.returns, d.investable, d.meta, d.asset_class, d.group)
    rng = np.random.default_rng(0)
    d2.X.loc[rows] = rng.normal(size=d2.X.loc[rows].shape).astype(d2.X.dtypes.iloc[0])
    holes = d.Y.loc[rows].isna().to_numpy()                                                           # the unlabelled tail stays unlabelled: filling it would lengthen the labelled axis and move every fold boundary
    d2.Y.loc[rows] = np.where(holes, np.nan, rng.normal(size=holes.shape)).astype(d2.Y.dtypes.iloc[0])
    res2 = tr.walk_forward_train(d2, cfg["model"], cfg["split"], seed=2)
    f0 = [x for x in res.diagnostics][0]["stable_selection"]
    f0b = [x for x in res2.diagnostics][0]["stable_selection"]
    assert f0["chosen"] == f0b["chosen"] and f0["frequencies"] == f0b["frequencies"], "the first fold's choice used only its own training rows"
    assert any(a["stable_selection"]["frequencies"] != b2["stable_selection"]["frequencies"] for a, b2 in zip(res.diagnostics[1:], res2.diagnostics[1:])) or len(res.diagnostics) == 1, "later folds do see the changed data"
