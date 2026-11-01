"""P0-19: a model that learns from other models' predictions must learn only from out-of-sample ones, labelled in time, and the record must say which model made each prediction and what it had seen.

Three layers are checked: the walk-forward stream (every prediction knows its fold, its model's fitting day and its labels' last day); the member ensemble, which calibrates and weights its members on
predictions they made for rows they were not fitted on (checked with an independent recomputation from recorded predictions); and meta-labelling, whose labels must be settled at its own fitting day and
whose base predictions come from folds fitted before them. A held-out change after a date may not move anything a meta model said on or before it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml.causal import CausalView
from src.ml import config as mc, datasets as dsm, ensemble_models as em, ensembles as en, models as md, training as tr
from tests.ml_helpers import changed_after, make_bundle

warnings.filterwarnings("ignore")
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility", "momentum"]}, "split": {"min_train_days": 250, "refit_every": 100}}, defaults=mc.DEFAULTS)
SPLIT = CFG["split"]


@pytest.fixture(scope="module")
def ds():
    return dsm.build_dataset(make_bundle(6, 900, seed=77, alpha=1.0), CFG["dataset"], "oof")


@pytest.fixture(scope="module")
def primary(ds):
    return tr.walk_forward_train(ds, {"kind": "ridge", "params": {"alpha": 10.0}}, SPLIT, 0)


# ------------------------------------------------------------------------------------------------------------------ the walk-forward stream
def test_every_prediction_names_the_fold_that_made_it_and_what_that_model_had_seen(ds, primary):
    prod = primary.producers()
    assert not prod["fold"].isna().any() and len(prod) == len(set(primary.predictions.index.get_level_values(0)))
    assert primary.oos_violations() == []
    folds = [f for f in primary.folds if not f.get("skipped")]
    assert sorted(prod["fold"].unique()) == [f["fold"] for f in folds]
    assert prod["fold"].is_monotonic_increasing
    for f, m in zip(folds, [m for m in primary.models if m is not None]):
        first = prod[prod["fold"] == f["fold"]].index.min()
        assert str(first.date()) == f["test_start"] == f["fit_cutoff"], "a model is fitted at the close of the first day it will be asked about"
        assert m.fit_info_["train_rows_hash"] == f["train_rows_hash"] and pd.Timestamp(f["label_end_max"]) <= pd.Timestamp(f["fit_cutoff"])


def test_a_prediction_from_a_model_fitted_after_its_date_is_reported():
    """The check has teeth: a stream whose fold says it was fitted after the date it scores, or with labels that end after it was fitted, is named."""
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=4), ["A"]], names=["date", "asset"])
    folds = [{"fold": 0, "skipped": False, "test_start": "2020-01-01", "test_end": "2020-01-02", "fit_cutoff": "2020-01-03", "label_end_max": "2020-01-02", "train_rows_hash": "x"},
             {"fold": 1, "skipped": False, "test_start": "2020-01-03", "test_end": "2020-01-06", "fit_cutoff": "2020-01-03", "label_end_max": "2020-01-06", "train_rows_hash": "y"}]
    result = tr.TrainResult(pd.Series(0.0, index=idx), pd.Series(dtype=float), folds, [None, None], [], pd.DataFrame(), None, "t", "regression", "ridge", [])
    found = result.oos_violations()
    assert any("fitted on 2020-01-03, after the date" in v for v in found) and any("labels that end on 2020-01-06" in v for v in found) and len(found) == 4


# ------------------------------------------------------------------------------------------------------------------ the member ensemble
class Spy(md.Model):
    """A ridge member that records the rows it was fitted on and every prediction it makes."""

    log: list = []

    def __init__(self, inner):
        super().__init__("spy", "regression", {}, 0)
        self.inner = inner
        self.rec = {"fit": None, "predict": []}
        Spy.log.append(self.rec)

    def fit(self, X, y, dates=None):
        self.rec["fit"] = np.asarray(dates).copy()
        self.inner.fit(X, y, dates)
        self.features_, self.n_train_ = self.inner.features_, len(X)
        return self

    def raw_predict(self, X):
        out = self.inner.predict(X)
        self.rec["predict"].append((X.index.copy(), np.asarray(out).copy()))
        return out


@pytest.mark.parametrize("refit_full", [1, 0])
def test_the_ensemble_weighs_and_calibrates_members_only_on_rows_none_of_them_was_fitted_on(ds, monkeypatch, refit_full):
    real = md.make_model
    ens = real("model_ensemble", {"members": "ridge,ridge[alpha=300]", "weighting": "equal", "refit_full": refit_full}, "regression", 0)
    monkeypatch.setattr(em, "make_model", lambda kind, params, task, seed: Spy(real(kind, params, task, seed)))
    pos_all, t_first = ds.positions(), 600
    labelled = ds.Y[ds.target].notna().to_numpy()
    rows = np.flatnonzero(labelled & (pos_all + ds.lag + ds.horizon <= t_first))                                      # what the trainer would hand over at the first decision of a block
    X, y, pos = ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos_all[rows]
    ens.bind(CausalView(ds, t_first))
    Spy.log = []
    ens.fit(X, y, pos)
    stage_1 = Spy.log[:2]
    upos = np.unique(pos)
    n_val = max(5, int(round(len(upos) * float(ens.params["val_fraction"]))))
    start, purge = int(upos[-n_val]), ds.lag + ds.horizon
    expected_fit, expected_val = set(map(int, np.flatnonzero(pos <= start - 1 - purge))), set(map(int, np.flatnonzero(pos >= start)))
    lin = ens.lineage_
    assert lin["stage_1_labels_settled_before_stage_2"] and lin["refit_full"] == bool(refit_full) and lin["stage_2"]["first"] == str(ds.dates[start].date())
    assert lin["stage_1"]["last"] == str(ds.dates[int(pos[list(expected_fit)].max())].date()) and lin["stage_1"]["fit_rows"] == len(expected_fit) and lin["stage_2"]["rows"] == len(expected_val)
    for rec in stage_1:
        assert len(rec["predict"]) == 1
        idx, _ = rec["predict"][0]
        predicted = set(map(int, np.flatnonzero(np.isin(pos, ds.dates.get_indexer(idx.get_level_values(0))))))
        assert int(rec["fit"].max()) + ds.lag + ds.horizon < start, "a stage-1 member's labels are settled before the weighing rows begin"
        assert len(rec["fit"]) == len(expected_fit) and predicted == expected_val, "the members predict exactly the later rows and were fitted exactly on the earlier ones"
        assert not (set(map(int, rec["fit"])) & set(pos[list(expected_val)])), "no weighing row was in a member's fit"
    for j, rec in enumerate(stage_1):                                                                                 # the calibration recomputed from the recorded out-of-sample predictions
        idx, out = rec["predict"][0]
        t = y.reindex(idx).to_numpy(dtype=float)
        ok = np.isfinite(t)
        s, t = out[ok], t[ok]
        b = max(float(np.cov(s, t)[0, 1] / (np.var(s, ddof=1) + 1e-18)), 0.0)
        a = float(t.mean() - b * s.mean())
        assert ens.b_[j] == pytest.approx(b, rel=1e-9, abs=1e-12) and ens.a_[j] == pytest.approx(a, rel=1e-9, abs=1e-12)
        assert ens.sigma_[j] == pytest.approx(float(np.std(t - a - b * s, ddof=1)), rel=1e-9)
    if refit_full:
        assert len(Spy.log) == 4 and int(Spy.log[2]["fit"].max()) == int(pos.max()), "the deployed members were refitted on every training row, after the weights were set"
    else:
        assert len(Spy.log) == 2


def test_the_ensemble_lineage_travels_with_each_fold_of_a_walk_forward(ds):
    res = tr.walk_forward_train(ds, {"kind": "model_ensemble", "params": {"members": "ridge,ridge[alpha=300]", "weighting": "dynamic"}}, SPLIT, 0)
    folds = [d for d in res.diagnostics]
    assert folds and all(d["lineage"]["stage_1_labels_settled_before_stage_2"] for d in folds)
    for f, d in zip([f for f in res.folds if not f.get("skipped")], folds):
        lin = d["lineage"]
        assert pd.Timestamp(lin["stage_2"]["last"]) <= pd.Timestamp(f["fit_cutoff"]), "the weighing rows are inside the training window"
        assert pd.Timestamp(lin["stage_1"]["label_end_max"]) < pd.Timestamp(lin["stage_2"]["first"])


# ------------------------------------------------------------------------------------------------------------------ meta-labelling
@pytest.fixture(scope="module")
def meta(ds, primary):
    return en.meta_label_scores(ds, primary, "logistic", None, min_rows=200)


def test_meta_labels_are_settled_at_the_meta_fit_cutoff_and_base_predictions_come_from_earlier_folds(ds, primary, meta):
    assert meta["n_models"] >= 2 and len(meta["lineage"]) == meta["n_models"]
    lag_h = ds.lag + ds.horizon
    for block in meta["lineage"]:
        cutoff = pd.Timestamp(block["meta_fit_cutoff"])
        assert block["labels_settled_at_cutoff"] and pd.Timestamp(block["meta_label_end_max"]) <= cutoff
        last = ds.dates.get_loc(pd.Timestamp(block["meta_train_last_decision"]))
        assert last + lag_h <= ds.dates.get_loc(cutoff), "the last training decision's label ends on or before the day the meta model is fitted"
        assert all(pd.Timestamp(c) <= cutoff for c in block["producer_fit_cutoffs"]) and block["producer_folds"] == sorted(block["producer_folds"])
        assert max(block["producer_folds"]) < len([f for f in primary.folds if not f.get("skipped")]), "base predictions come from folds that exist"


def test_a_change_after_a_date_cannot_move_what_the_meta_model_said_on_or_before_it(ds, primary, meta):
    ok = [f for f in primary.folds if not f.get("skipped")]
    p = int(ds.dates.get_indexer([pd.Timestamp(ok[2]["test_start"])])[0]) + 10                  # inside the third block: the meta models for blocks 1 and 2 were fitted before it
    other = changed_after(ds, p)
    primary_b = tr.walk_forward_train(other, {"kind": "ridge", "params": {"alpha": 10.0}}, SPLIT, 0)
    meta_b = en.meta_label_scores(other, primary_b, "logistic", None, min_rows=200)
    before = meta["probability"].index.get_level_values(0) <= ds.dates[p]
    pa = meta["probability"][before]
    pb = meta_b["probability"].reindex(pa.index)
    assert pa.notna().sum() > 100
    assert np.allclose(pa.to_numpy(dtype=float), pb.to_numpy(dtype=float), equal_nan=True, atol=1e-9), "a meta probability on or before the date moved"
    assert not np.allclose(meta["probability"][~before].to_numpy(dtype=float), meta_b["probability"].reindex(meta["probability"][~before].index).to_numpy(dtype=float), equal_nan=True, atol=1e-9)
