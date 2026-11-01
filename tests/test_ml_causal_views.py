"""P0-18, P0-20 and the causal part of P0-29: a bound model sees a restricted view, and changing anything after a date changes nothing a model fitted or said on or before it.

The end-to-end check runs the whole walk-forward twice, on a dataset and on a copy whose features, labels, prices and returns are all replaced after a date P, for every family that reads history. Every prediction
for a day on or before P must be identical. That is the distinction the specification draws between *prediction causality* (this file) and a model that is legitimately refitted later on data that has since arrived.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm, models as md, training as tr
from src.ml.causal import CausalAccessError, CausalView
from src.utils.optional import unavailable_reason
from tests.ml_helpers import FAMILIES, changed_after, make_bundle

warnings.filterwarnings("ignore")
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 250, "refit_every": 100}}, defaults=mc.DEFAULTS)
SPLIT = CFG["split"]
@pytest.fixture(scope="module")
def ds():
    return dsm.build_dataset(make_bundle(6, 800, seed=91, alpha=1.0), CFG["dataset"], "causal")


# ------------------------------------------------------------------------------------------------------------ the view
def test_a_view_holds_nothing_after_its_cutoff_and_no_labels(ds):
    cut = 500
    v = CausalView(ds, cut)
    assert v.dates[-1] == ds.dates[cut] and v.prices.index[-1] == ds.dates[cut] and v.returns.index[-1] == ds.dates[cut] and v.investable.index[-1] == ds.dates[cut]
    assert v.X.index.get_level_values(0).max() == ds.dates[cut] and len(v.X) < len(ds.X) and v.positions().max() == cut
    assert (v.lag, v.horizon, v.target) == (ds.lag, ds.horizon, ds.target) and v.assets == ds.assets and v.features == ds.features
    for name in ("Y", "labels", "meta_secret", "anything_else"):
        with pytest.raises(CausalAccessError):
            getattr(v, name)
    assert CausalView.of(v) is v and CausalView.of(v, 400).cutoff_pos == 400
    with pytest.raises(CausalAccessError, match="cannot be widened"):
        CausalView.of(v, 600)
    with pytest.raises(CausalAccessError, match="not a date"):
        CausalView(ds, 10_000)
    assert CausalView(ds, ds.dates[cut]).cutoff_pos == cut, "a cutoff can be named by its date"


def test_the_perturbed_copy_really_differs_after_the_date_and_not_before(ds):
    other = changed_after(ds, 500)
    late = ds.dates > ds.dates[500]
    assert not np.allclose(ds.returns[late], other.returns[late]) and not np.allclose(ds.X.iloc[-50:].to_numpy(dtype=float), other.X.iloc[-50:].to_numpy(dtype=float), equal_nan=True)
    assert ds.returns[~late].equals(other.returns[~late]) and not np.allclose(ds.returns.shift(-1).loc[ds.dates[500]], other.returns.shift(-1).loc[ds.dates[500]]), "a model that looked one day ahead would see a difference"


@pytest.mark.parametrize("name", [n for n, f in FAMILIES.items() if f[0] != "ridge"])
def test_a_raw_dataset_given_to_bind_becomes_a_view_without_labels(ds, name):
    kind, params, _, needs = FAMILIES[name]
    if any(unavailable_reason(p) for p in needs):
        pytest.skip(f"{needs} not available")
    model = md.make_model(kind, params, "regression", 0)
    assert model.needs_dataset
    model.bind(ds)
    bound = model._ds
    assert isinstance(bound, CausalView) and bound.cutoff_pos == len(ds.dates) - 1
    with pytest.raises(CausalAccessError):
        getattr(bound, "Y")


# ------------------------------------------------------------------------------------------------------------ the whole walk-forward, twice
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_everything_a_family_said_on_or_before_a_date_is_unchanged_by_anything_after_it(ds, name):
    kind, params, tol, needs = FAMILIES[name]
    for pkg in needs:
        if unavailable_reason(pkg):
            pytest.skip(f"{pkg} is not available here")
    cfg = {"kind": kind, "params": params}
    a = tr.walk_forward_train(ds, cfg, SPLIT, 0)
    ok = [f for f in a.folds if not f.get("skipped")]
    assert len(ok) >= 3
    mid = ok[1]
    p = int(ds.dates.get_indexer([pd.Timestamp(mid["test_start"])])[0]) + 20                       # inside the second block, so the model fitted for it is the one under test
    assert p < int(ds.dates.get_indexer([pd.Timestamp(mid["test_end"])])[0])
    b = tr.walk_forward_train(changed_after(ds, p), cfg, SPLIT, 0)
    upto = a.predictions.index.get_level_values(0) <= ds.dates[p]
    assert upto.sum() > 100
    assert np.allclose(a.predictions[upto], b.predictions.reindex(a.predictions.index)[upto], atol=tol, equal_nan=True), f"{name}: a prediction on or before the date moved"
    assert not np.allclose(a.predictions[~upto], b.predictions.reindex(a.predictions.index)[~upto], atol=tol, equal_nan=True), "the later predictions must move, or the test is vacuous"
    for key, frame in a.extras.items():
        before = frame.index.get_level_values(0) <= ds.dates[p]
        other = b.extras[key].reindex(frame.index)
        assert np.allclose(frame[before].to_numpy(dtype=float), other[before].to_numpy(dtype=float), atol=max(tol, 1e-6), equal_nan=True), f"{name}: the {key} reported on or before the date moved"
    assert [f["train_rows_hash"] for f in ok[:2]] == [f["train_rows_hash"] for f in b.folds if not f.get("skipped")][:2], "the folds fitted before the date used the same rows"


def test_a_model_fitted_before_the_date_is_the_same_model_whatever_happens_after_it(ds):
    """The fitted artifacts themselves, not just the predictions: the ridge coefficients of the folds fitted before the change are identical, and the fold fitted after it is not."""
    cfg = {"kind": "ridge", "params": {"alpha": 10.0}}
    a = tr.walk_forward_train(ds, cfg, SPLIT, 0)
    ok = [f for f in a.folds if not f.get("skipped")]
    p = int(ds.dates.get_indexer([pd.Timestamp(ok[1]["test_start"])])[0]) + 20                    # inside the second block: folds 0 and 1 were fitted before it, fold 2 after
    b = tr.walk_forward_train(changed_after(ds, p), cfg, SPLIT, 0)
    ma, mb = [m for m in a.models if m is not None], [m for m in b.models if m is not None]
    assert len(ma) == len(mb) == 3
    assert np.array_equal(ma[0].estimator.coef_, mb[0].estimator.coef_) and np.array_equal(ma[1].estimator.coef_, mb[1].estimator.coef_)
    assert not np.allclose(ma[2].estimator.coef_, mb[2].estimator.coef_), "the last fold is fitted after the change and must see it"


def test_a_final_model_remembers_what_it_has_seen_and_refuses_to_replay_an_earlier_decision(ds):
    model = tr.fit_final(ds, {"kind": "ridge", "params": {}}, 0)
    info = model.fit_info_
    assert info["fit_cutoff"] == str(ds.dates[-1].date()) and pd.Timestamp(info["label_end_max"]) <= ds.dates[-1] and info["schema"]["features"] == ds.features and len(info["schema"]["hash"]) == 16
    latest = tr.predict_latest(model, ds)
    assert len(latest) == len(ds.assets) and latest.name == str(ds.dates[-1].date())
    with pytest.raises(tr.ReplayError, match="replay, not a decision"):
        tr.predict_latest(model, ds, ds.dates[-30])
    assert len(tr.predict_latest(model, ds, ds.dates[-30], allow_replay=True)) == len(ds.assets)


def test_a_fold_model_records_what_it_was_fitted_to_and_the_schema_of_its_features(ds):
    res = tr.walk_forward_train(ds, {"kind": "ridge", "params": {}}, SPLIT, 0)
    for f, m in zip(res.folds, res.models):
        if m is None:
            continue
        assert m.fit_info_["train_rows_hash"] == f["train_rows_hash"] and m.fit_info_["fit_cutoff"] == f["fit_cutoff"] and m.fit_info_["schema"]["features"] == ds.features


# ------------------------------------------------------------------------------------------------------------ negative controls: the harness has teeth
class Peeker(md.Model):
    """Reads tomorrow's realised return from the dataset it was bound to. The checks above must see this."""

    needs_dataset = True

    def __init__(self):
        super().__init__("peeker", "regression", {}, 0)
        self._ds = None

    def bind(self, ds):
        self._ds = CausalView.of(ds)
        return self

    def fit(self, X, y, dates=None):
        self.features_, self.n_train_ = list(X.columns), len(X)
        return self

    def raw_predict(self, X):
        nxt = self._ds.returns.shift(-1)
        di = nxt.index.get_indexer(X.index.get_level_values(0))
        ai = nxt.columns.get_indexer(X.index.get_level_values(1))
        return np.nan_to_num(nxt.to_numpy()[di, ai])


class LabelReader(Peeker):
    """Reads the realised labels while fitting."""

    def fit(self, X, y, dates=None):
        self._ds.Y                                                                                         # noqa: B018 - the access itself is the test
        return super().fit(X, y, dates)


def test_a_model_that_reads_tomorrow_is_caught_by_the_perturbation_check(ds, monkeypatch):
    monkeypatch.setattr(tr, "make_model", lambda kind, params, task, seed: Peeker())
    cfg = {"kind": "peeker", "params": {}}
    a = tr.walk_forward_train(ds, cfg, SPLIT, 0)
    ok = [f for f in a.folds if not f.get("skipped")]
    p = int(ds.dates.get_indexer([pd.Timestamp(ok[1]["test_start"])])[0]) + 20
    b = tr.walk_forward_train(changed_after(ds, p), cfg, SPLIT, 0)
    upto = a.predictions.index.get_level_values(0) <= ds.dates[p]
    moved = ~np.isclose(a.predictions[upto], b.predictions.reindex(a.predictions.index)[upto], atol=1e-12)
    assert moved.sum() >= 1, "a model that reads the next day's return must show up as a prediction that moves"
    assert a.predictions.index.get_level_values(0)[upto][moved].max() == ds.dates[p], "and it moves on the last day before the change, where tomorrow is the first changed day"


def test_a_model_cannot_read_the_labels_through_its_view(ds, monkeypatch):
    monkeypatch.setattr(tr, "make_model", lambda kind, params, task, seed: LabelReader())
    with pytest.raises(CausalAccessError, match="labels are not part of a causal view"):
        tr.walk_forward_train(ds, {"kind": "label_reader", "params": {}}, SPLIT, 0)
