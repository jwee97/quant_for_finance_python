"""P0-13 (persisted label intervals) and P0-21 (concurrency, uniqueness and the weights that use them, and a record of the choice).

The weights are checked against numbers worked out by hand on an explicit calendar, then against a brute-force count on random gapped fixtures; the trainer's hook is checked for what it passes to the model, for the
models that cannot use per-row weights (which must say so), and for designs where weighting is pointless (a one-day horizon, an agent). The persisted label intervals are compared with an independent computation.
"""

import types
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm, experiments as xe, models as md, training as tr
from src.utils.optional import unavailable_reason
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")


def stub(rows, lag=1, horizon=3, n=60):
    """A dataset-shaped object holding only what ``dependence_weights`` reads: rows as (position, asset) pairs on an ``n``-day calendar."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    idx = pd.MultiIndex.from_tuples([(dates[p], a) for p, a in rows], names=["date", "asset"])
    pos = np.array([p for p, _ in rows])
    return types.SimpleNamespace(X=pd.DataFrame(index=idx), dates=dates, lag=lag, horizon=horizon, positions=lambda: pos)


def brute(rows, lag, horizon):
    """Average uniqueness of every row by counting, asset by asset, the labels alive on each of its days."""
    out = []
    for i, (p, a) in enumerate(rows):
        days = range(p + lag + 1, p + lag + horizon + 1)
        mine = [(q + lag + 1, q + lag + horizon) for q, b in rows if b == a]
        out.append(np.mean([1.0 / sum(s <= d <= e for s, e in mine) for d in days]))
    return np.array(out)


def test_uniqueness_equals_the_numbers_worked_out_by_hand():
    rows = [(p, "A") for p in (10, 11, 12, 20, 22, 23)]                # return days: 12-14, 13-15, 14-16, 22-24, 24-26, 25-27 (lag 1, horizon 3)
    dep = tr.dependence_weights(stub(rows), np.arange(len(rows)))
    first = np.array([(1 + 1 / 2 + 1 / 3) / 3, (1 / 2 + 1 / 3 + 1 / 2) / 3, (1 / 3 + 1 / 2 + 1) / 3, (1 + 1 + 1 / 2) / 3, (1 / 2 + 1 / 2 + 1 / 2) / 3, (1 / 2 + 1 / 2 + 1) / 3])
    assert dep["mean_uniqueness"] == pytest.approx(first.mean()) and dep["needed"]
    assert dep["weights"] == pytest.approx(first * len(first) / first.sum()) and dep["weights"].mean() == pytest.approx(1.0)
    assert dep["mean_concurrency"] == pytest.approx(np.mean([np.mean(c) for c in ([1, 2, 3], [2, 3, 2], [3, 2, 1], [1, 1, 2], [2, 2, 2], [2, 2, 1])]))


def test_events_of_different_assets_do_not_count_as_concurrent_with_each_other():
    rows = [(10, "A"), (10, "B"), (11, "A"), (11, "B")]
    dep = tr.dependence_weights(stub(rows), np.arange(4))
    alone = tr.dependence_weights(stub([(10, "A"), (11, "A")]), np.arange(2))
    assert dep["weights"][[0, 2]] == pytest.approx(alone["weights"]) and dep["weights"][[1, 3]] == pytest.approx(alone["weights"])


@pytest.mark.parametrize("seed", range(5))
def test_the_weights_equal_a_brute_force_count_on_random_gapped_fixtures(seed):
    rng = np.random.default_rng(seed)
    lag, horizon = int(rng.integers(0, 3)), int(rng.integers(2, 8))
    rows = sorted((int(p), str(a)) for p in rng.choice(np.arange(5, 120), 60, replace=False) for a in rng.choice(["A", "B", "C"], 2, replace=False))
    dep = tr.dependence_weights(stub(rows, lag, horizon, n=160), np.arange(len(rows)))
    expected = brute(rows, lag, horizon)
    assert dep["weights"] == pytest.approx(expected * len(expected) / expected.sum(), rel=1e-12) and dep["mean_uniqueness"] == pytest.approx(expected.mean(), rel=1e-12)


def test_a_one_day_horizon_has_nothing_to_weight():
    dep = tr.dependence_weights(stub([(p, "A") for p in range(10, 30)], horizon=1), np.arange(20))
    assert not dep["needed"] and np.allclose(dep["weights"], 1.0) and dep["mean_uniqueness"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------------------------ the trainer's hook
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 250, "refit_every": 100}}, defaults=mc.DEFAULTS)


def split(**extra):
    return {**CFG["split"], **extra}


@pytest.fixture(scope="module")
def ds():
    return dsm.build_dataset(make_bundle(6, 800, seed=91, alpha=1.0), CFG["dataset"], "dependence")


class Recorder(md.Model):
    """Records the weights it is handed and, like a model that uses them, calls ``_weights``."""

    seen: list = []

    def __init__(self, uses=True):
        super().__init__("recorder", "regression", {}, 0)
        self.uses = uses

    def fit(self, X, y, dates=None):
        self.features_, self.n_train_ = list(X.columns), len(X)
        Recorder.seen.append(None if getattr(self, "sample_weight_", None) is None else np.asarray(self.sample_weight_).copy())
        if self.uses:
            self._weights(dates)
        return self

    def raw_predict(self, X):
        return np.zeros(len(X))


def test_the_trainer_hands_the_model_the_uniqueness_weights_of_exactly_its_training_rows(ds, monkeypatch):
    monkeypatch.setattr(tr, "make_model", lambda kind, params, task, seed: Recorder())
    Recorder.seen = []
    res = tr.walk_forward_train(ds, {"kind": "recorder", "params": {}}, split(sample_weighting="uniqueness"), 0)
    ok = [f for f in res.folds if not f.get("skipped")]
    assert len(Recorder.seen) == len(ok) >= 3
    pos = ds.positions()
    for f, got in zip(ok, Recorder.seen):
        t_first = int(ds.dates.get_indexer([pd.Timestamp(f["fit_cutoff"])])[0])
        rows = np.flatnonzero(ds.Y[ds.target].notna().to_numpy() & (pos + ds.lag + ds.horizon <= t_first))
        want = tr.dependence_weights(ds, rows)
        assert got is not None and got == pytest.approx(want["weights"]) and got.mean() == pytest.approx(1.0)
        assert f["dependence"]["treatment"] == "uniqueness_weights" and f["dependence"]["applied"] and f["dependence"]["mean_uniqueness"] == pytest.approx(want["mean_uniqueness"], abs=1e-4)
        assert want["mean_uniqueness"] < 0.5, "five-day labels on consecutive days overlap heavily"
    for m in res.models:
        assert getattr(m, "sample_weight_", None) is None, "a saved model does not carry the fit's weights"


def test_without_the_setting_nothing_is_weighted_and_the_overlap_is_still_recorded(ds, monkeypatch):
    monkeypatch.setattr(tr, "make_model", lambda kind, params, task, seed: Recorder())
    Recorder.seen = []
    res = tr.walk_forward_train(ds, {"kind": "recorder", "params": {}}, split(), 0)
    assert all(w is None for w in Recorder.seen)
    dep = [f["dependence"] for f in res.folds if not f.get("skipped")]
    assert all(d["treatment"] == "none" and d["requested"] == "none" and d["mean_uniqueness"] < 0.5 for d in dep)


def test_a_model_that_cannot_use_per_row_weights_says_so(ds, monkeypatch):
    monkeypatch.setattr(tr, "make_model", lambda kind, params, task, seed: Recorder(uses=False))
    res = tr.walk_forward_train(ds, {"kind": "recorder", "params": {}}, split(sample_weighting="uniqueness"), 0)
    for f in (f for f in res.folds if not f.get("skipped")):
        assert f["dependence"]["treatment"] == "uniqueness_weights" and f["dependence"]["applied"] is False and "does not use per-row weights" in f["dependence"]["note"]
    summary = xe.dependence_summary(res.folds, "uniqueness")
    assert summary["treatment"] == "requested, not fully applied" and summary["notes"]


def test_ridge_and_the_regime_model_use_the_weights_and_change_their_fit(ds):
    for kind, params in (("ridge", {"alpha": 10.0}), ("hmm_regime", {"n_states": 2})):
        if kind == "hmm_regime" and unavailable_reason("hmmlearn"):
            continue
        plain = tr.walk_forward_train(ds, {"kind": kind, "params": params}, split(), 0)
        weighted = tr.walk_forward_train(ds, {"kind": kind, "params": params}, split(sample_weighting="uniqueness"), 0)
        assert all(f["dependence"]["applied"] for f in weighted.folds if not f.get("skipped")), kind
        assert not np.allclose(plain.predictions.to_numpy(), weighted.predictions.to_numpy()), f"{kind}: the weights must change the fit"


def test_weighting_is_not_attempted_where_it_is_pointless_or_meaningless(monkeypatch):
    one = dsm.build_dataset(make_bundle(6, 900, seed=3, alpha=1.0), mc.experiment_config({"dataset": {"feature_groups": ["price"], "target": "fwd_ret_1", "horizon": 1}}, defaults=mc.DEFAULTS)["dataset"], "h1")
    res = tr.walk_forward_train(one, {"kind": "ridge", "params": {}}, split(sample_weighting="uniqueness"), 0)
    d = [f["dependence"] for f in res.folds if not f.get("skipped")][0]
    assert d["treatment"] == "none" and "labels do not overlap" in d["note"] and d["mean_uniqueness"] == pytest.approx(1.0)
    assert tr.is_agent("ppo") and not tr.is_agent("ridge")


def test_the_setting_is_validated_and_does_not_change_an_existing_configuration_hash():
    assert "sample_weighting" not in CFG["split"]
    with pytest.raises(mc.MLConfigError, match="sample_weighting"):
        mc.experiment_config({"split": {"sample_weighting": "bootstrap"}}, defaults=mc.DEFAULTS)
    assert mc.config_hash(mc.experiment_config({"split": {"sample_weighting": "uniqueness"}}, defaults=mc.DEFAULTS)) != mc.config_hash(mc.experiment_config({}, defaults=mc.DEFAULTS))


def test_the_report_records_the_treatment_the_overlap_and_the_dataset_audit(ds):
    overrides = {"dataset": CFG["dataset"], "split": split(sample_weighting="uniqueness"), "model": {"kind": "ridge", "params": {"alpha": 10.0}}, "signal": {"n": 2}}
    exp = xe.run_experiment(overrides, dataset=ds, info={"source": "test", "label": "synthetic", "universe": "synthetic"}, registry=None, save=False, explain=False, final_model=False, robustness=False)
    dep = exp.report["dependence"]
    assert dep["treatment"] == "uniqueness_weights" and dep["requested"] == "uniqueness" and 0.1 < dep["mean_uniqueness"] < 0.5 and dep["folds"] >= 3
    assert exp.report["audit"]["label"]["target"] == ds.target and exp.report["audit"]["cells"] == ds.meta["audit"]["cells"]
    assert exp.report["folds"]["list"][0]["fit_cutoff"] and exp.report["folds"]["list"][0]["dependence"]["treatment"] == "uniqueness_weights"
    assert xe.dependence_summary([{"fold": 0}], "none")["treatment"] == "not recorded", "a run saved before the record existed says so"


# ------------------------------------------------------------------------------------------------------------------ persisted intervals
def test_the_label_intervals_are_persisted_with_the_dataset_and_match_an_independent_computation(ds, tmp_path):
    folder = ds.save(tmp_path / "d")
    saved = pd.read_parquet(folder / "label_intervals.parquet")
    assert list(saved.columns) == ["label_start_pos", "label_end_pos", "label_start_at", "label_end_at", "label_available_at"] and len(saved) == len(ds.dates)
    n, lag, h = len(ds.dates), ds.lag, ds.horizon
    for i in (0, 5, n - h - lag - 1, n - h - lag, n - 1):
        start, end = i + lag, i + lag + h
        assert saved["label_start_pos"].iloc[i] == start and saved["label_end_pos"].iloc[i] == end
        assert (pd.isna(saved["label_end_at"].iloc[i]) if end >= n else saved["label_end_at"].iloc[i] == ds.dates[end])
        assert (pd.isna(saved["label_available_at"].iloc[i]) if end >= n else saved["label_available_at"].iloc[i] == ds.dates[end]), "a label is available at the close of its last day"
    assert int(saved["label_end_at"].isna().sum()) == lag + h, "exactly the last lag+horizon decisions have a label that has not ended inside the data"
    assert ds.meta["label_convention"]["entry"] == "close of day t + lag"
