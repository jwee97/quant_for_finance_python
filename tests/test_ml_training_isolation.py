"""P0-18 (training isolation): what a model was fitted to, and every choice made while developing it, depends on its eligible training rows and nothing else.

Two datasets that agree up to a date and differ after it in everything that happened later (the outcomes only, or the features too) are run through the same walk-forward. Every fold fitted before the date must
hold the same fitted state: the same scalers, imputation fills, target scaling, early-stopping history and weights, seen through what the fold's model says about its own training rows and through what it reports
about itself. A search that stops at the end of the validation segment must reach the same score whatever the test and forward segments hold. The fold fitted after the date, which legally learns from what has
since settled, must move, or the comparison would prove nothing.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm, experiments as xe, optimization as opt, training as tr
from src.utils.optional import unavailable_reason
from tests.ml_helpers import FAMILIES, changed_after, make_bundle

warnings.filterwarnings("ignore")
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 250, "refit_every": 100}}, defaults=mc.DEFAULTS)
SPLIT = CFG["split"]
VOLATILE = {"seconds", "peak_memory_mb", "device_note"}


@pytest.fixture(scope="module")
def ds():
    return dsm.build_dataset(make_bundle(6, 800, seed=91, alpha=1.0), CFG["dataset"], "isolation")


def scrub(x):
    """A diagnostics dict with the keys that depend on the machine and the clock removed."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, (list, tuple)):
        return [scrub(v) for v in x]
    return x


def same(a, b, tol):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (a is None) == (b is None) and (a is None or (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * (1.0 + abs(a)))
    return a == b


def training_sample(ds, fold, n=80):
    """Rows the fold was certainly fitted on: the last ``n`` decision days whose labels were settled at the fold's first decision."""
    t_first = int(ds.dates.get_indexer([pd.Timestamp(fold["fit_cutoff"])])[0])
    pos = ds.positions()
    top = t_first - ds.lag - ds.horizon
    return np.flatnonzero((pos <= top) & (pos > top - n))


@pytest.mark.parametrize("parts", [("prices", "returns", "Y"), ("prices", "returns", "X", "Y")], ids=["outcomes_only", "everything"])
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_state_of_every_fold_fitted_before_a_date_is_the_same_whatever_happens_after_it(ds, name, parts):
    kind, params, tol, needs = FAMILIES[name]
    for pkg in needs:
        if unavailable_reason(pkg):
            pytest.skip(f"{pkg} is not available here")
    cfg = {"kind": kind, "params": params}
    a = tr.walk_forward_train(ds, cfg, SPLIT, 0)
    ok = [f for f in a.folds if not f.get("skipped")]
    assert len(ok) >= 3
    p = int(ds.dates.get_indexer([pd.Timestamp(ok[1]["test_start"])])[0]) + 20                    # folds 0 and 1 were fitted before it, fold 2 after
    other = changed_after(ds, p, parts)
    b = tr.walk_forward_train(other, cfg, SPLIT, 0)
    ma, mb = [m for m in a.models if m is not None], [m for m in b.models if m is not None]
    diag_a, diag_b = {d["fold"]: d for d in a.diagnostics}, {d["fold"]: d for d in b.diagnostics}
    for k in (0, 1):
        rows = training_sample(ds, ok[k])
        assert len(rows) >= 50
        Xs = ds.X.iloc[rows]
        for m in (ma[k], mb[k]):
            if m.needs_dataset:                                                                  # the view each fold's model was left with ends at its block's last day: rows of its training window are inside it
                assert m._ds.cutoff_pos >= int(ds.dates.get_indexer([pd.Timestamp(ok[k]["fit_cutoff"])])[0])
        pa, pb = np.asarray(ma[k].predict(Xs), dtype=float), np.asarray(mb[k].predict(Xs), dtype=float)
        assert np.allclose(pa, pb, atol=max(tol, 1e-7), equal_nan=True), f"{name}: fold {k} says something different about its own training rows"
        assert same(scrub(diag_a.get(ok[k]["fold"], {})), scrub(diag_b.get(ok[k]["fold"], {})), max(tol, 1e-6)), f"{name}: fold {k} reports a different fit (training history, early stopping, weights)"
        assert ma[k].fit_info_["train_rows_hash"] == mb[k].fit_info_["train_rows_hash"] == ok[k]["train_rows_hash"]
        assert (getattr(ma[k], "y_mean_", None), getattr(ma[k], "y_std_", None)) == (getattr(mb[k], "y_mean_", None), getattr(mb[k], "y_std_", None)), "the target scaling came from the training rows only"
    rows = training_sample(ds, ok[2])
    late_a, late_b = np.asarray(ma[2].predict(ds.X.iloc[rows]), dtype=float), np.asarray(mb[2].predict(ds.X.iloc[rows]), dtype=float)
    assert not np.allclose(late_a, late_b, atol=max(tol, 1e-7), equal_nan=True), "the fold fitted after the change learns from what has settled by then, so it must differ"


def test_the_fitted_scalers_and_fills_are_functions_of_the_training_rows_only(ds):
    """The imputation fill, the scaler and the target scaling of a ridge fold, read straight from the fitted pipeline and recomputed from the rows the fold used."""
    res = tr.walk_forward_train(ds, {"kind": "ridge", "params": {"alpha": 10.0}}, SPLIT, 0)
    ok = [f for f in res.folds if not f.get("skipped")]
    pos = ds.positions()
    for f, m in zip(ok, [m for m in res.models if m is not None]):
        t_first = int(ds.dates.get_indexer([pd.Timestamp(f["fit_cutoff"])])[0])
        rows = np.flatnonzero(ds.Y[ds.target].notna().to_numpy() & (pos + ds.lag + ds.horizon <= t_first))
        assert tr.row_hash(rows) == f["train_rows_hash"] == m.fit_info_["train_rows_hash"], "the oracle's training rows are the fold's"
        values = ds.X.iloc[rows].to_numpy(dtype=float)
        impute, scale = m.pipe_.named_steps["impute"], m.pipe_.named_steps["scale"]
        assert np.allclose(impute.fill_, np.nan_to_num(np.nanmedian(values, axis=0), nan=0.0), atol=1e-6)
        filled = np.where(np.isnan(values), impute.fill_, values)
        assert np.allclose(scale.mean_, filled.mean(axis=0), atol=1e-5) and np.allclose(scale.scale_, filled.std(axis=0), atol=1e-5)
        y = ds.Y[ds.target].iloc[rows].to_numpy(dtype=float)
        assert m.y_mean_ == pytest.approx(float(np.nanmean(y))) and m.y_std_ == pytest.approx(float(np.nanstd(y)))


# ------------------------------------------------------------------------------------------------------------------ the search
def run(ds, overrides):
    return xe.run_experiment(overrides, dataset=ds, info={"source": "test", "label": "synthetic", "universe": "synthetic"}, registry=None, save=False, explain=False, final_model=False, robustness=False,
                             stop_after="validation")


def test_a_search_that_stops_at_the_end_of_validation_scores_the_same_whatever_the_later_segments_hold(ds):
    overrides = {"dataset": CFG["dataset"], "split": SPLIT, "model": {"kind": "ridge", "params": {"alpha": 10.0}}, "signal": {"n": 2}}
    plan, upos = tr.plan_for(ds, SPLIT, ds.target)
    last_validation = int(upos[plan.bounds["validation"][1]])
    other = changed_after(ds, last_validation + 10)
    first, second = run(ds, overrides), run(other, overrides)
    score_a, _ = opt.validation_score(first.report)
    score_b, _ = opt.validation_score(second.report)
    assert score_a == score_b and np.isfinite(score_a)
    val_a, val_b = first.net[first.labels == "validation"], second.net[second.labels == "validation"]
    assert len(val_a) > 100 and np.array_equal(val_a.to_numpy(), val_b.to_numpy())
    assert json.dumps(first.report["ic"]["all"], sort_keys=True, default=str) == json.dumps(second.report["ic"]["all"], sort_keys=True, default=str)


def test_the_same_change_does_move_a_run_that_does_predict_the_later_segments(ds):
    """The negative control: without ``stop_after`` the test segment is predicted, and a change inside it moves the result."""
    overrides = {"dataset": CFG["dataset"], "split": SPLIT, "model": {"kind": "ridge", "params": {"alpha": 10.0}}, "signal": {"n": 2}}
    plan, upos = tr.plan_for(ds, SPLIT, ds.target)
    other = changed_after(ds, int(upos[plan.bounds["validation"][1]]) + 10)
    full = lambda data: xe.run_experiment(overrides, dataset=data, info={"source": "test", "label": "synthetic", "universe": "synthetic"}, registry=None, save=False, explain=False, final_model=False, robustness=False)  # noqa: E731
    a, b = full(ds), full(other)
    assert a.report["segments"]["stats"]["test"]["sharpe"] != b.report["segments"]["stats"]["test"]["sharpe"]
