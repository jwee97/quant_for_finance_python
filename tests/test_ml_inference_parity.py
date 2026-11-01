"""P0-20: sequence windows end at the prediction cutoff, scalers and early stopping use eligible history only, the heads map to their horizons, and inference gives the same answer however it is batched.

The window builder is checked against a loop; the scaler and the train/validation split of a sequence model against definitions written out independently; the head targets against independently computed forward
returns in the order the heads are reported. Inference parity compares, for every family that reads history, the full block, chunks of it, a prefix of it, and a live replay in which the model is bound to a view
that ends on the decision date. A model whose answer depends on how the call is batched is built here to show that the comparison would see it.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, datasets as dsm, models as md, training as tr
from src.ml.causal import CausalView
from src.ml.deep import data as dd, model as dm
from src.utils.optional import unavailable_reason
from tests.ml_helpers import FAMILIES, NET, make_bundle

warnings.filterwarnings("ignore")
CFG = mc.experiment_config({"dataset": {"feature_groups": ["price", "volatility"]}, "split": {"min_train_days": 250, "refit_every": 100}}, defaults=mc.DEFAULTS)
SPLIT = CFG["split"]


@pytest.fixture(scope="module")
def ds():
    return dsm.build_dataset(make_bundle(6, 800, seed=91, alpha=1.0), CFG["dataset"], "parity")


# ------------------------------------------------------------------------------------------------------------------ windows
def test_a_window_ends_at_its_row_is_nan_before_the_history_starts_and_ignores_later_rows():
    rng = np.random.default_rng(0)
    dates, assets, L = pd.bdate_range("2024-01-01", periods=40), ["A", "B", "C"], 6
    idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    frame = pd.DataFrame(rng.normal(size=(len(idx), 2)), index=idx, columns=["f1", "f2"])
    frame.iloc[rng.choice(len(frame), 15, replace=False)] = np.nan
    acc = dd.SeqAccessor.from_frame(frame)
    t = np.array([0, 2, 5, 6, 20, 39]); a = np.array([0, 1, 2, 0, 1, 2])
    got = acc.windows(t, a, L)
    cube = frame.to_numpy().reshape(len(dates), len(assets), 2)
    for i, (ti, ai) in enumerate(zip(t, a)):
        for k in range(L):
            step = ti - (L - 1) + k
            want = cube[step, ai] if step >= 0 else np.full(2, np.nan)
            np.testing.assert_allclose(got[i, k], want, rtol=1e-6, atol=1e-7, equal_nan=True, err_msg=str((ti, ai, k)))
    later = frame.copy()
    later.iloc[np.flatnonzero(frame.index.get_level_values(0) > dates[20])] = rng.normal(size=(int((frame.index.get_level_values(0) > dates[20]).sum()), 2))
    other = dd.SeqAccessor.from_frame(later).windows(t[:5], a[:5], L)
    assert np.array_equal(got[:5], other, equal_nan=True), "a window of a row on or before day 20 does not depend on anything after it"
    assert np.isnan(got[0, :-1]).all() and not np.isnan(got[0, -1]).all(), "the row on the first day has only its own bar: the earlier steps are missing, not borrowed"


# ------------------------------------------------------------------------------------------------------------------ scaler, split and heads
def _tiny_lstm(**extra):
    return {**NET, "max_rows": 100000, **extra}                              # no subsampling, so that the rows the model trains on can be compared with the definition


def fitted_sequence_model(ds, params, t_first=600):
    """A sequence model fitted the way the trainer fits one at ``t_first``, with the rows and positions it was given."""
    pos = ds.positions()
    rows = np.flatnonzero(ds.Y[ds.target].notna().to_numpy() & (pos + ds.lag + ds.horizon <= t_first))
    model = md.make_model("lstm", params, "regression", 0).bind(CausalView(ds, t_first))
    return model.fit(ds.X.iloc[rows], ds.Y[ds.target].iloc[rows], pos[rows]), rows, pos[rows]


def test_the_scaler_the_split_and_early_stopping_use_the_training_window_only(ds):
    pytest.importorskip("torch")
    params = _tiny_lstm(val_fraction=0.2)
    model, rows, pos = fitted_sequence_model(ds, params)
    upos = np.unique(pos)
    n_val = max(1, int(round(len(upos) * 0.2)))
    start, purge = int(upos[-n_val]), ds.lag + ds.horizon
    train, val = np.flatnonzero(pos <= start - 1 - purge), np.flatnonzero(pos >= start)
    got_train, got_val = model._split(pos, np.random.default_rng(0))
    assert np.array_equal(np.sort(got_train), train) and np.array_equal(np.sort(got_val), val)
    capped = md.make_model("lstm", {**params, "max_rows": 900}, "regression", 0).bind(CausalView(ds, 600))
    sub_train, sub_val = capped._split(pos, np.random.default_rng(0))
    assert len(sub_train) == 900 and len(sub_val) == 300 and set(sub_train) <= set(train) and set(sub_val) <= set(val), "a cap only samples the eligible rows; it never adds any"
    assert model.n_windows_ == (len(train), len(val)) and pos[train].max() + purge < pos[val].min(), "no training label reaches the validation days"
    X = ds.X.iloc[rows].to_numpy(dtype=np.float32)
    assert np.allclose(model.scaler_.mu_, np.nan_to_num(np.nanmean(X[train].astype(np.float64), axis=0)), atol=1e-5)
    assert np.allclose(model.scaler_.sd_, np.nanstd(X[train].astype(np.float64), axis=0), atol=1e-5), "the scaler is fitted on the training part of the window, not on the validation tail"
    assert len(model.history_.val_loss) == model.history_.to_dict()["epochs_run"] >= 1, "early stopping watched the validation loss of the inner split, once per epoch"
    assert pos[val].max() <= 600 - purge, "every validation label was settled by the fitting day"


def test_each_head_is_trained_on_the_forward_return_of_the_horizon_it_is_reported_for(ds, monkeypatch):
    pytest.importorskip("torch")
    seen = {}
    real = dm.fit_network

    def spy(net, batch_fn, loss_fn, n_train, n_val, cfg):
        seen["batch_fn"], seen["n_train"] = batch_fn, n_train
        return real(net, batch_fn, loss_fn, n_train, n_val, cfg)

    monkeypatch.setattr(dm, "fit_network", spy)
    t_first = 600
    model, rows, pos = fitted_sequence_model(ds, _tiny_lstm(horizons="2,3"), t_first)
    assert model.heads_ == [2, 3, ds.horizon] and list(model.predict_extras(ds.X.iloc[rows[:20]])["heads"].columns) == ["h2", "h3", f"h{ds.horizon}"]
    _, y, mask = seen["batch_fn"]("train", np.arange(seen["n_train"]))
    y, mask = y.numpy(), mask.numpy()
    train, _ = model._split(pos, np.random.default_rng(0))
    prices = ds.prices.iloc[: t_first + 1]                                                                                       # the prices known at the fitting day
    everything = ds.X.iloc[rows].index                                                                                           # the rows the model was fitted on: its target scaling comes from all of them
    index = everything[train]
    for j, h in enumerate((2, 3)):
        fwd = (prices.shift(-(ds.lag + h)) / prices.shift(-ds.lag) - 1.0).reindex(index=ds.dates, columns=ds.assets)               # P[t+lag+h] / P[t+lag] - 1
        wide = fwd.sub(fwd.mean(axis=1), axis=0).to_numpy()
        ta, aa = ds.dates.get_indexer(everything.get_level_values(0)), pd.Index(ds.assets).get_indexer(everything.get_level_values(1))
        all_rows = wide[ta, aa]
        mean, std = np.nanmean(all_rows), np.nanstd(all_rows)
        want = wide[ds.dates.get_indexer(index.get_level_values(0)), pd.Index(ds.assets).get_indexer(index.get_level_values(1))]
        ok = np.isfinite(want)
        assert ok.any() and np.array_equal(mask[:, j].astype(bool), ok)
        assert np.allclose(y[ok, j], (want[ok] - mean) / std, atol=1e-4), f"head {j} (horizon {h}) is not trained on its own horizon's return"
    primary = ds.Y[ds.target].iloc[rows].to_numpy(dtype=float)[train]
    ok = np.isfinite(primary)
    assert np.allclose(y[ok, 2], (primary[ok] - model.y_mean_) / model.y_std_, atol=1e-4), "the last head is the dataset's own target"
    with pytest.raises(ValueError, match="shorter than the dataset's horizon"):
        fitted_sequence_model(ds, _tiny_lstm(horizons=f"2,{ds.horizon}"))


# ------------------------------------------------------------------------------------------------------------------ parity
def block_of(ds, name, fold=0):
    kind, params, tol, needs = FAMILIES[name]
    for pkg in needs:
        if unavailable_reason(pkg):
            pytest.skip(f"{pkg} is not available here")
    res = tr.walk_forward_train(ds, {"kind": kind, "params": params}, SPLIT, 0)
    model = [m for m in res.models if m is not None][fold]
    rows = res.fold_rows[[i for i, m in enumerate(res.models) if m is not None][fold]][: 36 * len(ds.assets)]                      # the first 36 days of the block, every asset
    return res, model, ds.X.iloc[rows], tol


def at(ds, date):
    return CausalView(ds, int(ds.dates.get_indexer([date])[0]))


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_prefix_of_the_block_gives_the_values_the_whole_block_gives_for_the_same_rows(ds, name):
    """Nothing later in the call may change an earlier row, including for the models that carry a state through the call (dynamic weights, a filter, a simulated account)."""
    res, model, X, tol = block_of(ds, name)
    full = np.asarray(model.predict(X), dtype=float)
    dates = X.index.get_level_values(0)
    for k in (5, 12, 25):
        cut = dates.unique()[k]
        part = X[dates <= cut]
        got = np.asarray(model.predict(part), dtype=float)
        assert np.allclose(got, full[: len(part)], atol=max(tol, 1e-6), equal_nan=True), f"{name}: rows up to {cut.date()} moved when later days were removed from the call"
    saved = res.predictions.reindex(X.index).to_numpy(dtype=float)
    assert np.allclose(full, saved, atol=max(tol, 1e-6), equal_nan=True), f"{name}: the model does not reproduce the predictions the evaluation saved"


@pytest.mark.parametrize("name", ["ridge", "lstm", "gcn", "hmm"])
def test_chunked_and_one_row_at_a_time_inference_equal_the_full_batch_for_models_without_a_call_state(ds, name):
    res, model, X, tol = block_of(ds, name)
    full = np.asarray(model.predict(X), dtype=float)
    chunked = np.concatenate([np.asarray(model.predict(X.iloc[i:i + 7]), dtype=float) for i in range(0, len(X), 7)])
    single = np.array([float(np.asarray(model.predict(X.iloc[[i]]))[0]) for i in range(len(X))])
    assert np.allclose(chunked, full, atol=max(tol, 1e-6), equal_nan=True) and np.allclose(single, full, atol=max(tol, 1e-6), equal_nan=True), name


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_a_live_replay_bound_to_a_view_that_ends_on_the_decision_date_gives_the_batch_value(ds, name):
    """What a live system would do on each day: know only the history up to that date (the view), and the days of the block so far (the call's rows). The answer for that day is the batch answer."""
    res, model, X, tol = block_of(ds, name)
    full = pd.Series(np.asarray(model.predict(X), dtype=float), index=X.index)
    dates = X.index.get_level_values(0)
    for k in (4, 10, 20):
        day = dates.unique()[k]
        model.bind(at(ds, day))
        rows = X[dates <= day]
        got = pd.Series(np.asarray(model.predict(rows), dtype=float), index=rows.index)
        today = got[got.index.get_level_values(0) == day]
        assert np.allclose(today.to_numpy(), full.reindex(today.index).to_numpy(), atol=max(tol, 1e-6), equal_nan=True), f"{name}: the live value on {day.date()} differs from the batch value"


class BatchDependent(md.Model):
    """Subtracts the mean of the call from its answer: the value of a row depends on which other rows share the call."""

    def fit(self, X, y, dates=None):
        self.features_, self.n_train_ = list(X.columns), len(X)
        return self

    def raw_predict(self, X):
        v = X.iloc[:, 0].to_numpy(dtype=float)
        return v - np.nanmean(v)


def test_the_comparison_would_catch_a_model_whose_answer_depends_on_how_the_call_is_batched(ds):
    X = ds.X.iloc[:200]
    model = BatchDependent("batch_dependent", "regression", {}, 0).fit(X, ds.Y[ds.target].iloc[:200])
    full = model.predict(X)
    chunked = np.concatenate([model.predict(X.iloc[i:i + 10]) for i in range(0, 200, 10)])
    assert np.nanmax(np.abs(chunked - full)) > 1e-3 and not np.allclose(model.predict(X.iloc[:50]), full[:50], equal_nan=True)
